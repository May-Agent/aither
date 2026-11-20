//! ACP integration backed by `aither_agent`.
//!
//! Three pieces bridge the agent runtime and the protocol:
//!
//! - [`AitherHost`] is the [`SessionHost`] a [`crate::server::Server`]
//!   serves: it connects the session's declared MCP servers, builds an
//!   [`Agent`] per `session/new`, and hands out [`AitherSession`]s.
//! - [`AitherSession`] drives `agent.run(..)` per prompt turn, streaming
//!   the [`SessionUpdateMapper`]'s domain updates to the client.
//! - [`AcpChildAgent`] goes the other way: an external agent's
//!   [`ClientSession`] run as an `aither_agent` [`ChildAgent`], its
//!   session events forwarded onto the parent's stream as
//!   [`AgentEvent`]s.

use std::collections::HashMap;
use std::future::Future;
use std::ops::ControlFlow;
use std::path::PathBuf;

use aither_agent::{
    Agent, AgentError, AgentEvent, AgentPath, BlockId, ChildAgent, ChildEvents, OutputKind,
    RequestOutcome, RunOutcome, TimedEvent, ToolOrigin, ToolVerdict,
};
use aither_core::LanguageModel;
use aither_core::llm::tool::CancelSignal;
use aither_core::llm::{AgentIdentity, ToolResult};
use aither_mcp::McpConnection;
use aither_mcp::protocol::McpToolDefinition;
use askama::Template;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use executor_core::{AnyExecutor, Executor};
use futures_lite::StreamExt;
use futures_lite::future::or;
use serde_json::Value;

use crate::client::{ClientError, ClientSession, SessionEvent, TurnOutcome};
use crate::domain::{
    ContentBlock, ContentChunk, MessageId, Open, Patch, SessionSetupParams, SessionUpdate,
    StopReason, ToolCall, ToolCallContent, ToolCallStatus, ToolCallUpdate, ToolKind,
};
use crate::mcp::{ConnectError, connect_all};
use crate::rpc::RpcError;
use crate::server::{
    PendingPrompt, PromptError, ServerError, Session, SessionContext, SessionHost,
};

/// What the agent of a new session is built from.
#[derive(Debug)]
pub struct SessionSetup {
    /// The session's working directory.
    pub cwd: PathBuf,
    /// The MCP servers the client opened the session with, connected and in
    /// the order it listed them. The agent registers their tools.
    pub mcp: Vec<McpConnection>,
}

/// A [`SessionHost`] driving per-session `aither_agent::Agent` instances.
///
/// `make_agent` is the factory called for every `session/new`: it gets the
/// session's [`SessionSetup`] — its `cwd` and the already-connected
/// [`McpConnection`]s — and returns the [`Agent`] the session drives.
///
/// # Example
///
/// ```rust,ignore
/// use aither_acp::aither::{AitherHost, SessionSetup};
/// use aither_acp::domain::Implementation;
/// use aither_acp::server::Server;
/// use executor_core::tokio::TokioGlobal;
///
/// let server = Server::new(
///     AitherHost::new(TokioGlobal, |setup: SessionSetup| async move {
///         let mut builder = Agent::builder(llm).instruction("You are helpful");
///         for connection in setup.mcp {
///             builder = builder.mcp(connection);
///         }
///         Ok(builder.build())
///     }),
///     Implementation::new("my-agent", "1.0.0"),
/// );
/// let (_peer, serve) = server.serve_stdio();
/// serve.await;
/// ```
pub struct AitherHost<F> {
    executor: AnyExecutor,
    make_agent: F,
}

impl<F> std::fmt::Debug for AitherHost<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AitherHost").finish_non_exhaustive()
    }
}

impl<F> AitherHost<F> {
    /// A host whose sessions build agents with `make_agent`.
    ///
    /// `executor` runs the connections to the MCP servers sessions are
    /// opened with.
    pub fn new(executor: impl Executor + 'static, make_agent: F) -> Self {
        Self {
            executor: AnyExecutor::new(executor),
            make_agent,
        }
    }
}

impl<LLM, F, Fut> SessionHost for AitherHost<F>
where
    LLM: LanguageModel + 'static,
    F: Fn(SessionSetup) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Agent<LLM>, ServerError>> + Send + 'static,
{
    type Session = AitherSession<LLM>;

    fn capabilities(&self) -> crate::domain::AgentCapabilities {
        use crate::domain::{AgentCapabilities, EmptyObject, McpCaps, PromptCaps, SessionCaps};
        AgentCapabilities {
            session: Some(SessionCaps {
                prompt: Some(PromptCaps {
                    embedded_context: Some(EmptyObject),
                    ..PromptCaps::default()
                }),
                mcp: Some(McpCaps {
                    stdio: Some(EmptyObject),
                    http: Some(EmptyObject),
                    ..McpCaps::default()
                }),
                additional_directories: Some(EmptyObject),
                ..SessionCaps::default()
            }),
            ..AgentCapabilities::default()
        }
    }

    async fn new_session(
        &self,
        params: SessionSetupParams,
        _cx: &SessionContext,
    ) -> Result<Self::Session, ServerError> {
        let mcp = connect_all(&self.executor, &params.mcp_servers)
            .await
            .map_err(|error| match error {
                ConnectError::Unsupported { .. } => {
                    ServerError::Rpc(RpcError::invalid_params(error.to_string()))
                }
                ConnectError::Failed { .. } => {
                    ServerError::Rpc(RpcError::internal_error(error.to_string()))
                }
            })?;
        let mcp_definitions: Vec<McpToolDefinition> = mcp
            .iter()
            .flat_map(|connection| connection.mcp_definitions().as_ref().clone())
            .collect();
        let setup = SessionSetup {
            cwd: params.cwd.clone(),
            mcp,
        };
        let agent = (self.make_agent)(setup)
            .await
            .map_err(|error| ServerError::host(format!("could not start a session: {error}")))?;
        Ok(AitherSession::new(params.cwd, agent).with_tool_definitions(&mcp_definitions))
    }
}

/// One live ACP session backed by an `aither_agent::Agent`.
///
/// Prompt turns run `agent.run(..)`; every [`TimedEvent`] is mapped to a
/// [`SessionUpdate`] and emitted through the session context as it
/// arrives, so the client sees text and tool activity while the turn is
/// still running. `session/cancel` (or `session/close`) resolves the
/// turn's `cancelled` future, which stops the drive and ends the turn
/// `cancelled`.
///
/// The agent and its mapper live behind a mutex: [`Session::prompt`]
/// takes `&self` but the run needs `&mut agent`. Prompts are serialized
/// per session by the server, so the lock is uncontended in practice.
pub struct AitherSession<LLM: LanguageModel + 'static> {
    cwd: PathBuf,
    inner: futures_util::lock::Mutex<TurnDrive<LLM>>,
}

/// What a prompt turn drives mutably.
struct TurnDrive<LLM: LanguageModel + 'static> {
    agent: Agent<LLM>,
    mapper: SessionUpdateMapper,
}

impl<LLM: LanguageModel + 'static> std::fmt::Debug for AitherSession<LLM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AitherSession")
            .field("cwd", &self.cwd)
            .finish_non_exhaustive()
    }
}

impl<LLM: LanguageModel + 'static> AitherSession<LLM> {
    /// A session driving `agent` in `cwd`.
    #[must_use]
    pub fn new(cwd: PathBuf, agent: Agent<LLM>) -> Self {
        Self {
            cwd,
            inner: futures_util::lock::Mutex::new(TurnDrive {
                agent,
                mapper: SessionUpdateMapper::default(),
            }),
        }
    }

    /// The session's working directory.
    #[must_use]
    pub const fn cwd(&self) -> &PathBuf {
        &self.cwd
    }

    /// Attaches ACP display metadata extracted from `definitions`' extension
    /// keys (`x-acp-kind`, `x-acp-title`, `x-acp-title-field`).
    #[must_use]
    pub fn with_tool_definitions<'a>(
        self,
        definitions: impl IntoIterator<Item = &'a McpToolDefinition>,
    ) -> Self {
        let TurnDrive { agent, mapper } = self.inner.into_inner();
        Self {
            cwd: self.cwd,
            inner: futures_util::lock::Mutex::new(TurnDrive {
                agent,
                mapper: mapper.with_definitions(definitions),
            }),
        }
    }
}

impl<LLM: LanguageModel + 'static> Session for AitherSession<LLM> {
    async fn prompt(
        &self,
        pending: PendingPrompt,
        cx: &SessionContext,
    ) -> Result<StopReason, PromptError> {
        let text = match prompt_text(pending.content()) {
            Ok(text) => text,
            Err(error) => {
                pending.reject(error.clone()).await;
                return Err(PromptError::Failed(error));
            }
        };
        let turn = pending
            .accept()
            .await
            .map_err(|rejected| PromptError::from(rejected.into_inner()))?;

        let mut drive = self.inner.lock().await;
        // A cancelled turn ends like a completed one — `cancelled` is the
        // drive's stop future and `is_cancelled` disambiguates the outcome.
        let result = {
            let TurnDrive { agent, mapper } = &mut *drive;
            let flow = agent
                .run(&text, [])
                .drive(
                    (mapper, cx),
                    turn.cancelled(),
                    |(mapper, cx), timed| async move {
                        if let Some(update) = mapper.map(&timed) {
                            let _ = cx.update(update).await;
                        }
                        ControlFlow::Continue((mapper, cx))
                    },
                )
                .await;
            match flow {
                Ok(ControlFlow::Continue(_)) => Ok(if turn.is_cancelled() {
                    StopReason::Cancelled
                } else {
                    StopReason::EndTurn
                }),
                Ok(ControlFlow::Break(())) => Ok(StopReason::Cancelled),
                Err(error) => {
                    tracing::debug!(error = %error, "agent turn failed");
                    Err(PromptError::failed(error.to_string()))
                }
            }
        };
        drop(drive);
        result
    }
}

/// Text resource contents rendered for the model.
#[derive(Template)]
#[template(path = "embedded_resource_text.txt", escape = "none")]
struct EmbeddedResourceTextTemplate<'a> {
    uri: &'a str,
    mime_type: Option<&'a str>,
    text: &'a str,
}

/// A binary resource reported without its payload.
#[derive(Template)]
#[template(path = "embedded_resource_binary.txt", escape = "none")]
struct EmbeddedResourceBinaryTemplate<'a> {
    uri: &'a str,
    mime_type: Option<&'a str>,
    byte_len: usize,
}

/// A resource link rendered for the model.
#[derive(Template)]
#[template(path = "resource_link.txt", escape = "none")]
struct ResourceLinkTemplate<'a> {
    uri: &'a str,
    name: &'a str,
    title: Option<&'a str>,
    description: Option<&'a str>,
    mime_type: Option<&'a str>,
    size: Option<u64>,
}

/// Render an embedded [`ContentBlock::Resource`] for the model.
fn format_embedded_resource(
    resource: &crate::domain::EmbeddedResource,
) -> Result<String, RpcError> {
    match &resource.resource {
        crate::domain::ResourceContents::Text(text) => Ok(EmbeddedResourceTextTemplate {
            uri: &text.uri,
            mime_type: text.mime_type.as_deref(),
            text: &text.text,
        }
        .render()
        .expect("embedded text resource template must render")),
        crate::domain::ResourceContents::Blob(blob) => {
            let bytes = BASE64.decode(&blob.blob).map_err(|error| {
                RpcError::invalid_params(format!(
                    "invalid base64 in embedded resource '{}': {error}",
                    blob.uri
                ))
            })?;
            Ok(std::str::from_utf8(&bytes).map_or_else(
                |_| {
                    EmbeddedResourceBinaryTemplate {
                        uri: &blob.uri,
                        mime_type: blob.mime_type.as_deref(),
                        byte_len: bytes.len(),
                    }
                    .render()
                    .expect("embedded binary resource template must render")
                },
                |text| {
                    EmbeddedResourceTextTemplate {
                        uri: &blob.uri,
                        mime_type: blob.mime_type.as_deref(),
                        text,
                    }
                    .render()
                    .expect("embedded text resource template must render")
                },
            ))
        }
    }
}

/// Render a [`ContentBlock::ResourceLink`] for the model.
fn format_resource_link(link: &crate::domain::ResourceLink) -> String {
    ResourceLinkTemplate {
        uri: &link.uri,
        name: &link.name,
        title: link.title.as_deref(),
        description: link.description.as_deref(),
        mime_type: link.mime_type.as_deref(),
        size: link.size,
    }
    .render()
    .expect("resource link template must render")
}

/// Render a prompt's text blocks, embedded resources, and resource links
/// for the model — the `embeddedContext` capability this host advertises
/// means the client may send all three.
///
/// # Errors
///
/// `invalid params` when an embedded blob resource is not valid base64.
fn prompt_text(prompt: &[ContentBlock]) -> Result<String, RpcError> {
    let mut rendered = Vec::with_capacity(prompt.len());
    for block in prompt {
        match block {
            ContentBlock::Text(text) => rendered.push(text.text.clone()),
            ContentBlock::Resource(resource) => {
                rendered.push(format_embedded_resource(resource)?);
            }
            ContentBlock::ResourceLink(link) => rendered.push(format_resource_link(link)),
            // Image, audio, and extension blocks have no text rendering.
            _ => {}
        }
    }
    Ok(rendered.join("\n"))
}

/// An external ACP session run as a subagent.
///
/// [`AcpChildAgent`] implements aither-agent's [`ChildAgent`], so a session
/// of an agent behind ACP (such as `devin acp`) runs through
/// [`SubagentRun::run`](aither_agent::SubagentRun::run) exactly as an
/// in-process [`Agent`] does: the session's events are forwarded to the
/// parent's stream as [`AgentEvent`]s tagged with the child's
/// [`AgentPath`], and the parent's cancel signal sends `session/cancel`.
///
/// Session events come off the [`ClientSession`] itself —
/// [`ClientSession::prompt`] yields the turn's [`SessionEvent`] stream,
/// which is what the old `UpdateSink` fed the previous implementation.
///
/// # Events
///
/// Session events become these events; every other event has none:
///
/// | Session event | Agent event |
/// | --- | --- |
/// | `MessageDelta` agent/thought (text) | `BlockDelta`, `Text` or `Reasoning`; a `BlockEnded` closes the block when the kind changes, a tool call starts or the run ends |
/// | `ToolCall` snapshot, first seen | `ToolCallStarted` (local; the title is the name), then what its status implies |
/// | `ToolCall` snapshot, in progress | `ToolCallDecided`, `Allowed`, once per call |
/// | `ToolCall` snapshot, completed/failed/cancelled | `ToolCallEnded`, once per call |
///
/// The run is bracketed by `RunStarted` and `RunEnded`. ACP has no
/// requests, turns, tool progress, pauses, compaction or context
/// injection, so none of those events are emitted; plans, modes,
/// commands, usage and session-info events are dropped.
#[derive(Debug)]
pub struct AcpChildAgent {
    session: ClientSession,
    path: AgentPath,
}

impl AcpChildAgent {
    /// Runs `session` as the child `identity` names.
    ///
    /// The session is created (and authenticated) by the embedder; this
    /// drives its prompts. One [`run`](ChildAgent::run) is one
    /// `session/prompt`.
    #[must_use]
    pub fn new(session: ClientSession, identity: &AgentIdentity) -> Self {
        Self {
            session,
            path: AgentPath::of(identity),
        }
    }

    /// The session the child runs on.
    #[must_use]
    pub const fn session(&self) -> &ClientSession {
        &self.session
    }

    fn emit(&self, events: &ChildEvents, event: AgentEvent) {
        events.forward(TimedEvent::now(self.path.clone(), event));
    }

    fn end(&self, events: &ChildEvents, run: RunId, outcome: RunOutcome) -> RunOutcome {
        self.emit(events, AgentEvent::run_ended(run, outcome.clone()));
        outcome
    }
}

use aither_agent::RunId;

/// What one iteration of the run loop waited for.
enum Step {
    /// The next turn event (or stream end).
    Event(Option<Box<SessionEvent>>),
    /// The parent asked to cancel.
    Cancelled,
}

impl ChildAgent for AcpChildAgent {
    async fn run(
        &mut self,
        prompt: &str,
        events: &ChildEvents,
        cancel: CancelSignal,
    ) -> RunOutcome {
        let run_id = RunId::new(uuid::Uuid::new_v4().to_string());
        self.emit(
            events,
            AgentEvent::run_started(run_id.clone(), self.path.clone()),
        );

        let mut turn = match self.session.prompt(prompt) {
            Ok(turn) => turn,
            Err(error) => return self.end(events, run_id, failed_client(&error)),
        };

        let mut translator = Translator::new();
        let mut cancel_sent = false;
        loop {
            let step = or(
                async { Step::Event(turn.next().await.map(Box::new)) },
                async {
                    if cancel_sent {
                        std::future::pending::<()>().await;
                    } else {
                        cancel.cancelled().await;
                    }
                    Step::Cancelled
                },
            )
            .await;
            match step {
                Step::Event(Some(event)) => {
                    for event in translator.translate(&event) {
                        self.emit(events, event);
                    }
                }
                Step::Event(None) => break,
                Step::Cancelled => {
                    cancel_sent = true;
                    // The agent answers the pending prompt `cancelled`.
                    if self.session.cancel().await.is_err() {
                        break;
                    }
                }
            }
        }
        if let Some(event) = translator.close_block() {
            self.emit(events, event);
        }
        let outcome = outcome(turn.finish().await, translator.reply());
        self.end(events, run_id, outcome)
    }
}

fn failed_client(error: &ClientError) -> RunOutcome {
    RunOutcome::Failed {
        error: AgentError::llm_local(error.to_string()),
    }
}

/// Maps a finished turn's outcome onto a run's.
fn outcome(outcome: TurnOutcome, reply: String) -> RunOutcome {
    match outcome {
        TurnOutcome::Cancelled => RunOutcome::Cancelled,
        TurnOutcome::Rejected(error) => RunOutcome::Failed {
            error: AgentError::llm_local(error.to_string()),
        },
        TurnOutcome::Finished {
            stop_reason,
            text,
            error,
            ..
        } => {
            let reply = if text.is_empty() { reply } else { text };
            match stop_reason {
                Some(Open::Known(StopReason::Cancelled)) => RunOutcome::Cancelled,
                Some(Open::Known(StopReason::MaxTurnRequests)) => RunOutcome::IterationLimit,
                Some(Open::Known(StopReason::Error)) => RunOutcome::Failed {
                    error: AgentError::llm_local(error.map_or_else(
                        || "the external agent reported an error".to_string(),
                        |error| error.to_string(),
                    )),
                },
                Some(Open::Known(StopReason::MaxTokens)) => RunOutcome::Failed {
                    error: AgentError::llm_local("the external agent reached its token limit"),
                },
                Some(Open::Known(StopReason::Refusal)) => RunOutcome::Failed {
                    error: AgentError::llm_local("the external agent refused to respond"),
                },
                // `end_turn`, an absent reason, and extension reasons all
                // mean the turn ended normally.
                _ => RunOutcome::Completed { final_text: reply },
            }
        }
    }
}

/// Turns the events of one prompt into agent events.
#[derive(Debug)]
struct Translator {
    request: aither_agent::RequestId,
    next_block: u32,
    open_block: Option<(BlockId, OutputKind)>,
    /// Tool calls seen and not ended, with whether their decision was reported.
    calls: HashMap<String, bool>,
    /// The message text so far per message id, in first-seen order: the
    /// run's final text.
    texts: Vec<(MessageId, String)>,
}

impl Translator {
    fn new() -> Self {
        Self {
            request: aither_agent::RequestId::new(1),
            next_block: 0,
            open_block: None,
            calls: HashMap::new(),
            texts: Vec::new(),
        }
    }

    /// The run's reply: the agent messages' concatenated text.
    fn reply(&self) -> String {
        self.texts.iter().map(|(_, text)| text.as_str()).collect()
    }

    fn translate(&mut self, event: &SessionEvent) -> Vec<AgentEvent> {
        let mut events = Vec::new();
        match event {
            SessionEvent::MessageDelta {
                message_id,
                kind,
                text,
                ..
            } => {
                let kind = match kind {
                    crate::client::MessageKind::Agent => OutputKind::Text,
                    crate::client::MessageKind::Thought => OutputKind::Reasoning,
                    _ => return events,
                };
                if let Some(text) = text {
                    self.chunk(message_id.clone(), kind, text.clone(), &mut events);
                }
            }
            SessionEvent::ToolCall(call) => {
                events.extend(self.close_block());
                self.tool_call(call, &mut events);
            }
            _ => {}
        }
        events
    }

    fn chunk(
        &mut self,
        message_id: MessageId,
        kind: OutputKind,
        text: String,
        events: &mut Vec<AgentEvent>,
    ) {
        if self.open_block.is_some_and(|(_, open)| open != kind) {
            events.extend(self.close_block());
        }
        let block = if let Some((block, _)) = self.open_block {
            block
        } else {
            let block = BlockId::new(self.next_block);
            self.next_block += 1;
            self.open_block = Some((block, kind));
            block
        };
        if kind == OutputKind::Text {
            match self.texts.iter_mut().find(|(id, _)| *id == message_id) {
                Some((_, whole)) => whole.push_str(&text),
                None => self.texts.push((message_id, text.clone())),
            }
        }
        events.push(AgentEvent::block_delta(self.request, block, kind, text));
    }

    /// Ends the open block, if there is one.
    fn close_block(&mut self) -> Option<AgentEvent> {
        let (block, kind) = self.open_block.take()?;
        Some(AgentEvent::block_ended(self.request, block, kind))
    }

    fn tool_call(&mut self, call: &ToolCall, events: &mut Vec<AgentEvent>) {
        let id = aither_core::llm::ToolCallId::new(call.tool_call_id.as_str());
        if self
            .calls
            .insert(call.tool_call_id.to_string(), false)
            .is_none()
        {
            let name = call.name.clone().unwrap_or_else(|| call.title.clone());
            events.push(AgentEvent::tool_call_started(
                id.clone(),
                name,
                call.raw_input.clone().unwrap_or(Value::Null),
                ToolOrigin::Local,
            ));
        }
        if let Some(status) = &call.status {
            self.status(&id, status, &call.content, call.raw_output.as_ref(), events);
        }
    }

    /// Reports what `status` says happened to the call `id`.
    fn status(
        &mut self,
        id: &aither_core::llm::ToolCallId,
        status: &Open<ToolCallStatus>,
        content: &[ToolCallContent],
        raw_output: Option<&Value>,
        events: &mut Vec<AgentEvent>,
    ) {
        let Open::Known(status) = status else {
            return;
        };
        match status {
            ToolCallStatus::Pending => {}
            ToolCallStatus::InProgress | ToolCallStatus::Paused => {
                self.decide(id, events);
            }
            ToolCallStatus::Completed | ToolCallStatus::Failed | ToolCallStatus::Cancelled => {
                self.decide(id, events);
                if self.calls.remove(id.as_str()).is_some() {
                    let text = result_text(content, raw_output);
                    let result = match (status, text) {
                        (ToolCallStatus::Completed, Some(text)) => ToolResult::text(text),
                        (ToolCallStatus::Completed, None) => ToolResult::Done,
                        (_, text) => ToolResult::error(text.unwrap_or_default()),
                    };
                    events.push(AgentEvent::tool_call_ended(id.clone(), result));
                }
            }
        }
    }

    fn decide(&mut self, id: &aither_core::llm::ToolCallId, events: &mut Vec<AgentEvent>) {
        if let Some(decided) = self.calls.get_mut(id.as_str())
            && !std::mem::replace(decided, true)
        {
            events.push(AgentEvent::tool_call_decided(
                id.clone(),
                ToolVerdict::Allowed,
            ));
        }
    }
}

/// The text of a finished call: its text content, else its raw output.
fn result_text(content: &[ToolCallContent], raw_output: Option<&Value>) -> Option<String> {
    let text: Vec<&str> = content
        .iter()
        .filter_map(|content| match content {
            ToolCallContent::Content(crate::domain::tool::Content {
                content: ContentBlock::Text(text),
                ..
            }) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    if !text.is_empty() {
        return Some(text.join("\n"));
    }
    match raw_output? {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        other => Some(other.to_string()),
    }
}

/// Converts a stream of [`TimedEvent`]s into ACP [`SessionUpdate`]s.
///
/// The mapping, event by event:
///
/// | Event | ACP update |
/// | --- | --- |
/// | `BlockDelta` | `agent_message_chunk`, or `agent_thought_chunk` for reasoning |
/// | `ToolCallStarted` | `tool_call_update`, pending; a hosted call's detail is its title |
/// | `ToolCallDelta` | `tool_call_update` carrying the arguments received so far |
/// | `ToolCallDecided` | `tool_call_update`: in progress when the call runs, failed with the reason when denied |
/// | `ToolProgress` | `tool_call_update`, in progress, with the report as content |
/// | `ToolCallEnded` | `tool_call_update`, completed or failed, with the result as raw output |
/// | `SubagentStarted` | `tool_call_update` standing for the child |
/// | `SubagentEnded` | `tool_call_update` completing or failing it |
///
/// These events have no ACP update: `RunStarted`, `RunEnded`, `TurnStarted`,
/// `TurnEnded`, `RequestStarted`, `BlockEnded` (ACP chunks have no end),
/// `Paused` and `Resumed` (ACP's permission request is the client's own
/// channel), `CompactionStarted`, `CompactionEnded`, `CompactionRecovery`,
/// `ContextInjected` and `Checkpoint`. `RequestEnded` has none either, but a
/// retrying one clears the retried request's tool arguments; ACP cannot
/// withdraw chunks already sent, which is a protocol limitation.
///
/// Events are mapped by the agent that emitted them. A child's tool calls
/// are reported under ids and titles that name its path, so they never
/// collide with the parent's; its message and reasoning chunks are dropped,
/// since ACP would show them as part of the parent's reply.
///
/// `AgentEvent::ToolCallDelta` carries incremental argument fragments; the
/// mapper accumulates them per call so each emitted `ToolCallUpdate` reports
/// the arguments received so far.
/// Optional display metadata describing how a tool's calls map onto ACP
/// [`ToolCallUpdate`]s.
///
/// Extracted from a tool's raw argument-schema extension keys
/// (`x-acp-kind`/`x-tool-kind`, `x-acp-title`/`x-tool-title`, and
/// `x-acp-title-field`/`x-title-field`) or configured directly on
/// [`SessionUpdateMapper`]. When unset, the mapper falls back to
/// [`ToolKind::Other`] and `"Running <name>"` — no product-specific tool
/// names or argument fields are assumed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolMetadata {
    kind: Option<ToolKind>,
    title: Option<String>,
    title_field: Option<String>,
}

impl ToolMetadata {
    /// Empty metadata (uses generic fallbacks).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            kind: None,
            title: None,
            title_field: None,
        }
    }

    /// Sets the ACP [`ToolKind`] reported for calls to this tool.
    #[must_use]
    pub const fn with_kind(mut self, kind: ToolKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Sets a static human-readable title fallback for calls to this tool.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        let title = title.into();
        if !title.is_empty() {
            self.title = Some(title);
        }
        self
    }

    /// Names the JSON object argument property whose non-empty string
    /// value titles a call to this tool.
    #[must_use]
    pub fn with_title_field(mut self, field: impl Into<String>) -> Self {
        let field = field.into();
        if !field.is_empty() {
            self.title_field = Some(field);
        }
        self
    }

    /// Reads optional ACP display metadata from a tool's raw JSON
    /// argument schema (`x-acp-kind`/`x-tool-kind`, `x-acp-title`/
    /// `x-tool-title`, and `x-acp-title-field`/`x-title-field`).
    #[must_use]
    pub fn from_schema(schema: &Value) -> Self {
        let Some(schema) = schema.as_object() else {
            return Self::new();
        };

        let kind = schema
            .get("x-acp-kind")
            .or_else(|| schema.get("x-tool-kind"))
            .and_then(|val| serde_json::from_value::<ToolKind>(val.clone()).ok());

        let title = schema
            .get("x-acp-title")
            .or_else(|| schema.get("x-tool-title"))
            .and_then(serde_json::Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        let title_field = schema
            .get("x-acp-title-field")
            .or_else(|| schema.get("x-title-field"))
            .and_then(serde_json::Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        Self {
            kind,
            title,
            title_field,
        }
    }

    /// Reads the same keys from an [`McpToolDefinition`]'s raw input
    /// schema — the form extension keys survive in, since provider-facing
    /// schemas are cleaned of `x-*` keys.
    #[must_use]
    pub fn from_mcp_definition(definition: &McpToolDefinition) -> Self {
        Self::from_schema(&definition.input_schema)
    }
}

/// Converts a stream of [`TimedEvent`]s into ACP [`SessionUpdate`]s.
#[derive(Debug, Default)]
pub struct SessionUpdateMapper {
    /// Arguments accumulated per tool call, keyed by the agent and call id.
    tool_arguments: HashMap<(AgentPath, String), String>,
    /// Message ids allocated per response block, keyed by the request and
    /// block ids, so one block's chunks append to one message.
    messages: HashMap<(u64, u32), MessageId>,
    /// Optional per-tool display metadata keyed by tool name.
    tool_metadata: HashMap<String, ToolMetadata>,
}

impl SessionUpdateMapper {
    /// A mapper pre-populated with display metadata extracted from
    /// `definitions`.
    #[must_use]
    pub fn from_definitions<'a>(
        definitions: impl IntoIterator<Item = &'a McpToolDefinition>,
    ) -> Self {
        Self::default().with_definitions(definitions)
    }

    /// Registers display metadata extracted from each
    /// [`McpToolDefinition`] in `definitions`.
    #[must_use]
    pub fn with_definitions<'a>(
        mut self,
        definitions: impl IntoIterator<Item = &'a McpToolDefinition>,
    ) -> Self {
        for def in definitions {
            let meta = ToolMetadata::from_mcp_definition(def);
            if meta != ToolMetadata::default() {
                self.tool_metadata.insert(def.name.clone(), meta);
            }
        }
        self
    }

    /// Attaches explicit [`ToolMetadata`] for `name`.
    #[must_use]
    pub fn with_tool_metadata(mut self, name: impl Into<String>, metadata: ToolMetadata) -> Self {
        self.register_tool_metadata(name, metadata);
        self
    }

    /// Inserts or replaces [`ToolMetadata`] for `name`.
    pub fn register_tool_metadata(&mut self, name: impl Into<String>, metadata: ToolMetadata) {
        self.tool_metadata.insert(name.into(), metadata);
    }

    /// The [`ToolKind`] reported for calls to `name`.
    fn kind_for(&self, name: &str) -> ToolKind {
        self.tool_metadata
            .get(name)
            .and_then(|meta| meta.kind)
            .unwrap_or(ToolKind::Other)
    }

    /// The title reported for a call to `name` with `arguments`.
    fn title_for(&self, agent: &AgentPath, name: &str, arguments: &str) -> String {
        format_tool_title(agent, name, arguments, self.tool_metadata.get(name))
    }

    /// Convert a `TimedEvent` to an ACP `SessionUpdate`.
    ///
    /// Returns `None` for events that have no ACP mapping; see the type's
    /// documentation for which.
    #[must_use]
    pub fn map(&mut self, timed: &TimedEvent) -> Option<SessionUpdate> {
        let agent = &timed.agent;
        match &timed.event {
            AgentEvent::BlockDelta {
                request,
                block,
                kind,
                text,
                ..
            } => {
                if !agent.is_main() {
                    return None;
                }
                let message_id = match self.messages.entry((request.get(), block.get())) {
                    std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(MessageId::new()).clone()
                    }
                };
                let chunk = ContentChunk::new(message_id, ContentBlock::text(text.clone()));
                Some(match kind {
                    OutputKind::Reasoning => SessionUpdate::AgentThoughtChunk(chunk),
                    _ => SessionUpdate::AgentMessageChunk(chunk),
                })
            }

            AgentEvent::ToolCallStarted {
                call,
                name,
                arguments,
                origin,
                ..
            } => Some(self.tool_call_started(agent, call, name, arguments, *origin)),

            AgentEvent::ToolCallDelta {
                call,
                name,
                arguments_fragment,
                ..
            } => Some(self.tool_call_delta(agent, call, name, arguments_fragment)),

            AgentEvent::ToolCallDecided { call, decision, .. } => {
                let (status, content) = decision_update(decision)?;
                let mut update = update(call_id(agent, call.as_str()));
                update.status = Patch::Set(Open::Known(status));
                if let Some(content) = content {
                    update.content = Patch::Set(content);
                }
                Some(SessionUpdate::ToolCallUpdate(update))
            }

            AgentEvent::ToolProgress { call, progress, .. } => {
                let mut update = update(call_id(agent, call.as_str()));
                update.status = Patch::Set(Open::Known(ToolCallStatus::InProgress));
                update.content = Patch::Set(vec![text_content(progress_text(progress))]);
                Some(SessionUpdate::ToolCallUpdate(update))
            }

            AgentEvent::ToolCallEnded { call, result, .. } => {
                Some(self.tool_call_ended(agent, call, result))
            }

            // The broken attempt's calls never run and the replacement
            // streams its own, so the arguments this agent accumulated are
            // dropped. ACP has no update that withdraws streamed message
            // chunks.
            AgentEvent::RequestEnded {
                outcome: RequestOutcome::Retrying { .. },
                ..
            } => {
                self.tool_arguments.retain(|(owner, _), _| owner != agent);
                None
            }

            AgentEvent::SubagentStarted {
                child,
                name,
                prompt,
                ..
            } => {
                let mut update = update(subagent_call_id(child));
                update.status = Patch::Set(Open::Known(ToolCallStatus::InProgress));
                update.title = Patch::Set(format!("Subagent {name}"));
                update.kind = Patch::Set(Open::Known(ToolKind::Other));
                update.raw_input = Patch::Set(Value::String(prompt.clone()));
                Some(SessionUpdate::ToolCallUpdate(update))
            }

            AgentEvent::SubagentEnded { child, outcome, .. } => {
                let status = match outcome {
                    RunOutcome::Completed { .. } => ToolCallStatus::Completed,
                    _ => ToolCallStatus::Failed,
                };
                let mut update = update(subagent_call_id(child));
                update.status = Patch::Set(Open::Known(status));
                Some(SessionUpdate::ToolCallUpdate(update))
            }

            _ => None,
        }
    }

    /// A `tool_call_update` announcing a call.
    fn tool_call_started(
        &mut self,
        agent: &AgentPath,
        call: &aither_core::llm::ToolCallId,
        name: &str,
        arguments: &Value,
        origin: ToolOrigin,
    ) -> SessionUpdate {
        // A hosted call reports its detail as a string, not JSON arguments.
        let (title_arguments, raw_input) = match origin {
            ToolOrigin::Hosted => (arguments.as_str().unwrap_or_default().to_string(), None),
            _ => (arguments.to_string(), Some(arguments.clone())),
        };
        self.tool_arguments.insert(
            (agent.clone(), call.as_str().to_string()),
            title_arguments.clone(),
        );
        let mut update = update(call_id(agent, call.as_str()));
        update.status = Patch::Set(Open::Known(ToolCallStatus::Pending));
        update.name = Patch::Set(name.to_string());
        update.title = Patch::Set(self.title_for(agent, name, &title_arguments));
        update.kind = Patch::Set(Open::Known(self.kind_for(name)));
        if let Some(raw_input) = raw_input {
            update.raw_input = Patch::Set(raw_input);
        }
        SessionUpdate::ToolCallUpdate(update)
    }

    /// A `tool_call_update` carrying the arguments received so far.
    fn tool_call_delta(
        &mut self,
        agent: &AgentPath,
        call: &aither_core::llm::ToolCallId,
        name: &str,
        arguments_fragment: &str,
    ) -> SessionUpdate {
        let arguments = {
            let entry = self
                .tool_arguments
                .entry((agent.clone(), call.as_str().to_string()))
                .or_default();
            entry.push_str(arguments_fragment);
            entry.clone()
        };
        let mut update = update(call_id(agent, call.as_str()));
        update.status = Patch::Set(Open::Known(ToolCallStatus::Pending));
        update.name = Patch::Set(name.to_string());
        update.title = Patch::Set(self.title_for(agent, name, &arguments));
        update.kind = Patch::Set(Open::Known(self.kind_for(name)));
        update.raw_input = Patch::Set(Value::String(arguments));
        SessionUpdate::ToolCallUpdate(update)
    }

    /// A `tool_call_update` closing a call with its result.
    fn tool_call_ended(
        &mut self,
        agent: &AgentPath,
        call: &aither_core::llm::ToolCallId,
        result: &ToolResult,
    ) -> SessionUpdate {
        self.tool_arguments
            .remove(&(agent.clone(), call.as_str().to_string()));
        let mut update = update(call_id(agent, call.as_str()));
        update.status = Patch::Set(Open::Known(tool_result_status(result)));
        update.content = Patch::Set(vec![text_content(result.render_for_model())]);
        update.raw_output = Patch::Set(tool_result_raw_output(result));
        SessionUpdate::ToolCallUpdate(update)
    }
}

/// An update to `tool_call_id` that changes nothing yet.
fn update(tool_call_id: String) -> ToolCallUpdate {
    ToolCallUpdate::new(tool_call_id)
}

/// The id a call is reported under: the provider's own for the main agent,
/// qualified by the agent's path for a child, whose provider may reuse it.
fn call_id(agent: &AgentPath, call: &str) -> String {
    if agent.is_main() {
        call.to_string()
    } else {
        format!("{agent}:{call}")
    }
}

fn subagent_call_id(child: &AgentPath) -> String {
    format!("subagent:{child}")
}

/// A text block as tool-call content.
fn text_content(text: String) -> ToolCallContent {
    ToolCallContent::Content(crate::domain::tool::Content {
        content: ContentBlock::text(text),
        meta: None,
        extra: std::collections::BTreeMap::new(),
    })
}

/// What a progress report says, for a reader.
fn progress_text(progress: &aither_core::llm::tool::Progress) -> String {
    if let Some(message) = progress.message() {
        return message.to_string();
    }
    progress.total().map_or_else(
        || progress.progress().to_string(),
        |total| format!("{} of {total}", progress.progress()),
    )
}

const fn tool_result_status(result: &ToolResult) -> ToolCallStatus {
    if result.is_error() {
        ToolCallStatus::Failed
    } else {
        ToolCallStatus::Completed
    }
}

fn tool_result_raw_output(result: &ToolResult) -> Value {
    match serde_json::to_value(result) {
        Ok(value) => value,
        Err(error) => {
            panic!("failed to serialize ToolResult for ACP raw_output: {error}");
        }
    }
}

/// What a decision about a call changes in its tool call; `None` while it
/// stays pending.
fn decision_update(
    verdict: &ToolVerdict,
) -> Option<(ToolCallStatus, Option<Vec<ToolCallContent>>)> {
    match verdict {
        ToolVerdict::Allowed | ToolVerdict::Rewritten => Some((ToolCallStatus::InProgress, None)),
        ToolVerdict::Denied { reason } => Some((
            ToolCallStatus::Failed,
            Some(vec![text_content(reason.clone())]),
        )),
        // The call stays pending until the answer arrives as another
        // decision.
        _ => None,
    }
}

/// Maximum characters shown in a tool-call title before it is truncated.
const TOOL_TITLE_MAX_CHARS: usize = 50;

/// Truncate `text` to [`TOOL_TITLE_MAX_CHARS`] characters on a char boundary,
/// appending an ellipsis when anything was cut.
fn truncate_title(text: &str) -> String {
    match text.char_indices().nth(TOOL_TITLE_MAX_CHARS) {
        Some((byte_index, _)) => format!("{}...", &text[..byte_index]),
        None => text.to_string(),
    }
}

/// Format a human-readable title for a tool call from optional
/// [`ToolMetadata`], falling back to `"Running <name>"` when no metadata
/// is configured. A child's calls are prefixed with the child's path.
fn format_tool_title(
    agent: &AgentPath,
    name: &str,
    arguments: &str,
    metadata: Option<&ToolMetadata>,
) -> String {
    let dynamic_title = metadata
        .and_then(|meta| meta.title_field.as_deref())
        .and_then(|field| match serde_json::from_str(arguments) {
            Ok(Value::Object(map)) => map
                .get(field)
                .and_then(Value::as_str)
                .filter(|val| !val.is_empty())
                .map(truncate_title),
            _ => None,
        });
    let title = dynamic_title
        .or_else(|| metadata.and_then(|meta| meta.title.clone()))
        .unwrap_or_else(|| format!("Running {name}"));
    if agent.is_main() {
        title
    } else {
        format!("[{agent}] {title}")
    }
}

#[cfg(test)]
mod tests;
