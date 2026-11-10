//! An external ACP session run as a subagent.
//!
//! [`AcpChildAgent`] implements aither-agent's [`ChildAgent`], so a session of
//! an agent behind ACP (such as `devin acp`) runs through
//! [`SubagentRun::run`](aither_agent::SubagentRun::run) exactly as an
//! in-process [`Agent`](aither_agent::Agent) does: the session's updates are
//! forwarded to the parent's stream as [`AgentEvent`]s tagged with the
//! child's [`AgentPath`], and the parent's cancel signal sends
//! `session/cancel`.
//!
//! The connection's [`ClientHandler`] receives the session's updates; it
//! hands them to the child through an [`UpdateSink`] (see
//! [`UpdateSink::channel`]).
//!
//! # Events
//!
//! ACP updates become these events; every other update has none:
//!
//! | ACP update | Event |
//! | --- | --- |
//! | `agent_message_chunk`, `agent_thought_chunk` (text) | `BlockDelta`, `Text` or `Reasoning`; a `BlockEnded` closes the block when the kind changes, a tool call starts or the run ends |
//! | `tool_call` | `ToolCallStarted` (local; the title is the name), then what its status implies |
//! | `tool_call_update`, in progress | `ToolCallDecided`, `Allowed`, once per call |
//! | `tool_call_update`, completed or failed | `ToolCallEnded`, once per call |
//!
//! The run is bracketed by `RunStarted` and `RunEnded`. ACP has no requests,
//! turns, tool progress, pauses, compaction or context injection, so none of
//! those events are emitted, and a chunk that is not text, a plan, a mode,
//! command, usage or session-info update is dropped.

use std::collections::HashMap;
use std::future::pending;

use aither_agent::{
    AgentError, AgentEvent, AgentPath, BlockId, ChildAgent, ChildEvents, OutputKind, RequestId,
    RunId, RunOutcome, TimedEvent, ToolVerdict,
};
use aither_core::llm::tool::CancelSignal;
use aither_core::llm::{AgentIdentity, ToolCallId, ToolResult};
use futures_lite::future::or;

use crate::client::{AcpClient, ClientError, ClientHandler};
use crate::protocol::{
    ContentBlock, PromptParams, PromptResult, SessionCancelParams, SessionNotification,
    SessionUpdate, StopReason, TextContent, ToolCall, ToolCallContent, ToolCallStatus,
    ToolCallUpdate,
};

/// Where a [`ClientHandler`] puts the session updates it receives, for an
/// [`AcpChildAgent`] to forward.
#[derive(Debug, Clone)]
pub struct UpdateSink {
    sender: async_channel::Sender<SessionNotification>,
}

/// The updates an [`UpdateSink`] collected, read by an [`AcpChildAgent`].
#[derive(Debug)]
pub struct UpdateSource {
    receiver: async_channel::Receiver<SessionNotification>,
}

impl UpdateSink {
    /// A sink and the source that reads what is pushed into it.
    #[must_use]
    pub fn channel() -> (Self, UpdateSource) {
        let (sender, receiver) = async_channel::unbounded();
        (Self { sender }, UpdateSource { receiver })
    }

    /// Hands one `session/update` notification to the child. Call it from
    /// [`ClientHandler::session_update`], which is called in the order the
    /// agent sent the updates.
    pub fn push(&self, notification: SessionNotification) {
        // The source is gone only when the child agent is, and nothing reads
        // the update then.
        let _ = self.sender.try_send(notification);
    }
}

/// An ACP session that runs as a subagent.
///
/// The session is created and authenticated by the embedder; this drives its
/// prompts. One [`run`](ChildAgent::run) is one `session/prompt`.
#[derive(Debug)]
pub struct AcpChildAgent<H: ClientHandler> {
    client: AcpClient<H>,
    session_id: String,
    path: AgentPath,
    updates: UpdateSource,
}

impl<H: ClientHandler> AcpChildAgent<H> {
    /// Runs `session_id` of `client` as the child `identity` names, reading
    /// the session's updates from `updates`, the source of the sink the
    /// client's handler pushes into.
    #[must_use]
    pub fn new(
        client: AcpClient<H>,
        session_id: impl Into<String>,
        identity: &AgentIdentity,
        updates: UpdateSource,
    ) -> Self {
        Self {
            client,
            session_id: session_id.into(),
            path: AgentPath::of(identity),
            updates,
        }
    }

    /// The ACP client the session runs on.
    #[must_use]
    pub const fn client(&self) -> &AcpClient<H> {
        &self.client
    }

    fn emit(&self, events: &ChildEvents, event: AgentEvent) {
        events.forward(TimedEvent::now(self.path.clone(), event));
    }

    fn forward(
        &self,
        events: &ChildEvents,
        notification: SessionNotification,
        translator: &mut Translator,
    ) {
        if notification.session_id != self.session_id {
            return;
        }
        for event in translator.translate(notification.update) {
            self.emit(events, event);
        }
    }
}

/// What one iteration of the run loop waited for.
enum Step {
    Update(Option<Box<SessionNotification>>),
    Response(Result<PromptResult, ClientError>),
    Cancelled,
}

impl<H: ClientHandler> ChildAgent for AcpChildAgent<H> {
    async fn run(
        &mut self,
        prompt: &str,
        events: &ChildEvents,
        cancel: CancelSignal,
    ) -> RunOutcome {
        let run_id = RunId::new(uuid::Uuid::new_v4().to_string());
        // What an earlier prompt left behind is not this run's.
        while self.updates.receiver.try_recv().is_ok() {}
        self.emit(
            events,
            AgentEvent::run_started(run_id.clone(), self.path.clone()),
        );

        let params = PromptParams::new(
            &self.session_id,
            vec![ContentBlock::Text(TextContent {
                text: prompt.to_string(),
                annotations: None,
                meta: None,
            })],
        );
        let mut response = match self.client.start_prompt(&params).await {
            Ok(response) => response,
            Err(error) => return self.end(events, run_id, failed(&error)),
        };

        let mut translator = Translator::new();
        let mut cancel_sent = false;
        let result = loop {
            let step = or(
                async { Step::Update(self.updates.receiver.recv().await.ok().map(Box::new)) },
                or(async { Step::Response((&mut response).await) }, async {
                    if cancel_sent {
                        pending::<()>().await;
                    } else {
                        cancel.cancelled().await;
                    }
                    Step::Cancelled
                }),
            )
            .await;
            match step {
                Step::Update(Some(notification)) => {
                    self.forward(events, *notification, &mut translator);
                }
                Step::Update(None) => {
                    break Err(ClientError::Closed { status: None });
                }
                Step::Response(result) => break result,
                Step::Cancelled => {
                    cancel_sent = true;
                    // The agent answers the pending prompt `cancelled`.
                    if let Err(error) = self
                        .client
                        .cancel(SessionCancelParams::new(&self.session_id))
                        .await
                    {
                        break Err(error);
                    }
                }
            }
        };
        // Updates the agent sent before it answered.
        while let Ok(notification) = self.updates.receiver.try_recv() {
            self.forward(events, notification, &mut translator);
        }
        if let Some(event) = translator.close_block() {
            self.emit(events, event);
        }
        let outcome = match result {
            Ok(result) => outcome(result.stop_reason, translator.reply),
            Err(error) => failed(&error),
        };
        self.end(events, run_id, outcome)
    }
}

impl<H: ClientHandler> AcpChildAgent<H> {
    fn end(&self, events: &ChildEvents, run: RunId, outcome: RunOutcome) -> RunOutcome {
        self.emit(events, AgentEvent::run_ended(run, outcome.clone()));
        outcome
    }
}

fn failed(error: &ClientError) -> RunOutcome {
    RunOutcome::Failed {
        error: AgentError::llm_local(error.to_string()),
    }
}

fn outcome(stop: StopReason, reply: String) -> RunOutcome {
    match stop {
        StopReason::EndTurn => RunOutcome::Completed { final_text: reply },
        StopReason::Cancelled => RunOutcome::Cancelled,
        StopReason::MaxTurnRequests => RunOutcome::IterationLimit,
        StopReason::Error => RunOutcome::Failed {
            error: AgentError::llm_local("the external agent reported an error"),
        },
        StopReason::MaxTokens => RunOutcome::Failed {
            error: AgentError::llm_local("the external agent reached its token limit"),
        },
        StopReason::Refusal => RunOutcome::Failed {
            error: AgentError::llm_local("the external agent refused to respond"),
        },
    }
}

/// Turns the updates of one prompt into events.
#[derive(Debug)]
struct Translator {
    request: RequestId,
    next_block: u32,
    open_block: Option<(BlockId, OutputKind)>,
    /// Calls seen and not ended, with whether their decision was reported.
    calls: HashMap<String, bool>,
    /// The message text so far: the run's final text.
    reply: String,
}

impl Translator {
    fn new() -> Self {
        Self {
            request: RequestId::new(1),
            next_block: 0,
            open_block: None,
            calls: HashMap::new(),
            reply: String::new(),
        }
    }

    fn translate(&mut self, update: SessionUpdate) -> Vec<AgentEvent> {
        let mut events = Vec::new();
        match update {
            SessionUpdate::AgentMessageChunk(chunk) => {
                self.chunk(OutputKind::Text, chunk.content, &mut events);
            }
            SessionUpdate::AgentThoughtChunk(chunk) => {
                self.chunk(OutputKind::Reasoning, chunk.content, &mut events);
            }
            SessionUpdate::ToolCall(call) => {
                events.extend(self.close_block());
                self.tool_call(call, &mut events);
            }
            SessionUpdate::ToolCallUpdate(update) => self.tool_call_update(&update, &mut events),
            _ => {}
        }
        events
    }

    fn chunk(&mut self, kind: OutputKind, content: ContentBlock, events: &mut Vec<AgentEvent>) {
        let ContentBlock::Text(TextContent { text, .. }) = content else {
            return;
        };
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
            self.reply.push_str(&text);
        }
        events.push(AgentEvent::block_delta(self.request, block, kind, text));
    }

    /// Ends the open block, if there is one.
    fn close_block(&mut self) -> Option<AgentEvent> {
        let (block, kind) = self.open_block.take()?;
        Some(AgentEvent::block_ended(self.request, block, kind))
    }

    fn tool_call(&mut self, call: ToolCall, events: &mut Vec<AgentEvent>) {
        let id = ToolCallId::new(&call.tool_call_id);
        events.push(AgentEvent::tool_call_started(
            id.clone(),
            call.title,
            call.raw_input.unwrap_or(serde_json::Value::Null),
            aither_agent::ToolOrigin::Local,
        ));
        self.calls.insert(call.tool_call_id, false);
        if let Some(status) = call.status {
            self.status(&id, status, &call.content, call.raw_output.as_ref(), events);
        }
    }

    fn tool_call_update(&mut self, update: &ToolCallUpdate, events: &mut Vec<AgentEvent>) {
        // An update for a call never announced has nothing to attach to.
        if !self.calls.contains_key(&update.tool_call_id) {
            return;
        }
        if let Some(status) = update.status {
            let id = ToolCallId::new(&update.tool_call_id);
            self.status(
                &id,
                status,
                update.content.as_deref().unwrap_or_default(),
                update.raw_output.as_ref(),
                events,
            );
        }
    }

    /// Reports what `status` says happened to the call `id`.
    fn status(
        &mut self,
        id: &ToolCallId,
        status: ToolCallStatus,
        content: &[ToolCallContent],
        raw_output: Option<&serde_json::Value>,
        events: &mut Vec<AgentEvent>,
    ) {
        match status {
            ToolCallStatus::Pending => {}
            ToolCallStatus::InProgress => self.decide(id, events),
            ToolCallStatus::Completed | ToolCallStatus::Failed => {
                self.decide(id, events);
                if self.calls.remove(id.as_str()).is_some() {
                    let text = result_text(content, raw_output);
                    let result = match (status, text) {
                        (ToolCallStatus::Failed, text) => {
                            ToolResult::error(text.unwrap_or_default())
                        }
                        (_, Some(text)) => ToolResult::text(text),
                        (_, None) => ToolResult::Done,
                    };
                    events.push(AgentEvent::tool_call_ended(id.clone(), result));
                }
            }
        }
    }

    fn decide(&mut self, id: &ToolCallId, events: &mut Vec<AgentEvent>) {
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
fn result_text(
    content: &[ToolCallContent],
    raw_output: Option<&serde_json::Value>,
) -> Option<String> {
    let text: Vec<&str> = content
        .iter()
        .filter_map(|content| match content {
            ToolCallContent::Content {
                content: ContentBlock::Text(text),
            } => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    if !text.is_empty() {
        return Some(text.join("\n"));
    }
    match raw_output? {
        serde_json::Value::Null => None,
        serde_json::Value::String(text) => Some(text.clone()),
        other => Some(other.to_string()),
    }
}

#[cfg(test)]
#[path = "child_tests.rs"]
mod tests;
