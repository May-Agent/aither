//! ACP server backed by `aither_agent::Agent`, built on [`AgentHandler`].

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use aither_agent::Agent;
use aither_core::LanguageModel;
use aither_mcp::McpConnection;
use aither_mcp::transport::{BidirectionalTransport, StdioTransport};
use askama::Template;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use executor_core::{AnyExecutor, Executor};
use serde_json::Value;
use tracing::debug;

use super::{AgentHandler, ClientPeer, ServerError, drive_transport};
use crate::mcp::{ConnectError, connect_all};
use crate::protocol::{
    AcpError, AgentCapabilities, AuthenticateParams, AuthenticateResult, ContentBlock,
    EmbeddedResource, ExtMethod, Implementation, InitializeParams, InitializeResult, JsonRpcError,
    JsonRpcNotification, LogoutParams, LogoutResult, McpCapabilities, PROTOCOL_VERSION,
    PromptCapabilities, PromptParams, PromptResult, ResourceLink, Result, SessionCancelParams,
    SessionCapabilities, SessionCloseCapabilities, SessionCloseParams, SessionCloseResult,
    SessionDeleteCapabilities, SessionDeleteParams, SessionDeleteResult, SessionInfo,
    SessionListCapabilities, SessionListParams, SessionListResult, SessionLoadParams,
    SessionLoadResult, SessionNewParams, SessionNewResult, SessionNotification,
    SessionResumeParams, SessionResumeResult, SessionSetConfigOptionParams,
    SessionSetConfigOptionResult, SessionSetModeParams, SessionSetModeResult, SessionUpdate,
    StopReason,
};
use crate::session::{AcpSession, Cancellation};

/// Builds the agent backing a new session.
type AgentFactory<LLM> =
    Box<dyn Fn(SessionSetup) -> Pin<Box<dyn Future<Output = Result<Agent<LLM>>> + Send>> + Send>;

/// What the agent of a new session is built from.
#[derive(Debug)]
pub struct SessionSetup {
    /// The session's working directory.
    pub cwd: PathBuf,
    /// The MCP servers the client opened the session with, connected and in
    /// the order it listed them. The agent registers their tools.
    pub mcp: Vec<McpConnection>,
}

/// Shared state for a single live [`AcpSession`].
///
/// The session itself moves out through the 1-slot bounded channel for the
/// duration of a prompt turn while its working directory and [`Cancellation`]
/// handle stay reachable for `session/list` and `session/cancel`.
struct SessionEntry<LLM: LanguageModel + 'static> {
    cwd: PathBuf,
    cancellation: Cancellation,
    slot: (
        async_channel::Sender<AcpSession<LLM>>,
        async_channel::Receiver<AcpSession<LLM>>,
    ),
}

/// [`AgentHandler`] implementation driving per-session `aither_agent::Agent`
/// instances.
struct AitherAgentHandler<LLM: LanguageModel + 'static> {
    info: Implementation,
    sessions: Mutex<HashMap<String, Arc<SessionEntry<LLM>>>>,
    initialized: AtomicBool,
    make_agent: Mutex<AgentFactory<LLM>>,
    executor: AnyExecutor,
}

impl<LLM: LanguageModel + 'static> AitherAgentHandler<LLM> {
    fn ensure_initialized(&self) -> std::result::Result<(), JsonRpcError> {
        if self.initialized.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(JsonRpcError::invalid_request(
                "Agent server not initialized",
            ))
        }
    }

    fn create_agent(
        &self,
        setup: SessionSetup,
    ) -> Pin<Box<dyn Future<Output = Result<Agent<LLM>>> + Send>> {
        let make_agent = self
            .make_agent
            .lock()
            .expect("agent factory mutex poisoned");
        make_agent(setup)
    }

    fn insert_session(&self, session: AcpSession<LLM>) -> String {
        let session_id = session.id().to_string();
        let cwd = session.cwd().clone();
        let cancellation = session.cancellation();
        let slot = async_channel::bounded(1);
        let _ = slot.0.try_send(session);
        let entry = Arc::new(SessionEntry {
            cwd,
            cancellation,
            slot,
        });
        self.sessions
            .lock()
            .expect("sessions mutex poisoned")
            .insert(session_id.clone(), entry);
        session_id
    }

    fn get_session(&self, session_id: &str) -> Option<Arc<SessionEntry<LLM>>> {
        self.sessions
            .lock()
            .expect("sessions mutex poisoned")
            .get(session_id)
            .cloned()
    }

    fn end_session(&self, session_id: &str) -> bool {
        let removed = self
            .sessions
            .lock()
            .expect("sessions mutex poisoned")
            .remove(session_id);
        removed.is_some_and(|entry| {
            entry.cancellation.cancel();
            entry.slot.0.close();
            true
        })
    }
}

impl<LLM: LanguageModel + 'static> AgentHandler for AitherAgentHandler<LLM> {
    fn initialize(
        &self,
        _peer: &ClientPeer,
        _params: Option<InitializeParams>,
    ) -> impl Future<Output = std::result::Result<InitializeResult, JsonRpcError>> + Send {
        if self.initialized.swap(true, Ordering::AcqRel) {
            return std::future::ready(Err(JsonRpcError::invalid_request(
                "Agent server is already initialized",
            )));
        }
        let info = self.info.clone();
        std::future::ready(Ok(InitializeResult {
            protocol_version: PROTOCOL_VERSION,
            agent_capabilities: AgentCapabilities {
                load_session: false,
                prompt_capabilities: PromptCapabilities {
                    image: false,
                    audio: false,
                    embedded_context: true,
                },
                mcp_capabilities: McpCapabilities {
                    http: true,
                    sse: false,
                },
                session_capabilities: SessionCapabilities {
                    list: Some(SessionListCapabilities::default()),
                    delete: Some(SessionDeleteCapabilities::default()),
                    close: Some(SessionCloseCapabilities::default()),
                    ..SessionCapabilities::default()
                },
                ..AgentCapabilities::default()
            },
            agent_info: Some(info),
            auth_methods: vec![],
            meta: None,
        }))
    }

    fn authenticate(
        &self,
        _peer: &ClientPeer,
        _params: AuthenticateParams,
    ) -> impl Future<Output = std::result::Result<AuthenticateResult, JsonRpcError>> + Send {
        std::future::ready(
            self.ensure_initialized()
                .and_then(|()| Err(JsonRpcError::method_not_found("authenticate"))),
        )
    }

    fn logout(
        &self,
        _peer: &ClientPeer,
        _params: LogoutParams,
    ) -> impl Future<Output = std::result::Result<LogoutResult, JsonRpcError>> + Send {
        std::future::ready(
            self.ensure_initialized()
                .and_then(|()| Err(JsonRpcError::method_not_found("logout"))),
        )
    }

    async fn new_session(
        &self,
        _peer: &ClientPeer,
        params: SessionNewParams,
    ) -> std::result::Result<SessionNewResult, JsonRpcError> {
        self.ensure_initialized()?;
        let mcp = match connect_all(&self.executor, &params.mcp_servers).await {
            Ok(mcp) => mcp,
            Err(err @ ConnectError::Unsupported(_)) => {
                return Err(JsonRpcError::invalid_params(err.to_string()));
            }
            Err(err @ ConnectError::Failed { .. }) => {
                return Err(JsonRpcError::internal_error(err.to_string()));
            }
        };
        let mcp_definitions: Vec<_> = mcp.iter().flat_map(McpConnection::definitions).collect();
        let setup = SessionSetup {
            cwd: params.cwd.clone(),
            mcp,
        };
        let agent = self.create_agent(setup).await.map_err(|err| {
            JsonRpcError::internal_error(format!("could not start a session: {err}"))
        })?;

        let session = AcpSession::new(params.cwd, params.mcp_servers, agent)
            .with_tool_definitions(&mcp_definitions);
        let session_id = self.insert_session(session);

        Ok(SessionNewResult {
            session_id,
            ..SessionNewResult::default()
        })
    }

    fn load_session(
        &self,
        _peer: &ClientPeer,
        _params: SessionLoadParams,
    ) -> impl Future<Output = std::result::Result<SessionLoadResult, JsonRpcError>> + Send {
        std::future::ready(
            self.ensure_initialized()
                .and_then(|()| Err(JsonRpcError::method_not_found("session/load"))),
        )
    }

    fn resume_session(
        &self,
        _peer: &ClientPeer,
        _params: SessionResumeParams,
    ) -> impl Future<Output = std::result::Result<SessionResumeResult, JsonRpcError>> + Send {
        std::future::ready(
            self.ensure_initialized()
                .and_then(|()| Err(JsonRpcError::method_not_found("session/resume"))),
        )
    }

    async fn prompt(
        &self,
        peer: &ClientPeer,
        params: PromptParams,
    ) -> std::result::Result<PromptResult, JsonRpcError> {
        self.ensure_initialized()?;
        let text = prompt_text(&params.prompt)?;
        let Some(entry) = self.get_session(&params.session_id) else {
            return Err(session_not_found(&params.session_id));
        };
        let guard = entry.cancellation.begin_prompt();
        let Ok(mut session) = entry.slot.1.recv().await else {
            return Err(session_not_found(&params.session_id));
        };

        let session_id = params.session_id.clone();
        let outcome = session
            .prompt_with_guard(&guard, &text, |update| {
                let notification = JsonRpcNotification::with_params(
                    "session/update",
                    SessionNotification::new(&session_id, update),
                );
                if let Err(error) = peer.enqueue_notification(notification) {
                    debug!("failed to queue session update: {error}");
                }
            })
            .await;

        let was_cancelled = guard.is_cancelled();
        drop(guard);
        let _ = entry.slot.0.try_send(session);

        let stop_reason = match outcome {
            Ok(()) if was_cancelled => StopReason::Cancelled,
            Ok(()) => StopReason::EndTurn,
            Err(error) => {
                debug!("Agent error: {error}");
                StopReason::Error
            }
        };

        Ok(PromptResult {
            stop_reason,
            meta: None,
        })
    }

    fn list_sessions(
        &self,
        _peer: &ClientPeer,
        params: SessionListParams,
    ) -> impl Future<Output = std::result::Result<SessionListResult, JsonRpcError>> + Send {
        if let Err(err) = self.ensure_initialized() {
            return std::future::ready(Err(err));
        }
        let sessions = {
            let guard = self.sessions.lock().expect("sessions mutex poisoned");
            guard
                .iter()
                .filter(|(_, entry)| {
                    params
                        .cwd
                        .as_ref()
                        .is_none_or(|filter_cwd| filter_cwd == &entry.cwd)
                })
                .map(|(id, entry)| SessionInfo {
                    session_id: id.clone(),
                    cwd: entry.cwd.clone(),
                    additional_directories: Vec::new(),
                    title: None,
                    updated_at: None,
                    meta: None,
                    extra: BTreeMap::new(),
                })
                .collect()
        };
        std::future::ready(Ok(SessionListResult {
            sessions,
            ..SessionListResult::default()
        }))
    }

    fn close_session(
        &self,
        _peer: &ClientPeer,
        params: SessionCloseParams,
    ) -> impl Future<Output = std::result::Result<SessionCloseResult, JsonRpcError>> + Send {
        if let Err(err) = self.ensure_initialized() {
            return std::future::ready(Err(err));
        }
        let result = if self.end_session(&params.session_id) {
            Ok(SessionCloseResult::default())
        } else {
            Err(session_not_found(&params.session_id))
        };
        std::future::ready(result)
    }

    fn delete_session(
        &self,
        _peer: &ClientPeer,
        params: SessionDeleteParams,
    ) -> impl Future<Output = std::result::Result<SessionDeleteResult, JsonRpcError>> + Send {
        if let Err(err) = self.ensure_initialized() {
            return std::future::ready(Err(err));
        }
        let result = if self.end_session(&params.session_id) {
            Ok(SessionDeleteResult::default())
        } else {
            Err(session_not_found(&params.session_id))
        };
        std::future::ready(result)
    }

    fn set_mode(
        &self,
        _peer: &ClientPeer,
        _params: SessionSetModeParams,
    ) -> impl Future<Output = std::result::Result<SessionSetModeResult, JsonRpcError>> + Send {
        std::future::ready(
            self.ensure_initialized()
                .and_then(|()| Err(JsonRpcError::method_not_found("session/set_mode"))),
        )
    }

    fn set_config_option(
        &self,
        _peer: &ClientPeer,
        _params: SessionSetConfigOptionParams,
    ) -> impl Future<Output = std::result::Result<SessionSetConfigOptionResult, JsonRpcError>> + Send
    {
        std::future::ready(
            self.ensure_initialized()
                .and_then(|()| Err(JsonRpcError::method_not_found("session/set_config_option"))),
        )
    }

    fn ext_request(
        &self,
        _peer: &ClientPeer,
        method: ExtMethod,
        _params: Option<Value>,
    ) -> impl Future<Output = std::result::Result<Value, JsonRpcError>> + Send {
        let init_check = self.ensure_initialized();
        async move {
            init_check?;
            Err(JsonRpcError::method_not_found(method.as_str()))
        }
    }

    fn cancel(
        &self,
        _peer: &ClientPeer,
        params: SessionCancelParams,
    ) -> impl Future<Output = ()> + Send {
        if self.initialized.load(Ordering::Acquire) {
            if let Some(entry) = self.get_session(&params.session_id) {
                entry.cancellation.cancel_active();
            } else {
                debug!("session/cancel for unknown session: {}", params.session_id);
            }
        }
        std::future::ready(())
    }
}

/// ACP server that exposes aither agents to code editors.
///
/// Built onto the generic duplex server layer ([`serve`](super::serve) and
/// [`AgentHandler`]); bridges that wrap external agent runtimes without
/// implementing [`LanguageModel`] should implement [`AgentHandler`] directly
/// instead.
///
/// # Example
///
/// ```ignore
/// use aither_acp::AcpServer;
///
/// let mut server = AcpServer::stdio("my-agent", "1.0.0", executor, |setup| async move {
///     let mut builder = Agent::builder(llm).instruction("You are helpful");
///     for connection in setup.mcp {
///         builder = builder.mcp(connection);
///     }
///     Ok(builder.build())
/// });
/// server.run().await?;
/// ```
pub struct AcpServer<T: BidirectionalTransport, LLM: LanguageModel + 'static> {
    transport: T,
    handler: Arc<AitherAgentHandler<LLM>>,
}

impl<T: BidirectionalTransport, LLM: LanguageModel + 'static> std::fmt::Debug
    for AcpServer<T, LLM>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let session_count = self
            .handler
            .sessions
            .lock()
            .map_or(0, |sessions| sessions.len());
        f.debug_struct("AcpServer")
            .field("info", &self.handler.info)
            .field("sessions", &session_count)
            .field(
                "initialized",
                &self.handler.initialized.load(Ordering::SeqCst),
            )
            .finish_non_exhaustive()
    }
}

impl<LLM: LanguageModel + 'static> AcpServer<StdioTransport, LLM> {
    /// Create an ACP server using stdio transport.
    ///
    /// This is the standard way to create an ACP server that communicates
    /// via stdin/stdout (e.g., when run by Zed or other editors).
    ///
    /// # Arguments
    ///
    /// * `name` - The agent name.
    /// * `version` - The agent version.
    /// * `executor` - Runs the connections to the MCP servers sessions are
    ///   opened with.
    /// * `make_agent` - Builds the agent backing each new session from its
    ///   [`SessionSetup`].
    pub fn stdio<F, Fut>(
        name: impl Into<String>,
        version: impl Into<String>,
        executor: impl Executor + 'static,
        make_agent: F,
    ) -> Self
    where
        F: Fn(SessionSetup) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Agent<LLM>>> + Send + 'static,
    {
        Self::new(StdioTransport::new(), name, version, executor, make_agent)
    }
}

impl<T: BidirectionalTransport, LLM: LanguageModel + 'static> AcpServer<T, LLM> {
    /// Create an ACP server over a custom transport.
    ///
    /// For most use cases, prefer [`AcpServer::stdio`]. Arguments are as for
    /// that constructor.
    pub fn new<F, Fut>(
        transport: T,
        name: impl Into<String>,
        version: impl Into<String>,
        executor: impl Executor + 'static,
        make_agent: F,
    ) -> Self
    where
        F: Fn(SessionSetup) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Agent<LLM>>> + Send + 'static,
    {
        let handler = AitherAgentHandler {
            info: Implementation {
                name: name.into(),
                title: None,
                version: version.into(),
            },
            sessions: Mutex::new(HashMap::new()),
            initialized: AtomicBool::new(false),
            make_agent: Mutex::new(Box::new(move |setup| Box::pin(make_agent(setup)))),
            executor: AnyExecutor::new(executor),
        };
        Self {
            transport,
            handler: Arc::new(handler),
        }
    }

    /// Run the server main loop.
    ///
    /// Processes incoming client requests and notifications until the
    /// connection closes.
    ///
    /// # Errors
    ///
    /// Returns [`AcpError::Transport`] if a fatal transport error occurs.
    pub async fn run(&mut self) -> Result<()> {
        debug!("ACP server starting: {}", self.handler.info.name);
        let (peer, outbound_rx) = ClientPeer::channel();
        drive_transport(&mut self.transport, outbound_rx, peer, self.handler.clone())
            .await
            .map_err(|error| match error {
                ServerError::Transport(message) => AcpError::Transport(message),
                ServerError::Closed => AcpError::ConnectionClosed,
                ServerError::JsonRpc(rpc) => AcpError::JsonRpc(rpc),
                ServerError::Protocol(proto) => AcpError::InvalidState(proto),
            })
    }

    /// Send a session update notification directly over the underlying
    /// transport.
    ///
    /// # Errors
    ///
    /// Returns [`AcpError::Transport`] if the notification cannot be written to
    /// the client.
    pub async fn send_update(&mut self, session_id: &str, update: SessionUpdate) -> Result<()> {
        let notif = JsonRpcNotification::with_params(
            "session/update",
            SessionNotification::new(session_id, update),
        );
        self.transport
            .notify(notif)
            .await
            .map_err(|error| AcpError::Transport(error.to_string()))
    }
}

#[derive(Template)]
#[template(path = "embedded_resource_text.txt", escape = "none")]
struct EmbeddedResourceTextTemplate<'a> {
    uri: &'a str,
    mime_type: Option<&'a str>,
    text: &'a str,
}

#[derive(Template)]
#[template(path = "embedded_resource_binary.txt", escape = "none")]
struct EmbeddedResourceBinaryTemplate<'a> {
    uri: &'a str,
    mime_type: Option<&'a str>,
    byte_len: usize,
}

#[derive(Template)]
#[template(path = "embedded_resource_empty.txt", escape = "none")]
struct EmbeddedResourceEmptyTemplate<'a> {
    uri: &'a str,
    mime_type: Option<&'a str>,
}

#[derive(Template)]
#[template(path = "resource_link.txt", escape = "none")]
struct ResourceLinkTemplate<'a> {
    uri: &'a str,
    name: &'a str,
    title: Option<&'a str>,
    description: Option<&'a str>,
    mime_type: Option<&'a str>,
    size: Option<i64>,
}

fn format_embedded_resource(
    resource: &EmbeddedResource,
) -> std::result::Result<String, JsonRpcError> {
    let uri = resource.uri.as_str();
    let mime_type = resource.mime_type.as_deref();
    if let Some(text) = resource.text.as_deref() {
        return Ok(EmbeddedResourceTextTemplate {
            uri,
            mime_type,
            text,
        }
        .render()
        .expect("embedded text resource template must render"));
    }
    if let Some(blob) = resource.blob.as_deref() {
        let bytes = BASE64.decode(blob).map_err(|err| {
            JsonRpcError::invalid_params(format!(
                "invalid base64 in embedded resource '{uri}': {err}"
            ))
        })?;
        return std::str::from_utf8(&bytes).map_or_else(
            |_| {
                Ok(EmbeddedResourceBinaryTemplate {
                    uri,
                    mime_type,
                    byte_len: bytes.len(),
                }
                .render()
                .expect("embedded binary resource template must render"))
            },
            |text| {
                Ok(EmbeddedResourceTextTemplate {
                    uri,
                    mime_type,
                    text,
                }
                .render()
                .expect("embedded text resource template must render"))
            },
        );
    }
    Ok(EmbeddedResourceEmptyTemplate { uri, mime_type }
        .render()
        .expect("embedded empty resource template must render"))
}

fn format_resource_link(link: &ResourceLink) -> String {
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

/// Renders a prompt's text, embedded resources, and resource links for the model.
fn prompt_text(prompt: &[ContentBlock]) -> std::result::Result<String, JsonRpcError> {
    let mut rendered = Vec::with_capacity(prompt.len());
    for block in prompt {
        match block {
            ContentBlock::Text(text) => rendered.push(text.text.clone()),
            ContentBlock::Resource(content) => {
                rendered.push(format_embedded_resource(&content.resource)?);
            }
            ContentBlock::ResourceLink(link) => rendered.push(format_resource_link(link)),
            ContentBlock::Image(_) | ContentBlock::Audio(_) => {}
        }
    }
    Ok(rendered.join("\n"))
}

/// JSON-RPC invalid-params error naming a missing session.
fn session_not_found(session_id: &str) -> JsonRpcError {
    JsonRpcError::invalid_params(format!("Session not found: {session_id}"))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Duration;

    use aither_core::llm::{Event, LLMRequest, model::Profile};
    use aither_mcp::protocol::{ErrorCode, JsonRpcMessage, JsonRpcRequest, RequestId};
    use aither_mcp::transport::DuplexTransport;
    use futures_lite::{Stream, StreamExt};
    use serde_json::json;

    use super::*;
    use crate::client::{AcpClient, ClientError, ClientHandler};
    use crate::protocol::{
        RequestPermissionOutcome, RequestPermissionParams, RequestPermissionResult,
        ResourceContent, TextContent,
    };

    #[derive(Debug)]
    struct ModelErr;

    impl std::fmt::Display for ModelErr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("model error")
        }
    }

    impl std::error::Error for ModelErr {}

    impl aither_core::llm::Classify for ModelErr {
        fn failure(&self) -> aither_core::llm::Failure {
            aither_core::llm::Failure::permanent()
        }
    }

    /// Emits a text chunk, notifies `started_tx`, and then parks forever unless
    /// `complete_fast` is set.
    #[derive(Clone)]
    struct BlockingAfterChunkModel {
        started_tx: async_channel::Sender<()>,
        complete_fast: Arc<AtomicBool>,
        recorded_prompts: Arc<Mutex<Vec<String>>>,
    }

    impl BlockingAfterChunkModel {
        fn new(started_tx: async_channel::Sender<()>) -> Self {
            Self {
                started_tx,
                complete_fast: Arc::new(AtomicBool::new(false)),
                recorded_prompts: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl LanguageModel for BlockingAfterChunkModel {
        type Error = ModelErr;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = std::result::Result<Event, ModelErr>> + Send {
            let joined = request
                .messages()
                .iter()
                .map(|m| m.content().to_string())
                .collect::<Vec<_>>()
                .join("\n---\n");
            self.recorded_prompts.lock().expect("recorded").push(joined);

            let started_tx = self.started_tx.clone();
            let fast = self.complete_fast.load(Ordering::Acquire);
            futures_lite::stream::once_future(async move {
                let _ = started_tx.send(()).await;
                Ok(Event::Text("partial".to_string()))
            })
            .chain(futures_lite::stream::poll_fn(move |_cx| {
                if fast {
                    std::task::Poll::Ready(None)
                } else {
                    std::task::Poll::Pending
                }
            }))
        }

        fn profile(&self) -> impl Future<Output = std::result::Result<Profile, ModelErr>> + Send {
            std::future::ready(Ok(Profile::new("blk", "test", "blk", "Blocking", 128_000)))
        }
    }

    #[derive(Default)]
    struct TestClient {
        updates: Mutex<Vec<SessionNotification>>,
    }

    impl ClientHandler for TestClient {
        fn session_update(
            &self,
            notification: SessionNotification,
        ) -> impl Future<Output = ()> + Send {
            self.updates.lock().expect("updates").push(notification);
            std::future::ready(())
        }

        fn request_permission(
            &self,
            _params: RequestPermissionParams,
        ) -> impl Future<Output = std::result::Result<RequestPermissionResult, JsonRpcError>> + Send
        {
            std::future::ready(Ok(RequestPermissionResult {
                outcome: RequestPermissionOutcome::Cancelled,
                meta: None,
            }))
        }
    }

    #[tokio::test]
    async fn acp_server_handles_session_lifecycle_and_mid_turn_cancel() {
        let (started_tx, started_rx) = async_channel::bounded(1);
        let (client_end, server_end) = DuplexTransport::pair();
        let mut server = AcpServer::new(
            server_end,
            "aither-test",
            "0.1.0",
            executor_core::tokio::TokioGlobal,
            move |_setup| {
                let started_tx = started_tx.clone();
                async move { Ok(Agent::new(BlockingAfterChunkModel::new(started_tx))) }
            },
        );
        let server_task = tokio::spawn(async move { server.run().await });

        let (client, client_driver) = AcpClient::connect(client_end, TestClient::default());
        let client_task = tokio::spawn(client_driver);

        let init = client.initialize().await.expect("initialize");
        assert_eq!(
            init.agent_info.as_ref().map(|i| i.name.as_str()),
            Some("aither-test")
        );

        let session = client
            .new_session(SessionNewParams::new("/tmp/acp-server-test"))
            .await
            .expect("session/new");

        let listed = client
            .list_sessions(SessionListParams::new())
            .await
            .expect("session/list");
        assert_eq!(listed.sessions.len(), 1);
        assert_eq!(listed.sessions[0].session_id, session.session_id);

        let prompt_fut = client
            .start_prompt(&PromptParams::new(
                &session.session_id,
                vec![ContentBlock::Text(TextContent {
                    text: "hello".to_string(),
                    annotations: None,
                    meta: None,
                })],
            ))
            .await
            .expect("start_prompt");
        let prompt_task = tokio::spawn(prompt_fut);

        tokio::time::timeout(Duration::from_secs(5), started_rx.recv())
            .await
            .expect("turn must start")
            .expect("started_rx");

        client
            .cancel(SessionCancelParams::new(&session.session_id))
            .await
            .expect("cancel");

        let result = tokio::time::timeout(Duration::from_secs(5), prompt_task)
            .await
            .expect("cancelled prompt must finish")
            .expect("prompt_task")
            .expect("prompt result");
        assert_eq!(result.stop_reason, StopReason::Cancelled);

        client
            .close_session(session.session_id.as_str())
            .await
            .expect("close_session");
        // Closing an already-closed session reports invalid_params.
        assert!(matches!(
            client.delete_session(session.session_id.as_str()).await,
            Err(ClientError::JsonRpc(_))
        ));

        client.close();
        client_task.await.expect("client_task");
        server_task.await.expect("server_task").expect("server.run");
    }

    #[tokio::test]
    async fn rejects_requests_before_initialize_and_duplicate_initialize() {
        let (started_tx, _started_rx) = async_channel::bounded(1);
        let (client_end, server_end) = DuplexTransport::pair();
        let mut server = AcpServer::new(
            server_end,
            "aither-test",
            "0.1.0",
            executor_core::tokio::TokioGlobal,
            move |_setup| {
                let started_tx = started_tx.clone();
                async move { Ok(Agent::new(BlockingAfterChunkModel::new(started_tx))) }
            },
        );
        let server_task = tokio::spawn(async move { server.run().await });

        let (client, client_driver) = AcpClient::connect(client_end, TestClient::default());
        let client_task = tokio::spawn(client_driver);

        // Requests before `initialize` fail with INVALID_REQUEST (-32600).
        let err = client
            .new_session(SessionNewParams::new("/tmp/acp-pre-init"))
            .await
            .expect_err("session/new before initialize must fail");
        assert!(matches!(
            err,
            ClientError::JsonRpc(ref rpc) if rpc.code == ErrorCode::INVALID_REQUEST
        ));

        let list_err = client
            .list_sessions(SessionListParams::new())
            .await
            .expect_err("session/list before initialize must fail");
        assert!(matches!(
            list_err,
            ClientError::JsonRpc(ref rpc) if rpc.code == ErrorCode::INVALID_REQUEST
        ));

        // First `initialize` succeeds; a second `initialize` is rejected.
        client.initialize().await.expect("first initialize");
        let dup_init = client
            .initialize()
            .await
            .expect_err("second initialize must fail");
        assert!(matches!(
            dup_init,
            ClientError::JsonRpc(ref rpc) if rpc.code == ErrorCode::INVALID_REQUEST
        ));

        client.close();
        client_task.await.expect("client_task");
        server_task.await.expect("server_task").expect("server.run");
    }

    #[tokio::test]
    async fn rejects_duplicate_in_flight_request_id_without_clobbering_original() {
        let (started_tx, started_rx) = async_channel::bounded(2);
        let (mut raw_client, server_end) = DuplexTransport::pair();
        let mut server = AcpServer::new(
            server_end,
            "aither-test",
            "0.1.0",
            executor_core::tokio::TokioGlobal,
            move |_setup| {
                let started_tx = started_tx.clone();
                async move { Ok(Agent::new(BlockingAfterChunkModel::new(started_tx))) }
            },
        );
        let server_task = tokio::spawn(async move { server.run().await });

        raw_client
            .send_request(JsonRpcRequest::with_params(
                RequestId::Number(1),
                "initialize",
                json!({"protocolVersion": PROTOCOL_VERSION}),
            ))
            .await
            .expect("send init");
        let Some(JsonRpcMessage::Response(init_resp)) = raw_client.recv().await.expect("recv init")
        else {
            panic!("expected initialize response");
        };
        assert!(init_resp.error.is_none());

        raw_client
            .send_request(JsonRpcRequest::with_params(
                RequestId::Number(2),
                "session/new",
                json!({"cwd": "/tmp/acp-dup-id"}),
            ))
            .await
            .expect("send session/new");
        let Some(JsonRpcMessage::Response(new_resp)) = raw_client.recv().await.expect("recv new")
        else {
            panic!("expected session/new response");
        };
        let session_id = new_resp.result.expect("result")["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_string();

        // Start a blocking prompt with id = 7.
        raw_client
            .send_request(JsonRpcRequest::with_params(
                RequestId::Number(7),
                "session/prompt",
                json!({
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "slow"}],
                }),
            ))
            .await
            .expect("send prompt");
        started_rx.recv().await.expect("prompt started");

        // Send a duplicate request carrying the same in-flight id = 7.
        raw_client
            .send_request(JsonRpcRequest::with_params(
                RequestId::Number(7),
                "session/list",
                json!({}),
            ))
            .await
            .expect("send duplicate id");

        let dup_resp = loop {
            match raw_client.recv().await.expect("recv") {
                Some(JsonRpcMessage::Response(resp)) => break resp,
                Some(JsonRpcMessage::Notification(_)) => {}
                other => panic!("unexpected message: {other:?}"),
            }
        };
        assert_eq!(dup_resp.id, RequestId::Number(7));
        assert_eq!(
            dup_resp.error.expect("duplicate must error").code,
            ErrorCode::INVALID_REQUEST
        );

        // Cancel the original prompt and verify its own response still arrives cleanly.
        aither_mcp::transport::Transport::notify(
            &mut raw_client,
            JsonRpcNotification::with_params("session/cancel", json!({"sessionId": session_id})),
        )
        .await
        .expect("notify cancel");

        let orig_resp = loop {
            match raw_client.recv().await.expect("recv") {
                Some(JsonRpcMessage::Response(resp)) => break resp,
                Some(JsonRpcMessage::Notification(_)) => {}
                other => panic!("unexpected message: {other:?}"),
            }
        };
        assert_eq!(orig_resp.id, RequestId::Number(7));
        assert_eq!(
            orig_resp.result.expect("orig result")["stopReason"],
            "cancelled"
        );

        aither_mcp::transport::Transport::close(&mut raw_client)
            .await
            .expect("close");
        server_task.await.expect("server_task").expect("server.run");
    }

    #[tokio::test]
    async fn reentrant_client_request_inside_session_update_does_not_deadlock() {
        struct ReentrantHandler {
            client: Arc<OnceLock<AcpClient<Self>>>,
            observed_sessions: Arc<Mutex<Vec<usize>>>,
        }

        impl ClientHandler for ReentrantHandler {
            async fn session_update(&self, _notification: SessionNotification) {
                let client = self.client.get().expect("client set before prompt").clone();
                let listed = client
                    .list_sessions(SessionListParams::new())
                    .await
                    .expect("reentrant list_sessions inside session_update must succeed");
                self.observed_sessions
                    .lock()
                    .expect("observed")
                    .push(listed.sessions.len());
            }

            fn request_permission(
                &self,
                _params: RequestPermissionParams,
            ) -> impl Future<Output = std::result::Result<RequestPermissionResult, JsonRpcError>> + Send
            {
                std::future::ready(Ok(RequestPermissionResult {
                    outcome: RequestPermissionOutcome::Cancelled,
                    meta: None,
                }))
            }
        }

        let (started_tx, _started_rx) = async_channel::bounded(4);
        let model = BlockingAfterChunkModel::new(started_tx);
        model.complete_fast.store(true, Ordering::Release);

        let (client_end, server_end) = DuplexTransport::pair();
        let mut server = AcpServer::new(
            server_end,
            "aither-test",
            "0.1.0",
            executor_core::tokio::TokioGlobal,
            move |_setup| {
                let model = model.clone();
                async move { Ok(Agent::new(model)) }
            },
        );
        let server_task = tokio::spawn(async move { server.run().await });

        let shared_client = Arc::new(OnceLock::new());
        let observed = Arc::new(Mutex::new(Vec::new()));
        let (client, client_driver) = AcpClient::connect(
            client_end,
            ReentrantHandler {
                client: Arc::clone(&shared_client),
                observed_sessions: Arc::clone(&observed),
            },
        );
        let _ = shared_client.set(client.clone());
        let client_task = tokio::spawn(client_driver);

        client.initialize().await.expect("initialize");
        let session = client
            .new_session(SessionNewParams::new("/tmp/acp-reentrant"))
            .await
            .expect("session/new");

        let prompt_res = tokio::time::timeout(
            Duration::from_secs(5),
            client.prompt(PromptParams::new(
                &session.session_id,
                vec![ContentBlock::Text(TextContent {
                    text: "stream and re-enter".to_string(),
                    annotations: None,
                    meta: None,
                })],
            )),
        )
        .await
        .expect("reentrant session_update must not deadlock")
        .expect("prompt");
        assert_eq!(prompt_res.stop_reason, StopReason::EndTurn);
        assert_eq!(*observed.lock().expect("observed"), vec![1]);

        client.close();
        client_task.await.expect("client_task");
        server_task.await.expect("server_task").expect("server.run");
    }

    #[tokio::test]
    async fn immediate_wire_cancel_and_embedded_resources_work_end_to_end() {
        let (started_tx, _started_rx) = async_channel::bounded(8);
        let model = BlockingAfterChunkModel::new(started_tx);
        let fast_flag = Arc::clone(&model.complete_fast);
        let recorded = Arc::clone(&model.recorded_prompts);

        let (client_end, server_end) = DuplexTransport::pair();
        let mut server = AcpServer::new(
            server_end,
            "aither-test",
            "0.1.0",
            executor_core::tokio::TokioGlobal,
            move |_setup| {
                let model = model.clone();
                async move { Ok(Agent::new(model)) }
            },
        );
        let server_task = tokio::spawn(async move { server.run().await });

        let (client, client_driver) = AcpClient::connect(client_end, TestClient::default());
        let client_task = tokio::spawn(client_driver);

        client.initialize().await.expect("initialize");
        let session = client
            .new_session(SessionNewParams::new("/tmp/acp-immediate-cancel"))
            .await
            .expect("session/new");

        // Fire `start_prompt` and `cancel` back-to-back with zero sleep.
        let prompt_fut = client
            .start_prompt(&PromptParams::new(
                &session.session_id,
                vec![ContentBlock::Text(TextContent {
                    text: "cancel-me".to_string(),
                    annotations: None,
                    meta: None,
                })],
            ))
            .await
            .expect("start_prompt");
        client
            .cancel(SessionCancelParams::new(&session.session_id))
            .await
            .expect("immediate cancel");

        let cancelled = tokio::time::timeout(Duration::from_secs(5), prompt_fut)
            .await
            .expect("immediately cancelled prompt must not hang")
            .expect("prompt");
        assert_eq!(cancelled.stop_reason, StopReason::Cancelled);

        // Send a late cancel while idle — must NOT poison the next turn!
        client
            .cancel(SessionCancelParams::new(&session.session_id))
            .await
            .expect("idle cancel");

        fast_flag.store(true, Ordering::Release);
        let completed = client
            .prompt(PromptParams::new(
                &session.session_id,
                sample_context_blocks(),
            ))
            .await
            .expect("second prompt");
        assert_eq!(completed.stop_reason, StopReason::EndTurn);

        let captured = recorded.lock().expect("recorded").last().cloned().unwrap();
        assert!(captured.contains("Inspect context"));
        assert!(captured.contains("file:///workspace/notes.txt"));
        assert!(captured.contains("你好 / こんにちは / 안녕하세요"));
        assert!(captured.contains("Embedded binary resource file:///workspace/icon.bin"));
        assert!(captured.contains("(4 bytes)"));
        assert!(
            captured.contains("Resource link Specification (spec.md): file:///workspace/spec.md")
        );
        assert!(captured.contains("[4096 bytes]"));
        assert!(captured.contains("Protocol overview"));

        client.close();
        client_task.await.expect("client_task");
        server_task.await.expect("server_task").expect("server.run");
    }

    fn sample_context_blocks() -> Vec<ContentBlock> {
        let binary_blob = BASE64.encode([0xff_u8, 0x00, 0xfe, 0x80]);
        let mut link = ResourceLink::new("file:///workspace/spec.md", "spec.md");
        link.title = Some("Specification".to_string());
        link.mime_type = Some("text/markdown".to_string());
        link.size = Some(4096);
        link.description = Some("Protocol overview".to_string());

        vec![
            ContentBlock::Text(TextContent {
                text: "Inspect context".to_string(),
                annotations: None,
                meta: None,
            }),
            ContentBlock::Resource(ResourceContent {
                resource: EmbeddedResource {
                    uri: "file:///workspace/notes.txt".to_string(),
                    mime_type: Some("text/plain".to_string()),
                    text: Some("你好 / こんにちは / 안녕하세요".to_string()),
                    blob: None,
                    meta: None,
                },
                annotations: None,
                meta: None,
            }),
            ContentBlock::Resource(ResourceContent {
                resource: EmbeddedResource {
                    uri: "file:///workspace/icon.bin".to_string(),
                    mime_type: Some("application/octet-stream".to_string()),
                    text: None,
                    blob: Some(binary_blob),
                    meta: None,
                },
                annotations: None,
                meta: None,
            }),
            ContentBlock::ResourceLink(link),
        ]
    }
}
