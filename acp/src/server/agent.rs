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
use executor_core::{AnyExecutor, Executor};
use tracing::debug;

use super::{AgentHandler, ClientPeer, ServerError, drive_transport};
use crate::mcp::{ConnectError, connect_all};
use crate::protocol::{
    AcpError, AgentCapabilities, ContentBlock, Implementation, InitializeParams, InitializeResult,
    JsonRpcError, JsonRpcNotification, McpCapabilities, PROTOCOL_VERSION, PromptCapabilities,
    PromptParams, PromptResult, Result, SessionCancelParams, SessionCapabilities,
    SessionCloseCapabilities, SessionCloseParams, SessionCloseResult, SessionDeleteCapabilities,
    SessionDeleteParams, SessionDeleteResult, SessionInfo, SessionListCapabilities,
    SessionListParams, SessionListResult, SessionNewParams, SessionNewResult, SessionNotification,
    SessionUpdate, StopReason,
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
        self.initialized.store(true, Ordering::SeqCst);
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

    async fn new_session(
        &self,
        _peer: &ClientPeer,
        params: SessionNewParams,
    ) -> std::result::Result<SessionNewResult, JsonRpcError> {
        let mcp = match connect_all(&self.executor, &params.mcp_servers).await {
            Ok(mcp) => mcp,
            Err(err @ ConnectError::Unsupported(_)) => {
                return Err(JsonRpcError::invalid_params(err.to_string()));
            }
            Err(err @ ConnectError::Failed { .. }) => {
                return Err(JsonRpcError::internal_error(err.to_string()));
            }
        };
        let setup = SessionSetup {
            cwd: params.cwd.clone(),
            mcp,
        };
        let agent = self.create_agent(setup).await.map_err(|err| {
            JsonRpcError::internal_error(format!("could not start a session: {err}"))
        })?;

        let session = AcpSession::new(params.cwd, params.mcp_servers, agent);
        let session_id = self.insert_session(session);

        Ok(SessionNewResult {
            session_id,
            ..SessionNewResult::default()
        })
    }

    async fn prompt(
        &self,
        peer: &ClientPeer,
        params: PromptParams,
    ) -> std::result::Result<PromptResult, JsonRpcError> {
        let Some(entry) = self.get_session(&params.session_id) else {
            return Err(session_not_found(&params.session_id));
        };
        let Ok(mut session) = entry.slot.1.recv().await else {
            return Err(session_not_found(&params.session_id));
        };

        let text = prompt_text(&params.prompt);
        let session_id = params.session_id.clone();
        let outcome = session
            .prompt(&text, |update| {
                let notification = JsonRpcNotification::with_params(
                    "session/update",
                    SessionNotification::new(&session_id, update),
                );
                if let Err(error) = peer.enqueue_notification(notification) {
                    debug!("failed to queue session update: {error}");
                }
            })
            .await;

        let was_cancelled = session.is_cancelled();
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
        let result = if self.end_session(&params.session_id) {
            Ok(SessionDeleteResult::default())
        } else {
            Err(session_not_found(&params.session_id))
        };
        std::future::ready(result)
    }

    fn cancel(
        &self,
        _peer: &ClientPeer,
        params: SessionCancelParams,
    ) -> impl Future<Output = ()> + Send {
        if let Some(entry) = self.get_session(&params.session_id) {
            entry.cancellation.cancel();
        } else {
            debug!("session/cancel for unknown session: {}", params.session_id);
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

/// The concatenated text of a prompt's text blocks.
fn prompt_text(prompt: &[ContentBlock]) -> String {
    prompt
        .iter()
        .filter_map(|block| {
            if let ContentBlock::Text(text) = block {
                Some(text.text.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// JSON-RPC invalid-params error naming a missing session.
fn session_not_found(session_id: &str) -> JsonRpcError {
    JsonRpcError::invalid_params(format!("Session not found: {session_id}"))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use aither_core::llm::{Event, LLMRequest, model::Profile};
    use aither_mcp::transport::DuplexTransport;
    use futures_lite::{Stream, StreamExt};

    use super::*;
    use crate::client::{AcpClient, ClientError, ClientHandler};
    use crate::protocol::{
        RequestPermissionOutcome, RequestPermissionParams, RequestPermissionResult, TextContent,
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

    /// Emits a text chunk, notifies `started_tx`, and then parks forever so
    /// mid-turn `session/cancel` can be tested deterministically.
    #[derive(Clone)]
    struct BlockingAfterChunkModel {
        started_tx: async_channel::Sender<()>,
    }

    impl LanguageModel for BlockingAfterChunkModel {
        type Error = ModelErr;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = std::result::Result<Event, ModelErr>> + Send {
            let started_tx = self.started_tx.clone();
            futures_lite::stream::once_future(async move {
                let _ = started_tx.send(()).await;
                Ok(Event::Text("partial".to_string()))
            })
            .chain(futures_lite::stream::pending())
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
                async move { Ok(Agent::new(BlockingAfterChunkModel { started_tx })) }
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
}
