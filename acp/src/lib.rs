//! ACP (Agent Client Protocol) client and server for aither.
//!
//! One ergonomic, runtime-agnostic API drives both ACP v1 and v2:
//! [`domain`] types are canonical v2 shapes, and the [`wire`] codec
//! layer degrades each connection to v1 when the peer negotiates it.
//!
//! - **Serve an agent**: implement [`SessionHost`] and [`Session`], then
//!   [`Server::serve`] over a [`Transport`] (or [`Server::serve_stdio`]
//!   with the `stdio` feature).
//! - **Drive an agent**: [`Client::builder`] registers agent→client
//!   handlers (permissions, elicitation, extension methods), then
//!   [`ClientBuilder::connect`]s over a transport, `spawn`s a child
//!   process (`process` feature), or [`ClientBuilder::in_process`] /
//!   [`ClientBuilder::in_process_server`] attaches an in-process
//!   [`SessionHost`] or configured [`Server`] over the zero-serde direct
//!   lane.
//! - [`state`]: the canonical [`SessionReducer`] folding
//!   `session/update`s into a [`SessionState`] snapshot plus
//!   [`SessionEvent`]s.
//!
//! # Serving an agent over stdio
//!
//! ```rust,no_run
//! use aither_acp::{
//!     Implementation, PendingPrompt, PromptError, Server, Session, SessionContext, SessionHost,
//!     SessionSetupParams, StopReason, server::ServerError,
//! };
//!
//! struct Echo;
//! impl Session for Echo {
//!     async fn prompt(
//!         &self,
//!         pending: PendingPrompt,
//!         _cx: &SessionContext,
//!     ) -> Result<StopReason, PromptError> {
//!         let turn = pending
//!             .accept()
//!             .await
//!             .map_err(|e| PromptError::from(e.into_inner()))?;
//!         turn.message().text("echo").await.ok();
//!         Ok(StopReason::EndTurn)
//!     }
//! }
//!
//! struct EchoHost;
//! impl SessionHost for EchoHost {
//!     type Session = Echo;
//!     async fn new_session(
//!         &self,
//!         _params: SessionSetupParams,
//!         _cx: &SessionContext,
//!     ) -> Result<Echo, ServerError> {
//!         Ok(Echo)
//!     }
//! }
//!
//! # async fn run() {
//! let server = Server::new(EchoHost, Implementation::new("echo-agent", "0.1.0"));
//! let (_client_end, agent_end) = aither_acp::rpc::Memory::pair();
//! let (peer, serve) = server.serve(agent_end); // or serve_stdio() with `stdio`
//! // Drive `serve` on an executor: it dispatches requests and runs turns.
//! # let _ = (peer, serve);
//! # }
//! ```
//!
//! # Driving an agent
//!
//! [`ClientBuilder::spawn`] (`process` feature) launches an agent child,
//! [`ClientBuilder::connect`] attaches any [`Transport`], and
//! [`ClientBuilder::in_process`] / [`ClientBuilder::in_process_server`]
//! attach a [`SessionHost`] or configured [`Server`] with zero
//! serialization:
//!
//! ```rust,no_run
//! use aither_acp::domain::PermissionOptionId;
//! use aither_acp::rpc;
//! use aither_acp::{
//!     Client, Implementation, PermissionOutcome, PermissionRequest, domain::SelectedOutcome,
//! };
//!
//! # async fn run() {
//! let (client_end, _agent_end) = rpc::Memory::pair();
//! let (client, conn) = Client::builder(Implementation::new("my-editor", "1.0.0"))
//!     .permissions(|_req: PermissionRequest| async move {
//!         PermissionOutcome::Selected(SelectedOutcome {
//!             option_id: PermissionOptionId::from("allow"),
//!             meta: None,
//!         })
//!     })
//!     .connect(client_end)
//!     .await
//!     .expect("connect");
//! tokio::spawn(conn); // drive the connection
//!
//! let session = client.new_session("/tmp").open().await.expect("session/new");
//! let outcome = session.prompt_run("hello").await.expect("prompt");
//! # let _ = outcome;
//! # }
//! ```

pub mod client;
pub mod domain;
pub mod ext;
pub mod rpc;
pub mod server;
pub mod state;
pub mod vendor;
pub mod wire;

#[cfg(feature = "agent")]
pub mod aither;
#[cfg(feature = "mcp")]
pub mod mcp;

// --- the ergonomic flat surface ---------------------------------------
//
// The modules stay public for advanced use; the names below are the
// everyday API: client, server, domain primitives, errors, transport.

pub use client::{
    Client, ClientBuilder, ClientError, ClientSession, IntoPrompt, PromptTurn, TurnOutcome,
};
pub use domain::ids::ConfigId;
pub use domain::{
    AgentCapabilities, AuthMethod, AvailableCommand, ClientCapabilities, ConfigOption, ConfigValue,
    ContentBlock, ExtIdent, ExtMethod, ForegroundState, Implementation, McpServer, MessageId,
    MetaPatch, Open, Patch, PermissionOutcome, PermissionRequest, PlanId, ResumeParams, SessionId,
    SessionInfo, SessionPage, SessionSetupParams, SessionUpdate, SetupResult, StopReason,
    TerminalId, ToolCallId,
};
#[cfg(feature = "process")]
pub use rpc::ChildProcess;
#[cfg(feature = "stdio")]
pub use rpc::Stdio;
pub use rpc::{Lines, Memory, PeerError, RpcError, Transport};
pub use server::{
    PendingPrompt, PromptError, Server, ServerBuilder, ServerError, Session, SessionContext,
    SessionHost, ToolCallBuilder, ToolCallHandle, Turn,
};
pub use state::{SessionEvent, SessionReducer, SessionState};

pub use server::ProtocolVersion;
pub use server::{AuthConfig, AuthLoginRequest, AuthRequest};

#[cfg(feature = "agent")]
pub use aither::{AcpChildAgent, AitherHost, SessionSetup};
#[cfg(feature = "mcp")]
pub use mcp::connect_all;
