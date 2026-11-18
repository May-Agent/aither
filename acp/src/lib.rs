//! # aither-acp
//!
//! `Agent Client Protocol` (ACP) client and server implementation.
//!
//! This crate implements both directions of ACP v1:
//!
//! - [`serve`] and [`serve_stdio`] drive any [`AgentHandler`] over a
//!   [`BidirectionalTransport`](aither_mcp::transport::BidirectionalTransport),
//!   handing each call a [`ClientPeer`] that streams [`SessionNotification`]
//!   updates and initiates correlated agent-to-client requests
//!   (`session/request_permission`, `fs/*`, `terminal/*`,
//!   `elicitation/create`, `_` extensions) while concurrent inbound
//!   notifications such as `session/cancel` continue to flow.
//! - With the default `agent` feature enabled, `AcpServer` runs an aither
//!   `LanguageModel` as an ACP agent over stdio or any bidirectional
//!   transport, connecting session MCP servers via [`mcp::connect_all`].
//! - [`AcpClient`] connects to a spawned or piped ACP agent (such as
//!   `devin acp`) over any
//!   [`BidirectionalTransport`](aither_mcp::transport::BidirectionalTransport),
//!   driving sessions, prompts, modes, config options, authentication, and
//!   session lifecycle while a [`ClientHandler`] receives streamed updates
//!   and answers permission, file-system, terminal, and elicitation requests
//!   from the agent.
//!
//! Disable default features (`default-features = false`) to depend on the
//! protocol types, [`ext`], [`vendor`], [`mcp`], [`AcpClient`], and the generic
//! [`serve`] / [`AgentHandler`] server without pulling `aither-agent`.
//!
//! Extensions follow ACP's `_`-prefixed method convention: provider-neutral
//! conventions live under [`ext`], vendor-specific surface under [`vendor`],
//! and [`AcpClient::ext_request`]/[`AcpClient::ext_notify`] and
//! [`ClientPeer::ext_request`]/[`ClientPeer::ext_notify`] carry any
//! [`ExtMethod`] a provider defines.
//!
//! # Protocol Overview
//!
//! ACP uses JSON-RPC 2.0 over newline-delimited stdio or pipes. ACP describes
//! the client direction (editor → agent): [`serve`] / [`AgentHandler`] play the
//! ACP **server** (agent) role and [`AcpClient`] plays the ACP **client**
//! (editor) role.
//!
//! # Client
//!
//! ```no_run
//! use aither_acp::{AcpClient, ContentBlock, PromptParams, SessionNewParams, TextContent};
//! # use aither_acp::{ClientCapabilities, ClientHandler, SessionNotification};
//! # use aither_acp::{RequestPermissionParams, RequestPermissionResult, RequestPermissionOutcome};
//! # use aither_mcp::protocol::JsonRpcError;
//! # use std::future::Future;
//!
//! # struct H;
//! # impl ClientHandler for H {
//! #     fn session_update(&self, _: SessionNotification) -> impl Future<Output = ()> + Send {
//! #         async {}
//! #     }
//! #     fn request_permission(&self, _: RequestPermissionParams)
//! #         -> impl Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send
//! #     {
//! #         async { Ok(RequestPermissionResult { outcome: RequestPermissionOutcome::Cancelled, meta: None }) }
//! #     }
//! # }
//! # async fn run() -> Result<(), aither_acp::ClientError> {
//! let (client, connection) = AcpClient::spawn(
//!     "devin",
//!     &["acp"],
//!     [],
//!     "/tmp",
//!     H,
//! )?;
//! tokio::spawn(connection);
//!
//! let init = client.initialize().await?;
//! let session = client
//!     .new_session(SessionNewParams::new("/tmp"))
//!     .await?;
//! let result = client
//!     .prompt(PromptParams::new(
//!         &session.session_id,
//!         vec![ContentBlock::Text(TextContent {
//!             text: "hi".to_string(),
//!             annotations: None,
//!             meta: None,
//!         })],
//!     ))
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Generic Agent Server
//!
//! ```no_run
//! use aither_acp::{
//!     AgentCapabilities, AgentHandler, ClientPeer, ContentBlock, ContentChunk, Implementation,
//!     InitializeParams, InitializeResult, JsonRpcError, PROTOCOL_VERSION, PromptParams,
//!     PromptResult, SessionNewParams, SessionNewResult, SessionNotification, SessionUpdate,
//!     StopReason, TextContent, serve_stdio,
//! };
//! use std::future::Future;
//!
//! struct EchoHandler;
//!
//! impl AgentHandler for EchoHandler {
//!     fn initialize(
//!         &self,
//!         _peer: &ClientPeer,
//!         _params: Option<InitializeParams>,
//!     ) -> impl Future<Output = Result<InitializeResult, JsonRpcError>> + Send {
//!         async {
//!             Ok(InitializeResult {
//!                 protocol_version: PROTOCOL_VERSION,
//!                 agent_capabilities: AgentCapabilities::default(),
//!                 agent_info: Some(Implementation {
//!                     name: "echo-bridge".to_string(),
//!                     title: None,
//!                     version: "0.1.0".to_string(),
//!                 }),
//!                 auth_methods: vec![],
//!                 meta: None,
//!             })
//!         }
//!     }
//!
//!     fn new_session(
//!         &self,
//!         _peer: &ClientPeer,
//!         _params: SessionNewParams,
//!     ) -> impl Future<Output = Result<SessionNewResult, JsonRpcError>> + Send {
//!         async {
//!             Ok(SessionNewResult {
//!                 session_id: "sess-1".to_string(),
//!                 ..SessionNewResult::default()
//!             })
//!         }
//!     }
//!
//!     fn prompt(
//!         &self,
//!         peer: &ClientPeer,
//!         params: PromptParams,
//!     ) -> impl Future<Output = Result<PromptResult, JsonRpcError>> + Send {
//!         let peer = peer.clone();
//!         async move {
//!             let _ = peer
//!                 .session_update(SessionNotification::new(
//!                     &params.session_id,
//!                     SessionUpdate::AgentMessageChunk(ContentChunk {
//!                         content: ContentBlock::Text(TextContent {
//!                             text: "hello".to_string(),
//!                             annotations: None,
//!                             meta: None,
//!                         }),
//!                         message_id: None,
//!                         meta: None,
//!                     }),
//!                 ))
//!                 .await;
//!             Ok(PromptResult {
//!                 stop_reason: StopReason::EndTurn,
//!                 meta: None,
//!             })
//!         }
//!     }
//! }
//!
//! # async fn run() {
//! let (_peer, connection) = serve_stdio(EchoHandler);
//! connection.await;
//! # }
//! ```
//!
//! # Aither Agent Server (`agent` feature)
//!
//! ```no_run
//! # #[cfg(feature = "agent")]
//! # mod agent_example {
//! use aither_acp::{AcpError, AcpServer};
//! use aither_agent::Agent;
//! use aither_core::LanguageModel;
//! use aither_core::llm::{Event, LLMRequest, model::Profile};
//! use futures_lite::{Stream, stream};
//! use std::future::Future;
//!
//! #[derive(Clone)]
//! struct EchoModel;
//!
//! #[derive(Debug)]
//! struct EchoError;
//! impl std::fmt::Display for EchoError {
//!     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//!         f.write_str("echo model error")
//!     }
//! }
//! impl std::error::Error for EchoError {}
//! impl aither_core::llm::Classify for EchoError {
//!     fn failure(&self) -> aither_core::llm::Failure {
//!         aither_core::llm::Failure::permanent()
//!     }
//! }
//!
//! impl LanguageModel for EchoModel {
//!     type Error = EchoError;
//!
//!     fn respond(&self, _request: LLMRequest) -> impl Stream<Item = Result<Event, EchoError>> + Send {
//!         stream::once(Ok(Event::Text("Hello from ACP".to_string())))
//!     }
//!
//!     fn profile(&self) -> impl Future<Output = Result<Profile, EchoError>> + Send {
//!         std::future::ready(Ok(Profile::new("echo", "test", "echo", "Echo model", 128_000)))
//!     }
//! }
//!
//! pub async fn run() -> Result<(), AcpError> {
//!     let mut server = AcpServer::stdio(
//!         "echo-agent",
//!         "0.1.0",
//!         executor_core::tokio::TokioGlobal,
//!         |_setup| async { Ok::<_, AcpError>(Agent::new(EchoModel)) },
//!     );
//!     server.run().await
//! }
//! # }
//! # fn main() {}
//! ```
//!
//! # Error Handling
//!
//! [`ClientError`] and [`ServerError`] distinguish a closed transport from a
//! JSON-RPC error returned by the peer and from a protocol violation. When the
//! connection closes, every pending request fails rather than hanging.

#[cfg(feature = "agent")]
mod adapter;
#[cfg(feature = "agent")]
mod child;
mod client;
pub mod ext;
mod macros;
pub mod mcp;
pub mod protocol;
mod server;
#[cfg(feature = "agent")]
mod session;
pub mod vendor;

#[cfg(feature = "agent")]
pub use adapter::SessionUpdateMapper;
#[cfg(feature = "agent")]
pub use child::{AcpChildAgent, UpdateSink, UpdateSource};
pub use client::{AcpClient, ClientError, ClientHandler, ResponseFuture};
pub use protocol::*;
#[cfg(feature = "agent")]
pub use server::{AcpServer, SessionSetup};
pub use server::{AgentHandler, ClientPeer, ServerError, ServerResponseFuture, serve, serve_stdio};
#[cfg(feature = "agent")]
pub use session::AcpSession;
