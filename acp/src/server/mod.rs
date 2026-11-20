//! The agent-facing ACP server API.
//!
//! A [`Server`] runs one [`SessionHost`] implementation for both ACP v1
//! and ACP v2 clients. The transport-independent [`ServerCore`]
//! dispatching logic is fronted by [`WireAdapter`] on a JSON-RPC
//! transport ([`Server::serve`]) or called directly by
//! `client::in_process` (the direct lane).
//!
//! ```rust,no_run
//! use std::path::PathBuf;
//! use std::sync::Arc;
//!
//! use futures_lite::future;
//!
//! use aither_acp::domain::{Implementation, SessionSetupParams, StopReason};
//! use aither_acp::rpc;
//! use aither_acp::server::{
//!     PendingPrompt, PromptError, Server, ServerError, Session, SessionContext,
//!     SessionHost,
//! };
//!
//! # struct Demo;
//! # impl SessionHost for Demo {
//! #     type Session = MySession;
//! #     async fn new_session(
//! #         &self,
//! #         _params: SessionSetupParams,
//! #         _cx: &SessionContext,
//! #     ) -> Result<MySession, ServerError> {
//! #         Ok(MySession)
//! #     }
//! # }
//! # struct MySession;
//! # impl Session for MySession {
//! #     async fn prompt(
//! #         &self,
//! #         prompt: PendingPrompt,
//! #         _cx: &SessionContext,
//! #     ) -> Result<StopReason, PromptError> {
//! #         let turn = prompt.accept().await.map_err(|_| PromptError::Cancelled)?;
//! #         turn.message().text("hi").await.ok();
//! #         Ok(StopReason::EndTurn)
//! #     }
//! # }
//! # fn build() {
//! let server = Server::new(Demo, Implementation::new("demo", "0.1.0"));
//! let (client, agent) = rpc::Memory::pair();
//! let (peer, serve) = server.serve(agent);
//! // Drive `serve` on an executor alongside the client connection.
//! # }
//! ```

mod context;
mod dispatch;
mod error;
mod host;
mod prompt;
mod session;
mod table;
mod turn;

#[cfg(test)]
mod tests;

pub use context::{
    AgentHooks, AnyAgentHooks, ProtocolVersion, SessionContext, SubagentHandle, TerminalHandle,
    V1TerminalLane,
};
pub use dispatch::{ExtHandler, InitializeParams, ServerCore, UnknownHandler, WireAdapter};
pub use error::{PromptError, PromptRejected, ServerError};
pub use host::{
    AnySessionHost, AuthConfig, AuthLoginRequest, AuthRequest, ReplaySink, ResumeRequest,
    SessionHost,
};
pub use prompt::{PendingPrompt, PromptTicket};
pub use session::{AnySession, Session};
pub use turn::{MessageStream, ToolCallBuilder, ToolCallHandle, Turn};

use std::future::Future;
use std::sync::Arc;

use futures_lite::future;

use crate::domain::Implementation;
use crate::domain::open::ExtMethod;
use crate::rpc::{self, Peer, Transport};

/// A bound server: a [`ServerCore`] plus its host, ready to
/// [`serve`](Server::serve) a transport.
///
/// Create with [`Server::new`] or [`Server::builder`].
pub struct Server {
    core: Arc<ServerCore>,
}

impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server").finish_non_exhaustive()
    }
}

impl Server {
    /// A server for `host` under implementation `info` (name/version
    /// reported in the `initialize` response).
    pub fn new<H: SessionHost>(host: H, info: Implementation) -> Self {
        Self::builder(host, info).build()
    }

    /// Start configuring a server.
    pub fn builder<H: SessionHost>(host: H, info: Implementation) -> ServerBuilder {
        ServerBuilder {
            core: ServerCore::new(host, info),
        }
    }

    /// The core dispatcher — the object the in-process direct lane
    /// (`client::in_process`) calls into.
    #[must_use]
    pub const fn core(&self) -> &Arc<ServerCore> {
        &self.core
    }

    /// Serve `transport`.
    ///
    /// Returns the [`Peer`] the host's client-side requests ride on,
    /// and a future to poll on the caller's executor: it runs the
    /// transport connection *and* the prompt scheduler together, then
    /// shuts the host down at EOF (`host.shutdown()` fires exactly
    /// once, after every session has been closed).
    pub fn serve<T: Transport + 'static>(
        self,
        transport: T,
    ) -> (Peer, impl Future<Output = ()> + Send) {
        let adapter = WireAdapter::new(self.core.clone());
        let (peer, conn) = rpc::connection(transport, adapter.clone());
        adapter.attach(peer.clone());
        let core = self.core;
        let done = async move {
            core.host_start().await;
            // The engine ends when the transport closes; `or` then drops
            // the scheduler and we close out.
            future::or(conn, core.drive()).await;
            core.on_disconnect().await;
        };
        (peer, done)
    }

    /// Serve over stdin/stdout lines (the classic stdio transport).
    ///
    /// Available with the `stdio` feature.
    #[cfg(feature = "stdio")]
    pub fn serve_stdio(self) -> (Peer, impl Future<Output = ()> + Send) {
        self.serve(rpc::Stdio::new())
    }
}

/// Builder for [`Server`].
pub struct ServerBuilder {
    core: ServerCore,
}

impl std::fmt::Debug for ServerBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerBuilder").finish_non_exhaustive()
    }
}

impl ServerBuilder {
    /// Install auth configuration (`auth/login`/`auth/logout` —
    /// `authenticate`/`logout` on v1).
    #[must_use]
    pub fn auth(mut self, auth: AuthConfig) -> Self {
        self.core = self.core.with_auth(auth);
        self
    }

    /// Install the `_`-prefixed extension-method handler.
    #[must_use]
    pub fn ext_handler(mut self, handler: ExtHandler) -> Self {
        self.core = self.core.with_ext_handler(handler);
        self
    }

    /// Convenience: install an extension handler from an async closure.
    ///
    /// The closure is invoked as
    /// `handler(method, params) -> Result<Value, RpcError>`.
    #[must_use]
    pub fn ext<F, Fut>(self, f: F) -> Self
    where
        F: Fn(ExtMethod, Option<serde_json::Value>) -> Fut + Send + Sync + 'static,
        Fut:
            std::future::Future<Output = Result<serde_json::Value, rpc::RpcError>> + Send + 'static,
    {
        self.ext_handler(std::sync::Arc::new(move |method, params| {
            Box::pin(f(method, params))
        }))
    }

    /// Install the unknown non-extension method fallback.
    #[must_use]
    pub fn unknown_handler(mut self, handler: UnknownHandler) -> Self {
        self.core = self.core.with_unknown_handler(handler);
        self
    }

    /// Finish building the [`Server`].
    pub fn build(self) -> Server {
        Server {
            core: Arc::new(self.core),
        }
    }
}
