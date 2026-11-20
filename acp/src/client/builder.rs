//! [`ClientBuilder`]: handler registration and connection entry points.

use std::collections::BTreeMap;
use std::fmt;
#[cfg(feature = "process")]
use std::path::Path;

use serde_json::json;

use crate::domain::{ClientCapabilities, ElicitCaps, Implementation, InitResult, SessionCaps};
use crate::rpc::{self, Transport};

use super::connection::{AnyAgentLane, Connection, SharedVersion, WireHooks, WireLane};
use super::error::ClientError;
use super::handle::{Client, SessionRegistry};
use super::handler::{
    AnyAgentHooks, AnyElicitationHandler, AnyExtRequestHandler, AnyFsHandler,
    AnyNotificationHandler, AnyPermissionHandler, AnyV1Terminal, AssembledHooks,
    ElicitationHandler, ExtRequestHandler, FsHandler, NotificationHandler, PermissionHandler,
    V1Terminal,
};

/// Builds a [`Client`]: register agent→client handlers, then pick a
/// transport.
pub struct ClientBuilder {
    info: Implementation,
    permission: Option<AnyPermissionHandler>,
    elicitation: Option<AnyElicitationHandler>,
    fs: Option<AnyFsHandler>,
    terminal: Option<AnyV1Terminal>,
    ext: Option<AnyExtRequestHandler>,
    notification: Option<AnyNotificationHandler>,
    auth_terminal: bool,
}

impl fmt::Debug for ClientBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientBuilder")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl ClientBuilder {
    pub(crate) const fn new(info: Implementation) -> Self {
        Self {
            info,
            permission: None,
            elicitation: None,
            fs: None,
            terminal: None,
            ext: None,
            notification: None,
            auth_terminal: false,
        }
    }

    /// Handle `session/request_permission`.
    #[must_use]
    pub fn permissions(mut self, handler: impl PermissionHandler) -> Self {
        self.permission = Some(AnyPermissionHandler::new(handler));
        self
    }

    /// Handle `elicitation/create` (+ `elicitation/complete`).
    /// Advertises elicitation capability in `initialize`.
    #[must_use]
    pub fn elicitation(mut self, handler: impl ElicitationHandler) -> Self {
        self.elicitation = Some(AnyElicitationHandler::new(handler));
        self
    }

    /// Handle the v1 `fs/*` lane. Advertised only in the v1-shaped
    /// capability payload (recovered from `capabilities.extra` on v1
    /// handshakes; never sent as a v2 field).
    #[must_use]
    pub fn v1_fs(mut self, handler: impl FsHandler) -> Self {
        self.fs = Some(AnyFsHandler::new(handler));
        self
    }

    /// Handle the v1 `terminal/*` lane. This does **not** advertise
    /// terminal *authentication* — use [`auth_terminal`](Self::auth_terminal)
    /// for that.
    #[must_use]
    pub fn v1_terminal(mut self, handler: impl V1Terminal) -> Self {
        self.terminal = Some(AnyV1Terminal::new(handler));
        self
    }

    /// Advertise terminal-authentication support
    /// (`clientCapabilities.auth.terminal`). Opt-in, independent of
    /// [`v1_terminal`](Self::v1_terminal).
    #[must_use]
    pub const fn auth_terminal(mut self) -> Self {
        self.auth_terminal = true;
        self
    }

    /// Handle `_`-prefixed extension requests.
    #[must_use]
    pub fn ext_request(mut self, handler: impl ExtRequestHandler) -> Self {
        self.ext = Some(AnyExtRequestHandler::new(handler));
        self
    }

    /// Catch-all handler for unmodelled agent notifications.
    #[must_use]
    pub fn notification(mut self, handler: impl NotificationHandler) -> Self {
        self.notification = Some(AnyNotificationHandler::new(handler));
        self
    }

    /// The capabilities this client will advertise.
    ///
    /// v1-only lanes (`fs`, `terminal`) are encoded into `capabilities`
    /// `extra` — the codec reads them back out on v1 negotiation, so
    /// they never appear as dedicated fields in the v2 handshake.
    fn client_capabilities(&self) -> ClientCapabilities {
        let mut caps = ClientCapabilities::default();
        if self.elicitation.is_some() {
            caps.elicitation = Some(ElicitCaps {
                form: Some(crate::domain::EmptyObject),
                url: Some(crate::domain::EmptyObject),
                meta: None,
                extra: BTreeMap::new(),
            });
        }
        if self.auth_terminal {
            caps.auth = Some(crate::domain::ClientAuthCaps {
                terminal: Some(crate::domain::EmptyObject),
                meta: None,
                extra: BTreeMap::new(),
            });
        }
        if let Some(fs) = &self.fs {
            let _ = fs; // presence is what matters
            caps.extra.insert(
                "fs".to_string(),
                json!({"readTextFile": true, "writeTextFile": true}),
            );
        }
        if self.terminal.is_some() {
            caps.extra.insert("terminal".to_string(), json!(true));
        }
        caps
    }

    fn hooks(&self, sessions: SessionRegistry) -> AnyAgentHooks {
        AnyAgentHooks::new(AssembledHooks {
            sessions,
            permission: self.permission.clone(),
            elicitation: self.elicitation.clone(),
            fs: self.fs.clone(),
            terminal: self.terminal.clone(),
            ext: self.ext.clone(),
            notification: self.notification.clone(),
        })
    }

    /// Connect over `transport`, run the `initialize` handshake, and
    /// return the client plus the [`Connection`] the caller must drive.
    ///
    /// # Errors
    ///
    /// Handshake failures: transport errors, rejection (including
    /// version negotiation), and malformed init results.
    pub async fn connect(
        self,
        transport: impl Transport + 'static,
    ) -> Result<(Client, Connection), ClientError> {
        let sessions = SessionRegistry::new();
        let version = std::sync::Arc::new(SharedVersion::default());
        let hooks = self.hooks(sessions.clone());
        let wire = WireHooks::new(version.clone(), sessions.clone(), hooks);
        let (peer, mut engine) = rpc::connection(transport, wire);
        let lane = AnyAgentLane::new(WireLane::new(peer, version, sessions.clone()));
        let caps = self.client_capabilities();
        // The handshake only resolves once the engine starts pumping, so
        // drive it alongside `initialize`.
        let init = {
            let handshake = lane.initialize(caps.clone(), self.info.clone());
            futures_lite::pin!(handshake);
            std::future::poll_fn(|cx| {
                if let std::task::Poll::Ready(result) = handshake.as_mut().poll(cx) {
                    return std::task::Poll::Ready(result);
                }
                match std::pin::Pin::new(&mut engine).poll(cx) {
                    std::task::Poll::Ready(()) => {
                        std::task::Poll::Ready(Err(ClientError::Transport(
                            rpc::RpcError::internal_error("connection closed during initialize"),
                        )))
                    }
                    std::task::Poll::Pending => std::task::Poll::Pending,
                }
            })
            .await?
        };
        let client = Client::new(lane, init, caps, self.info, sessions.clone());
        Ok((client, Connection::wire(engine, sessions)))
    }

    /// Spawn `bin` (a `process` feature transport) and connect to it.
    ///
    /// # Errors
    ///
    /// Spawn and handshake failures.
    #[cfg(feature = "process")]
    pub async fn spawn(
        self,
        bin: &str,
        args: &[String],
        env: &[(String, String)],
        cwd: &Path,
    ) -> Result<(Client, Connection), ClientError> {
        let transport =
            rpc::ChildProcess::spawn(bin, args, env, cwd).map_err(ClientError::Transport)?;
        self.connect(transport).await
    }

    /// Connect to an agent over this process's stdio (`stdio` feature).
    ///
    /// # Errors
    ///
    /// Handshake failures.
    #[cfg(feature = "stdio")]
    pub async fn connect_stdio(self) -> Result<(Client, Connection), ClientError> {
        self.connect(rpc::Stdio::new()).await
    }

    /// Attach to an in-process [`crate::server::SessionHost`] — the
    /// zero-serde direct lane (always v2 semantics): the client's
    /// [`AnyAgentLane`] calls [`crate::server::ServerCore`] domain
    /// methods directly and the server's `SessionContext` reaches the
    /// client's handler table with no serialization in between.
    ///
    /// The returned [`Connection`] drives the agent's prompt scheduler —
    /// spawn (or `or`) it for turns to progress. The agent's
    /// `Implementation` reported at `initialize` is the fixed
    /// in-process label; `SessionHost` carries no identity of its own.
    /// Closing the [`Connection`] — or dropping the last [`Client`]
    /// clone — ends the host.
    ///
    /// # Errors
    ///
    /// The `initialize` capability exchange failed.
    pub async fn in_process<H: crate::server::SessionHost>(
        self,
        host: H,
    ) -> Result<(Client, Connection), ClientError> {
        let sessions = SessionRegistry::new();
        let hooks = self.hooks(sessions.clone());
        let caps = self.client_capabilities();
        // The client's handler table answers agent→client traffic.
        let server_hooks =
            crate::server::AnyAgentHooks::new(super::direct::DirectHooks::new(hooks, caps.clone()));
        let core = std::sync::Arc::new(crate::server::ServerCore::new(
            host,
            crate::domain::Implementation::new("in-process-agent", env!("CARGO_PKG_VERSION")),
        ));
        let (shutdown_tx, shutdown_rx) = async_channel::bounded::<()>(1);
        let lane = AnyAgentLane::new(super::direct::DirectLane::new(
            super::direct::DirectServer::new(core.clone(), server_hooks, shutdown_tx),
        ));
        // Initialize goes through the real capability exchange; the
        // prompt scheduler only needs the returned `Connection` polled.
        let init = lane.initialize(caps.clone(), self.info.clone()).await?;
        let drive = async move {
            core.host_start().await;
            futures_lite::future::or(core.drive(), async move {
                let _ = shutdown_rx.recv().await;
            })
            .await;
            core.on_disconnect().await;
        };
        let client = Client::new(lane, init, caps, self.info, sessions.clone());
        Ok((client, Connection::in_process(sessions, Box::pin(drive))))
    }
}

/// Read a v1 session-capability flag folded into
/// `capabilities.session.extra` by the codec (`"list"`, `"resume"`,
/// `"close"`, `"loadSession"`).
pub fn session_cap(init: &InitResult, key: &str) -> bool {
    init.capabilities
        .session
        .as_ref()
        .and_then(|s: &SessionCaps| s.extra.get(key))
        .is_some_and(|v| v.is_object() || v.as_bool().unwrap_or(false))
}
