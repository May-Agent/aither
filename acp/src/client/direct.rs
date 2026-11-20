//! The direct (in-process) lane — zero serde, always v2 semantics.
//!
//! [`DirectLane`] forwards the client's [`AgentLaneImpl`] calls to a
//! [`DirectCore`]: [`DirectServer`] adapts
//! [`crate::server::ServerCore`] (its per-call `hooks` parameter bound to
//! the client's handler table), while [`DirectHooks`] adapts that same
//! handler table to the server's [`crate::server::AgentHooks`] trait so
//! [`crate::server::SessionContext`] calls reach it directly. No message
//! is ever serialized and no turn end is fabricated — the agent's own
//! `idle` `state_update` ends the turn, exactly like on the wire.
//!
//! The `fs/*` and `terminal/*` client lanes are v1-only and never
//! reachable here: [`crate::server::SessionContext`] gates them on
//! `protocol.is_v1()` and the direct lane always reports
//! [`crate::server::ProtocolVersion::V2`].

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::domain::{
    AuthMethodId, ClientCapabilities, ElicitCreate, ElicitMode, ElicitResult, ExtMethod,
    Implementation, InitResult, MessageId, Open, PermissionOutcome, PermissionRequest,
    PromptAccepted, PromptParams, ResumeParams, SessionId, SessionListParams, SessionNotification,
    SessionPage, SessionSetupParams, SessionUpdate, SetConfigParams, SetupResult, TerminalId,
    ToolCallId, config::ConfigOption,
};
use crate::rpc::{PeerError, RpcError};
use crate::server::{AgentHooks as ServerAgentHooks, AnyAgentHooks as ServerHooks, ServerCore};

use super::connection::{AgentLaneImpl, PromptReply};
use super::error::ClientError;
use super::handler::AnyAgentHooks as ClientHooks;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The surface [`DirectLane`] needs from `crate::server::ServerCore`.
///
/// Object-safety is not required — the lane is generic. When
/// `ServerCore` exists, `impl DirectCore for ServerCore` (mechanical
/// forwarding) or adapt its methods here if its signatures differ.
pub trait DirectCore: Send + Sync + 'static {
    /// `initialize`.
    fn initialize(
        &self,
        capabilities: ClientCapabilities,
        info: Implementation,
    ) -> impl Future<Output = Result<InitResult, RpcError>> + Send;
    /// `session/new`.
    fn new_session(
        &self,
        params: SessionSetupParams,
    ) -> impl Future<Output = Result<SetupResult, RpcError>> + Send;
    /// `session/resume` (replay waits are the caller's concern).
    fn resume_session(
        &self,
        params: ResumeParams,
    ) -> impl Future<Output = Result<SetupResult, RpcError>> + Send;
    /// `session/fork`.
    fn fork_session(
        &self,
        params: ResumeParams,
    ) -> impl Future<Output = Result<SetupResult, RpcError>> + Send;
    /// `session/list`.
    fn list_sessions(
        &self,
        params: SessionListParams,
    ) -> impl Future<Output = Result<SessionPage, RpcError>> + Send;
    /// `session/delete`.
    fn delete_session(&self, id: SessionId) -> impl Future<Output = Result<(), RpcError>> + Send;
    /// `session/close`.
    fn close_session(&self, id: SessionId) -> impl Future<Output = Result<(), RpcError>> + Send;
    /// `session/prompt` — returns when the prompt is *inserted*; the turn
    /// ends via `client_hooks.session_update` idle `state_update`.
    fn prompt(
        &self,
        params: PromptParams,
    ) -> impl Future<Output = Result<crate::domain::PromptAccepted, RpcError>> + Send;
    /// `session/cancel`.
    fn cancel_session(&self, id: SessionId) -> impl Future<Output = Result<(), RpcError>> + Send;
    /// `session/set_config_option` — options arrive via update, so the
    /// result is expected empty.
    fn set_config(
        &self,
        params: SetConfigParams,
    ) -> impl Future<Output = Result<Vec<ConfigOption>, RpcError>> + Send;
    /// `session/tool_call_cancel`.
    fn cancel_tool_call(
        &self,
        session: SessionId,
        tool_call: ToolCallId,
    ) -> impl Future<Output = Result<(), RpcError>> + Send;
    /// `session/terminal_cancel`.
    fn cancel_terminal(
        &self,
        session: SessionId,
        terminal: TerminalId,
    ) -> impl Future<Output = Result<(), RpcError>> + Send;
    /// `auth/login`.
    fn authenticate(
        &self,
        method: AuthMethodId,
    ) -> impl Future<Output = Result<(), RpcError>> + Send;
    /// `auth/logout`.
    fn logout(
        &self,
        method: Option<AuthMethodId>,
    ) -> impl Future<Output = Result<(), RpcError>> + Send;
    /// `_`-prefixed request.
    fn ext_request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> impl Future<Output = Result<Value, RpcError>> + Send;
    /// `_`-prefixed notification.
    fn ext_notify(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> impl Future<Output = Result<(), RpcError>> + Send;
    /// End the in-process agent.
    fn shutdown(&self) -> impl Future<Output = ()> + Send;
}

const fn lane_err(error: RpcError) -> ClientError {
    ClientError::Rejected(error)
}

/// The in-process lane: forwards domain calls to a [`DirectCore`]
/// (the future `crate::server::ServerCore`) with no serialization.
pub struct DirectLane<C> {
    core: C,
}

impl<C: DirectCore> DirectLane<C> {
    /// Wrap `core`.
    pub(crate) const fn new(core: C) -> Self {
        Self { core }
    }
}

impl<C: DirectCore> AgentLaneImpl for DirectLane<C> {
    fn initialize(
        &self,
        capabilities: ClientCapabilities,
        info: Implementation,
    ) -> BoxFuture<'_, Result<InitResult, ClientError>> {
        Box::pin(async move {
            self.core
                .initialize(capabilities, info)
                .await
                .map_err(lane_err)
        })
    }

    fn protocol_version(&self) -> u16 {
        2
    }

    fn new_session(
        &self,
        params: SessionSetupParams,
    ) -> BoxFuture<'_, Result<SetupResult, ClientError>> {
        Box::pin(async move { self.core.new_session(params).await.map_err(lane_err) })
    }

    fn resume_session(
        &self,
        params: ResumeParams,
    ) -> BoxFuture<'_, Result<SetupResult, ClientError>> {
        Box::pin(async move { self.core.resume_session(params).await.map_err(lane_err) })
    }

    fn fork_session(
        &self,
        params: ResumeParams,
    ) -> BoxFuture<'_, Result<SetupResult, ClientError>> {
        Box::pin(async move { self.core.fork_session(params).await.map_err(lane_err) })
    }

    fn list_sessions(
        &self,
        params: SessionListParams,
    ) -> BoxFuture<'_, Result<SessionPage, ClientError>> {
        Box::pin(async move { self.core.list_sessions(params).await.map_err(lane_err) })
    }

    fn delete_session(&self, id: SessionId) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move { self.core.delete_session(id).await.map_err(lane_err) })
    }

    fn close_session(&self, id: SessionId) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move { self.core.close_session(id).await.map_err(lane_err) })
    }

    fn prompt(&self, params: PromptParams) -> BoxFuture<'_, Result<PromptReply, ClientError>> {
        Box::pin(async move {
            let accepted = self.core.prompt(params).await.map_err(lane_err)?;
            Ok(PromptReply::Accepted {
                message_id: Some(accepted.message_id),
            })
        })
    }

    fn cancel_session(&self, id: SessionId) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move { self.core.cancel_session(id).await.map_err(lane_err) })
    }

    fn set_config(
        &self,
        params: SetConfigParams,
    ) -> BoxFuture<'_, Result<Vec<ConfigOption>, ClientError>> {
        Box::pin(async move { self.core.set_config(params).await.map_err(lane_err) })
    }

    fn cancel_tool_call(
        &self,
        session: SessionId,
        tool_call: ToolCallId,
    ) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move {
            self.core
                .cancel_tool_call(session, tool_call)
                .await
                .map_err(lane_err)
        })
    }

    fn cancel_terminal(
        &self,
        session: SessionId,
        terminal: TerminalId,
    ) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move {
            self.core
                .cancel_terminal(session, terminal)
                .await
                .map_err(lane_err)
        })
    }

    fn authenticate(&self, method: AuthMethodId) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move { self.core.authenticate(method).await.map_err(lane_err) })
    }

    fn logout(&self, method: Option<AuthMethodId>) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move { self.core.logout(method).await.map_err(lane_err) })
    }

    fn ext_request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> BoxFuture<'_, Result<Value, ClientError>> {
        Box::pin(async move {
            self.core
                .ext_request(method, params)
                .await
                .map_err(lane_err)
        })
    }

    fn ext_notify(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move { self.core.ext_notify(method, params).await.map_err(lane_err) })
    }

    fn legacy_request(
        &self,
        method: &str,
        _params: Value,
    ) -> BoxFuture<'_, Result<Value, ClientError>> {
        let method = method.to_string();
        Box::pin(async move {
            Err(ClientError::Unsupported(format!(
                "legacy method `{method}` has no direct-lane mapping"
            )))
        })
    }

    fn shutdown(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move { self.core.shutdown().await })
    }
}

/// The client's handler table seen through [`crate::server::AgentHooks`]
/// — what `SessionContext` calls on the direct lane.
///
/// `WireHooks` plays this role on the wire lane; `DirectHooks` performs
/// the identical capability gating (elicitation) with no codec in the
/// way. The v1-only `fs`/`terminal` methods keep their default `None`:
/// the direct lane is always v2, so `SessionContext` never reaches them.
pub struct DirectHooks {
    /// The assembled client handler table.
    hooks: ClientHooks,
    /// What the client advertised at `initialize` (for the elicit gate).
    caps: ClientCapabilities,
}

impl DirectHooks {
    /// Wrap the client's handler table.
    pub const fn new(hooks: ClientHooks, caps: ClientCapabilities) -> Self {
        Self { hooks, caps }
    }
}

impl ServerAgentHooks for DirectHooks {
    async fn session_update(&self, session_id: &SessionId, update: SessionUpdate) {
        self.hooks
            .session_update(SessionNotification {
                session_id: session_id.clone(),
                update,
                meta: None,
                extra: BTreeMap::new(),
            })
            .await;
    }

    async fn request_permission(
        &self,
        params: PermissionRequest,
    ) -> Result<PermissionOutcome, PeerError> {
        self.hooks
            .request_permission(params)
            .await
            .map_err(PeerError::Rejected)
    }

    async fn elicit(
        &self,
        _session_id: &SessionId,
        params: ElicitCreate,
    ) -> Option<Result<ElicitResult, PeerError>> {
        // `None` means "client cannot elicit" — same gate the wire
        // lane's `WireHooks` applies from the advertised capabilities.
        let supported = self
            .caps
            .elicitation
            .as_ref()
            .is_some_and(|caps| match &params.mode {
                Open::Known(ElicitMode::Form(_)) => caps.form.is_some(),
                Open::Known(ElicitMode::Url(_)) => caps.url.is_some(),
                _ => true,
            });
        if !supported {
            return None;
        }
        Some(self.hooks.elicit(params).await.map_err(PeerError::Rejected))
    }

    async fn ext_request(
        &self,
        method: &ExtMethod,
        params: Option<Value>,
    ) -> Result<Value, PeerError> {
        self.hooks
            .ext_request(method.clone(), params)
            .await
            .map_err(PeerError::Rejected)
    }

    async fn notification(&self, method: &str, params: Option<Value>) -> Result<(), PeerError> {
        self.hooks
            .notification(crate::rpc::Notification {
                method: method.to_owned(),
                params,
            })
            .await;
        Ok(())
    }
}

/// A [`ServerCore`] bound to one client's hook table — the
/// [`DirectCore`] the direct lane drives.
///
/// `ServerCore` takes agent→client hooks per session call (the wire
/// lane builds one codec adapter per connection); `DirectServer` binds
/// the client's [`ClientHooks`] once, so every session reaches the same
/// handler table. Closing `shutdown` ends the [`Connection`]'s driver
/// (and with it the host); the channel also dies with the lane, so the
/// last [`Client`](super::Client) clone dropping shuts the host down.
pub struct DirectServer {
    /// The transport-independent server core.
    core: Arc<ServerCore>,
    /// The client's handler table, wrapped for the server.
    hooks: ServerHooks,
    /// Closing this channel ends the connection driver's `drive()` poll.
    shutdown: async_channel::Sender<()>,
}

impl DirectServer {
    /// Bind `core` to the client's handler table; `shutdown` is closed by
    /// [`DirectCore::shutdown`] to end the connection's driver.
    pub const fn new(
        core: Arc<ServerCore>,
        hooks: ServerHooks,
        shutdown: async_channel::Sender<()>,
    ) -> Self {
        Self {
            core,
            hooks,
            shutdown,
        }
    }
}

impl DirectCore for DirectServer {
    async fn initialize(
        &self,
        capabilities: ClientCapabilities,
        info: Implementation,
    ) -> Result<InitResult, RpcError> {
        // The real capability-exchange path: `initialize` must be the
        // connection's first request, exactly as on the wire.
        self.core.gate_request("initialize")?;
        self.core
            .initialize(crate::server::InitializeParams {
                protocol_version: 2,
                capabilities,
                info: Some(info),
                meta: None,
            })
            .await
    }

    async fn new_session(&self, params: SessionSetupParams) -> Result<SetupResult, RpcError> {
        self.core.new_session(params, self.hooks.clone()).await
    }

    async fn resume_session(&self, params: ResumeParams) -> Result<SetupResult, RpcError> {
        self.core.resume_session(params, self.hooks.clone()).await
    }

    async fn fork_session(&self, params: ResumeParams) -> Result<SetupResult, RpcError> {
        self.core.fork_session(params, self.hooks.clone()).await
    }

    async fn list_sessions(&self, params: SessionListParams) -> Result<SessionPage, RpcError> {
        self.core.list_sessions(params).await
    }

    async fn delete_session(&self, id: SessionId) -> Result<(), RpcError> {
        self.core.delete_session(&id).await
    }

    async fn close_session(&self, id: SessionId) -> Result<(), RpcError> {
        self.core.close_session(&id).await
    }

    async fn prompt(&self, mut params: PromptParams) -> Result<PromptAccepted, RpcError> {
        // The v2 wire adapter fills an absent idempotency key itself;
        // keep parity so every direct prompt is idempotent.
        if params.idempotency_key.is_none() {
            params.idempotency_key = Some(MessageId::new().to_string());
        }
        let ticket = self.core.prompt(params).await?;
        Ok(PromptAccepted {
            message_id: ticket.message_id().clone(),
            meta: None,
            extra: BTreeMap::new(),
        })
    }

    fn cancel_session(&self, id: SessionId) -> impl Future<Output = Result<(), RpcError>> + Send {
        // A notification on the wire: unknown sessions are ignored.
        self.core.cancel_session(&id);
        std::future::ready(Ok(()))
    }

    async fn set_config(&self, params: SetConfigParams) -> Result<Vec<ConfigOption>, RpcError> {
        self.core.set_config(params).await
    }

    fn cancel_tool_call(
        &self,
        session: SessionId,
        tool_call: ToolCallId,
    ) -> impl Future<Output = Result<(), RpcError>> + Send {
        self.core.tool_call_cancel(&session, &tool_call);
        std::future::ready(Ok(()))
    }

    fn cancel_terminal(
        &self,
        session: SessionId,
        terminal: TerminalId,
    ) -> impl Future<Output = Result<(), RpcError>> + Send {
        self.core.terminal_cancel(&session, &terminal);
        std::future::ready(Ok(()))
    }

    async fn authenticate(&self, method: AuthMethodId) -> Result<(), RpcError> {
        self.core
            .auth_login(crate::server::AuthLoginRequest {
                method_id: method,
                meta: None,
                extra: std::collections::BTreeMap::new(),
            })
            .await
    }

    async fn logout(&self, method: Option<AuthMethodId>) -> Result<(), RpcError> {
        self.core.auth_logout(method).await
    }

    async fn ext_request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> Result<Value, RpcError> {
        self.core.ext_request(method.as_str(), params).await
    }

    async fn ext_notify(&self, method: ExtMethod, params: Option<Value>) -> Result<(), RpcError> {
        // The wire lane drops `_` notifications outright; in-process the
        // ext handler still observes them — its result is discarded and
        // its error reaches the caller.
        self.core
            .ext_request(method.as_str(), params)
            .await
            .map(|_| ())
    }

    fn shutdown(&self) -> impl Future<Output = ()> + Send {
        // Ends the `drive()` side of the connection driver's `or`: the
        // core's `on_disconnect` teardown then closes sessions and shuts
        // the host down.
        self.shutdown.close();
        std::future::ready(())
    }
}
