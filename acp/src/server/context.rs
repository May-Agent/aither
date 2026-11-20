//! [`SessionContext`] — the handle a [`SessionHost`] and its
//! [`Session`]s use to talk back to the client.
//!
//! The context is a thin, `Clone`-able façade over [`AgentHooks`], the
//! lane abstraction implemented by `WireAdapter` (JSON-RPC) and by the
//! in-process client (direct lane). Everything the host can do —
//! emitting `session/update`s, requesting permissions and elicitations,
//! v1 `fs`/`terminal` client lanes, v2 agent-owned terminals and
//! subagents — goes through here, with `requires_action` bookkeeping
//! applied automatically to interactive requests.

use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::domain::{
    AvailableCommand, AvailableCommandsUpdate, ClientCapabilities, ConfigOption,
    ConfigOptionUpdate, ElicitComplete, ElicitCreate, ElicitResult, ForegroundState, Notice,
    NoticeSeverity, Open, Patch, PermissionOutcome, PermissionRequest, ReplayComplete, SessionId,
    SessionInfoUpdate, SessionUpdate, StateUpdate, StopReason, SubagentUpdate, TerminalExitStatus,
    TerminalId, TerminalOutputChunk, TerminalUpdate, UsageUpdate, open::ExtMethod,
};
use crate::rpc::{PeerError, RpcError};
use crate::wire::v1;

use super::table::SessionShared;

/// The protocol version negotiated on this connection.
///
/// The in-process direct lane always reports [`ProtocolVersion::V2`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtocolVersion {
    /// ACP v1 (`protocolVersion: 1`).
    V1,
    /// ACP v2 (`protocolVersion: 2`).
    V2,
}

impl ProtocolVersion {
    /// The numeric `protocolVersion` advertised on the wire.
    #[must_use]
    pub const fn number(self) -> u16 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
        }
    }

    /// Whether the negotiated version is v1.
    #[must_use]
    pub const fn is_v1(self) -> bool {
        matches!(self, Self::V1)
    }
}

/// Agent→client traffic, implemented by the transport lane.
///
/// `WireAdapter` implements this over [`crate::rpc::Peer`]; the
/// in-process client implements it over its handler table. Methods that
/// a version cannot express return `None` (v1-only `fs`/`terminal` on a
/// v2 connection, or elicitation when the client did not advertise it).
///
/// All request/notification methods are fire-and-forget w.r.t. update
/// ordering: items queued on the lane precede the response of whichever
/// in-flight request queued them (the RPC engine flushes causally).
pub trait AgentHooks: Send + Sync + 'static {
    /// Emit a `session/update` notification. Implementations translate
    /// (v1 codec) or pass through (v2/direct) as appropriate.
    fn session_update(
        &self,
        session_id: &SessionId,
        update: SessionUpdate,
    ) -> impl Future<Output = ()> + Send;

    /// `session/request_permission`.
    fn request_permission(
        &self,
        params: PermissionRequest,
    ) -> impl Future<Output = Result<PermissionOutcome, PeerError>> + Send;

    /// `elicitation/create`. `None` means the client cannot elicit.
    fn elicit(
        &self,
        session_id: &SessionId,
        params: ElicitCreate,
    ) -> impl Future<Output = Option<Result<ElicitResult, PeerError>>> + Send;

    /// A custom `_`-prefixed request to the client.
    fn ext_request(
        &self,
        method: &ExtMethod,
        params: Option<Value>,
    ) -> impl Future<Output = Result<Value, PeerError>> + Send;

    /// A raw JSON-RPC notification to the client.
    fn notification(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> impl Future<Output = Result<(), PeerError>> + Send;

    /// `fs/read_text_file` (v1 only). `None` when unsupported.
    fn fs_read_text(
        &self,
        session_id: &SessionId,
        path: &Path,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> impl Future<Output = Option<Result<String, PeerError>>> + Send {
        let _ = (session_id, path, line, limit);
        std::future::ready(None)
    }

    /// `fs/write_text_file` (v1 only). `None` when unsupported.
    fn fs_write_text(
        &self,
        session_id: &SessionId,
        path: &Path,
        content: &str,
    ) -> impl Future<Output = Option<Result<(), PeerError>>> + Send {
        let _ = (session_id, path, content);
        std::future::ready(None)
    }

    /// `terminal/create` (v1 only). `None` when unsupported.
    fn terminal_create(
        &self,
        session_id: &SessionId,
        params: v1::TerminalCreateParams,
    ) -> impl Future<Output = Option<Result<v1::TerminalCreateResult, PeerError>>> + Send {
        let _ = (session_id, params);
        std::future::ready(None)
    }

    /// `terminal/output` (v1 only). `None` when unsupported.
    fn terminal_output(
        &self,
        session_id: &SessionId,
        terminal_id: &str,
    ) -> impl Future<Output = Option<Result<v1::TerminalOutputResult, PeerError>>> + Send {
        let _ = (session_id, terminal_id);
        std::future::ready(None)
    }

    /// `terminal/wait_for_exit` (v1 only). `None` when unsupported.
    fn terminal_wait_for_exit(
        &self,
        session_id: &SessionId,
        terminal_id: &str,
    ) -> impl Future<Output = Option<Result<v1::TerminalExitStatus, PeerError>>> + Send {
        let _ = (session_id, terminal_id);
        std::future::ready(None)
    }

    /// `terminal/kill` (v1 only). `None` when unsupported.
    fn terminal_kill(
        &self,
        session_id: &SessionId,
        terminal_id: &str,
    ) -> impl Future<Output = Option<Result<(), PeerError>>> + Send {
        let _ = (session_id, terminal_id);
        std::future::ready(None)
    }

    /// `terminal/release` (v1 only). `None` when unsupported.
    fn terminal_release(
        &self,
        session_id: &SessionId,
        terminal_id: &str,
    ) -> impl Future<Output = Option<Result<(), PeerError>>> + Send {
        let _ = (session_id, terminal_id);
        std::future::ready(None)
    }
}

/// A boxed agent→client lane future.
type HookFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A boxed lane reply that is `None` when the lane cannot serve the
/// call (the caller then reports `method not found`).
type HookReply<'a, T> = HookFuture<'a, Option<Result<T, PeerError>>>;

/// Object-safe twin of [`AgentHooks`] (private, boxing).
trait AgentHooksImpl: Send + Sync + 'static {
    fn session_update<'a>(
        &'a self,
        session_id: &'a SessionId,
        update: SessionUpdate,
    ) -> HookFuture<'a, ()>;
    fn request_permission(
        &self,
        params: PermissionRequest,
    ) -> HookFuture<'_, Result<PermissionOutcome, PeerError>>;
    fn elicit<'a>(
        &'a self,
        session_id: &'a SessionId,
        params: ElicitCreate,
    ) -> HookReply<'a, ElicitResult>;
    fn ext_request<'a>(
        &'a self,
        method: &'a ExtMethod,
        params: Option<Value>,
    ) -> HookFuture<'a, Result<Value, PeerError>>;
    fn notification<'a>(
        &'a self,
        method: &'a str,
        params: Option<Value>,
    ) -> HookFuture<'a, Result<(), PeerError>>;
    fn fs_read_text<'a>(
        &'a self,
        session_id: &'a SessionId,
        path: &'a Path,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> HookReply<'a, String>;
    fn fs_write_text<'a>(
        &'a self,
        session_id: &'a SessionId,
        path: &'a Path,
        content: &'a str,
    ) -> HookReply<'a, ()>;
    fn terminal_create<'a>(
        &'a self,
        session_id: &'a SessionId,
        params: v1::TerminalCreateParams,
    ) -> HookReply<'a, v1::TerminalCreateResult>;
    fn terminal_output<'a>(
        &'a self,
        session_id: &'a SessionId,
        terminal_id: &'a str,
    ) -> HookReply<'a, v1::TerminalOutputResult>;
    fn terminal_wait_for_exit<'a>(
        &'a self,
        session_id: &'a SessionId,
        terminal_id: &'a str,
    ) -> HookReply<'a, v1::TerminalExitStatus>;
    fn terminal_kill<'a>(
        &'a self,
        session_id: &'a SessionId,
        terminal_id: &'a str,
    ) -> HookReply<'a, ()>;
    fn terminal_release<'a>(
        &'a self,
        session_id: &'a SessionId,
        terminal_id: &'a str,
    ) -> HookReply<'a, ()>;
}

impl<H: AgentHooks> AgentHooksImpl for H {
    fn session_update<'a>(
        &'a self,
        session_id: &'a SessionId,
        update: SessionUpdate,
    ) -> HookFuture<'a, ()> {
        Box::pin(AgentHooks::session_update(self, session_id, update))
    }

    fn request_permission(
        &self,
        params: PermissionRequest,
    ) -> HookFuture<'_, Result<PermissionOutcome, PeerError>> {
        Box::pin(AgentHooks::request_permission(self, params))
    }

    fn elicit<'a>(
        &'a self,
        session_id: &'a SessionId,
        params: ElicitCreate,
    ) -> HookReply<'a, ElicitResult> {
        Box::pin(AgentHooks::elicit(self, session_id, params))
    }

    fn ext_request<'a>(
        &'a self,
        method: &'a ExtMethod,
        params: Option<Value>,
    ) -> HookFuture<'a, Result<Value, PeerError>> {
        Box::pin(AgentHooks::ext_request(self, method, params))
    }

    fn notification<'a>(
        &'a self,
        method: &'a str,
        params: Option<Value>,
    ) -> HookFuture<'a, Result<(), PeerError>> {
        Box::pin(AgentHooks::notification(self, method, params))
    }

    fn fs_read_text<'a>(
        &'a self,
        session_id: &'a SessionId,
        path: &'a Path,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> HookReply<'a, String> {
        Box::pin(AgentHooks::fs_read_text(
            self, session_id, path, line, limit,
        ))
    }

    fn fs_write_text<'a>(
        &'a self,
        session_id: &'a SessionId,
        path: &'a Path,
        content: &'a str,
    ) -> HookReply<'a, ()> {
        Box::pin(AgentHooks::fs_write_text(self, session_id, path, content))
    }

    fn terminal_create<'a>(
        &'a self,
        session_id: &'a SessionId,
        params: v1::TerminalCreateParams,
    ) -> HookReply<'a, v1::TerminalCreateResult> {
        Box::pin(AgentHooks::terminal_create(self, session_id, params))
    }

    fn terminal_output<'a>(
        &'a self,
        session_id: &'a SessionId,
        terminal_id: &'a str,
    ) -> HookReply<'a, v1::TerminalOutputResult> {
        Box::pin(AgentHooks::terminal_output(self, session_id, terminal_id))
    }

    fn terminal_wait_for_exit<'a>(
        &'a self,
        session_id: &'a SessionId,
        terminal_id: &'a str,
    ) -> HookReply<'a, v1::TerminalExitStatus> {
        Box::pin(AgentHooks::terminal_wait_for_exit(
            self,
            session_id,
            terminal_id,
        ))
    }

    fn terminal_kill<'a>(
        &'a self,
        session_id: &'a SessionId,
        terminal_id: &'a str,
    ) -> HookReply<'a, ()> {
        Box::pin(AgentHooks::terminal_kill(self, session_id, terminal_id))
    }

    fn terminal_release<'a>(
        &'a self,
        session_id: &'a SessionId,
        terminal_id: &'a str,
    ) -> HookReply<'a, ()> {
        Box::pin(AgentHooks::terminal_release(self, session_id, terminal_id))
    }
}

/// A type-erased [`AgentHooks`] — what [`SessionContext`] stores.
///
/// Construct with [`AnyAgentHooks::new`].
#[derive(Clone)]
pub struct AnyAgentHooks {
    inner: Arc<dyn AgentHooksImpl>,
}

impl std::fmt::Debug for AnyAgentHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyAgentHooks").finish_non_exhaustive()
    }
}

impl AnyAgentHooks {
    /// Erase `hooks`.
    pub fn new<H: AgentHooks>(hooks: H) -> Self {
        Self {
            inner: Arc::new(hooks),
        }
    }
}

/// Map a lane error to the [`RpcError`] the host should see.
fn lane_error(error: PeerError) -> RpcError {
    match error {
        PeerError::Rejected(error) => error,
        other => RpcError::internal_error(format!("client lane failed: {other}")),
    }
}

/// The context a [`crate::server::Session`] (or
/// [`crate::server::SessionHost`]) uses to reach its client.
///
/// Cheap to clone; every clone addresses the same session. On the wire
/// lane each call becomes `session/update` notifications or client
/// requests; on the in-process direct lane (always
/// [`ProtocolVersion::V2`]) calls reach the client handler table
/// directly.
#[derive(Clone)]
pub struct SessionContext {
    pub(crate) inner: Arc<ContextInner>,
}

impl std::fmt::Debug for SessionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionContext")
            .field("session_id", &self.inner.session_id)
            .field("protocol", &self.inner.protocol)
            .finish_non_exhaustive()
    }
}

/// Shared context state (one per session).
#[derive(Debug)]
pub struct ContextInner {
    /// The session id.
    pub(crate) session_id: SessionId,
    /// Agent→client lane.
    pub(crate) hooks: AnyAgentHooks,
    /// Negotiated protocol version.
    pub(crate) protocol: ProtocolVersion,
    /// What the client advertised at `initialize`.
    pub(crate) client_caps: ClientCapabilities,
    /// What the client called itself.
    pub(crate) client_info: Option<crate::domain::Implementation>,
    /// Per-session scheduling state.
    pub(crate) shared: Arc<SessionShared>,
}

impl SessionContext {
    /// Build the context for a session (called by [`ServerCore`]).
    pub(crate) fn new(
        session_id: SessionId,
        hooks: AnyAgentHooks,
        protocol: ProtocolVersion,
        client_caps: ClientCapabilities,
        client_info: Option<crate::domain::Implementation>,
        shared: Arc<SessionShared>,
    ) -> Self {
        Self {
            inner: Arc::new(ContextInner {
                session_id,
                hooks,
                protocol,
                client_caps,
                client_info,
                shared,
            }),
        }
    }

    /// The session this context addresses.
    #[must_use]
    pub fn session_id(&self) -> &SessionId {
        &self.inner.session_id
    }

    /// The negotiated protocol version.
    #[must_use]
    pub fn protocol_version(&self) -> ProtocolVersion {
        self.inner.protocol
    }

    /// What the client advertised at `initialize`.
    #[must_use]
    pub fn client_caps(&self) -> &ClientCapabilities {
        &self.inner.client_caps
    }

    /// The client's `Implementation` from `initialize`, if any.
    #[must_use]
    pub fn client_info(&self) -> Option<&crate::domain::Implementation> {
        self.inner.client_info.as_ref()
    }

    /// Shared scheduling state (crate-internal).
    pub(crate) fn shared(&self) -> &Arc<SessionShared> {
        &self.inner.shared
    }

    /// Emit an arbitrary `session/update`.
    ///
    /// This is the escape hatch behind every typed convenience method.
    /// On v1 connections updates the codec cannot express are dropped;
    /// `Err` only fires when the lane itself is gone.
    ///
    /// # Errors
    ///
    /// Reserved for lane emission failures; the bundled lanes absorb
    /// send failures internally and currently always return `Ok`.
    pub async fn update(&self, update: impl Into<SessionUpdate>) -> Result<(), RpcError> {
        self.inner
            .hooks
            .inner
            .session_update(&self.inner.session_id, update.into())
            .await;
        Ok(())
    }

    /// Emit a `state_update` with a foreground state (internal helper —
    /// never counted for `requires_action` bookkeeping).
    pub(crate) async fn emit_state(
        &self,
        state: ForegroundState,
        stop_reason: Option<Open<StopReason>>,
        error: Option<Value>,
    ) {
        self.inner
            .hooks
            .inner
            .session_update(
                &self.inner.session_id,
                SessionUpdate::StateUpdate(StateUpdate {
                    state: Open::Known(state),
                    stop_reason,
                    background: Vec::new(),
                    error,
                    meta: None,
                    extra: BTreeMap::new(),
                }),
            )
            .await;
    }

    /// Emit `replay_complete` (internal, end of a `session/resume`
    /// drain). On v1 the codec drops it — replay ends with the response.
    pub(crate) async fn emit_replay_complete(&self) {
        self.inner
            .hooks
            .inner
            .session_update(
                &self.inner.session_id,
                SessionUpdate::ReplayComplete(ReplayComplete {
                    meta: None,
                    extra: BTreeMap::new(),
                }),
            )
            .await;
    }

    /// Begin an interactive agent→client request; emits the
    /// `requires_action` transition when this is the turn's first.
    pub(crate) async fn action_begin(&self) -> Option<Arc<super::table::TurnState>> {
        let turn = self.inner.shared.current.lock().expect("current").clone();
        if let Some(turn) = &turn
            && turn.action_begin()
        {
            self.emit_state(ForegroundState::RequiresAction, None, None)
                .await;
        }
        turn
    }

    /// End an interactive request; emits `running` when the turn's last
    /// outstanding request resolved.
    pub(crate) async fn action_end(&self, turn: Option<Arc<super::table::TurnState>>) {
        if let Some(turn) = turn
            && turn.action_end()
        {
            self.emit_state(ForegroundState::Running, None, None).await;
        }
    }

    /// `session/request_permission`, tracked for `requires_action`.
    ///
    /// # Errors
    ///
    /// Propagates the client's error response or a lane failure.
    pub async fn request_permission(
        &self,
        request: PermissionRequest,
    ) -> Result<PermissionOutcome, RpcError> {
        let turn = self.action_begin().await;
        let result = self.inner.hooks.inner.request_permission(request).await;
        self.action_end(turn).await;
        result.map_err(lane_error)
    }

    /// `elicitation/create`; `None` when the client cannot elicit.
    ///
    /// For url-mode elicitations the request resolves once the client
    /// acknowledges; [`elicit_wait_complete`](Self::elicit_wait_complete)
    /// waits for the client's `elicitation/complete` notification.
    pub async fn elicit(&self, create: ElicitCreate) -> Option<Result<ElicitResult, RpcError>> {
        let turn = self.action_begin().await;
        let result = self
            .inner
            .hooks
            .inner
            .elicit(&self.inner.session_id, create)
            .await;
        self.action_end(turn).await;
        result.map(|r| r.map_err(lane_error))
    }

    /// Wait for the client's `elicitation/complete` notification for
    /// `elicitation_id` (url-mode flow). Returns the notification's
    /// payload; a dropped waiter means the session ended.
    ///
    /// # Panics
    ///
    /// Panics if the session's elicit-waiter mutex is poisoned.
    pub async fn elicit_wait_complete(
        &self,
        elicitation_id: &crate::domain::ElicitationId,
    ) -> Option<ElicitComplete> {
        let (tx, rx) = async_channel::bounded(1);
        self.inner
            .shared
            .elicits
            .lock()
            .expect("elicits")
            .insert(elicitation_id.clone(), tx);
        rx.recv().await.ok()
    }

    /// `fs/read_text_file` on a v1 client. `None` on v2 or when the
    /// client did not advertise `fs.readTextFile`.
    pub async fn fs_read(
        &self,
        path: impl AsRef<Path>,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> Option<Result<String, RpcError>> {
        if self.inner.protocol.is_v1() && self.fs_cap("readTextFile") {
            let turn = self.action_begin().await;
            let result = self
                .inner
                .hooks
                .inner
                .fs_read_text(&self.inner.session_id, path.as_ref(), line, limit)
                .await;
            self.action_end(turn).await;
            result.map(|r| r.map_err(lane_error))
        } else {
            None
        }
    }

    /// `fs/write_text_file` on a v1 client. `None` on v2 or when the
    /// client did not advertise `fs.writeTextFile`.
    pub async fn fs_write(
        &self,
        path: impl AsRef<Path>,
        content: impl AsRef<str>,
    ) -> Option<Result<(), RpcError>> {
        if self.inner.protocol.is_v1() && self.fs_cap("writeTextFile") {
            let turn = self.action_begin().await;
            let result = self
                .inner
                .hooks
                .inner
                .fs_write_text(&self.inner.session_id, path.as_ref(), content.as_ref())
                .await;
            self.action_end(turn).await;
            result.map(|r| r.map_err(lane_error))
        } else {
            None
        }
    }

    /// The client's `fs` sub-capability flag (`clientCapabilities.fs`).
    fn fs_cap(&self, key: &str) -> bool {
        self.inner
            .client_caps
            .extra
            .get("fs")
            .and_then(|fs| fs.get(key))
            .is_some_and(|v| v.as_bool().unwrap_or(false))
    }

    /// Whether the v1 client advertises `terminal` support.
    fn terminal_cap(&self) -> bool {
        self.inner
            .client_caps
            .extra
            .get("terminal")
            .is_some_and(|v| v.as_bool().unwrap_or(false))
    }

    /// The v1 client-owned terminal lane (`terminal/*` requests).
    ///
    /// `None` on v2 (including the direct lane) or when the client did
    /// not advertise `clientCapabilities.terminal`.
    #[must_use]
    pub fn v1_terminal(&self) -> Option<V1TerminalLane> {
        if self.inner.protocol.is_v1() && self.terminal_cap() {
            Some(V1TerminalLane { cx: self.clone() })
        } else {
            None
        }
    }

    /// Open a v2 agent-owned display terminal.
    ///
    /// Emits a `terminal_update` carrying `command` (the client renders
    /// and optionally runs it); use the handle to append output and
    /// report exit. On v1 these updates are dropped by the codec — the
    /// v1 client-side lane is [`v1_terminal`](Self::v1_terminal).
    pub async fn terminal(&self, command: impl Into<String>) -> TerminalHandle {
        let terminal_id = TerminalId::new();
        let mut update = TerminalUpdate::new(terminal_id.clone());
        update.command = Patch::Set(command.into());
        let _ = self.update(SessionUpdate::TerminalUpdate(update)).await;
        TerminalHandle {
            cx: self.clone(),
            terminal_id,
        }
    }

    /// Open a v2 subagent lane bound to `session_id` (the child session).
    ///
    /// Emits `subagent_update`s so the client can render delegated work.
    /// The v1 codec drops them — v1 subagent presentation is a client
    /// extension concern the `ext` lane can fill.
    pub async fn subagent(
        &self,
        session_id: impl Into<SessionId>,
        title: Option<String>,
    ) -> SubagentHandle {
        let session_id = session_id.into();
        let mut update = SubagentUpdate {
            session_id: session_id.clone(),
            ..Default::default()
        };
        if let Some(title) = title {
            update.title = Patch::Set(title);
        }
        let _ = self.update(SessionUpdate::SubagentUpdate(update)).await;
        SubagentHandle {
            cx: self.clone(),
            session_id,
        }
    }

    /// Emit the full current `config_option` set.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn set_config_options(&self, options: Vec<ConfigOption>) -> Result<(), RpcError> {
        self.update(SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate {
            config_options: options,
            meta: None,
            extra: BTreeMap::new(),
        }))
        .await
    }

    /// Emit the full current available-commands list.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn set_commands(&self, commands: Vec<AvailableCommand>) -> Result<(), RpcError> {
        self.update(SessionUpdate::AvailableCommandsUpdate(
            AvailableCommandsUpdate {
                available_commands: commands,
                meta: None,
                extra: BTreeMap::new(),
            },
        ))
        .await
    }

    /// Patch the session's `title` (and `updatedAt`) as seen in
    /// `session/list`.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn set_session_info(
        &self,
        title: Patch<String>,
        updated_at: Patch<String>,
    ) -> Result<(), RpcError> {
        self.update(SessionUpdate::SessionInfoUpdate(SessionInfoUpdate {
            title,
            updated_at,
            meta: Patch::Unchanged,
            extra: BTreeMap::new(),
        }))
        .await
    }

    /// Emit a `usage_update`.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn usage(&self, usage: UsageUpdate) -> Result<(), RpcError> {
        self.update(SessionUpdate::UsageUpdate(usage)).await
    }

    /// Emit a transient `notice`.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn notice(
        &self,
        severity: impl Into<Open<NoticeSeverity>>,
        title: impl Into<String>,
        description: Option<String>,
    ) -> Result<(), RpcError> {
        self.update(SessionUpdate::Notice(Notice {
            severity: severity.into(),
            title: title.into(),
            description,
            meta: None,
            extra: BTreeMap::new(),
        }))
        .await
    }

    /// A `_`-prefixed extension request to the client, tracked for
    /// `requires_action`.
    ///
    /// # Errors
    ///
    /// Propagates the client's error response or a lane failure.
    pub async fn ext_request(
        &self,
        method: &ExtMethod,
        params: Option<Value>,
    ) -> Result<Value, RpcError> {
        let turn = self.action_begin().await;
        let result = self.inner.hooks.inner.ext_request(method, params).await;
        self.action_end(turn).await;
        result.map_err(lane_error)
    }

    /// A raw extension notification to the client (not tracked — the
    /// agent is not waiting on anything).
    ///
    /// # Errors
    ///
    /// `Err` when the client lane is gone (unattached or closed).
    pub async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), RpcError> {
        self.inner
            .hooks
            .inner
            .notification(method, params)
            .await
            .map_err(lane_error)
    }
}

/// The v1 client-owned terminal lane (`terminal/create` &c.).
///
/// Only obtainable through [`SessionContext::v1_terminal`] on v1
/// connections where the client advertised `terminal` support.
#[derive(Clone, Debug)]
pub struct V1TerminalLane {
    cx: SessionContext,
}

impl V1TerminalLane {
    /// `terminal/create`.
    ///
    /// # Errors
    ///
    /// Propagates the client's error response or a lane failure.
    pub async fn create(
        &self,
        command: impl Into<String>,
        args: Vec<String>,
        cwd: Option<PathBuf>,
        output_byte_limit: Option<u64>,
    ) -> Result<TerminalId, RpcError> {
        let params = v1::TerminalCreateParams {
            session_id: self.cx.session_id().to_string(),
            command: command.into(),
            args,
            env: Vec::new(),
            cwd,
            output_byte_limit,
            meta: None,
        };
        let turn = self.cx.action_begin().await;
        let result = self
            .cx
            .inner
            .hooks
            .inner
            .terminal_create(&self.cx.inner.session_id, params)
            .await;
        self.cx.action_end(turn).await;
        match result {
            Some(Ok(result)) => Ok(TerminalId::from(result.terminal_id)),
            Some(Err(e)) => Err(lane_error(e)),
            None => Err(RpcError::method_not_found("terminal/create")),
        }
    }

    /// `terminal/output`.
    ///
    /// # Errors
    ///
    /// Propagates the client's error response or a lane failure.
    pub async fn output(
        &self,
        terminal_id: &TerminalId,
    ) -> Result<v1::TerminalOutputResult, RpcError> {
        let turn = self.cx.action_begin().await;
        let result = self
            .cx
            .inner
            .hooks
            .inner
            .terminal_output(&self.cx.inner.session_id, terminal_id.as_str())
            .await;
        self.cx.action_end(turn).await;
        result.map_or_else(
            || Err(RpcError::method_not_found("terminal/output")),
            |r| r.map_err(lane_error),
        )
    }

    /// `terminal/wait_for_exit`.
    ///
    /// # Errors
    ///
    /// Propagates the client's error response or a lane failure.
    pub async fn wait_for_exit(
        &self,
        terminal_id: &TerminalId,
    ) -> Result<v1::TerminalExitStatus, RpcError> {
        let turn = self.cx.action_begin().await;
        let result = self
            .cx
            .inner
            .hooks
            .inner
            .terminal_wait_for_exit(&self.cx.inner.session_id, terminal_id.as_str())
            .await;
        self.cx.action_end(turn).await;
        result.map_or_else(
            || Err(RpcError::method_not_found("terminal/wait_for_exit")),
            |r| r.map_err(lane_error),
        )
    }

    /// `terminal/kill`.
    ///
    /// # Errors
    ///
    /// Propagates the client's error response or a lane failure.
    pub async fn kill(&self, terminal_id: &TerminalId) -> Result<(), RpcError> {
        let turn = self.cx.action_begin().await;
        let result = self
            .cx
            .inner
            .hooks
            .inner
            .terminal_kill(&self.cx.inner.session_id, terminal_id.as_str())
            .await;
        self.cx.action_end(turn).await;
        result.map_or_else(
            || Err(RpcError::method_not_found("terminal/kill")),
            |r| r.map_err(lane_error),
        )
    }

    /// `terminal/release`.
    ///
    /// # Errors
    ///
    /// Propagates the client's error response or a lane failure.
    pub async fn release(&self, terminal_id: &TerminalId) -> Result<(), RpcError> {
        let turn = self.cx.action_begin().await;
        let result = self
            .cx
            .inner
            .hooks
            .inner
            .terminal_release(&self.cx.inner.session_id, terminal_id.as_str())
            .await;
        self.cx.action_end(turn).await;
        result.map_or_else(
            || Err(RpcError::method_not_found("terminal/release")),
            |r| r.map_err(lane_error),
        )
    }
}

/// Handle for a v2 agent-owned terminal (display lane).
#[derive(Clone, Debug)]
pub struct TerminalHandle {
    cx: SessionContext,
    terminal_id: TerminalId,
}

impl TerminalHandle {
    /// This terminal's id.
    #[must_use]
    pub const fn id(&self) -> &TerminalId {
        &self.terminal_id
    }

    /// Patch the terminal's public fields (`pid`, `cwd`, `size`,
    /// `capabilities`, ...) while keeping `terminal_id` fixed.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn update(&self, update: TerminalUpdate) -> Result<(), RpcError> {
        let mut update = update;
        update.terminal_id = self.terminal_id.clone();
        self.cx.update(SessionUpdate::TerminalUpdate(update)).await
    }

    /// Append output (`terminal_output_chunk`).
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn write(&self, data: impl Into<String>) -> Result<(), RpcError> {
        self.cx
            .update(SessionUpdate::TerminalOutputChunk(TerminalOutputChunk {
                terminal_id: self.terminal_id.clone(),
                data: data.into(),
                truncated: None,
                meta: None,
                extra: BTreeMap::new(),
            }))
            .await
    }

    /// Report process exit (`exitStatus` patch).
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn exit(&self, status: TerminalExitStatus) -> Result<(), RpcError> {
        let mut update = TerminalUpdate::new(self.terminal_id.clone());
        update.exit_status = Patch::Set(status);
        self.cx.update(SessionUpdate::TerminalUpdate(update)).await
    }

    /// Wait for the client to cancel this terminal
    /// (`session/terminal_cancel`).
    ///
    /// # Panics
    ///
    /// Panics if the session's cancel-waiter mutex is poisoned.
    pub async fn cancelled(&self) {
        let (tx, rx) = async_channel::bounded(1);
        self.cx
            .shared()
            .terminal_cancels
            .lock()
            .expect("terminal_cancels")
            .insert(self.terminal_id.clone(), tx);
        let _ = rx.recv().await;
    }
}

/// Handle for a v2 subagent lane.
#[derive(Clone, Debug)]
pub struct SubagentHandle {
    cx: SessionContext,
    session_id: SessionId,
}

impl SubagentHandle {
    /// The child session id.
    #[must_use]
    pub const fn id(&self) -> &SessionId {
        &self.session_id
    }

    /// Patch the subagent presentation (`title`, `description`,
    /// `state`, ...).
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn update(&self, update: SubagentUpdate) -> Result<(), RpcError> {
        let mut update = update;
        update.session_id = self.session_id.clone();
        self.cx.update(SessionUpdate::SubagentUpdate(update)).await
    }
}
