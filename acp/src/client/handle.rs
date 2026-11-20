//! The [`Client`] handle and the per-session registry.
//!
//! `SessionRegistry` maps `SessionId -> Arc<SessionCell>`; a cell owns the
//! [`SessionReducer`], the per-session [`V1Codec`] (harmless on v2), and
//! the subscriber fan-out. Cells are created lazily — an early
//! `session/update` or a `session/load` replay can arrive before the
//! lane's `open()` continuation registers the cell — so `get_or_create`
//! is the only insertion path and demux never loses updates.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde_json::Value;

use crate::domain::{
    AuthMethod, AuthMethodId, ClientCapabilities, ExtMethod, Implementation, InitResult, McpServer,
    Open, ReplayCursor, ReplayStart, SessionId, SessionListParams, SessionNotification,
    SessionPage, SessionSetupParams, SessionUpdate,
};
use crate::wire::codec::V1Codec;

use super::connection::AnyAgentLane;
use super::error::ClientError;
use super::session::ClientSession;
use crate::state::{SessionEvent, SessionReducer, SessionState};

/// Per-session state shared by the lane, hooks, and `ClientSession`.
///
/// `codec` translates v1 traffic for this session; it is unused (fresh)
/// on v2 connections. `subscribers` is the hand-rolled unbounded
/// broadcast: every registered `Sender` receives every event.
pub struct SessionCell {
    /// The session id.
    pub(crate) id: SessionId,
    /// Folded session state.
    pub(crate) reducer: Mutex<SessionReducer>,
    /// v1 codec state for this session's traffic.
    pub(crate) codec: Mutex<V1Codec>,
    /// Live event subscribers.
    subscribers: Mutex<Vec<async_channel::Sender<SessionEvent>>>,
}

impl fmt::Debug for SessionCell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionCell")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl SessionCell {
    fn new(id: SessionId) -> Self {
        Self {
            id,
            reducer: Mutex::new(SessionReducer::new()),
            codec: Mutex::new(V1Codec::new()),
            subscribers: Mutex::new(Vec::new()),
        }
    }

    /// Register a new event subscriber (unbounded broadcast channel).
    pub(crate) fn subscribe(&self) -> async_channel::Receiver<SessionEvent> {
        let (tx, rx) = async_channel::unbounded();
        self.subscribers.lock().expect("subs poisoned").push(tx);
        rx
    }

    /// Fold `update` into the reducer and broadcast the emitted events.
    pub(crate) fn apply(&self, update: &SessionUpdate) {
        let events = self.reducer.lock().expect("reducer poisoned").apply(update);
        self.broadcast(&events);
    }

    /// Broadcast pre-computed events (for synthesized turns/results).
    pub(crate) fn broadcast(&self, events: &[SessionEvent]) {
        if events.is_empty() {
            return;
        }
        self.subscribers
            .lock()
            .expect("subs poisoned")
            .retain(|tx| events.iter().all(|e| tx.try_send(e.clone()).is_ok()) || !tx.is_closed());
    }

    /// The current snapshot.
    pub(crate) fn state(&self) -> SessionState {
        self.reducer
            .lock()
            .expect("reducer poisoned")
            .state()
            .clone()
    }

    /// Mark the session as replaying (v2 `replay_from: start`).
    pub(crate) fn set_replaying(&self) {
        self.reducer
            .lock()
            .expect("reducer poisoned")
            .begin_replay();
    }

    /// Mark the session replay complete and emit the event (v1 `load`,
    /// where no `replay_complete` update arrives).
    pub(crate) fn mark_replay_complete(&self) {
        self.apply(&SessionUpdate::ReplayComplete(
            crate::domain::ReplayComplete {
                meta: None,
                extra: BTreeMap::new(),
            },
        ));
    }

    /// Drop all subscribers — receivers end (`None`) so turn streams and
    /// `finish()` terminate instead of hanging on a dead connection.
    fn close(&self) {
        self.subscribers.lock().expect("subs poisoned").clear();
    }
}

/// `SessionId -> Arc<SessionCell>`. Cheap to clone; shared between the
/// `Client`, the `WireLane`, and the `WireHooks` demux.
#[derive(Clone, Default)]
pub struct SessionRegistry {
    map: Arc<Mutex<HashMap<SessionId, Arc<SessionCell>>>>,
}

impl fmt::Debug for SessionRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionRegistry").finish_non_exhaustive()
    }
}

impl SessionRegistry {
    /// An empty registry.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The cell for `id`, creating it on first sight (lazy demux).
    pub(crate) fn get_or_create(&self, id: SessionId) -> Arc<SessionCell> {
        self.map
            .lock()
            .expect("sessions poisoned")
            .entry(id.clone())
            .or_insert_with(|| Arc::new(SessionCell::new(id)))
            .clone()
    }

    /// Drop `id`'s cell (session closed).
    pub(crate) fn remove(&self, id: &SessionId) {
        let cell = self.map.lock().expect("sessions poisoned").remove(id);
        if let Some(cell) = cell {
            cell.close();
        }
    }

    /// Demux entry point from [`AgentHooks`](super::handler::AnyAgentHooks):
    /// fold the notification into the target session's cell.
    pub(crate) fn apply(&self, notification: &SessionNotification) {
        self.get_or_create(notification.session_id.clone())
            .apply(&notification.update);
    }

    /// End every subscriber stream (connection closed).
    pub(crate) fn close_all(&self) {
        for cell in self.map.lock().expect("sessions poisoned").values() {
            cell.close();
        }
    }
}

/// A live ACP connection to an agent.
///
/// Created by [`ClientBuilder::connect`](super::ClientBuilder::connect)
/// (and friends) — it already holds the negotiated [`InitResult`]. Cheap
/// to clone; all clones share the lane and session registry.
#[derive(Clone)]
pub struct Client {
    inner: Arc<ClientInner>,
}

pub struct ClientInner {
    /// The client→agent lane (wire or direct).
    pub(crate) lane: AnyAgentLane,
    /// Negotiated init result.
    pub(crate) init: InitResult,
    /// The capabilities this client advertised.
    pub(crate) capabilities: ClientCapabilities,
    /// This client's `Implementation`.
    pub(crate) info: Implementation,
    /// Live sessions.
    pub(crate) sessions: SessionRegistry,
}

impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client")
            .field("protocol_version", &self.protocol_version())
            .field("agent", &self.inner.init.info)
            .finish_non_exhaustive()
    }
}

impl Client {
    pub(crate) fn new(
        lane: AnyAgentLane,
        init: InitResult,
        capabilities: ClientCapabilities,
        info: Implementation,
        sessions: SessionRegistry,
    ) -> Self {
        Self {
            inner: Arc::new(ClientInner {
                lane,
                init,
                capabilities,
                info,
                sessions,
            }),
        }
    }

    /// A [`ClientBuilder`](crate::client::ClientBuilder) for
    /// `implementation` (`clientInfo.name/version`).
    #[must_use]
    pub const fn builder(implementation: Implementation) -> super::builder::ClientBuilder {
        super::builder::ClientBuilder::new(implementation)
    }

    /// The negotiated ACP protocol version (`1` or `2`).
    #[must_use]
    pub fn protocol_version(&self) -> u16 {
        self.inner.lane.protocol_version()
    }

    /// What the agent declared in `initialize`.
    #[must_use]
    pub fn agent_capabilities(&self) -> &crate::domain::AgentCapabilities {
        &self.inner.init.capabilities
    }

    /// The capabilities this client advertised at `initialize`.
    #[must_use]
    pub fn capabilities(&self) -> &ClientCapabilities {
        &self.inner.capabilities
    }

    /// The agent's `Implementation`, if it sent one.
    #[must_use]
    pub fn agent_info(&self) -> Option<&Implementation> {
        self.inner.init.info.as_ref()
    }

    /// Authentication methods the agent offers.
    #[must_use]
    pub fn auth_methods(&self) -> &[AuthMethod] {
        &self.inner.init.auth_methods
    }

    /// This client's `Implementation` (as advertised in `initialize`).
    #[must_use]
    pub fn implementation(&self) -> &Implementation {
        &self.inner.info
    }

    /// The raw negotiated init result.
    #[must_use]
    pub(crate) fn init(&self) -> &InitResult {
        &self.inner.init
    }

    /// The lane (crate-internal: sessions and builders use it).
    pub(crate) fn lane(&self) -> &AnyAgentLane {
        &self.inner.lane
    }

    /// The session registry (crate-internal).
    pub(crate) fn registry(&self) -> &SessionRegistry {
        &self.inner.sessions
    }

    /// Whether the negotiated protocol is v2.
    pub(crate) fn is_v2(&self) -> bool {
        self.protocol_version() >= 2
    }

    /// Start a new session rooted at `cwd`.
    #[must_use]
    pub fn new_session(&self, cwd: impl Into<PathBuf>) -> SessionBuilder {
        SessionBuilder {
            client: self.clone(),
            setup: SessionSetupParams {
                cwd: cwd.into(),
                ..SessionSetupParams::default()
            },
        }
    }

    /// Re-attach to a previously created session.
    ///
    /// The builder defaults to [`ResumeBuilder::replay_from_start`].
    /// `cwd` is required because v1 `session/load`/`session/resume` echo
    /// it back to the agent.
    #[must_use]
    pub fn resume_session(&self, id: SessionId, cwd: impl Into<PathBuf>) -> ResumeBuilder {
        ResumeBuilder {
            client: self.clone(),
            params: crate::domain::ResumeParams {
                setup: SessionSetupParams {
                    cwd: cwd.into(),
                    ..SessionSetupParams::default()
                },
                session_id: id,
                replay_from: Some(Open::Known(ReplayCursor::Start(ReplayStart { meta: None }))),
            },
        }
    }

    /// Fork a session: `session/fork` with setup fields.
    ///
    /// # Errors
    ///
    /// Propagates transport, rejection, and protocol errors.
    pub async fn fork_session(
        &self,
        id: &SessionId,
        cwd: impl Into<PathBuf>,
    ) -> Result<ClientSession, ClientError> {
        let params = crate::domain::ResumeParams {
            setup: SessionSetupParams {
                cwd: cwd.into(),
                ..SessionSetupParams::default()
            },
            session_id: id.clone(),
            replay_from: None,
        };
        let result = self.inner.lane.fork_session(params).await?;
        Ok(self.adopt(result))
    }

    /// List sessions the agent knows about.
    ///
    /// # Errors
    ///
    /// `session/list` on v1 requires the `list` session capability.
    pub async fn sessions(&self, cursor: Option<String>) -> Result<SessionPage, ClientError> {
        if !self.is_v2() && !super::builder::session_cap(&self.inner.init, "list") {
            return Err(ClientError::Unsupported(
                "agent does not advertise session list support".to_string(),
            ));
        }
        self.inner
            .lane
            .list_sessions(SessionListParams {
                cwd: None,
                cursor,
                meta: None,
                extra: BTreeMap::new(),
            })
            .await
    }

    /// Delete a session from the agent's history.
    ///
    /// # Errors
    ///
    /// `session/delete` on v1 requires the `delete` session capability.
    pub async fn delete_session(&self, id: &SessionId) -> Result<(), ClientError> {
        if !self.is_v2() && !super::builder::session_cap(&self.inner.init, "delete") {
            return Err(ClientError::Unsupported(
                "agent does not advertise session delete support".to_string(),
            ));
        }
        self.inner.lane.delete_session(id.clone()).await
    }

    /// `auth/login` (v2) / `authenticate` (v1) with `method`.
    ///
    /// # Errors
    ///
    /// Propagates transport and rejection errors.
    pub async fn authenticate(&self, method: impl Into<AuthMethodId>) -> Result<(), ClientError> {
        self.inner.lane.authenticate(method.into()).await
    }

    /// `auth/logout` (v2) / `logout` (v1).
    ///
    /// # Errors
    ///
    /// Propagates transport and rejection errors.
    pub async fn logout(&self, method: Option<AuthMethodId>) -> Result<(), ClientError> {
        self.inner.lane.logout(method).await
    }

    /// A `_`-prefixed extension request.
    ///
    /// # Errors
    ///
    /// Propagates transport, rejection, and serialization errors.
    pub fn ext_request<'a>(
        &'a self,
        method: &'a ExtMethod,
        params: &impl Serialize,
    ) -> impl Future<Output = Result<Value, ClientError>> + Send + 'a {
        // Serialize eagerly so the `!Sync` params reference never crosses
        // an await — keeps the returned future `Send`.
        let params = serde_json::to_value(params)
            .map_err(|e| ClientError::Protocol(format!("ext params: {e}")));
        async move {
            let params = params?;
            self.inner
                .lane
                .ext_request(method.clone(), (!params.is_null()).then_some(params))
                .await
        }
    }

    /// A `_`-prefixed extension notification.
    ///
    /// # Errors
    ///
    /// Propagates transport and serialization errors.
    pub fn ext_notify<'a>(
        &'a self,
        method: &'a ExtMethod,
        params: &impl Serialize,
    ) -> impl Future<Output = Result<(), ClientError>> + Send + 'a {
        // See `ext_request`: serialize eagerly for `Send`.
        let params = serde_json::to_value(params)
            .map_err(|e| ClientError::Protocol(format!("ext params: {e}")));
        async move {
            let params = params?;
            self.inner
                .lane
                .ext_notify(method.clone(), (!params.is_null()).then_some(params))
                .await
        }
    }

    /// Ask the peer to close the connection (queued outbound flushes
    /// first; the [`Connection`](super::Connection) then resolves).
    pub async fn close(&self) {
        self.inner.lane.shutdown().await;
    }

    /// Adopt a session the lane just created: find-or-create its cell,
    /// seed the reducer with the setup result's config options and
    /// commands, and build the [`ClientSession`].
    pub(crate) fn adopt(&self, result: crate::domain::SetupResult) -> ClientSession {
        let cell = self.inner.sessions.get_or_create(result.session_id.clone());
        if !result.config_options.is_empty() {
            cell.apply(&SessionUpdate::ConfigOptionUpdate(
                crate::domain::ConfigOptionUpdate {
                    config_options: result.config_options.clone(),
                    meta: None,
                    extra: BTreeMap::new(),
                },
            ));
        }
        if !result.available_commands.is_empty() {
            cell.apply(&SessionUpdate::AvailableCommandsUpdate(
                crate::domain::AvailableCommandsUpdate {
                    available_commands: result.available_commands.clone(),
                    meta: None,
                    extra: BTreeMap::new(),
                },
            ));
        }
        ClientSession::new(self.clone(), cell, result.extra)
    }
}

/// Builder for `session/new`.
#[derive(Debug)]
pub struct SessionBuilder {
    client: Client,
    setup: SessionSetupParams,
}

impl SessionBuilder {
    /// Add one MCP server (`mcp_servers`).
    #[must_use]
    pub fn mcp_server(mut self, server: McpServer) -> Self {
        self.setup.mcp_servers.push(server);
        self
    }

    /// Add several MCP servers.
    #[must_use]
    pub fn mcp_servers(mut self, servers: impl IntoIterator<Item = McpServer>) -> Self {
        self.setup.mcp_servers.extend(servers);
        self
    }

    /// Additional workspace roots.
    #[must_use]
    pub fn additional_directories(mut self, dirs: impl IntoIterator<Item = PathBuf>) -> Self {
        self.setup.additional_directories.extend(dirs);
        self
    }

    /// `_meta` for the setup params.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.setup.meta = Some(meta);
        self
    }

    /// `session/new` → [`ClientSession`].
    ///
    /// # Errors
    ///
    /// Propagates transport, rejection, and protocol errors.
    pub async fn open(self) -> Result<ClientSession, ClientError> {
        let result = self.client.lane().new_session(self.setup).await?;
        Ok(self.client.adopt(result))
    }
}

/// Builder for `session/resume` / `session/load`.
///
/// Defaults to *replay from start* (`session/load` on v1,
/// `replayFrom: {"type":"start"}` on v2). [`ResumeBuilder::no_replay`]
/// attaches without replay (`session/resume` on both versions).
#[derive(Debug)]
pub struct ResumeBuilder {
    client: Client,
    params: crate::domain::ResumeParams,
}

impl ResumeBuilder {
    /// Replay the transcript from the beginning (the default).
    ///
    /// On v1 requires `agentCapabilities.loadSession`; on v2 `open()` waits
    /// for the `replay_complete` update before returning.
    #[must_use]
    pub fn replay_from_start(mut self) -> Self {
        self.params.replay_from =
            Some(Open::Known(ReplayCursor::Start(ReplayStart { meta: None })));
        self
    }

    /// Attach without replay (`session/resume`).
    ///
    /// On v1 requires `sessionCapabilities.resume`.
    #[must_use]
    pub fn no_replay(mut self) -> Self {
        self.params.replay_from = None;
        self
    }

    /// Add one MCP server.
    #[must_use]
    pub fn mcp_server(mut self, server: McpServer) -> Self {
        self.params.setup.mcp_servers.push(server);
        self
    }

    /// Additional workspace roots (must be re-sent on v1).
    #[must_use]
    pub fn additional_directories(mut self, dirs: impl IntoIterator<Item = PathBuf>) -> Self {
        self.params.setup.additional_directories.extend(dirs);
        self
    }

    /// `session/resume` (or v1 `session/load`) → [`ClientSession`].
    ///
    /// # Errors
    ///
    /// Capability-gated on v1 (`loadSession` / `sessionCapabilities.resume`
    /// respectively), plus transport, rejection, and protocol errors. A
    /// v2 replay that ends without `replay_complete` is a protocol error.
    pub async fn open(self) -> Result<ClientSession, ClientError> {
        let replay = self.params.replay_from.is_some();
        let v2 = self.client.is_v2();
        // Capability gates only exist on v1.
        if !v2 {
            let cap = if replay { "loadSession" } else { "resume" };
            let advertised = if replay {
                super::builder::session_cap(self.client.init(), "loadSession")
            } else {
                super::builder::session_cap(self.client.init(), "resume")
            };
            if !advertised {
                return Err(ClientError::Unsupported(format!(
                    "agent does not advertise `{cap}`"
                )));
            }
        }
        let session_id = self.params.session_id.clone();
        let cell = self.client.registry().get_or_create(session_id);
        if replay && v2 {
            cell.set_replaying();
        }
        // Subscribe before the request so no replay update is missed.
        let rx = cell.subscribe();
        let result = self.client.lane().resume_session(self.params).await?;
        let session = self.client.adopt(result);
        if replay && v2 {
            // Wait for `replay_complete` before returning the session.
            while let Ok(event) = rx.recv().await {
                if matches!(event, SessionEvent::ReplayComplete) {
                    break;
                }
            }
            if !cell.state().replay_done {
                return Err(ClientError::Protocol(
                    "session/resume replay ended without `replay_complete`".to_string(),
                ));
            }
        } else if replay {
            // v1 `session/load` replays before the result; mark complete.
            cell.mark_replay_complete();
        } else {
            drop(rx);
        }
        Ok(session)
    }
}
