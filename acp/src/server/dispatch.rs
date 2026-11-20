//! [`ServerCore`] — the transport-agnostic dispatcher — and
//! [`WireAdapter`], the [`rpc::Handler`] that serves it over JSON-RPC.
//!
//! `ServerCore` owns the session table, negotiation state, the FIFO
//! prompt scheduler, cancellation, and idempotency. [`WireAdapter`]
//! decodes wire params (through the per-session [`V1Codec`] on v1),
//! calls `ServerCore`, and encodes results. `client::in_process` calls
//! the same domain-level API through the direct lane.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, Weak};

use futures_lite::StreamExt;
use futures_lite::future;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tracing::{debug, warn};

use crate::domain::{
    AuthMethodId, ClientCapabilities, ElicitComplete, ForegroundState, Implementation, InitResult,
    MessageId, Open, PromptFinished, PromptParams, ResumeParams, SessionEndParams, SessionId,
    SessionInfo, SessionListParams, SessionNotification, SessionPage, SessionSetupParams,
    SessionUpdate, SetConfigParams, SetupResult, StopReason, TerminalId, ToolCallId,
    open::ExtMethod,
};
use crate::rpc::{
    Handler, Notification, Peer, PeerError, Request, RequestContext, Response, RpcError,
};
use crate::wire::codec::V1Codec;
use crate::wire::{v1, v2};

use super::context::{AgentHooks, AnyAgentHooks, ProtocolVersion, SessionContext};
use super::error::PromptError;
use super::host::{
    AnySessionHost, AuthConfig, AuthLoginRequest, AuthRequest, ResumeRequest, SessionHost,
};
use super::prompt::{PendingPrompt, PromptTicket};
use super::table::{QueuedItem, SessionEntry, SessionShared, TurnState};

/// Decoded `initialize` parameters, normalized from either wire shape.
#[derive(Debug, Clone)]
pub struct InitializeParams {
    /// `protocolVersion` requested by the client.
    pub protocol_version: u16,
    /// Client capabilities (domain form).
    pub capabilities: ClientCapabilities,
    /// Client implementation info, when provided.
    pub info: Option<Implementation>,
    /// Extension metadata.
    pub meta: Option<Value>,
}

/// Per-connection mutable state.
#[derive(Debug, Default)]
struct ConnState {
    /// Negotiated version once `initialize` completed.
    protocol: Option<ProtocolVersion>,
    /// The client's capabilities.
    client_caps: ClientCapabilities,
    /// The client's implementation.
    client_info: Option<Implementation>,
    /// `initialize` completed successfully.
    initialized: bool,
    /// Any request (including `initialize`) was already seen — used for
    /// the "initialize must be first" gate.
    seen_request: bool,
}

/// An extension-method handler (`_`-prefixed methods).
pub type ExtHandler = Arc<
    dyn Fn(
            ExtMethod,
            Option<Value>,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, RpcError>> + Send>>
        + Send
        + Sync,
>;

/// A fallback handler for unknown non-extension methods.
pub type UnknownHandler = Arc<
    dyn Fn(
            &str,
            Option<Value>,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, RpcError>> + Send>>
        + Send
        + Sync,
>;

/// The transport-independent ACP server core.
///
/// One `ServerCore` serves one connection: it tracks negotiated
/// protocol version, client capabilities, the live session table, and
/// per-session prompt scheduling. The same domain-level API is consumed
/// by [`WireAdapter`] (JSON-RPC) and by `client::in_process` (the
/// direct lane, always v2 semantics).
///
/// Prompt turns do not run inside `prompt()` — they are driven by
/// [`drive`](Self::drive), which `Server::serve` polls alongside the
/// transport; an in-process client must poll it for turns to progress.
pub struct ServerCore {
    host: AnySessionHost,
    info: Implementation,
    sessions: Mutex<HashMap<SessionId, Arc<SessionEntry>>>,
    state: Mutex<ConnState>,
    auth: Option<Arc<AuthConfig>>,
    ext: Option<ExtHandler>,
    unknown: Option<UnknownHandler>,
    wake_tx: async_channel::Sender<SessionId>,
    wake_rx: async_channel::Receiver<SessionId>,
}

impl std::fmt::Debug for ServerCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerCore")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl ServerCore {
    /// A core serving `host` under implementation `info`.
    pub fn new<H: SessionHost>(host: H, info: Implementation) -> Self {
        let (wake_tx, wake_rx) = async_channel::unbounded();
        Self {
            host: AnySessionHost::new(host),
            info,
            sessions: Mutex::new(HashMap::new()),
            state: Mutex::new(ConnState::default()),
            auth: None,
            ext: None,
            unknown: None,
            wake_tx,
            wake_rx,
        }
    }

    /// A core serving an already-erased host.
    #[must_use]
    pub fn from_parts(host: AnySessionHost, info: Implementation) -> Self {
        let (wake_tx, wake_rx) = async_channel::unbounded();
        Self {
            host,
            info,
            sessions: Mutex::new(HashMap::new()),
            state: Mutex::new(ConnState::default()),
            auth: None,
            ext: None,
            unknown: None,
            wake_tx,
            wake_rx,
        }
    }

    /// Install auth configuration (`auth/login`/`auth/logout`).
    #[must_use]
    pub fn with_auth(mut self, auth: AuthConfig) -> Self {
        self.auth = Some(Arc::new(auth));
        self
    }

    /// Install the `_`-method extension handler.
    #[must_use]
    pub fn with_ext_handler(mut self, handler: ExtHandler) -> Self {
        self.ext = Some(handler);
        self
    }

    /// Install the fallback unknown-method handler.
    #[must_use]
    pub fn with_unknown_handler(mut self, handler: UnknownHandler) -> Self {
        self.unknown = Some(handler);
        self
    }

    /// The negotiated protocol version (`None` before `initialize`;
    /// the direct lane pins it to v2 explicitly).
    ///
    /// # Panics
    ///
    /// Panics if the connection-state mutex is poisoned.
    #[must_use]
    pub fn protocol(&self) -> Option<ProtocolVersion> {
        self.state.lock().expect("state").protocol
    }

    /// The client's capabilities (`Default` before `initialize`).
    ///
    /// # Panics
    ///
    /// Panics if the connection-state mutex is poisoned.
    #[must_use]
    pub fn client_caps(&self) -> ClientCapabilities {
        self.state.lock().expect("state").client_caps.clone()
    }

    /// The capabilities advertised at `initialize`.
    #[must_use]
    pub fn capabilities(&self) -> crate::domain::AgentCapabilities {
        self.host.capabilities()
    }

    /// Whether `initialize` has completed.
    ///
    /// # Panics
    ///
    /// Panics if the connection-state mutex is poisoned.
    #[must_use]
    pub fn initialized(&self) -> bool {
        self.state.lock().expect("state").initialized
    }

    /// Enforce the "initialize must be the first request" rule.
    ///
    /// Returns `-32600` for a non-`initialize` request before
    /// initialization or a second `initialize`.
    pub(crate) fn gate_request(&self, method: &str) -> Result<(), RpcError> {
        {
            let mut st = self.state.lock().expect("state");
            let seen = st.seen_request;
            st.seen_request = true;
            if method == "initialize" {
                if seen {
                    Err(RpcError::invalid_request())
                } else {
                    Ok(())
                }
            } else if st.initialized {
                Ok(())
            } else {
                Err(RpcError::invalid_request())
            }
        }
    }

    /// Version-aware unknown-session error: `-32002` (with
    /// `data.sessionId`) on v2, `-32602` on v1.
    fn unknown_session(&self, session_id: &SessionId) -> RpcError {
        match self.protocol() {
            Some(ProtocolVersion::V1) => {
                RpcError::invalid_params(format!("unknown session {session_id}"))
            }
            _ => RpcError::session_not_found(session_id.as_str()),
        }
    }

    /// Look up a live session entry.
    pub(crate) fn entry(&self, session_id: &SessionId) -> Option<Arc<SessionEntry>> {
        self.sessions
            .lock()
            .expect("sessions")
            .get(session_id)
            .cloned()
    }

    /// Negotiate the protocol version and record client capabilities.
    ///
    /// Selection: request v1 → v1; request ≥ v2 → v2; request 0 →
    /// `-32022` with `{requested, supported: [1,2]}`.
    ///
    /// # Errors
    ///
    /// `-32600` if not the first request; `-32022` for unsupported
    /// versions.
    ///
    /// # Panics
    ///
    /// Panics if the connection-state mutex is poisoned.
    pub fn initialize(
        &self,
        params: InitializeParams,
    ) -> impl Future<Output = Result<InitResult, RpcError>> + Send {
        let protocol = match params.protocol_version {
            0 => {
                return future::ready(Err(RpcError::protocol_version_unsupported(0, &[1, 2])));
            }
            1 => ProtocolVersion::V1,
            _ => ProtocolVersion::V2,
        };
        {
            let mut st = self.state.lock().expect("state");
            st.protocol = Some(protocol);
            st.client_caps = params.capabilities;
            st.client_info = params.info;
            st.initialized = true;
        }
        future::ready(Ok(InitResult {
            protocol_version: protocol.number(),
            capabilities: self.host.capabilities(),
            info: Some(self.info.clone()),
            auth_methods: self
                .auth
                .as_ref()
                .map_or_else(Vec::new, |a| a.methods.clone()),
            meta: params.meta,
        }))
    }

    /// Build the [`SessionContext`] for a new session entry.
    fn make_context(&self, session_id: SessionId, hooks: AnyAgentHooks) -> SessionContext {
        let st = self.state.lock().expect("state");
        SessionContext::new(
            session_id,
            hooks,
            st.protocol.unwrap_or(ProtocolVersion::V2),
            st.client_caps.clone(),
            st.client_info.clone(),
            Arc::new(SessionShared::new()),
        )
    }

    /// Finish a session registration: read its declared surface and
    /// store the entry.
    fn register(&self, cx: &SessionContext, session: super::session::AnySession) -> SetupResult {
        let result = SetupResult {
            session_id: cx.session_id().clone(),
            config_options: session.config_options(),
            available_commands: session.available_commands(),
            meta: None,
            extra: BTreeMap::new(),
        };
        let entry = Arc::new(SessionEntry {
            id: cx.session_id().clone(),
            session,
            shared: cx.shared().clone(),
            cx: cx.clone(),
        });
        self.sessions
            .lock()
            .expect("sessions")
            .insert(cx.session_id().clone(), entry);
        result
    }

    /// `session/new`.
    ///
    /// # Errors
    ///
    /// Propagates the host's setup failure as an RPC error.
    pub async fn new_session(
        &self,
        params: SessionSetupParams,
        hooks: AnyAgentHooks,
    ) -> Result<SetupResult, RpcError> {
        let cx = self.make_context(SessionId::new(), hooks);
        let session = self
            .host
            .new_session(params, &cx)
            .await
            .map_err(RpcError::from)?;
        Ok(self.register(&cx, session))
    }

    /// `session/resume` (and v1 `session/load`/`session/fork` behind the
    /// codec — `params.replay_from` selects the replay window).
    ///
    /// The host fills the [`ReplaySink`](crate::server::ReplaySink);
    /// drained updates are emitted
    /// before this returns, followed by `replay_complete` on v2.
    ///
    /// # Errors
    ///
    /// `method not found` when the host does not implement resume;
    /// `invalid params` when the session is already live.
    pub async fn resume_session(
        &self,
        params: ResumeParams,
        hooks: AnyAgentHooks,
    ) -> Result<SetupResult, RpcError> {
        self.resume_like(params, hooks, false).await
    }

    /// `session/fork` — same mechanics as resume under a fresh id.
    ///
    /// # Errors
    ///
    /// `method not found` when the host does not implement forking.
    pub async fn fork_session(
        &self,
        params: ResumeParams,
        hooks: AnyAgentHooks,
    ) -> Result<SetupResult, RpcError> {
        self.resume_like(params, hooks, true).await
    }

    async fn resume_like(
        &self,
        params: ResumeParams,
        hooks: AnyAgentHooks,
        fork: bool,
    ) -> Result<SetupResult, RpcError> {
        let session_id = if fork {
            SessionId::new()
        } else {
            params.session_id.clone()
        };
        if self.entry(&session_id).is_some() {
            return Err(RpcError::invalid_params(format!(
                "session {session_id} is already live"
            )));
        }
        let cx = self.make_context(session_id, hooks);
        let (replay, drained) = super::host::ReplaySink::channel();
        let req = ResumeRequest { params, replay };
        let session = if fork {
            self.host
                .fork_session(req, &cx)
                .await
                .map_err(RpcError::from)?
        } else {
            self.host
                .resume_session(req, &cx)
                .await
                .map_err(RpcError::from)?
        };
        // The sink is consumed by the request; once the host returns it
        // is dropped and the drain terminates.
        while let Ok(update) = drained.recv().await {
            let _ = cx.update(update).await;
        }
        cx.emit_replay_complete().await;
        Ok(self.register(&cx, session))
    }

    /// `session/list`.
    ///
    /// # Errors
    ///
    /// Propagates the host's error.
    pub async fn list_sessions(&self, params: SessionListParams) -> Result<SessionPage, RpcError> {
        self.host.list_sessions(params).await
    }

    /// `session/delete` — removes stored session state on the host.
    /// Does not attach to a live session; live-close is
    /// [`close_session`](Self::close_session).
    ///
    /// # Errors
    ///
    /// Propagates the host's error.
    pub async fn delete_session(&self, session_id: &SessionId) -> Result<(), RpcError> {
        self.host.delete_session(session_id).await
    }

    /// `session/close` — cancels the active turn, waits for its final
    /// updates, closes the session, and drops the entry.
    ///
    /// # Errors
    ///
    /// `session not found` (version-aware) for unknown or closed ids.
    ///
    /// # Panics
    ///
    /// Panics if the session-table mutex is poisoned.
    pub async fn close_session(&self, session_id: &SessionId) -> Result<(), RpcError> {
        let entry = self
            .sessions
            .lock()
            .expect("sessions")
            .remove(session_id)
            .ok_or_else(|| self.unknown_session(session_id))?;
        self.teardown(&entry, true).await;
        Ok(())
    }

    /// Shared teardown: mark closed, cancel the turn + queue, wait for
    /// the turn's final updates, then `session.close()`.
    async fn teardown(&self, entry: &Arc<SessionEntry>, wait_turn: bool) {
        let turn = entry.shared.cancel_all(&RpcError::request_cancelled());
        if wait_turn && let Some(turn) = turn {
            turn.ended().await;
        }
        entry.session.close().await;
    }

    /// `session/prompt` — insert a prompt into the session's FIFO queue.
    ///
    /// Returns the [`PromptTicket`] immediately: v2 callers respond
    /// `{messageId}` at once while the prompt is driven asynchronously
    /// by [`drive`](Self::drive); v1 callers additionally await
    /// [`PromptTicket::finish`] for the turn's `stopReason`.
    ///
    /// Idempotent: a repeated `idempotencyKey` yields the original
    /// `messageId` without re-inserting.
    ///
    /// # Errors
    ///
    /// `session not found` (version-aware) for unknown ids;
    /// `invalid params` for a malformed idempotency key;
    /// `request cancelled` when the session is closing.
    ///
    /// # Panics
    ///
    /// Panics if any of the session bookkeeping mutexes is poisoned.
    pub fn prompt(
        &self,
        params: PromptParams,
    ) -> impl Future<Output = Result<PromptTicket, RpcError>> + Send {
        let entry = self
            .entry(&params.session_id)
            .ok_or_else(|| self.unknown_session(&params.session_id));
        future::ready((|| {
            let entry = entry?;
            if let Some(key) = &params.idempotency_key {
                if key.is_empty() || key.len() > 255 {
                    return Err(RpcError::invalid_params("malformed idempotency key"));
                }
                if let Some(item) = entry.shared.idempotent.lock().expect("idempotent").get(key) {
                    let (tx, rx) = async_channel::bounded(1);
                    item.add_finisher(tx);
                    return Ok(PromptTicket::new(item.message_id.clone(), rx));
                }
            }
            if entry.shared.closed.load(Ordering::SeqCst) {
                return Err(RpcError::request_cancelled());
            }
            let (tx, rx) = async_channel::bounded(1);
            let item = Arc::new(QueuedItem::new(params, MessageId::new()));
            item.add_finisher(tx);
            if let Some(key) = item.params.idempotency_key.clone() {
                entry
                    .shared
                    .idempotent
                    .lock()
                    .expect("idempotent")
                    .insert(key, item.clone());
            }
            entry
                .shared
                .by_message
                .lock()
                .expect("by_message")
                .insert(item.message_id.clone(), item.clone());
            let message_id = item.message_id.clone();
            entry.shared.queue.lock().expect("queue").push_back(item);
            let _ = self.wake_tx.try_send(entry.id.clone());
            Ok(PromptTicket::new(message_id, rx))
        })())
    }

    /// `session/cancel` — cancel the running turn and fail queued
    /// prompts with `-32800`. A notification: no response, unknown
    /// sessions are ignored.
    pub fn cancel_session(&self, session_id: &SessionId) {
        if let Some(entry) = self.entry(session_id) {
            let _ = entry.shared.cancel_turn();
        }
    }

    /// `session/set_config_option` — call
    /// [`Session::set_config`](super::Session::set_config) and emit the
    /// resulting `config_option_update` (which the v1 codec also turns
    /// into `current_mode_update` for the mode option).
    ///
    /// # Errors
    ///
    /// `session not found` for unknown ids; propagates the session's
    /// error.
    pub async fn set_config(
        &self,
        params: SetConfigParams,
    ) -> Result<Vec<crate::domain::ConfigOption>, RpcError> {
        let entry = self
            .entry(&params.session_id)
            .ok_or_else(|| self.unknown_session(&params.session_id))?;
        let options = entry
            .session
            .set_config(&params.config_id, params.value, &entry.cx)
            .await?;
        let _ = entry.cx.set_config_options(options.clone()).await;
        Ok(options)
    }

    /// `session/tool_call_cancel` — resolve a tool call's
    /// [`ToolCallHandle::cancelled`](super::ToolCallHandle::cancelled)
    /// waiter.
    ///
    /// # Panics
    ///
    /// Panics if the session's cancel-waiter mutex is poisoned.
    pub fn tool_call_cancel(&self, session_id: &SessionId, tool_call_id: &ToolCallId) {
        if let Some(entry) = self.entry(session_id)
            && let Some(tx) = entry
                .shared
                .tool_cancels
                .lock()
                .expect("tool_cancels")
                .remove(tool_call_id)
        {
            let _ = tx.try_send(());
        }
    }

    /// `session/terminal_cancel` — resolve a terminal's
    /// [`TerminalHandle::cancelled`](super::TerminalHandle::cancelled)
    /// waiter.
    ///
    /// # Panics
    ///
    /// Panics if the session's cancel-waiter mutex is poisoned.
    pub fn terminal_cancel(&self, session_id: &SessionId, terminal_id: &TerminalId) {
        if let Some(entry) = self.entry(session_id)
            && let Some(tx) = entry
                .shared
                .terminal_cancels
                .lock()
                .expect("terminal_cancels")
                .remove(terminal_id)
        {
            let _ = tx.try_send(());
        }
    }

    /// `elicitation/complete` — resolve a pending url-mode elicitation
    /// waiter registered by
    /// [`SessionContext::elicit_wait_complete`].
    ///
    /// # Panics
    ///
    /// Panics if the session's elicit-waiter mutex is poisoned.
    pub fn elicit_complete(&self, session_id: &SessionId, complete: &ElicitComplete) {
        if let Some(entry) = self.entry(session_id)
            && let Some(tx) = entry
                .shared
                .elicits
                .lock()
                .expect("elicits")
                .remove(&complete.elicitation_id)
        {
            let _ = tx.try_send(complete.clone());
        }
    }

    /// `auth/login` (`authenticate` on v1).
    ///
    /// # Errors
    ///
    /// `method not found` when no auth methods are configured.
    pub async fn auth_login(&self, request: AuthLoginRequest) -> Result<(), RpcError> {
        let auth = self
            .auth
            .as_ref()
            .filter(|a| !a.methods.is_empty())
            .ok_or_else(|| RpcError::method_not_found("auth/login"))?;
        (auth.handler)(AuthRequest::Login(request)).await
    }

    /// `auth/logout` (`logout` on v1).
    ///
    /// # Errors
    ///
    /// `method not found` when no auth methods are configured.
    pub async fn auth_logout(&self, method_id: Option<AuthMethodId>) -> Result<(), RpcError> {
        let auth = self
            .auth
            .as_ref()
            .filter(|a| !a.methods.is_empty())
            .ok_or_else(|| RpcError::method_not_found("auth/logout"))?;
        (auth.handler)(AuthRequest::Logout(method_id)).await
    }

    /// Dispatch a `_`-prefixed extension request.
    ///
    /// # Errors
    ///
    /// `method not found` when no extension handler is installed.
    pub async fn ext_request(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, RpcError> {
        let method =
            ExtMethod::try_new(method).map_err(|e| RpcError::invalid_params(e.to_string()))?;
        match &self.ext {
            Some(handler) => handler(method, params).await,
            None => Err(RpcError::method_not_found(method.as_str())),
        }
    }

    /// Dispatch a non-extension method no standard dispatch covered.
    ///
    /// # Errors
    ///
    /// `method not found` when no fallback handler is installed.
    pub async fn unknown_request(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, RpcError> {
        match &self.unknown {
            Some(handler) => handler(method, params).await,
            None => Err(RpcError::method_not_found(method)),
        }
    }

    /// The prompt scheduler. Poll this for turns to make progress —
    /// `Server::serve` does so alongside the transport; the in-process
    /// lane must arrange it itself.
    ///
    /// Runs until the connection ends (the returned future is dropped
    /// at EOF); never completes on its own.
    ///
    /// # Panics
    ///
    /// Panics if a session's prompt-queue mutex is poisoned.
    pub async fn drive(&self) {
        let mut active: HashSet<SessionId> = HashSet::new();
        let mut running = futures_util::stream::FuturesUnordered::new();
        loop {
            enum Event {
                Wake(SessionId),
                Done(SessionId),
            }
            let event = future::or(
                async {
                    Event::Wake(
                        self.wake_rx
                            .recv()
                            .await
                            .unwrap_or_else(|_| SessionId::from(String::new())),
                    )
                },
                async {
                    match running.next().await {
                        Some(id) => Event::Done(id),
                        None => future::pending().await,
                    }
                },
            )
            .await;
            match event {
                Event::Wake(id) => {
                    if !active.insert(id.clone()) {
                        continue;
                    }
                    match self.entry(&id) {
                        Some(entry) => {
                            running.push(Box::pin(async move {
                                run_session(&entry).await;
                                id
                            }));
                        }
                        None => {
                            active.remove(&id);
                        }
                    }
                }
                Event::Done(id) => {
                    active.remove(&id);
                    // A prompt enqueued after the queue drained but before
                    // this completion must re-kick the session.
                    if let Some(entry) = self.entry(&id)
                        && !entry.shared.queue.lock().expect("queue").is_empty()
                    {
                        let _ = self.wake_tx.try_send(id);
                    }
                }
            }
        }
    }

    /// Start the host (`SessionHost::start`), called by `Server::serve`
    /// before dispatching begins.
    pub(crate) async fn host_start(&self) {
        if let Err(e) = self.host.start().await {
            tracing::error!("session host failed to start: {e}");
        }
    }

    /// Called when the transport ends: close every session and shut the
    /// host down.
    ///
    /// # Panics
    ///
    /// Panics if the session-table mutex is poisoned.
    pub async fn on_disconnect(&self) {
        let entries: Vec<Arc<SessionEntry>> = self
            .sessions
            .lock()
            .expect("sessions")
            .drain()
            .map(|(_, e)| e)
            .collect();
        for entry in entries {
            self.teardown(&entry, false).await;
        }
        self.host.shutdown().await;
    }
}

/// Run one session's queue to completion (called from
/// [`ServerCore::drive`]).
async fn run_session(entry: &Arc<SessionEntry>) {
    loop {
        let item = entry.shared.queue.lock().expect("queue").pop_front();
        let Some(item) = item else { break };
        run_turn(entry, item).await;
        if entry.shared.closed.load(Ordering::SeqCst) {
            for item in entry.shared.queue.lock().expect("queue").drain(..) {
                item.complete(&Err(RpcError::request_cancelled()));
            }
            break;
        }
    }
}

/// Run one turn for `item`.
async fn run_turn(entry: &Arc<SessionEntry>, item: Arc<QueuedItem>) {
    let turn = Arc::new(TurnState::new());
    *entry.shared.current.lock().expect("current") = Some(turn.clone());
    let pending = PendingPrompt::driving(
        item.clone(),
        entry.shared.clone(),
        entry.cx.clone(),
        turn.clone(),
    );
    let result = entry.session.prompt(pending, &entry.cx).await;
    let (stop, error) = match result {
        Ok(stop) => (Open::Known(stop), None),
        Err(PromptError::Cancelled) => (Open::Known(StopReason::Cancelled), None),
        Err(PromptError::Failed(e)) => (
            Open::Known(StopReason::Error),
            serde_json::to_value(&e).ok(),
        ),
    };
    if turn.is_accepted() {
        entry
            .cx
            .emit_state(ForegroundState::Idle, Some(stop.clone()), error)
            .await;
    } else {
        // Rejected prompts already emitted their terminal state via
        // `PendingPrompt::reject`; a dropped prompt gets cancelled here.
        if !item.was_rejected() {
            entry
                .cx
                .emit_state(
                    ForegroundState::Idle,
                    Some(Open::Known(StopReason::Error)),
                    serde_json::to_value(RpcError::request_cancelled()).ok(),
                )
                .await;
        }
        item.complete(&Err(RpcError::request_cancelled()));
    }
    turn.finish(&Ok(PromptFinished {
        stop_reason: stop,
        usage: None,
        meta: None,
    }));
    *entry.shared.current.lock().expect("current") = None;
}

// =============================================================================
// WireAdapter
// =============================================================================

/// The [`Handler`](crate::rpc::Handler) serving [`ServerCore`] over a
/// JSON-RPC transport.
///
/// `WireAdapter` tracks the negotiated protocol version, keeps a
/// per-session [`V1Codec`], and supplies each session's
/// [`SessionContext`] with a wire `AgentHooks` lane over [`Peer`]. Build
/// it
/// through [`crate::server::Server::serve`], or `attach` a `Peer`
/// obtained from [`rpc::connection`](crate::rpc::connection).
#[derive(Clone)]
pub struct WireAdapter {
    core: Arc<ServerCore>,
    peer: Arc<Mutex<Option<Peer>>>,
    codecs: Arc<Mutex<HashMap<SessionId, V1Codec>>>,
    init_codec: Arc<Mutex<V1Codec>>,
}

impl std::fmt::Debug for WireAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WireAdapter").finish_non_exhaustive()
    }
}

impl WireAdapter {
    /// An adapter dispatching to `core`.
    #[must_use]
    pub fn new(core: Arc<ServerCore>) -> Self {
        Self {
            core,
            peer: Arc::new(Mutex::new(None)),
            codecs: Arc::new(Mutex::new(HashMap::new())),
            init_codec: Arc::new(Mutex::new(V1Codec::new())),
        }
    }

    /// Hand the adapter the [`Peer`] `rpc::connection` produced. Calls
    /// before `attach` fail the outbound lane.
    ///
    /// # Panics
    ///
    /// Panics if the peer mutex is poisoned.
    pub fn attach(&self, peer: Peer) {
        *self.peer.lock().expect("peer") = Some(peer);
    }

    /// The core this adapter serves.
    #[must_use]
    pub const fn core(&self) -> &Arc<ServerCore> {
        &self.core
    }

    fn peer(&self) -> Result<Peer, RpcError> {
        self.peer
            .lock()
            .expect("peer")
            .clone()
            .ok_or_else(|| RpcError::internal_error("wire adapter not attached"))
    }

    /// Run `f` against the per-session codec (created on first use).
    fn with_codec<R>(&self, session_id: &SessionId, f: impl FnOnce(&mut V1Codec) -> R) -> R {
        f(self
            .codecs
            .lock()
            .expect("codecs")
            .entry(session_id.clone())
            .or_default())
    }

    /// Session hooks for a session being registered.
    fn hooks(&self) -> AnyAgentHooks {
        AnyAgentHooks::new(WireHooks {
            peer: self.peer.clone(),
            codecs: self.codecs.clone(),
            core: Arc::downgrade(&self.core),
        })
    }

    /// Remove session-side wire state on close/disconnect.
    fn drop_codec(&self, session_id: &SessionId) {
        self.codecs.lock().expect("codecs").remove(session_id);
    }
}

/// Decode params or fail with `invalid params`.
fn decode<T: DeserializeOwned>(params: Option<Value>) -> Result<T, RpcError> {
    serde_json::from_value(params.unwrap_or(Value::Null))
        .map_err(|e| RpcError::invalid_params(format!("{e}")))
}

/// Encode `value` as the response payload.
fn encode<T: Serialize>(value: &T) -> Result<Value, RpcError> {
    serde_json::to_value(value).map_err(|e| RpcError::internal_error(e.to_string()))
}

impl Handler for WireAdapter {
    async fn request(&self, req: Request, _cx: RequestContext) -> Response {
        match self.dispatch(&req.method, req.params).await {
            Ok(value) => Response::success(req.id, value),
            Err(error) => Response::error(req.id, error),
        }
    }

    async fn notification(&self, n: Notification) {
        self.dispatch_notification(n).await;
    }
}

impl WireAdapter {
    /// Route an inbound request.
    async fn dispatch(&self, method: &str, params: Option<Value>) -> Result<Value, RpcError> {
        self.core.gate_request(method)?;
        let v1_mode = self.core.protocol() == Some(ProtocolVersion::V1);
        match method {
            "initialize" => self.initialize(params).await,
            "session/new" | "session/load" | "session/resume" | "session/fork" => {
                self.dispatch_setup(method, params, v1_mode).await
            }
            "session/list" | "session/delete" | "session/close" => {
                self.dispatch_admin(method, params, v1_mode).await
            }
            "session/prompt"
            | "session/set_mode"
            | "session/set_config_option"
            | "session/tool_call_cancel"
            | "session/terminal_cancel" => self.dispatch_turn(method, params, v1_mode).await,
            // ---- auth ----------------------------------------------------------------------
            "authenticate" if v1_mode => {
                let p: v1::AuthenticateParams = decode(params)?;
                self.core
                    .auth_login(AuthLoginRequest {
                        method_id: AuthMethodId::from(p.method_id),
                        meta: p.meta,
                        extra: BTreeMap::new(),
                    })
                    .await?;
                encode(&serde_json::json!({}))
            }
            "auth/login" => {
                let p: v2::AuthLoginParams = decode(params)?;
                self.core
                    .auth_login(AuthLoginRequest {
                        method_id: p.method_id,
                        meta: p.meta,
                        extra: p.extra,
                    })
                    .await?;
                encode(&serde_json::json!({}))
            }
            "logout" | "auth/logout" => {
                let method_id = params
                    .as_ref()
                    .and_then(|p| p.get("methodId"))
                    .and_then(Value::as_str)
                    .map(AuthMethodId::from);
                self.core.auth_logout(method_id).await?;
                encode(&serde_json::json!({}))
            }
            // ---- extensions ------------------------------------------------------------------
            _ if method.starts_with('_') => self.core.ext_request(method, params).await,
            _ => self.core.unknown_request(method, params).await,
        }
    }

    /// `session/new`/`load`/`resume`/`fork` (the setup family).
    async fn dispatch_setup(
        &self,
        method: &str,
        params: Option<Value>,
        v1_mode: bool,
    ) -> Result<Value, RpcError> {
        match method {
            "session/new" if v1_mode => {
                let p: v1::SessionNewParams = decode(params)?;
                let domain = self
                    .init_codec
                    .lock()
                    .expect("init_codec")
                    .new_to_domain(&p);
                let result = self.core.new_session(domain, self.hooks()).await?;
                self.setup_response_v1(&result.session_id, &result, true)
            }
            "session/new" => {
                let p: SessionSetupParams = decode(params)?;
                encode(&self.core.new_session(p, self.hooks()).await?)
            }
            "session/load" if v1_mode => {
                let p: v1::SessionLoadParams = decode(params)?;
                let mut resume = self
                    .init_codec
                    .lock()
                    .expect("init_codec")
                    .load_to_domain(&p);
                resume.replay_from = Some(Open::Known(crate::domain::ReplayCursor::Start(
                    crate::domain::ReplayStart { meta: None },
                )));
                self.session_id_exists_gate(&resume.session_id)?;
                let result = self.core.resume_session(resume, self.hooks()).await?;
                self.setup_response_v1(&result.session_id, &result, false)
            }
            "session/resume" if v1_mode => {
                let p: v1::SessionResumeParams = decode(params)?;
                let resume = self
                    .init_codec
                    .lock()
                    .expect("init_codec")
                    .resume_to_domain(&p);
                self.session_id_exists_gate(&resume.session_id)?;
                let result = self.core.resume_session(resume, self.hooks()).await?;
                self.setup_response_v1(&result.session_id, &result, false)
            }
            "session/resume" => {
                let p: ResumeParams = decode(params)?;
                encode(&self.core.resume_session(p, self.hooks()).await?)
            }
            "session/fork" if v1_mode => {
                let p: v1::SessionForkParams = decode(params)?;
                let resume = domain_fork_params(&p);
                let result = self.core.fork_session(resume, self.hooks()).await?;
                self.fork_response_v1(&result)
            }
            "session/fork" => {
                self.require_fork_cap()?;
                let p: ResumeParams = decode(params)?;
                encode(&self.core.fork_session(p, self.hooks()).await?)
            }
            _ => self.core.unknown_request(method, params).await,
        }
    }

    /// `session/list`/`delete`/`close` (session administration).
    async fn dispatch_admin(
        &self,
        method: &str,
        params: Option<Value>,
        v1_mode: bool,
    ) -> Result<Value, RpcError> {
        match method {
            "session/list" if v1_mode => {
                let p: v1::SessionListParams = decode(params)?;
                let page = self.core.list_sessions(list_to_domain(&p)).await?;
                encode(&v1::SessionListResult {
                    sessions: page.sessions.iter().map(session_info_to_v1).collect(),
                    next_cursor: page.next_cursor,
                    meta: page.meta,
                })
            }
            "session/list" => {
                let p: SessionListParams = decode(params)?;
                encode(&self.core.list_sessions(p).await?)
            }
            "session/delete" if v1_mode => {
                let p: v1::SessionDeleteParams = decode(params)?;
                self.core
                    .delete_session(&SessionId::from(p.session_id))
                    .await?;
                encode(&v1::SessionDeleteResult::default())
            }
            "session/delete" => {
                self.require_delete_cap()?;
                let p: SessionEndParams = decode(params)?;
                self.core.delete_session(&p.session_id).await?;
                encode(&serde_json::json!({}))
            }
            "session/close" if v1_mode => {
                let p: v1::SessionCloseParams = decode(params)?;
                let id = SessionId::from(p.session_id);
                self.core.close_session(&id).await?;
                self.drop_codec(&id);
                encode(&v1::SessionCloseResult::default())
            }
            "session/close" => {
                let p: SessionEndParams = decode(params)?;
                self.core.close_session(&p.session_id).await?;
                self.drop_codec(&p.session_id);
                encode(&serde_json::json!({}))
            }
            _ => self.core.unknown_request(method, params).await,
        }
    }

    /// `session/prompt`, configuration, and turn/session cancellations.
    async fn dispatch_turn(
        &self,
        method: &str,
        params: Option<Value>,
        v1_mode: bool,
    ) -> Result<Value, RpcError> {
        match method {
            "session/prompt" if v1_mode => {
                let p: v1::PromptParams = decode(params)?;
                let session_id = SessionId::from(p.session_id.clone());
                let domain = self.with_codec(&session_id, |c| c.prompt_to_domain(&p));
                let ticket = self.core.prompt(domain).await?;
                // v1: the response carries the turn's stopReason, so it
                // waits for the turn end (queueing/merging handled in core).
                let finished = ticket.finish().await?;
                let result = self.with_codec(&session_id, |c| c.prompt_finished_to_v1(&finished));
                encode(&result)
            }
            "session/prompt" => {
                let mut p: PromptParams = decode(params)?;
                if p.idempotency_key.is_none() {
                    p.idempotency_key = Some(MessageId::new().to_string());
                }
                let ticket = self.core.prompt(p).await?;
                encode(&v2::PromptResult {
                    message_id: ticket.message_id().clone(),
                    meta: None,
                    extra: BTreeMap::new(),
                })
            }
            "session/set_mode" if v1_mode => {
                let p: v1::SessionSetModeParams = decode(params)?;
                let session_id = SessionId::from(p.session_id.clone());
                let domain = self.with_codec(&session_id, |c| c.set_mode_to_domain(&p));
                self.core.set_config(domain).await?;
                encode(&v1::SessionSetModeResult::default())
            }
            "session/set_config_option" if v1_mode => {
                let p: v1::SessionSetConfigOptionParams = decode(params)?;
                let session_id = SessionId::from(p.session_id.clone());
                let domain = self.with_codec(&session_id, |c| c.set_config_to_domain(&p));
                let options = self.core.set_config(domain).await?;
                let config_options = self.with_codec(&session_id, |c| {
                    options.iter().map(|o| c.config_option_to_v1(o)).collect()
                });
                encode(&v1::SessionSetConfigOptionResult {
                    config_options,
                    meta: None,
                })
            }
            "session/set_config_option" => {
                let p: SetConfigParams = decode(params)?;
                self.core.set_config(p).await?;
                encode(&serde_json::json!({}))
            }
            "session/tool_call_cancel" => {
                #[derive(serde::Deserialize)]
                #[serde(rename_all = "camelCase")]
                struct Cancel {
                    session_id: SessionId,
                    tool_call_id: ToolCallId,
                }
                let p: Cancel = decode(params)?;
                self.core.tool_call_cancel(&p.session_id, &p.tool_call_id);
                encode(&serde_json::json!({}))
            }
            "session/terminal_cancel" => {
                #[derive(serde::Deserialize)]
                #[serde(rename_all = "camelCase")]
                struct Cancel {
                    session_id: SessionId,
                    terminal_id: TerminalId,
                }
                let p: Cancel = decode(params)?;
                self.core.terminal_cancel(&p.session_id, &p.terminal_id);
                encode(&serde_json::json!({}))
            }
            _ => self.core.unknown_request(method, params).await,
        }
    }

    /// Route an inbound notification.
    fn dispatch_notification(&self, n: Notification) -> impl Future<Output = ()> + Send {
        match n.method.as_str() {
            "session/cancel" => {
                #[derive(serde::Deserialize)]
                #[serde(rename_all = "camelCase")]
                struct Cancel {
                    session_id: String,
                }
                match decode::<Cancel>(n.params) {
                    Ok(p) => self.core.cancel_session(&SessionId::from(p.session_id)),
                    Err(e) => warn!("bad session/cancel params: {e}"),
                }
            }
            "elicitation/complete" => {
                match decode::<ElicitComplete>(n.params) {
                    Ok(p) => {
                        // Route through every live session; the waiter
                        // lives on the owning session's shared state.
                        // Collect ids first: `elicit_complete` re-locks
                        // the same `sessions` mutex via `entry`.
                        let ids: Vec<SessionId> = self
                            .core
                            .sessions
                            .lock()
                            .expect("sessions")
                            .values()
                            .map(|entry| entry.id.clone())
                            .collect();
                        for id in &ids {
                            self.core.elicit_complete(id, &p);
                        }
                    }
                    Err(e) => warn!("bad elicitation/complete params: {e}"),
                }
            }
            method => debug!("dropping unknown notification {method}"),
        }
        future::ready(())
    }

    /// `initialize`: detect the requested version, decode via the right
    /// codec, and respond with the negotiated version's shape.
    async fn initialize(&self, params: Option<Value>) -> Result<Value, RpcError> {
        let params = params.ok_or_else(|| RpcError::invalid_params("missing params"))?;
        let requested = params
            .get("protocolVersion")
            .and_then(Value::as_u64)
            .ok_or_else(|| RpcError::invalid_params("missing protocolVersion"))?;
        let requested = u16::try_from(requested)
            .map_err(|_| RpcError::invalid_params("protocolVersion out of range"))?;
        let domain_params = if requested == 1 {
            let p: v1::InitializeParams = decode(Some(params.clone()))?;
            InitializeParams {
                protocol_version: p.protocol_version,
                capabilities: self
                    .init_codec
                    .lock()
                    .expect("init_codec")
                    .client_caps_to_domain(&p.client_capabilities),
                info: p.client_info.map(|i| Implementation {
                    name: i.name,
                    title: i.title,
                    version: i.version,
                    meta: None,
                    extra: BTreeMap::new(),
                }),
                meta: p.meta,
            }
        } else {
            // v2 shape (`capabilities`/`info`), tolerating dual-shape
            // clients that also send the v1 field names.
            let v = &params;
            let caps = v
                .get("capabilities")
                .or_else(|| v.get("clientCapabilities"));
            let capabilities = match caps {
                Some(caps) => serde_json::from_value(caps.clone())
                    .map_err(|e| RpcError::invalid_params(format!("capabilities: {e}")))?,
                None => ClientCapabilities::default(),
            };
            let info = v
                .get("info")
                .or_else(|| v.get("clientInfo"))
                .and_then(|i| serde_json::from_value(i.clone()).ok());
            InitializeParams {
                protocol_version: requested,
                capabilities,
                info,
                meta: v.get("_meta").cloned(),
            }
        };
        let result = self.core.initialize(domain_params).await?;
        if requested == 1 {
            encode(
                &self
                    .init_codec
                    .lock()
                    .expect("init_codec")
                    .initialize_result_to_v1(&result),
            )
        } else {
            encode(&v2::InitializeResponse {
                protocol_version: result.protocol_version,
                capabilities: result.capabilities,
                info: result.info,
                auth_methods: result.auth_methods,
                meta: result.meta,
                extra: BTreeMap::new(),
            })
        }
    }

    /// Respond to a v1 session setup (`session/new`/`session/load`/
    /// `session/resume`) with the codec-translated fields, emitting the
    /// codec's `pending_updates` first (the engine writes them before
    /// the response — the only causal ordering available without an
    /// executor task).
    fn setup_response_v1(
        &self,
        session_id: &SessionId,
        result: &SetupResult,
        new: bool,
    ) -> Result<Value, RpcError> {
        let mut codec = V1Codec::new();
        // Feed the session codec the setup so mode/config state is cached.
        let wire = codec.setup_result_to_v1(result);
        let updates = wire.pending_updates.clone();
        let value = if new {
            encode(&v1::SessionNewResult {
                session_id: session_id.to_string(),
                modes: wire.modes,
                config_options: Some(wire.config_options),
                meta: result.meta.clone(),
                extra: BTreeMap::new(),
            })?
        } else {
            encode(&v1::SessionLoadResult {
                modes: wire.modes,
                config_options: Some(wire.config_options),
                meta: result.meta.clone(),
                extra: BTreeMap::new(),
            })?
        };
        // Queue the pending updates ahead of the response.
        if let Ok(peer) = self.peer() {
            for update in updates {
                let notification = v1::SessionNotification::new(session_id.to_string(), update);
                drop(peer.notify("session/update", &notification));
            }
        }
        // Keep the codec for the session's later traffic.
        *self
            .codecs
            .lock()
            .expect("codecs")
            .entry(session_id.clone())
            .or_default() = codec;
        Ok(value)
    }

    /// v1 `session/fork` response (same fields + `sessionId`).
    fn fork_response_v1(&self, result: &SetupResult) -> Result<Value, RpcError> {
        self.setup_response_v1(&result.session_id, result, false)
            .map(|mut v| {
                if let Value::Object(map) = &mut v {
                    map.insert(
                        "sessionId".to_string(),
                        Value::String(result.session_id.to_string()),
                    );
                }
                v
            })
    }

    /// Gate `session/fork` on the advertised capability.
    fn require_fork_cap(&self) -> Result<(), RpcError> {
        let caps = self.core.capabilities();
        if caps.session.and_then(|s| s.fork).is_some() {
            Ok(())
        } else {
            Err(RpcError::method_not_found("session/fork"))
        }
    }

    /// Gate `session/delete` on the advertised capability.
    fn require_delete_cap(&self) -> Result<(), RpcError> {
        let caps = self.core.capabilities();
        if caps.session.and_then(|s| s.delete).is_some() {
            Ok(())
        } else {
            Err(RpcError::method_not_found("session/delete"))
        }
    }

    /// `session/load`/`session/resume` on an id already live is invalid.
    fn session_id_exists_gate(&self, session_id: &SessionId) -> Result<(), RpcError> {
        if self.core.entry(session_id).is_some() {
            return Err(RpcError::invalid_params(format!(
                "session {session_id} is already live"
            )));
        }
        Ok(())
    }
}

/// Map v1 `session/fork` params to domain [`ResumeParams`].
fn domain_fork_params(p: &v1::SessionForkParams) -> ResumeParams {
    ResumeParams {
        setup: SessionSetupParams {
            cwd: p.cwd.clone(),
            additional_directories: p.additional_directories.clone(),
            mcp_servers: Vec::new(),
            system_prompt: None,
            meta: None,
            extra: BTreeMap::new(),
        },
        session_id: SessionId::from(p.session_id.clone()),
        replay_from: None,
    }
}

/// Map v1 `session/list` params to domain.
fn list_to_domain(p: &v1::SessionListParams) -> SessionListParams {
    SessionListParams {
        cwd: p.cwd.clone(),
        cursor: p.cursor.clone(),
        meta: p.meta.clone(),
        extra: BTreeMap::new(),
    }
}

/// Domain `SessionInfo` → v1 `SessionInfo`.
fn session_info_to_v1(s: &SessionInfo) -> v1::SessionInfo {
    v1::SessionInfo {
        session_id: s.session_id.to_string(),
        cwd: s.cwd.clone(),
        additional_directories: Vec::new(),
        title: s.title.clone(),
        updated_at: s.updated_at.clone(),
        meta: s.meta.clone(),
        extra: s.extra.clone(),
    }
}

// =============================================================================
// WireHooks — AgentHooks over Peer
// =============================================================================

/// The agent→client lane for a connection, shared across sessions.
#[derive(Clone)]
struct WireHooks {
    peer: Arc<Mutex<Option<Peer>>>,
    codecs: Arc<Mutex<HashMap<SessionId, V1Codec>>>,
    core: Weak<ServerCore>,
}

impl WireHooks {
    fn peer(&self) -> Result<Peer, PeerError> {
        self.peer
            .lock()
            .expect("peer")
            .clone()
            .ok_or(PeerError::Closed { status: None })
    }

    fn protocol(&self) -> ProtocolVersion {
        self.core
            .upgrade()
            .and_then(|c| c.protocol())
            .unwrap_or(ProtocolVersion::V2)
    }

    fn caps(&self) -> ClientCapabilities {
        self.core
            .upgrade()
            .map_or_else(ClientCapabilities::default, |c| c.client_caps())
    }

    fn emit_v1(&self, session_id: &SessionId, update: &SessionUpdate) {
        let Ok(peer) = self.peer() else { return };
        let updates = self
            .codecs
            .lock()
            .expect("codecs")
            .entry(session_id.clone())
            .or_default()
            .update_to_v1(update);
        for update in updates {
            let notification = v1::SessionNotification::new(session_id.to_string(), update);
            drop(peer.notify("session/update", &notification));
        }
    }
}

impl AgentHooks for WireHooks {
    fn session_update(
        &self,
        session_id: &SessionId,
        update: SessionUpdate,
    ) -> impl Future<Output = ()> + Send {
        match self.protocol() {
            ProtocolVersion::V1 => self.emit_v1(session_id, &update),
            ProtocolVersion::V2 => {
                if let Ok(peer) = self.peer() {
                    let notification = SessionNotification {
                        session_id: session_id.clone(),
                        update,
                        meta: None,
                        extra: BTreeMap::new(),
                    };
                    drop(peer.notify("session/update", &notification));
                }
            }
        }
        future::ready(())
    }

    async fn request_permission(
        &self,
        params: crate::domain::PermissionRequest,
    ) -> Result<crate::domain::PermissionOutcome, PeerError> {
        let peer = self.peer()?;
        match self.protocol() {
            ProtocolVersion::V1 => {
                let v1_params = {
                    let mut codecs = self.codecs.lock().expect("codecs");
                    codecs
                        .entry(params.session_id.clone())
                        .or_default()
                        .permission_to_v1(&params)
                };
                let result: v1::RequestPermissionResult = peer
                    .request("session/request_permission", &v1_params)
                    .into_future()
                    .await?;
                Ok(self
                    .codecs
                    .lock()
                    .expect("codecs")
                    .get(&params.session_id)
                    .map_or_else(
                        || {
                            crate::domain::PermissionOutcome::Cancelled(
                                crate::domain::CancelledOutcome { meta: None },
                            )
                        },
                        |c| c.outcome_to_domain(&result.outcome),
                    ))
            }
            ProtocolVersion::V2 => {
                let result: crate::domain::PermissionResult = peer
                    .request("session/request_permission", &params)
                    .into_future()
                    .await?;
                Ok(result.outcome)
            }
        }
    }

    async fn elicit(
        &self,
        session_id: &SessionId,
        params: crate::domain::ElicitCreate,
    ) -> Option<Result<crate::domain::ElicitResult, PeerError>> {
        // Capability gate.
        let supported = {
            let caps = self.caps();
            caps.elicitation
                .as_ref()
                .is_some_and(|e| match &params.mode {
                    Open::Known(crate::domain::ElicitMode::Form(_)) => e.form.is_some(),
                    Open::Known(crate::domain::ElicitMode::Url(_)) => e.url.is_some(),
                    _ => true,
                })
        };
        if !supported {
            return None;
        }
        let peer = match self.peer() {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };
        Some(match self.protocol() {
            ProtocolVersion::V1 => {
                let v1_params = {
                    let mut codecs = self.codecs.lock().expect("codecs");
                    match codecs
                        .entry(session_id.clone())
                        .or_default()
                        .elicit_to_v1(&params, session_id)
                    {
                        Ok(p) => p,
                        Err(raw) => {
                            return Some(Err(PeerError::Rejected(RpcError::invalid_params(
                                format!("elicitation not expressible on v1: {raw}"),
                            ))));
                        }
                    }
                };
                peer.request::<_, v1::ElicitationCreateResult>("elicitation/create", &v1_params)
                    .into_future()
                    .await
                    .map(|r| {
                        self.codecs
                            .lock()
                            .expect("codecs")
                            .get(session_id)
                            .map_or_else(
                                || crate::domain::ElicitResult {
                                    action: Open::Known(crate::domain::ElicitAction::Cancel(
                                        crate::domain::CancelAction { meta: None },
                                    )),
                                    meta: None,
                                    extra: BTreeMap::new(),
                                },
                                |c| c.elicit_result_to_domain(&r),
                            )
                    })
            }
            ProtocolVersion::V2 => {
                peer.request::<_, crate::domain::ElicitResult>("elicitation/create", &params)
                    .into_future()
                    .await
            }
        })
    }

    async fn ext_request(
        &self,
        method: &ExtMethod,
        params: Option<Value>,
    ) -> Result<Value, PeerError> {
        self.peer()?
            .request_untyped(method.as_str(), params)
            .into_future()
            .await
    }

    async fn notification(&self, method: &str, params: Option<Value>) -> Result<(), PeerError> {
        self.peer()?
            .notify(method, &params.unwrap_or(Value::Null))
            .await
            .map_err(|_| PeerError::Closed { status: None })
    }

    async fn fs_read_text(
        &self,
        session_id: &SessionId,
        path: &std::path::Path,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> Option<Result<String, PeerError>> {
        if self.protocol() != ProtocolVersion::V1 {
            return None;
        }
        let peer = match self.peer() {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };
        let params = v1::ReadTextFileParams {
            session_id: session_id.to_string(),
            path: path.to_path_buf(),
            line,
            limit,
            meta: None,
        };
        Some(
            peer.request::<_, v1::ReadTextFileResult>("fs/read_text_file", &params)
                .into_future()
                .await
                .map(|r| r.content),
        )
    }

    async fn fs_write_text(
        &self,
        session_id: &SessionId,
        path: &std::path::Path,
        content: &str,
    ) -> Option<Result<(), PeerError>> {
        if self.protocol() != ProtocolVersion::V1 {
            return None;
        }
        let peer = match self.peer() {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };
        let params = v1::WriteTextFileParams {
            session_id: session_id.to_string(),
            path: path.to_path_buf(),
            content: content.to_string(),
            meta: None,
        };
        Some(
            peer.request::<_, v1::WriteTextFileResult>("fs/write_text_file", &params)
                .into_future()
                .await
                .map(|_| ()),
        )
    }

    async fn terminal_create(
        &self,
        _session_id: &SessionId,
        params: v1::TerminalCreateParams,
    ) -> Option<Result<v1::TerminalCreateResult, PeerError>> {
        if self.protocol() != ProtocolVersion::V1 {
            return None;
        }
        let peer = match self.peer() {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };
        Some(
            peer.request::<_, v1::TerminalCreateResult>("terminal/create", &params)
                .into_future()
                .await,
        )
    }

    async fn terminal_output(
        &self,
        session_id: &SessionId,
        terminal_id: &str,
    ) -> Option<Result<v1::TerminalOutputResult, PeerError>> {
        if self.protocol() != ProtocolVersion::V1 {
            return None;
        }
        let peer = match self.peer() {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };
        let params = v1::TerminalOutputParams {
            session_id: session_id.to_string(),
            terminal_id: terminal_id.to_string(),
            meta: None,
        };
        Some(
            peer.request::<_, v1::TerminalOutputResult>("terminal/output", &params)
                .into_future()
                .await,
        )
    }

    async fn terminal_wait_for_exit(
        &self,
        session_id: &SessionId,
        terminal_id: &str,
    ) -> Option<Result<v1::TerminalExitStatus, PeerError>> {
        if self.protocol() != ProtocolVersion::V1 {
            return None;
        }
        let peer = match self.peer() {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };
        let params = v1::TerminalWaitForExitParams {
            session_id: session_id.to_string(),
            terminal_id: terminal_id.to_string(),
            meta: None,
        };
        Some(
            peer.request::<_, v1::TerminalExitStatus>("terminal/wait_for_exit", &params)
                .into_future()
                .await,
        )
    }

    async fn terminal_kill(
        &self,
        session_id: &SessionId,
        terminal_id: &str,
    ) -> Option<Result<(), PeerError>> {
        if self.protocol() != ProtocolVersion::V1 {
            return None;
        }
        let peer = match self.peer() {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };
        let params = v1::TerminalKillParams {
            session_id: session_id.to_string(),
            terminal_id: terminal_id.to_string(),
            meta: None,
        };
        Some(
            peer.request::<_, v1::TerminalKillResult>("terminal/kill", &params)
                .into_future()
                .await
                .map(|_| ()),
        )
    }

    async fn terminal_release(
        &self,
        session_id: &SessionId,
        terminal_id: &str,
    ) -> Option<Result<(), PeerError>> {
        if self.protocol() != ProtocolVersion::V1 {
            return None;
        }
        let peer = match self.peer() {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };
        let params = v1::TerminalReleaseParams {
            session_id: session_id.to_string(),
            terminal_id: terminal_id.to_string(),
            meta: None,
        };
        Some(
            peer.request::<_, v1::TerminalReleaseResult>("terminal/release", &params)
                .into_future()
                .await
                .map(|_| ()),
        )
    }
}
