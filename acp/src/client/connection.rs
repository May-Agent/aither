//! The client→agent lane abstraction and the JSON-RPC transport binding.
//!
//! Every `Client` talks to its agent through [`AnyAgentLane`], a
//! domain-typed, version-neutral operation set. Two implementations exist:
//!
//! - [`WireLane`] — a [`Peer`] over an [`rpc::Connection`], negotiated to
//!   ACP v1 or v2. v1 traffic is translated through per-session
//!   [`V1Codec`]s that live in the session cells shared with
//!   [`WireHooks`].
//! - `DirectLane` ([`super::direct`]) — the zero-serde in-process lane.

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tracing::{debug, warn};

use crate::domain::{
    AuthMethodId, ClientCapabilities, ElicitComplete, ExtMethod, Implementation, InitResult,
    MessageId, PermissionOutcome, PermissionRequest, PromptFinished, PromptParams, ResumeParams,
    SessionId, SessionListParams, SessionNotification, SessionPage, SessionSetupParams,
    SetConfigParams, SetupResult, TerminalId, ToolCallId, config::ConfigOption,
};
use crate::rpc::{
    Handler, Notification, Peer, PeerError, Request, RequestContext, Response, RpcError,
};
use crate::wire::codec::{self, Protocol, V1Codec, V1Resume, V1SetConfig};
use crate::wire::v1;
use crate::wire::v2;

use super::error::ClientError;
use super::handle::{SessionCell, SessionRegistry};
use super::handler::{AnyAgentHooks, respond_ok};

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// What `session/prompt` answered.
#[derive(Debug, Clone)]
pub enum PromptReply {
    /// v2 semantics: the prompt was inserted; the turn ends on the first
    /// idle `state_update` afterwards.
    Accepted {
        /// Server-assigned message id, if any.
        message_id: Option<MessageId>,
    },
    /// v1 semantics: the response already carries the finished turn.
    Finished(PromptFinished),
}

/// The negotiated wire protocol, shared between [`WireLane`] and
/// [`WireHooks`] so both sides encode for the same version.
#[derive(Debug, Default)]
pub struct SharedVersion(AtomicU8);

impl SharedVersion {
    const V1: u8 = 1;
    const V2: u8 = 2;

    /// Record the negotiated version.
    pub(crate) fn set(&self, version: Protocol) {
        let raw = match version {
            Protocol::V1 => Self::V1,
            Protocol::V2 => Self::V2,
        };
        self.0.store(raw, Ordering::Release);
    }

    /// The negotiated version. [`Protocol::V2`] while the handshake
    /// is still in flight — domain-shaped decoding is the safer default
    /// for protocol-violating early traffic.
    pub(crate) fn get(&self) -> Protocol {
        match self.0.load(Ordering::Acquire) {
            Self::V1 => Protocol::V1,
            _ => Protocol::V2,
        }
    }
}

/// Object-safe twin of the public lane surface.
///
/// Every client→agent operation, domain-typed. Methods return boxed
/// futures internally; [`AnyAgentLane`] re-exposes them as `impl Future`.
pub trait AgentLaneImpl: Send + Sync {
    /// `initialize` — negotiate and return the init result.
    fn initialize(
        &self,
        capabilities: ClientCapabilities,
        info: Implementation,
    ) -> BoxFuture<'_, Result<InitResult, ClientError>>;
    /// The negotiated ACP protocol version (1 or 2).
    fn protocol_version(&self) -> u16;
    /// `session/new`.
    fn new_session(
        &self,
        params: SessionSetupParams,
    ) -> BoxFuture<'_, Result<SetupResult, ClientError>>;
    /// `session/resume` / `session/load`.
    fn resume_session(
        &self,
        params: ResumeParams,
    ) -> BoxFuture<'_, Result<SetupResult, ClientError>>;
    /// `session/fork`.
    fn fork_session(&self, params: ResumeParams)
    -> BoxFuture<'_, Result<SetupResult, ClientError>>;
    /// `session/list`.
    fn list_sessions(
        &self,
        params: SessionListParams,
    ) -> BoxFuture<'_, Result<SessionPage, ClientError>>;
    /// `session/delete`.
    fn delete_session(&self, id: SessionId) -> BoxFuture<'_, Result<(), ClientError>>;
    /// `session/close`.
    fn close_session(&self, id: SessionId) -> BoxFuture<'_, Result<(), ClientError>>;
    /// `session/prompt`.
    fn prompt(&self, params: PromptParams) -> BoxFuture<'_, Result<PromptReply, ClientError>>;
    /// `session/cancel` notification.
    fn cancel_session(&self, id: SessionId) -> BoxFuture<'_, Result<(), ClientError>>;
    /// `session/set_config_option` (v1 codec routes `session/set_mode`).
    /// Returns the new option list when the peer reports one (v1 does).
    fn set_config(
        &self,
        params: SetConfigParams,
    ) -> BoxFuture<'_, Result<Vec<ConfigOption>, ClientError>>;
    /// `session/tool_call_cancel` notification (v2 only).
    fn cancel_tool_call(
        &self,
        session: SessionId,
        tool_call: ToolCallId,
    ) -> BoxFuture<'_, Result<(), ClientError>>;
    /// `session/terminal_cancel` notification (v2 only).
    fn cancel_terminal(
        &self,
        session: SessionId,
        terminal: TerminalId,
    ) -> BoxFuture<'_, Result<(), ClientError>>;
    /// `auth/login` (v2) / `authenticate` (v1).
    fn authenticate(&self, method: AuthMethodId) -> BoxFuture<'_, Result<(), ClientError>>;
    /// `auth/logout` (v2) / `logout` (v1).
    fn logout(&self, method: Option<AuthMethodId>) -> BoxFuture<'_, Result<(), ClientError>>;
    /// A `_`-prefixed request; raw JSON in and out.
    fn ext_request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> BoxFuture<'_, Result<Value, ClientError>>;
    /// A `_`-prefixed notification.
    fn ext_notify(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> BoxFuture<'_, Result<(), ClientError>>;
    /// A legacy (non-`_`) request; used by the v1 codex
    /// `session/set_model` fallback.
    fn legacy_request(
        &self,
        method: &str,
        params: Value,
    ) -> BoxFuture<'_, Result<Value, ClientError>>;
    /// Begin shutting the lane down.
    fn shutdown(&self) -> BoxFuture<'_, ()>;
}

/// The public lane wrapper: one type for wire and in-process clients.
#[derive(Clone)]
pub struct AnyAgentLane(Arc<dyn AgentLaneImpl>);

impl fmt::Debug for AnyAgentLane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyAgentLane").finish_non_exhaustive()
    }
}

impl AnyAgentLane {
    /// Wrap a lane implementation.
    pub(crate) fn new(lane: impl AgentLaneImpl + 'static) -> Self {
        Self(Arc::new(lane))
    }

    /// `initialize`.
    pub fn initialize(
        &self,
        capabilities: ClientCapabilities,
        info: Implementation,
    ) -> impl Future<Output = Result<InitResult, ClientError>> + Send + '_ {
        self.0.initialize(capabilities, info)
    }
    /// The negotiated protocol version (1 or 2).
    #[must_use]
    pub fn protocol_version(&self) -> u16 {
        self.0.protocol_version()
    }
    /// `session/new`.
    pub fn new_session(
        &self,
        params: SessionSetupParams,
    ) -> impl Future<Output = Result<SetupResult, ClientError>> + Send + '_ {
        self.0.new_session(params)
    }
    /// `session/resume` / `session/load`.
    pub fn resume_session(
        &self,
        params: ResumeParams,
    ) -> impl Future<Output = Result<SetupResult, ClientError>> + Send + '_ {
        self.0.resume_session(params)
    }
    /// `session/fork`.
    pub fn fork_session(
        &self,
        params: ResumeParams,
    ) -> impl Future<Output = Result<SetupResult, ClientError>> + Send + '_ {
        self.0.fork_session(params)
    }
    /// `session/list`.
    pub fn list_sessions(
        &self,
        params: SessionListParams,
    ) -> impl Future<Output = Result<SessionPage, ClientError>> + Send + '_ {
        self.0.list_sessions(params)
    }
    /// `session/delete`.
    pub fn delete_session(
        &self,
        id: SessionId,
    ) -> impl Future<Output = Result<(), ClientError>> + Send + '_ {
        self.0.delete_session(id)
    }
    /// `session/close`.
    pub fn close_session(
        &self,
        id: SessionId,
    ) -> impl Future<Output = Result<(), ClientError>> + Send + '_ {
        self.0.close_session(id)
    }
    /// `session/prompt`.
    pub(crate) fn prompt(
        &self,
        params: PromptParams,
    ) -> impl Future<Output = Result<PromptReply, ClientError>> + Send + '_ {
        self.0.prompt(params)
    }
    /// `session/cancel` notification.
    pub fn cancel_session(
        &self,
        id: SessionId,
    ) -> impl Future<Output = Result<(), ClientError>> + Send + '_ {
        self.0.cancel_session(id)
    }
    /// `session/set_config_option`.
    pub(crate) fn set_config(
        &self,
        params: SetConfigParams,
    ) -> impl Future<Output = Result<Vec<ConfigOption>, ClientError>> + Send + '_ {
        self.0.set_config(params)
    }
    /// `session/tool_call_cancel` notification (v2 only).
    pub fn cancel_tool_call(
        &self,
        session: SessionId,
        tool_call: ToolCallId,
    ) -> impl Future<Output = Result<(), ClientError>> + Send + '_ {
        self.0.cancel_tool_call(session, tool_call)
    }
    /// `session/terminal_cancel` notification (v2 only).
    pub fn cancel_terminal(
        &self,
        session: SessionId,
        terminal: TerminalId,
    ) -> impl Future<Output = Result<(), ClientError>> + Send + '_ {
        self.0.cancel_terminal(session, terminal)
    }
    /// `auth/login` (v2) / `authenticate` (v1).
    pub fn authenticate(
        &self,
        method: AuthMethodId,
    ) -> impl Future<Output = Result<(), ClientError>> + Send + '_ {
        self.0.authenticate(method)
    }
    /// `auth/logout` (v2) / `logout` (v1).
    pub fn logout(
        &self,
        method: Option<AuthMethodId>,
    ) -> impl Future<Output = Result<(), ClientError>> + Send + '_ {
        self.0.logout(method)
    }
    /// `_`-prefixed extension request.
    pub fn ext_request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> impl Future<Output = Result<Value, ClientError>> + Send + '_ {
        self.0.ext_request(method, params)
    }
    /// `_`-prefixed extension notification.
    pub fn ext_notify(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> impl Future<Output = Result<(), ClientError>> + Send + '_ {
        self.0.ext_notify(method, params)
    }
    /// Legacy non-`_` request (v1 vendor fallbacks).
    pub(crate) fn legacy_request(
        &self,
        method: &str,
        params: Value,
    ) -> impl Future<Output = Result<Value, ClientError>> + Send + '_ {
        self.0.legacy_request(method, params)
    }
    /// Shut the lane down.
    pub(crate) fn shutdown(&self) -> impl Future<Output = ()> + Send + '_ {
        self.0.shutdown()
    }
}

/// Map a [`PeerError`] into [`ClientError`] (the shared `From` impl,
/// spelled out for call sites).
fn lane_error(error: PeerError) -> ClientError {
    ClientError::from(error)
}

/// Typed request: [`Peer::request`] serializes `params` eagerly, so no
/// `&P` is held across the await — keeps the returned future `Send`.
fn send<P: Serialize, R: DeserializeOwned>(
    peer: &Peer,
    method: &str,
    params: &P,
) -> impl Future<Output = Result<R, ClientError>> + Send {
    let reply = peer.request::<P, R>(method, params);
    async move { reply.await.map_err(lane_error) }
}

/// Typed notification — see [`send`].
fn notify<P: Serialize>(
    peer: &Peer,
    method: &str,
    params: &P,
) -> impl Future<Output = Result<(), ClientError>> + Send {
    let sent = peer.notify(method, params);
    async move { sent.await.map_err(ClientError::from) }
}

/// A [`Peer`]-backed lane over the JSON-RPC connection.
pub struct WireLane {
    peer: Peer,
    version: Arc<SharedVersion>,
    sessions: SessionRegistry,
}

impl WireLane {
    /// A lane bound to `peer`, `version`, and the shared session registry.
    pub(crate) const fn new(
        peer: Peer,
        version: Arc<SharedVersion>,
        sessions: SessionRegistry,
    ) -> Self {
        Self {
            peer,
            version,
            sessions,
        }
    }

    fn v2(&self) -> bool {
        matches!(self.version.get(), Protocol::V2)
    }

    /// Translate a setup result through the per-session v1 codec.
    fn v1_setup_result(&self, session_id: &str, result: &v1::SessionNewResult) -> SetupResult {
        let cell = self
            .sessions
            .get_or_create(SessionId::from_string(session_id));
        let mut codec = cell.codec.lock().expect("codec poisoned");
        codec.setup_result_from_v1(result)
    }

    fn v1_codec(&self, session_id: &SessionId) -> Arc<SessionCell> {
        self.sessions.get_or_create(session_id.clone())
    }

    async fn v1_resume(&self, params: ResumeParams) -> Result<SetupResult, ClientError> {
        let codec = V1Codec::new();
        match codec.resume_to_v1(&params) {
            V1Resume::Load(p) => {
                let result: v1::SessionLoadResult = send(&self.peer, "session/load", &p).await?;
                let new = v1::SessionNewResult {
                    session_id: p.session_id.clone(),
                    modes: result.modes,
                    config_options: result.config_options,
                    meta: result.meta,
                    extra: result.extra,
                };
                Ok(self.v1_setup_result(&p.session_id, &new))
            }
            V1Resume::Resume(p) => {
                let result: v1::SessionResumeResult =
                    send(&self.peer, "session/resume", &p).await?;
                let new = v1::SessionNewResult {
                    session_id: p.session_id.clone(),
                    modes: result.modes,
                    config_options: result.config_options,
                    meta: result.meta,
                    extra: result.extra,
                };
                Ok(self.v1_setup_result(&p.session_id, &new))
            }
            V1Resume::UnsupportedCursor(raw) => Err(ClientError::Protocol(format!(
                "resume replay cursor {raw:?} has no v1 representation"
            ))),
        }
    }
}

impl AgentLaneImpl for WireLane {
    fn initialize(
        &self,
        capabilities: ClientCapabilities,
        info: Implementation,
    ) -> BoxFuture<'_, Result<InitResult, ClientError>> {
        Box::pin(async move {
            let params = codec::initialize_params_dual(&capabilities, &info, None);
            match self.peer.request_untyped("initialize", Some(params)).await {
                Ok(value) => {
                    let version = codec::detect_initialize(&value);
                    self.version.set(version);
                    let init = codec::initialize_result_detect(&value)
                        .map_err(|e| ClientError::Protocol(e.to_string()))?;
                    debug!(
                        negotiated = init.protocol_version,
                        shape = ?version,
                        "initialize negotiated"
                    );
                    Ok(init)
                }
                Err(PeerError::Rejected(e))
                    if e.code() == crate::rpc::PROTOCOL_VERSION_UNSUPPORTED =>
                {
                    // The agent flatly refuses version 2: retry as a pure
                    // v1 handshake (a tolerant agent would have answered
                    // the dual request itself).
                    let v1_params = v1::InitializeParams {
                        protocol_version: 1,
                        client_capabilities: V1Codec::new().client_caps_to_v1(&capabilities),
                        client_info: Some(v1::Implementation {
                            name: info.name,
                            title: info.title,
                            version: info.version,
                        }),
                        meta: None,
                    };
                    let result: v1::InitializeResult =
                        send(&self.peer, "initialize", &v1_params).await?;
                    self.version.set(Protocol::V1);
                    Ok(V1Codec::new().initialize_result_to_domain(&result))
                }
                Err(e) => Err(lane_error(e)),
            }
        })
    }

    fn protocol_version(&self) -> u16 {
        match self.version.get() {
            Protocol::V1 => 1,
            Protocol::V2 => 2,
        }
    }

    fn new_session(
        &self,
        params: SessionSetupParams,
    ) -> BoxFuture<'_, Result<SetupResult, ClientError>> {
        Box::pin(async move {
            if self.v2() {
                send(&self.peer, "session/new", &params).await
            } else {
                let p = V1Codec::new().setup_to_v1(&params);
                let result: v1::SessionNewResult = send(&self.peer, "session/new", &p).await?;
                let id = result.session_id.clone();
                Ok(self.v1_setup_result(&id, &result))
            }
        })
    }

    fn resume_session(
        &self,
        params: ResumeParams,
    ) -> BoxFuture<'_, Result<SetupResult, ClientError>> {
        Box::pin(async move {
            if self.v2() {
                send(&self.peer, "session/resume", &params).await
            } else {
                self.v1_resume(params).await
            }
        })
    }

    fn fork_session(
        &self,
        params: ResumeParams,
    ) -> BoxFuture<'_, Result<SetupResult, ClientError>> {
        Box::pin(async move {
            if self.v2() {
                send(&self.peer, "session/fork", &params).await
            } else {
                let codec = V1Codec::new();
                let mcp_servers = params
                    .setup
                    .mcp_servers
                    .iter()
                    .filter_map(|s| codec.mcp_server_to_v1(s))
                    .collect();
                let mut p =
                    v1::SessionForkParams::new(params.session_id.as_str(), params.setup.cwd)
                        .mcp_servers(mcp_servers)
                        .additional_directories(params.setup.additional_directories);
                if let Some(meta) = params.setup.meta {
                    p = p.meta(meta);
                }
                let result: v1::SessionForkResult = send(&self.peer, "session/fork", &p).await?;
                let new = v1::SessionNewResult {
                    session_id: result.session_id.clone(),
                    modes: result.modes,
                    config_options: result.config_options,
                    meta: result.meta,
                    extra: result.extra,
                };
                Ok(self.v1_setup_result(&result.session_id, &new))
            }
        })
    }

    fn list_sessions(
        &self,
        params: SessionListParams,
    ) -> BoxFuture<'_, Result<SessionPage, ClientError>> {
        Box::pin(async move {
            if self.v2() {
                let result: SessionPage = send(&self.peer, "session/list", &params).await?;
                Ok(result)
            } else {
                let p = v1::SessionListParams {
                    cwd: params.cwd,
                    cursor: params.cursor,
                    meta: params.meta,
                };
                let result: v1::SessionListResult = send(&self.peer, "session/list", &p).await?;
                let sessions = result
                    .sessions
                    .into_iter()
                    .map(|s| {
                        let mut extra = s.extra;
                        if !s.additional_directories.is_empty() {
                            extra.insert(
                                "additionalDirectories".to_string(),
                                json!(s.additional_directories),
                            );
                        }
                        crate::domain::SessionInfo {
                            session_id: SessionId::from_string(s.session_id),
                            cwd: s.cwd,
                            title: s.title,
                            updated_at: s.updated_at,
                            meta: s.meta,
                            extra,
                        }
                    })
                    .collect();
                Ok(SessionPage {
                    sessions,
                    next_cursor: result.next_cursor,
                    meta: result.meta,
                    extra: BTreeMap::new(),
                })
            }
        })
    }

    fn delete_session(&self, id: SessionId) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move {
            let params = json!({"sessionId": id.as_str()});
            let _: Value = self
                .peer
                .request_untyped("session/delete", Some(params))
                .await
                .map_err(lane_error)?;
            Ok(())
        })
    }

    fn close_session(&self, id: SessionId) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move {
            let params = json!({"sessionId": id.as_str()});
            let _: Value = self
                .peer
                .request_untyped("session/close", Some(params))
                .await
                .map_err(lane_error)?;
            self.sessions.remove(&id);
            Ok(())
        })
    }

    fn prompt(&self, params: PromptParams) -> BoxFuture<'_, Result<PromptReply, ClientError>> {
        Box::pin(async move {
            if self.v2() {
                let result: v2::PromptResult = send(&self.peer, "session/prompt", &params).await?;
                Ok(PromptReply::Accepted {
                    message_id: Some(result.message_id),
                })
            } else {
                let cell = self.v1_codec(&params.session_id);
                let p = {
                    let codec = cell.codec.lock().expect("codec poisoned");
                    codec.prompt_to_v1(&params)
                };
                let result: v1::PromptResult = send(&self.peer, "session/prompt", &p).await?;
                let finished = {
                    let codec = cell.codec.lock().expect("codec poisoned");
                    codec.prompt_result_to_domain(&result)
                };
                Ok(PromptReply::Finished(finished))
            }
        })
    }

    fn cancel_session(&self, id: SessionId) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move {
            notify(
                &self.peer,
                "session/cancel",
                &json!({"sessionId": id.as_str()}),
            )
            .await
        })
    }

    fn set_config(
        &self,
        params: SetConfigParams,
    ) -> BoxFuture<'_, Result<Vec<ConfigOption>, ClientError>> {
        Box::pin(async move {
            if self.v2() {
                let _: crate::domain::SetConfigResult =
                    send(&self.peer, "session/set_config_option", &params).await?;
                // v2 reports the new list via `config_option_update`.
                Ok(Vec::new())
            } else {
                let cell = self.v1_codec(&params.session_id);
                let routed = {
                    let codec = cell.codec.lock().expect("codec poisoned");
                    codec.set_config_to_v1(&params)
                };
                match routed {
                    V1SetConfig::SetMode(p) => {
                        let _: v1::SessionSetModeResult =
                            send(&self.peer, "session/set_mode", &p).await?;
                        Ok(Vec::new())
                    }
                    V1SetConfig::SetConfig(p) => {
                        let result: v1::SessionSetConfigOptionResult =
                            send(&self.peer, "session/set_config_option", &p).await?;
                        let codec = cell.codec.lock().expect("codec poisoned");
                        let options = result
                            .config_options
                            .iter()
                            .map(|o| codec.config_option_to_domain(o))
                            .collect();
                        Ok(options)
                    }
                    V1SetConfig::Unsupported(raw) => Err(ClientError::Protocol(format!(
                        "config value {raw} has no v1 representation"
                    ))),
                }
            }
        })
    }

    fn cancel_tool_call(
        &self,
        session: SessionId,
        tool_call: ToolCallId,
    ) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move {
            if self.v2() {
                notify(
                    &self.peer,
                    "session/tool_call_cancel",
                    &json!({"sessionId": session.as_str(), "toolCallId": tool_call.as_str()}),
                )
                .await
            } else {
                Err(ClientError::Unsupported(
                    "tool-call cancellation is v2-only".to_string(),
                ))
            }
        })
    }

    fn cancel_terminal(
        &self,
        session: SessionId,
        terminal: TerminalId,
    ) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move {
            if self.v2() {
                notify(
                    &self.peer,
                    "session/terminal_cancel",
                    &json!({"sessionId": session.as_str(), "terminalId": terminal.as_str()}),
                )
                .await
            } else {
                Err(ClientError::Unsupported(
                    "terminal cancellation is v2-only".to_string(),
                ))
            }
        })
    }

    fn authenticate(&self, method: AuthMethodId) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move {
            if self.v2() {
                let p = v2::AuthLoginParams {
                    method_id: method,
                    meta: None,
                    extra: BTreeMap::new(),
                };
                let _: Value = self
                    .peer
                    .request_untyped(
                        "auth/login",
                        Some(
                            serde_json::to_value(p)
                                .map_err(|e| ClientError::Protocol(e.to_string()))?,
                        ),
                    )
                    .await
                    .map_err(lane_error)?;
            } else {
                let p = v1::AuthenticateParams::new(method.as_str());
                let _: v1::AuthenticateResult = send(&self.peer, "authenticate", &p).await?;
            }
            Ok(())
        })
    }

    fn logout(&self, method: Option<AuthMethodId>) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move {
            if self.v2() {
                let params = method.map_or_else(|| json!({}), |m| json!({"methodId": m.as_str()}));
                let _: Value = self
                    .peer
                    .request_untyped("auth/logout", Some(params))
                    .await
                    .map_err(lane_error)?;
            } else {
                let _: v1::LogoutResult =
                    send(&self.peer, "logout", &v1::LogoutParams::new()).await?;
            }
            Ok(())
        })
    }

    fn ext_request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> BoxFuture<'_, Result<Value, ClientError>> {
        Box::pin(async move {
            self.peer
                .request_untyped(method.as_str(), params)
                .await
                .map_err(lane_error)
        })
    }

    fn ext_notify(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(async move {
            self.peer
                .notify(method.as_str(), &params.unwrap_or(Value::Null))
                .await
                .map_err(ClientError::from)
        })
    }

    fn legacy_request(
        &self,
        method: &str,
        params: Value,
    ) -> BoxFuture<'_, Result<Value, ClientError>> {
        let method = method.to_owned();
        Box::pin(async move {
            self.peer
                .request_untyped(&method, Some(params))
                .await
                .map_err(lane_error)
        })
    }

    fn shutdown(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            self.peer.close();
        })
    }
}

/// The agent→client [`Handler`]: decodes wire traffic for the negotiated
/// version and dispatches into [`AnyAgentHooks`].
pub struct WireHooks {
    version: Arc<SharedVersion>,
    sessions: SessionRegistry,
    hooks: AnyAgentHooks,
}

impl WireHooks {
    /// A handler bound to the negotiated-version cell, registry, hooks.
    pub(crate) const fn new(
        version: Arc<SharedVersion>,
        sessions: SessionRegistry,
        hooks: AnyAgentHooks,
    ) -> Self {
        Self {
            version,
            sessions,
            hooks,
        }
    }

    fn v2(&self) -> bool {
        matches!(self.version.get(), Protocol::V2)
    }

    fn params(req: &Request) -> Result<Value, Response> {
        req.params.clone().ok_or_else(|| {
            Response::error(
                Some(req.id.clone()),
                RpcError::invalid_params("missing params"),
            )
        })
    }

    fn invalid(id: crate::rpc::RequestId, e: impl fmt::Display) -> Response {
        Response::error(
            Some(id),
            RpcError::invalid_params(format!("invalid params: {e}")),
        )
    }

    async fn on_request(&self, req: &Request) -> Response {
        let id = req.id.clone();
        match req.method.as_str() {
            "session/request_permission" => self.request_permission(id, req).await,
            "elicitation/create" => self.elicitation_create(id, req).await,
            "fs/read_text_file" => {
                self.lane_call(id, req, |p: v1::ReadTextFileParams| self.hooks.fs_read(p))
                    .await
            }
            "fs/write_text_file" => {
                self.lane_call(id, req, |p: v1::WriteTextFileParams| self.hooks.fs_write(p))
                    .await
            }
            "terminal/create" => {
                self.lane_call(id, req, |p: v1::TerminalCreateParams| {
                    self.hooks.terminal_create(p)
                })
                .await
            }
            "terminal/output" => {
                self.lane_call(id, req, |p: v1::TerminalOutputParams| {
                    self.hooks.terminal_output(p)
                })
                .await
            }
            "terminal/kill" => {
                self.lane_call(id, req, |p: v1::TerminalKillParams| {
                    self.hooks.terminal_kill(p)
                })
                .await
            }
            "terminal/wait_for_exit" => {
                self.lane_call(id, req, |p: v1::TerminalWaitForExitParams| {
                    self.hooks.terminal_wait_for_exit(p)
                })
                .await
            }
            "terminal/release" => {
                self.lane_call(id, req, |p: v1::TerminalReleaseParams| {
                    self.hooks.terminal_release(p)
                })
                .await
            }
            method if method.starts_with('_') => {
                let Ok(ext) = ExtMethod::try_new(method) else {
                    return Response::error(Some(id), RpcError::method_not_found(method));
                };
                match self.hooks.ext_request(ext, req.params.clone()).await {
                    Ok(value) => Response::success(id, value),
                    Err(e) => Response::error(Some(id), e),
                }
            }
            method => Response::error(Some(id), RpcError::method_not_found(method)),
        }
    }

    /// `session/request_permission` on either wire version.
    async fn request_permission(&self, id: crate::rpc::RequestId, req: &Request) -> Response {
        if self.v2() {
            let Ok(params) = Self::params(req).and_then(|v| {
                serde_json::from_value::<PermissionRequest>(v)
                    .map_err(|e| Self::invalid(id.clone(), e))
            }) else {
                return Self::invalid(id, "malformed");
            };
            match self.hooks.request_permission(params).await {
                Ok(outcome) => {
                    // Custom/unknown outcomes degrade to cancelled.
                    let outcome = match outcome {
                        PermissionOutcome::Selected(_) | PermissionOutcome::Cancelled(_) => outcome,
                        PermissionOutcome::Custom(..) | PermissionOutcome::Unknown(..) => {
                            PermissionOutcome::Cancelled(crate::domain::CancelledOutcome {
                                meta: None,
                            })
                        }
                    };
                    respond_ok(id, json!({"outcome": outcome}))
                }
                Err(e) => Response::error(Some(id), e),
            }
        } else {
            let Ok(p) = Self::params(req).and_then(|v| {
                serde_json::from_value::<v1::RequestPermissionParams>(v)
                    .map_err(|e| Self::invalid(id.clone(), e))
            }) else {
                return Self::invalid(id, "malformed");
            };
            let cell = self
                .sessions
                .get_or_create(SessionId::from_string(&p.session_id));
            let request = {
                let codec = cell.codec.lock().expect("codec poisoned");
                codec.permission_to_domain(&p)
            };
            match self.hooks.request_permission(request).await {
                Ok(outcome) => {
                    let result = {
                        let codec = cell.codec.lock().expect("codec poisoned");
                        codec.outcome_to_v1(&outcome)
                    };
                    respond_ok(
                        id,
                        &v1::RequestPermissionResult {
                            outcome: result,
                            meta: None,
                        },
                    )
                }
                Err(e) => Response::error(Some(id), e),
            }
        }
    }

    /// `elicitation/create` on either wire version.
    async fn elicitation_create(&self, id: crate::rpc::RequestId, req: &Request) -> Response {
        if self.v2() {
            let Ok(params) = Self::params(req).and_then(|v| {
                serde_json::from_value::<crate::domain::ElicitCreate>(v)
                    .map_err(|e| Self::invalid(id.clone(), e))
            }) else {
                return Self::invalid(id, "malformed");
            };
            match self.hooks.elicit(params).await {
                Ok(result) => respond_ok(id, &result),
                Err(e) => Response::error(Some(id), e),
            }
        } else {
            let Ok(p) = Self::params(req).and_then(|v| {
                serde_json::from_value::<v1::ElicitationCreateParams>(v)
                    .map_err(|e| Self::invalid(id.clone(), e))
            }) else {
                return Self::invalid(id, "malformed");
            };
            let create = {
                let codec = V1Codec::new();
                codec.elicit_to_domain(&p)
            };
            match self.hooks.elicit(create).await {
                Ok(result) => {
                    let codec = V1Codec::new();
                    respond_ok(id, codec.elicit_result_to_v1(&result))
                }
                Err(e) => Response::error(Some(id), e),
            }
        }
    }

    /// A v1 client-lane request (`fs/*`, `terminal/*`): decode `P`, run
    /// the hook, encode `R`.
    async fn lane_call<P, R, F, Fut>(
        &self,
        id: crate::rpc::RequestId,
        req: &Request,
        call: F,
    ) -> Response
    where
        P: DeserializeOwned,
        R: Serialize,
        F: FnOnce(P) -> Fut,
        Fut: Future<Output = Result<R, RpcError>>,
    {
        let Ok(p) = Self::params(req)
            .and_then(|v| serde_json::from_value::<P>(v).map_err(|e| Self::invalid(id.clone(), e)))
        else {
            return Self::invalid(id, "malformed");
        };
        match call(p).await {
            Ok(r) => respond_ok(id, &r),
            Err(e) => Response::error(Some(id), e),
        }
    }

    async fn on_notification(&self, n: Notification) {
        match n.method.as_str() {
            "session/update" => {
                if self.v2() {
                    match n
                        .params
                        .clone()
                        .map(serde_json::from_value::<SessionNotification>)
                    {
                        Some(Ok(note)) => self.hooks.session_update(note).await,
                        Some(Err(e)) => {
                            warn!(error = %e, "malformed session/update notification");
                        }
                        None => warn!("session/update without params"),
                    }
                } else {
                    match n
                        .params
                        .clone()
                        .map(serde_json::from_value::<v1::SessionNotification>)
                    {
                        Some(Ok(note)) => {
                            let cell = self
                                .sessions
                                .get_or_create(SessionId::from_string(&note.session_id));
                            let update = {
                                let mut codec = cell.codec.lock().expect("codec poisoned");
                                codec.update_from_v1(note.update)
                            };
                            let notification = SessionNotification {
                                session_id: cell.id.clone(),
                                update,
                                meta: note.meta,
                                extra: note.extra,
                            };
                            self.hooks.session_update(notification).await;
                        }
                        Some(Err(e)) => {
                            warn!(error = %e, "malformed v1 session/update notification");
                        }
                        None => warn!("session/update without params"),
                    }
                }
            }
            "elicitation/complete" => {
                if self.v2() {
                    if let Some(Ok(p)) = n
                        .params
                        .clone()
                        .map(serde_json::from_value::<ElicitComplete>)
                    {
                        self.hooks.elicit_complete(p).await;
                    }
                } else if let Some(Ok(p)) = n
                    .params
                    .clone()
                    .map(serde_json::from_value::<v1::ElicitationCompleteParams>)
                {
                    self.hooks
                        .elicit_complete(ElicitComplete {
                            elicitation_id: p.elicitation_id.into(),
                            meta: p.meta,
                            extra: BTreeMap::new(),
                        })
                        .await;
                }
            }
            _ => self.hooks.notification(n).await,
        }
    }
}

impl Handler for WireHooks {
    async fn request(&self, req: Request, _cx: RequestContext) -> Response {
        self.on_request(&req).await
    }

    async fn notification(&self, n: Notification) {
        self.on_notification(n).await;
    }
}

/// The connection driver the caller polls.
///
/// Wraps the engine future for a wire lane — and completes immediately for
/// the in-process lane, which needs no transport task. When it resolves,
/// every session's subscriber streams are closed so `PromptTurn::finish`
/// and `subscribe()` receivers end instead of hanging.
pub struct Connection {
    inner: BoxFuture<'static, ()>,
    sessions: SessionRegistry,
}

impl fmt::Debug for Connection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Connection").finish_non_exhaustive()
    }
}

impl Connection {
    /// Wrap the wire engine.
    pub(crate) fn wire(engine: crate::rpc::Connection, sessions: SessionRegistry) -> Self {
        Self {
            inner: Box::pin(async move {
                engine.await;
            }),
            sessions,
        }
    }

    /// The direct lane's driver: `drive` polls `ServerCore::drive` for
    /// the prompt scheduler and runs the core's `on_disconnect` teardown
    /// once the lane's shutdown channel closes.
    pub(crate) fn in_process(sessions: SessionRegistry, drive: BoxFuture<'static, ()>) -> Self {
        Self {
            inner: drive,
            sessions,
        }
    }
}

impl Future for Connection {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<()> {
        match self.inner.as_mut().poll(cx) {
            std::task::Poll::Ready(()) => {
                self.sessions.close_all();
                std::task::Poll::Ready(())
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

// ---- small helpers reused by session.rs ---------------------------------

/// The synthesized domain `state_update` injected when a v1 prompt result
/// ends the turn.
pub fn v1_turn_end_update(finished: &PromptFinished) -> crate::domain::StateUpdate {
    crate::domain::StateUpdate {
        state: crate::domain::Open::Known(crate::domain::ForegroundState::Idle),
        stop_reason: Some(finished.stop_reason.clone()),
        error: None,
        background: Vec::new(),
        meta: None,
        extra: BTreeMap::new(),
    }
}

/// Steering extension method (v1 fallback lane). Kept local until
/// `crate::ext` lands; wire-compatible with `ext::steering`.
pub const STEERING_METHOD: &str = "_session/steering";

/// Whether the agent advertised `_session/steering` in
/// `initialize._meta.steering.supported`.
pub fn steering_advertised(init: &InitResult) -> bool {
    init.meta
        .as_ref()
        .and_then(|m| m.get("steering"))
        .and_then(|s| s.get("supported"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Whether the setup result carries the codex `models` extra — the legacy
/// signal that `session/set_model` is the only model-switch method.
pub fn codex_models(extra: &BTreeMap<String, Value>) -> bool {
    extra.contains_key("models")
}
