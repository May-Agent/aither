//! The [`SessionHost`] trait and its object-safe storage wrapper.
//!
//! A `SessionHost` is the agent's front door: the server calls it to
//! create, resume, list, delete, and fork sessions. Implementations
//! write async code with `impl Future + Send` signatures; the private
//! [`SessionHostImpl`] twin erases associated types for dynamic storage
//! inside [`AnySessionHost`], which is what [`ServerCore`] and the
//! builders hold.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::domain::{
    AgentCapabilities, AuthMethod, ResumeParams, SessionId, SessionListParams, SessionPage,
    SessionSetupParams, SessionUpdate,
};
use crate::rpc::RpcError;

use super::context::SessionContext;
use super::error::ServerError;
use super::session::{AnySession, Session};

/// Everything [`SessionHost::resume_session`] needs: the resume
/// parameters plus the [`ReplaySink`] the host fills with the session's
/// stored updates.
#[derive(Debug)]
pub struct ResumeRequest {
    /// Decoded `session/resume` (or `session/load`/`session/fork`) params.
    pub params: ResumeParams,
    /// Replay channel drained to the client before the resume response.
    pub replay: ReplaySink,
}

impl ResumeRequest {
    /// The session being resumed.
    #[must_use]
    pub const fn session_id(&self) -> &SessionId {
        &self.params.session_id
    }
}

/// The channel a host fills with replayed [`SessionUpdate`]s during
/// `resume_session`.
///
/// Every update sent is emitted to the client before the resume response
/// is written; on v2 the framework appends a `replay_complete` update
/// automatically once the sink is drained. The sink is consumed by the
/// request — cloning is impossible, so when `resume_session` returns the
/// stream is closed and drained.
#[derive(Debug)]
pub struct ReplaySink {
    tx: async_channel::Sender<SessionUpdate>,
}

impl ReplaySink {
    /// A sink plus the receiver [`ServerCore`] drains.
    pub(crate) fn channel() -> (Self, async_channel::Receiver<SessionUpdate>) {
        let (tx, rx) = async_channel::unbounded();
        (Self { tx }, rx)
    }

    /// Queue `update` for replay to the client.
    ///
    /// # Errors
    ///
    /// Returns the update (boxed — it is large) when the server stopped
    /// draining (session resume abandoned).
    pub fn send(&self, update: SessionUpdate) -> Result<(), Box<SessionUpdate>> {
        self.tx
            .try_send(update)
            .map_err(|e| Box::new(e.into_inner()))
    }
}

/// The agent implementation an [`crate::server::Server`] serves.
///
/// Only [`new_session`](SessionHost::new_session) and [`Session::prompt`]
/// are mandatory; every other method has a default that either reports
/// `method not found` or is a no-op. Advertised capabilities come from
/// [`capabilities`](SessionHost::capabilities) — it must reflect the
/// methods actually overridden; returning wrong capabilities is user
/// error (a `session/fork` request on a host that did not advertise
/// `fork` is rejected with -32601 before reaching `fork_session`).
pub trait SessionHost: Send + Sync + 'static {
    /// The session type this host creates.
    type Session: Session;

    /// The capabilities advertised in `initialize`.
    ///
    /// Defaults to a minimal session capability set: prompting plus
    /// `session/list` and `session/close` support. A host that supports
    /// resume (required for the v2 baseline), forking, deletion, MCP, or
    /// prompt-content kinds beyond text MUST override this so the
    /// client learns the truth.
    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities {
            session: Some(crate::domain::SessionCaps {
                prompt: Some(crate::domain::PromptCaps::default()),
                mcp: Some(crate::domain::McpCaps {
                    stdio: Some(crate::domain::EmptyObject),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Create a fresh session.
    ///
    /// `cx.session_id()` carries the id the server assigned; the result's
    /// `config_options`/`available_commands` come from the returned
    /// [`Session`] itself.
    fn new_session(
        &self,
        params: SessionSetupParams,
        cx: &SessionContext,
    ) -> impl Future<Output = Result<Self::Session, ServerError>> + Send;

    /// Resume an existing session.
    ///
    /// `req.replay` is the [`ReplaySink`] to fill with the session's
    /// stored updates before returning. The default reports
    /// `method not found` — a host claiming the v2 baseline MUST
    /// override this (v2 requires baseline resume support), even though
    /// the method stays dispatchable without the capability flag.
    fn resume_session(
        &self,
        req: ResumeRequest,
        cx: &SessionContext,
    ) -> impl Future<Output = Result<Self::Session, ServerError>> + Send {
        let _ = (req, cx);
        std::future::ready(Err(ServerError::Rpc(RpcError::method_not_found(
            "session/resume",
        ))))
    }

    /// List sessions the host can resume.
    ///
    /// The default returns an empty page.
    fn list_sessions(
        &self,
        params: SessionListParams,
    ) -> impl Future<Output = Result<SessionPage, RpcError>> + Send {
        let _ = params;
        std::future::ready(Ok(SessionPage::default()))
    }

    /// Permanently delete a stored session.
    ///
    /// The default reports `method not found`. Advertise
    /// `capabilities.session.delete` only when overriding.
    fn delete_session(
        &self,
        session_id: &SessionId,
    ) -> impl Future<Output = Result<(), RpcError>> + Send {
        let _ = session_id;
        std::future::ready(Err(RpcError::method_not_found("session/delete")))
    }

    /// Fork an existing session into a new one.
    ///
    /// `req.replay` may be used to emit the forked history. The default
    /// reports `method not found`; advertise `capabilities.session.fork`
    /// only when overriding (ungated requests are rejected first).
    fn fork_session(
        &self,
        req: ResumeRequest,
        cx: &SessionContext,
    ) -> impl Future<Output = Result<Self::Session, ServerError>> + Send {
        let _ = (req, cx);
        std::future::ready(Err(ServerError::Rpc(RpcError::method_not_found(
            "session/fork",
        ))))
    }

    /// Called once when the server starts serving.
    ///
    /// # Errors
    ///
    /// A failure aborts the connection before any request is handled.
    fn start(&self) -> impl Future<Output = Result<(), ServerError>> + Send {
        std::future::ready(Ok(()))
    }

    /// Called once after the transport ends and every session is closed.
    fn shutdown(&self) -> impl Future<Output = ()> + Send {
        std::future::ready(())
    }
}

/// Object-safe twin of [`SessionHost`] (private, boxing).
trait SessionHostImpl: Send + Sync + 'static {
    fn capabilities(&self) -> AgentCapabilities;
    fn new_session(
        &self,
        params: SessionSetupParams,
        cx: SessionContext,
    ) -> Pin<Box<dyn Future<Output = Result<AnySession, ServerError>> + Send + '_>>;
    fn resume_session(
        &self,
        req: ResumeRequest,
        cx: SessionContext,
    ) -> Pin<Box<dyn Future<Output = Result<AnySession, ServerError>> + Send + '_>>;
    fn list_sessions(
        &self,
        params: SessionListParams,
    ) -> Pin<Box<dyn Future<Output = Result<SessionPage, RpcError>> + Send + '_>>;
    fn delete_session(
        &self,
        session_id: SessionId,
    ) -> Pin<Box<dyn Future<Output = Result<(), RpcError>> + Send + '_>>;
    fn fork_session(
        &self,
        req: ResumeRequest,
        cx: SessionContext,
    ) -> Pin<Box<dyn Future<Output = Result<AnySession, ServerError>> + Send + '_>>;
    fn start(&self) -> Pin<Box<dyn Future<Output = Result<(), ServerError>> + Send + '_>>;
    fn shutdown(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

impl<H: SessionHost> SessionHostImpl for H {
    fn capabilities(&self) -> AgentCapabilities {
        SessionHost::capabilities(self)
    }

    fn new_session(
        &self,
        params: SessionSetupParams,
        cx: SessionContext,
    ) -> Pin<Box<dyn Future<Output = Result<AnySession, ServerError>> + Send + '_>> {
        Box::pin(async move {
            let session = SessionHost::new_session(self, params, &cx).await?;
            Ok(AnySession::new(session))
        })
    }

    fn resume_session(
        &self,
        req: ResumeRequest,
        cx: SessionContext,
    ) -> Pin<Box<dyn Future<Output = Result<AnySession, ServerError>> + Send + '_>> {
        Box::pin(async move {
            let session = SessionHost::resume_session(self, req, &cx).await?;
            Ok(AnySession::new(session))
        })
    }

    fn list_sessions(
        &self,
        params: SessionListParams,
    ) -> Pin<Box<dyn Future<Output = Result<SessionPage, RpcError>> + Send + '_>> {
        Box::pin(async move { SessionHost::list_sessions(self, params).await })
    }

    fn delete_session(
        &self,
        session_id: SessionId,
    ) -> Pin<Box<dyn Future<Output = Result<(), RpcError>> + Send + '_>> {
        Box::pin(async move { SessionHost::delete_session(self, &session_id).await })
    }

    fn fork_session(
        &self,
        req: ResumeRequest,
        cx: SessionContext,
    ) -> Pin<Box<dyn Future<Output = Result<AnySession, ServerError>> + Send + '_>> {
        Box::pin(async move {
            let session = SessionHost::fork_session(self, req, &cx).await?;
            Ok(AnySession::new(session))
        })
    }

    fn start(&self) -> Pin<Box<dyn Future<Output = Result<(), ServerError>> + Send + '_>> {
        Box::pin(SessionHost::start(self))
    }

    fn shutdown(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(SessionHost::shutdown(self))
    }
}

/// A type-erased [`SessionHost`] — the concrete handle the server stores.
///
/// Construct with [`AnySessionHost::new`]; all methods mirror
/// [`SessionHost`] with the associated [`Session`] erased to
/// [`AnySession`].
#[derive(Clone)]
pub struct AnySessionHost {
    inner: Arc<dyn SessionHostImpl>,
}

impl std::fmt::Debug for AnySessionHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnySessionHost").finish_non_exhaustive()
    }
}

impl AnySessionHost {
    /// Erase `host`.
    pub fn new<H: SessionHost>(host: H) -> Self {
        Self {
            inner: Arc::new(host),
        }
    }

    /// See [`SessionHost::capabilities`].
    pub(crate) fn capabilities(&self) -> AgentCapabilities {
        self.inner.capabilities()
    }

    /// See [`SessionHost::new_session`].
    pub(crate) async fn new_session(
        &self,
        params: SessionSetupParams,
        cx: &SessionContext,
    ) -> Result<AnySession, ServerError> {
        self.inner.new_session(params, cx.clone()).await
    }

    /// See [`SessionHost::resume_session`].
    pub(crate) async fn resume_session(
        &self,
        req: ResumeRequest,
        cx: &SessionContext,
    ) -> Result<AnySession, ServerError> {
        self.inner.resume_session(req, cx.clone()).await
    }

    /// See [`SessionHost::list_sessions`].
    pub(crate) async fn list_sessions(
        &self,
        params: SessionListParams,
    ) -> Result<SessionPage, RpcError> {
        self.inner.list_sessions(params).await
    }

    /// See [`SessionHost::delete_session`].
    pub(crate) async fn delete_session(&self, session_id: &SessionId) -> Result<(), RpcError> {
        self.inner.delete_session(session_id.clone()).await
    }

    /// See [`SessionHost::fork_session`].
    pub(crate) async fn fork_session(
        &self,
        req: ResumeRequest,
        cx: &SessionContext,
    ) -> Result<AnySession, ServerError> {
        self.inner.fork_session(req, cx.clone()).await
    }

    /// See [`SessionHost::start`].
    pub(crate) async fn start(&self) -> Result<(), ServerError> {
        self.inner.start().await
    }

    /// See [`SessionHost::shutdown`].
    pub(crate) async fn shutdown(&self) {
        self.inner.shutdown().await;
    }
}

/// How a client authenticates: `auth/login` (`authenticate` on v1) or
/// `auth/logout` (`logout` on v1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthRequest {
    /// `auth/login` / `authenticate`: run this method.
    Login(crate::domain::AuthMethodId),
    /// `auth/logout` / `logout`: end this method's session, or the
    /// current login when `None`.
    Logout(Option<crate::domain::AuthMethodId>),
}

/// The handler inside [`AuthConfig`]: an [`AuthRequest`] in, a boxed
/// `Result<(), RpcError>` future out.
pub type AuthHandler = Arc<
    dyn Fn(AuthRequest) -> Pin<Box<dyn Future<Output = Result<(), RpcError>> + Send>> + Send + Sync,
>;

/// The authentication surface a server advertises and delegates to.
///
/// `methods` are returned in the `initialize` response; `handler` runs
/// the flows. An empty `methods` means `auth/login`/`auth/logout`
/// (`authenticate`/`logout` on v1) are reported `method not found`
/// (-32601), per the dispatch rules.
pub struct AuthConfig {
    /// Methods advertised at `initialize`.
    pub methods: Vec<AuthMethod>,
    /// Runs login/logout requests.
    pub handler: AuthHandler,
}

impl std::fmt::Debug for AuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthConfig")
            .field("methods", &self.methods)
            .finish_non_exhaustive()
    }
}

impl AuthConfig {
    /// An `AuthConfig` whose handler is `f`.
    ///
    /// `f` receives [`AuthRequest::Login`] or [`AuthRequest::Logout`]
    /// and returns the outcome as a future.
    pub fn new<F, Fut>(methods: Vec<AuthMethod>, f: F) -> Self
    where
        F: Fn(AuthRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), RpcError>> + Send + 'static,
    {
        Self {
            methods,
            handler: Arc::new(move |req| Box::pin(f(req))),
        }
    }
}
