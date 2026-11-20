//! Client-side handler traits and the assembled [`AgentHooks`] table.
//!
//! The user registers handlers on [`ClientBuilder`](super::ClientBuilder);
//! they are assembled into one [`AnyAgentHooks`] object that the wire
//! handler ([`super::connection`]) or — on the direct lane — the server's
//! `SessionContext` calls.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::Serialize;
use serde_json::Value;

use crate::domain::{
    ElicitComplete, ElicitCreate, ElicitResult, ExtMethod, PermissionOutcome, PermissionRequest,
    SessionNotification,
};
use crate::rpc::RpcError;
use crate::wire::v1;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Answers `session/request_permission`.
///
/// Returning [`PermissionOutcome::Cancelled`] (or an unknown outcome,
/// which degrades to cancelled) declines the request. When the agent
/// aborts the request mid-flight (`$/cancel_request`) the engine drops
/// this future and answers `-32800` itself.
pub trait PermissionHandler: Send + Sync + 'static {
    /// Decide the outcome of `request`.
    fn request(&self, request: PermissionRequest)
    -> impl Future<Output = PermissionOutcome> + Send;
}

impl<F, Fut> PermissionHandler for F
where
    F: Fn(PermissionRequest) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = PermissionOutcome> + Send,
{
    fn request(
        &self,
        request: PermissionRequest,
    ) -> impl Future<Output = PermissionOutcome> + Send {
        (self)(request)
    }
}

/// Object-safe twin of [`PermissionHandler`].
pub trait PermissionHandlerImpl: Send + Sync {
    fn request(&self, request: PermissionRequest) -> BoxFuture<'_, PermissionOutcome>;
}

impl<T: PermissionHandler> PermissionHandlerImpl for T {
    fn request(&self, request: PermissionRequest) -> BoxFuture<'_, PermissionOutcome> {
        Box::pin(PermissionHandler::request(self, request))
    }
}

/// Type-erased [`PermissionHandler`].
#[derive(Clone)]
pub struct AnyPermissionHandler(Arc<dyn PermissionHandlerImpl>);

impl AnyPermissionHandler {
    pub(crate) fn new(handler: impl PermissionHandler) -> Self {
        Self(Arc::new(handler))
    }
    async fn request(&self, request: PermissionRequest) -> PermissionOutcome {
        self.0.request(request).await
    }
}

/// Answers `elicitation/create` (and sees `elicitation/complete`).
pub trait ElicitationHandler: Send + Sync + 'static {
    /// Resolve an elicitation request.
    fn create(&self, create: ElicitCreate) -> impl Future<Output = ElicitResult> + Send;
    /// A URL-mode elicitation finished. Default: ignore.
    fn complete(&self, _complete: ElicitComplete) -> impl Future<Output = ()> + Send {
        std::future::ready(())
    }
}

impl<F, Fut> ElicitationHandler for F
where
    F: Fn(ElicitCreate) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ElicitResult> + Send,
{
    fn create(&self, create: ElicitCreate) -> impl Future<Output = ElicitResult> + Send {
        (self)(create)
    }
}

/// Object-safe twin of [`ElicitationHandler`].
pub trait ElicitationHandlerImpl: Send + Sync {
    fn create(&self, create: ElicitCreate) -> BoxFuture<'_, ElicitResult>;
    fn complete(&self, complete: ElicitComplete) -> BoxFuture<'_, ()>;
}

impl<T: ElicitationHandler> ElicitationHandlerImpl for T {
    fn create(&self, create: ElicitCreate) -> BoxFuture<'_, ElicitResult> {
        Box::pin(ElicitationHandler::create(self, create))
    }
    fn complete(&self, complete: ElicitComplete) -> BoxFuture<'_, ()> {
        Box::pin(ElicitationHandler::complete(self, complete))
    }
}

/// Type-erased [`ElicitationHandler`].
#[derive(Clone)]
pub struct AnyElicitationHandler(Arc<dyn ElicitationHandlerImpl>);

impl AnyElicitationHandler {
    pub(crate) fn new(handler: impl ElicitationHandler) -> Self {
        Self(Arc::new(handler))
    }
    async fn create(&self, create: ElicitCreate) -> ElicitResult {
        self.0.create(create).await
    }
    async fn complete(&self, complete: ElicitComplete) {
        self.0.complete(complete).await;
    }
}

/// v1-only `fs/*` client lane.
pub trait FsHandler: Send + Sync + 'static {
    /// `fs/read_text_file`.
    fn read(
        &self,
        params: v1::ReadTextFileParams,
    ) -> impl Future<Output = Result<v1::ReadTextFileResult, RpcError>> + Send;
    /// `fs/write_text_file`.
    fn write(
        &self,
        params: v1::WriteTextFileParams,
    ) -> impl Future<Output = Result<v1::WriteTextFileResult, RpcError>> + Send;
}

/// Object-safe twin of [`FsHandler`].
pub trait FsHandlerImpl: Send + Sync {
    fn read(
        &self,
        params: v1::ReadTextFileParams,
    ) -> BoxFuture<'_, Result<v1::ReadTextFileResult, RpcError>>;
    fn write(
        &self,
        params: v1::WriteTextFileParams,
    ) -> BoxFuture<'_, Result<v1::WriteTextFileResult, RpcError>>;
}

impl<T: FsHandler> FsHandlerImpl for T {
    fn read(
        &self,
        params: v1::ReadTextFileParams,
    ) -> BoxFuture<'_, Result<v1::ReadTextFileResult, RpcError>> {
        Box::pin(FsHandler::read(self, params))
    }
    fn write(
        &self,
        params: v1::WriteTextFileParams,
    ) -> BoxFuture<'_, Result<v1::WriteTextFileResult, RpcError>> {
        Box::pin(FsHandler::write(self, params))
    }
}

/// Type-erased [`FsHandler`].
#[derive(Clone)]
pub struct AnyFsHandler(Arc<dyn FsHandlerImpl>);

impl AnyFsHandler {
    pub(crate) fn new(handler: impl FsHandler) -> Self {
        Self(Arc::new(handler))
    }
    async fn read(&self, p: v1::ReadTextFileParams) -> Result<v1::ReadTextFileResult, RpcError> {
        self.0.read(p).await
    }
    async fn write(&self, p: v1::WriteTextFileParams) -> Result<v1::WriteTextFileResult, RpcError> {
        self.0.write(p).await
    }
}

/// v1-only `terminal/*` client lane.
pub trait V1Terminal: Send + Sync + 'static {
    /// `terminal/create`.
    fn create(
        &self,
        params: v1::TerminalCreateParams,
    ) -> impl Future<Output = Result<v1::TerminalCreateResult, RpcError>> + Send;
    /// `terminal/output`.
    fn output(
        &self,
        params: v1::TerminalOutputParams,
    ) -> impl Future<Output = Result<v1::TerminalOutputResult, RpcError>> + Send;
    /// `terminal/kill`.
    fn kill(
        &self,
        params: v1::TerminalKillParams,
    ) -> impl Future<Output = Result<v1::TerminalKillResult, RpcError>> + Send;
    /// `terminal/wait_for_exit`.
    fn wait_for_exit(
        &self,
        params: v1::TerminalWaitForExitParams,
    ) -> impl Future<Output = Result<v1::TerminalExitStatus, RpcError>> + Send;
    /// `terminal/release`.
    fn release(
        &self,
        params: v1::TerminalReleaseParams,
    ) -> impl Future<Output = Result<v1::TerminalReleaseResult, RpcError>> + Send;
}

/// Object-safe twin of [`V1Terminal`].
pub trait V1TerminalImpl: Send + Sync {
    fn create(
        &self,
        p: v1::TerminalCreateParams,
    ) -> BoxFuture<'_, Result<v1::TerminalCreateResult, RpcError>>;
    fn output(
        &self,
        p: v1::TerminalOutputParams,
    ) -> BoxFuture<'_, Result<v1::TerminalOutputResult, RpcError>>;
    fn kill(
        &self,
        p: v1::TerminalKillParams,
    ) -> BoxFuture<'_, Result<v1::TerminalKillResult, RpcError>>;
    fn wait_for_exit(
        &self,
        p: v1::TerminalWaitForExitParams,
    ) -> BoxFuture<'_, Result<v1::TerminalExitStatus, RpcError>>;
    fn release(
        &self,
        p: v1::TerminalReleaseParams,
    ) -> BoxFuture<'_, Result<v1::TerminalReleaseResult, RpcError>>;
}

impl<T: V1Terminal> V1TerminalImpl for T {
    fn create(
        &self,
        p: v1::TerminalCreateParams,
    ) -> BoxFuture<'_, Result<v1::TerminalCreateResult, RpcError>> {
        Box::pin(V1Terminal::create(self, p))
    }
    fn output(
        &self,
        p: v1::TerminalOutputParams,
    ) -> BoxFuture<'_, Result<v1::TerminalOutputResult, RpcError>> {
        Box::pin(V1Terminal::output(self, p))
    }
    fn kill(
        &self,
        p: v1::TerminalKillParams,
    ) -> BoxFuture<'_, Result<v1::TerminalKillResult, RpcError>> {
        Box::pin(V1Terminal::kill(self, p))
    }
    fn wait_for_exit(
        &self,
        p: v1::TerminalWaitForExitParams,
    ) -> BoxFuture<'_, Result<v1::TerminalExitStatus, RpcError>> {
        Box::pin(V1Terminal::wait_for_exit(self, p))
    }
    fn release(
        &self,
        p: v1::TerminalReleaseParams,
    ) -> BoxFuture<'_, Result<v1::TerminalReleaseResult, RpcError>> {
        Box::pin(V1Terminal::release(self, p))
    }
}

/// Type-erased [`V1Terminal`].
#[derive(Clone)]
pub struct AnyV1Terminal(Arc<dyn V1TerminalImpl>);

impl AnyV1Terminal {
    pub(crate) fn new(handler: impl V1Terminal) -> Self {
        Self(Arc::new(handler))
    }
}

/// Answers `_`-prefixed extension requests.
pub trait ExtRequestHandler: Send + Sync + 'static {
    /// Handle `method` with raw `params`; the returned value becomes the result.
    fn request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> impl Future<Output = Result<Value, RpcError>> + Send;
}

impl<F, Fut> ExtRequestHandler for F
where
    F: Fn(ExtMethod, Option<Value>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Value, RpcError>> + Send,
{
    fn request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> impl Future<Output = Result<Value, RpcError>> + Send {
        (self)(method, params)
    }
}

/// Object-safe twin of [`ExtRequestHandler`].
pub trait ExtRequestHandlerImpl: Send + Sync {
    fn request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> BoxFuture<'_, Result<Value, RpcError>>;
}

impl<T: ExtRequestHandler> ExtRequestHandlerImpl for T {
    fn request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> BoxFuture<'_, Result<Value, RpcError>> {
        Box::pin(ExtRequestHandler::request(self, method, params))
    }
}

/// Type-erased [`ExtRequestHandler`].
#[derive(Clone)]
pub struct AnyExtRequestHandler(Arc<dyn ExtRequestHandlerImpl>);

impl AnyExtRequestHandler {
    pub(crate) fn new(handler: impl ExtRequestHandler) -> Self {
        Self(Arc::new(handler))
    }
}

/// Catch-all for agent→client notifications the crate does not model.
pub trait NotificationHandler: Send + Sync + 'static {
    /// Handle `notification`.
    fn notify(&self, notification: crate::rpc::Notification) -> impl Future<Output = ()> + Send;
}

impl<F, Fut> NotificationHandler for F
where
    F: Fn(crate::rpc::Notification) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send,
{
    fn notify(&self, notification: crate::rpc::Notification) -> impl Future<Output = ()> + Send {
        (self)(notification)
    }
}

/// Object-safe twin of [`NotificationHandler`].
pub trait NotificationHandlerImpl: Send + Sync {
    fn notify(&self, notification: crate::rpc::Notification) -> BoxFuture<'_, ()>;
}

impl<T: NotificationHandler> NotificationHandlerImpl for T {
    fn notify(&self, notification: crate::rpc::Notification) -> BoxFuture<'_, ()> {
        Box::pin(NotificationHandler::notify(self, notification))
    }
}

/// Type-erased [`NotificationHandler`].
#[derive(Clone)]
pub struct AnyNotificationHandler(Arc<dyn NotificationHandlerImpl>);

impl AnyNotificationHandler {
    pub(crate) fn new(handler: impl NotificationHandler) -> Self {
        Self(Arc::new(handler))
    }
}

/// The object the agent calls: `session/update` demux plus interactive
/// requests. On the wire a codec sits in front; on the direct lane the
/// server's `SessionContext` calls these methods directly — always
/// domain-typed, always v2 semantics (the `fs`/`terminal` lanes are not
/// reachable there).
pub trait AgentHooksImpl: Send + Sync {
    /// `session/update` notification (domain-decoded already).
    fn session_update(&self, notification: SessionNotification) -> BoxFuture<'_, ()>;
    /// `session/request_permission`.
    fn request_permission(
        &self,
        request: PermissionRequest,
    ) -> BoxFuture<'_, Result<PermissionOutcome, RpcError>>;
    /// `elicitation/create`.
    fn elicit(&self, create: ElicitCreate) -> BoxFuture<'_, Result<ElicitResult, RpcError>>;
    /// `elicitation/complete` notification.
    fn elicit_complete(&self, complete: ElicitComplete) -> BoxFuture<'_, ()>;
    /// `fs/read_text_file` (v1 only).
    fn fs_read(
        &self,
        params: v1::ReadTextFileParams,
    ) -> BoxFuture<'_, Result<v1::ReadTextFileResult, RpcError>>;
    /// `fs/write_text_file` (v1 only).
    fn fs_write(
        &self,
        params: v1::WriteTextFileParams,
    ) -> BoxFuture<'_, Result<v1::WriteTextFileResult, RpcError>>;
    /// `terminal/create` (v1 only).
    fn terminal_create(
        &self,
        params: v1::TerminalCreateParams,
    ) -> BoxFuture<'_, Result<v1::TerminalCreateResult, RpcError>>;
    /// `terminal/output` (v1 only).
    fn terminal_output(
        &self,
        params: v1::TerminalOutputParams,
    ) -> BoxFuture<'_, Result<v1::TerminalOutputResult, RpcError>>;
    /// `terminal/kill` (v1 only).
    fn terminal_kill(
        &self,
        params: v1::TerminalKillParams,
    ) -> BoxFuture<'_, Result<v1::TerminalKillResult, RpcError>>;
    /// `terminal/wait_for_exit` (v1 only).
    fn terminal_wait_for_exit(
        &self,
        params: v1::TerminalWaitForExitParams,
    ) -> BoxFuture<'_, Result<v1::TerminalExitStatus, RpcError>>;
    /// `terminal/release` (v1 only).
    fn terminal_release(
        &self,
        params: v1::TerminalReleaseParams,
    ) -> BoxFuture<'_, Result<v1::TerminalReleaseResult, RpcError>>;
    /// Any `_`-prefixed request.
    fn ext_request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> BoxFuture<'_, Result<Value, RpcError>>;
    /// Any unmodelled notification.
    fn notification(&self, notification: crate::rpc::Notification) -> BoxFuture<'_, ()>;
}

/// A cloneable agent-hooks trait object: the handler table the
/// builder assembles and both lanes share.
#[derive(Clone)]
pub struct AnyAgentHooks(Arc<dyn AgentHooksImpl>);

impl fmt::Debug for AnyAgentHooks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyAgentHooks").finish_non_exhaustive()
    }
}

impl AnyAgentHooks {
    /// Wrap `hooks`.
    pub(crate) fn new(hooks: impl AgentHooksImpl + 'static) -> Self {
        Self(Arc::new(hooks))
    }

    /// `session/update` notification.
    pub fn session_update(
        &self,
        notification: SessionNotification,
    ) -> impl Future<Output = ()> + Send + '_ {
        self.0.session_update(notification)
    }
    /// `session/request_permission`.
    pub fn request_permission(
        &self,
        request: PermissionRequest,
    ) -> impl Future<Output = Result<PermissionOutcome, RpcError>> + Send + '_ {
        self.0.request_permission(request)
    }
    /// `elicitation/create`.
    pub fn elicit(
        &self,
        create: ElicitCreate,
    ) -> impl Future<Output = Result<ElicitResult, RpcError>> + Send + '_ {
        self.0.elicit(create)
    }
    /// `elicitation/complete` notification.
    pub fn elicit_complete(
        &self,
        complete: ElicitComplete,
    ) -> impl Future<Output = ()> + Send + '_ {
        self.0.elicit_complete(complete)
    }
    /// `fs/read_text_file` (v1 only).
    pub fn fs_read(
        &self,
        params: v1::ReadTextFileParams,
    ) -> impl Future<Output = Result<v1::ReadTextFileResult, RpcError>> + Send + '_ {
        self.0.fs_read(params)
    }
    /// `fs/write_text_file` (v1 only).
    pub fn fs_write(
        &self,
        params: v1::WriteTextFileParams,
    ) -> impl Future<Output = Result<v1::WriteTextFileResult, RpcError>> + Send + '_ {
        self.0.fs_write(params)
    }
    /// `terminal/create` (v1 only).
    pub fn terminal_create(
        &self,
        params: v1::TerminalCreateParams,
    ) -> impl Future<Output = Result<v1::TerminalCreateResult, RpcError>> + Send + '_ {
        self.0.terminal_create(params)
    }
    /// `terminal/output` (v1 only).
    pub fn terminal_output(
        &self,
        params: v1::TerminalOutputParams,
    ) -> impl Future<Output = Result<v1::TerminalOutputResult, RpcError>> + Send + '_ {
        self.0.terminal_output(params)
    }
    /// `terminal/kill` (v1 only).
    pub fn terminal_kill(
        &self,
        params: v1::TerminalKillParams,
    ) -> impl Future<Output = Result<v1::TerminalKillResult, RpcError>> + Send + '_ {
        self.0.terminal_kill(params)
    }
    /// `terminal/wait_for_exit` (v1 only).
    pub fn terminal_wait_for_exit(
        &self,
        params: v1::TerminalWaitForExitParams,
    ) -> impl Future<Output = Result<v1::TerminalExitStatus, RpcError>> + Send + '_ {
        self.0.terminal_wait_for_exit(params)
    }
    /// `terminal/release` (v1 only).
    pub fn terminal_release(
        &self,
        params: v1::TerminalReleaseParams,
    ) -> impl Future<Output = Result<v1::TerminalReleaseResult, RpcError>> + Send + '_ {
        self.0.terminal_release(params)
    }
    /// `_`-prefixed extension request.
    pub fn ext_request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> impl Future<Output = Result<Value, RpcError>> + Send + '_ {
        self.0.ext_request(method, params)
    }
    /// Unmodelled notification catch-all.
    pub fn notification(
        &self,
        notification: crate::rpc::Notification,
    ) -> impl Future<Output = ()> + Send + '_ {
        self.0.notification(notification)
    }
}

/// The handler table the builder assembles: demux of `session/update`
/// into the [`SessionRegistry`](super::handle::SessionRegistry) plus the
/// user's optional handlers.
pub struct AssembledHooks {
    /// Session demux target.
    pub(crate) sessions: super::handle::SessionRegistry,
    /// User permission handler.
    pub(crate) permission: Option<AnyPermissionHandler>,
    /// User elicitation handler.
    pub(crate) elicitation: Option<AnyElicitationHandler>,
    /// v1 filesystem lane.
    pub(crate) fs: Option<AnyFsHandler>,
    /// v1 terminal lane.
    pub(crate) terminal: Option<AnyV1Terminal>,
    /// Extension request handler.
    pub(crate) ext: Option<AnyExtRequestHandler>,
    /// Notification catch-all.
    pub(crate) notification: Option<AnyNotificationHandler>,
}

impl AgentHooksImpl for AssembledHooks {
    fn session_update(&self, notification: SessionNotification) -> BoxFuture<'_, ()> {
        let sessions = self.sessions.clone();
        Box::pin(async move {
            sessions.apply(&notification);
        })
    }

    fn request_permission(
        &self,
        request: PermissionRequest,
    ) -> BoxFuture<'_, Result<PermissionOutcome, RpcError>> {
        Box::pin(async move {
            match &self.permission {
                Some(h) => Ok(h.request(request).await),
                None => Err(RpcError::method_not_found("session/request_permission")),
            }
        })
    }

    fn elicit(&self, create: ElicitCreate) -> BoxFuture<'_, Result<ElicitResult, RpcError>> {
        Box::pin(async move {
            match &self.elicitation {
                Some(h) => Ok(h.create(create).await),
                None => Err(RpcError::method_not_found("elicitation/create")),
            }
        })
    }

    fn elicit_complete(&self, complete: ElicitComplete) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            if let Some(h) = &self.elicitation {
                h.complete(complete).await;
            }
        })
    }

    fn fs_read(
        &self,
        params: v1::ReadTextFileParams,
    ) -> BoxFuture<'_, Result<v1::ReadTextFileResult, RpcError>> {
        Box::pin(async move {
            match &self.fs {
                Some(h) => h.read(params).await,
                None => Err(RpcError::method_not_found("fs/read_text_file")),
            }
        })
    }

    fn fs_write(
        &self,
        params: v1::WriteTextFileParams,
    ) -> BoxFuture<'_, Result<v1::WriteTextFileResult, RpcError>> {
        Box::pin(async move {
            match &self.fs {
                Some(h) => h.write(params).await,
                None => Err(RpcError::method_not_found("fs/write_text_file")),
            }
        })
    }

    fn terminal_create(
        &self,
        params: v1::TerminalCreateParams,
    ) -> BoxFuture<'_, Result<v1::TerminalCreateResult, RpcError>> {
        Box::pin(async move {
            match &self.terminal {
                Some(h) => h.0.create(params).await,
                None => Err(RpcError::method_not_found("terminal/create")),
            }
        })
    }

    fn terminal_output(
        &self,
        params: v1::TerminalOutputParams,
    ) -> BoxFuture<'_, Result<v1::TerminalOutputResult, RpcError>> {
        Box::pin(async move {
            match &self.terminal {
                Some(h) => h.0.output(params).await,
                None => Err(RpcError::method_not_found("terminal/output")),
            }
        })
    }

    fn terminal_kill(
        &self,
        params: v1::TerminalKillParams,
    ) -> BoxFuture<'_, Result<v1::TerminalKillResult, RpcError>> {
        Box::pin(async move {
            match &self.terminal {
                Some(h) => h.0.kill(params).await,
                None => Err(RpcError::method_not_found("terminal/kill")),
            }
        })
    }

    fn terminal_wait_for_exit(
        &self,
        params: v1::TerminalWaitForExitParams,
    ) -> BoxFuture<'_, Result<v1::TerminalExitStatus, RpcError>> {
        Box::pin(async move {
            match &self.terminal {
                Some(h) => h.0.wait_for_exit(params).await,
                None => Err(RpcError::method_not_found("terminal/wait_for_exit")),
            }
        })
    }

    fn terminal_release(
        &self,
        params: v1::TerminalReleaseParams,
    ) -> BoxFuture<'_, Result<v1::TerminalReleaseResult, RpcError>> {
        Box::pin(async move {
            match &self.terminal {
                Some(h) => h.0.release(params).await,
                None => Err(RpcError::method_not_found("terminal/release")),
            }
        })
    }

    fn ext_request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> BoxFuture<'_, Result<Value, RpcError>> {
        Box::pin(async move {
            match &self.ext {
                Some(h) => h.0.request(method.clone(), params).await,
                None => Err(RpcError::method_not_found(method.as_str())),
            }
        })
    }

    fn notification(&self, notification: crate::rpc::Notification) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            match &self.notification {
                Some(h) => h.0.notify(notification).await,
                None => {
                    tracing::debug!(method = %notification.method, "ignoring agent notification");
                }
            }
        })
    }
}

/// Serialize a result payload into a [`crate::rpc::Response`].
pub fn respond_ok(id: crate::rpc::RequestId, result: impl Serialize) -> crate::rpc::Response {
    match serde_json::to_value(result) {
        Ok(value) => crate::rpc::Response::success(id, value),
        Err(error) => crate::rpc::Response::error(
            Some(id),
            RpcError::internal_error(format!("failed to serialize response: {error}")),
        ),
    }
}
