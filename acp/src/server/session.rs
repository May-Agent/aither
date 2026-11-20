//! The [`Session`] trait and its object-safe storage wrapper.
//!
//! A `Session` is one live conversation the host created or resumed.
//! [`Session::prompt`] is the only required method — the
//! [`PendingPrompt`] handed in is consumed by
//! [`PendingPrompt::accept`](crate::server::PendingPrompt::accept)
//! (yielding a [`Turn`]) or
//! [`PendingPrompt::reject`](crate::server::PendingPrompt::reject).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::domain::ids::ConfigId;
use crate::domain::{AvailableCommand, ConfigOption, ConfigValue, StopReason};
use crate::rpc::RpcError;

use super::context::SessionContext;
use super::error::PromptError;
use super::prompt::PendingPrompt;

/// One live ACP session.
///
/// The server serializes `prompt` calls per session (FIFO); other
/// methods may be invoked concurrently with a running turn.
pub trait Session: Send + Sync + 'static {
    /// Configuration options this session exposes at setup, and the
    /// authoritative list whenever the set changes.
    ///
    /// Used to populate `session/new` (and `session/load`/
    /// `session/resume`/`session/fork`) responses.
    fn config_options(&self) -> Vec<ConfigOption> {
        Vec::new()
    }

    /// Slash commands available in this session.
    ///
    /// On v1 these are delivered as an `available_commands_update`
    /// notification (there is no response field for them).
    fn available_commands(&self) -> Vec<AvailableCommand> {
        Vec::new()
    }

    /// Set a configuration option (`session/set_config_option`, and
    /// `session/set_mode` on v1 — decoded as the config option whose
    /// category is `mode`).
    ///
    /// Returns the full, updated option set; the framework emits it as a
    /// `config_option_update`. The default reports `method not found`.
    fn set_config(
        &self,
        id: &ConfigId,
        value: ConfigValue,
        cx: &SessionContext,
    ) -> impl Future<Output = Result<Vec<ConfigOption>, RpcError>> + Send {
        let _ = (id, value, cx);
        std::future::ready(Err(RpcError::method_not_found("session/set_config_option")))
    }

    /// Run one prompt turn.
    ///
    /// `prompt` was inserted into the session's queue by the server; the
    /// implementation either [`accept`](PendingPrompt::accept)s it into
    /// a [`Turn`](crate::server::Turn) and drives it to a [`StopReason`],
    /// or [`reject`](PendingPrompt::reject)s it outright.
    ///
    /// Returning ends the turn: the framework emits the v2 `state_update`
    /// `idle` (or completes the v1 prompt response) with the stop reason.
    /// A [`PromptError`](crate::server::PromptError) maps to
    /// `stopReason: "cancelled"`/`"error"`.
    fn prompt(
        &self,
        prompt: PendingPrompt,
        cx: &SessionContext,
    ) -> impl Future<Output = Result<StopReason, PromptError>> + Send;

    /// Called when the session is torn down (`session/close`, transport
    /// end, or server shutdown). The default is a no-op.
    fn close(&self) -> impl Future<Output = ()> + Send {
        std::future::ready(())
    }
}

/// Object-safe twin of [`Session`] (private, boxing).
trait SessionImpl: Send + Sync + 'static {
    fn config_options(&self) -> Vec<ConfigOption>;
    fn available_commands(&self) -> Vec<AvailableCommand>;
    fn set_config(
        &self,
        id: ConfigId,
        value: ConfigValue,
        cx: SessionContext,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ConfigOption>, RpcError>> + Send + '_>>;
    fn prompt(
        &self,
        prompt: PendingPrompt,
        cx: SessionContext,
    ) -> Pin<Box<dyn Future<Output = Result<StopReason, PromptError>> + Send + '_>>;
    fn close(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

impl<S: Session> SessionImpl for S {
    fn config_options(&self) -> Vec<ConfigOption> {
        Session::config_options(self)
    }

    fn available_commands(&self) -> Vec<AvailableCommand> {
        Session::available_commands(self)
    }

    fn set_config(
        &self,
        id: ConfigId,
        value: ConfigValue,
        cx: SessionContext,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ConfigOption>, RpcError>> + Send + '_>> {
        Box::pin(async move { Session::set_config(self, &id, value, &cx).await })
    }

    fn prompt(
        &self,
        prompt: PendingPrompt,
        cx: SessionContext,
    ) -> Pin<Box<dyn Future<Output = Result<StopReason, PromptError>> + Send + '_>> {
        Box::pin(async move { Session::prompt(self, prompt, &cx).await })
    }

    fn close(&self) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(Session::close(self))
    }
}

/// A type-erased [`Session`] stored by
/// [`ServerCore`](crate::server::ServerCore).
///
/// Construct with [`AnySession::new`]; methods mirror [`Session`].
pub struct AnySession {
    inner: Arc<dyn SessionImpl>,
}

impl std::fmt::Debug for AnySession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnySession").finish_non_exhaustive()
    }
}

impl Clone for AnySession {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl AnySession {
    /// Erase `session`.
    pub fn new<S: Session>(session: S) -> Self {
        Self {
            inner: Arc::new(session),
        }
    }

    /// See [`Session::config_options`].
    pub(crate) fn config_options(&self) -> Vec<ConfigOption> {
        self.inner.config_options()
    }

    /// See [`Session::available_commands`].
    pub(crate) fn available_commands(&self) -> Vec<AvailableCommand> {
        self.inner.available_commands()
    }

    /// See [`Session::set_config`].
    pub(crate) async fn set_config(
        &self,
        id: &ConfigId,
        value: ConfigValue,
        cx: &SessionContext,
    ) -> Result<Vec<ConfigOption>, RpcError> {
        self.inner.set_config(id.clone(), value, cx.clone()).await
    }

    /// See [`Session::prompt`].
    pub(crate) async fn prompt(
        &self,
        prompt: PendingPrompt,
        cx: &SessionContext,
    ) -> Result<StopReason, PromptError> {
        self.inner.prompt(prompt, cx.clone()).await
    }

    /// See [`Session::close`].
    pub(crate) async fn close(&self) {
        self.inner.close().await;
    }
}
