//! Server-side error types.
//!
//! [`ServerError`] covers host-level failures (session setup, shutdown);
//! [`PromptError`] covers turn-level failures reported through the v2
//! `state_update` idle payload (or a v1 `stopReason`) rather than as an
//! RPC error; [`PromptRejected`] is handed back when
//! [`PendingPrompt::accept`](crate::server::PendingPrompt::accept) fails.

use thiserror::Error;

use crate::rpc::RpcError;

/// A failure in session setup, host lifecycle, or serving.
///
/// `Rpc` is forwarded to the peer verbatim as the request's error
/// response; the other variants collapse into an `internal error`
/// (-32603) when they have to cross the wire.
#[derive(Debug, Error)]
pub enum ServerError {
    /// A JSON-RPC error reported to the client unchanged.
    #[error(transparent)]
    Rpc(#[from] RpcError),
    /// The host failed a request that has no dedicated wire error.
    #[error("{0}")]
    Host(String),
    /// The server is shutting down; no new work is accepted.
    #[error("server is shutting down")]
    Shutdown,
}

impl ServerError {
    /// A host-side failure from a displayable message.
    #[must_use]
    pub fn host(message: impl Into<String>) -> Self {
        Self::Host(message.into())
    }
}

impl From<ServerError> for RpcError {
    fn from(error: ServerError) -> Self {
        match error {
            ServerError::Rpc(error) => error,
            ServerError::Host(message) => Self::internal_error(message),
            ServerError::Shutdown => Self::internal_error("server is shutting down"),
        }
    }
}

/// Why a turn ended badly.
///
/// A prompt handler returns this when the turn it accepted fails. The
/// framework never turns it into a JSON-RPC error for an already-accepted
/// prompt: it becomes a `state_update` `idle` with `stopReason`
/// (`cancelled` or `error`) on v2, and a `PromptResult` carrying the same
/// stop reason on v1.
#[derive(Debug, Error)]
pub enum PromptError {
    /// The turn was cancelled (`session/cancel` or `session/close`).
    ///
    /// Maps to `stopReason: "cancelled"`.
    #[error("turn cancelled")]
    Cancelled,
    /// The turn failed. Maps to `stopReason: "error"` with the wrapped
    /// [`RpcError`] serialized into the `state_update` `error` field.
    #[error(transparent)]
    Failed(RpcError),
}

impl PromptError {
    /// A failure with an `internal error` (-32603) payload.
    #[must_use]
    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed(RpcError::internal_error(message))
    }
}

impl From<RpcError> for PromptError {
    fn from(error: RpcError) -> Self {
        Self::Failed(error)
    }
}

/// A [`PendingPrompt`](crate::server::PendingPrompt) could not be
/// accepted — it was already cancelled or its session is gone.
///
/// The client has already been told the prompt did not run; the wrapped
/// [`RpcError`] is what was (or will be) delivered.
#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub struct PromptRejected(RpcError);

impl PromptRejected {
    /// Wrap the error already reported to the client.
    #[must_use]
    pub(crate) const fn new(error: RpcError) -> Self {
        Self(error)
    }

    /// The error delivered to the client.
    #[must_use]
    pub fn into_inner(self) -> RpcError {
        self.0
    }

    /// The error delivered to the client.
    #[must_use]
    pub const fn error(&self) -> &RpcError {
        &self.0
    }
}
