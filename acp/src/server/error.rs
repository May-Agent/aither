//! Errors reported by the ACP server peer handle.

use aither_mcp::protocol::JsonRpcError;
use thiserror::Error;

/// Error returned by [`ClientPeer`](super::ClientPeer) methods.
///
/// Distinguishes a broken or closed transport from JSON-RPC errors returned by
/// the client and from client messages that violate the protocol.
#[derive(Debug, Clone, Error)]
pub enum ServerError {
    /// A transport-level failure: I/O or an internal channel error.
    ///
    /// The underlying error is stringified so the error can be cloned and
    /// delivered to every pending request at once.
    #[error("transport error: {0}")]
    Transport(String),

    /// The client connection closed.
    #[error("client connection closed")]
    Closed,

    /// The client returned a JSON-RPC error response.
    #[error("JSON-RPC error: {0}")]
    JsonRpc(#[from] JsonRpcError),

    /// The client sent a message that violates the protocol: a response that
    /// does not match its request's shape, or a malformed payload.
    #[error("protocol violation: {0}")]
    Protocol(String),
}
