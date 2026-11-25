//! Error types for the Debug Adapter Protocol client and session layer.

use std::io;

/// Errors produced by DAP framing, transports, client calls, and session operations.
#[derive(Debug, thiserror::Error)]
pub enum DapError {
    /// Underlying I/O failure on a transport or source-file read.
    #[error("DAP I/O error: {0}")]
    Io(#[from] io::Error),

    /// JSON serialization or deserialization failure on a DAP message.
    #[error("DAP JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Malformed Content-Length header or protocol framing.
    #[error("invalid DAP frame: {0}")]
    Protocol(String),

    /// Frame declared a `Content-Length` exceeding the safety limit.
    #[error("DAP frame length {length} bytes exceeds maximum {limit} bytes")]
    FrameTooLarge {
        /// Declared `Content-Length` in bytes.
        length: usize,
        /// Configured maximum in bytes.
        limit: usize,
    },

    /// Transport closed before completing the operation.
    #[error("DAP transport closed")]
    ConnectionClosed,

    /// Adapter returned a response with `success: false`.
    #[error("DAP request `{command}` failed: {message}")]
    RequestFailed {
        /// DAP command name that failed.
        command: String,
        /// Error message reported by the adapter.
        message: String,
    },

    /// Adapter did not emit `initialized` or reply before the session closed.
    #[error("DAP initialization failed: {0}")]
    InitializationFailed(String),

    /// Operation requires a stopped debug target, but the target is not stopped.
    #[error("target is not stopped (current state: {state})")]
    NotStopped {
        /// Current lifecycle state name.
        state: String,
    },

    /// Operation requires an active running or stopped target, but the session ended.
    #[error("debug session is no longer active (current state: {state})")]
    SessionEnded {
        /// Current lifecycle state name.
        state: String,
    },

    /// Requested breakpoint ID does not exist in the session's breakpoint table.
    #[error("breakpoint #{0} not found")]
    BreakpointNotFound(u32),

    /// Requested stack frame index is out of range for the current stopped snapshot.
    #[error("frame #{index} is out of range (available frames: {count})")]
    InvalidFrameIndex {
        /// Requested 0-based frame index.
        index: usize,
        /// Number of frames in the current snapshot.
        count: usize,
    },

    /// Requested variable expansion handle does not exist or was invalidated on resume.
    #[error("variable handle #{0} is invalid or was invalidated when execution resumed")]
    InvalidVariableHandle(usize),

    /// Adapter does not support the requested optional capability.
    #[error("debug adapter does not support {0}")]
    Unsupported(&'static str),

    /// Requested command must be handled by the host broker that spawns/connects adapters.
    #[error("command `{0}` must be handled by the debug session host")]
    RequiresHostBroker(&'static str),

    /// Session actor stopped before answering the request.
    #[error("debug session actor closed")]
    SessionClosed,
}
