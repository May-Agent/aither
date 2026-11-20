//! The client error type.

use std::process::ExitStatus;

use thiserror::Error;

use crate::rpc::{PeerError, RpcError};

/// Everything that can go wrong driving an ACP agent.
///
/// Variants mirror the SPEC: transport failures, the connection dropping,
/// agent-reported errors, protocol-shape violations, version negotiation,
/// and capability-gated calls the agent does not support.
#[derive(Debug, Error)]
pub enum ClientError {
    /// The transport itself failed.
    #[error("transport error: {0}")]
    Transport(RpcError),
    /// The connection ended before a response arrived. `status` carries
    /// the agent's exit status when it is a process we spawned.
    #[error("connection closed (exit status: {status:?})")]
    PeerClosed {
        /// Exit status of a process-backed peer, if any.
        status: Option<ExitStatus>,
    },
    /// The agent replied with a payload that does not fit the protocol
    /// shape (malformed result, unrepresentable downgrade).
    #[error("protocol error: {0}")]
    Protocol(String),
    /// The peer answered the request with a JSON-RPC error.
    #[error("{0}")]
    Rejected(RpcError),
    /// The agent does not implement the requested protocol version.
    #[error("protocol version {requested} is not supported (agent supports {supported:?})")]
    VersionUnsupported {
        /// The version the client asked for.
        requested: u16,
        /// Versions the agent supports.
        supported: Vec<u16>,
    },
    /// The call is gated by a capability the agent did not advertise
    /// (e.g. `session/load` on a v1 agent without `loadSession`), or is
    /// otherwise unimplementable on this connection.
    #[error("unsupported: {0}")]
    Unsupported(String),
}

impl From<PeerError> for ClientError {
    fn from(error: PeerError) -> Self {
        match error {
            PeerError::Transport(e) => Self::Transport(e),
            PeerError::Closed { status } => Self::PeerClosed { status },
            PeerError::Rejected(e) => {
                if e.code() == crate::rpc::PROTOCOL_VERSION_UNSUPPORTED {
                    Self::VersionUnsupported {
                        requested: e
                            .data
                            .as_ref()
                            .and_then(|d| d.get("requested"))
                            .and_then(serde_json::Value::as_u64)
                            .map_or(0, |v| u16::try_from(v).unwrap_or(0)),
                        supported: e
                            .data
                            .as_ref()
                            .and_then(|d| d.get("supported"))
                            .and_then(serde_json::Value::as_array)
                            .map_or_default(|a| {
                                a.iter()
                                    .filter_map(|v| u16::try_from(v.as_u64()?).ok())
                                    .collect()
                            }),
                    }
                } else {
                    Self::Rejected(e)
                }
            }
            PeerError::Malformed(m) => Self::Protocol(m),
        }
    }
}

impl From<RpcError> for ClientError {
    fn from(error: RpcError) -> Self {
        Self::Transport(error)
    }
}
