//! JSON-RPC 2.0 error object and the error codes ACP assigns.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

/// JSON-RPC `parse error` — invalid JSON was received.
pub const PARSE_ERROR: i64 = -32700;
/// JSON-RPC `invalid request` — the value sent is not a request object.
pub const INVALID_REQUEST: i64 = -32600;
/// JSON-RPC `method not found`.
pub const METHOD_NOT_FOUND: i64 = -32601;
/// JSON-RPC `invalid params`.
pub const INVALID_PARAMS: i64 = -32602;
/// JSON-RPC `internal error`.
pub const INTERNAL_ERROR: i64 = -32603;
/// ACP `request cancelled` — the peer aborted an in-flight request.
pub const REQUEST_CANCELLED: i64 = -32800;
/// ACP `authentication required`.
pub const AUTH_REQUIRED: i64 = -32000;
/// ACP `resource not found` — sessions and other named resources.
pub const RESOURCE_NOT_FOUND: i64 = -32002;
/// ACP `protocol version not supported`.
pub const PROTOCOL_VERSION_UNSUPPORTED: i64 = -32022;

/// A JSON-RPC 2.0 error object.
///
/// Carries the wire `code`/`message`/`data` triple verbatim; also used to
/// report transport-level failures from [`Transport`](crate::rpc::Transport)
/// implementations (the `code` is meaningless off the wire, the `message`
/// describes the failure).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Error)]
#[error("{message} ({code})")]
pub struct RpcError {
    /// Error code.
    pub code: i64,
    /// Human-readable description of the failure.
    pub message: String,
    /// Optional structured data attached by the sender.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    /// An error with `code` and `message` and no `data`.
    #[must_use]
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// Attach structured `data`, consuming and returning `self`.
    ///
    /// A serialization failure is folded into a plain string rather than
    /// dropped — an error object must never fail to be built.
    #[must_use]
    pub fn data(mut self, data: impl Serialize) -> Self {
        self.data = Some(match serde_json::to_value(data) {
            Ok(value) => value,
            Err(error) => Value::String(format!("failed to serialize error data: {error}")),
        });
        self
    }

    /// The error code.
    #[must_use]
    pub const fn code(&self) -> i64 {
        self.code
    }

    /// Whether this is `method not found` (-32601).
    #[must_use]
    pub const fn is_method_not_found(&self) -> bool {
        self.code == METHOD_NOT_FOUND
    }

    /// `Parse error` (-32700).
    #[must_use]
    pub fn parse_error() -> Self {
        Self::new(PARSE_ERROR, "Parse error")
    }

    /// `Invalid request` (-32600).
    #[must_use]
    pub fn invalid_request() -> Self {
        Self::new(INVALID_REQUEST, "Invalid request")
    }

    /// `Method not found` (-32601) for `method`.
    #[must_use]
    pub fn method_not_found(method: &str) -> Self {
        Self::new(METHOD_NOT_FOUND, format!("Method not found: {method}"))
    }

    /// `Invalid params` (-32602).
    #[must_use]
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self::new(INVALID_PARAMS, message)
    }

    /// `Internal error` (-32603).
    #[must_use]
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::new(INTERNAL_ERROR, message)
    }

    /// `Request cancelled` (-32800).
    #[must_use]
    pub fn request_cancelled() -> Self {
        Self::new(REQUEST_CANCELLED, "Request cancelled")
    }

    /// `Authentication required` (-32000).
    #[must_use]
    pub fn auth_required() -> Self {
        Self::new(AUTH_REQUIRED, "Authentication required")
    }

    /// `Resource not found` (-32002) for `resource`.
    #[must_use]
    pub fn resource_not_found(resource: impl std::fmt::Display) -> Self {
        Self::new(
            RESOURCE_NOT_FOUND,
            format!("Resource not found: {resource}"),
        )
    }

    /// `Session not found` (-32002) with `data.sessionId` set.
    #[must_use]
    pub fn session_not_found(session_id: &str) -> Self {
        Self::resource_not_found(format_args!("session {session_id}"))
            .data(json!({ "sessionId": session_id }))
    }

    /// `Protocol version not supported` (-32022) with `data.requested` and
    /// `data.supported` set.
    #[must_use]
    pub fn protocol_version_unsupported(requested: u16, supported: &[u16]) -> Self {
        Self::new(
            PROTOCOL_VERSION_UNSUPPORTED,
            format!("Protocol version {requested} is not supported"),
        )
        .data(json!({ "requested": requested, "supported": supported }))
    }
}

impl From<std::io::Error> for RpcError {
    /// Map a transport-level I/O failure to an `internal error`.
    fn from(error: std::io::Error) -> Self {
        Self::internal_error(format!("I/O error: {error}"))
    }
}

impl From<serde_json::Error> for RpcError {
    /// Map a serialization failure to an `internal error`.
    fn from(error: serde_json::Error) -> Self {
        Self::internal_error(format!("serialization error: {error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_carry_their_codes() {
        assert_eq!(RpcError::parse_error().code(), PARSE_ERROR);
        assert_eq!(RpcError::invalid_request().code(), INVALID_REQUEST);
        assert_eq!(RpcError::method_not_found("m").code(), METHOD_NOT_FOUND);
        assert_eq!(RpcError::invalid_params("p").code(), INVALID_PARAMS);
        assert_eq!(RpcError::internal_error("i").code(), INTERNAL_ERROR);
        assert_eq!(RpcError::request_cancelled().code(), REQUEST_CANCELLED);
        assert_eq!(RpcError::auth_required().code(), AUTH_REQUIRED);
        assert_eq!(RpcError::resource_not_found("r").code(), RESOURCE_NOT_FOUND);
        assert_eq!(
            RpcError::protocol_version_unsupported(3, &[1, 2]).code(),
            PROTOCOL_VERSION_UNSUPPORTED
        );
    }

    #[test]
    fn display_is_message_then_code() {
        assert_eq!(RpcError::new(42, "boom").to_string(), "boom (42)");
    }

    #[test]
    fn is_method_not_found_matches_only_its_code() {
        assert!(RpcError::method_not_found("m").is_method_not_found());
        assert!(!RpcError::parse_error().is_method_not_found());
    }

    #[test]
    fn session_not_found_carries_the_session_id() {
        let error = RpcError::session_not_found("s-1");
        assert_eq!(error.code(), RESOURCE_NOT_FOUND);
        assert_eq!(error.data, Some(json!({ "sessionId": "s-1" })));
    }

    #[test]
    fn version_unsupported_carries_both_versions() {
        let error = RpcError::protocol_version_unsupported(3, &[1, 2]);
        assert_eq!(
            error.data,
            Some(json!({ "requested": 3, "supported": [1, 2] }))
        );
    }

    #[test]
    fn roundtrips_through_json() {
        let error = RpcError::invalid_params("nope").data(json!({ "a": 1 }));
        let value = serde_json::to_value(&error).unwrap();
        assert_eq!(value["code"], json!(INVALID_PARAMS));
        assert_eq!(serde_json::from_value::<RpcError>(value).unwrap(), error);
    }
}
