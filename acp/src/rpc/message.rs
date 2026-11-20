//! JSON-RPC 2.0 message types and inbound classification.
//!
//! Every wire message carries `"jsonrpc": "2.0"`; the field is emitted as a
//! constant on serialize and tolerated on parse, so the owned types do not
//! store it.

use std::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value, json};

use super::error::RpcError;

/// Method name of the `$/cancel_request` meta-notification, which asks the
/// receiver to abort an in-flight request.
pub const CANCEL_REQUEST_METHOD: &str = "$/cancel_request";

/// A JSON-RPC request id: a string or an integer, never `null` on messages
/// we produce.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    /// String id.
    String(String),
    /// Integer id.
    Number(i64),
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(id) => f.write_str(id),
            Self::Number(id) => write!(f, "{id}"),
        }
    }
}

impl From<i64> for RequestId {
    fn from(id: i64) -> Self {
        Self::Number(id)
    }
}

impl From<String> for RequestId {
    fn from(id: String) -> Self {
        Self::String(id)
    }
}

impl From<&str> for RequestId {
    fn from(id: &str) -> Self {
        Self::String(id.to_owned())
    }
}

/// A JSON-RPC request: `method` + `id`, optionally `params`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Request {
    /// Correlation id the response must echo back.
    pub id: RequestId,
    /// Method name.
    pub method: String,
    /// Structured parameters; absent on the wire when `None`.
    #[serde(default)]
    pub params: Option<Value>,
}

impl Serialize for Request {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Request", 4)?;
        state.serialize_field("jsonrpc", "2.0")?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("method", &self.method)?;
        if let Some(params) = &self.params {
            state.serialize_field("params", params)?;
        }
        state.end()
    }
}

impl Request {
    /// A request without parameters.
    #[must_use]
    pub fn new(id: impl Into<RequestId>, method: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            method: method.into(),
            params: None,
        }
    }

    /// A request with parameters. `params` values that serialize to `null`
    /// are omitted on the wire.
    ///
    /// # Panics
    ///
    /// Panics if `params` fails to serialize to a JSON value.
    #[must_use]
    pub fn with_params(
        id: impl Into<RequestId>,
        method: impl Into<String>,
        params: impl Serialize,
    ) -> Self {
        let params = serde_json::to_value(params).expect("failed to serialize request params");
        Self {
            id: id.into(),
            method: method.into(),
            params: (!params.is_null()).then_some(params),
        }
    }
}

/// A JSON-RPC notification: `method` without `id`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Notification {
    /// Method name.
    pub method: String,
    /// Structured parameters; absent on the wire when `None`.
    #[serde(default)]
    pub params: Option<Value>,
}

impl Serialize for Notification {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Notification", 3)?;
        state.serialize_field("jsonrpc", "2.0")?;
        state.serialize_field("method", &self.method)?;
        if let Some(params) = &self.params {
            state.serialize_field("params", params)?;
        }
        state.end()
    }
}

impl Notification {
    /// A notification without parameters.
    #[must_use]
    pub fn new(method: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            params: None,
        }
    }

    /// A notification with parameters. `params` values that serialize to
    /// `null` are omitted on the wire.
    ///
    /// # Panics
    ///
    /// Panics if `params` fails to serialize to a JSON value.
    #[must_use]
    pub fn with_params(method: impl Into<String>, params: impl Serialize) -> Self {
        let params = serde_json::to_value(params).expect("failed to serialize notification params");
        Self {
            method: method.into(),
            params: (!params.is_null()).then_some(params),
        }
    }

    /// The `$/cancel_request` notification cancelling the request `id`.
    #[must_use]
    pub fn cancel_request(id: impl Into<RequestId>) -> Self {
        Self {
            method: CANCEL_REQUEST_METHOD.to_owned(),
            params: Some(json!({ "id": id.into() })),
        }
    }
}

/// A JSON-RPC response: `id` plus either `result` or `error`.
///
/// `id` is `Option` because the wire value may be `null` — peers answering
/// input they could not attribute (a parse failure, a batch) respond with
/// `"id": null`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Response {
    /// Correlation id of the answered request; `null` when unanswerable.
    pub id: Option<RequestId>,
    /// Result payload on success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error object on failure.
    #[serde(default)]
    pub error: Option<RpcError>,
}

impl Serialize for Response {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Response", 4)?;
        state.serialize_field("jsonrpc", "2.0")?;
        state.serialize_field("id", &self.id)?;
        if let Some(result) = &self.result {
            state.serialize_field("result", result)?;
        }
        if let Some(error) = &self.error {
            state.serialize_field("error", error)?;
        }
        state.end()
    }
}

impl Response {
    /// A successful response to the request `id`.
    ///
    /// # Panics
    ///
    /// Panics if `result` fails to serialize to a JSON value.
    #[must_use]
    pub fn success(id: impl Into<RequestId>, result: impl Serialize) -> Self {
        Self {
            id: Some(id.into()),
            result: Some(serde_json::to_value(result).expect("failed to serialize result")),
            error: None,
        }
    }

    /// An error response. `id` may be `None` for a `null` id — the only
    /// correct answer to a request whose id could not be determined.
    #[must_use]
    pub fn error(id: impl Into<Option<RequestId>>, error: RpcError) -> Self {
        Self {
            id: id.into(),
            result: None,
            error: Some(error),
        }
    }

    /// Whether this response carries an error.
    #[must_use]
    pub const fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// Convert to a `Result` of the raw result value.
    ///
    /// # Errors
    ///
    /// Returns the [`RpcError`] when this is an error response.
    pub fn into_result(self) -> Result<Value, RpcError> {
        if let Some(error) = self.error {
            Err(error)
        } else {
            Ok(self.result.unwrap_or(Value::Null))
        }
    }
}

/// A message received from the peer, classified.
///
/// The classification encodes JSON-RPC 2.0 strictly: batches (arrays) are
/// `Invalid`, a request with `"id": null` is `Invalid`, a response keeps a
/// `null` id, and anything that is not an object is `Invalid`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inbound {
    /// A request expecting a response.
    Request(Request),
    /// A notification expecting nothing.
    Notification(Notification),
    /// A response to a request we sent.
    Response(Response),
    /// Not a JSON-RPC message (bad JSON, batch, `null` request id, or a
    /// shape matching nothing). The engine answers `-32600` with `id: null`.
    Invalid,
}

impl Inbound {
    /// Parse one raw NDJSON line (without its trailing `\n`).
    ///
    /// Never fails: an unparseable line is `Invalid`.
    #[must_use]
    pub fn parse(line: &[u8]) -> Self {
        serde_json::from_slice::<Value>(line).map_or(Self::Invalid, Self::from_value)
    }

    /// Classify an already-parsed JSON value.
    #[must_use]
    pub fn from_value(value: Value) -> Self {
        let Value::Object(mut map) = value else {
            return Self::Invalid;
        };

        if let Some(method) = map.remove("method") {
            let Some(method) = method.as_str().map(str::to_owned) else {
                return Self::Invalid;
            };
            return match map.remove("id") {
                // `method` without `id` is a notification.
                None => Self::Notification(Notification {
                    method,
                    params: take_params(&mut map),
                }),
                // A request may never carry `"id": null`.
                Some(Value::Null) => Self::Invalid,
                Some(id) => serde_json::from_value::<RequestId>(id).map_or(Self::Invalid, |id| {
                    Self::Request(Request {
                        id,
                        method,
                        params: take_params(&mut map),
                    })
                }),
            };
        }

        // A response needs `result` or `error` plus an `id` that may be null.
        if map.contains_key("id") && (map.contains_key("result") || map.contains_key("error")) {
            let id = match map.remove("id") {
                Some(Value::Null) | None => None,
                Some(raw) => match serde_json::from_value::<RequestId>(raw) {
                    Ok(id) => Some(id),
                    Err(_) => return Self::Invalid,
                },
            };
            let error = match map.remove("error") {
                Some(raw) => match serde_json::from_value::<RpcError>(raw) {
                    Ok(error) => Some(error),
                    Err(_) => return Self::Invalid,
                },
                None => None,
            };
            return Self::Response(Response {
                id,
                result: map.remove("result"),
                error,
            });
        }

        Self::Invalid
    }
}

/// Extract `params` from `map`, treating an explicit `null` as absent.
fn take_params(map: &mut Map<String, Value>) -> Option<Value> {
    map.remove("params").filter(|params| !params.is_null())
}

/// A message queued for the peer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum Outbound {
    /// A request we initiated.
    Request(Request),
    /// A notification we sent.
    Notification(Notification),
    /// Our response to the peer's request.
    Response(Response),
}

impl From<Request> for Outbound {
    fn from(request: Request) -> Self {
        Self::Request(request)
    }
}

impl From<Notification> for Outbound {
    fn from(notification: Notification) -> Self {
        Self::Notification(notification)
    }
}

impl From<Response> for Outbound {
    fn from(response: Response) -> Self {
        Self::Response(response)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn request_id_shapes() {
        assert_eq!(
            serde_json::from_value::<RequestId>(json!("abc")).unwrap(),
            RequestId::String("abc".to_owned())
        );
        assert_eq!(
            serde_json::from_value::<RequestId>(json!(-5)).unwrap(),
            RequestId::Number(-5)
        );
        assert!(serde_json::from_value::<RequestId>(json!(null)).is_err());
        assert!(serde_json::from_value::<RequestId>(json!(1.5)).is_err());
        assert!(serde_json::from_value::<RequestId>(json!(true)).is_err());
        assert_eq!(
            serde_json::to_value(RequestId::from("s")).unwrap(),
            json!("s")
        );
        assert_eq!(
            serde_json::to_value(RequestId::from(7_i64)).unwrap(),
            json!(7)
        );
    }

    #[test]
    fn request_serializes_with_jsonrpc_constant() {
        let request = Request::with_params(1_i64, "session/prompt", json!({ "x": 1 }));
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(
            value,
            json!({"jsonrpc": "2.0", "id": 1, "method": "session/prompt", "params": {"x": 1}})
        );
        // Deserialization tolerates and drops the version tag.
        assert_eq!(serde_json::from_value::<Request>(value).unwrap(), request);
    }

    #[test]
    fn notification_omits_absent_params() {
        let value = serde_json::to_value(Notification::new("ping")).unwrap();
        assert_eq!(value, json!({"jsonrpc": "2.0", "method": "ping"}));
    }

    #[test]
    fn response_serializes_null_id() {
        let value =
            serde_json::to_value(Response::error(None, RpcError::invalid_request())).unwrap();
        assert_eq!(value["id"], json!(null));
        assert_eq!(value["error"]["code"], json!(-32600));
        assert!(value.get("result").is_none());
    }

    #[test]
    fn classifies_a_request() {
        let inbound = Inbound::parse(br#"{"jsonrpc":"2.0","id":1,"method":"m","params":{"a":2}}"#);
        let Inbound::Request(request) = inbound else {
            panic!("expected a request: {inbound:?}");
        };
        assert_eq!(request.id, RequestId::Number(1));
        assert_eq!(request.method, "m");
        assert_eq!(request.params, Some(json!({"a": 2})));
    }

    #[test]
    fn classifies_a_string_id_request() {
        let Inbound::Request(request) = Inbound::parse(br#"{"id":"s-1","method":"m"}"#) else {
            panic!("expected a request");
        };
        assert_eq!(request.id, RequestId::String("s-1".to_owned()));
        assert_eq!(request.params, None);
    }

    #[test]
    fn classifies_a_notification() {
        let Inbound::Notification(notification) =
            Inbound::parse(br#"{"jsonrpc":"2.0","method":"session/update","params":{}}"#)
        else {
            panic!("expected a notification");
        };
        assert_eq!(notification.method, "session/update");
    }

    #[test]
    fn params_null_is_absent() {
        let Inbound::Notification(notification) =
            Inbound::parse(br#"{"method":"m","params":null}"#)
        else {
            panic!("expected a notification");
        };
        assert_eq!(notification.params, None);
    }

    #[test]
    fn classifies_result_and_error_responses() {
        let Inbound::Response(ok) = Inbound::parse(br#"{"id":2,"result":{"v":1}}"#) else {
            panic!("expected a response");
        };
        assert_eq!(ok.id, Some(RequestId::Number(2)));
        assert_eq!(ok.into_result().unwrap(), json!({"v": 1}));

        let Inbound::Response(err) =
            Inbound::parse(br#"{"id":"x","error":{"code":-32601,"message":"nope"}}"#)
        else {
            panic!("expected a response");
        };
        assert_eq!(err.id, Some(RequestId::String("x".to_owned())));
        assert_eq!(err.error.unwrap().code(), -32601);
    }

    #[test]
    fn classifies_a_null_id_response() {
        let Inbound::Response(response) =
            Inbound::parse(br#"{"id":null,"error":{"code":-32600,"message":"bad"}}"#)
        else {
            panic!("expected a response");
        };
        assert_eq!(response.id, None);
        assert!(response.is_error());
    }

    #[test]
    fn rejects_batches() {
        assert_eq!(
            Inbound::parse(br#"[{"id":1,"method":"m"}]"#),
            Inbound::Invalid
        );
    }

    #[test]
    fn rejects_a_null_request_id() {
        assert_eq!(
            Inbound::parse(br#"{"id":null,"method":"m"}"#),
            Inbound::Invalid
        );
    }

    #[test]
    fn rejects_non_objects() {
        for line in [&b"5"[..], b"\"text\"", b"null", b"true"] {
            assert_eq!(Inbound::parse(line), Inbound::Invalid, "line: {line:?}");
        }
    }

    #[test]
    fn rejects_unparseable_lines() {
        assert_eq!(Inbound::parse(b"not json"), Inbound::Invalid);
        assert_eq!(Inbound::parse(b"{"), Inbound::Invalid);
    }

    #[test]
    fn rejects_bad_member_types() {
        for line in [
            &br#"{"id":1,"method":5}"#[..],
            br#"{"id":true,"method":"m"}"#,
            br#"{"id":null}"#,
            br#"{"result":1}"#,
            br#"{"id":1}"#,
            br#"{"id":true,"result":1}"#,
            br#"{"id":1,"error":{"code":"x"}}"#,
        ] {
            assert_eq!(Inbound::parse(line), Inbound::Invalid, "line: {line:?}");
        }
    }
}
