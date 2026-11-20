use std::time::Duration;

use aither_core::llm::{
    Classify, Failure, FailureKind,
    failure::zenwave::{body_failure, failure as http_failure, io_is_network, sse_failure},
};
use zenwave::{BodyError, Error as ZenwaveError, sse::ParseError as SseParseError};

/// Errors that can arise when calling the `OpenAI` API.
#[derive(Debug, thiserror::Error)]
pub enum OpenAIError {
    /// HTTP layer errors.
    #[error("HTTP error: {0}")]
    Http(#[source] ZenwaveError),
    /// Response body parsing failures.
    #[error("Body error: {0}")]
    Body(#[from] BodyError),
    /// SSE parsing failures.
    #[error("SSE error: {0}")]
    Stream(#[from] SseParseError),
    /// JSON serialization/deserialization errors.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Base64 decoding failures (e.g., image payloads).
    #[error("Base64 decode error: {0}")]
    Decode(#[from] base64::DecodeError),
    /// Local I/O failures (e.g., reading attachments or cache files).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// API contract violations or unsupported operations.
    #[error("{0}")]
    Api(String),
    /// Rate limit exceeded (HTTP 429).
    #[error("Rate limit exceeded: {message}{}", retry_after.map_or_default(|d| format!(" (retry after {}s)", d.as_secs())))]
    RateLimit {
        /// Message from the API.
        message: String,
        /// Suggested retry delay from Retry-After header.
        retry_after: Option<Duration>,
    },
    /// The account's quota or billing limit is exhausted (HTTP 429
    /// `insufficient_quota`).
    #[error("Quota exceeded: {message}")]
    QuotaExceeded {
        /// Message from the API.
        message: String,
    },
    /// Server error (HTTP 5xx).
    #[error("Server error {status}: {message}")]
    ServerError {
        /// HTTP status code.
        status: u16,
        /// Error message.
        message: String,
        /// Suggested retry delay from Retry-After header.
        retry_after: Option<Duration>,
    },
    /// The API reported an error in an event of an open stream.
    #[error("{message}")]
    StreamEvent {
        /// The error's `type`, such as `server_error`.
        error_type: Option<String>,
        /// The error's `code`, such as `insufficient_quota`.
        code: Option<String>,
        /// Message from the API.
        message: String,
    },
    /// The response did not start, or the response stream sent nothing, for the client's idle timeout.
    #[error("the response was silent for {idle:?}")]
    IdleTimeout {
        /// How long the client waited.
        idle: Duration,
    },
}

/// The `error` object of an `OpenAI` error body, `{"error":{"type":..,"code":..}}`.
#[derive(Debug, Default, serde::Deserialize)]
struct ErrorObject {
    #[serde(rename = "type")]
    error_type: Option<String>,
    code: Option<String>,
    message: Option<String>,
}

impl ErrorObject {
    fn from_body(body: &str) -> Option<Self> {
        #[derive(serde::Deserialize)]
        struct Body {
            error: ErrorObject,
        }
        serde_json::from_str::<Body>(body).ok().map(|b| b.error)
    }

    /// The kind this error object forces regardless of status, if any.
    fn kind(&self) -> Option<FailureKind> {
        let is = |wanted: &[&str]| {
            [&self.code, &self.error_type]
                .into_iter()
                .flatten()
                .any(|value| wanted.contains(&value.as_str()))
        };
        if is(&["insufficient_quota", "billing_hard_limit_reached"]) {
            Some(FailureKind::Quota)
        } else if is(&[
            "rate_limit_exceeded",
            "server_error",
            "service_unavailable_error",
            "overloaded_error",
        ]) {
            Some(FailureKind::Transient)
        } else {
            None
        }
    }
}

impl OpenAIError {
    /// Convert a zenwave error to an `OpenAIError`.
    pub(crate) fn from_http(err: zenwave::Error) -> Self {
        let failure = http_failure(&err);
        if let zenwave::Error::Http { status, .. } = &err {
            let object = err.response_body().and_then(ErrorObject::from_body);
            if status.as_u16() == 429 {
                if failure.kind == FailureKind::Quota
                    || object.as_ref().and_then(ErrorObject::kind) == Some(FailureKind::Quota)
                {
                    return Self::QuotaExceeded {
                        message: object
                            .and_then(|o| o.message)
                            .unwrap_or_else(|| "Quota exceeded".to_string()),
                    };
                }
                return Self::RateLimit {
                    message: "Rate limit exceeded".to_string(),
                    retry_after: failure.retry_after,
                };
            }
            if status.as_u16() >= 500 {
                return Self::ServerError {
                    status: status.as_u16(),
                    message: err.to_string(),
                    retry_after: failure.retry_after,
                };
            }
        }
        Self::Http(err)
    }

    /// Builds the error for an `error` object that arrived as a stream event.
    pub(crate) fn from_stream_event(error: &serde_json::Value) -> Self {
        let text = |key: &str| error.get(key).and_then(|v| v.as_str()).map(str::to_string);
        Self::StreamEvent {
            error_type: text("type"),
            code: text("code"),
            message: text("message").unwrap_or_else(|| "Unknown API error".to_string()),
        }
    }
}

impl Classify for OpenAIError {
    fn failure(&self) -> Failure {
        match self {
            Self::Http(err) => {
                let mut failure = http_failure(err);
                if let Some(kind) = err
                    .response_body()
                    .and_then(ErrorObject::from_body)
                    .and_then(|object| object.kind())
                {
                    failure.kind = kind;
                }
                failure
            }
            Self::Body(err) => body_failure(err),
            Self::Stream(err) => sse_failure(err),
            // An EOF while parsing JSON is a body cut off mid-response.
            Self::Json(err) if err.is_eof() => Failure::network(),
            Self::Io(err) if io_is_network(err.kind()) => Failure::network(),
            Self::Json(_) | Self::Decode(_) | Self::Io(_) | Self::Api(_) => Failure::permanent(),
            Self::RateLimit { retry_after, .. } | Self::ServerError { retry_after, .. } => {
                Failure::transient().with_retry_after(*retry_after)
            }
            Self::QuotaExceeded { .. } => Failure::quota(),
            Self::IdleTimeout { .. } => Failure::network(),
            Self::StreamEvent {
                error_type, code, ..
            } => {
                let object = ErrorObject {
                    error_type: error_type.clone(),
                    code: code.clone(),
                    message: None,
                };
                Failure::new(object.kind().unwrap_or(FailureKind::Permanent))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use aither_core::{
        LanguageModel,
        llm::{Classify, FailureKind, failure::zenwave::http_error, oneshot},
    };
    use futures_lite::StreamExt;

    use super::OpenAIError;
    use crate::OpenAI;

    fn from_status(status: u16, headers: &[(&str, &str)], body: &str) -> OpenAIError {
        OpenAIError::from_http(http_error(status, headers, body))
    }

    fn raw_http(status: u16, headers: &[(&str, &str)], body: &str) -> OpenAIError {
        OpenAIError::Http(http_error(status, headers, body))
    }

    #[test]
    fn server_error_is_transient() {
        assert_eq!(
            from_status(500, &[], "").failure().kind,
            FailureKind::Transient
        );
        assert_eq!(
            raw_http(503, &[], "").failure().kind,
            FailureKind::Transient
        );
    }

    #[test]
    fn overloaded_is_transient() {
        assert_eq!(
            raw_http(529, &[], "").failure().kind,
            FailureKind::Transient
        );
        let event = OpenAIError::StreamEvent {
            error_type: Some("server_error".into()),
            code: None,
            message: "overloaded".into(),
        };
        assert_eq!(event.failure().kind, FailureKind::Transient);
    }

    #[test]
    fn quota_429_is_quota_and_rate_limit_429_is_transient() {
        let quota = r#"{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":"insufficient_quota"}}"#;
        assert_eq!(
            from_status(429, &[], quota).failure().kind,
            FailureKind::Quota
        );
        assert_eq!(raw_http(429, &[], quota).failure().kind, FailureKind::Quota);
        let rate =
            r#"{"error":{"message":"Slow down","type":"requests","code":"rate_limit_exceeded"}}"#;
        assert_eq!(
            from_status(429, &[], rate).failure().kind,
            FailureKind::Transient
        );
        assert_eq!(
            raw_http(429, &[], rate).failure().kind,
            FailureKind::Transient
        );
    }

    #[test]
    fn unauthorized_is_permanent() {
        assert_eq!(
            from_status(401, &[], "").failure().kind,
            FailureKind::Permanent
        );
        assert_eq!(
            raw_http(401, &[], "").failure().kind,
            FailureKind::Permanent
        );
    }

    #[test]
    fn retry_after_seconds_and_date() {
        let failure = from_status(429, &[("retry-after", "7")], "").failure();
        assert_eq!(failure.retry_after, Some(Duration::from_secs(7)));
        let failure =
            raw_http(503, &[("retry-after", "Fri, 01 Jan 2100 00:00:00 GMT")], "").failure();
        assert!(
            failure
                .retry_after
                .is_some_and(|d| d > Duration::from_secs(3600))
        );
    }

    #[test]
    fn truncated_json_is_network() {
        let error = serde_json::from_str::<serde_json::Value>("{\"a\":").unwrap_err();
        assert_eq!(
            OpenAIError::Json(error).failure().kind,
            FailureKind::Network
        );
    }

    #[test]
    fn error_source_is_preserved() {
        use std::error::Error as _;

        let json_err =
            OpenAIError::from(serde_json::from_str::<serde_json::Value>("!").unwrap_err());
        assert!(json_err.source().is_some());

        let io_err = OpenAIError::from(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "reset",
        ));
        assert!(io_err.source().is_some());
        assert_eq!(io_err.failure().kind, FailureKind::Network);

        let http_err = raw_http(500, &[], "");
        assert!(http_err.source().is_some());
    }

    #[tokio::test]
    async fn refused_connection_is_network() {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let client = OpenAI::builder("key")
            .base_url(format!("http://127.0.0.1:{port}"))
            .no_retry()
            .build();
        let stream = client.respond(oneshot("system", "hello"));
        futures_lite::pin!(stream);
        let error = stream
            .next()
            .await
            .expect("stream yields")
            .expect_err("connection is refused");
        assert_eq!(error.failure().kind, FailureKind::Network, "{error}");
    }
}
