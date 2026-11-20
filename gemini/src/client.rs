use aither_core::llm::{Classify, RetryPolicy};
use std::{future::Future, time::Duration};

use aither_core::time::sleep;
use futures_lite::StreamExt;

use serde::{Deserialize, Serialize};
use url::Url;
use zenwave::{Client, client, header};

use crate::{
    config::{AuthMode, GeminiConfig, USER_AGENT},
    error::{ApiErrorResponse, GeminiError},
    types::{
        EmbedContentRequest, EmbedContentResponse, GenerateContentRequest, GenerateContentResponse,
    },
};

pub async fn within_idle<T>(
    idle: Duration,
    fut: impl Future<Output = T>,
) -> Result<T, GeminiError> {
    futures_lite::future::or(async move { Ok(fut.await) }, async move {
        sleep(idle).await;
        Err(GeminiError::IdleTimeout { idle })
    })
    .await
}

pub fn redact_url(raw: &str) -> String {
    if let Ok(mut parsed) = Url::parse(raw) {
        if parsed.query().is_some() {
            let pairs: Vec<(String, String)> = parsed
                .query_pairs()
                .map(|(k, v)| {
                    if k.eq_ignore_ascii_case("alt") {
                        (k.into_owned(), v.into_owned())
                    } else {
                        (k.into_owned(), "[REDACTED]".to_string())
                    }
                })
                .collect();
            parsed.set_query(None);
            {
                let mut qp = parsed.query_pairs_mut();
                for (k, v) in pairs {
                    qp.append_pair(&k, &v);
                }
            }
        }
        return parsed.to_string();
    }
    match raw.split_once('?') {
        Some((prefix, _)) => format!("{prefix}?[REDACTED]"),
        None => raw.to_string(),
    }
}

/// Response from GET /models/{model} endpoint.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    /// Model name (e.g., "models/gemini-2.0-flash")
    #[allow(dead_code)]
    pub name: String,
    /// Maximum input tokens the model can accept.
    pub input_token_limit: u32,
    /// Maximum output tokens the model can generate.
    #[allow(dead_code)]
    pub output_token_limit: u32,
}

pub async fn call_generate(
    cfg: &GeminiConfig,
    model: &str,
    request: GenerateContentRequest,
) -> Result<GenerateContentResponse, GeminiError> {
    post_json(cfg, cfg.model_endpoint(model, "generateContent"), &request).await
}

/// How a request that failed with a retryable failure is retried: twice,
/// after 2 s and then 4 s, or after the delay the upstream requested.
pub const RETRY_POLICY: RetryPolicy = RetryPolicy {
    max_retries: 2,
    initial_delay: Duration::from_secs(2),
    max_delay: Duration::from_secs(60),
    multiplier: 2,
};

fn sse_endpoint(cfg: &GeminiConfig, model: &str) -> String {
    let base_endpoint = cfg.model_endpoint(model, "streamGenerateContent");
    Url::parse(&base_endpoint).map_or_else(
        |_| {
            let separator = if base_endpoint.contains('?') {
                '&'
            } else {
                '?'
            };
            format!("{base_endpoint}{separator}alt=sse")
        },
        |mut parsed| {
            parsed.query_pairs_mut().append_pair("alt", "sse");
            parsed.to_string()
        },
    )
}

async fn connect_sse(
    cfg: &GeminiConfig,
    endpoint: &str,
    request: &GenerateContentRequest,
    debug: bool,
) -> Result<zenwave::sse::SseStream, GeminiError> {
    let mut retries = 0u32;
    loop {
        let mut backend = client();
        let mut builder = backend
            .post(endpoint)
            .map_err(GeminiError::from_http)?
            .header(header::USER_AGENT.as_str(), USER_AGENT)
            .map_err(GeminiError::from_http)?;

        if cfg.auth == AuthMode::Header {
            builder = builder
                .header("x-goog-api-key", cfg.api_key.clone())
                .map_err(GeminiError::from_http)?;
        }

        let builder = builder.json_body(request).map_err(GeminiError::from_http)?;

        let err = match within_idle(cfg.idle_timeout, builder.sse()).await {
            Ok(Ok(stream)) => return Ok(stream),
            Ok(Err(http_err)) => GeminiError::from_http(http_err),
            Err(timeout_err) => timeout_err,
        };
        let failure = err.failure();
        if !RETRY_POLICY.allows(retries, &failure) {
            return Err(err);
        }
        let delay = RETRY_POLICY.delay(retries, &failure);
        retries += 1;
        if debug {
            tracing::info!(
                "Gemini streaming error (retry {}/{}), retrying in {:?}: {}",
                retries,
                RETRY_POLICY.max_retries,
                delay,
                err
            );
        }
        sleep(delay).await;
    }
}

/// Stream generation responses using SSE with retry logic.
pub async fn stream_generate(
    cfg: &GeminiConfig,
    model: &str,
    request: GenerateContentRequest,
) -> Result<
    impl futures_core::Stream<Item = Result<GenerateContentResponse, GeminiError>>,
    GeminiError,
> {
    let endpoint = sse_endpoint(cfg, model);
    let debug = std::env::var("AITHER_GEMINI_DEBUG").as_deref() == Ok("1");
    if debug && let Ok(json) = serde_json::to_string_pretty(&request) {
        tracing::debug!("Gemini SSE request to {}:\n{json}", redact_url(&endpoint));
    }

    let sse_stream = connect_sse(cfg, &endpoint, &request, debug).await?;

    let idle_timeout = cfg.idle_timeout;
    Ok(async_stream::stream! {
        let mut stream = sse_stream;
        loop {
            let next_item = match within_idle(idle_timeout, stream.next()).await {
                Ok(Some(result)) => result,
                Ok(None) => break,
                Err(timeout_err) => {
                    yield Err(timeout_err);
                    return;
                }
            };
            let event = match next_item {
                Ok(e) => e,
                Err(e) => {
                    yield Err(GeminiError::Stream(e));
                    return;
                }
            };
            let data = event.text_data();
            if data.is_empty() || data == "[DONE]" || data.eq_ignore_ascii_case(": ping") {
                continue;
            }
            if let Ok(api_err) = serde_json::from_str::<ApiErrorResponse>(data)
                && api_err.error.is_some()
            {
                yield Err(GeminiError::Api(api_err.friendly_message()));
                return;
            }
            match serde_json::from_str::<GenerateContentResponse>(data) {
                Ok(response) => yield Ok(response),
                Err(e) => {
                    yield Err(GeminiError::Json(e));
                    return;
                }
            }
        }
    })
}

/// Fetch model info including context window size.
pub async fn get_model_info(cfg: &GeminiConfig, model: &str) -> Result<ModelInfo, GeminiError> {
    let model = crate::config::sanitize_model(model);
    get_json(cfg, cfg.endpoint(&model)).await
}

pub async fn embed_content(
    cfg: &GeminiConfig,
    request: EmbedContentRequest,
) -> Result<EmbedContentResponse, GeminiError> {
    post_json(
        cfg,
        cfg.model_endpoint(&cfg.embedding_model, "embedContent"),
        &request,
    )
    .await
}

#[allow(clippy::future_not_send)]
async fn get_json<T: for<'de> serde::Deserialize<'de>>(
    cfg: &GeminiConfig,
    endpoint: String,
) -> Result<T, GeminiError> {
    let mut backend = client();
    let mut builder = backend
        .get(endpoint)
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?;
    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }
    within_idle(cfg.idle_timeout, builder.json())
        .await?
        .map_err(GeminiError::from_http)
}

#[allow(clippy::future_not_send)]
pub async fn post_json<T: for<'de> serde::Deserialize<'de> + serde::Serialize, S: Serialize>(
    cfg: &GeminiConfig,
    endpoint: String,
    body: &S,
) -> Result<T, GeminiError> {
    let debug = std::env::var("AITHER_GEMINI_DEBUG").as_deref() == Ok("1");
    let safe_endpoint = debug.then(|| redact_url(&endpoint));
    if let Some(ref redacted) = safe_endpoint
        && let Ok(json) = serde_json::to_string_pretty(body)
    {
        tracing::debug!("Gemini request to {redacted}:\n{json}");
    }

    let mut retries = 0u32;
    loop {
        let mut backend = client();
        let mut builder = backend
            .post(endpoint.clone())
            .map_err(GeminiError::from_http)?
            .header(header::USER_AGENT.as_str(), USER_AGENT)
            .map_err(GeminiError::from_http)?;
        if cfg.auth == AuthMode::Header {
            builder = builder
                .header("x-goog-api-key", cfg.api_key.clone())
                .map_err(GeminiError::from_http)?;
        }
        let builder = builder.json_body(body).map_err(GeminiError::from_http)?;

        match within_idle(cfg.idle_timeout, builder.json()).await {
            Ok(Ok(res)) => {
                if let Some(ref redacted) = safe_endpoint
                    && let Ok(json) = serde_json::to_string_pretty(&res)
                {
                    tracing::debug!("Gemini response from {redacted}:\n{json}");
                }
                return Ok(res);
            }
            Ok(Err(e)) => {
                let err = GeminiError::from_http(e);
                let failure = err.failure();
                if RETRY_POLICY.allows(retries, &failure) {
                    let delay = RETRY_POLICY.delay(retries, &failure);
                    retries += 1;
                    if debug {
                        tracing::info!(
                            "Gemini POST error (retry {}/{}), retrying in {:?}: {}",
                            retries,
                            RETRY_POLICY.max_retries,
                            delay,
                            err
                        );
                    }
                    sleep(delay).await;
                    continue;
                }
                return Err(err);
            }
            Err(err) => {
                let failure = err.failure();
                if RETRY_POLICY.allows(retries, &failure) {
                    let delay = RETRY_POLICY.delay(retries, &failure);
                    retries += 1;
                    if debug {
                        tracing::info!(
                            "Gemini POST timeout (retry {}/{}), retrying in {:?}: {}",
                            retries,
                            RETRY_POLICY.max_retries,
                            delay,
                            err
                        );
                    }
                    sleep(delay).await;
                    continue;
                }
                return Err(err);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gemini, GeminiProvider};
    use aither_core::{LanguageModel, llm::oneshot};
    use std::io::{Read, Write};

    #[test]
    fn default_auth_mode_is_header_and_debug_redacts_api_keys() {
        let client = Gemini::new("AIzaSySecretKey123");
        assert_eq!(client.config().auth, AuthMode::Header);
        assert!(
            !client
                .config()
                .endpoint("models/gemini-2.5-flash:generateContent")
                .contains("AIzaSySecretKey123")
        );
        let client_dbg = format!("{client:?}");
        assert!(!client_dbg.contains("AIzaSySecretKey123"));
        assert!(client_dbg.contains("[REDACTED]"));

        let provider = GeminiProvider::new("AIzaSyProviderSecret");
        let provider_dbg = format!("{provider:?}");
        assert!(!provider_dbg.contains("AIzaSyProviderSecret"));
        assert!(provider_dbg.contains("[REDACTED]"));
    }

    #[test]
    fn redact_url_strips_query_secrets_while_preserving_alt() {
        let raw = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:streamGenerateContent?key=AIzaSySuperSecret&alt=sse";
        let redacted = redact_url(raw);
        assert!(!redacted.contains("AIzaSySuperSecret"));
        assert!(redacted.contains("alt=sse"));
    }

    #[tokio::test]
    async fn sse_stream_enforces_idle_timeout_when_server_stalls() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind stub");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n",
            );
            let _ = stream.flush();
            // Hold connection open without sending SSE events.
            while stream.read(&mut buf).is_ok_and(|n| n > 0) {}
        });

        let client = Gemini::new("key")
            .with_base_url(format!("http://{addr}"))
            .with_idle_timeout(Duration::from_millis(80));
        let stream = client.respond(oneshot("sys", "hello"));
        futures_lite::pin!(stream);
        let err = stream
            .next()
            .await
            .expect("stream yields")
            .expect_err("stalled SSE stream must time out");
        assert!(matches!(err, GeminiError::IdleTimeout { .. }), "{err:?}");
        assert!(stream.next().await.is_none());
    }
}
