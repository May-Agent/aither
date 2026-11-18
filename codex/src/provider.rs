//! [`LanguageModelProvider`] for the Codex subscription backend.
//!
//! The backend exposes its own model catalog at
//! `GET {base_url}/models?client_version=…`, which is authoritative for what a
//! subscription can run — [`CodexProvider::list_models`] surfaces it directly.

use std::sync::Arc;

use async_lock::Mutex;
use serde::Deserialize;
use zenwave::{Client as _, client};

use aither_core::llm::model::Profile as ModelProfile;
use aither_core::llm::provider::{LanguageModelProvider, Profile};

use crate::auth::CodexCredentials;
use crate::client::{Codex, OnRefresh, codex_from_shared, resolve_shared_credentials};
use crate::constant::{ACCOUNT_ID_HEADER, CLIENT_VERSION, CODEX_BASE_URL, ORIGINATOR, TOKEN_URL};
use crate::error::CodexError;

/// Provider over the `ChatGPT` Codex backend.
///
/// Holds the shared credential cell used for both catalog listing and model
/// clients so single-use refresh-token rotations propagate everywhere.
#[derive(Clone)]
pub struct CodexProvider {
    shared: Arc<Mutex<CodexCredentials>>,
    base_url: String,
    token_url: String,
    originator: String,
    on_refresh: Option<OnRefresh>,
}

impl std::fmt::Debug for CodexProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodexProvider")
            .field("base_url", &self.base_url)
            .field("originator", &self.originator)
            .finish_non_exhaustive()
    }
}

impl CodexProvider {
    /// Create a provider from OAuth credentials.
    #[must_use]
    pub fn new(credentials: CodexCredentials) -> Self {
        Self {
            shared: Arc::new(Mutex::new(credentials)),
            base_url: CODEX_BASE_URL.into(),
            token_url: TOKEN_URL.into(),
            originator: ORIGINATOR.into(),
            on_refresh: None,
        }
    }

    /// Override the backend base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Override the `originator` header sent by instantiated model clients.
    #[must_use]
    pub fn with_originator(mut self, originator: impl Into<String>) -> Self {
        self.originator = originator.into();
        self
    }

    /// Propagate a token-rotation persistence callback to model clients.
    #[must_use]
    pub fn on_refresh(
        mut self,
        callback: impl Fn(&CodexCredentials) + Send + Sync + 'static,
    ) -> Self {
        self.on_refresh = Some(Arc::new(callback));
        self
    }

    #[cfg(test)]
    pub(crate) fn with_token_url(mut self, token_url: impl Into<String>) -> Self {
        self.token_url = token_url.into();
        self
    }

    /// A snapshot of the credentials backing this provider.
    pub async fn credentials(&self) -> CodexCredentials {
        self.shared.lock().await.clone()
    }
}

/// Response of `GET /codex/models`.
#[derive(Debug, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    models: Vec<ModelDescriptor>,
}

#[derive(Debug, Deserialize)]
struct ModelDescriptor {
    slug: String,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    context_window: Option<u32>,
    /// Catalog visibility; `"hide"` entries are filtered out.
    #[serde(default)]
    visibility: Option<String>,
    /// Sort priority (higher first); missing sorts last.
    #[serde(default)]
    priority: Option<i64>,
}

impl ModelDescriptor {
    fn visible(&self) -> bool {
        self.visibility.as_deref() != Some("hide")
    }

    fn into_profile(self) -> Option<ModelProfile> {
        let context_length = self.context_window.filter(|&w| w > 0).or_else(|| {
            aither_models::lookup(&self.slug).and_then(aither_models::ModelEntry::max_input_tokens)
        });
        let Some(context_length) = context_length else {
            tracing::warn!(
                model = %self.slug,
                "skip model from Codex list: missing context_window and no aither-models fallback"
            );
            return None;
        };
        let description = self
            .description
            .filter(|d| !d.trim().is_empty())
            .or(self.display_name)
            .unwrap_or_default();
        Some(ModelProfile::new(
            self.slug.clone(),
            "openai",
            self.slug,
            description,
            context_length,
        ))
    }
}

async fn fetch_models_once(
    base_url: &str,
    credentials: &CodexCredentials,
) -> Result<ModelsResponse, CodexError> {
    let endpoint = format!("{}/models", base_url.trim_end_matches('/'));
    let mut url = url::Url::parse(&endpoint)
        .map_err(|e| CodexError::Api(format!("invalid Codex base URL '{base_url}': {e}")))?;
    url.query_pairs_mut()
        .append_pair("client_version", CLIENT_VERSION);

    let mut backend = client();
    let mut request = backend.get(url.as_str())?.header(
        "authorization",
        format!("Bearer {}", credentials.access_token),
    )?;
    if let Some(account_id) = &credentials.account_id {
        request = request.header(ACCOUNT_ID_HEADER, account_id)?;
    }
    Ok(request.json().await?)
}

const fn is_http_unauthorized(err: &CodexError) -> bool {
    matches!(
        err,
        CodexError::Http(zenwave::Error::Http { status, .. }) if status.as_u16() == 401
    )
}

impl LanguageModelProvider for CodexProvider {
    type Model = Codex;
    type Error = CodexError;

    async fn list_models(&self) -> Result<Vec<ModelProfile>, Self::Error> {
        let credentials = resolve_shared_credentials(
            &self.shared,
            &self.token_url,
            self.on_refresh.as_ref(),
            false,
            None,
        )
        .await?;
        let mut response = match fetch_models_once(&self.base_url, &credentials).await {
            Ok(resp) => resp,
            Err(err) if is_http_unauthorized(&err) && credentials.can_refresh() => {
                let refreshed = resolve_shared_credentials(
                    &self.shared,
                    &self.token_url,
                    self.on_refresh.as_ref(),
                    true,
                    Some(&credentials.access_token),
                )
                .await?;
                fetch_models_once(&self.base_url, &refreshed).await?
            }
            Err(err) => return Err(err),
        };
        response
            .models
            .sort_by_key(|m| std::cmp::Reverse(m.priority.unwrap_or(0)));
        Ok(response
            .models
            .into_iter()
            .filter(ModelDescriptor::visible)
            .filter_map(ModelDescriptor::into_profile)
            .collect())
    }

    fn get_model(
        &self,
        name: &str,
    ) -> impl core::future::Future<Output = Result<Self::Model, Self::Error>> + Send {
        let model = codex_from_shared(
            Arc::clone(&self.shared),
            name,
            &self.base_url,
            &self.token_url,
            &self.originator,
            self.on_refresh.clone(),
        );
        std::future::ready(Ok(model))
    }

    fn profile() -> Profile {
        Profile::new("codex", "OpenAI models through a ChatGPT subscription")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::{Event, LanguageModel, oneshot};
    use futures_lite::StreamExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn model_descriptor_into_profile_uses_slug_and_registry_fallback() {
        let with_window = ModelDescriptor {
            slug: "gpt-5.3-codex".into(),
            display_name: Some("GPT-5.3 Codex".into()),
            description: Some("Fast coding model".into()),
            context_window: Some(272_000),
            visibility: None,
            priority: Some(10),
        }
        .into_profile()
        .expect("profile");
        assert_eq!(with_window.name, "gpt-5.3-codex");
        assert_eq!(with_window.slug, "gpt-5.3-codex");
        assert_eq!(with_window.context_length, 272_000);

        let fallback = ModelDescriptor {
            slug: "gpt-4o".into(),
            display_name: Some("GPT-4o".into()),
            description: None,
            context_window: Some(0),
            visibility: None,
            priority: None,
        }
        .into_profile()
        .expect("registry fallback for gpt-4o");
        assert_eq!(fallback.name, "gpt-4o");
        assert!(fallback.context_length > 0);
        assert_eq!(fallback.description, "GPT-4o");

        let unknown_zero = ModelDescriptor {
            slug: "unregistered-codex-model".into(),
            display_name: Some("Unknown".into()),
            description: None,
            context_window: None,
            visibility: None,
            priority: None,
        }
        .into_profile();
        assert!(unknown_zero.is_none());
    }

    #[test]
    fn cloned_codex_setters_do_not_panic_and_default_originator_is_aither() {
        assert_eq!(ORIGINATOR, "aither");
        let c1 = Codex::new(CodexCredentials::new("at", "rt"), "gpt-4o");
        let c2 = c1
            .clone()
            .with_base_url("http://127.0.0.1:9999")
            .with_originator("custom-app")
            .on_refresh(|_| {});
        assert_eq!(c1.retry_policy(), c2.retry_policy());
        assert_eq!(c1.token_counting(), c2.token_counting());
        assert_eq!(c1.count_text("hello"), c2.count_text("hello"));
    }

    #[tokio::test]
    async fn access_only_credentials_skip_refresh_when_expired() {
        let creds = CodexCredentials::access_only("expired-at").with_expires_at(1);
        let client = Codex::new(creds, "gpt-5.3-codex").with_base_url("http://127.0.0.1:1");
        let mut stream = std::pin::pin!(client.respond(oneshot("sys", "hi")));
        let err = stream
            .next()
            .await
            .expect("yields error")
            .expect_err("must fail without hitting refresh endpoint");
        assert!(
            err.to_string().contains("no refresh token"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn provider_and_models_share_single_use_refresh_rotation_and_base_url() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let base = format!("http://{addr}");
        let refresh_hits = Arc::new(AtomicUsize::new(0));
        let refresh_hits_srv = Arc::clone(&refresh_hits);

        std::thread::spawn(move || {
            for _ in 0..4 {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let mut req = Vec::new();
                let mut buf = [0u8; 4096];
                let head_end = loop {
                    let n = stream.read(&mut buf).expect("read");
                    if n == 0 {
                        return;
                    }
                    req.extend_from_slice(&buf[..n]);
                    if let Some(pos) = req.windows(4).position(|w| w == b"\r\n\r\n") {
                        break pos + 4;
                    }
                };
                let head = String::from_utf8_lossy(&req[..head_end]).into_owned();
                let lower = head.to_ascii_lowercase();
                let content_len = lower
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                while req.len() < head_end + content_len {
                    let n = stream.read(&mut buf).expect("read body");
                    if n == 0 {
                        break;
                    }
                    req.extend_from_slice(&buf[..n]);
                }
                let body_str = String::from_utf8_lossy(&req[head_end..]).into_owned();

                if head.starts_with("POST /oauth/token ") {
                    let count = refresh_hits_srv.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(count, 0, "single-use refresh token must only be spent once");
                    assert!(body_str.contains("\"refresh_token\":\"rt-initial\""));
                    let token_body = serde_json::json!({
                        "access_token": "at-rotated",
                        "refresh_token": "rt-rotated",
                        "expires_in": 3600
                    })
                    .to_string();
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{token_body}",
                        token_body.len()
                    );
                    let _ = stream.write_all(resp.as_bytes());
                } else if head.starts_with("POST /responses ")
                    && lower.contains("authorization: bearer at-rotated")
                    && lower.contains("originator: aither")
                {
                    let sse = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"shared\"}\n\ndata: [DONE]\n\n";
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{sse}",
                        sse.len()
                    );
                    let _ = stream.write_all(resp.as_bytes());
                } else {
                    let err_body = r#"{"error":{"message":"unauthorized"}}"#;
                    let resp = format!(
                        "HTTP/1.1 401 Unauthorized\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{err_body}",
                        err_body.len()
                    );
                    let _ = stream.write_all(resp.as_bytes());
                }
            }
        });

        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_clone = Arc::clone(&callback_count);
        let provider = CodexProvider::new(CodexCredentials::new("at-stale", "rt-initial"))
            .with_base_url(&base)
            .with_token_url(format!("{base}/oauth/token"))
            .on_refresh(move |creds| {
                assert_eq!(creds.refresh_token, "rt-rotated");
                callback_count_clone.fetch_add(1, Ordering::SeqCst);
            });

        let model_a = provider.get_model("gpt-5.3-codex").await.expect("model a");
        let model_b = provider.get_model("gpt-5.4").await.expect("model b");

        // model_a hits 401 on `at-stale`, refreshes `rt-initial` -> `at-rotated` / `rt-rotated`, and succeeds.
        let out_a = Box::pin(aither_core::llm::collect_text(
            model_a.respond(oneshot("sys", "hi")),
        ))
        .await
        .expect("model_a succeeds after refresh");
        assert_eq!(out_a, "shared");
        assert_eq!(refresh_hits.load(Ordering::SeqCst), 1);
        assert_eq!(callback_count.load(Ordering::SeqCst), 1);

        // model_b immediately observes `at-rotated` + `rt-rotated` from shared state without refreshing.
        let out_b = Box::pin(aither_core::llm::collect_text(
            model_b.respond(oneshot("sys", "hi")),
        ))
        .await
        .expect("model_b reuses rotated token");
        assert_eq!(out_b, "shared");
        assert_eq!(refresh_hits.load(Ordering::SeqCst), 1);
        assert_eq!(provider.credentials().await.refresh_token, "rt-rotated");
    }

    #[tokio::test]
    async fn mid_stream_events_prevent_401_retry_duplication() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let base = format!("http://{addr}");

        // Stream yields one text event and then ends cleanly; ensure normal flow works and
        // check `yielded_any` guard logic via a 2-attempt check where initial 401 retries once.
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_srv = Arc::clone(&hits);
        std::thread::spawn(move || {
            for _ in 0..2 {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                hits_srv.fetch_add(1, Ordering::SeqCst);
                let sse = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"once\"}\n\ndata: {\"type\":\"error\",\"code\":\"invalid_api_key\",\"message\":\"expired mid stream\"}\n\n";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{sse}",
                    sse.len()
                );
                let _ = stream.write_all(resp.as_bytes());
                std::thread::sleep(Duration::from_millis(10));
            }
        });

        let client =
            Codex::new(CodexCredentials::new("at", "rt"), "gpt-5.3-codex").with_base_url(&base);
        let events: Vec<Result<Event, CodexError>> =
            client.respond(oneshot("sys", "hi")).collect().await;
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        assert!(matches!(&events[0], Ok(Event::Text(t)) if t == "once"));
        assert!(events.iter().any(Result::is_err));
    }
}
