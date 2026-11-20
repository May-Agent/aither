use std::{
    sync::{Arc, OnceLock, atomic::AtomicU64},
    time::Duration,
};

use aither_core::llm::model::Ability;
use url::Url;

/// Gemini REST base URL used by the Developer API.
pub const GEMINI_API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Default idle timeout per response header and per SSE event.
pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

pub const USER_AGENT: &str = "aither-gemini/0.1";
/// Google's latest and most capable Flash model, built for coding and agentic
/// workflows. Thinking depth is steered with `thinkingLevel`, not a token budget.
pub const DEFAULT_MODEL: &str = "gemini-3.7-flash";
pub const DEFAULT_EMBEDDING_MODEL: &str = "gemini-embedding-001";
/// Nano Banana 2.
pub const DEFAULT_IMAGE_MODEL: &str = "gemini-3.1-flash-image";
pub const DEFAULT_TTS_MODEL: &str = "gemini-3.1-flash-tts-preview";
pub const DEFAULT_TTS_VOICE: &str = "Kore";

/// Authentication strategy supported by the Gemini backend.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    /// Attach `?key=API_KEY` to every request.
    Query,
    /// Send the API key via `x-goog-api-key` header (default).
    #[default]
    Header,
}

/// Native Gemini backend wired up to the `aither` traits.
#[derive(Clone, Debug)]
pub struct Gemini {
    inner: Arc<GeminiConfig>,
}

impl Gemini {
    /// Create a backend using the default chat/embedding models.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(GeminiConfig {
                api_key: api_key.into(),
                base_url: GEMINI_API_BASE_URL.to_string(),
                auth: AuthMode::Header,
                text_model: sanitize_model(DEFAULT_MODEL),
                embedding_model: sanitize_model(DEFAULT_EMBEDDING_MODEL),
                embedding_dimensions: 3072,
                image_model: Some(sanitize_model(DEFAULT_IMAGE_MODEL)),
                tts_model: Some(sanitize_model(DEFAULT_TTS_MODEL)),
                tts_voice: DEFAULT_TTS_VOICE.to_string(),
                native_abilities: vec![Ability::Pdf],
                idle_timeout: DEFAULT_IDLE_TIMEOUT,
                tool_call_seq: Arc::new(AtomicU64::new(0)),
                cached_context_length: Arc::new(OnceLock::new()),
            }),
        }
    }

    /// Override the REST base URL (useful for sandboxes or proxies).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let cfg = Arc::make_mut(&mut self.inner);
        cfg.base_url = base_url.into();
        cfg.cached_context_length = Arc::new(OnceLock::new());
        self
    }

    /// Select authentication mode (`AuthMode::Header` by default).
    #[must_use]
    pub fn with_auth_mode(mut self, mode: AuthMode) -> Self {
        Arc::make_mut(&mut self.inner).auth = mode;
        self
    }

    /// Override the per-event and initial-response idle timeout.
    #[must_use]
    pub fn with_idle_timeout(mut self, idle_timeout: Duration) -> Self {
        Arc::make_mut(&mut self.inner).idle_timeout = idle_timeout;
        self
    }

    /// Override the default chat model.
    #[must_use]
    pub fn with_text_model(mut self, model: impl Into<String>) -> Self {
        let cfg = Arc::make_mut(&mut self.inner);
        cfg.text_model = sanitize_model(model);
        cfg.cached_context_length = Arc::new(OnceLock::new());
        self
    }

    /// Override the embedding model and (optionally) its dimensionality.
    #[must_use]
    pub fn with_embedding_model(mut self, model: impl Into<String>, dim: usize) -> Self {
        let cfg = Arc::make_mut(&mut self.inner);
        cfg.embedding_model = sanitize_model(model);
        cfg.embedding_dimensions = dim;
        self
    }

    /// Override the optional image model.
    #[must_use]
    pub fn with_image_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).image_model = Some(sanitize_model(model));
        self
    }

    /// Disable image generation support.
    #[must_use]
    pub fn without_image_model(mut self) -> Self {
        Arc::make_mut(&mut self.inner).image_model = None;
        self
    }

    /// Override the optional text-to-speech model + voice.
    #[must_use]
    pub fn with_tts(mut self, model: impl Into<String>, voice: impl Into<String>) -> Self {
        let cfg = Arc::make_mut(&mut self.inner);
        cfg.tts_model = Some(sanitize_model(model));
        cfg.tts_voice = voice.into();
        self
    }

    pub(crate) fn config(&self) -> &GeminiConfig {
        &self.inner
    }

    /// Declare native capabilities exposed by the selected Gemini model.
    #[must_use]
    pub fn with_native_capabilities(
        mut self,
        abilities: impl IntoIterator<Item = Ability>,
    ) -> Self {
        let cfg = Arc::make_mut(&mut self.inner);
        for ability in abilities {
            if !cfg.native_abilities.contains(&ability) {
                cfg.native_abilities.push(ability);
            }
        }
        self
    }

    /// Declare a single native capability.
    #[must_use]
    pub fn with_native_capability(self, ability: Ability) -> Self {
        self.with_native_capabilities([ability])
    }
}

#[derive(Clone)]
pub struct GeminiConfig {
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) auth: AuthMode,
    pub(crate) text_model: String,
    pub(crate) embedding_model: String,
    pub(crate) embedding_dimensions: usize,
    pub(crate) image_model: Option<String>,
    pub(crate) tts_model: Option<String>,
    pub(crate) tts_voice: String,
    pub(crate) native_abilities: Vec<Ability>,
    pub(crate) idle_timeout: Duration,
    pub(crate) tool_call_seq: Arc<AtomicU64>,
    pub(crate) cached_context_length: Arc<OnceLock<u32>>,
}

impl std::fmt::Debug for GeminiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiConfig")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .field("auth", &self.auth)
            .field("text_model", &self.text_model)
            .field("embedding_model", &self.embedding_model)
            .field("embedding_dimensions", &self.embedding_dimensions)
            .field("image_model", &self.image_model)
            .field("tts_model", &self.tts_model)
            .field("tts_voice", &self.tts_voice)
            .field("native_abilities", &self.native_abilities)
            .field("idle_timeout", &self.idle_timeout)
            .finish_non_exhaustive()
    }
}

impl GeminiConfig {
    pub(crate) fn endpoint(&self, suffix: &str) -> String {
        let raw = format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            suffix.trim_start_matches('/')
        );
        if self.auth == AuthMode::Query {
            if let Ok(mut parsed) = Url::parse(&raw) {
                parsed.query_pairs_mut().append_pair("key", &self.api_key);
                return parsed.to_string();
            }
            let separator = if raw.contains('?') { '&' } else { '?' };
            return format!("{raw}{separator}key={}", self.api_key);
        }
        raw
    }

    pub(crate) fn model_endpoint(&self, model: &str, action: &str) -> String {
        let model = sanitize_model(model);
        self.endpoint(&format!("{model}:{action}"))
    }
}

pub fn sanitize_model(model: impl Into<String>) -> String {
    let model = model.into();
    if model.starts_with("models/") {
        model
    } else {
        format!("models/{model}")
    }
}
