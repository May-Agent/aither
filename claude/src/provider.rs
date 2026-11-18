//! Claude provider for listing and instantiating models.

use crate::{
    Claude,
    client::{is_anthropic_host, resolve_v1_url},
    constant::{ANTHROPIC_VERSION, CLAUDE_BASE_URL},
    error::ClaudeError,
};
use aither_core::llm::{
    LanguageModelProvider, model::Profile as ModelProfile, provider::Profile as ProviderProfile,
};
use aither_models::lookup as lookup_model_info;
use serde::Deserialize;
use std::{future::Future, sync::Arc};
use zenwave::{Client, client, header};

/// Provider capable of listing and instantiating `Claude` models.
#[derive(Clone, Debug)]
pub struct ClaudeProvider {
    inner: Arc<ProviderConfig>,
}

impl ClaudeProvider {
    /// Create a new provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(ProviderConfig {
                api_key: api_key.into(),
                base_url: CLAUDE_BASE_URL.to_string(),
            }),
        }
    }

    /// Override the REST base URL.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = url.into();
        self
    }
}

impl LanguageModelProvider for ClaudeProvider {
    type Model = Claude;
    type Error = ClaudeError;

    fn list_models(&self) -> impl Future<Output = Result<Vec<ModelProfile>, Self::Error>> + Send {
        let cfg = self.inner.clone();
        async move {
            let endpoint = resolve_v1_url(&cfg.base_url, "/v1/models");
            let mut backend = client();
            let mut req = backend
                .get(endpoint)
                .map_err(ClaudeError::Http)?
                .header("x-api-key", cfg.api_key.clone())
                .map_err(ClaudeError::Http)?
                .header("anthropic-version", ANTHROPIC_VERSION)
                .map_err(ClaudeError::Http)?;
            if !is_anthropic_host(&cfg.base_url) {
                req = req
                    .header(
                        header::AUTHORIZATION.as_str(),
                        format!("Bearer {}", cfg.api_key),
                    )
                    .map_err(ClaudeError::Http)?;
            }
            let response: ModelListResponse = req.json().await.map_err(ClaudeError::Http)?;

            Ok(response
                .data
                .into_iter()
                .filter_map(|model| {
                    let context_length = model
                        .input_token_limit
                        .or(model.context_length)
                        .or(model.max_tokens)
                        .or_else(|| lookup_model_info(&model.id).and_then(aither_models::ModelEntry::max_input_tokens));
                    let Some(context_length) = context_length else {
                        tracing::warn!(
                            model = %model.id,
                            "skip model from Claude list: missing context_length and no aither-models fallback"
                        );
                        return None;
                    };
                    Some(ModelProfile::new(
                        model.id.clone(),
                        "anthropic",
                        model.id,
                        model.display_name,
                        context_length,
                    ))
                })
                .collect())
        }
    }

    fn get_model(
        &self,
        name: &str,
    ) -> impl Future<Output = Result<Self::Model, Self::Error>> + Send {
        let cfg = self.inner.clone();
        let name = name.to_string();
        async move {
            Ok(Claude::new(cfg.api_key.clone())
                .with_base_url(cfg.base_url.clone())
                .with_model(name))
        }
    }

    fn profile() -> ProviderProfile {
        ProviderProfile::new("anthropic", "Anthropic Claude API provider")
    }
}

#[derive(Clone)]
struct ProviderConfig {
    api_key: String,
    base_url: String,
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .finish()
    }
}

#[derive(Debug, Deserialize)]
struct ModelListResponse {
    data: Vec<ModelDescriptor>,
}

#[derive(Debug, Deserialize)]
struct ModelDescriptor {
    id: String,
    #[serde(default)]
    display_name: String,
    #[serde(default, alias = "inputTokenLimit", alias = "max_input_tokens")]
    input_token_limit: Option<u32>,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
}
