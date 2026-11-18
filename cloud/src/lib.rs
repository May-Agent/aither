//! Unified cloud provider wrapping `OpenAI`, Claude, Gemini, and GitHub Copilot.
//!
//! This crate provides a single `CloudProvider` enum that implements `LanguageModel`,
//! allowing seamless switching between providers.
//!
//! It also provides `CloudModelProvider` enum that implements `LanguageModelProvider`,
//! allowing unified model listing and instantiation across providers.

pub use aither_claude::{self as claude, Claude, ClaudeProvider};
pub use aither_codex::{self as codex, Codex, CodexProvider};
pub use aither_copilot::{self as copilot, Copilot, CopilotProvider};
pub use aither_gemini::{self as gemini, Gemini, GeminiProvider};
pub use aither_openai::{self as openai, OpenAI, OpenAIProvider};

use aither_core::{
    EmbeddingModel, ImageGenerator, LanguageModel,
    image::{Data as ImageData, Prompt as ImagePrompt, Size as ImageSize},
    llm::{
        Classify, Event, GenerateError, LLMRequest, LanguageModelProvider, RetryPolicy,
        TokenCounting, model::Profile, provider::Profile as ProviderProfile,
    },
};
use futures_core::Stream;
use futures_lite::StreamExt;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

/// Unified cloud provider wrapping `OpenAI`, Claude, Gemini, and GitHub Copilot.
#[derive(Clone)]
pub enum CloudProvider {
    /// `OpenAI` GPT models.
    OpenAI(OpenAI),
    /// Anthropic Claude models.
    Claude(Claude),
    /// Google Gemini models.
    Gemini(Gemini),
    /// GitHub Copilot models.
    Copilot(Copilot),
    /// `ChatGPT` Codex subscription models.
    Codex(Codex),
}

/// Unified embedding-capable cloud provider.
#[derive(Clone, Debug)]
pub enum CloudEmbedder {
    /// `OpenAI` embeddings.
    OpenAI(OpenAI),
    /// Google Gemini embeddings.
    Gemini(Gemini),
}

impl From<OpenAI> for CloudEmbedder {
    fn from(client: OpenAI) -> Self {
        Self::OpenAI(client)
    }
}

impl From<Gemini> for CloudEmbedder {
    fn from(client: Gemini) -> Self {
        Self::Gemini(client)
    }
}

/// Error returned when a cloud provider cannot be used as an embedder.
#[derive(Debug, thiserror::Error)]
pub enum CloudEmbedderError {
    /// The provider does not implement embeddings.
    #[error("provider does not support embeddings: {0}")]
    UnsupportedProvider(&'static str),
}

impl TryFrom<CloudProvider> for CloudEmbedder {
    type Error = CloudEmbedderError;

    fn try_from(value: CloudProvider) -> Result<Self, Self::Error> {
        match value {
            CloudProvider::OpenAI(client) => Ok(Self::OpenAI(client)),
            CloudProvider::Gemini(client) => Ok(Self::Gemini(client)),
            CloudProvider::Claude(_) => Err(CloudEmbedderError::UnsupportedProvider("claude")),
            CloudProvider::Copilot(_) => Err(CloudEmbedderError::UnsupportedProvider("copilot")),
            CloudProvider::Codex(_) => Err(CloudEmbedderError::UnsupportedProvider("codex")),
        }
    }
}

impl EmbeddingModel for CloudEmbedder {
    fn dim(&self) -> usize {
        match self {
            Self::OpenAI(client) => client.dim(),
            Self::Gemini(client) => client.dim(),
        }
    }

    fn embed(
        &self,
        text: &str,
    ) -> impl std::future::Future<Output = aither_core::Result<Vec<f32>>> + Send {
        let embedder = self.clone();
        let text = text.to_string();
        async move {
            match embedder {
                Self::OpenAI(client) => client.embed(text.as_str()).await,
                Self::Gemini(client) => client.embed(text.as_str()).await,
            }
        }
    }
}

/// Unified image-generation cloud provider.
#[derive(Clone, Debug)]
pub enum CloudImageGenerator {
    /// `OpenAI` image generation.
    OpenAI(OpenAI),
    /// Google Gemini image generation.
    Gemini(Gemini),
}

impl From<OpenAI> for CloudImageGenerator {
    fn from(client: OpenAI) -> Self {
        Self::OpenAI(client)
    }
}

impl From<Gemini> for CloudImageGenerator {
    fn from(client: Gemini) -> Self {
        Self::Gemini(client)
    }
}

/// Error returned when a cloud provider cannot be used as an image generator.
#[derive(Debug, thiserror::Error)]
pub enum CloudImageGeneratorError {
    /// The provider does not implement image generation.
    #[error("provider does not support image generation: {0}")]
    UnsupportedProvider(&'static str),
}

impl TryFrom<CloudProvider> for CloudImageGenerator {
    type Error = CloudImageGeneratorError;

    fn try_from(value: CloudProvider) -> Result<Self, Self::Error> {
        match value {
            CloudProvider::OpenAI(client) => Ok(Self::OpenAI(client)),
            CloudProvider::Gemini(client) => Ok(Self::Gemini(client)),
            CloudProvider::Claude(_) => {
                Err(CloudImageGeneratorError::UnsupportedProvider("claude"))
            }
            CloudProvider::Copilot(_) => {
                Err(CloudImageGeneratorError::UnsupportedProvider("copilot"))
            }
            CloudProvider::Codex(_) => Err(CloudImageGeneratorError::UnsupportedProvider("codex")),
        }
    }
}

impl ImageGenerator for CloudImageGenerator {
    type Error = CloudError;

    fn create(
        &self,
        prompt: ImagePrompt,
        size: ImageSize,
    ) -> impl Stream<Item = Result<ImageData, Self::Error>> + Send {
        let generator = self.clone();
        async_stream::stream! {
            match generator {
                Self::OpenAI(client) => {
                    let mut stream = std::pin::pin!(client.create(prompt, size));
                    while let Some(item) = stream.next().await {
                        yield item.map_err(CloudError::from);
                    }
                }
                Self::Gemini(client) => {
                    let mut stream = std::pin::pin!(client.create(prompt, size));
                    while let Some(item) = stream.next().await {
                        yield item.map_err(CloudError::from);
                    }
                }
            }
        }
    }

    fn edit(
        &self,
        prompt: ImagePrompt,
        mask: &[u8],
    ) -> impl Stream<Item = Result<ImageData, Self::Error>> + Send {
        let generator = self.clone();
        let mask = mask.to_vec();
        async_stream::stream! {
            match generator {
                Self::OpenAI(client) => {
                    let mut stream = std::pin::pin!(client.edit(prompt, &mask));
                    while let Some(item) = stream.next().await {
                        yield item.map_err(CloudError::from);
                    }
                }
                Self::Gemini(client) => {
                    let mut stream = std::pin::pin!(client.edit(prompt, &mask));
                    while let Some(item) = stream.next().await {
                        yield item.map_err(CloudError::from);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CloudEmbedder, CloudEmbedderError, CloudError, CloudImageGenerator,
        CloudImageGeneratorError, CloudProvider,
    };
    use crate::{Claude, Codex, Copilot, Gemini, OpenAI, codex::CodexCredentials};
    use aither_core::{
        ImageGenerator, LanguageModel,
        image::{Prompt, Size},
        llm::oneshot,
    };
    use futures_lite::StreamExt;
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[test]
    fn openai_and_gemini_convert_to_cloud_embedder() {
        assert!(matches!(
            CloudEmbedder::try_from(CloudProvider::OpenAI(OpenAI::new("test"))),
            Ok(CloudEmbedder::OpenAI(_))
        ));
        assert!(matches!(
            CloudEmbedder::try_from(CloudProvider::Gemini(Gemini::new("test"))),
            Ok(CloudEmbedder::Gemini(_))
        ));
    }

    #[test]
    fn unsupported_cloud_providers_fail_fast() {
        assert!(matches!(
            CloudEmbedder::try_from(CloudProvider::Claude(Claude::new("test"))),
            Err(CloudEmbedderError::UnsupportedProvider("claude"))
        ));
        assert!(matches!(
            CloudEmbedder::try_from(CloudProvider::Copilot(Copilot::new("test"))),
            Err(CloudEmbedderError::UnsupportedProvider("copilot"))
        ));
    }

    #[test]
    fn cloud_image_generator_converts_supported_and_rejects_unsupported() {
        assert!(matches!(
            CloudImageGenerator::try_from(CloudProvider::OpenAI(OpenAI::new("test"))),
            Ok(CloudImageGenerator::OpenAI(_))
        ));
        assert!(matches!(
            CloudImageGenerator::try_from(CloudProvider::Gemini(Gemini::new("test"))),
            Ok(CloudImageGenerator::Gemini(_))
        ));
        assert!(matches!(
            CloudImageGenerator::try_from(CloudProvider::Claude(Claude::new("test"))),
            Err(CloudImageGeneratorError::UnsupportedProvider("claude"))
        ));
        assert!(matches!(
            CloudImageGenerator::try_from(CloudProvider::Copilot(Copilot::new("test"))),
            Err(CloudImageGeneratorError::UnsupportedProvider("copilot"))
        ));
        let codex = Codex::new(CodexCredentials::new("at", "rt"), "gpt-5.3-codex");
        assert!(matches!(
            CloudImageGenerator::try_from(CloudProvider::Codex(codex)),
            Err(CloudImageGeneratorError::UnsupportedProvider("codex"))
        ));
    }

    #[tokio::test]
    async fn cloud_provider_generate_forwards_native_gemini_schema_generation() {
        use std::io::{Read, Write};

        #[derive(Debug, Deserialize, JsonSchema, PartialEq, Eq)]
        struct StructuredAnswer {
            answer: i32,
        }

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind stub");
        let addr = listener.local_addr().expect("addr");
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut req = Vec::new();
            let mut buf = [0u8; 4096];
            let body_start = loop {
                let n = stream.read(&mut buf).expect("read");
                req.extend_from_slice(&buf[..n]);
                if let Some(pos) = req.windows(4).position(|w| w == b"\r\n\r\n") {
                    break pos + 4;
                }
            };
            let head = String::from_utf8_lossy(&req[..body_start]).to_ascii_lowercase();
            let chunked = head.contains("transfer-encoding: chunked");
            let content_len = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            while if chunked {
                !req.ends_with(b"0\r\n\r\n")
            } else {
                req.len() < body_start + content_len
            } {
                let n = stream.read(&mut buf).expect("read body");
                if n == 0 {
                    break;
                }
                req.extend_from_slice(&buf[..n]);
            }
            let raw = String::from_utf8_lossy(&req[body_start..]).into_owned();
            let body = if chunked {
                let mut decoded = String::new();
                let mut rest = raw.as_str();
                while let Some((size_hex, after)) = rest.split_once("\r\n") {
                    let size = usize::from_str_radix(size_hex.trim(), 16).expect("chunk hex");
                    if size == 0 {
                        break;
                    }
                    decoded.push_str(after.get(..size).expect("chunk bytes"));
                    rest = after.get(size + 2..).unwrap_or("");
                }
                decoded
            } else {
                raw
            };
            tx.send(body).expect("send body");

            let sse = "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"{\\\"answer\\\":42}\"}]},\"finishReason\":\"STOP\"}]}\n\n";
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{sse}",
                sse.len()
            );
            let _ = stream.write_all(resp.as_bytes());
        });

        let provider = CloudProvider::Gemini(
            Gemini::new("secret-key").with_base_url(format!("http://{addr}/v1beta")),
        );
        let out: StructuredAnswer =
            Box::pin(provider.generate(oneshot("You are a calculator.", "6 * 7")))
                .await
                .expect("structured generate succeeds");
        assert_eq!(out, StructuredAnswer { answer: 42 });

        let request_body: serde_json::Value =
            serde_json::from_str(&rx.recv().expect("captured body")).expect("json request");
        assert_eq!(
            request_body["generationConfig"]["responseMimeType"],
            "application/json"
        );
        assert!(request_body["generationConfig"]["responseJsonSchema"].is_object());
    }

    #[tokio::test]
    async fn cloud_image_generator_propagates_provider_errors() {
        let generator = CloudImageGenerator::Gemini(Gemini::new("key").without_image_model());
        let mut stream = std::pin::pin!(generator.create(Prompt::new("a fox"), Size::square(512)));
        let err = stream
            .next()
            .await
            .expect("stream yields error")
            .expect_err("disabled image model errors");
        assert!(matches!(err, CloudError::Gemini(_)));
    }
}

impl From<OpenAI> for CloudProvider {
    fn from(client: OpenAI) -> Self {
        Self::OpenAI(client)
    }
}

impl From<Claude> for CloudProvider {
    fn from(client: Claude) -> Self {
        Self::Claude(client)
    }
}

impl From<Gemini> for CloudProvider {
    fn from(client: Gemini) -> Self {
        Self::Gemini(client)
    }
}

impl From<Copilot> for CloudProvider {
    fn from(client: Copilot) -> Self {
        Self::Copilot(client)
    }
}

impl From<Codex> for CloudProvider {
    fn from(client: Codex) -> Self {
        Self::Codex(client)
    }
}

impl std::fmt::Debug for CloudProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAI(_) => f.debug_tuple("CloudProvider::OpenAI").finish(),
            Self::Claude(_) => f.debug_tuple("CloudProvider::Claude").finish(),
            Self::Gemini(_) => f.debug_tuple("CloudProvider::Gemini").finish(),
            Self::Copilot(_) => f.debug_tuple("CloudProvider::Copilot").finish(),
            Self::Codex(_) => f.debug_tuple("CloudProvider::Codex").finish(),
        }
    }
}

/// Error type for cloud provider operations.
#[derive(Debug, thiserror::Error)]
pub enum CloudError {
    /// `OpenAI` API error.
    #[error("OpenAI error: {0}")]
    OpenAI(#[from] aither_openai::OpenAIError),
    /// Claude API error.
    #[error("Claude error: {0}")]
    Claude(#[from] aither_claude::ClaudeError),
    /// Gemini API error.
    #[error("Gemini error: {0}")]
    Gemini(#[from] aither_gemini::GeminiError),
    /// GitHub Copilot API error.
    #[error("Copilot error: {0}")]
    Copilot(#[from] aither_copilot::CopilotError),
    /// `ChatGPT` Codex API error.
    #[error("Codex error: {0}")]
    Codex(#[from] aither_codex::CodexError),
}

impl Classify for CloudError {
    fn failure(&self) -> aither_core::llm::Failure {
        match self {
            Self::OpenAI(error) => error.failure(),
            Self::Claude(error) => error.failure(),
            Self::Gemini(error) => error.failure(),
            Self::Copilot(error) => error.failure(),
            Self::Codex(error) => error.failure(),
        }
    }
}

/// Rewraps a `count_tokens` failure as a `CloudError`, keeping parse
/// failures (which name no provider) intact.
fn map_cloud_error<E>(error: GenerateError<E>) -> GenerateError<CloudError>
where
    CloudError: From<E>,
{
    match error {
        GenerateError::Provider(source) => GenerateError::Provider(CloudError::from(source)),
        GenerateError::Parse { source, response } => GenerateError::Parse { source, response },
    }
}

impl LanguageModel for CloudProvider {
    type Error = CloudError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let provider = match self {
            Self::OpenAI(inner) => ProviderInner::OpenAI(inner.clone()),
            Self::Claude(inner) => ProviderInner::Claude(inner.clone()),
            Self::Gemini(inner) => ProviderInner::Gemini(inner.clone()),
            Self::Copilot(inner) => ProviderInner::Copilot(inner.clone()),
            Self::Codex(inner) => ProviderInner::Codex(inner.clone()),
        };

        async_stream::stream! {
            match provider {
                ProviderInner::OpenAI(inner) => {
                    let mut stream = std::pin::pin!(inner.respond(request));
                    while let Some(result) = stream.next().await {
                        yield result.map_err(CloudError::from);
                    }
                }
                ProviderInner::Claude(inner) => {
                    let mut stream = std::pin::pin!(inner.respond(request));
                    while let Some(result) = stream.next().await {
                        yield result.map_err(CloudError::from);
                    }
                }
                ProviderInner::Gemini(inner) => {
                    let mut stream = std::pin::pin!(inner.respond(request));
                    while let Some(result) = stream.next().await {
                        yield result.map_err(CloudError::from);
                    }
                }
                ProviderInner::Copilot(inner) => {
                    let mut stream = std::pin::pin!(inner.respond(request));
                    while let Some(result) = stream.next().await {
                        yield result.map_err(CloudError::from);
                    }
                }
                ProviderInner::Codex(inner) => {
                    let mut stream = std::pin::pin!(inner.respond(request));
                    while let Some(result) = stream.next().await {
                        yield result.map_err(CloudError::from);
                    }
                }
            }
        }
    }

    fn generate<T: JsonSchema + DeserializeOwned + 'static>(
        &self,
        request: LLMRequest,
    ) -> impl std::future::Future<Output = Result<T, GenerateError<Self::Error>>> + Send {
        let provider = self.clone();
        async move {
            match provider {
                Self::OpenAI(inner) => Box::pin(inner.generate(request))
                    .await
                    .map_err(map_cloud_error),
                Self::Claude(inner) => Box::pin(inner.generate(request))
                    .await
                    .map_err(map_cloud_error),
                Self::Gemini(inner) => Box::pin(inner.generate(request))
                    .await
                    .map_err(map_cloud_error),
                Self::Copilot(inner) => Box::pin(inner.generate(request))
                    .await
                    .map_err(map_cloud_error),
                Self::Codex(inner) => Box::pin(inner.generate(request))
                    .await
                    .map_err(map_cloud_error),
            }
        }
    }

    /// `count_tokens` reports `GenerateError` over each provider's own error
    /// type; rewrap the provider arm so callers see one `CloudError`.
    fn token_counting(&self) -> TokenCounting {
        match self {
            Self::OpenAI(inner) => inner.token_counting(),
            Self::Claude(inner) => inner.token_counting(),
            Self::Gemini(inner) => inner.token_counting(),
            Self::Copilot(inner) => inner.token_counting(),
            Self::Codex(inner) => inner.token_counting(),
        }
    }

    async fn count_tokens(
        &self,
        request: &LLMRequest,
    ) -> Result<Option<u32>, GenerateError<Self::Error>> {
        match self {
            Self::OpenAI(inner) => inner.count_tokens(request).await.map_err(map_cloud_error),
            Self::Claude(inner) => inner.count_tokens(request).await.map_err(map_cloud_error),
            Self::Gemini(inner) => inner.count_tokens(request).await.map_err(map_cloud_error),
            Self::Copilot(inner) => inner.count_tokens(request).await.map_err(map_cloud_error),
            Self::Codex(inner) => inner.count_tokens(request).await.map_err(map_cloud_error),
        }
    }

    fn count_text(&self, text: &str) -> Option<u32> {
        match self {
            Self::OpenAI(inner) => inner.count_text(text),
            Self::Claude(inner) => inner.count_text(text),
            Self::Gemini(inner) => inner.count_text(text),
            Self::Copilot(inner) => inner.count_text(text),
            Self::Codex(inner) => inner.count_text(text),
        }
    }

    fn retry_policy(&self) -> RetryPolicy {
        match self {
            Self::OpenAI(inner) => inner.retry_policy(),
            Self::Claude(inner) => inner.retry_policy(),
            Self::Gemini(inner) => inner.retry_policy(),
            Self::Copilot(inner) => inner.retry_policy(),
            Self::Codex(inner) => inner.retry_policy(),
        }
    }

    async fn profile(&self) -> Result<Profile, Self::Error> {
        match self {
            Self::OpenAI(inner) => inner.profile().await.map_err(CloudError::from),
            Self::Claude(inner) => inner.profile().await.map_err(CloudError::from),
            Self::Gemini(inner) => inner.profile().await.map_err(CloudError::from),
            Self::Copilot(inner) => inner.profile().await.map_err(CloudError::from),
            Self::Codex(inner) => inner.profile().await.map_err(CloudError::from),
        }
    }
}

/// Internal enum to allow cloning the provider for async block.
enum ProviderInner {
    OpenAI(OpenAI),
    Claude(Claude),
    Gemini(Gemini),
    Copilot(Copilot),
    Codex(Codex),
}

/// Unified model provider wrapping `OpenAI`, Claude, Gemini, and Copilot providers.
///
/// Implements `LanguageModelProvider` to allow unified model listing and instantiation.
#[derive(Clone, Debug)]
pub enum CloudModelProvider {
    /// `OpenAI` provider.
    OpenAI(OpenAIProvider),
    /// Anthropic Claude provider.
    Claude(ClaudeProvider),
    /// Google Gemini provider.
    Gemini(GeminiProvider),
    /// GitHub Copilot provider.
    Copilot(CopilotProvider),
    /// `ChatGPT` Codex provider.
    Codex(CodexProvider),
}

impl From<OpenAIProvider> for CloudModelProvider {
    fn from(provider: OpenAIProvider) -> Self {
        Self::OpenAI(provider)
    }
}

impl From<ClaudeProvider> for CloudModelProvider {
    fn from(provider: ClaudeProvider) -> Self {
        Self::Claude(provider)
    }
}

impl From<GeminiProvider> for CloudModelProvider {
    fn from(provider: GeminiProvider) -> Self {
        Self::Gemini(provider)
    }
}

impl From<CopilotProvider> for CloudModelProvider {
    fn from(provider: CopilotProvider) -> Self {
        Self::Copilot(provider)
    }
}

impl From<CodexProvider> for CloudModelProvider {
    fn from(provider: CodexProvider) -> Self {
        Self::Codex(provider)
    }
}

impl LanguageModelProvider for CloudModelProvider {
    type Model = CloudProvider;
    type Error = CloudError;

    fn list_models(
        &self,
    ) -> impl std::future::Future<Output = Result<Vec<Profile>, Self::Error>> + Send {
        let provider = self.clone();
        async move {
            match provider {
                Self::OpenAI(p) => p.list_models().await.map_err(CloudError::from),
                Self::Claude(p) => p.list_models().await.map_err(CloudError::from),
                Self::Gemini(p) => p.list_models().await.map_err(CloudError::from),
                Self::Copilot(p) => p.list_models().await.map_err(CloudError::from),
                Self::Codex(p) => p.list_models().await.map_err(CloudError::from),
            }
        }
    }

    fn get_model(
        &self,
        name: &str,
    ) -> impl std::future::Future<Output = Result<Self::Model, Self::Error>> + Send {
        let provider = self.clone();
        let name = name.to_string();
        async move {
            match provider {
                Self::OpenAI(p) => p
                    .get_model(&name)
                    .await
                    .map(CloudProvider::from)
                    .map_err(CloudError::from),
                Self::Claude(p) => p
                    .get_model(&name)
                    .await
                    .map(CloudProvider::from)
                    .map_err(CloudError::from),
                Self::Gemini(p) => p
                    .get_model(&name)
                    .await
                    .map(CloudProvider::from)
                    .map_err(CloudError::from),
                Self::Copilot(p) => p
                    .get_model(&name)
                    .await
                    .map(CloudProvider::from)
                    .map_err(CloudError::from),
                Self::Codex(p) => p
                    .get_model(&name)
                    .await
                    .map(CloudProvider::from)
                    .map_err(CloudError::from),
            }
        }
    }

    fn profile() -> ProviderProfile {
        ProviderProfile::new("cloud", "Unified cloud provider")
    }
}
