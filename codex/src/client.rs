//! [`LanguageModel`] implementation backed by the `ChatGPT` Codex Responses
//! backend.
//!
//! [`Codex`] wraps [`aither_openai::Client`] pointed at
//! `https://chatgpt.com/backend-api/codex` with subscription-auth request
//! headers. It owns the OAuth token lifecycle: expired access tokens are
//! refreshed lazily inside `respond`, a `401` mid-stream triggers one forced
//! refresh and retry, and rotated credentials are pushed to a persistence
//! callback so the refresh token (single-use) is never lost.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use aither_core::llm::model::{Ability, Profile as ModelProfile};
use aither_core::llm::{
    Event, GenerateError, LLMRequest, LanguageModel, RetryPolicy, TokenCounting,
};
use aither_openai::OpenAI;
use async_lock::Mutex;
use futures_lite::StreamExt as _;

use crate::auth::{CodexCredentials, refresh_tokens_at};
use crate::constant::{ACCOUNT_ID_HEADER, CODEX_BASE_URL, ORIGINATOR, RESIDENCY_HEADER, TOKEN_URL};
use crate::error::CodexError;

/// Refresh the access token this long before its recorded expiry.
const EXPIRY_SKEW: Duration = Duration::from_secs(60);

/// Residency sentinel that means "no constraint" — the header is omitted.
const NO_RESIDENCY_CONSTRAINT: &str = "no_constraint";

/// Callback invoked whenever the credential set changes (token rotation).
///
/// The implementation is expected to persist the credentials; it is invoked
/// synchronously while the token lock is held, so it should schedule any I/O
/// rather than perform it inline.
pub type OnRefresh = Arc<dyn Fn(&CodexCredentials) + Send + Sync>;

pub async fn resolve_shared_credentials(
    shared: &Mutex<CodexCredentials>,
    token_url: &str,
    on_refresh: Option<&OnRefresh>,
    force: bool,
    failed_access_token: Option<&str>,
) -> Result<CodexCredentials, CodexError> {
    let mut credentials = shared.lock().await;
    if !force && !credentials.is_expired(EXPIRY_SKEW) {
        return Ok(credentials.clone());
    }
    if force
        && let Some(failed) = failed_access_token
        && credentials.access_token != failed
    {
        return Ok(credentials.clone());
    }
    let Some(refresh_token) = credentials.refresh_token_opt() else {
        if !force && !credentials.is_expired(Duration::ZERO) {
            return Ok(credentials.clone());
        }
        return Err(CodexError::Api(
            "ChatGPT access token expired and no refresh token is configured".into(),
        ));
    };

    let fresh = refresh_tokens_at(refresh_token, token_url).await?;
    tracing::debug!("refreshed ChatGPT access token");
    if let Some(callback) = on_refresh {
        callback(&fresh);
    }
    *credentials = fresh.clone();
    drop(credentials);
    Ok(fresh)
}

/// A [`LanguageModel`] for a single model on the Codex subscription backend.
#[derive(Clone)]
pub struct Codex {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    model: String,
    base_url: String,
    token_url: String,
    originator: String,
    session_id: String,
    shared: Arc<Mutex<CodexCredentials>>,
    on_refresh: Option<OnRefresh>,
    template_client: OpenAI,
}

pub fn codex_from_shared(
    shared: Arc<Mutex<CodexCredentials>>,
    model: impl Into<String>,
    base_url: impl Into<String>,
    token_url: impl Into<String>,
    originator: impl Into<String>,
    on_refresh: Option<OnRefresh>,
) -> Codex {
    let model = model.into();
    let base_url = base_url.into();
    let template_client = OpenAI::builder("codex-template")
        .base_url(&base_url)
        .model(&model)
        .store(false)
        .build();
    Codex {
        inner: Arc::new(Inner {
            model,
            base_url,
            token_url: token_url.into(),
            originator: originator.into(),
            session_id: session_id(),
            shared,
            on_refresh,
            template_client,
        }),
    }
}

impl Codex {
    /// Create a client for `model` using the given credentials.
    #[must_use]
    pub fn new(credentials: CodexCredentials, model: impl Into<String>) -> Self {
        codex_from_shared(
            Arc::new(Mutex::new(credentials)),
            model,
            CODEX_BASE_URL,
            TOKEN_URL,
            ORIGINATOR,
            None,
        )
    }

    /// Override the backend base URL (defaults to the Codex subscription
    /// endpoint).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        let inner = Arc::make_mut(&mut self.inner);
        inner.template_client = inner.template_client.clone().with_base_url(&base_url);
        inner.base_url = base_url;
        self
    }

    /// Override the `originator` header (defaults to [`ORIGINATOR`]).
    #[must_use]
    pub fn with_originator(mut self, originator: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).originator = originator.into();
        self
    }

    /// Register a callback fired whenever tokens rotate.
    #[must_use]
    pub fn on_refresh(
        mut self,
        callback: impl Fn(&CodexCredentials) + Send + Sync + 'static,
    ) -> Self {
        Arc::make_mut(&mut self.inner).on_refresh = Some(Arc::new(callback));
        self
    }

    /// A snapshot of the current credentials.
    pub async fn credentials(&self) -> CodexCredentials {
        self.inner.shared.lock().await.clone()
    }
}

impl Inner {
    /// Build the inner `OpenAI` Responses client for a credential set.
    fn openai_client(&self, credentials: &CodexCredentials) -> OpenAI {
        let mut builder = OpenAI::builder(&credentials.access_token)
            .base_url(&self.base_url)
            .model(&self.model)
            .store(false)
            .extra_header("originator", &self.originator)
            .extra_header("session-id", &self.session_id);
        if let Some(account_id) = &credentials.account_id {
            builder = builder.extra_header(ACCOUNT_ID_HEADER, account_id);
        }
        if let Some(residency) = credentials
            .residency
            .as_deref()
            .filter(|r| *r != NO_RESIDENCY_CONSTRAINT)
        {
            builder = builder.extra_header(RESIDENCY_HEADER, residency);
        }
        builder.build()
    }

    async fn can_refresh(&self) -> bool {
        self.shared.lock().await.can_refresh()
    }

    /// Build a request-ready client (fresh token + credential headers), returning
    /// the access token used so single-flight 401 retries know whether a sibling
    /// already rotated it.
    async fn request_client(
        &self,
        force_refresh: bool,
        failed_access_token: Option<&str>,
    ) -> Result<(OpenAI, String), CodexError> {
        let credentials = resolve_shared_credentials(
            &self.shared,
            &self.token_url,
            self.on_refresh.as_ref(),
            force_refresh,
            failed_access_token,
        )
        .await?;
        let client = self.openai_client(&credentials);
        Ok((client, credentials.access_token))
    }
}

impl fmt::Debug for Codex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Codex")
            .field("model", &self.inner.model)
            .field("base_url", &self.inner.base_url)
            .field("originator", &self.inner.originator)
            .field("session_id", &self.inner.session_id)
            .finish_non_exhaustive()
    }
}

impl LanguageModel for Codex {
    type Error = CodexError;

    fn token_counting(&self) -> TokenCounting {
        self.inner.template_client.token_counting()
    }

    async fn count_tokens(
        &self,
        request: &LLMRequest,
    ) -> Result<Option<u32>, GenerateError<Self::Error>> {
        self.inner
            .template_client
            .count_tokens(request)
            .await
            .map_err(|err| match err {
                GenerateError::Provider(source) => {
                    GenerateError::Provider(CodexError::OpenAI(source))
                }
                GenerateError::Parse { source, response } => {
                    GenerateError::Parse { source, response }
                }
            })
    }

    fn count_text(&self, text: &str) -> Option<u32> {
        self.inner.template_client.count_text(text)
    }

    fn retry_policy(&self) -> RetryPolicy {
        self.inner.template_client.retry_policy()
    }

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl futures_core::Stream<Item = Result<Event, Self::Error>> + Send {
        let inner = Arc::clone(&self.inner);
        async_stream::stream! {
            let mut retried = false;
            let mut failed_token: Option<String> = None;
            loop {
                let (client, used_token) =
                    match inner.request_client(retried, failed_token.as_deref()).await {
                        Ok(pair) => pair,
                        Err(err) => {
                            yield Err(err);
                            return;
                        }
                    };
                let mut stream = std::pin::pin!(client.respond(request.clone()));
                let mut unauthorized = false;
                let mut yielded_any = false;
                while let Some(event) = stream.next().await {
                    match event {
                        Err(err)
                            if is_unauthorized(&err)
                                && !retried
                                && !yielded_any
                                && inner.can_refresh().await =>
                        {
                            unauthorized = true;
                            failed_token = Some(used_token.clone());
                            break;
                        }
                        event => {
                            yielded_any = true;
                            yield event.map_err(CodexError::OpenAI);
                        }
                    }
                }
                if !unauthorized {
                    return;
                }
                tracing::warn!("Codex request unauthorized; refreshing token and retrying");
                retried = true;
            }
        }
    }

    fn profile(
        &self,
    ) -> impl core::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
        let model = self.inner.model.clone();
        async move {
            let entry = aither_models::lookup(&model);
            let context_length = entry
                .and_then(aither_models::ModelEntry::max_input_tokens)
                .ok_or_else(|| {
                    CodexError::Api(format!(
                        "cannot determine context length for model '{model}': \
                         aither-models has no entry"
                    ))
                })?;
            let mut profile = ModelProfile::new(
                model.clone(),
                "openai",
                model,
                "OpenAI model through a ChatGPT subscription",
                context_length,
            )
            .with_ability(Ability::ToolUse);
            if let Some(entry) = entry {
                // The registry's per-message effort is the public Responses
                // API's; the ChatGPT backend does not document it.
                for ability in entry
                    .abilities()
                    .iter()
                    .filter(|ability| **ability != Ability::PerMessageEffort)
                {
                    if !profile.abilities.contains(ability) {
                        profile.abilities.push(*ability);
                    }
                }
            }
            Ok(profile)
        }
    }
}

/// Whether an `OpenAIError` is an HTTP 401.
const fn is_unauthorized(err: &aither_openai::OpenAIError) -> bool {
    matches!(
        err,
        aither_openai::OpenAIError::Http(zenwave::Error::Http { status, .. })
            if status.as_u16() == 401
    )
}

/// Random session identifier for the `session-id` header.
fn session_id() -> String {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("secure random source unavailable");
    bytes.iter().fold(String::with_capacity(32), |mut s, b| {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
        s
    })
}
