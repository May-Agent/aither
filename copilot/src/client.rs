//! GitHub Copilot client implementation.

use crate::{
    CopilotError,
    constant::{
        COPILOT_BASE_URL, COPILOT_INTEGRATION_ID, COPILOT_TOKEN_URL, DEFAULT_MODEL, EDITOR_VERSION,
    },
};
use aither_core::time::sleep;
use aither_core::{
    LanguageModel,
    llm::{
        Event, LLMRequest, Message, Role, ToolCall, Usage,
        model::{
            Ability, OpenAIPromptCacheRetention, Parameters, Profile as ModelProfile, ToolChoice,
        },
        tool::ToolDefinition,
    },
};
use async_lock::Mutex;
use futures_core::Stream;
use futures_lite::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use zenwave::{Client, client, header};

/// Shared short-lived session token and endpoint refreshed on 401 responses.
#[derive(Clone)]
pub struct SessionState {
    pub token: String,
    pub base_url: String,
}

impl std::fmt::Debug for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionState")
            .field("token", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .finish()
    }
}

pub fn copilot_from_shared(
    session: Arc<Mutex<SessionState>>,
    token_exchange_url: String,
    model: &str,
    editor_version: String,
    integration_id: String,
    oauth_token: Option<String>,
) -> Copilot {
    Copilot {
        inner: Arc::new(Config {
            session,
            token_exchange_url,
            model: model.trim().to_string(),
            editor_version,
            integration_id,
            oauth_token,
        }),
    }
}

pub async fn refresh_shared_session(
    session: &Mutex<SessionState>,
    failed_token: &str,
    oauth_token: Option<&str>,
    token_exchange_url: &str,
) -> Result<Option<SessionState>, CopilotError> {
    let Some(oauth_token) = oauth_token else {
        return Ok(None);
    };

    let mut guard = session.lock().await;
    if guard.token != failed_token {
        return Ok(Some(guard.clone()));
    }

    let fresh = crate::auth::get_session_token_at(oauth_token, token_exchange_url).await?;
    guard.token = fresh.token;
    guard.base_url = fresh.api_endpoint;
    Ok(Some(guard.clone()))
}

/// GitHub Copilot language model client.
///
/// Uses the OpenAI-compatible chat completions API at `api.githubcopilot.com`.
#[derive(Clone, Debug)]
pub struct Copilot {
    inner: Arc<Config>,
}

impl Copilot {
    /// Create a new Copilot client with the given OAuth token.
    pub fn new(token: impl Into<String>) -> Self {
        Self::builder(token).build()
    }

    /// Create a builder for configuring the Copilot client.
    #[must_use]
    pub fn builder(token: impl Into<String>) -> Builder {
        Builder::new(token)
    }

    /// Override the default chat model.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).model = model.into().trim().to_string();
        self
    }

    /// Override the REST base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        let cfg = Arc::make_mut(&mut self.inner);
        if Arc::strong_count(&cfg.session) == 1 {
            cfg.session.lock_blocking().base_url = base_url;
        } else {
            let token = cfg.session.lock_blocking().token.clone();
            cfg.session = Arc::new(Mutex::new(SessionState { token, base_url }));
        }
        self
    }

    /// Provide an OAuth token so the client can refresh session tokens on 401s.
    #[must_use]
    pub fn with_oauth_token(mut self, token: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).oauth_token = Some(token.into());
        self
    }
}

impl LanguageModel for Copilot {
    type Error = CopilotError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let cfg = self.inner.clone();
        let (messages, parameters, tool_defs) = request.into_parts();
        let tool_defs = filter_tool_definitions(tool_defs, &parameters.tool_choice);

        async_stream::stream! {
            let payload_messages = match to_chat_messages(&messages).await {
                Ok(messages) => messages,
                Err(error) => {
                    yield Err(CopilotError::Api(error));
                    return;
                }
            };
            let openai_tools = if tool_defs.is_empty() {
                None
            } else {
                Some(convert_tools(&tool_defs))
            };

            let mut events = chat_completions_stream(cfg, payload_messages, &parameters, openai_tools);

            while let Some(event) = events.next().await {
                yield event;
            }
        }
    }

    fn profile(
        &self,
    ) -> impl std::future::Future<Output = Result<ModelProfile, Self::Error>> + Send {
        let cfg = self.inner.clone();
        async move {
            // Try to get context window from models database
            let context_length = aither_models::lookup(&cfg.model)
                .and_then(aither_models::ModelEntry::max_input_tokens)
                .ok_or_else(|| {
                    CopilotError::Api(format!(
                        "cannot determine context length for model '{model}': \
                         aither-models has no entry",
                        model = cfg.model,
                    ))
                })?;

            Ok(ModelProfile::new(
                cfg.model.clone(),
                "GitHub Copilot",
                cfg.model.clone(),
                "GitHub Copilot model",
                context_length,
            )
            .with_ability(Ability::ToolUse))
        }
    }
}

#[derive(Clone)]
struct Config {
    session: Arc<Mutex<SessionState>>,
    token_exchange_url: String,
    model: String,
    editor_version: String,
    integration_id: String,
    oauth_token: Option<String>,
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let base_url = self
            .session
            .try_lock()
            .map_or_else(|| "<locked>".to_string(), |guard| guard.base_url.clone());
        f.debug_struct("Config")
            .field("token", &"[REDACTED]")
            .field("base_url", &base_url)
            .field("model", &self.model)
            .field("editor_version", &self.editor_version)
            .field("integration_id", &self.integration_id)
            .field(
                "oauth_token",
                &self.oauth_token.as_ref().map(|_| "[REDACTED]"),
            )
            .finish_non_exhaustive()
    }
}

/// Builder for [`Copilot`] clients.
pub struct Builder {
    token: String,
    base_url: String,
    token_exchange_url: String,
    model: String,
    editor_version: String,
    integration_id: String,
    oauth_token: Option<String>,
}

impl std::fmt::Debug for Builder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Builder")
            .field("token", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("editor_version", &self.editor_version)
            .field("integration_id", &self.integration_id)
            .field(
                "oauth_token",
                &self.oauth_token.as_ref().map(|_| "[REDACTED]"),
            )
            .finish_non_exhaustive()
    }
}

impl Builder {
    fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            base_url: COPILOT_BASE_URL.to_string(),
            token_exchange_url: COPILOT_TOKEN_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            editor_version: EDITOR_VERSION.to_string(),
            integration_id: COPILOT_INTEGRATION_ID.to_string(),
            oauth_token: None,
        }
    }

    /// Set a custom base URL.
    #[must_use]
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the model identifier.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into().trim().to_string();
        self
    }

    /// Set the editor version header.
    #[must_use]
    pub fn editor_version(mut self, version: impl Into<String>) -> Self {
        self.editor_version = version.into();
        self
    }

    /// Set the integration ID header.
    #[must_use]
    pub fn integration_id(mut self, id: impl Into<String>) -> Self {
        self.integration_id = id.into();
        self
    }

    /// Provide an OAuth token so the client can refresh session tokens on 401s.
    #[must_use]
    pub fn oauth_token(mut self, token: impl Into<String>) -> Self {
        self.oauth_token = Some(token.into());
        self
    }

    /// Build the Copilot client.
    #[must_use]
    pub fn build(self) -> Copilot {
        Copilot {
            inner: Arc::new(Config {
                session: Arc::new(Mutex::new(SessionState {
                    token: self.token,
                    base_url: self.base_url,
                })),
                token_exchange_url: self.token_exchange_url,
                model: self.model,
                editor_version: self.editor_version,
                integration_id: self.integration_id,
                oauth_token: self.oauth_token,
            }),
        }
    }
}

// === Request/Response Types ===

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessagePayload>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolPayload>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoicePayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<&'static str>,
}

#[derive(Debug, Serialize, Clone)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize, Clone)]
struct ChatMessagePayload {
    role: &'static str,
    content: ContentPayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCallPayload>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum ContentPayload {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlPayload },
}

#[derive(Debug, Clone, Serialize)]
struct ImageUrlPayload {
    url: String,
}

#[derive(Debug, Serialize, Clone)]
struct ChatToolCallPayload {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: ChatToolCallFunction,
}

#[derive(Debug, Serialize, Clone)]
struct ChatToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize, Clone)]
struct ToolPayload {
    r#type: &'static str,
    function: ToolFunction,
}

#[derive(Debug, Serialize, Clone)]
struct ToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ToolChoicePayload {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Serialize, Clone)]
struct ToolChoiceFunction {
    name: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    #[serde(default)]
    usage: Option<ChatCompletionUsage>,
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct ChatCompletionUsage {
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokenDetails>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokenDetails>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct PromptTokenDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct CompletionTokenDetails {
    #[serde(default)]
    reasoning_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct ChunkDelta {
    content: Option<String>,
    #[serde(
        default,
        alias = "reasoning",
        alias = "reasoning_text",
        alias = "reasoningContent",
        alias = "thinking",
        alias = "thinking_content",
        alias = "thinkingContent"
    )]
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolCall {
    index: Option<usize>,
    id: Option<String>,
    function: Option<ChunkToolFunction>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolFunction {
    name: Option<String>,
    arguments: Option<String>,
}

// === Conversion Functions ===

async fn to_chat_messages(messages: &[Message]) -> Result<Vec<ChatMessagePayload>, String> {
    let mut payloads = Vec::with_capacity(messages.len());
    for msg in messages {
        let (role, tool_calls, tool_call_id) = match msg.role() {
            Role::Configuration => {
                return Err("Copilot takes no per-message effort update".to_string());
            }
            Role::System => ("system", None, None),
            Role::User => ("user", None, None),
            Role::Assistant => {
                let calls = if msg.tool_calls().is_empty() {
                    None
                } else {
                    Some(
                        msg.tool_calls()
                            .iter()
                            .map(|tc| ChatToolCallPayload {
                                id: tc.id.clone(),
                                kind: "function",
                                function: ChatToolCallFunction {
                                    name: tc.name.clone(),
                                    arguments: tc.arguments.to_string(),
                                },
                            })
                            .collect(),
                    )
                };
                ("assistant", calls, None)
            }
            Role::Tool => (
                "tool",
                None,
                Some(msg.tool_call_id().unwrap_or_default().to_string()),
            ),
        };
        if matches!(msg.role(), Role::Tool) && !msg.attachments().is_empty() {
            payloads.push(ChatMessagePayload {
                role,
                content: ContentPayload::Text(msg.content().to_owned()),
                tool_calls,
                tool_call_id,
            });
            let attachment_only = Message::user("").with_attachments(msg.attachments().to_vec());
            payloads.push(ChatMessagePayload {
                role: "user",
                content: build_content(&attachment_only).await?,
                tool_calls: None,
                tool_call_id: None,
            });
        } else {
            payloads.push(ChatMessagePayload {
                role,
                content: build_content(msg).await?,
                tool_calls,
                tool_call_id,
            });
        }
    }
    Ok(payloads)
}

async fn build_content(message: &Message) -> Result<ContentPayload, String> {
    let attachments = message.attachments();

    if attachments.is_empty() {
        return Ok(ContentPayload::Text(message.content().to_owned()));
    }

    let mut parts = Vec::with_capacity(attachments.len() + 1);
    for attachment in attachments {
        let media_type = attachment.media_type().as_ref();
        if !media_type.starts_with("image/") {
            return Err(format!(
                "Copilot Chat Completions does not support attachment MIME type '{media_type}'"
            ));
        }
        let data_url = url_to_data_url(attachment.url(), media_type).await?;
        parts.push(ContentPart::ImageUrl {
            image_url: ImageUrlPayload { url: data_url },
        });
    }

    if !message.content().is_empty() {
        parts.push(ContentPart::Text {
            text: message.content().to_owned(),
        });
    }

    Ok(ContentPayload::Parts(parts))
}

async fn url_to_data_url(url: &url::Url, media_type: &str) -> Result<String, String> {
    match url.scheme() {
        "data" | "http" | "https" => Ok(url.as_str().to_string()),
        "file" => read_file_to_data_url(url, media_type).await,
        scheme => Err(format!(
            "Copilot does not support attachment URL scheme '{scheme}'"
        )),
    }
}

async fn read_file_to_data_url(url: &url::Url, media_type: &str) -> Result<String, String> {
    use base64::Engine;

    let path = url
        .to_file_path()
        .map_err(|()| "attachment file URL could not be converted to a path".to_string())?;
    let data = async_fs::read(&path)
        .await
        .map_err(|error| format!("failed to read attachment '{}': {error}", path.display()))?;
    let base64_data = base64::engine::general_purpose::STANDARD.encode(data);
    Ok(format!("data:{media_type};base64,{base64_data}"))
}

fn convert_tools(tool_defs: &[ToolDefinition]) -> Vec<ToolPayload> {
    tool_defs
        .iter()
        .map(|def| ToolPayload {
            r#type: "function",
            function: ToolFunction {
                name: def.name().to_string(),
                description: def.description().to_string(),
                parameters: def.arguments_openai_schema(),
            },
        })
        .collect()
}

fn tool_choice(params: &Parameters, has_tools: bool) -> Option<ToolChoicePayload> {
    if !has_tools {
        return None;
    }
    match &params.tool_choice {
        ToolChoice::Auto => Some(ToolChoicePayload::Mode("auto")),
        ToolChoice::Required => Some(ToolChoicePayload::Mode("required")),
        ToolChoice::None => Some(ToolChoicePayload::Mode("none")),
        ToolChoice::Exact(name) => Some(ToolChoicePayload::Function {
            kind: "function",
            function: ToolChoiceFunction { name: name.clone() },
        }),
    }
}

fn filter_tool_definitions(defs: Vec<ToolDefinition>, choice: &ToolChoice) -> Vec<ToolDefinition> {
    match choice {
        ToolChoice::None => Vec::new(),
        ToolChoice::Exact(name) => defs
            .into_iter()
            .filter(|tool| tool.name() == name)
            .collect(),
        ToolChoice::Auto | ToolChoice::Required => defs,
    }
}

fn prompt_cache_retention(params: &Parameters) -> Option<&'static str> {
    params
        .cache
        .openai
        .as_ref()
        .and_then(|cache| cache.retention)
        .map(OpenAIPromptCacheRetention::as_str)
}

// === Streaming ===

const SSE_FIRST_EVENT_TIMEOUT: Duration = Duration::from_secs(90);
const SSE_IDLE_TIMEOUT: Duration = Duration::from_secs(30);

async fn open_sse_stream(
    session: &SessionState,
    cfg: &Config,
    request: &ChatCompletionRequest,
) -> Result<zenwave::sse::SseStream, CopilotError> {
    let endpoint = format!(
        "{}/chat/completions",
        session.base_url.trim_end_matches('/')
    );
    let mut backend = client();

    let build_result = backend
        .post(&endpoint)
        .and_then(|b| {
            b.header(
                header::AUTHORIZATION.as_str(),
                format!("Bearer {}", session.token),
            )
        })
        .and_then(|b| b.header(header::USER_AGENT.as_str(), "aither-copilot/0.1"))
        .and_then(|b| b.header(header::ACCEPT.as_str(), "text/event-stream"))
        .and_then(|b| b.header("editor-version", cfg.editor_version.clone()))
        .and_then(|b| b.header("Copilot-Integration-Id", cfg.integration_id.clone()));

    let builder = build_result.map_err(CopilotError::Http)?;

    match builder
        .json_body(request)
        .map_err(CopilotError::Http)?
        .sse()
        .await
    {
        Ok(stream) => Ok(stream),
        Err(zenwave::Error::Timeout) => Err(CopilotError::Timeout),
        Err(e) => Err(CopilotError::Http(e)),
    }
}

const fn is_unauthorized(err: &CopilotError) -> bool {
    matches!(
        err,
        CopilotError::Http(zenwave::Error::Http { status, .. }) if status.as_u16() == 401
    )
}

async fn open_sse_stream_with_refresh(
    cfg: &Config,
    request: &ChatCompletionRequest,
) -> Result<zenwave::sse::SseStream, CopilotError> {
    let snapshot = cfg.session.lock().await.clone();
    match open_sse_stream(&snapshot, cfg, request).await {
        Ok(stream) => Ok(stream),
        Err(err) if is_unauthorized(&err) => {
            match refresh_shared_session(
                &cfg.session,
                &snapshot.token,
                cfg.oauth_token.as_deref(),
                &cfg.token_exchange_url,
            )
            .await?
            {
                Some(refreshed) => open_sse_stream(&refreshed, cfg, request).await,
                None => Err(err),
            }
        }
        Err(err) => Err(err),
    }
}

struct ProcessedSse {
    events: Vec<Event>,
    saw_payload: bool,
    saw_finish_reason: bool,
    done: bool,
}

enum SseStep {
    Event(Result<zenwave::sse::Event, zenwave::sse::ParseError>),
    Closed,
    IdleTimeout,
    FirstEventTimeout,
}

async fn next_sse_event(
    sse_stream: &mut zenwave::sse::SseStream,
    saw_payload: bool,
    last_progress: Instant,
    first_event_timeout: Duration,
    idle_timeout: Duration,
) -> SseStep {
    let timeout = if saw_payload {
        idle_timeout
    } else {
        first_event_timeout
    };
    let elapsed = Instant::now().saturating_duration_since(last_progress);
    if elapsed >= timeout {
        return if saw_payload {
            SseStep::IdleTimeout
        } else {
            SseStep::FirstEventTimeout
        };
    }
    futures_lite::future::race(
        async {
            sse_stream
                .next()
                .await
                .map_or(SseStep::Closed, SseStep::Event)
        },
        async {
            sleep(timeout.saturating_sub(elapsed)).await;
            if saw_payload {
                SseStep::IdleTimeout
            } else {
                SseStep::FirstEventTimeout
            }
        },
    )
    .await
}

fn process_sse_data(
    data: &str,
    include_reasoning: bool,
    usage: &mut Option<Usage>,
    tool_calls: &mut HashMap<usize, ToolCallAccumulator>,
) -> Result<ProcessedSse, CopilotError> {
    if data.is_empty() {
        return Ok(ProcessedSse {
            events: Vec::new(),
            saw_payload: false,
            saw_finish_reason: false,
            done: false,
        });
    }
    if data == "[DONE]" {
        return Ok(ProcessedSse {
            events: Vec::new(),
            saw_payload: false,
            saw_finish_reason: false,
            done: true,
        });
    }
    if let Ok(error_obj) = serde_json::from_str::<Value>(data)
        && let Some(error) = error_obj.get("error")
    {
        let msg = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Unknown API error");
        return Err(CopilotError::Api(msg.to_string()));
    }

    let chunk: ChatCompletionChunk = serde_json::from_str(data).map_err(CopilotError::Json)?;
    let mut events = Vec::new();
    let mut saw_payload = chunk.usage.is_some();
    let mut saw_finish_reason = false;
    if let Some(chunk_usage) = &chunk.usage {
        *usage = Some(usage_from_chat_completion(chunk_usage));
    }
    for choice in &chunk.choices {
        if choice.finish_reason.is_some() {
            saw_finish_reason = true;
            saw_payload = true;
        }
        if let Some(content) = &choice.delta.content
            && !content.is_empty()
        {
            saw_payload = true;
            events.push(Event::Text(content.clone()));
        }
        if let Some(reasoning) = &choice.delta.reasoning_content
            && !reasoning.is_empty()
        {
            saw_payload = true;
            if include_reasoning {
                events.push(Event::Reasoning(reasoning.clone()));
            }
        }
        if let Some(calls) = &choice.delta.tool_calls
            && !calls.is_empty()
        {
            saw_payload = true;
            accumulate_tool_calls(calls, tool_calls);
        }
    }
    Ok(ProcessedSse {
        events,
        saw_payload,
        saw_finish_reason,
        done: false,
    })
}

fn accumulate_tool_calls(
    calls: &[ChunkToolCall],
    tool_calls: &mut HashMap<usize, ToolCallAccumulator>,
) {
    for call in calls {
        let index = call.index.unwrap_or(0);
        let acc = tool_calls.entry(index).or_default();
        if let Some(id) = &call.id
            && !id.is_empty()
        {
            acc.id = Some(id.clone());
        }
        if let Some(function) = &call.function {
            if let Some(name) = &function.name
                && !name.is_empty()
            {
                acc.name = Some(name.clone());
            }
            if let Some(args) = &function.arguments {
                acc.arguments.push_str(args);
            }
        }
    }
}

fn accumulated_tool_call_events(
    tool_calls: HashMap<usize, ToolCallAccumulator>,
    saw_completion: bool,
) -> Result<Vec<Event>, CopilotError> {
    let mut sorted_calls: Vec<_> = tool_calls.into_iter().collect();
    sorted_calls.sort_by_key(|(index, _)| *index);
    let mut events = Vec::with_capacity(sorted_calls.len());
    for (_, acc) in sorted_calls {
        let (Some(id), Some(name)) = (acc.id, acc.name) else {
            continue;
        };
        let arguments = if acc.arguments.is_empty() {
            if !saw_completion {
                return Err(CopilotError::Api(
                    "Copilot stream closed before tool call arguments completed".to_string(),
                ));
            }
            Value::Object(serde_json::Map::default())
        } else {
            serde_json::from_str(&acc.arguments).map_err(CopilotError::Json)?
        };
        events.push(Event::ToolCall(ToolCall {
            id,
            name,
            arguments,
            // Chat Completions carries no reasoning signature to round-trip.
            reasoning_state: None,
        }));
    }
    Ok(events)
}

fn chat_completions_stream(
    cfg: Arc<Config>,
    payload_messages: Vec<ChatMessagePayload>,
    params: &Parameters,
    tools: Option<Vec<ToolPayload>>,
) -> impl Stream<Item = Result<Event, CopilotError>> + Send + Unpin {
    let params = params.clone();
    Box::pin(chat_completions_stream_inner(
        cfg,
        payload_messages,
        params,
        tools,
    ))
}

fn chat_completions_stream_inner(
    cfg: Arc<Config>,
    payload_messages: Vec<ChatMessagePayload>,
    params: Parameters,
    tools: Option<Vec<ToolPayload>>,
) -> impl Stream<Item = Result<Event, CopilotError>> + Send {
    async_stream::stream! {
        let has_tools = tools.as_ref().is_some_and(|t| !t.is_empty());
        let request = ChatCompletionRequest {
            model: cfg.model.clone(),
            messages: payload_messages,
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            temperature: params.temperature,
            top_p: params.top_p,
            max_tokens: params.max_tokens,
            tools,
            tool_choice: tool_choice(&params, has_tools),
            prompt_cache_key: params
                .cache
                .openai
                .as_ref()
                .and_then(|cache| cache.key.clone()),
            prompt_cache_retention: prompt_cache_retention(&params),
        };

        tracing::debug!(
            request = %serde_json::to_string_pretty(&request).unwrap_or_default(),
            "Sending Copilot chat completion request"
        );

        let sse_stream = match open_sse_stream_with_refresh(&cfg, &request).await {
            Ok(stream) => stream,
            Err(err) => {
                yield Err(err);
                return;
            }
        };

        let events = stream_chat_completion_events(sse_stream, params.include_reasoning);
        futures_lite::pin!(events);
        while let Some(event) = events.next().await {
            yield event;
        }
    }
}

fn stream_chat_completion_events(
    sse_stream: zenwave::sse::SseStream,
    include_reasoning: bool,
) -> impl Stream<Item = Result<Event, CopilotError>> + Send {
    stream_chat_completion_events_with_timeouts(
        sse_stream,
        include_reasoning,
        SSE_FIRST_EVENT_TIMEOUT,
        SSE_IDLE_TIMEOUT,
    )
}

fn stream_chat_completion_events_with_timeouts(
    sse_stream: zenwave::sse::SseStream,
    include_reasoning: bool,
    first_event_timeout: Duration,
    idle_timeout: Duration,
) -> impl Stream<Item = Result<Event, CopilotError>> + Send {
    async_stream::stream! {
        let mut event_count = 0usize;
        let mut saw_payload = false;
        let mut saw_completion = false;
        let mut last_progress = Instant::now();
        let mut tool_calls: HashMap<usize, ToolCallAccumulator> = HashMap::new();
        let mut usage: Option<Usage> = None;

        futures_lite::pin!(sse_stream);
        loop {
            let next = next_sse_event(
                &mut sse_stream,
                saw_payload,
                last_progress,
                first_event_timeout,
                idle_timeout,
            )
            .await;
            match next {
                SseStep::Event(Ok(event)) => {
                    // Every SSE frame received over the socket — keepalive comments,
                    // empty data pings, reasoning chunks, or content deltas — proves
                    // the stream is alive and resets the idle timer.
                    last_progress = Instant::now();
                    let data = event.text_data();
                    let data = data.trim();
                    if data.is_empty() {
                        continue;
                    }
                    tracing::trace!(sse_event = %data, "Received Copilot SSE event");
                    event_count += 1;
                    // A `finish_reason` only ends the content: with `include_usage`
                    // the usage arrives in a later chunk, before `[DONE]`.
                    let processed = match process_sse_data(
                        data,
                        include_reasoning,
                        &mut usage,
                        &mut tool_calls,
                    ) {
                        Ok(processed) => processed,
                        Err(err) => {
                            yield Err(err);
                            return;
                        }
                    };
                    if processed.saw_finish_reason {
                        saw_completion = true;
                    }
                    if processed.done {
                        saw_completion = true;
                        break;
                    }
                    if processed.saw_payload {
                        saw_payload = true;
                    }
                    for event in processed.events {
                        yield Ok(event);
                    }
                }
                SseStep::Event(Err(err)) => {
                    yield Err(CopilotError::Stream(err));
                    return;
                }
                SseStep::Closed => break,
                SseStep::IdleTimeout | SseStep::FirstEventTimeout => {
                    tracing::warn!("Copilot SSE stream timed out");
                    yield Err(CopilotError::Timeout);
                    return;
                }
            }
        }

        tracing::debug!(event_count, "Processed Copilot SSE events");
        match accumulated_tool_call_events(tool_calls, saw_completion) {
            Ok(events) => {
                for event in events {
                    yield Ok(event);
                }
            }
            Err(err) => {
                yield Err(err);
                return;
            }
        }
        if let Some(final_usage) = usage {
            yield Ok(Event::Usage(final_usage));
        }
    }
}

#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

fn usage_from_chat_completion(raw: &ChatCompletionUsage) -> Usage {
    Usage {
        prompt_tokens: raw.prompt_tokens,
        completion_tokens: raw.completion_tokens,
        total_tokens: raw.total_tokens,
        reasoning_tokens: raw
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens),
        cache_read_tokens: raw
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens),
        cache_write_tokens: None,
        cost_usd: None,
        stop_reason: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenwave::ResponseExt;

    fn sse_body(chunks: &[&str]) -> zenwave::sse::SseStream {
        let body = chunks
            .iter()
            .map(|data| format!("data: {data}\n\n"))
            .collect::<Vec<_>>()
            .concat();
        zenwave::Response::new(body.into()).into_sse()
    }

    /// With `include_usage` the finish chunk is followed by a usage chunk with
    /// empty `choices`, then `[DONE]`.
    #[tokio::test]
    async fn usage_after_finish_reason_is_reported() {
        let stream = sse_body(&[
            r#"{"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18,"prompt_tokens_details":{"cached_tokens":3},"completion_tokens_details":{"reasoning_tokens":2}}}"#,
            "[DONE]",
        ]);
        let events: Vec<Event> = stream_chat_completion_events(stream, false)
            .map(|event| event.expect("stream event"))
            .collect()
            .await;

        let text: String = events
            .iter()
            .filter_map(|event| match event {
                Event::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "Hello");
        let Some(Event::Usage(usage)) = events.last() else {
            panic!("expected trailing usage event, got {events:?}");
        };
        assert_eq!(usage.prompt_tokens, Some(11));
        assert_eq!(usage.completion_tokens, Some(7));
        assert_eq!(usage.total_tokens, Some(18));
        assert_eq!(usage.reasoning_tokens, Some(2));
        assert_eq!(usage.cache_read_tokens, Some(3));
    }

    #[tokio::test]
    async fn tool_message_with_attachment_emits_followup_user_message() {
        let image = aither_core::llm::Attachment::new(
            "data:image/png;base64,AA==".parse().expect("image URL"),
            "image/png".parse().expect("image MIME"),
        );
        let payloads = to_chat_messages(&[Message::tool("c1", "seen").with_attachment(image)])
            .await
            .expect("copilot messages");
        let value = serde_json::to_value(&payloads).expect("serialize");
        let items = value.as_array().expect("array");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["role"], "tool");
        assert_eq!(items[0]["tool_call_id"], "c1");
        assert_eq!(items[0]["content"], "seen");
        assert_eq!(items[1]["role"], "user");
        assert_eq!(items[1]["content"][0]["type"], "image_url");
    }

    #[tokio::test]
    async fn idle_timeout_after_payload_returns_timeout_error() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 2048];
            let _ = stream.read(&mut buf);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
                )
                .expect("write sse");
            stream.flush().expect("flush");
            let _ = release_rx.recv();
        });

        let mut backend = zenwave::client();
        let sse = backend
            .get(format!("http://{addr}"))
            .expect("get")
            .sse()
            .await
            .expect("sse");
        let events = stream_chat_completion_events_with_timeouts(
            sse,
            false,
            Duration::from_secs(1),
            Duration::from_millis(80),
        );
        futures_lite::pin!(events);

        let first = events.next().await.expect("first item").expect("text ok");
        assert!(matches!(first, Event::Text(ref t) if t == "partial"));

        let second = events
            .next()
            .await
            .expect("timeout item")
            .expect_err("must yield Timeout");
        assert!(matches!(second, CopilotError::Timeout), "{second:?}");
        assert!(events.next().await.is_none());
        drop(release_tx);
    }

    #[tokio::test]
    async fn keepalive_and_reasoning_frames_reset_idle_timer() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let _ = stream.set_nodelay(true);
            let mut buf = [0u8; 2048];
            let _ = stream.read(&mut buf);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n",
                )
                .expect("head");
            stream
                .write_all(
                    b"data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"step1\"},\"finish_reason\":null}]}\n\n",
                )
                .expect("reasoning 1");
            stream.flush().expect("flush");
            // 14 frames spaced 50 ms apart outlast the 500 ms idle timeout while
            // leaving a 450 ms scheduling margin per gap on loaded CI runners.
            for idx in 0..14 {
                std::thread::sleep(Duration::from_millis(50));
                if idx % 2 == 0 {
                    stream.write_all(b"data: \n\n").expect("keepalive");
                } else {
                    stream
                        .write_all(
                            b"data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"step\"},\"finish_reason\":null}]}\n\n",
                        )
                        .expect("reasoning keepalive");
                }
                stream.flush().expect("flush");
            }
            stream
                .write_all(
                    b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
                )
                .expect("final");
            stream.flush().expect("flush");
        });

        let mut backend = zenwave::client();
        let sse = backend
            .get(format!("http://{addr}"))
            .expect("get")
            .sse()
            .await
            .expect("sse");
        let collected: Vec<Event> = stream_chat_completion_events_with_timeouts(
            sse,
            true,
            Duration::from_millis(500),
            Duration::from_millis(500),
        )
        .map(|res| res.expect("keepalives prevent false timeout"))
        .collect()
        .await;

        assert!(
            collected
                .iter()
                .any(|e| matches!(e, Event::Reasoning(r) if r == "step1"))
        );
        assert!(
            collected
                .iter()
                .any(|e| matches!(e, Event::Text(t) if t == "done"))
        );
    }

    #[tokio::test]
    async fn truncated_tool_call_json_returns_error_instead_of_empty_object() {
        let stream = sse_body(&[
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"delete_file","arguments":"{\"path\":\"/important"}}]},"finish_reason":null}]}"#,
        ]);
        let results: Vec<Result<Event, CopilotError>> =
            stream_chat_completion_events(stream, false).collect().await;
        assert_eq!(results.len(), 1);
        let err = results
            .into_iter()
            .next()
            .unwrap()
            .expect_err("truncated JSON must error");
        assert!(matches!(err, CopilotError::Json(_)), "{err:?}");
    }

    #[tokio::test]
    async fn refreshed_session_token_is_shared_across_clones_and_provider() {
        use aither_core::llm::{LanguageModelProvider, oneshot};
        use std::io::{Read, Write};
        use std::sync::atomic::{AtomicUsize, Ordering};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let base = format!("http://{addr}");
        let base_for_server = base.clone();
        let exchange_hits = Arc::new(AtomicUsize::new(0));
        let exchange_hits_server = Arc::clone(&exchange_hits);

        std::thread::spawn(move || {
            while let Ok((mut stream, _)) = listener.accept() {
                let mut req = Vec::new();
                let mut buf = [0u8; 4096];
                let head_end = loop {
                    let Ok(n) = stream.read(&mut buf) else {
                        break 0;
                    };
                    if n == 0 {
                        break 0;
                    }
                    req.extend_from_slice(&buf[..n]);
                    if let Some(pos) = req.windows(4).position(|w| w == b"\r\n\r\n") {
                        break pos + 4;
                    }
                };
                if head_end == 0 {
                    continue;
                }
                let head = String::from_utf8_lossy(&req[..head_end]).into_owned();
                let lower = head.to_ascii_lowercase();
                let chunked = lower.contains("transfer-encoding: chunked");
                let content_len = lower
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                while if chunked {
                    !req.ends_with(b"0\r\n\r\n")
                } else {
                    req.len() < head_end + content_len
                } {
                    let Ok(n) = stream.read(&mut buf) else {
                        break;
                    };
                    if n == 0 {
                        break;
                    }
                    req.extend_from_slice(&buf[..n]);
                }

                if head.starts_with("GET /exchange ") {
                    exchange_hits_server.fetch_add(1, Ordering::SeqCst);
                    let body = serde_json::json!({
                        "token": "fresh-session-token",
                        "endpoints": { "api": base_for_server },
                        "expires_at": 1_999_999_999_i64
                    })
                    .to_string();
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(resp.as_bytes());
                } else if lower.contains("authorization: bearer fresh-session-token") {
                    let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{sse}",
                        sse.len()
                    );
                    let _ = stream.write_all(resp.as_bytes());
                } else {
                    let body = r#"{"error":{"message":"expired"}}"#;
                    let resp = format!(
                        "HTTP/1.1 401 Unauthorized\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(resp.as_bytes());
                }
            }
        });

        let provider = crate::CopilotProvider::new("stale-session-token")
            .base_url(&base)
            .oauth_token("gho_secret_oauth")
            .with_token_exchange_url(format!("{base}/exchange"));
        let client_a = provider.get_model("gpt-5.3-codex").await.expect("model a");
        let client_b = client_a.clone().with_model("gpt-5.4");
        let client_c = provider.get_model("gpt-5.4").await.expect("model c");

        let text_a = aither_core::llm::collect_text(client_a.respond(oneshot("sys", "hi")))
            .await
            .expect("client_a refreshes on 401 and succeeds");
        assert_eq!(text_a, "ok");
        assert_eq!(exchange_hits.load(Ordering::SeqCst), 1);

        let text_b = aither_core::llm::collect_text(client_b.respond(oneshot("sys", "hi")))
            .await
            .expect("client_b reuses refreshed session token without 401");
        assert_eq!(text_b, "ok");

        let text_c = aither_core::llm::collect_text(client_c.respond(oneshot("sys", "hi")))
            .await
            .expect("provider-spawned client_c shares refreshed session token");
        assert_eq!(text_c, "ok");
        assert_eq!(exchange_hits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn debug_impls_redact_copilot_secrets() {
        let builder = Copilot::builder("secret-session-tok").oauth_token("gho_secret_oauth");
        let builder_dbg = format!("{builder:?}");
        assert!(!builder_dbg.contains("secret-session-tok"), "{builder_dbg}");
        assert!(!builder_dbg.contains("gho_secret_oauth"), "{builder_dbg}");

        let client = builder.build();
        let client_dbg = format!("{client:?}");
        assert!(!client_dbg.contains("secret-session-tok"), "{client_dbg}");
        assert!(!client_dbg.contains("gho_secret_oauth"), "{client_dbg}");

        let provider =
            crate::CopilotProvider::new("secret-session-tok").oauth_token("gho_secret_oauth");
        let provider_dbg = format!("{provider:?}");
        assert!(
            !provider_dbg.contains("secret-session-tok"),
            "{provider_dbg}"
        );
        assert!(!provider_dbg.contains("gho_secret_oauth"), "{provider_dbg}");

        let tok = crate::auth::CopilotToken {
            access_token: "gho_live_secret".into(),
            token_type: "bearer".into(),
            scope: "copilot".into(),
            expires_at: None,
        };
        assert!(!format!("{tok:?}").contains("gho_live_secret"));

        let session = crate::auth::SessionToken {
            token: "tid=secret_tid".into(),
            api_endpoint: "https://api.githubcopilot.com".into(),
            expires_at: 123,
        };
        assert!(!format!("{session:?}").contains("tid=secret_tid"));
    }
}
