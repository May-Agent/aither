//! [`ClientSession`]: one live session on a [`Client`].

use std::collections::BTreeMap;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_lite::Stream;
use serde::Serialize;
use serde_json::{Value, json};

use crate::domain::{
    ConfigValue, ConfigValueById, ConfigValueId, ContentBlock, ExtMethod, Open, SessionId,
    SessionUpdate, SetConfigParams,
    config::{ConfigCategory, ConfigOptionUpdate},
};
use crate::wire::codec::DEFAULT_MODE_OPTION_ID;

use super::connection::{codex_models, steering_advertised};
use super::error::ClientError;
use super::handle::{Client, SessionCell};
use super::turn::PromptTurn;
use crate::state::{SessionEvent, SessionState};

/// Anything that becomes a prompt's content blocks.
pub trait IntoPrompt {
    /// Build the prompt payload.
    fn into_prompt(self) -> Vec<ContentBlock>;
}

impl IntoPrompt for &str {
    fn into_prompt(self) -> Vec<ContentBlock> {
        vec![ContentBlock::text(self)]
    }
}

impl IntoPrompt for String {
    fn into_prompt(self) -> Vec<ContentBlock> {
        vec![ContentBlock::text(self)]
    }
}

impl IntoPrompt for &String {
    fn into_prompt(self) -> Vec<ContentBlock> {
        vec![ContentBlock::text(self.clone())]
    }
}

impl IntoPrompt for &[ContentBlock] {
    fn into_prompt(self) -> Vec<ContentBlock> {
        self.to_vec()
    }
}

impl<const N: usize> IntoPrompt for &[ContentBlock; N] {
    fn into_prompt(self) -> Vec<ContentBlock> {
        self.to_vec()
    }
}

impl IntoPrompt for Vec<ContentBlock> {
    fn into_prompt(self) -> Vec<ContentBlock> {
        self
    }
}

impl IntoPrompt for ContentBlock {
    fn into_prompt(self) -> Vec<ContentBlock> {
        vec![self]
    }
}

/// A session event subscription: the snapshot taken at subscribe time,
/// then every subsequent event.
pub struct SessionEvents {
    /// The [`ClientSession::state`] snapshot captured when `subscribe()`
    /// ran — what the events in this stream fold onto.
    pub snapshot: SessionState,
    rx: Pin<Box<async_channel::Receiver<SessionEvent>>>,
}

impl fmt::Debug for SessionEvents {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionEvents").finish_non_exhaustive()
    }
}

impl Stream for SessionEvents {
    type Item = SessionEvent;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<SessionEvent>> {
        self.get_mut().rx.as_mut().poll_next(cx)
    }
}

/// One live session. Cloneable — clones share state, events, and the lane.
#[derive(Clone)]
pub struct ClientSession {
    client: Client,
    cell: Arc<SessionCell>,
    /// Vendor extras from the setup result (codex `models` etc.).
    setup_extra: BTreeMap<String, Value>,
}

impl fmt::Debug for ClientSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientSession")
            .field("id", &self.id())
            .finish_non_exhaustive()
    }
}

impl ClientSession {
    pub(crate) const fn new(
        client: Client,
        cell: Arc<SessionCell>,
        setup_extra: BTreeMap<String, Value>,
    ) -> Self {
        Self {
            client,
            cell,
            setup_extra,
        }
    }

    /// The session id.
    #[must_use]
    pub fn id(&self) -> &SessionId {
        &self.cell.id
    }

    /// The owning client.
    #[must_use]
    pub const fn client(&self) -> &Client {
        &self.client
    }

    /// The current folded snapshot.
    #[must_use]
    pub fn state(&self) -> SessionState {
        self.cell.state()
    }

    /// Subscribe to session events: current snapshot plus every later
    /// event (including background work — this stream is not per-turn).
    #[must_use]
    pub fn subscribe(&self) -> SessionEvents {
        SessionEvents {
            snapshot: self.state(),
            rx: Box::pin(self.cell.subscribe()),
        }
    }

    /// Send a prompt; returns the [`PromptTurn`] stream.
    ///
    /// The request future lives inside the turn — `prompt()` returns
    /// immediately so the caller can drive events while the agent works
    /// (v2: until the idle `state_update`; v1: until the response).
    ///
    /// # Errors
    ///
    /// Only fails synchronously on a closed lane; agent rejections
    /// surface as [`TurnOutcome::Rejected`](super::TurnOutcome).
    pub fn prompt(&self, input: impl IntoPrompt) -> Result<PromptTurn, ClientError> {
        let params = crate::domain::PromptParams {
            session_id: self.id().clone(),
            prompt: input.into_prompt(),
            idempotency_key: Some(uuid::Uuid::new_v4().to_string()),
            meta: None,
            extra: BTreeMap::new(),
        };
        let rx = Box::pin(self.cell.subscribe());
        let lane = self.client.lane().clone();
        let pending = Box::pin(async move { lane.prompt(params).await });
        Ok(PromptTurn::new(self.clone(), rx, pending))
    }

    /// `prompt` + `finish()`: send and await the outcome.
    ///
    /// # Errors
    ///
    /// Same as [`prompt`](Self::prompt).
    pub async fn prompt_run(
        &self,
        input: impl IntoPrompt,
    ) -> Result<super::turn::TurnOutcome, ClientError> {
        let mut turn = self.prompt(input)?;
        Ok(turn.finish().await)
    }

    /// `session/cancel` — ask the agent to stop the current turn.
    ///
    /// # Errors
    ///
    /// Propagates transport errors.
    pub async fn cancel(&self) -> Result<(), ClientError> {
        self.client.lane().cancel_session(self.id().clone()).await
    }

    /// `session/close` — end the session without deleting its history.
    ///
    /// # Errors
    ///
    /// Propagates transport and rejection errors.
    pub async fn close(&self) -> Result<(), ClientError> {
        self.client.lane().close_session(self.id().clone()).await
    }

    /// `session/set_config_option` (v1 codec routes `session/set_mode`
    /// for the mode category).
    ///
    /// Applies the returned option list to local state when the agent
    /// reports it (v1 does; v2 reports via `config_option_update`).
    ///
    /// # Errors
    ///
    /// Propagates transport, rejection, and downgrade errors.
    pub async fn set_config(
        &self,
        id: impl Into<crate::domain::ids::ConfigId>,
        value: ConfigValue,
    ) -> Result<(), ClientError> {
        let options = self
            .client
            .lane()
            .set_config(SetConfigParams {
                session_id: self.id().clone(),
                config_id: id.into(),
                value,
                meta: None,
                extra: BTreeMap::new(),
            })
            .await?;
        if !options.is_empty() {
            self.cell
                .apply(&SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate {
                    config_options: options,
                    meta: None,
                    extra: BTreeMap::new(),
                }));
        }
        Ok(())
    }

    /// Activate session mode `mode_id` (a mode-category option value).
    ///
    /// v2: `session/set_config_option` on the mode-category option. v1:
    /// the codec maps the mode option onto `session/set_mode`.
    ///
    /// # Errors
    ///
    /// `Unsupported` when no mode option is advertised and the version
    /// cannot synthesize one.
    pub async fn set_mode(&self, mode_id: impl Into<ConfigValueId>) -> Result<(), ClientError> {
        let mode_id = mode_id.into();
        // The mode-category option's config id; on v1 the codec also
        // routes the synthesized DEFAULT_MODE_OPTION_ID to set_mode, so
        // it is a sound fallback when no option list was reported.
        let option_id = self
            .state()
            .config_options
            .iter()
            .find(|o| o.is_mode())
            .map_or_else(
                || crate::domain::ids::ConfigId::from_string(DEFAULT_MODE_OPTION_ID),
                |o| o.config_id.clone(),
            );
        self.set_config(
            option_id,
            ConfigValue::Id(ConfigValueById { value: mode_id }),
        )
        .await
    }

    /// Switch model to `model_id`.
    ///
    /// - v2 / v1 with a model-category config option → `set_config`.
    /// - v1 codex legacy: no model option but a `models` extra on the
    ///   setup result → `session/set_model` (wire-compatible with
    ///   `vendor::codex` until that module lands).
    ///
    /// # Errors
    ///
    /// `Unsupported` when neither path is available.
    pub async fn set_model(&self, model_id: impl Into<ConfigValueId>) -> Result<(), ClientError> {
        let model_id = model_id.into();
        if let Some(option) = self
            .state()
            .config_options
            .iter()
            .find(|o| matches!(&o.category, Some(Open::Known(ConfigCategory::Model))))
        {
            return self
                .set_config(
                    option.config_id.clone(),
                    ConfigValue::Id(ConfigValueById { value: model_id }),
                )
                .await;
        }
        if !self.client.is_v2() && codex_models(&self.setup_extra) {
            self.client
                .lane()
                .legacy_request(
                    "session/set_model",
                    json!({"sessionId": self.id().as_str(), "modelId": model_id.as_str()}),
                )
                .await?;
            return Ok(());
        }
        Err(ClientError::Unsupported(
            "agent advertises no model option".to_string(),
        ))
    }

    /// Inject content into the running turn.
    ///
    /// - v2: a concurrent `session/prompt` — returns the turn handle.
    /// - v1: `_session/steering` when the agent advertised it
    ///   (`_meta.steering.supported`), else a plain queued prompt.
    ///
    /// # Errors
    ///
    /// Propagates transport and rejection errors.
    ///
    /// # Panics
    ///
    /// Panics if the session's v1 codec mutex is poisoned.
    pub async fn steer(&self, blocks: impl IntoPrompt) -> Result<PromptTurn, ClientError> {
        let blocks = blocks.into_prompt();
        if self.client.is_v2() || !steering_advertised(self.client.init()) {
            return self.prompt(blocks);
        }
        // v1 + steering advertised: encode the prompt through the codec.
        let params = crate::domain::PromptParams {
            session_id: self.id().clone(),
            prompt: blocks,
            idempotency_key: None,
            meta: None,
            extra: BTreeMap::new(),
        };
        let v1_prompt = {
            let codec = self.cell.codec.lock().expect("codec poisoned");
            codec.prompt_to_v1(&params)
        };
        self.client
            .lane()
            .legacy_request(
                super::connection::STEERING_METHOD,
                json!({"sessionId": self.id().as_str(), "prompt": v1_prompt.prompt}),
            )
            .await?;
        // The steering response is an outcome, not a turn — return a
        // finished `PromptTurn` shaped like a detached subscription.
        let rx = Box::pin(self.cell.subscribe());
        Ok(PromptTurn::detached(self.clone(), rx))
    }

    /// A `_`-prefixed extension request, session-scoped by convention
    /// (params pass through verbatim).
    ///
    /// # Errors
    ///
    /// Propagates transport and rejection errors.
    pub fn ext_request<'a>(
        &'a self,
        method: &'a ExtMethod,
        params: &impl Serialize,
    ) -> impl Future<Output = Result<Value, ClientError>> + Send + 'a {
        self.client.ext_request(method, params)
    }

    /// `session/tool_call_cancel` (v2 only; the call must advertise
    /// `capabilities.cancel`).
    ///
    /// # Errors
    ///
    /// `Unsupported` on v1; transport errors otherwise.
    pub async fn cancel_tool_call(
        &self,
        tool_call: impl Into<crate::domain::ToolCallId>,
    ) -> Result<(), ClientError> {
        self.client
            .lane()
            .cancel_tool_call(self.id().clone(), tool_call.into())
            .await
    }

    /// `session/terminal_cancel` (v2 only).
    ///
    /// # Errors
    ///
    /// `Unsupported` on v1; transport errors otherwise.
    pub async fn cancel_terminal(
        &self,
        terminal: impl Into<crate::domain::TerminalId>,
    ) -> Result<(), ClientError> {
        self.client
            .lane()
            .cancel_terminal(self.id().clone(), terminal.into())
            .await
    }

    /// Inject a raw update into this session's reducer/broadcast —
    /// used by the v1 turn machinery to synthesize the turn end.
    pub(crate) fn push_update(&self, update: &SessionUpdate) {
        self.cell.apply(update);
    }
}
