//! Model groups with budget tracking and automatic fallback.
//!
//! A [`ModelGroup`] is a language model made of an ordered list of budgeted
//! models: when one exhausts its budget, the next serves the request.

use std::sync::atomic::{AtomicU64, Ordering};

use aither_core::llm::Usage;
use num_traits::ToPrimitive;

/// Budget limit for a model.
#[derive(Debug, Clone)]
pub enum Budget {
    /// Unlimited usage (no budget tracking).
    Unlimited,
    /// Limit by total tokens.
    Tokens(u64),
    /// Limit by estimated cost in USD (stored as microdollars for precision).
    Cost(f64),
}

impl Budget {
    /// Creates a token-based budget.
    #[must_use]
    pub const fn tokens(limit: u64) -> Self {
        Self::Tokens(limit)
    }

    /// Creates a cost-based budget in USD.
    #[must_use]
    pub const fn usd(limit: f64) -> Self {
        Self::Cost(limit)
    }
}

/// A model with budget tracking.
#[derive(Debug)]
pub struct BudgetedModel<M> {
    /// The underlying model.
    pub model: M,
    /// Budget limit.
    pub budget: Budget,
    /// Tokens used (atomic for thread safety).
    tokens_used: AtomicU64,
    /// Cost used in microdollars (1 USD = `1_000_000` microdollars).
    cost_used_micro: AtomicU64,
    /// Whether this model's budget is exhausted.
    exhausted: std::sync::atomic::AtomicBool,
}

impl<M> BudgetedModel<M> {
    /// Creates a new budgeted model.
    #[must_use]
    pub const fn new(model: M, budget: Budget) -> Self {
        Self {
            model,
            budget,
            tokens_used: AtomicU64::new(0),
            cost_used_micro: AtomicU64::new(0),
            exhausted: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Creates a model with unlimited budget.
    #[must_use]
    pub const fn unlimited(model: M) -> Self {
        Self::new(model, Budget::Unlimited)
    }

    /// Records usage from a response.
    pub fn record_usage(&self, usage: &Usage) {
        if let Some(tokens) = usage.total_tokens {
            self.tokens_used
                .fetch_add(u64::from(tokens), Ordering::Relaxed);
        }
        if let Some(cost) = usage.cost_usd {
            // Convert to microdollars
            let micro = (cost * 1_000_000.0).round().to_u64().unwrap_or(u64::MAX);
            self.cost_used_micro.fetch_add(micro, Ordering::Relaxed);
        }
    }

    /// Marks this model as exhausted (e.g., from API quota error).
    pub fn mark_exhausted(&self) {
        self.exhausted.store(true, Ordering::Release);
    }

    /// Checks if the budget is exhausted.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        if self.exhausted.load(Ordering::Acquire) {
            return true;
        }

        match &self.budget {
            Budget::Unlimited => false,
            Budget::Tokens(limit) => self.tokens_used.load(Ordering::Relaxed) >= *limit,
            Budget::Cost(limit) => {
                let micro_limit = (*limit * 1_000_000.0).round().to_u64().unwrap_or(u64::MAX);
                self.cost_used_micro.load(Ordering::Relaxed) >= micro_limit
            }
        }
    }

    /// Returns the current token usage.
    #[must_use]
    pub fn tokens_used(&self) -> u64 {
        self.tokens_used.load(Ordering::Relaxed)
    }

    /// Returns the current cost usage in USD.
    #[must_use]
    pub fn cost_used(&self) -> f64 {
        self.cost_used_micro
            .load(Ordering::Relaxed)
            .to_f64()
            .unwrap_or(f64::MAX)
            / 1_000_000.0
    }

    /// Resets the usage counters (but not the exhausted flag).
    pub fn reset_usage(&self) {
        self.tokens_used.store(0, Ordering::Relaxed);
        self.cost_used_micro.store(0, Ordering::Relaxed);
    }

    /// Resets everything including the exhausted flag.
    pub fn reset_all(&self) {
        self.reset_usage();
        self.exhausted.store(false, Ordering::Release);
    }

    /// Returns a reference to the underlying model.
    #[must_use]
    pub const fn inner(&self) -> &M {
        &self.model
    }
}

/// A group of models with the same tier, ordered by preference.
#[derive(Debug)]
pub struct ModelGroup<M> {
    /// Models in preference order (first = preferred).
    models: Vec<BudgetedModel<M>>,
    /// Current active model index.
    current: std::sync::atomic::AtomicUsize,
}

impl<M> ModelGroup<M> {
    /// Creates a new model group with a single model.
    #[must_use]
    pub fn new(model: BudgetedModel<M>) -> Self {
        Self {
            models: vec![model],
            current: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Creates a group from multiple budgeted models.
    #[must_use]
    pub const fn from_models(models: Vec<BudgetedModel<M>>) -> Self {
        Self {
            models,
            current: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Adds a fallback model to the group.
    #[must_use]
    pub fn with_fallback(mut self, model: BudgetedModel<M>) -> Self {
        self.models.push(model);
        self
    }

    /// Returns the current active model, or None if all are exhausted.
    #[must_use]
    pub fn current(&self) -> Option<&BudgetedModel<M>> {
        self.current_with_index().map(|(_, model)| model)
    }

    fn current_with_index(&self) -> Option<(usize, &BudgetedModel<M>)> {
        if !self.try_advance() {
            return None;
        }
        let idx = self.current.load(Ordering::Acquire);
        self.models
            .get(idx)
            .filter(|m| !m.is_exhausted())
            .map(|m| (idx, m))
    }

    /// Advances to the next model if the current one is exhausted.
    /// Returns true if a new model is available, false if all are exhausted.
    pub fn try_advance(&self) -> bool {
        loop {
            let idx = self.current.load(Ordering::Acquire);
            if idx >= self.models.len() {
                return false;
            }

            if let Some(model) = self.models.get(idx)
                && !model.is_exhausted()
            {
                return true;
            }

            // Try to advance
            let next = idx + 1;
            if next >= self.models.len() {
                return false;
            }

            // CAS to advance
            if self
                .current
                .compare_exchange(idx, next, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                // Another thread advanced, retry
                continue;
            }

            // Check if the new model is available
            if let Some(model) = self.models.get(next)
                && !model.is_exhausted()
            {
                return true;
            }
        }
    }

    /// Records usage for the current model.
    pub fn record_usage(&self, usage: &Usage) {
        let idx = self.current.load(Ordering::Acquire);
        if let Some(model) = self.models.get(idx) {
            model.record_usage(usage);
            if model.is_exhausted() {
                let _ = self.try_advance();
            }
        }
    }

    /// Marks the current model as exhausted and tries to advance.
    pub fn mark_current_exhausted(&self) -> bool {
        let idx = self.current.load(Ordering::Acquire);
        if let Some(model) = self.models.get(idx) {
            model.mark_exhausted();
        }
        self.try_advance()
    }

    /// Returns true if all models in the group are exhausted.
    #[must_use]
    pub fn all_exhausted(&self) -> bool {
        self.models.iter().all(BudgetedModel::is_exhausted)
    }

    /// Resets all models in the group.
    pub fn reset_all(&self) {
        for model in &self.models {
            model.reset_all();
        }
        self.current.store(0, Ordering::Release);
    }
}

// ============================================================================
// LanguageModel implementation for ModelGroup
// ============================================================================

use aither_core::{
    LanguageModel,
    llm::{
        Classify as _, Event, FailureKind, GenerateError, LLMRequest, RetryPolicy, TokenCounting,
        model::Profile,
    },
};
use futures_core::{FusedStream, Stream};

/// Error type for model group operations.
#[derive(Debug, thiserror::Error)]
pub enum ModelGroupError<E> {
    /// All models in the group are exhausted.
    #[error("all models in group are exhausted")]
    AllModelsExhausted,
    /// Underlying model error.
    #[error("{0}")]
    Model(#[source] E),
}

impl<E: aither_core::llm::Classify> aither_core::llm::Classify for ModelGroupError<E> {
    fn failure(&self) -> aither_core::llm::Failure {
        match self {
            Self::AllModelsExhausted => aither_core::llm::Failure::quota(),
            Self::Model(e) => e.failure(),
        }
    }
}

impl<M> LanguageModel for ModelGroup<M>
where
    M: LanguageModel + Send + Sync,
    M::Error: Send + Sync + 'static,
{
    type Error = ModelGroupError<M::Error>;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        ModelGroupStream::new(self, request)
    }

    /// The policy of the model the next request goes to; with every model
    /// exhausted there is nothing to retry against.
    fn retry_policy(&self) -> RetryPolicy {
        self.current()
            .map_or_else(RetryPolicy::none, |model| model.inner().retry_policy())
    }

    fn token_counting(&self) -> TokenCounting {
        self.current()
            .map_or(TokenCounting::None, |model| model.inner().token_counting())
    }

    async fn count_tokens(
        &self,
        request: &LLMRequest,
    ) -> Result<Option<u32>, GenerateError<Self::Error>> {
        let Some(model) = self.current() else {
            return Ok(None);
        };
        model
            .inner()
            .count_tokens(request)
            .await
            .map_err(|error| match error {
                GenerateError::Provider(source) => {
                    GenerateError::Provider(ModelGroupError::Model(source))
                }
                GenerateError::Parse { source, response } => {
                    GenerateError::Parse { source, response }
                }
            })
    }

    fn count_text(&self, text: &str) -> Option<u32> {
        self.current()
            .and_then(|model| model.inner().count_text(text))
    }

    async fn profile(&self) -> Result<Profile, Self::Error> {
        if let Some(model) = self.current() {
            model
                .inner()
                .profile()
                .await
                .map_err(ModelGroupError::Model)
        } else if let Some(model) = self.models.first() {
            model
                .inner()
                .profile()
                .await
                .map_err(ModelGroupError::Model)
        } else {
            Err(ModelGroupError::AllModelsExhausted)
        }
    }
}

use std::pin::Pin;
use std::task::{Context, Poll};

/// A boxed event stream from one of the group's models.
/// Stream wrapper for `ModelGroup` that tracks usage and detects quota errors.
type BoxedModelStream<'a, M> =
    Pin<Box<dyn Stream<Item = Result<Event, <M as LanguageModel>::Error>> + Send + 'a>>;

enum StreamState<'a, M: LanguageModel> {
    Active {
        model_index: usize,
        stream: BoxedModelStream<'a, M>,
    },
    Exhausted,
    Done,
}

struct ModelGroupStream<'a, M: LanguageModel> {
    group: &'a ModelGroup<M>,
    state: StreamState<'a, M>,
}

impl<'a, M: LanguageModel> ModelGroupStream<'a, M> {
    fn new(group: &'a ModelGroup<M>, request: LLMRequest) -> Self {
        let state = match group.current_with_index() {
            Some((model_index, model)) => {
                let stream = Box::pin(model.inner().respond(request)) as BoxedModelStream<'a, M>;
                StreamState::Active {
                    model_index,
                    stream,
                }
            }
            None => StreamState::Exhausted,
        };
        Self { group, state }
    }
}

impl<M: LanguageModel> Stream for ModelGroupStream<'_, M>
where
    M::Error: Send + 'static,
{
    type Item = Result<Event, ModelGroupError<M::Error>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match &mut self.state {
            StreamState::Done => Poll::Ready(None),
            StreamState::Exhausted => {
                self.state = StreamState::Done;
                Poll::Ready(Some(Err(ModelGroupError::AllModelsExhausted)))
            }
            StreamState::Active {
                model_index,
                stream,
            } => {
                let idx = *model_index;
                match stream.as_mut().poll_next(cx) {
                    Poll::Ready(Some(Ok(event))) => {
                        if let Event::Usage(ref usage) = event
                            && let Some(model) = self.group.models.get(idx)
                        {
                            model.record_usage(usage);
                            if model.is_exhausted() {
                                let _ = self.group.try_advance();
                            }
                        }
                        Poll::Ready(Some(Ok(event)))
                    }
                    Poll::Ready(Some(Err(e))) => {
                        if e.failure().kind == FailureKind::Quota {
                            if let Some(model) = self.group.models.get(idx) {
                                model.mark_exhausted();
                            }
                            let _ = self.group.try_advance();
                            tracing::warn!(error = %e, "Model quota exhausted");
                        }
                        self.state = StreamState::Done;
                        Poll::Ready(Some(Err(ModelGroupError::Model(e))))
                    }
                    Poll::Ready(None) => {
                        self.state = StreamState::Done;
                        Poll::Ready(None)
                    }
                    Poll::Pending => Poll::Pending,
                }
            }
        }
    }
}

impl<M: LanguageModel> FusedStream for ModelGroupStream<'_, M>
where
    M::Error: Send + 'static,
{
    fn is_terminated(&self) -> bool {
        matches!(self.state, StreamState::Done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::{
        LanguageModel,
        llm::{Event, LLMRequest, model::Profile},
    };
    use futures_lite::{StreamExt, stream};

    #[derive(Debug, Clone)]
    struct DummyModel {
        name: &'static str,
    }

    #[derive(Debug)]
    struct DummyError;

    impl std::fmt::Display for DummyError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("dummy error")
        }
    }

    impl std::error::Error for DummyError {}

    impl aither_core::llm::Classify for DummyError {
        fn failure(&self) -> aither_core::llm::Failure {
            aither_core::llm::Failure::permanent()
        }
    }

    impl LanguageModel for DummyModel {
        type Error = DummyError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl futures_core::Stream<Item = Result<Event, Self::Error>> + Send {
            stream::iter([Ok(Event::Text(self.name.to_string()))])
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<Profile, Self::Error>> + Send {
            std::future::ready(Ok(Profile::new("dummy", self.name, "test", "dummy", 0)))
        }
    }

    #[test]
    fn test_budget_tokens() {
        let model = BudgetedModel::new("gpt-4", Budget::tokens(1000));
        assert!(!model.is_exhausted());

        model.record_usage(&Usage::new(500, 400));
        assert!(!model.is_exhausted());

        model.record_usage(&Usage::new(50, 51));
        assert!(model.is_exhausted());
    }

    #[test]
    fn test_budget_cost() {
        let model = BudgetedModel::new("gpt-4", Budget::usd(1.0));
        assert!(!model.is_exhausted());

        model.record_usage(&Usage::new(0, 0).with_cost(0.5));
        assert!(!model.is_exhausted());

        model.record_usage(&Usage::new(0, 0).with_cost(0.6));
        assert!(model.is_exhausted());
    }

    #[test]
    fn test_model_group_fallback() {
        let group = ModelGroup::from_models(vec![
            BudgetedModel::new("primary", Budget::tokens(100)),
            BudgetedModel::new("fallback", Budget::tokens(100)),
        ]);

        // Initially on primary
        assert_eq!(
            group.current().map(super::BudgetedModel::inner),
            Some(&"primary")
        );

        // Exhaust primary
        group.record_usage(&Usage::new(50, 51));
        assert_eq!(
            group.current().map(super::BudgetedModel::inner),
            Some(&"fallback")
        );

        // Exhaust fallback
        group.record_usage(&Usage::new(50, 51));
        assert!(!group.try_advance());
        assert!(group.all_exhausted());
    }

    #[test]
    fn test_mark_exhausted() {
        let group = ModelGroup::from_models(vec![
            BudgetedModel::unlimited("primary"),
            BudgetedModel::unlimited("fallback"),
        ]);

        assert_eq!(
            group.current().map(super::BudgetedModel::inner),
            Some(&"primary")
        );

        // Mark as exhausted (e.g., from API quota error)
        assert!(group.mark_current_exhausted());
        assert_eq!(
            group.current().map(super::BudgetedModel::inner),
            Some(&"fallback")
        );
    }

    #[test]
    fn current_advances_past_exhausted_models() {
        let group = ModelGroup::from_models(vec![
            BudgetedModel::new("primary", Budget::tokens(1)),
            BudgetedModel::new("fallback", Budget::tokens(100)),
        ]);
        group.models[0].mark_exhausted();
        assert_eq!(
            group.current().map(super::BudgetedModel::inner),
            Some(&"fallback")
        );
    }

    #[tokio::test]
    async fn stream_uses_fallback_after_primary_is_exhausted() {
        let group = ModelGroup::from_models(vec![
            BudgetedModel::new(DummyModel { name: "primary" }, Budget::tokens(1)),
            BudgetedModel::new(DummyModel { name: "fallback" }, Budget::tokens(100)),
        ]);
        group.models[0].mark_exhausted();

        let request = LLMRequest::new([aither_core::llm::Message::user("hello")]);
        let mut stream = group.respond(request);
        let first = stream.next().await;

        match first {
            Some(Ok(Event::Text(text))) => assert_eq!(text, "fallback"),
            other => panic!("expected fallback model text event, got {other:?}"),
        }
    }

    #[derive(Debug, Clone)]
    struct FailingError {
        msg: &'static str,
        failure: aither_core::llm::Failure,
    }

    impl std::fmt::Display for FailingError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.msg)
        }
    }

    impl std::error::Error for FailingError {}

    impl aither_core::llm::Classify for FailingError {
        fn failure(&self) -> aither_core::llm::Failure {
            self.failure
        }
    }

    #[derive(Clone)]
    struct ScriptedGroupModel {
        events: Vec<Result<Event, FailingError>>,
    }

    impl LanguageModel for ScriptedGroupModel {
        type Error = FailingError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl futures_core::Stream<Item = Result<Event, Self::Error>> + Send {
            stream::iter(self.events.clone())
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<Output = Result<Profile, Self::Error>> + Send {
            std::future::ready(Ok(Profile::new("scripted", "m", "test", "m", 0)))
        }
    }

    #[tokio::test]
    async fn exhausted_stream_yields_one_error_and_terminates() {
        let group = ModelGroup::new(BudgetedModel::new(
            DummyModel { name: "only" },
            Budget::tokens(1),
        ));
        group.models[0].mark_exhausted();

        let request = LLMRequest::new([aither_core::llm::Message::user("hello")]);
        let items: Vec<_> = group.respond(request).collect().await;
        assert_eq!(items.len(), 1);
        assert!(matches!(items[0], Err(ModelGroupError::AllModelsExhausted)));
    }

    #[tokio::test]
    async fn quota_classification_and_bound_model_index_govern_exhaustion() {
        // Non-quota error containing "exceeded" must not exhaust the model, and
        // collecting the stream after an error must terminate cleanly.
        let permanent_exceeded = ScriptedGroupModel {
            events: vec![Err(FailingError {
                msg: "context_length_exceeded",
                failure: aither_core::llm::Failure::permanent(),
            })],
        };
        let group = ModelGroup::from_models(vec![
            BudgetedModel::unlimited(permanent_exceeded),
            BudgetedModel::unlimited(ScriptedGroupModel {
                events: vec![Ok(Event::Text("fallback".into()))],
            }),
        ]);
        let items: Vec<_> = group
            .respond(LLMRequest::new([aither_core::llm::Message::user("hi")]))
            .collect()
            .await;
        assert_eq!(items.len(), 1);
        assert!(matches!(items[0], Err(ModelGroupError::Model(_))));
        assert!(
            !group.models[0].is_exhausted(),
            "substring 'exceeded' without FailureKind::Quota must not exhaust primary"
        );

        // A stream opened on primary records usage and quota exhaustion on
        // primary even if `group` advances while the stream is open.
        let quota_stream = ScriptedGroupModel {
            events: vec![
                Ok(Event::Usage(Usage::new(10, 5))),
                Err(FailingError {
                    msg: "limit reached",
                    failure: aither_core::llm::Failure::quota(),
                }),
            ],
        };
        let group = ModelGroup::from_models(vec![
            BudgetedModel::unlimited(quota_stream),
            BudgetedModel::unlimited(ScriptedGroupModel {
                events: vec![Ok(Event::Text("fallback".into()))],
            }),
        ]);
        let mut open_stream =
            group.respond(LLMRequest::new([aither_core::llm::Message::user("hi")]));
        // Simulate concurrent advance to fallback before `open_stream` is polled.
        group.current.store(1, Ordering::Release);

        let collected: Vec<_> = (&mut open_stream).collect().await;
        assert_eq!(collected.len(), 2);
        assert_eq!(group.models[0].tokens_used(), 15);
        assert_eq!(group.models[1].tokens_used(), 0);
        assert!(group.models[0].is_exhausted());
        assert!(!group.models[1].is_exhausted());
    }
}
