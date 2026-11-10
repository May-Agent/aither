//! Keeping the prompt-cache entry warm: an idle agent's refresh, on an
//! injected clock.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_core::llm::{Event, LLMRequest, LanguageModel, Usage, model::Profile};
use futures_core::Stream;

use crate::context::CacheLifetime;
use crate::{Agent, AgentBuilder, ManualClock};

#[derive(Debug)]
struct NeverError;

impl std::fmt::Display for NeverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("never")
    }
}

impl std::error::Error for NeverError {}

impl aither_core::llm::Classify for NeverError {
    fn failure(&self) -> aither_core::llm::Failure {
        aither_core::llm::Failure::permanent()
    }
}

/// A model that records its requests and answers each with the same text and
/// usage.
#[derive(Clone, Default)]
struct UsageLlm {
    requests: Arc<Mutex<Vec<LLMRequest>>>,
}

impl UsageLlm {
    /// The keep-alive requests seen so far: those asking for no output.
    fn keep_alives(&self) -> usize {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|request| request.parameters().max_tokens == Some(0))
            .count()
    }
}

impl LanguageModel for UsageLlm {
    type Error = NeverError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        self.requests.lock().unwrap().push(request);
        futures_lite::stream::iter([
            Ok(Event::Text("ok".to_string())),
            Ok(Event::Usage(Usage::new(1200, 0))),
        ])
    }

    fn profile(&self) -> impl std::future::Future<Output = Result<Profile, Self::Error>> + Send {
        std::future::ready(Ok(Profile::new(
            "usage",
            "test",
            "usage",
            "usage model",
            128_000,
        )))
    }
}

/// An agent that answered one request, with `lifetime` as its cache lifetime.
async fn agent_after_one_request(
    llm: &UsageLlm,
    clock: &ManualClock,
    lifetime: Option<CacheLifetime>,
) -> Agent<UsageLlm> {
    let mut builder = AgentBuilder::new(llm.clone()).clock(clock.clone());
    if let Some(lifetime) = lifetime {
        builder = builder.cache_lifetime(lifetime);
    }
    let mut agent = builder.build();
    agent.query("hello").await.expect("the first request");
    agent
}

#[tokio::test]
async fn an_idle_agent_refreshes_its_entry_without_touching_its_history() {
    let llm = UsageLlm::default();
    let clock = ManualClock::new();
    let mut agent = agent_after_one_request(&llm, &clock, None).await;
    clock.advance(Duration::from_secs(3590));
    let before = agent.context().len_recent();
    let requests = agent.cache_stats().requests();
    let cold_in = agent.cache_warmth().cold_in().expect("a limited entry");
    assert_eq!(cold_in, Duration::from_secs(10));

    let usage = agent
        .keep_cache_warm()
        .await
        .expect("the refresh succeeds")
        .expect("there was an entry to refresh");

    assert_eq!(usage, Usage::new(1200, 0));
    assert_eq!(llm.keep_alives(), 1);
    assert_eq!(agent.context().len_recent(), before, "history is untouched");
    assert_eq!(
        agent.cache_stats().requests(),
        requests + 1,
        "the usage is recorded"
    );
    assert_eq!(
        agent.cache_warmth().cold_in(),
        Some(Duration::from_secs(3600)),
        "the entry's lifetime restarts"
    );
}

#[tokio::test]
async fn nothing_is_refreshed_without_a_live_limited_entry() {
    let llm = UsageLlm::default();
    let clock = ManualClock::new();

    let mut fresh = AgentBuilder::new(llm.clone()).clock(clock.clone()).build();
    assert_eq!(fresh.keep_cache_warm().await.expect("no request"), None);

    let mut cold = agent_after_one_request(&llm, &clock, None).await;
    clock.advance(Duration::from_secs(7200));
    assert_eq!(cold.keep_cache_warm().await.expect("cold entry"), None);

    let mut permanent = agent_after_one_request(&llm, &clock, Some(CacheLifetime::Permanent)).await;
    clock.advance(Duration::from_secs(7200));
    assert_eq!(
        permanent.keep_cache_warm().await.expect("permanent entry"),
        None
    );
    assert_eq!(llm.keep_alives(), 0);
}
