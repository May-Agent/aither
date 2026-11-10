//! Idle compaction driven through the public builder only, on an injected
//! clock, the way a downstream host exercises it.

use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use aither_core::llm::{Event, LLMRequest, LanguageModel, model::Profile};
use futures_core::Stream;

use crate::context::CacheLifetime;
use crate::{
    Agent, AgentBuilder, AgentError, AgentEvent, CacheState, CompactionOutcome, CompactionTrigger,
    ManualClock, Run,
};

const HANDOFF_JSON: &str = r#"{"current_task":"keep going","key_decisions":[],"context_files":[],"errors_fixes":[],"pending_work":["continue"],"recovery_hints":[],"skills":[]}"#;

#[derive(Debug)]
struct MockError;

impl std::fmt::Display for MockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("mock error")
    }
}

impl std::error::Error for MockError {}

impl aither_core::llm::Classify for MockError {
    fn failure(&self) -> aither_core::llm::Failure {
        aither_core::llm::Failure::permanent()
    }
}

/// Replies "ok" to the run, then the handoff document to a compaction, and
/// counts the requests it received.
#[derive(Clone, Default)]
struct ScriptedLlm {
    requests: Arc<AtomicUsize>,
}

impl ScriptedLlm {
    fn requests(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }
}

impl LanguageModel for ScriptedLlm {
    type Error = MockError;

    fn respond(
        &self,
        _request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let seen = self.requests.fetch_add(1, Ordering::SeqCst);
        let text = if seen == 0 { "ok" } else { HANDOFF_JSON };
        futures_lite::stream::iter([Ok(Event::Text(text.to_string()))])
    }

    fn profile(&self) -> impl std::future::Future<Output = Result<Profile, Self::Error>> + Send {
        std::future::ready(Ok(Profile::new(
            "mock",
            "test",
            "mock-model",
            "mock model",
            1_000_000,
        )))
    }
}

/// Builds an agent and runs one turn on it.
async fn agent_after_one_turn(
    llm: ScriptedLlm,
    lifetime: Option<CacheLifetime>,
    clock: &ManualClock,
) -> Agent<ScriptedLlm> {
    let mut builder = AgentBuilder::new(llm).clock(clock.clone());
    if let Some(lifetime) = lifetime {
        builder = builder.cache_lifetime(lifetime);
    }
    let mut agent = builder.build();
    // Over the default compaction floor, which leaves short contexts alone.
    agent
        .query(&"word ".repeat(40_000))
        .await
        .expect("the first turn");
    agent
}

/// The compaction events a driven `run` emitted, in order.
async fn compaction_events(
    run: Run<impl std::future::Future<Output = Result<(), AgentError>>>,
) -> Vec<String> {
    let ControlFlow::Continue(seen) = run
        .drive(
            Vec::new(),
            std::future::pending(),
            |mut seen, timed| async move {
                match timed.event {
                    AgentEvent::CompactionStarted { trigger } => {
                        assert_eq!(trigger, CompactionTrigger::Idle);
                        seen.push("started".to_string());
                    }
                    AgentEvent::CompactionEnded { outcome } => seen.push(match outcome {
                        CompactionOutcome::Applied(_) => "applied".to_string(),
                        CompactionOutcome::NothingToCompact => "nothing to compact".to_string(),
                        CompactionOutcome::Failed(error) => format!("failed: {error}"),
                        CompactionOutcome::Discarded => "discarded".to_string(),
                    }),
                    _ => {}
                }
                ControlFlow::<std::convert::Infallible, _>::Continue(seen)
            },
        )
        .await
        .expect("the idle compaction succeeds");
    seen
}

/// Whether calling `compact_when_idle` emits no compaction event at all.
async fn compacts_nothing(agent: &mut Agent<ScriptedLlm>) -> bool {
    compaction_events(agent.compact_when_idle())
        .await
        .is_empty()
}

#[tokio::test]
async fn a_limited_lifetime_schedules_one_idle_compaction_on_the_injected_clock() {
    let llm = ScriptedLlm::default();
    let clock = ManualClock::new();
    let lifetime = CacheLifetime::Limited(Duration::from_secs(600));
    let mut agent = agent_after_one_turn(llm.clone(), Some(lifetime), &clock).await;
    assert_eq!(agent.cache_lifetime(), lifetime);
    assert_eq!(llm.requests(), 1);

    // Nothing is due while the clock stands still, and calling is a no-op.
    let due_in = agent
        .idle_compaction_due_in()
        .expect("a compaction is scheduled");
    assert!(due_in > Duration::ZERO && due_in < Duration::from_secs(600));
    assert!(compacts_nothing(&mut agent).await);
    assert_eq!(llm.requests(), 1);

    // Just short of the due point: still not due.
    clock.advance(due_in.saturating_sub(Duration::from_secs(1)));
    assert!(compacts_nothing(&mut agent).await);
    assert_eq!(llm.requests(), 1);

    // At the due point: exactly one compaction, reported as started and applied.
    clock.advance(Duration::from_secs(1));
    assert_eq!(agent.idle_compaction_due_in(), Some(Duration::ZERO));
    assert_eq!(
        compaction_events(agent.compact_when_idle()).await,
        ["started", "applied"]
    );
    assert_eq!(llm.requests(), 2);

    // Still idle, later in the same gap: nothing more.
    clock.advance(Duration::from_secs(30));
    assert!(compacts_nothing(&mut agent).await);
    assert_eq!(llm.requests(), 2);
}

#[tokio::test]
async fn a_permanent_lifetime_schedules_no_idle_compaction() {
    let llm = ScriptedLlm::default();
    let clock = ManualClock::new();
    let mut agent = agent_after_one_turn(llm.clone(), Some(CacheLifetime::Permanent), &clock).await;
    assert_eq!(agent.cache_lifetime(), CacheLifetime::Permanent);

    clock.advance(Duration::from_secs(10 * 86_400));
    assert_eq!(agent.idle_compaction_due_in(), None);
    assert!(compacts_nothing(&mut agent).await);
    assert_eq!(llm.requests(), 1);
}

#[tokio::test]
async fn the_default_lifetime_is_one_hour() {
    let clock = ManualClock::new();
    let agent = agent_after_one_turn(ScriptedLlm::default(), None, &clock).await;
    assert_eq!(
        agent.cache_lifetime(),
        CacheLifetime::Limited(Duration::from_secs(3600))
    );
}

#[tokio::test]
async fn the_agent_hands_hosts_a_warmth_handle_that_follows_its_requests() {
    let clock = ManualClock::new();
    let lifetime = CacheLifetime::Limited(Duration::from_secs(600));
    let mut agent = agent_after_one_turn(ScriptedLlm::default(), Some(lifetime), &clock).await;
    let warmth = agent.cache_warmth();
    assert_eq!(
        warmth.state(),
        CacheState::Warm {
            cold_in: Some(Duration::from_secs(600))
        }
    );

    clock.advance(Duration::from_secs(600));
    assert!(!warmth.is_warm());

    // A request after the entry went cold refreshes it for the same handle.
    let _ = agent.query("again").await;
    assert_eq!(warmth.cold_in(), Some(Duration::from_secs(600)));
}
