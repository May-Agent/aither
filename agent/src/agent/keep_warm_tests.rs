//! Keeping the prompt-cache entry warm: an idle agent's refresh, and the
//! in-run keep-alive that sends only while a command outlives the entry.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_core::llm::{Event, LLMRequest, LanguageModel, Message, Usage, model::Profile};
use aither_sandbox::{SandboxMode, background_task_channel, job_registry_channel, test_ids::lease};
use futures_core::Stream;

use super::*;
use crate::{AgentBuilder, AgentOptions, BackgroundJobPolicy};

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

/// A model that records its requests and reports the same usage for each.
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
        futures_lite::stream::iter([Ok(Event::Usage(Usage::new(1200, 0)))])
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

/// How far into the hour the entry was started, leaving it cold in ten seconds.
const NEARLY_COLD: Duration = Duration::from_secs(3590);

fn waiting_agent(llm: &UsageLlm) -> Agent<UsageLlm> {
    let mut agent = AgentBuilder::new(llm.clone())
        .agent_kind(crate::AgentKind::Chatbot)
        .background_jobs(BackgroundJobPolicy::Await)
        .cache_keep_alive(crate::config::CacheKeepAlive::default())
        .build();
    agent.last_request = Some(LLMRequest::new(vec![Message::user("hello")]));
    agent.backdate_cache_entry(NEARLY_COLD);
    agent
}

#[tokio::test]
async fn an_idle_agent_refreshes_its_entry_without_touching_its_history() {
    let llm = UsageLlm::default();
    let mut agent = waiting_agent(&llm);
    let before = agent.context().len_recent();
    let cold_in = agent.cache_warmth().cold_in().expect("a limited entry");
    assert!(cold_in <= Duration::from_secs(10), "{cold_in:?}");

    let usage = agent
        .keep_cache_warm()
        .await
        .expect("the refresh succeeds")
        .expect("there was an entry to refresh");

    assert_eq!(usage, Usage::new(1200, 0));
    assert_eq!(llm.keep_alives(), 1);
    assert_eq!(agent.context().len_recent(), before, "history is untouched");
    assert_eq!(agent.cache_stats().requests(), 1, "the usage is recorded");
    let cold_in = agent.cache_warmth().cold_in().expect("a limited entry");
    assert!(cold_in > Duration::from_secs(3500), "{cold_in:?}");
}

#[tokio::test]
async fn nothing_is_refreshed_without_a_live_limited_entry() {
    let llm = UsageLlm::default();

    let mut fresh = AgentBuilder::new(llm.clone())
        .agent_kind(crate::AgentKind::Chatbot)
        .build();
    assert_eq!(fresh.keep_cache_warm().await.expect("no request"), None);

    let mut cold = waiting_agent(&llm);
    cold.backdate_cache_entry(Duration::from_secs(7200));
    assert_eq!(cold.keep_cache_warm().await.expect("cold entry"), None);

    let mut permanent = waiting_agent(&llm);
    permanent.use_cache_lifetime(CacheLifetime::Permanent);
    permanent.warmth.start_entry_now();
    assert_eq!(
        permanent.keep_cache_warm().await.expect("permanent entry"),
        None
    );
    assert_eq!(llm.keep_alives(), 0);
}

/// Parks `agent` on one running command bounded by `timeout`, for a little
/// while, and returns how many keep-alives went out meanwhile.
async fn keep_alives_while_waiting_on(timeout: Duration) -> usize {
    let llm = UsageLlm::default();
    let mut agent = waiting_agent(&llm);
    let (jobs, service) = job_registry_channel(aither_sandbox::test_budgets::output_policy());
    tokio::spawn(service.serve());
    let task = lease("build");
    jobs.bound_by_timeout(&task, timeout).await;
    jobs.register(
        7,
        &task,
        "exec-build",
        "cargo check",
        SandboxMode::Sandboxed,
        None,
    )
    .await;
    agent.job_registry = Some(jobs);

    let (sender, receiver) = background_task_channel();
    sender.task_started();
    let sink = EventSink::discarding();
    let wait = agent.wait_for_background_events(&receiver, &sink);
    let finish = async {
        tokio::time::sleep(Duration::from_millis(450)).await;
        sender.task_finished();
    };
    let (waited, ()) =
        tokio::time::timeout(Duration::from_secs(30), futures::future::join(wait, finish))
            .await
            .expect("the wait ends once the command does");
    waited.expect("the wait succeeds");
    llm.keep_alives()
}

#[tokio::test]
async fn a_command_outliving_the_entry_gets_a_keep_alive_before_expiry() {
    assert!(keep_alives_while_waiting_on(Duration::from_secs(3600)).await >= 1);
}

#[tokio::test]
async fn commands_ending_before_the_entry_goes_cold_get_no_keep_alive() {
    // The entry goes cold in ten seconds; the command ends in five.
    assert_eq!(
        keep_alives_while_waiting_on(Duration::from_secs(5)).await,
        0
    );
}
