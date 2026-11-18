//! A response stream that breaks after it opened is sent again under the
//! model's retry policy, and only the replacement reaches the conversation.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_core::llm::{
    Classify, Event, Failure, LLMRequest, LanguageModel, Message, ReasoningState, RetryPolicy,
    Tool, ToolCall, ToolContext, ToolResult, model::Profile,
};
use futures_core::Stream;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::{AgentBuilder, AgentError, AgentEvent, OutputKind, RequestOutcome, RunOutcome};

type TestAgent = crate::Agent<BreakingLlm>;

/// A provider failure of a chosen kind.
#[derive(Debug, Clone, Copy)]
struct StreamError(Failure);

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stream failed: {:?}", self.0.kind)
    }
}

impl std::error::Error for StreamError {}

impl Classify for StreamError {
    fn failure(&self) -> Failure {
        self.0
    }
}

type Response = Vec<Result<Event, StreamError>>;

/// Answers each request with the next scripted response under `policy`, and
/// records every request it receives.
#[derive(Clone)]
struct BreakingLlm {
    script: Arc<Mutex<VecDeque<Response>>>,
    requests: Arc<Mutex<Vec<LLMRequest>>>,
    policy: RetryPolicy,
}

impl BreakingLlm {
    fn new(policy: RetryPolicy, script: impl IntoIterator<Item = Response>) -> Self {
        Self {
            script: Arc::new(Mutex::new(script.into_iter().collect())),
            requests: Arc::new(Mutex::new(Vec::new())),
            policy,
        }
    }

    fn requests(&self) -> Vec<LLMRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl LanguageModel for BreakingLlm {
    type Error = StreamError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        self.requests.lock().unwrap().push(request);
        let response = self
            .script
            .lock()
            .unwrap()
            .pop_front()
            .expect("the script has a response for every request");
        futures_lite::stream::iter(response)
    }

    fn retry_policy(&self) -> RetryPolicy {
        self.policy
    }

    fn profile(&self) -> impl std::future::Future<Output = Result<Profile, Self::Error>> + Send {
        std::future::ready(Ok(Profile::new(
            "breaking",
            "test",
            "breaking",
            "breaking model",
            128_000,
        )))
    }
}

/// Retries without waiting, at most `max_retries` times.
fn immediate(max_retries: u32) -> RetryPolicy {
    RetryPolicy {
        max_retries,
        initial_delay: Duration::ZERO,
        max_delay: Duration::ZERO,
        multiplier: 2,
    }
}

fn network_break() -> Result<Event, StreamError> {
    Err(StreamError(Failure::network()))
}

fn text(chunk: &str) -> Event {
    Event::Text(chunk.to_string())
}

/// Counts that a call reached it.
#[derive(Debug, Deserialize, JsonSchema)]
struct ProbeArgs {}

/// Counts the calls that reach it.
#[derive(Clone, Default)]
struct Probe(Arc<Mutex<usize>>);

impl Tool for Probe {
    fn name(&self) -> Cow<'static, str> {
        "probe".into()
    }

    type Arguments = ProbeArgs;
    type Res = ToolResult;

    fn call(
        &self,
        _arguments: Self::Arguments,
        _cx: ToolContext,
    ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
        *self.0.lock().unwrap() += 1;
        std::future::ready(Ok(ToolResult::text("probed")))
    }
}

/// Runs `prompt` to its end, returning what it ended with and every event it
/// emitted on the way.
async fn run(agent: &mut TestAgent, prompt: &str) -> (Result<(), AgentError>, Vec<AgentEvent>) {
    let mut events = Vec::new();
    let result = agent
        .run(prompt, std::iter::empty())
        .drive(
            &mut events,
            std::future::pending::<()>(),
            |events, timed| async move {
                events.push(timed.event);
                ControlFlow::Continue(events)
            },
        )
        .await
        .map(|_| ());
    (result, events)
}

fn retries(events: &[AgentEvent]) -> Vec<(u32, u32, Duration)> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::RequestEnded {
                outcome:
                    RequestOutcome::Retrying {
                        retry,
                        max_retries,
                        delay,
                        ..
                    },
                ..
            } => Some((*retry, *max_retries, *delay)),
            _ => None,
        })
        .collect()
}

fn is_retry(event: &AgentEvent) -> bool {
    matches!(
        event,
        AgentEvent::RequestEnded {
            outcome: RequestOutcome::Retrying { .. },
            ..
        }
    )
}

fn assistants(agent: &TestAgent) -> Vec<Message> {
    agent
        .context()
        .recent()
        .into_iter()
        .filter(|message| matches!(message, Message::Assistant { .. }))
        .cloned()
        .collect()
}

#[tokio::test]
async fn a_stream_that_breaks_mid_response_is_sent_again_and_only_the_replacement_is_kept() {
    let llm = BreakingLlm::new(
        immediate(3),
        [
            vec![
                Ok(Event::ReasoningState(ReasoningState::new("test", "broken"))),
                Ok(text("half an ans")),
                Ok(Event::ToolCall(ToolCall::new(
                    "call_broken",
                    "probe",
                    serde_json::json!({}),
                ))),
                network_break(),
            ],
            vec![
                Ok(Event::ReasoningState(ReasoningState::new("test", "whole"))),
                Ok(text("the whole answer")),
            ],
        ],
    );
    let probe = Probe::default();
    let mut agent = AgentBuilder::new(llm.clone()).tool(probe.clone()).build();

    let (result, events) = run(&mut agent, "answer").await;

    result.expect("the retried run completes");
    let requests = llm.requests();
    assert_eq!(requests.len(), 2, "the request is sent again once");
    assert_eq!(
        format!("{:?}", requests[0]),
        format!("{:?}", requests[1]),
        "the request sent again is the one that broke"
    );
    assert_eq!(
        *probe.0.lock().unwrap(),
        0,
        "a call from the broken stream never runs"
    );
    let kept = assistants(&agent);
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].content(), "the whole answer");
    assert_eq!(
        kept[0].reasoning(),
        [ReasoningState::new("test", "whole")],
        "only the replacement's reasoning is kept"
    );
    assert!(
        kept[0].tool_calls().is_empty(),
        "the broken stream's call is not in the conversation"
    );
    assert_eq!(retries(&events), [(1, 3, Duration::ZERO)]);
    let retry_at = events.iter().position(is_retry).expect("a retry event");
    let texts: Vec<_> = events
        .iter()
        .enumerate()
        .filter_map(|(at, event)| match event {
            AgentEvent::BlockDelta {
                kind: OutputKind::Text,
                text,
                ..
            } => Some((at, text.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(
        texts.iter().map(|(_, chunk)| *chunk).collect::<Vec<_>>(),
        ["half an ans", "the whole answer"]
    );
    assert!(
        texts[0].0 < retry_at && retry_at < texts[1].0,
        "clients learn of the replacement between the two attempts"
    );
    assert!(matches!(
        events.last(),
        Some(AgentEvent::RunEnded {
            outcome: RunOutcome::Completed { final_text },
            ..
        }) if final_text == "the whole answer"
    ));
}

#[tokio::test]
async fn retries_stop_at_the_policy_limit() {
    let llm = BreakingLlm::new(
        immediate(2),
        (0..3).map(|_| vec![Ok(text("partial")), network_break()]),
    );
    let mut agent = AgentBuilder::new(llm.clone()).build();

    let (result, events) = run(&mut agent, "answer").await;

    let error = result.expect_err("the run fails once the retries are spent");
    assert_eq!(error.failure(), Some(&Failure::network()));
    assert_eq!(llm.requests().len(), 3, "the first attempt and two retries");
    assert_eq!(
        retries(&events),
        [(1, 2, Duration::ZERO), (2, 2, Duration::ZERO)]
    );
    assert_eq!(assistants(&agent), [], "no broken attempt is kept");
}

#[tokio::test]
async fn a_non_retryable_break_mid_response_ends_the_turn() {
    let llm = BreakingLlm::new(
        immediate(3),
        [vec![
            Ok(text("partial")),
            Err(StreamError(Failure::permanent())),
        ]],
    );
    let mut agent = AgentBuilder::new(llm.clone()).build();

    let (result, events) = run(&mut agent, "answer").await;

    let error = result.expect_err("a permanent failure ends the run");
    assert_eq!(error.failure(), Some(&Failure::permanent()));
    assert_eq!(llm.requests().len(), 1);
    assert_eq!(retries(&events), []);
    assert_eq!(assistants(&agent), []);
}

#[tokio::test]
async fn a_failure_before_the_stream_opened_is_left_to_the_provider() {
    let llm = BreakingLlm::new(immediate(3), [vec![network_break()]]);
    let mut agent = AgentBuilder::new(llm.clone()).build();

    let (result, events) = run(&mut agent, "answer").await;

    result.expect_err("the provider already retried establishing the stream");
    assert_eq!(llm.requests().len(), 1);
    assert_eq!(retries(&events), []);
}

#[tokio::test]
async fn a_retry_waits_the_delay_the_upstream_requested() {
    let requested = Duration::from_millis(5);
    let policy = RetryPolicy {
        max_retries: 1,
        initial_delay: Duration::from_secs(1),
        max_delay: Duration::from_secs(1),
        multiplier: 2,
    };
    let llm = BreakingLlm::new(
        policy,
        [
            vec![
                Ok(text("partial")),
                Err(StreamError(
                    Failure::transient().with_retry_after(Some(requested)),
                )),
            ],
            vec![Ok(text("done"))],
        ],
    );
    let mut agent = AgentBuilder::new(llm.clone()).build();

    let (result, events) = run(&mut agent, "answer").await;

    result.expect("the retried run completes");
    assert_eq!(retries(&events), [(1, 1, requested)]);
}

#[derive(Debug, Clone)]
struct NamedStreamError {
    message: &'static str,
    failure: Failure,
}

impl std::fmt::Display for NamedStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message)
    }
}

impl std::error::Error for NamedStreamError {}

impl Classify for NamedStreamError {
    fn failure(&self) -> Failure {
        self.failure
    }
}

type NamedResponse = Vec<Result<Event, NamedStreamError>>;

#[derive(Clone)]
struct MalformedLlm {
    script: Arc<Mutex<VecDeque<NamedResponse>>>,
    requests: Arc<Mutex<Vec<LLMRequest>>>,
    policy: RetryPolicy,
}

impl LanguageModel for MalformedLlm {
    type Error = NamedStreamError;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        self.requests.lock().unwrap().push(request);
        let response = self
            .script
            .lock()
            .unwrap()
            .pop_front()
            .expect("scripted response");
        futures_lite::stream::iter(response)
    }

    fn retry_policy(&self) -> RetryPolicy {
        self.policy
    }

    fn profile(&self) -> impl std::future::Future<Output = Result<Profile, Self::Error>> + Send {
        std::future::ready(Ok(Profile::new(
            "malformed",
            "test",
            "malformed",
            "malformed model",
            128_000,
        )))
    }
}

#[tokio::test]
async fn a_malformed_function_call_emits_retrying_and_succeeds_on_retry() {
    let llm = MalformedLlm {
        script: Arc::new(Mutex::new(VecDeque::from([
            vec![Err(NamedStreamError {
                message: "provider rejected malformed function call payload",
                failure: Failure::permanent(),
            })],
            vec![Ok(text("recovered"))],
        ]))),
        requests: Arc::new(Mutex::new(Vec::new())),
        policy: immediate(2),
    };
    let mut agent = AgentBuilder::new(llm.clone()).build();
    let mut events = Vec::new();
    let result = agent
        .run("go", std::iter::empty())
        .drive(
            &mut events,
            std::future::pending::<()>(),
            |events, timed| async move {
                events.push(timed.event);
                ControlFlow::Continue(events)
            },
        )
        .await
        .map(|_| ());

    result.expect("retry after malformed function call succeeds");
    assert_eq!(llm.requests.lock().unwrap().len(), 2);
    assert_eq!(retries(&events), [(1, 2, Duration::ZERO)]);
    let outcomes: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::RequestEnded { outcome, .. } => Some(outcome),
            _ => None,
        })
        .collect();
    assert_eq!(outcomes.len(), 2);
    assert!(matches!(
        outcomes[0],
        RequestOutcome::Retrying { retry: 1, .. }
    ));
    assert!(matches!(outcomes[1], RequestOutcome::Completed));
}

struct RecordingTailSource {
    committed: Arc<Mutex<Vec<String>>>,
}

impl crate::ContextSource for RecordingTailSource {
    fn id(&self) -> crate::SourceId {
        crate::SourceId::new("tail-probe")
    }

    async fn blocks(&mut self, moment: crate::Moment) -> Vec<crate::ContextBlock> {
        self.peek(moment).await
    }

    fn peek(
        &self,
        moment: crate::Moment,
    ) -> impl std::future::Future<Output = Vec<crate::ContextBlock>> + Send {
        std::future::ready(match moment {
            crate::Moment::Request { .. } => {
                vec![crate::ContextBlock::tail("clock", "volatile-tail")]
            }
            _ => Vec::new(),
        })
    }

    async fn committed(&mut self, tail: &[crate::Block]) {
        self.committed
            .lock()
            .unwrap()
            .extend(tail.iter().map(|block| block.content().to_string()));
    }
}

struct MutableLiveSource {
    value: Arc<Mutex<String>>,
}

impl crate::LiveSource for MutableLiveSource {
    const TAG: &'static str = "live_doc";

    fn render(&self) -> impl std::future::Future<Output = Option<String>> + Send {
        std::future::ready(Some(self.value.lock().unwrap().clone()))
    }

    fn announce(&self, _change: &crate::LiveChange<'_>) -> crate::Block {
        crate::Block::new("live_doc_changed", "live_doc changed")
    }
}

#[tokio::test]
async fn a_pre_stream_failure_leaves_tail_prefix_alerts_and_warmth_uncommitted() {
    let llm = BreakingLlm::new(immediate(2), [vec![network_break()], vec![Ok(text("ok"))]]);
    let clock = crate::ManualClock::new();
    let committed = Arc::new(Mutex::new(Vec::new()));
    let live_value = Arc::new(Mutex::new("v1".to_string()));
    let mut agent = AgentBuilder::new(llm.clone())
        .clock(clock)
        .cache_lifetime(crate::context::CacheLifetime::Limited(Duration::from_secs(
            600,
        )))
        .live_source(MutableLiveSource {
            value: Arc::clone(&live_value),
        })
        .context_source(RecordingTailSource {
            committed: Arc::clone(&committed),
        })
        .build();
    agent
        .context_mut()
        .push(Message::system("undelivered-before-fail"));

    let (first_result, _) = run(&mut agent, "first").await;
    first_result.expect_err("first attempt fails before opening stream");

    assert!(
        committed.lock().unwrap().is_empty(),
        "tail must not commit when stream never opened"
    );
    assert!(
        !agent.cache_warmth().is_warm(),
        "cache warmth must remain cold when stream never opened"
    );
    assert_eq!(
        agent.context().fork().reassemble().system_messages_pruned,
        0,
        "alert must remain undelivered when stream never opened"
    );

    // Updating the live source before the first successful stream edits the
    // prefix in place rather than emitting a `<system-reminder>` live-change alert.
    *live_value.lock().unwrap() = "v2".to_string();
    let (second_result, second_events) = run(&mut agent, "second").await;
    second_result.expect("second attempt succeeds");

    assert_eq!(*committed.lock().unwrap(), ["volatile-tail"]);
    assert!(agent.cache_warmth().is_warm());
    assert_eq!(
        agent.context().fork().reassemble().system_messages_pruned,
        1,
        "delivered alert is now eligible for reassembly pruning"
    );
    assert!(
        !second_events.iter().any(|event| matches!(
            event,
            AgentEvent::ContextInjected {
                kind: crate::InjectionKind::LiveChange,
                ..
            }
        )),
        "live source update before first opened stream must edit prefix in place"
    );
}

#[derive(Clone)]
struct HangingStreamLlm {
    entered: async_channel::Sender<()>,
}

impl LanguageModel for HangingStreamLlm {
    type Error = StreamError;

    fn respond(
        &self,
        _request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        let entered = self.entered.clone();
        futures_lite::stream::once_future(async move {
            let _ = entered.send(()).await;
            std::future::pending::<Result<Event, StreamError>>().await
        })
    }

    fn profile(&self) -> impl std::future::Future<Output = Result<Profile, Self::Error>> + Send {
        std::future::ready(Ok(Profile::new(
            "hanging",
            "test",
            "hanging",
            "hanging model",
            128_000,
        )))
    }
}

#[tokio::test]
async fn cancelling_before_the_stream_yields_leaves_tail_and_warmth_uncommitted() {
    let (entered_tx, entered_rx) = async_channel::bounded(1);
    let committed = Arc::new(Mutex::new(Vec::new()));
    let clock = crate::ManualClock::new();
    let mut agent = AgentBuilder::new(HangingStreamLlm {
        entered: entered_tx,
    })
    .clock(clock)
    .cache_lifetime(crate::context::CacheLifetime::Limited(Duration::from_secs(
        600,
    )))
    .context_source(RecordingTailSource {
        committed: Arc::clone(&committed),
    })
    .build();

    let run = agent.run("hello", std::iter::empty());
    let cancel = run.cancel_handle();
    let stopper = async move {
        entered_rx.recv().await.expect("stream polled");
        cancel.cancel();
    };
    let (_, ()) = futures_lite::future::zip(
        run.drive((), std::future::pending::<()>(), |(), _timed| async move {
            ControlFlow::Continue(())
        }),
        stopper,
    )
    .await;

    assert!(committed.lock().unwrap().is_empty());
    assert!(!agent.cache_warmth().is_warm());
}
