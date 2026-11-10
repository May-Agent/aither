//! The loop's contract: what a run emits, what hooks decide, and how a run
//! ends.

#![allow(clippy::unused_async, clippy::unused_async_trait_impl)]

use std::borrow::Cow;
use std::collections::VecDeque;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_core::llm::tool::{Approval, ApprovalKind, ApprovalRequest, Approver, ApproverFailure};
use aither_core::llm::{
    Classify, Event, Failure, LLMRequest, LanguageModel, RetryPolicy, Tool, ToolCall, ToolContext,
    ToolResult, model::Profile,
};
use futures_core::Stream;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::{
    Agent, AgentBuilder, AgentError, AgentEvent, AgentPath, Block, ContextBlock, ContextSource,
    Hooks, InjectionKind, Moment, RunOutcome, SourceId, StopCx, StopDecision, SubagentRuns,
    TimedEvent, ToolDecision, ToolInvocation, ToolVerdict,
};

#[derive(Debug)]
struct NeverFails;

impl std::fmt::Display for NeverFails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("never fails")
    }
}

impl std::error::Error for NeverFails {}

impl Classify for NeverFails {
    fn failure(&self) -> Failure {
        Failure::permanent()
    }
}

/// Answers each request with the next scripted response and records every
/// request it receives.
#[derive(Clone)]
struct Scripted {
    script: Arc<Mutex<VecDeque<Vec<Event>>>>,
    requests: Arc<Mutex<Vec<LLMRequest>>>,
}

impl Scripted {
    fn new(script: impl IntoIterator<Item = Vec<Event>>) -> Self {
        Self {
            script: Arc::new(Mutex::new(script.into_iter().collect())),
            requests: Arc::default(),
        }
    }

    fn requests(&self) -> Vec<LLMRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl LanguageModel for Scripted {
    type Error = NeverFails;

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
        futures_lite::stream::iter(response.into_iter().map(Ok))
    }

    fn retry_policy(&self) -> RetryPolicy {
        RetryPolicy::default()
    }

    fn profile(&self) -> impl std::future::Future<Output = Result<Profile, Self::Error>> + Send {
        std::future::ready(Ok(Profile::new(
            "scripted",
            "test",
            "scripted",
            "scripted model",
            128_000,
        )))
    }
}

fn say(text: &str) -> Vec<Event> {
    vec![Event::Text(text.to_string())]
}

fn call(id: &str, name: &str) -> Vec<Event> {
    vec![Event::ToolCall(ToolCall::new(
        id,
        name,
        serde_json::json!({}),
    ))]
}

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArguments {}

/// Counts the calls that reach it.
#[derive(Clone, Default)]
struct Probe(Arc<Mutex<usize>>);

impl Tool for Probe {
    fn name(&self) -> Cow<'static, str> {
        "probe".into()
    }

    fn description(&self) -> Cow<'static, str> {
        "Counts the calls that reach it.".into()
    }

    type Arguments = NoArguments;
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

async fn run_to_end<M, H, C, S>(
    agent: &mut Agent<M, H, C, S>,
    prompt: &str,
) -> (Result<(), AgentError>, Vec<TimedEvent>)
where
    S: crate::HandoffSchema,
    M: LanguageModel + 'static,
    C: LanguageModel + 'static,
    H: Hooks,
{
    let mut events = Vec::new();
    let result = agent
        .run(prompt, std::iter::empty())
        .drive(
            &mut events,
            std::future::pending::<()>(),
            |events, timed| async move {
                events.push(timed);
                ControlFlow::Continue(events)
            },
        )
        .await
        .map(|_| ());
    (result, events)
}

fn kinds(events: &[TimedEvent]) -> Vec<&'static str> {
    events
        .iter()
        .map(|timed| match &timed.event {
            AgentEvent::RunStarted { .. } => "run-started",
            AgentEvent::RunEnded { .. } => "run-ended",
            AgentEvent::TurnStarted { .. } => "turn-started",
            AgentEvent::TurnEnded { .. } => "turn-ended",
            AgentEvent::ToolCallStarted { .. } => "tool-started",
            AgentEvent::ToolCallDecided { .. } => "tool-decided",
            AgentEvent::ToolCallEnded { .. } => "tool-ended",
            AgentEvent::Paused { .. } => "paused",
            AgentEvent::Resumed { .. } => "resumed",
            _ => "other",
        })
        .filter(|kind| *kind != "other")
        .collect()
}

fn outcome(events: &[TimedEvent]) -> &RunOutcome {
    events
        .iter()
        .rev()
        .find_map(|timed| match &timed.event {
            AgentEvent::RunEnded { outcome, .. } => Some(outcome),
            _ => None,
        })
        .expect("a run ends with RunEnded")
}

#[tokio::test]
async fn a_run_is_bracketed_by_its_start_and_end() {
    let llm = Scripted::new([say("hello")]);
    let mut agent = AgentBuilder::new(llm).build();

    let (result, events) = run_to_end(&mut agent, "hi").await;

    result.expect("the run completes");
    assert_eq!(
        kinds(&events),
        ["run-started", "turn-started", "turn-ended", "run-ended"]
    );
    assert!(events.iter().all(|timed| timed.agent == AgentPath::main()));
    assert!(matches!(
        outcome(&events),
        RunOutcome::Completed { final_text } if final_text == "hello"
    ));
}

#[tokio::test]
async fn query_returns_the_reply() {
    let mut agent = AgentBuilder::new(Scripted::new([say("four")])).build();
    assert_eq!(agent.query("2+2").await.unwrap(), "four");
}

#[tokio::test]
async fn a_tool_call_is_decided_run_and_reported() {
    let llm = Scripted::new([call("c1", "probe"), say("done")]);
    let probe = Probe::default();
    let mut agent = AgentBuilder::new(llm).tool(probe.clone()).build();

    let (result, events) = run_to_end(&mut agent, "go").await;

    result.expect("the run completes");
    assert_eq!(*probe.0.lock().unwrap(), 1);
    let tool: Vec<_> = kinds(&events)
        .into_iter()
        .filter(|kind| kind.starts_with("tool"))
        .collect();
    assert_eq!(tool, ["tool-started", "tool-decided", "tool-ended"]);
}

struct DenyProbe;

impl Hooks for DenyProbe {
    async fn before_tool(&self, call: &ToolInvocation<'_>) -> ToolDecision {
        assert_eq!(call.name, "probe");
        ToolDecision::Deny("probing is off".into())
    }
}

#[tokio::test]
async fn a_denied_call_does_not_run_and_the_model_reads_the_reason() {
    let llm = Scripted::new([call("c1", "probe"), say("ok")]);
    let probe = Probe::default();
    let mut agent = AgentBuilder::new(llm.clone())
        .tool(probe.clone())
        .hooks(DenyProbe)
        .build();

    let (result, events) = run_to_end(&mut agent, "go").await;

    result.expect("the run completes");
    assert_eq!(*probe.0.lock().unwrap(), 0);
    assert!(events.iter().any(|timed| matches!(
        &timed.event,
        AgentEvent::ToolCallDecided {
            decision: ToolVerdict::Denied { reason },
            ..
        } if reason == "probing is off"
    )));
    let second = format!("{:?}", llm.requests()[1].messages());
    assert!(second.contains("probing is off"), "{second}");
}

struct AskProbe;

impl Hooks for AskProbe {
    async fn before_tool(&self, _call: &ToolInvocation<'_>) -> ToolDecision {
        ToolDecision::Ask(ApprovalRequest::new(ApprovalKind::new("test"), "probe"))
    }
}

/// Approves once the test sends the go-ahead.
struct Gate(async_channel::Receiver<()>);

impl Approver for Gate {
    fn approve(
        &self,
        _request: ApprovalRequest,
    ) -> impl std::future::Future<Output = Result<Approval, ApproverFailure>> + Send {
        let gate = self.0.clone();
        async move {
            gate.recv().await.expect("the test answers");
            Ok(Approval::Approved)
        }
    }
}

#[tokio::test]
async fn a_pending_approval_pauses_the_run_and_resumes_it_on_the_answer() {
    let llm = Scripted::new([call("c1", "probe"), say("done")]);
    let probe = Probe::default();
    let (answer, gate) = async_channel::bounded(1);
    let mut agent = AgentBuilder::new(llm)
        .tool(probe.clone())
        .hooks(AskProbe)
        .approver(Gate(gate))
        .build();

    let mut events = Vec::new();
    let _ = agent
        .run("go", std::iter::empty())
        .drive(
            (&mut events, answer),
            std::future::pending::<()>(),
            |(events, answer), timed| async move {
                if let AgentEvent::Paused { call, kind, .. } = &timed.event {
                    assert!(call.is_some(), "the pause names the call");
                    assert_eq!(kind.as_str(), "test");
                    answer.send(()).await.unwrap();
                }
                events.push(timed);
                ControlFlow::Continue((events, answer))
            },
        )
        .await
        .expect("the run completes");

    assert_eq!(*probe.0.lock().unwrap(), 1);
    let order = kinds(&events);
    let paused = order.iter().position(|kind| *kind == "paused").unwrap();
    let resumed = order.iter().position(|kind| *kind == "resumed").unwrap();
    let ended = order.iter().position(|kind| *kind == "tool-ended").unwrap();
    assert!(paused < resumed && resumed < ended, "{order:?}");
}

#[tokio::test]
async fn an_approval_with_no_approver_fails_the_run() {
    let llm = Scripted::new([call("c1", "probe")]);
    let mut agent = AgentBuilder::new(llm)
        .tool(Probe::default())
        .hooks(AskProbe)
        .build();

    let (result, _) = run_to_end(&mut agent, "go").await;

    assert!(matches!(
        result.expect_err("no approver"),
        AgentError::Approval(_)
    ));
}

#[tokio::test]
async fn cancelling_a_run_ends_it_as_cancelled() {
    let llm = Scripted::new([say("never read")]);
    let mut agent = AgentBuilder::new(llm).build();
    let run = agent.run("hi", std::iter::empty());
    run.cancel();

    let mut events = Vec::new();
    let _ = run
        .drive(
            &mut events,
            std::future::pending::<()>(),
            |events, timed| async move {
                events.push(timed);
                ControlFlow::Continue(events)
            },
        )
        .await
        .expect("a cancelled run is not an error");

    assert!(matches!(outcome(&events), RunOutcome::Cancelled));
}

#[tokio::test]
async fn stopping_a_drive_still_delivers_the_cancelled_end() {
    let llm = Scripted::new([say("never read")]);
    let mut agent = AgentBuilder::new(llm).build();
    let mut events = Vec::new();

    let flow = agent
        .run("hi", std::iter::empty())
        .drive(
            &mut events,
            std::future::ready("stop"),
            |events, timed| async move {
                events.push(timed);
                ControlFlow::Continue(events)
            },
        )
        .await
        .expect("a stopped run is not an error");

    assert!(matches!(flow, ControlFlow::Break("stop")));
    assert!(matches!(outcome(&events), RunOutcome::Cancelled));
}

struct NotYet(Mutex<bool>);

impl Hooks for NotYet {
    async fn on_stop(&self, cx: &StopCx<'_>) -> StopDecision {
        let mut held = self.0.lock().unwrap();
        if std::mem::replace(&mut *held, true) {
            StopDecision::End
        } else {
            assert_eq!(cx.final_text, "first");
            StopDecision::Continue {
                reason: "check your work".into(),
            }
        }
    }
}

#[tokio::test]
async fn on_stop_can_hold_the_run_for_another_turn() {
    let llm = Scripted::new([say("first"), say("second")]);
    let mut agent = AgentBuilder::new(llm.clone())
        .hooks(NotYet(Mutex::new(false)))
        .build();

    let (result, events) = run_to_end(&mut agent, "go").await;

    result.expect("the run completes");
    assert!(matches!(
        outcome(&events),
        RunOutcome::Completed { final_text } if final_text == "second"
    ));
    let second = format!("{:?}", llm.requests()[1].messages());
    assert!(second.contains("check your work"), "{second}");
}

/// Puts a tail block in every request and a stable one at the first.
struct Notes {
    committed: Arc<Mutex<Vec<String>>>,
}

impl ContextSource for Notes {
    fn id(&self) -> SourceId {
        SourceId::new("notes")
    }

    async fn blocks(&mut self, moment: Moment) -> Vec<ContextBlock> {
        self.peek(moment).await
    }

    async fn peek(&self, moment: Moment) -> Vec<ContextBlock> {
        match moment {
            Moment::Request { reassembled: true } => vec![
                ContextBlock::stable("note", "stable-note"),
                ContextBlock::tail("clock", "tail-note"),
            ],
            Moment::Request { .. } => vec![ContextBlock::tail("clock", "tail-note")],
            _ => Vec::new(),
        }
    }

    async fn committed(&mut self, tail: &[Block]) {
        let mut committed = self.committed.lock().unwrap();
        committed.extend(tail.iter().map(|block| block.content().to_string()));
    }
}

#[tokio::test]
async fn a_source_contributes_stable_and_tail_blocks_and_hears_what_was_sent() {
    let llm = Scripted::new([say("hi")]);
    let committed = Arc::new(Mutex::new(Vec::new()));
    let mut agent = AgentBuilder::new(llm.clone())
        .context_source(Notes {
            committed: Arc::clone(&committed),
        })
        .build();

    let (result, events) = run_to_end(&mut agent, "go").await;

    result.expect("the run completes");
    let sent = format!("{:?}", llm.requests()[0].messages());
    assert!(sent.contains("tail-note"), "{sent}");
    assert_eq!(*committed.lock().unwrap(), ["tail-note"]);
    let kinds: Vec<_> = events
        .iter()
        .filter_map(|timed| match &timed.event {
            AgentEvent::ContextInjected { source, kind } if source.as_str() == "notes" => {
                Some(*kind)
            }
            _ => None,
        })
        .collect();
    assert_eq!(kinds, [InjectionKind::Tail]);
}

/// Returns a tail block where only a request has a tail.
struct BadTail;

impl ContextSource for BadTail {
    fn id(&self) -> SourceId {
        SourceId::new("bad")
    }

    async fn blocks(&mut self, moment: Moment) -> Vec<ContextBlock> {
        self.peek(moment).await
    }

    async fn peek(&self, moment: Moment) -> Vec<ContextBlock> {
        match moment {
            Moment::FinalTurn => vec![ContextBlock::tail("oops", "tail at the end")],
            _ => Vec::new(),
        }
    }
}

#[tokio::test]
#[should_panic(expected = "only a request has a tail")]
async fn a_tail_block_outside_a_request_is_a_contract_breach() {
    let mut agent = AgentBuilder::new(Scripted::new([say("hi")]))
        .context_source(BadTail)
        .build();
    let _ = run_to_end(&mut agent, "go").await;
}

#[tokio::test]
async fn instructions_render_in_the_order_they_were_added() {
    let llm = Scripted::new([say("hi")]);
    let mut agent = AgentBuilder::new(llm.clone())
        .instruction("first rule")
        .instruction("second rule")
        .build();

    let (result, _) = run_to_end(&mut agent, "go").await;

    result.expect("the run completes");
    let sent = format!("{:?}", llm.requests()[0].messages());
    let first = sent.find("first rule").expect("first instruction");
    let second = sent.find("second rule").expect("second instruction");
    assert!(first < second);
}

#[tokio::test]
async fn a_child_is_registered_bracketed_and_cancelled_through_the_registry() {
    let runs = SubagentRuns::default();
    let started_by = aither_core::llm::CallIdentity::new(
        aither_core::llm::ToolCallId::new("c1"),
        aither_core::llm::AgentIdentity::Main,
    );
    let run = runs.start(&started_by, "researcher", "look into it");
    assert_eq!(runs.list().len(), 1);
    assert_eq!(runs.list()[0].name, "researcher");

    let llm = Scripted::new([say("found it")]);
    let mut child = AgentBuilder::new(llm).identity(run.identity()).build();
    let signal = run.cancel_signal();
    assert!(!signal.is_cancelled());
    assert!(runs.cancel(run.id()));
    assert!(signal.is_cancelled());
    assert!(!runs.cancel(999));

    let outcome = run.run(&mut child, "look into it").await;

    assert!(matches!(outcome, RunOutcome::Cancelled));
    assert!(runs.list().is_empty());
}

#[tokio::test]
async fn a_childs_events_reach_the_parents_stream_with_its_path() {
    let runs = SubagentRuns::default();
    let started_by = aither_core::llm::CallIdentity::new(
        aither_core::llm::ToolCallId::new("c1"),
        aither_core::llm::AgentIdentity::Main,
    );
    let run = runs.start(&started_by, "researcher", "look");
    let path = run.path().clone();
    let mut child = AgentBuilder::new(Scripted::new([say("found")]))
        .identity(run.identity())
        .build();
    let outcome = run.run(&mut child, "look").await;
    assert!(matches!(outcome, RunOutcome::Completed { .. }));

    let mut parent = AgentBuilder::new(Scripted::new([say("hi")]))
        .subagents(runs)
        .build();
    let (result, events) = run_to_end(&mut parent, "go").await;
    result.expect("the parent completes");

    let child_events: Vec<_> = events.iter().filter(|timed| timed.agent == path).collect();
    assert!(
        child_events
            .iter()
            .any(|timed| matches!(timed.event, AgentEvent::RunStarted { .. }))
    );
    assert!(events.iter().any(|timed| matches!(
        &timed.event,
        AgentEvent::SubagentStarted { child, name, .. } if *child == path && name == "researcher"
    )));
    assert!(events.iter().any(|timed| matches!(
        &timed.event,
        AgentEvent::SubagentEnded { child, outcome: RunOutcome::Completed { .. } } if *child == path
    )));
    let _ = Duration::ZERO;
}

/// A tail block that reads the source's state and a one-shot stable block
/// that `blocks` consumes and `peek` leaves in place.
struct OneShot {
    pending: bool,
}

impl ContextSource for OneShot {
    fn id(&self) -> SourceId {
        SourceId::new("one-shot")
    }

    async fn blocks(&mut self, moment: Moment) -> Vec<ContextBlock> {
        let blocks = self.peek(moment).await;
        if matches!(moment, Moment::Request { .. }) {
            self.pending = false;
        }
        blocks
    }

    async fn peek(&self, moment: Moment) -> Vec<ContextBlock> {
        let Moment::Request { .. } = moment else {
            return Vec::new();
        };
        let mut blocks = vec![ContextBlock::tail("state", "tail-state")];
        if self.pending {
            blocks.push(ContextBlock::stable("correction", "one-shot-correction"));
        }
        blocks
    }
}

fn texts(messages: &[aither_core::llm::Message]) -> Vec<String> {
    messages
        .iter()
        .map(|message| format!("{message:?}"))
        .collect()
}

#[tokio::test]
async fn a_preview_carries_what_the_run_then_sends_and_consumes_nothing() {
    let llm = Scripted::new([say("hi")]);
    let mut agent = AgentBuilder::new(llm.clone())
        .context_source(OneShot { pending: true })
        .build();

    let previewed = agent
        .preview_request_messages("go", std::iter::empty())
        .await
        .expect("the preview renders");
    let again = agent
        .preview_request_messages("go", std::iter::empty())
        .await
        .expect("a second preview renders");
    assert_eq!(
        texts(&previewed),
        texts(&again),
        "a preview consumes nothing"
    );

    let (result, _) = run_to_end(&mut agent, "go").await;
    result.expect("the run completes");

    let sent = llm.requests()[0].messages().to_vec();
    assert_eq!(
        texts(&previewed),
        texts(&sent),
        "the preview is the request the run sends"
    );
    let sent_text = texts(&sent).join("\n");
    assert!(sent_text.contains("tail-state"), "{sent_text}");
    assert!(
        sent_text.contains("one-shot-correction"),
        "the one-shot block reaches the run after a preview: {sent_text}"
    );
}
