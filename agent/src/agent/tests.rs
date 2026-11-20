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
    Agent, AgentBuilder, AgentCheckpoint, AgentError, AgentEvent, AgentPath, Block, ContextBlock,
    ContextSource, Hooks, InjectionKind, Moment, RunOutcome, SourceId, StopCx, StopDecision,
    SubagentRuns, TimedEvent, ToolDecision, ToolInvocation, ToolVerdict,
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

/// Panics on every call.
struct Panicker;

impl Tool for Panicker {
    fn name(&self) -> Cow<'static, str> {
        "panicker".into()
    }

    fn description(&self) -> Cow<'static, str> {
        "Panics on every call.".into()
    }

    type Arguments = NoArguments;
    type Res = ToolResult;

    async fn call(
        &self,
        _arguments: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        panic!("the disk is on fire");
    }
}

fn tool_results(llm: &Scripted) -> Vec<String> {
    llm.requests()
        .last()
        .expect("the model was asked at least once")
        .messages()
        .iter()
        .filter(|message| message.role() == aither_core::llm::Role::Tool)
        .map(|message| message.content().to_string())
        .collect()
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

#[tokio::test]
async fn a_panicking_tool_fails_its_call_and_the_next_call_runs() {
    let llm = Scripted::new([call("c1", "panicker"), call("c2", "probe"), say("done")]);
    let probe = Probe::default();
    let mut agent = AgentBuilder::new(llm.clone())
        .tool(Panicker)
        .tool(probe.clone())
        .build();

    let (result, _) = run_to_end(&mut agent, "go").await;

    result.expect("the run survives the panic");
    assert_eq!(*probe.0.lock().unwrap(), 1, "the next call ran normally");
    let results = tool_results(&llm);
    assert_eq!(
        results,
        [
            "Error: tool 'panicker' panicked: the disk is on fire",
            "probed"
        ]
    );
}

#[tokio::test]
async fn a_panic_in_one_parallel_call_leaves_the_other_result_intact() {
    let llm = Scripted::new([
        vec![
            Event::ToolCall(ToolCall::new("c1", "panicker", serde_json::json!({}))),
            Event::ToolCall(ToolCall::new("c2", "probe", serde_json::json!({}))),
        ],
        say("done"),
    ]);
    let probe = Probe::default();
    let mut agent = AgentBuilder::new(llm.clone())
        .tool(Panicker)
        .tool(probe.clone())
        .build();

    let (result, _) = run_to_end(&mut agent, "go").await;

    result.expect("the run survives the panic");
    assert_eq!(*probe.0.lock().unwrap(), 1);
    let results = tool_results(&llm);
    assert_eq!(results.len(), 2);
    assert!(results.contains(&"probed".to_owned()), "{results:?}");
    assert!(
        results.contains(&"Error: tool 'panicker' panicked: the disk is on fire".to_owned()),
        "{results:?}"
    );
}

#[tokio::test]
async fn a_checkpoint_taken_mid_run_restores_the_completed_steps() {
    let llm = Scripted::new([call("c1", "probe"), say("done")]);
    let mut agent = AgentBuilder::new(llm).tool(Probe::default()).build();
    let run = agent.run("go", std::iter::empty());
    let checkpoints = run.checkpoints();

    let flow = run
        .drive(
            (),
            std::future::pending::<Option<AgentCheckpoint>>(),
            |(), timed| {
                let taken = match timed.event {
                    AgentEvent::Checkpoint { .. } => Some(checkpoints.latest()),
                    _ => None,
                };
                async move { taken.map_or(ControlFlow::Continue(()), ControlFlow::Break) }
            },
        )
        .await
        .expect("the run is stopped, not failed");
    let ControlFlow::Break(checkpoint) = flow else {
        panic!("the run emitted a checkpoint");
    };
    let checkpoint = checkpoint.expect("a checkpoint event follows a recorded checkpoint");

    let mut restored = AgentBuilder::new(Scripted::new([]))
        .tool(Probe::default())
        .build();
    restored
        .restore_checkpoint(checkpoint)
        .expect("the checkpoint restores");
    let history = format!("{:?}", restored.history());
    assert!(history.contains("go"), "the user message: {history}");
    assert!(history.contains("c1"), "the completed tool call: {history}");
}

struct AttachImageHook(aither_core::llm::Attachment);

impl Hooks for AttachImageHook {
    async fn after_tool(&self, _outcome: &crate::ToolOutcome) -> crate::AfterTool {
        crate::AfterTool::ReplaceWithAttachments {
            result: ToolResult::text("attached image"),
            attachments: vec![self.0.clone()],
        }
    }
}

#[tokio::test]
async fn after_tool_replace_with_attachments_reaches_the_model() {
    let attachment = aither_core::llm::Attachment::new(
        "data:image/png;base64,AA==".parse().expect("data url"),
        "image/png".parse().expect("image MIME"),
    );
    let llm = Scripted::new([call("c1", "probe"), say("done")]);
    let mut agent = AgentBuilder::new(llm.clone())
        .tool(Probe::default())
        .hooks(AttachImageHook(attachment.clone()))
        .build();

    let (result, _) = run_to_end(&mut agent, "go").await;

    result.expect("the run completes");
    let requests = llm.requests();
    let tool_message = requests[1]
        .messages()
        .iter()
        .find(|msg| msg.tool_call_id() == Some("c1"))
        .expect("tool message in second request");
    assert_eq!(tool_message.content(), "attached image");
    assert_eq!(tool_message.attachments(), &[attachment]);
}

/// Validates Claude's `MidConversationSystem` placement rule: after the
/// leading `Role::System` prefix, every `Role::System` turn must follow a
/// `Role::User` or `Role::Tool` turn, never follow another `Role::System`, and
/// be either last or followed by `Role::Assistant` (skipping `Configuration`).
fn assert_valid_claude_system_turns(messages: &[aither_core::llm::Message]) {
    use aither_core::llm::Role;

    let leading = messages
        .iter()
        .take_while(|msg| msg.role() == Role::System)
        .count();
    let turns: Vec<(usize, Role)> = messages[leading..]
        .iter()
        .enumerate()
        .filter(|(_, msg)| msg.role() != Role::Configuration)
        .map(|(offset, msg)| (leading + offset, msg.role()))
        .collect();
    for (pos, &(index, role)) in turns.iter().enumerate() {
        if role != Role::System {
            continue;
        }
        let follows_user = pos
            .checked_sub(1)
            .is_some_and(|prev| matches!(turns[prev].1, Role::User | Role::Tool));
        assert!(
            follows_user,
            "mid-conversation system turn at index {index} must follow User/Tool: {messages:?}"
        );
        if let Some(&(_, next_role)) = turns.get(pos + 1) {
            assert_eq!(
                next_role,
                Role::Assistant,
                "mid-conversation system turn at index {index} must be last or before Assistant: {messages:?}"
            );
        }
    }
}

#[derive(Clone)]
struct ClaudeScripted {
    script: Arc<Mutex<VecDeque<Vec<Event>>>>,
    requests: Arc<Mutex<Vec<LLMRequest>>>,
    context_length: u32,
}

impl ClaudeScripted {
    fn new(context_length: u32, script: impl IntoIterator<Item = Vec<Event>>) -> Self {
        Self {
            script: Arc::new(Mutex::new(script.into_iter().collect())),
            requests: Arc::default(),
            context_length,
        }
    }

    fn requests(&self) -> Vec<LLMRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl LanguageModel for ClaudeScripted {
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
            .expect("scripted response");
        futures_lite::stream::iter(response.into_iter().map(Ok))
    }

    fn profile(&self) -> impl std::future::Future<Output = Result<Profile, Self::Error>> + Send {
        let context_length = self.context_length;
        std::future::ready(Ok(Profile::new(
            "claude-sonnet-5-5",
            "anthropic",
            "claude-sonnet-5-5",
            "Claude",
            context_length,
        )
        .with_ability(aither_core::llm::model::Ability::MidConversationSystem)))
    }
}

struct MultiAlertSource {
    object_once: Mutex<bool>,
}

impl ContextSource for MultiAlertSource {
    fn id(&self) -> SourceId {
        SourceId::new("multi-alert")
    }

    async fn blocks(&mut self, moment: Moment) -> Vec<ContextBlock> {
        match moment {
            Moment::Request { .. } => vec![
                ContextBlock::stable("alert-a", "source-stable-a"),
                ContextBlock::stable("alert-b", "source-stable-b"),
                ContextBlock::tail("clock", "live-tail-clock"),
            ],
            Moment::FinalTurn => {
                let mut pending = self.object_once.lock().unwrap();
                if std::mem::take(&mut *pending) {
                    vec![ContextBlock::stable("objection", "source-final-objection")]
                } else {
                    Vec::new()
                }
            }
            Moment::ToolTurnEnd => Vec::new(),
        }
    }

    async fn peek(&self, _moment: Moment) -> Vec<ContextBlock> {
        Vec::new()
    }
}

struct HookAlertsAndHold {
    hold_once: Mutex<bool>,
}

impl Hooks for HookAlertsAndHold {
    async fn before_request(&self, _cx: &crate::RequestCx<'_>) -> Vec<ContextBlock> {
        vec![ContextBlock::stable("hook-note", "hook-stable-note")]
    }

    async fn on_stop(&self, _cx: &StopCx<'_>) -> StopDecision {
        let mut hold = self.hold_once.lock().unwrap();
        if std::mem::take(&mut *hold) {
            StopDecision::Continue {
                reason: "verify output".into(),
            }
        } else {
            StopDecision::End
        }
    }
}

#[tokio::test]
async fn claude_mid_conversation_system_adjacency_holds_across_hooks_compaction_and_next_run() {
    let restore_root = tempfile::tempdir().expect("tempdir");
    async_fs::write(restore_root.path().join("guide.md"), "important context")
        .await
        .expect("write guide");

    let handoff_json = r#"{"current_task":"finish","key_decisions":[],"context_files":[{"path":"guide.md","why":"keep handy"}],"errors_fixes":[],"pending_work":["reply"],"recovery_hints":[],"skills":[]}"#;
    let llm = ClaudeScripted::new(
        128_000,
        [
            say("first draft"),
            say("revised draft"),
            say(handoff_json),
            say("follow-up answer"),
        ],
    );
    let mut agent = AgentBuilder::new(llm.clone())
        .instruction("system prefix")
        .restore_roots([restore_root.path().to_path_buf()])
        .context_source(MultiAlertSource {
            object_once: Mutex::new(true),
        })
        .hooks(HookAlertsAndHold {
            hold_once: Mutex::new(true),
        })
        .build();

    let (first_run, _) = run_to_end(&mut agent, "start task").await;
    first_run.expect("first run completes");

    let compact_flow = agent
        .compact(None)
        .drive((), std::future::pending::<()>(), |(), _timed| async move {
            ControlFlow::Continue(())
        })
        .await;
    let _ = compact_flow.expect("compaction completes");

    let (second_run, _) = run_to_end(&mut agent, "next user turn").await;
    second_run.expect("second run completes");

    let requests = llm.requests();
    assert_eq!(requests.len(), 4);
    for request in &requests {
        assert_valid_claude_system_turns(request.messages());
    }
    let last_text: String = requests[3]
        .messages()
        .iter()
        .map(aither_core::llm::Message::content)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        last_text.contains("<external-content boundary=\""),
        "restored file must be wrapped in ExternalContent: {last_text}"
    );
}

#[derive(Clone)]
struct NamedTool {
    name: &'static str,
    reply: String,
}

impl Tool for NamedTool {
    fn name(&self) -> Cow<'static, str> {
        self.name.into()
    }

    fn description(&self) -> Cow<'static, str> {
        "Test tool.".into()
    }

    type Arguments = NoArguments;
    type Res = ToolResult;

    fn call(
        &self,
        _arguments: Self::Arguments,
        _cx: ToolContext,
    ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
        std::future::ready(Ok(ToolResult::text(self.reply.clone())))
    }
}

struct KeepDecisionHook;

impl Hooks for KeepDecisionHook {
    async fn after_tool(&self, outcome: &crate::ToolOutcome) -> crate::AfterTool {
        if outcome.name == "decision" {
            crate::AfterTool::Keep.kept()
        } else {
            crate::AfterTool::Keep
        }
    }
}

#[tokio::test]
async fn live_after_tool_kept_exchanges_survive_in_run_compactions_in_order() {
    let handoff_json = r#"{"current_task":"keep going","key_decisions":[],"context_files":[],"errors_fixes":[],"pending_work":["finish"],"recovery_hints":[],"skills":[]}"#;
    let pad = "x".repeat(6_000);
    let compactor = ClaudeScripted::new(128_000, [say(handoff_json), say(handoff_json)]);
    let main_llm = ClaudeScripted::new(
        10_000,
        [
            vec![
                Event::ToolCall(ToolCall::new("k1", "decision", serde_json::json!({}))),
                Event::ToolCall(ToolCall::new("s1", "scratch", serde_json::json!({}))),
            ],
            call("k2", "decision"),
            say("all done"),
            say("after cold cache"),
        ],
    );
    let clock = crate::ManualClock::new();
    let policy = crate::ContextPolicy {
        handoff_reserve: 2_000,
        compact_ceiling: 1_200,
        start_distance: 100,
        warn_distance: 200,
        user_message_budget: 0,
        ..crate::ContextPolicy::default()
    };
    let config = crate::AgentConfig {
        context_policy: policy,
        ..crate::AgentConfig::default()
    };
    let mut agent = AgentBuilder::new(main_llm.clone())
        .config(config)
        .clock(clock.clone())
        .cache_lifetime(crate::context::CacheLifetime::Limited(Duration::from_secs(
            600,
        )))
        .compaction_model(compactor, aither_core::llm::model::Parameters::default())
        .tool(NamedTool {
            name: "decision",
            reply: "approved-choice".into(),
        })
        .tool(NamedTool {
            name: "scratch",
            reply: pad,
        })
        .hooks(KeepDecisionHook)
        .build();

    let (result, _) = run_to_end(&mut agent, "go").await;
    result.expect("in-run pause-point compaction succeeds");

    // Advance past the cache lifetime to trigger ColdCache compaction on the next run too.
    clock.advance(Duration::from_secs(601));
    let (second, _) = run_to_end(&mut agent, "resume").await;
    second.expect("cold-cache compaction succeeds");

    let final_messages = main_llm
        .requests()
        .last()
        .expect("final request")
        .messages()
        .to_vec();
    let kept_tool_ids: Vec<&str> = final_messages
        .iter()
        .filter_map(|msg| msg.tool_call_id())
        .collect();
    assert_eq!(
        kept_tool_ids,
        ["k1", "k2"],
        "kept exchanges survive in chronological order without duplicates while unkept `s1` is compacted"
    );
    let assistant_call_ids: Vec<&str> = final_messages
        .iter()
        .flat_map(|msg| msg.tool_calls().iter().map(|call| call.id.as_str()))
        .collect();
    assert_eq!(assistant_call_ids, ["k1", "k2"]);
}

#[derive(Clone)]
struct YieldingClaudeScripted {
    inner: ClaudeScripted,
}

impl LanguageModel for YieldingClaudeScripted {
    type Error = NeverFails;

    fn respond(
        &self,
        request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, Self::Error>> + Send {
        self.inner.requests.lock().unwrap().push(request);
        let response = self
            .inner
            .script
            .lock()
            .unwrap()
            .pop_front()
            .expect("scripted response");
        futures_lite::stream::unfold(Some(response), |state| async move {
            let mut events = state?;
            futures_lite::future::yield_now().await;
            if events.is_empty() {
                return None;
            }
            let next = events.remove(0);
            Some((Ok(next), Some(events)))
        })
    }

    fn profile(&self) -> impl std::future::Future<Output = Result<Profile, Self::Error>> + Send {
        self.inner.profile()
    }
}

#[tokio::test]
async fn generic_async_compaction_folds_post_fork_turns_into_handoff_and_rotates_epoch() {
    use aither_core::llm::ReasoningState;

    let workdir = tempfile::tempdir().expect("tempdir");
    let spill_dir = workdir.path().join("outputs");
    async_fs::write(workdir.path().join("spec.md"), "spec-invariant-42")
        .await
        .expect("write spec");

    let handoff_json = r#"{"current_task":"post-fork","key_decisions":[],"context_files":[{"path":"spec.md","why":"core spec"}],"errors_fixes":[],"pending_work":["finish"],"recovery_hints":[],"skills":[]}"#;
    let claude_thinking = ReasoningState::new(
        "claude",
        serde_json::json!({
            "type": "thinking",
            "thinking": "bound to pre-compaction prefix",
            "signature": "sig-old"
        })
        .to_string(),
    );

    let compactor = ClaudeScripted::new(128_000, [say(handoff_json), say(handoff_json)]);
    let main_llm = YieldingClaudeScripted {
        inner: ClaudeScripted::new(
            20_000,
            [
                vec![
                    Event::ReasoningState(claude_thinking),
                    Event::Text("post-fork tool step".to_string()),
                    Event::ToolCall(ToolCall::new("c_post", "big_tool", serde_json::json!({}))),
                ],
                say("finished"),
            ],
        ),
    };
    let policy = crate::ContextPolicy {
        handoff_reserve: 2_000,
        compact_ceiling: 10_000,
        // High `start_distance` forks `WindowFilling` compaction on Turn 1 without hitting `should_pause`.
        start_distance: 9_900,
        warn_distance: 9_950,
        spill_threshold: 50,
        spill_head: 10,
        user_message_budget: 2_000,
        ..crate::ContextPolicy::default()
    };
    let mut agent = AgentBuilder::new(main_llm.clone())
        .config(crate::AgentConfig {
            context_policy: policy,
            ..crate::AgentConfig::default()
        })
        .restore_roots([workdir.path().to_path_buf()])
        .spill_dir(spill_dir.clone())
        .prompt_cache(crate::PromptCache::new("sess-196"))
        .compaction_model(compactor, aither_core::llm::model::Parameters::default())
        .tool(NamedTool {
            name: "big_tool",
            reply: "post-fork-result-line\n".repeat(40),
        })
        .build();

    let (result, events) = run_to_end(&mut agent, &"seed ".repeat(300)).await;
    result.expect("run with async WindowFilling compaction completes");

    assert!(events.iter().any(|timed| matches!(
        timed.event,
        AgentEvent::CompactionStarted {
            trigger: crate::CompactionTrigger::WindowFilling
        }
    )));
    assert!(events.iter().any(|timed| matches!(
        timed.event,
        AgentEvent::CompactionEnded {
            outcome: crate::CompactionOutcome::Applied(_)
        }
    )));

    let requests = main_llm.inner.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0]
            .parameters()
            .cache
            .openai
            .as_ref()
            .and_then(|c| c.key.as_deref()),
        Some("sess-196:0")
    );
    assert_eq!(
        requests[1]
            .parameters()
            .cache
            .openai
            .as_ref()
            .and_then(|c| c.key.as_deref()),
        Some("sess-196:1"),
        "PromptCache epoch must advance when compaction rewrites the prefix"
    );

    assert_folded_async_handoff_request(requests[1].messages());
}

fn assert_folded_async_handoff_request(post_messages: &[aither_core::llm::Message]) {
    assert_valid_claude_system_turns(post_messages);
    assert!(
        post_messages
            .iter()
            .all(|msg| msg.role() != aither_core::llm::Role::Assistant
                && msg.role() != aither_core::llm::Role::Tool),
        "post-fork assistant/tool chain must be folded into <handoff> rather than left trailing: {post_messages:?}"
    );
    let handoff_msg = post_messages
        .iter()
        .find(|msg| {
            msg.role() == aither_core::llm::Role::System && msg.content().contains("<handoff>")
        })
        .expect("system handoff message");
    assert!(handoff_msg.content().contains("## User Messages"));
    assert!(
        handoff_msg
            .content()
            .contains("## Activity During Compaction")
    );
    assert!(handoff_msg.content().contains("post-fork tool step"));
    assert!(handoff_msg.content().contains("big_tool"));
    assert!(handoff_msg.content().contains("c_post"));
    assert!(
        handoff_msg.content().contains("outputs/compaction-"),
        "oversized post-fork tool output must spill to outputs/: {}",
        handoff_msg.content()
    );

    let resume_user = post_messages
        .iter()
        .find(|msg| msg.role() == aither_core::llm::Role::User)
        .expect("opening user resume turn");
    assert!(
        resume_user
            .content()
            .contains("while compaction was in flight")
    );
    assert!(
        resume_user.content().contains("spec-invariant-42"),
        "restored context file must be inlined in the post-compaction resume turn: {}",
        resume_user.content()
    );
}

struct StableEveryRequestSource;

impl ContextSource for StableEveryRequestSource {
    fn id(&self) -> SourceId {
        SourceId::new("dup-check")
    }

    async fn blocks(&mut self, moment: Moment) -> Vec<ContextBlock> {
        match moment {
            Moment::Request { .. } => vec![ContextBlock::stable("dup-src", "unique-source-alert")],
            _ => Vec::new(),
        }
    }

    async fn peek(&self, _moment: Moment) -> Vec<ContextBlock> {
        Vec::new()
    }
}

struct StableBeforeRequestHook;

impl Hooks for StableBeforeRequestHook {
    async fn before_request(&self, _cx: &crate::RequestCx<'_>) -> Vec<ContextBlock> {
        vec![ContextBlock::stable("dup-hook", "unique-hook-alert")]
    }
}

#[tokio::test]
async fn pause_point_compaction_deduplicates_before_request_and_source_stable_blocks() {
    let handoff_json = r#"{"current_task":"dedup","key_decisions":[],"context_files":[],"errors_fixes":[],"pending_work":["done"],"recovery_hints":[],"skills":[]}"#;
    let compactor = ClaudeScripted::new(128_000, [say(handoff_json), say(handoff_json)]);
    let main_llm = ClaudeScripted::new(20_000, [say("ok")]);
    let policy = crate::ContextPolicy {
        handoff_reserve: 2_000,
        compact_ceiling: 600,
        start_distance: 200,
        warn_distance: 300,
        user_message_budget: 0,
        ..crate::ContextPolicy::default()
    };
    let mut agent = AgentBuilder::new(main_llm.clone())
        .config(crate::AgentConfig {
            context_policy: policy,
            ..crate::AgentConfig::default()
        })
        .compaction_model(compactor, aither_core::llm::model::Parameters::default())
        .context_source(StableEveryRequestSource)
        .hooks(StableBeforeRequestHook)
        .build();
    // Pre-populate delivered conversation above `compact_ceiling` (600 tokens) so
    // `prepare_request` enters pause-point `make_room` and runs `collect_request_context` twice.
    agent
        .context_mut()
        .push(aither_core::llm::Message::user("x".repeat(4_000)));
    agent
        .context_mut()
        .push(aither_core::llm::Message::assistant("y".repeat(4_000)));
    agent.context_mut().mark_prefix_sent();

    let (result, _) = run_to_end(&mut agent, "short").await;
    result.expect("pause-point compaction succeeds");

    let sent = format!("{:?}", main_llm.requests()[0].messages());
    assert_eq!(
        sent.matches("unique-source-alert").count(),
        1,
        "source stable alert must appear once across pause-point make_room: {sent}"
    );
    assert_eq!(
        sent.matches("unique-hook-alert").count(),
        1,
        "before_request stable alert must appear once across pause-point make_room: {sent}"
    );
}

struct PendingBeforeRequestHook {
    entered: async_channel::Sender<()>,
}

impl Hooks for PendingBeforeRequestHook {
    async fn before_request(&self, _cx: &crate::RequestCx<'_>) -> Vec<ContextBlock> {
        let _ = self.entered.send(()).await;
        std::future::pending::<Vec<ContextBlock>>().await
    }
}

#[tokio::test]
async fn cancelling_preview_request_messages_with_leaves_context_and_seq_untouched() {
    let (entered_tx, entered_rx) = async_channel::bounded(1);
    let mut agent = AgentBuilder::new(Scripted::new([say("ok")]))
        .hooks(PendingBeforeRequestHook {
            entered: entered_tx,
        })
        .build();
    agent
        .context_mut()
        .push(aither_core::llm::Message::user("initial"));
    let seq_before = agent.context().fork_point();
    let len_before = agent.context().len_recent();

    {
        let preview_fut = agent.preview_request_messages_with(
            "preview prompt",
            [aither_core::llm::Message::user("extra hidden")],
            std::iter::empty(),
        );
        futures_lite::pin!(preview_fut);
        futures_lite::future::or(
            async {
                let _ = (&mut preview_fut).await;
            },
            async {
                entered_rx.recv().await.expect("reached before_request");
            },
        )
        .await;
    }

    assert_eq!(agent.context().len_recent(), len_before);
    assert_eq!(agent.context().fork_point(), seq_before);
    assert_eq!(agent.history().len(), 1);
    assert_eq!(agent.history()[0].content(), "initial");
}

struct BarrierSource {
    name: &'static str,
    tx: async_channel::Sender<()>,
    rx: async_channel::Receiver<()>,
    committed_order: Arc<Mutex<Vec<&'static str>>>,
}

impl ContextSource for BarrierSource {
    fn id(&self) -> SourceId {
        SourceId::new(self.name)
    }

    async fn blocks(&mut self, moment: Moment) -> Vec<ContextBlock> {
        self.peek(moment).await
    }

    async fn peek(&self, moment: Moment) -> Vec<ContextBlock> {
        self.tx.send(()).await.expect("barrier send");
        self.rx.recv().await.expect("barrier recv");
        match moment {
            Moment::Request { .. } => vec![ContextBlock::tail(self.name, self.name)],
            _ => Vec::new(),
        }
    }

    async fn committed(&mut self, _tail: &[Block]) {
        self.tx.send(()).await.expect("commit barrier send");
        self.rx.recv().await.expect("commit barrier recv");
        self.committed_order.lock().unwrap().push(self.name);
    }
}

#[tokio::test]
async fn context_sources_execute_concurrently_and_preserve_registration_order() {
    let (a_to_b_tx, a_to_b_rx) = async_channel::bounded(2);
    let (b_to_a_tx, b_to_a_rx) = async_channel::bounded(2);
    let committed_order = Arc::new(Mutex::new(Vec::new()));

    let llm = Scripted::new([say("done")]);
    let mut agent = AgentBuilder::new(llm.clone())
        .context_source(BarrierSource {
            name: "first-src",
            tx: a_to_b_tx,
            rx: b_to_a_rx,
            committed_order: Arc::clone(&committed_order),
        })
        .context_source(BarrierSource {
            name: "second-src",
            tx: b_to_a_tx,
            rx: a_to_b_rx,
            committed_order: Arc::clone(&committed_order),
        })
        .build();

    let preview = agent
        .preview_request_messages("check", std::iter::empty())
        .await
        .expect("concurrent peek succeeds without deadlocking");
    let preview_tail = preview.last().expect("tail message").content();
    let first_pos = preview_tail.find("first-src").expect("first-src present");
    let second_pos = preview_tail.find("second-src").expect("second-src present");
    assert!(
        first_pos < second_pos,
        "registration order preserved: {preview_tail}"
    );

    let (result, _) = run_to_end(&mut agent, "check").await;
    result.expect("concurrent blocks and committed succeed without deadlocking");
    assert_eq!(committed_order.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn manual_clock_utc_now_and_live_cache_setters_take_effect_immediately() {
    use aither_core::llm::model::ClaudePromptCacheTtl;

    let start = jiff::Timestamp::from_second(1_800_000_000).expect("valid timestamp");
    let clock = crate::ManualClock::at_utc(start);
    let llm = ClaudeScripted::new(128_000, [say("one"), say("two")]);
    let mut agent = AgentBuilder::new(llm.clone())
        .clock(clock.clone())
        .cache_lifetime(crate::context::CacheLifetime::Limited(Duration::from_secs(
            600,
        )))
        .claude_cache_ttl(ClaudePromptCacheTtl::OneHour)
        .instruction("system")
        .build();
    let warmth = agent.cache_warmth();

    let (first, events) = run_to_end(&mut agent, "hello").await;
    first.expect("first run");
    assert!(events.iter().all(|timed| timed.at == start));
    assert!(
        agent
            .context()
            .checkpoint()
            .items()
            .iter()
            .all(|item| item.committed_at() == start)
    );
    assert_eq!(warmth.cold_in(), Some(Duration::from_secs(600)));

    // Live updates take effect immediately on existing `CacheWarmth` clones and next request.
    agent.set_cache_lifetime(crate::context::CacheLifetime::Limited(Duration::from_secs(
        120,
    )));
    assert_eq!(warmth.cold_in(), Some(Duration::from_secs(120)));
    agent.set_claude_cache_ttl(ClaudePromptCacheTtl::FiveMinutes);
    assert_eq!(
        agent.cache_lifetime(),
        crate::context::CacheLifetime::Limited(Duration::from_secs(120))
    );
    assert_eq!(agent.claude_cache_ttl(), ClaudePromptCacheTtl::FiveMinutes);

    clock.advance(Duration::from_secs(30));
    let expected_second =
        jiff::Timestamp::from_second(1_800_000_030).expect("valid advanced timestamp");
    let (second, second_events) = run_to_end(&mut agent, "again").await;
    second.expect("second run");
    assert!(
        second_events
            .iter()
            .all(|timed| timed.at == expected_second)
    );
    assert_eq!(
        llm.requests()[1].parameters().cache.claude.map(|c| c.ttl),
        Some(ClaudePromptCacheTtl::FiveMinutes)
    );
}

#[tokio::test]
async fn a_compaction_on_a_compaction_model_carries_no_session_cache_controls() {
    let handoff_json = r#"{"current_task":"t","key_decisions":[],"context_files":[],"errors_fixes":[],"pending_work":["reply"],"recovery_hints":[],"skills":[]}"#;

    // A compaction on the session's own model reads the warm cache: its
    // request keeps the controls.
    let same = ClaudeScripted::new(128_000, [say("hi"), say(handoff_json)]);
    let mut agent = AgentBuilder::new(same.clone()).build();
    let (first, _) = run_to_end(&mut agent, "hello").await;
    first.expect("first run");
    let _ = agent
        .compact(None)
        .drive((), std::future::pending::<()>(), |(), _timed| async move {
            ControlFlow::Continue(())
        })
        .await
        .expect("same-model compaction");
    assert!(
        !same
            .requests()
            .last()
            .expect("compaction request")
            .parameters()
            .cache
            .is_empty(),
        "a compaction on the session model keeps the cache controls"
    );

    // One on a configured compaction model reads none of that cache, so its
    // request writes no entry for the prefix being discarded.
    let compactor = ClaudeScripted::new(128_000, [say(handoff_json)]);
    let main = ClaudeScripted::new(128_000, [say("hi")]);
    let mut agent = AgentBuilder::new(main)
        .compaction_model(
            compactor.clone(),
            aither_core::llm::model::Parameters::default(),
        )
        .build();
    let (first, _) = run_to_end(&mut agent, "hello").await;
    first.expect("first run");
    let _ = agent
        .compact(None)
        .drive((), std::future::pending::<()>(), |(), _timed| async move {
            ControlFlow::Continue(())
        })
        .await
        .expect("compaction-model compaction");
    let request = compactor
        .requests()
        .last()
        .expect("compaction request")
        .clone();
    assert!(
        request.parameters().cache.is_empty(),
        "a compaction model's request carries no session cache controls"
    );
}
