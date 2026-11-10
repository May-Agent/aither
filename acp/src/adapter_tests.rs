//! The mapper's behaviour on the events a real agent run emits.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use aither_agent::Agent;
use aither_core::llm::{
    Classify, Event, Failure, LLMRequest, LanguageModel, RetryPolicy, ToolCall, model::Profile,
};
use futures_lite::Stream;

use super::*;
use aither_core::llm::tool::Progress;

#[derive(Debug)]
struct Broken;

impl std::fmt::Display for Broken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("stream broke")
    }
}

impl std::error::Error for Broken {}

impl Classify for Broken {
    fn failure(&self) -> Failure {
        Failure::network()
    }
}

type Response = Vec<Result<Event, Broken>>;

/// Answers each request with the next scripted response.
#[derive(Clone)]
struct Scripted(Arc<Mutex<VecDeque<Response>>>);

impl Scripted {
    fn new(script: impl IntoIterator<Item = Response>) -> Self {
        Self(Arc::new(Mutex::new(script.into_iter().collect())))
    }
}

impl LanguageModel for Scripted {
    type Error = Broken;

    fn respond(&self, _request: LLMRequest) -> impl Stream<Item = Result<Event, Broken>> + Send {
        let response = self.0.lock().unwrap().pop_front().expect("scripted");
        futures_lite::stream::iter(response)
    }

    fn retry_policy(&self) -> RetryPolicy {
        RetryPolicy {
            max_retries: 1,
            initial_delay: std::time::Duration::ZERO,
            max_delay: std::time::Duration::ZERO,
            multiplier: 2,
        }
    }

    fn profile(&self) -> impl Future<Output = Result<Profile, Broken>> + Send {
        std::future::ready(Ok(Profile::new(
            "scripted",
            "test",
            "scripted",
            "Scripted model",
            128_000,
        )))
    }
}

/// Runs `script` through an agent and maps every event of the main agent.
async fn updates(script: Vec<Response>) -> Vec<SessionUpdate> {
    let mut agent = Agent::builder(Scripted::new(script)).build();
    let mut mapper = SessionUpdateMapper::default();
    let mut collected = Vec::new();
    let _ = agent
        .run("go", [])
        .drive(
            (&mut mapper, &mut collected),
            std::future::pending::<()>(),
            |(mapper, collected), timed| async move {
                collected.extend(mapper.map(&timed));
                std::ops::ControlFlow::Continue((mapper, collected))
            },
        )
        .await
        .expect("the run completes");
    collected
}

fn call(arguments: serde_json::Value) -> Event {
    Event::ToolCall(ToolCall::new("call-1", "terminal", arguments))
}

fn delta(fragment: &str) -> Event {
    Event::ToolCallDelta {
        id: "call-1".to_string(),
        name: "terminal".to_string(),
        arguments_fragment: fragment.to_string(),
    }
}

fn text_of(update: &SessionUpdate) -> &str {
    let (SessionUpdate::AgentMessageChunk(chunk) | SessionUpdate::AgentThoughtChunk(chunk)) =
        update
    else {
        panic!("expected a message or thought chunk, got {update:?}");
    };
    let ContentBlock::Text(text) = &chunk.content else {
        panic!("expected text content");
    };
    &text.text
}

#[tokio::test]
async fn text_and_reasoning_become_message_and_thought_chunks() {
    let updates = updates(vec![vec![
        Ok(Event::Reasoning("hmm".to_string())),
        Ok(Event::Text("Hello".to_string())),
    ]])
    .await;

    assert!(matches!(updates[0], SessionUpdate::AgentThoughtChunk(_)));
    assert_eq!(text_of(&updates[0]), "hmm");
    assert!(matches!(updates[1], SessionUpdate::AgentMessageChunk(_)));
    assert_eq!(text_of(&updates[1]), "Hello");
}

#[tokio::test]
async fn a_tool_call_is_announced_and_then_completed() {
    let updates = updates(vec![
        vec![Ok(call(serde_json::json!({
            "description": "List files",
            "script": "ls -la",
        })))],
        vec![Ok(Event::Text("done".to_string()))],
    ])
    .await;

    let SessionUpdate::ToolCall(started) = &updates[0] else {
        panic!("expected ToolCall, got {:?}", updates[0]);
    };
    assert_eq!(started.tool_call_id, "call-1");
    assert_eq!(started.title, "List files");
    assert_eq!(started.kind, Some(ToolKind::Execute));
    assert_eq!(started.status, Some(ToolCallStatus::Pending));
    let SessionUpdate::ToolCallUpdate(decided) = &updates[1] else {
        panic!("expected ToolCallUpdate, got {:?}", updates[1]);
    };
    assert_eq!(decided.status, Some(ToolCallStatus::InProgress));
    let SessionUpdate::ToolCallUpdate(ended) = &updates[2] else {
        panic!("expected ToolCallUpdate, got {:?}", updates[2]);
    };
    assert_eq!(ended.tool_call_id, "call-1");
    // The terminal tool is not registered, so the call ends in an error.
    assert_eq!(ended.status, Some(ToolCallStatus::Failed));
}

/// `ToolCallDelta` fragments are incremental; the mapper accumulates them so
/// every `ToolCallUpdate` still carries the arguments so far.
#[tokio::test]
async fn tool_call_deltas_accumulate() {
    let updates = updates(vec![
        vec![
            Ok(delta(r#"{"description""#)),
            Ok(delta(r#":"List files"}"#)),
            Ok(call(serde_json::json!({"description": "List files"}))),
        ],
        vec![Ok(Event::Text("done".to_string()))],
    ])
    .await;

    let SessionUpdate::ToolCallUpdate(last_delta) = &updates[1] else {
        panic!("expected ToolCallUpdate, got {:?}", updates[1]);
    };
    assert_eq!(
        last_delta.raw_input,
        Some(serde_json::Value::String(
            r#"{"description":"List files"}"#.to_string()
        ))
    );
    assert_eq!(last_delta.title.as_deref(), Some("List files"));
}

#[tokio::test]
async fn a_response_retry_drops_the_broken_attempts_arguments() {
    let updates = updates(vec![
        vec![Ok(delta(r#"{"description":"bro"#)), Err(Broken)],
        vec![
            Ok(delta(r#"{"description":"whole"}"#)),
            Ok(Event::Text("done".to_string())),
        ],
    ])
    .await;

    let SessionUpdate::ToolCallUpdate(replacement) = &updates[1] else {
        panic!("expected ToolCallUpdate, got {:?}", updates[1]);
    };
    assert_eq!(
        replacement.raw_input,
        Some(serde_json::Value::String(
            r#"{"description":"whole"}"#.to_string()
        ))
    );
}

/// Arguments without a usable description fall back to a generic title.
#[test]
fn tool_title_fallbacks() {
    assert_eq!(
        format_tool_title(&AgentPath::main(), "terminal", "{}"),
        "Running terminal"
    );
    assert_eq!(
        format_tool_title(
            &AgentPath::main(),
            "terminal",
            r#"{"description":"","script":"true"}"#
        ),
        "Running terminal"
    );
    assert_eq!(
        format_tool_title(&AgentPath::main(), "terminal", "not json"),
        "Running terminal"
    );
    assert_eq!(infer_tool_kind("unknown_tool"), ToolKind::Other);
}

/// Titles are cut on a char boundary: a multi-byte description longer than
/// the limit must not panic on byte slicing.
#[test]
fn tool_title_truncates_on_char_boundary() {
    let description = "あ".repeat(TOOL_TITLE_MAX_CHARS + 10);
    let arguments = serde_json::json!({"description": description}).to_string();
    let title = format_tool_title(&AgentPath::main(), "terminal", &arguments);
    assert_eq!(title, format!("{}...", "あ".repeat(TOOL_TITLE_MAX_CHARS)));

    let boundary_case = serde_json::json!({
        "description": format!("{:46}é{}", "x", "y".repeat(20)),
    })
    .to_string();
    let title = format_tool_title(&AgentPath::main(), "terminal", &boundary_case);
    assert!(title.contains('é'));
    assert_eq!(title.chars().count(), TOOL_TITLE_MAX_CHARS + 3);
    assert!(title.ends_with("..."));
}

fn timed(agent: &AgentPath, event: AgentEvent) -> TimedEvent {
    TimedEvent::now(agent.clone(), event)
}

fn started(id: &str) -> AgentEvent {
    AgentEvent::tool_call_started(
        aither_core::llm::ToolCallId::new(id),
        "terminal".to_string(),
        serde_json::json!({"description": "List files"}),
        ToolOrigin::Local,
    )
}

fn tool_update(update: Option<SessionUpdate>) -> ToolCallUpdate {
    let Some(SessionUpdate::ToolCallUpdate(update)) = update else {
        panic!("expected ToolCallUpdate, got {update:?}");
    };
    update
}

#[test]
fn a_decision_progress_and_end_update_the_same_call() {
    let main = AgentPath::main();
    let id = || aither_core::llm::ToolCallId::new("call-1");
    let mut mapper = SessionUpdateMapper::default();
    let _ = mapper.map(&timed(&main, started("call-1")));

    let allowed = tool_update(mapper.map(&timed(
        &main,
        AgentEvent::tool_call_decided(id(), ToolVerdict::Allowed),
    )));
    assert_eq!(allowed.tool_call_id, "call-1");
    assert_eq!(allowed.status, Some(ToolCallStatus::InProgress));

    let progress = tool_update(mapper.map(&timed(
        &main,
        AgentEvent::tool_progress(id(), Progress::new(1.0).with_message("compiling")),
    )));
    assert_eq!(progress.status, Some(ToolCallStatus::InProgress));
    let Some(content) = progress.content else {
        panic!("progress carries content");
    };
    assert!(matches!(
        &content[..],
        [ToolCallContent::Content { content: ContentBlock::Text(text) }] if text.text == "compiling"
    ));

    let denied = tool_update(mapper.map(&timed(
        &main,
        AgentEvent::tool_call_decided(
            id(),
            ToolVerdict::Denied {
                reason: "not allowed".to_string(),
            },
        ),
    )));
    assert_eq!(denied.status, Some(ToolCallStatus::Failed));

    assert!(
        mapper
            .map(&timed(
                &main,
                AgentEvent::tool_call_decided(id(), ToolVerdict::AwaitingApproval)
            ))
            .is_none()
    );
}

#[test]
fn a_childs_calls_are_namespaced_and_its_text_is_dropped() {
    let main = AgentPath::main();
    let child = main.child("researcher");
    let mut mapper = SessionUpdateMapper::default();

    let Some(SessionUpdate::ToolCall(call)) = mapper.map(&timed(&child, started("call-1"))) else {
        panic!("a child's tool call is announced");
    };
    assert_eq!(call.tool_call_id, "researcher:call-1");
    assert_eq!(call.title, "[researcher] List files");
    // The parent's call of the same id is a different call.
    let Some(SessionUpdate::ToolCall(parent_call)) = mapper.map(&timed(&main, started("call-1")))
    else {
        panic!("the parent's tool call is announced");
    };
    assert_eq!(parent_call.tool_call_id, "call-1");

    let text = AgentEvent::block_delta(
        aither_agent::RequestId::new(1),
        aither_agent::BlockId::new(0),
        OutputKind::Text,
        "hello".to_string(),
    );
    assert!(mapper.map(&timed(&child, text.clone())).is_none());
    assert!(mapper.map(&timed(&main, text)).is_some());
}

#[tokio::test]
async fn a_subagent_is_a_tool_call_from_start_to_end() {
    // The registry announces a child on the parent's stream.
    let runs = aither_agent::SubagentRuns::default();
    let started_by = aither_core::llm::CallIdentity::new(
        aither_core::llm::ToolCallId::new("c1"),
        aither_core::llm::AgentIdentity::Main,
    );
    runs.start(&started_by, "researcher", "look")
        .finish(RunOutcome::Cancelled);
    let mut parent = Agent::builder(Scripted::new([vec![Ok(Event::Text("hi".to_string()))]]))
        .subagents(runs)
        .build();
    let mut mapper = SessionUpdateMapper::default();
    let mut collected = Vec::new();
    let _ = parent
        .run("go", [])
        .drive(
            (&mut mapper, &mut collected),
            std::future::pending::<()>(),
            |(mapper, collected), timed| async move {
                collected.extend(mapper.map(&timed));
                std::ops::ControlFlow::Continue((mapper, collected))
            },
        )
        .await
        .expect("the run completes");

    let Some(SessionUpdate::ToolCall(call)) = collected
        .iter()
        .find(|update| matches!(update, SessionUpdate::ToolCall(_)))
    else {
        panic!("a subagent start is a tool call: {collected:?}");
    };
    assert_eq!(call.tool_call_id, "subagent:researcher");
    assert_eq!(call.status, Some(ToolCallStatus::InProgress));
    let ended = collected
        .iter()
        .find_map(|update| match update {
            SessionUpdate::ToolCallUpdate(update)
                if update.tool_call_id == "subagent:researcher" =>
            {
                Some(update)
            }
            _ => None,
        })
        .expect("the end is reported");
    assert_eq!(ended.status, Some(ToolCallStatus::Failed));
}
