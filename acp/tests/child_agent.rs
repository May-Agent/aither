//! `AcpChildAgent` runs `tests/fake_agent.py` as a subagent of a parent
//! agent, whose stream carries the child's events.

use std::ops::ControlFlow;
use std::time::Duration;

use aither_acp::{
    AcpChildAgent, AcpClient, ClientHandler, RequestPermissionOutcome, RequestPermissionParams,
    RequestPermissionResult, SessionNewParams, SessionNotification, UpdateSink,
};
use aither_agent::{
    Agent, AgentEvent, AgentPath, OutputKind, RunOutcome, SubagentRuns, TimedEvent, ToolVerdict,
};
use aither_core::LanguageModel;
use aither_core::llm::{
    AgentIdentity, CallIdentity, Classify, Event, Failure, LLMRequest, ToolCallId, model::Profile,
};
use aither_mcp::protocol::JsonRpcError;
use futures_lite::{Stream, stream};

/// Pushes the session's updates to the child and approves every permission.
struct Forwarding(UpdateSink);

impl ClientHandler for Forwarding {
    fn session_update(
        &self,
        notification: SessionNotification,
    ) -> impl std::future::Future<Output = ()> + Send {
        self.0.push(notification);
        std::future::ready(())
    }

    fn request_permission(
        &self,
        _: RequestPermissionParams,
    ) -> impl std::future::Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send
    {
        std::future::ready(Ok(RequestPermissionResult {
            outcome: RequestPermissionOutcome::Selected {
                option_id: "allow-1".to_string(),
            },
            meta: None,
        }))
    }
}

#[derive(Debug)]
struct Never;

impl std::fmt::Display for Never {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("never")
    }
}

impl std::error::Error for Never {}

impl Classify for Never {
    fn failure(&self) -> Failure {
        Failure::permanent()
    }
}

/// A parent model that only answers.
#[derive(Clone)]
struct Parent;

impl LanguageModel for Parent {
    type Error = Never;

    fn respond(&self, _request: LLMRequest) -> impl Stream<Item = Result<Event, Never>> + Send {
        stream::once(Ok(Event::Text("ok".to_string())))
    }

    fn profile(&self) -> impl std::future::Future<Output = Result<Profile, Never>> + Send {
        std::future::ready(Ok(Profile::new("p", "test", "p", "Parent", 128_000)))
    }
}

/// Runs the fake agent as the child `researcher` on `prompt`, cancelling it
/// after `cancel_after` if given, and returns its outcome and the events the
/// parent's stream carried.
async fn run_child(
    prompt: &str,
    cancel_after: Option<Duration>,
) -> Option<(RunOutcome, AgentPath, Vec<TimedEvent>)> {
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fake_agent.py");
    let (sink, source) = UpdateSink::channel();
    let Ok((client, connection)) =
        AcpClient::spawn("python3", &[script], [], "/tmp", Forwarding(sink))
    else {
        eprintln!("skipping: cannot spawn python3");
        return None;
    };
    let connection = tokio::spawn(connection);
    client.initialize().await.expect("initialize");
    let session = client
        .new_session(SessionNewParams::new("/tmp"))
        .await
        .expect("new session");

    let runs = SubagentRuns::default();
    let started_by = CallIdentity::new(ToolCallId::new("c1"), AgentIdentity::Main);
    let run = runs.start(&started_by, "researcher", prompt);
    let path = run.path().clone();
    let mut child = AcpChildAgent::new(client, session.session_id, &run.identity(), source);

    let outcome = match cancel_after {
        None => run.run(&mut child, prompt).await,
        Some(delay) => {
            let id = run.id();
            let canceller = runs.clone();
            let cancel = async move {
                tokio::time::sleep(delay).await;
                assert!(canceller.cancel(id));
            };
            tokio::time::timeout(Duration::from_secs(10), async {
                futures_util::future::join(run.run(&mut child, prompt), cancel)
                    .await
                    .0
            })
            .await
            .expect("a cancelled child must end")
        }
    };
    connection.abort();

    let mut parent = Agent::builder(Parent).subagents(runs).build();
    let mut events = Vec::new();
    let _ = parent
        .run("go", [])
        .drive(
            &mut events,
            std::future::pending::<()>(),
            |events, timed| async move {
                events.push(timed);
                ControlFlow::Continue(events)
            },
        )
        .await
        .expect("the parent completes");
    Some((outcome, path, events))
}

#[tokio::test]
async fn an_acp_session_streams_its_events_through_the_parent() {
    let Some((outcome, path, events)) = run_child("go", None).await else {
        return;
    };

    assert!(
        matches!(&outcome, RunOutcome::Completed { final_text } if final_text == "Hello abworld"),
        "{outcome:?}"
    );
    let child: Vec<_> = events
        .iter()
        .filter(|timed| timed.agent == path)
        .map(|timed| &timed.event)
        .collect();
    assert!(matches!(child.first(), Some(AgentEvent::RunStarted { .. })));
    assert!(matches!(
        child.last(),
        Some(AgentEvent::RunEnded {
            outcome: RunOutcome::Completed { .. },
            ..
        })
    ));
    let text: String = child
        .iter()
        .filter_map(|event| match event {
            AgentEvent::BlockDelta {
                kind: OutputKind::Text,
                text,
                ..
            } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello abworld");
    assert!(child.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallStarted { call, name, .. } if call.as_str() == "tc-1" && name == "fake-tool"
    )));
    assert!(child.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallEnded { call, .. } if call.as_str() == "tc-1"
    )));
    assert!(child.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallDecided {
            decision: ToolVerdict::Allowed,
            ..
        }
    )));
    // The parent brackets the child.
    assert!(events.iter().any(|timed| matches!(
        &timed.event,
        AgentEvent::SubagentEnded { child, outcome: RunOutcome::Completed { .. }, .. } if *child == path
    )));
}

#[tokio::test]
async fn the_parents_cancel_handle_cancels_the_session() {
    let Some((outcome, path, events)) = run_child("wait", Some(Duration::from_millis(500))).await
    else {
        return;
    };

    assert!(matches!(outcome, RunOutcome::Cancelled), "{outcome:?}");
    assert!(events.iter().any(|timed| timed.agent == path
        && matches!(
            timed.event,
            AgentEvent::RunEnded {
                outcome: RunOutcome::Cancelled,
                ..
            }
        )));
}
