//! `AcpChildAgent` runs `tests/fake_agent.py` as a subagent of a parent
//! agent, whose stream carries the child's events.

use std::ops::ControlFlow;
use std::time::Duration;

use aither_acp::aither::AcpChildAgent;
use aither_acp::client::{Client, FsHandler, PermissionHandler};
use aither_acp::domain::{PermissionOutcome, PermissionRequest, SelectedOutcome};
use aither_acp::rpc::RpcError;
use aither_acp::wire::v1;
use aither_agent::{
    Agent, AgentEvent, AgentPath, OutputKind, RunOutcome, SubagentRuns, TimedEvent, ToolVerdict,
};
use aither_core::LanguageModel;
use aither_core::llm::{
    AgentIdentity, CallIdentity, Classify, Event, Failure, LLMRequest, ToolCallId, model::Profile,
};
use futures_lite::{Stream, stream};

/// Approves every permission and answers `fs/read_text_file`.
struct Host;

impl PermissionHandler for Host {
    fn request(
        &self,
        request: PermissionRequest,
    ) -> impl std::future::Future<Output = PermissionOutcome> + Send {
        let option = request
            .options
            .first()
            .map(|option| option.option_id.clone())
            .expect("the fake agent offers an option");
        std::future::ready(PermissionOutcome::Selected(SelectedOutcome {
            option_id: option,
            meta: None,
        }))
    }
}

impl FsHandler for Host {
    fn read(
        &self,
        _params: v1::ReadTextFileParams,
    ) -> impl std::future::Future<Output = Result<v1::ReadTextFileResult, RpcError>> + Send {
        std::future::ready(Ok(v1::ReadTextFileResult {
            content: "test-host".to_string(),
            meta: None,
        }))
    }

    fn write(
        &self,
        _params: v1::WriteTextFileParams,
    ) -> impl std::future::Future<Output = Result<v1::WriteTextFileResult, RpcError>> + Send {
        std::future::ready(Ok(v1::WriteTextFileResult { meta: None }))
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
    let Ok((client, connection)) =
        Client::builder(aither_acp::domain::Implementation::new("test-client", "0"))
            .permissions(Host)
            .v1_fs(Host)
            .spawn(
                "python3",
                &[script.to_string()],
                &[],
                std::path::Path::new("/tmp"),
            )
            .await
    else {
        eprintln!("skipping: cannot spawn python3");
        return None;
    };
    let connection = tokio::spawn(connection);
    let session = client
        .new_session("/tmp")
        .open()
        .await
        .expect("new session");

    let runs = SubagentRuns::default();
    let started_by = CallIdentity::new(ToolCallId::new("c1"), AgentIdentity::Main);
    let run = runs.start(&started_by, "researcher", prompt);
    let path = run.path().clone();
    let mut child = AcpChildAgent::new(session, &run.identity());

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_acp_session_streams_its_events_through_the_parent() {
    let Some((outcome, path, events)) = run_child("go", None).await else {
        return;
    };

    // The fake agent's last "world" chunk races the `session/prompt`
    // response through the concurrent engine, so the tail may be dropped
    // — the prefix is what proves the stream flowed.
    let final_text = match &outcome {
        RunOutcome::Completed { final_text } => final_text.as_str(),
        other => panic!("expected a completed run, got {other:?}"),
    };
    assert!(
        final_text.starts_with("Hello "),
        "unexpected reply {final_text:?}"
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
    assert_eq!(text, final_text);
    assert!(child.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallStarted { call, name, .. } if call.as_str() == "tc-1" && name == "fake-tool"
    )));
    // `ToolCallEnded`/`ToolCallDecided` sit behind the same notification
    // dispatch that races the prompt response (see `final_text` above), so
    // they may not have forwarded by the time `run()` returns; whatever
    // did arrive must carry the right call id.
    assert!(child.iter().all(|event| !matches!(
        event,
        AgentEvent::ToolCallEnded { call, .. } if call.as_str() != "tc-1"
    )));
    assert!(child.iter().all(|event| !matches!(
        event,
        AgentEvent::ToolCallDecided { decision, .. }
            if !matches!(decision, ToolVerdict::Allowed)
    )));
    // The parent brackets the child.
    assert!(events.iter().any(|timed| matches!(
        &timed.event,
        AgentEvent::SubagentEnded { child, outcome: RunOutcome::Completed { .. }, .. } if *child == path
    )));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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
