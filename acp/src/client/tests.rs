//! Client unit tests: init detection, demux/reducer, turn ending.

use std::collections::BTreeMap;

use serde_json::json;

use crate::domain::{
    ConfigValueId, ContentBlock, CurrentModeUpdate, ForegroundState, MessageId, Open,
    ReplayComplete, SessionId, SessionUpdate, StopReason, update::StateUpdate,
};
use crate::wire::codec::{Protocol, detect_initialize};

use super::handle::SessionRegistry;
use crate::state::{MessageKind, SessionEvent, SessionReducer};

#[test]
fn detect_v2_and_v1_shapes() {
    assert_eq!(
        detect_initialize(&json!({"protocolVersion": 2, "capabilities": {}})),
        Protocol::V2
    );
    assert_eq!(
        detect_initialize(&json!({"protocolVersion": 1, "agentCapabilities": {}})),
        Protocol::V1
    );
}

fn idle_update(stop: Option<Open<StopReason>>) -> SessionUpdate {
    SessionUpdate::StateUpdate(StateUpdate {
        state: Open::Known(ForegroundState::Idle),
        stop_reason: stop,
        error: None,
        background: Vec::new(),
        meta: None,
        extra: BTreeMap::new(),
    })
}

fn foreground_update() -> SessionUpdate {
    SessionUpdate::StateUpdate(StateUpdate {
        state: Open::Known(ForegroundState::Running),
        stop_reason: None,
        error: None,
        background: Vec::new(),
        meta: None,
        extra: BTreeMap::new(),
    })
}

#[test]
fn reducer_turn_end_only_on_idle_transition() {
    let mut r = SessionReducer::new();
    // A bare `idle` while none was ever seen is informational, not a
    // turn end.
    let events = r.apply(&idle_update(None));
    assert!(
        events
            .iter()
            .all(|e| !matches!(e, SessionEvent::TurnEnded { .. }))
    );
    // Running -> idle: TurnEnded, carrying the update's stop reason.
    r.apply(&foreground_update());
    let events = r.apply(&idle_update(Some(Open::Known(StopReason::EndTurn))));
    assert!(events.iter().any(|e| matches!(
        e,
        SessionEvent::TurnEnded {
            stop_reason: Some(Open::Known(StopReason::EndTurn)),
            ..
        }
    )));
    // A second *bare* idle while already idle: no duplicate TurnEnded.
    let events = r.apply(&idle_update(None));
    assert!(
        events
            .iter()
            .all(|e| !matches!(e, SessionEvent::TurnEnded { .. }))
    );
}

#[test]
fn reducer_chunk_creates_message_and_mode_update() {
    let mut r = SessionReducer::new();
    let chunk = SessionUpdate::AgentMessageChunk(crate::domain::ContentChunk::new(
        "m1",
        ContentBlock::text("hi"),
    ));
    let events = r.apply(&chunk);
    assert!(matches!(
        events.first(),
        Some(SessionEvent::MessageDelta {
            kind: MessageKind::Agent,
            text: Some(t),
            ..
        }) if t == "hi"
    ));
    assert!(
        r.state()
            .messages
            .contains_key(&(MessageKind::Agent, MessageId::from("m1")))
    );

    let events = r.apply(&SessionUpdate::CurrentModeUpdate(CurrentModeUpdate {
        current_mode_id: ConfigValueId::from("code"),
        meta: None,
        extra: BTreeMap::new(),
    }));
    assert!(matches!(
        events.first(),
        Some(SessionEvent::ModeChanged(v)) if v.as_str() == "code"
    ));
    assert_eq!(
        r.state().current_mode.as_ref().map(ConfigValueId::as_str),
        Some("code")
    );
}

#[test]
fn demux_routes_updates_to_the_right_cell() {
    let registry = SessionRegistry::new();
    let s1 = SessionId::from("s1");
    let s2 = SessionId::from("s2");
    let c1 = registry.get_or_create(s1.clone());
    let rx2 = registry.get_or_create(s2).subscribe();

    registry.apply(&crate::domain::SessionNotification {
        session_id: s1,
        update: idle_update(Some(Open::Known(StopReason::EndTurn))),
        meta: None,
        extra: BTreeMap::new(),
    });

    // s1's cell saw the update; s2's subscriber did not.
    assert!(c1.reducer.lock().unwrap().state().foreground.is_some());
    assert!(rx2.is_empty());
}

#[tokio::test]
async fn subscribe_snapshot_then_live_events() {
    let registry = SessionRegistry::new();
    let cell = registry.get_or_create(SessionId::from("s"));
    cell.apply(&foreground_update());
    let (tx, rx) = async_channel::unbounded::<SessionEvent>();
    drop(tx);
    let sub = cell.subscribe();
    cell.mark_replay_complete();
    assert!(cell.state().replay_done);
    let event = sub.recv().await.expect("event");
    assert!(matches!(event, SessionEvent::ReplayComplete));
    drop(rx);
}

#[test]
fn v1_turn_end_update_synthesizes_idle() {
    let finished = crate::domain::PromptFinished {
        stop_reason: Open::Known(StopReason::EndTurn),
        usage: None,
        meta: None,
    };
    let update = SessionUpdate::StateUpdate(super::connection::v1_turn_end_update(&finished));
    let mut r = SessionReducer::new();
    r.apply(&foreground_update());
    let events = r.apply(&update);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SessionEvent::TurnEnded { .. }))
    );
}

#[test]
fn replay_complete_marks_state() {
    let mut r = SessionReducer::new();
    r.begin_replay();
    let events = r.apply(&SessionUpdate::ReplayComplete(ReplayComplete {
        meta: None,
        extra: BTreeMap::new(),
    }));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SessionEvent::ReplayComplete))
    );
    assert!(r.state().replay_done);
}

/// The in-process direct lane end to end: a `SessionHost` served by
/// `Client::in_process` — initialize, `session/new`, `session/prompt`,
/// the `running`/`idle` transition, and a finished turn outcome — all
/// without a transport or a serialized message.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn in_process_initialize_session_prompt_idle() {
    use crate::domain::SessionSetupParams;
    use crate::server::{
        PendingPrompt, PromptError, ServerError, Session, SessionContext, SessionHost,
    };

    /// A session that echoes the prompt back and ends the turn.
    struct Echo;
    impl Session for Echo {
        async fn prompt(
            &self,
            pending: PendingPrompt,
            _cx: &SessionContext,
        ) -> Result<StopReason, PromptError> {
            let text = pending
                .content()
                .iter()
                .filter_map(ContentBlock::as_text)
                .collect::<String>();
            let turn = pending
                .accept()
                .await
                .map_err(|rejected| PromptError::from(rejected.into_inner()))?;
            turn.message()
                .text(format!("echo: {text}"))
                .await
                .map_err(|_| PromptError::failed("echo failed"))?;
            Ok(StopReason::EndTurn)
        }
    }

    struct EchoHost;
    impl SessionHost for EchoHost {
        type Session = Echo;
        fn new_session(
            &self,
            _params: SessionSetupParams,
            _cx: &SessionContext,
        ) -> impl std::future::Future<Output = Result<Echo, ServerError>> + Send {
            std::future::ready(Ok(Echo))
        }
    }

    let (client, conn) =
        crate::client::Client::builder(crate::domain::Implementation::new("test-client", "0.0.0"))
            .in_process(EchoHost)
            .await
            .expect("in_process initialize");
    // The drive future must be polled for the turn scheduler to run.
    tokio::spawn(conn);

    assert_eq!(client.protocol_version(), 2);
    let session = client
        .new_session("/tmp")
        .open()
        .await
        .expect("session/new");
    let outcome = session.prompt_run("hi").await.expect("prompt");
    let crate::client::TurnOutcome::Finished {
        stop_reason, text, ..
    } = outcome
    else {
        panic!("expected a finished turn, got {outcome:?}");
    };
    assert_eq!(stop_reason, Some(Open::Known(StopReason::EndTurn)));
    assert_eq!(text, "echo: hi");
    assert!(matches!(
        session.state().foreground,
        Some(Open::Known(ForegroundState::Idle))
    ));
    assert!(
        session
            .state()
            .messages
            .values()
            .any(|m| m.content.as_option().is_some_and(|c| !c.is_empty()))
    );
}
