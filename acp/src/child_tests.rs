//! How the updates of an external session become events.

use super::*;

fn message(text: &str) -> SessionUpdate {
    SessionUpdate::AgentMessageChunk(chunk(text))
}

fn thought(text: &str) -> SessionUpdate {
    SessionUpdate::AgentThoughtChunk(chunk(text))
}

fn chunk(text: &str) -> crate::protocol::ContentChunk {
    crate::protocol::ContentChunk {
        content: ContentBlock::Text(TextContent {
            text: text.to_string(),
            annotations: None,
            meta: None,
        }),
        message_id: None,
        meta: None,
    }
}

fn call(id: &str, status: ToolCallStatus) -> SessionUpdate {
    SessionUpdate::ToolCall(ToolCall {
        tool_call_id: id.to_string(),
        title: "fake-tool".to_string(),
        kind: None,
        status: Some(status),
        content: vec![],
        locations: vec![],
        raw_input: Some(serde_json::json!({"x": 1})),
        raw_output: None,
        meta: None,
    })
}

fn call_update(id: &str, status: ToolCallStatus, text: Option<&str>) -> SessionUpdate {
    SessionUpdate::ToolCallUpdate(ToolCallUpdate {
        tool_call_id: id.to_string(),
        status: Some(status),
        content: text.map(|text| {
            vec![ToolCallContent::Content {
                content: ContentBlock::Text(TextContent {
                    text: text.to_string(),
                    annotations: None,
                    meta: None,
                }),
            }]
        }),
        title: None,
        kind: None,
        locations: None,
        raw_input: None,
        raw_output: None,
        meta: None,
    })
}

#[test]
fn chunks_of_one_kind_share_a_block_that_closes_when_the_kind_changes() {
    let mut translator = Translator::new();
    let mut events = Vec::new();
    for update in [
        thought("hm"),
        thought("m"),
        message("Hello"),
        message(" there"),
    ] {
        events.extend(translator.translate(update));
    }
    events.extend(translator.close_block());

    let shape: Vec<_> = events
        .iter()
        .map(|event| match event {
            AgentEvent::BlockDelta {
                block, kind, text, ..
            } => format!("delta {} {kind:?} {text}", block.get()),
            AgentEvent::BlockEnded { block, kind, .. } => {
                format!("end {} {kind:?}", block.get())
            }
            other => panic!("unexpected {other:?}"),
        })
        .collect();
    assert_eq!(
        shape,
        [
            "delta 0 Reasoning hm",
            "delta 0 Reasoning m",
            "end 0 Reasoning",
            "delta 1 Text Hello",
            "delta 1 Text  there",
            "end 1 Text",
        ]
    );
    assert_eq!(translator.reply, "Hello there");
}

#[test]
fn a_tool_call_closes_the_open_block_and_runs_to_its_end() {
    let mut translator = Translator::new();
    let mut events = translator.translate(message("Let me look"));
    events.extend(translator.translate(call("tc-1", ToolCallStatus::Pending)));
    events.extend(translator.translate(call_update("tc-1", ToolCallStatus::InProgress, None)));
    events.extend(translator.translate(call_update("tc-1", ToolCallStatus::InProgress, None)));
    events.extend(translator.translate(call_update(
        "tc-1",
        ToolCallStatus::Completed,
        Some("done"),
    )));
    // A second completion of the same call is no new event.
    events.extend(translator.translate(call_update("tc-1", ToolCallStatus::Completed, None)));

    assert!(matches!(events[0], AgentEvent::BlockDelta { .. }));
    assert!(matches!(events[1], AgentEvent::BlockEnded { .. }));
    let AgentEvent::ToolCallStarted {
        call,
        name,
        arguments,
        origin,
        ..
    } = &events[2]
    else {
        panic!("expected ToolCallStarted, got {:?}", events[2]);
    };
    assert_eq!(call.as_str(), "tc-1");
    assert_eq!(name, "fake-tool");
    assert_eq!(arguments, &serde_json::json!({"x": 1}));
    assert_eq!(*origin, aither_agent::ToolOrigin::Local);
    assert!(matches!(
        &events[3],
        AgentEvent::ToolCallDecided {
            decision: ToolVerdict::Allowed,
            ..
        }
    ));
    let AgentEvent::ToolCallEnded { result, .. } = &events[4] else {
        panic!("expected ToolCallEnded, got {:?}", events[4]);
    };
    assert_eq!(result, &ToolResult::text("done"));
    assert_eq!(events.len(), 5);
}

#[test]
fn a_failed_call_ends_as_an_error() {
    let mut translator = Translator::new();
    translator.translate(call("tc-1", ToolCallStatus::Pending));
    let events = translator.translate(call_update("tc-1", ToolCallStatus::Failed, Some("boom")));

    let AgentEvent::ToolCallEnded { result, .. } = events.last().expect("an end") else {
        panic!("expected ToolCallEnded");
    };
    assert_eq!(result, &ToolResult::error("boom"));
}

#[test]
fn an_update_for_a_call_never_announced_is_dropped() {
    let mut translator = Translator::new();
    assert!(
        translator
            .translate(call_update("ghost", ToolCallStatus::Completed, None))
            .is_empty()
    );
}

#[test]
fn updates_without_an_event_are_dropped() {
    let mut translator = Translator::new();
    assert!(
        translator
            .translate(SessionUpdate::Other(serde_json::json!({})))
            .is_empty()
    );
}

#[test]
fn stop_reasons_map_to_outcomes() {
    assert!(matches!(
        outcome(StopReason::EndTurn, "hi".to_string()),
        RunOutcome::Completed { final_text } if final_text == "hi"
    ));
    assert!(matches!(
        outcome(StopReason::Cancelled, String::new()),
        RunOutcome::Cancelled
    ));
    assert!(matches!(
        outcome(StopReason::MaxTurnRequests, String::new()),
        RunOutcome::IterationLimit
    ));
    assert!(matches!(
        outcome(StopReason::Refusal, String::new()),
        RunOutcome::Failed { .. }
    ));
}
