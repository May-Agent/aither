//! Adapter for converting aither agent events to ACP session updates.

use std::collections::HashMap;

use aither_agent::{
    AgentEvent, AgentPath, OutputKind, RequestOutcome, RunOutcome, TimedEvent, ToolOrigin,
    ToolVerdict,
};
use aither_core::llm::ToolResult;
use aither_core::llm::tool::Progress;

use crate::protocol::{
    ContentBlock, ContentChunk, SessionUpdate, TextContent, ToolCall, ToolCallContent,
    ToolCallStatus, ToolCallUpdate, ToolKind,
};

/// Converts a stream of [`TimedEvent`]s into ACP [`SessionUpdate`]s.
///
/// The mapping, event by event:
///
/// | Event | ACP update |
/// | --- | --- |
/// | `BlockDelta` | `agent_message_chunk`, or `agent_thought_chunk` for reasoning |
/// | `ToolCallStarted` | `tool_call`, pending; a hosted call's detail is its title |
/// | `ToolCallDelta` | `tool_call_update` carrying the arguments received so far |
/// | `ToolCallDecided` | `tool_call_update`: in progress when the call runs, failed with the reason when denied |
/// | `ToolProgress` | `tool_call_update`, in progress, with the report as content |
/// | `ToolCallEnded` | `tool_call_update`, completed or failed, with the result as raw output |
/// | `SubagentStarted` | `tool_call` standing for the child |
/// | `SubagentEnded` | `tool_call_update` completing or failing it |
///
/// These events have no ACP update: `RunStarted`, `RunEnded`, `TurnStarted`,
/// `TurnEnded`, `RequestStarted`, `BlockEnded` (ACP chunks have no end),
/// `Paused` and `Resumed` (ACP's permission request is the client's own
/// channel), `CompactionStarted`, `CompactionEnded`, `CompactionRecovery`,
/// `ContextInjected` and `Checkpoint`. `RequestEnded` has none either, but a
/// retrying one clears the retried request's tool arguments; ACP cannot
/// withdraw chunks already sent, which is a protocol limitation.
///
/// Events are mapped by the agent that emitted them. A child's tool calls
/// are reported under ids and titles that name its path, so they never
/// collide with the parent's; its message and reasoning chunks are dropped,
/// since ACP would show them as part of the parent's reply.
///
/// `AgentEvent::ToolCallDelta` carries incremental argument fragments; the
/// mapper accumulates them per call so each emitted `ToolCallUpdate` reports
/// the arguments received so far.
#[derive(Debug, Default)]
pub struct SessionUpdateMapper {
    /// Arguments accumulated per tool call, keyed by the agent and call id.
    tool_arguments: HashMap<(AgentPath, String), String>,
}

impl SessionUpdateMapper {
    /// Convert a `TimedEvent` to an ACP `SessionUpdate`.
    ///
    /// Returns `None` for events that have no ACP mapping; see the type's
    /// documentation for which.
    #[must_use]
    pub fn map(&mut self, timed: &TimedEvent) -> Option<SessionUpdate> {
        let agent = &timed.agent;
        match &timed.event {
            AgentEvent::BlockDelta { kind, text, .. } => {
                if !agent.is_main() {
                    return None;
                }
                let chunk = text_chunk(text.clone());
                Some(match kind {
                    OutputKind::Reasoning => SessionUpdate::AgentThoughtChunk(chunk),
                    _ => SessionUpdate::AgentMessageChunk(chunk),
                })
            }

            AgentEvent::ToolCallStarted {
                call,
                name,
                arguments,
                origin,
                ..
            } => {
                // A hosted call reports its detail as a string, not JSON
                // arguments.
                let (title_arguments, raw_input) = match origin {
                    ToolOrigin::Hosted => {
                        (arguments.as_str().unwrap_or_default().to_string(), None)
                    }
                    _ => (arguments.to_string(), Some(arguments.clone())),
                };
                self.tool_arguments.insert(
                    (agent.clone(), call.as_str().to_string()),
                    title_arguments.clone(),
                );
                Some(tool_call_pending(
                    call_id(agent, call.as_str()),
                    format_tool_title(agent, name, &title_arguments),
                    infer_tool_kind(name),
                    raw_input,
                ))
            }

            AgentEvent::ToolCallDelta {
                call,
                name,
                arguments_fragment,
                ..
            } => {
                let arguments = self
                    .tool_arguments
                    .entry((agent.clone(), call.as_str().to_string()))
                    .or_default();
                arguments.push_str(arguments_fragment);
                let arguments = arguments.as_str();
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    status: Some(ToolCallStatus::Pending),
                    title: Some(format_tool_title(agent, name, arguments)),
                    kind: Some(infer_tool_kind(name)),
                    raw_input: Some(serde_json::Value::String(arguments.to_string())),
                    ..update(call_id(agent, call.as_str()))
                }))
            }

            AgentEvent::ToolCallDecided { call, decision, .. } => {
                let (status, content) = decision_update(decision)?;
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    status: Some(status),
                    content,
                    ..update(call_id(agent, call.as_str()))
                }))
            }

            AgentEvent::ToolProgress { call, progress, .. } => {
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    status: Some(ToolCallStatus::InProgress),
                    content: Some(vec![text_content(progress_text(progress))]),
                    ..update(call_id(agent, call.as_str()))
                }))
            }

            AgentEvent::ToolCallEnded { call, result, .. } => {
                self.tool_arguments
                    .remove(&(agent.clone(), call.as_str().to_string()));
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    status: Some(tool_result_status(result)),
                    raw_output: Some(tool_result_raw_output(result)),
                    ..update(call_id(agent, call.as_str()))
                }))
            }

            // The broken attempt's calls never run and the replacement
            // streams its own, so the arguments this agent accumulated are
            // dropped. ACP has no update that withdraws streamed message
            // chunks.
            AgentEvent::RequestEnded {
                outcome: RequestOutcome::Retrying { .. },
                ..
            } => {
                self.tool_arguments.retain(|(owner, _), _| owner != agent);
                None
            }

            AgentEvent::SubagentStarted {
                child,
                name,
                prompt,
                ..
            } => Some(subagent_started(child, name, prompt)),

            AgentEvent::SubagentEnded { child, outcome, .. } => {
                Some(subagent_ended(child, outcome))
            }

            _ => None,
        }
    }
}

/// A tool call that has been made and not yet decided.
const fn tool_call_pending(
    tool_call_id: String,
    title: String,
    kind: ToolKind,
    raw_input: Option<serde_json::Value>,
) -> SessionUpdate {
    SessionUpdate::ToolCall(ToolCall {
        tool_call_id,
        title,
        kind: Some(kind),
        status: Some(ToolCallStatus::Pending),
        content: vec![],
        locations: vec![],
        raw_input,
        raw_output: None,
        meta: None,
    })
}

/// A subagent's start, as the tool call that stands for it.
fn subagent_started(child: &AgentPath, name: &str, prompt: &str) -> SessionUpdate {
    SessionUpdate::ToolCall(ToolCall {
        tool_call_id: subagent_call_id(child),
        title: format!("Subagent {name}"),
        kind: Some(ToolKind::Other),
        status: Some(ToolCallStatus::InProgress),
        content: vec![],
        locations: vec![],
        raw_input: Some(serde_json::Value::String(prompt.to_string())),
        raw_output: None,
        meta: None,
    })
}

/// A subagent's end, as the update that completes or fails its tool call.
fn subagent_ended(child: &AgentPath, outcome: &RunOutcome) -> SessionUpdate {
    let status = match outcome {
        RunOutcome::Completed { .. } => ToolCallStatus::Completed,
        _ => ToolCallStatus::Failed,
    };
    SessionUpdate::ToolCallUpdate(ToolCallUpdate {
        status: Some(status),
        ..update(subagent_call_id(child))
    })
}

/// What a decision about a call changes in its tool call; `None` while it
/// stays pending.
fn decision_update(
    verdict: &ToolVerdict,
) -> Option<(ToolCallStatus, Option<Vec<ToolCallContent>>)> {
    match verdict {
        ToolVerdict::Allowed | ToolVerdict::Rewritten => Some((ToolCallStatus::InProgress, None)),
        ToolVerdict::Denied { reason } => Some((
            ToolCallStatus::Failed,
            Some(vec![text_content(reason.clone())]),
        )),
        // The call stays pending until the answer arrives as another
        // decision.
        _ => None,
    }
}

/// An update to `tool_call_id` that changes nothing yet.
const fn update(tool_call_id: String) -> ToolCallUpdate {
    ToolCallUpdate {
        tool_call_id,
        status: None,
        content: None,
        title: None,
        kind: None,
        locations: None,
        raw_input: None,
        raw_output: None,
        meta: None,
    }
}

/// The id a call is reported under: the provider's own for the main agent,
/// qualified by the agent's path for a child, whose provider may reuse it.
fn call_id(agent: &AgentPath, call: &str) -> String {
    if agent.is_main() {
        call.to_string()
    } else {
        format!("{agent}:{call}")
    }
}

fn subagent_call_id(child: &AgentPath) -> String {
    format!("subagent:{child}")
}

/// A text block as tool-call content.
const fn text_content(text: String) -> ToolCallContent {
    ToolCallContent::Content {
        content: ContentBlock::Text(TextContent {
            text,
            annotations: None,
            meta: None,
        }),
    }
}

/// What a progress report says, for a reader.
fn progress_text(progress: &Progress) -> String {
    if let Some(message) = progress.message() {
        return message.to_string();
    }
    progress.total().map_or_else(
        || progress.progress().to_string(),
        |total| format!("{} of {total}", progress.progress()),
    )
}

/// Wraps plain text in the content-chunk shape ACP expects.
const fn text_chunk(text: String) -> ContentChunk {
    ContentChunk {
        content: ContentBlock::Text(TextContent {
            text,
            annotations: None,
            meta: None,
        }),
        message_id: None,
        meta: None,
    }
}

const fn tool_result_status(result: &ToolResult) -> ToolCallStatus {
    if result.is_error() {
        ToolCallStatus::Failed
    } else {
        ToolCallStatus::Completed
    }
}

fn tool_result_raw_output(result: &ToolResult) -> serde_json::Value {
    match serde_json::to_value(result) {
        Ok(value) => value,
        Err(error) => {
            panic!("failed to serialize ToolResult for ACP raw_output: {error}");
        }
    }
}

/// Maximum characters shown in a tool-call title before it is truncated.
const TOOL_TITLE_MAX_CHARS: usize = 50;

/// Truncate `text` to [`TOOL_TITLE_MAX_CHARS`] characters on a char boundary,
/// appending an ellipsis when anything was cut.
fn truncate_title(text: &str) -> String {
    match text.char_indices().nth(TOOL_TITLE_MAX_CHARS) {
        Some((byte_index, _)) => format!("{}...", &text[..byte_index]),
        None => text.to_string(),
    }
}

/// Format a human-readable title for a tool call.
///
/// A call whose arguments carry a non-empty `description` is titled by it:
/// that is the UI-facing explanation a model writes for what it runs.
/// Every other call gets the generic title. A child's calls are prefixed
/// with the child's path.
fn format_tool_title(agent: &AgentPath, name: &str, arguments: &str) -> String {
    let title = match serde_json::from_str(arguments) {
        Ok(serde_json::Value::Object(arguments)) => arguments
            .get("description")
            .and_then(serde_json::Value::as_str)
            .filter(|description| !description.is_empty())
            .map_or_else(|| format!("Running {name}"), truncate_title),
        _ => format!("Running {name}"),
    };
    if agent.is_main() {
        title
    } else {
        format!("[{agent}] {title}")
    }
}

/// Infer the tool kind from the tool name.
fn infer_tool_kind(name: &str) -> ToolKind {
    match name {
        "terminal" | "terminal_kill" | "terminal_input" => ToolKind::Execute,
        "terminal_read" => ToolKind::Read,
        _ => ToolKind::Other,
    }
}

#[cfg(test)]
#[path = "adapter_tests.rs"]
mod tests;
