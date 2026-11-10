//! Adapter for converting aither agent events to ACP session updates.

use aither_agent::{AgentEvent, OutputKind, RequestOutcome, ToolOrigin};
use aither_core::llm::ToolResult;

use crate::protocol::{
    ContentBlock, ContentChunk, SessionUpdate, TextContent, ToolCall, ToolCallStatus,
    ToolCallUpdate, ToolKind,
};

/// Converts a stream of [`AgentEvent`]s into ACP [`SessionUpdate`]s.
///
/// `AgentEvent::ToolCallDelta` carries incremental argument fragments; the
/// mapper accumulates them per call so each emitted `ToolCallUpdate` reports
/// the arguments received so far.
#[derive(Debug, Default)]
pub struct SessionUpdateMapper {
    /// Arguments accumulated per tool call, keyed by call id.
    tool_arguments: std::collections::HashMap<String, String>,
}

impl SessionUpdateMapper {
    /// Convert an `AgentEvent` to an ACP `SessionUpdate`.
    ///
    /// Returns `None` for events that don't have a direct ACP mapping
    /// (like the run and turn brackets, which are handled separately).
    #[must_use]
    pub fn map(&mut self, event: &AgentEvent) -> Option<SessionUpdate> {
        match event {
            AgentEvent::BlockDelta { kind, text, .. } => {
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
                let id = call.as_str().to_string();
                // A hosted call reports its detail as a string, not JSON
                // arguments.
                let (title_arguments, raw_input) = match origin {
                    ToolOrigin::Hosted => {
                        (arguments.as_str().unwrap_or_default().to_string(), None)
                    }
                    _ => (arguments.to_string(), Some(arguments.clone())),
                };
                self.tool_arguments
                    .insert(id.clone(), title_arguments.clone());
                Some(SessionUpdate::ToolCall(ToolCall {
                    tool_call_id: id,
                    title: format_tool_title(name, &title_arguments),
                    kind: Some(infer_tool_kind(name)),
                    status: Some(ToolCallStatus::Pending),
                    content: vec![],
                    locations: vec![],
                    raw_input,
                    raw_output: None,
                    meta: None,
                }))
            }

            AgentEvent::ToolCallDelta {
                call,
                name,
                arguments_fragment,
                ..
            } => {
                let id = call.as_str().to_string();
                let arguments = self.tool_arguments.entry(id.clone()).or_default();
                arguments.push_str(arguments_fragment);
                let arguments = arguments.as_str();
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    tool_call_id: id,
                    status: Some(ToolCallStatus::Pending),
                    content: None,
                    title: Some(format_tool_title(name, arguments)),
                    kind: Some(infer_tool_kind(name)),
                    locations: None,
                    raw_input: Some(serde_json::Value::String(arguments.to_string())),
                    raw_output: None,
                    meta: None,
                }))
            }

            AgentEvent::ToolCallEnded { call, result, .. } => {
                let id = call.as_str();
                self.tool_arguments.remove(id);
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    tool_call_id: id.to_string(),
                    status: Some(tool_result_status(result)),
                    content: None,
                    title: None,
                    kind: None,
                    locations: None,
                    raw_input: None,
                    raw_output: Some(tool_result_raw_output(result)),
                    meta: None,
                }))
            }

            // The broken attempt's calls never run and the replacement
            // streams its own, so their accumulated arguments are dropped.
            // ACP has no update that withdraws streamed message chunks.
            AgentEvent::RequestEnded {
                outcome: RequestOutcome::Retrying { .. },
                ..
            } => {
                self.tool_arguments.clear();
                None
            }

            // Everything else — run and turn brackets, requests, decisions,
            // pauses, compaction, context injection, subagents — has no ACP
            // update here.
            _ => None,
        }
    }
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
/// Every other call gets the generic title.
fn format_tool_title(name: &str, arguments: &str) -> String {
    if let Ok(serde_json::Value::Object(arguments)) = serde_json::from_str(arguments)
        && let Some(description) = arguments
            .get("description")
            .and_then(serde_json::Value::as_str)
            .filter(|description| !description.is_empty())
    {
        return truncate_title(description);
    }
    format!("Running {name}")
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
