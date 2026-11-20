//! Adapter for converting aither agent events to ACP session updates.

use std::collections::HashMap;

use aither_agent::{
    AgentEvent, AgentPath, OutputKind, RequestOutcome, RunOutcome, TimedEvent, ToolOrigin,
    ToolVerdict,
};
use aither_core::llm::ToolResult;
use aither_core::llm::tool::{Progress, ToolDefinition};

use crate::protocol::{
    ContentBlock, ContentChunk, SessionUpdate, TextContent, ToolCall, ToolCallContent,
    ToolCallStatus, ToolCallUpdate, ToolKind,
};

/// Optional display metadata describing how a tool's calls map onto ACP
/// [`ToolCall`] updates.
///
/// Extracted from a [`ToolDefinition`]'s JSON schema extension keys
/// (`x-acp-kind` / `x-tool-kind`, `x-acp-title` / `x-tool-title`, and
/// `x-acp-title-field` / `x-title-field`) or configured directly on
/// [`SessionUpdateMapper`]. When unset, the mapper falls back to
/// [`ToolKind::Other`] and `"Running <name>"` without assuming product-specific
/// tool names or argument fields.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolMetadata {
    kind: Option<ToolKind>,
    title: Option<String>,
    title_field: Option<String>,
}

impl ToolMetadata {
    /// Empty metadata (uses generic fallbacks).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            kind: None,
            title: None,
            title_field: None,
        }
    }

    /// Sets the ACP [`ToolKind`] reported for calls to this tool.
    #[must_use]
    pub const fn with_kind(mut self, kind: ToolKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Sets a static human-readable title fallback for calls to this tool.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        let title = title.into();
        if !title.is_empty() {
            self.title = Some(title);
        }
        self
    }

    /// Names the JSON object argument property whose non-empty string value
    /// titles a call to this tool.
    #[must_use]
    pub fn with_title_field(mut self, field: impl Into<String>) -> Self {
        let field = field.into();
        if !field.is_empty() {
            self.title_field = Some(field);
        }
        self
    }

    /// Reads optional ACP display metadata from a [`ToolDefinition`]'s raw
    /// argument schema (`x-acp-kind` / `x-tool-kind`, `x-acp-title` /
    /// `x-tool-title`, and `x-acp-title-field` / `x-title-field`).
    #[must_use]
    pub fn from_definition(definition: &ToolDefinition) -> Self {
        let Ok(serialized) = serde_json::to_value(definition) else {
            return Self::new();
        };
        let Some(schema) = serialized
            .get("arguments")
            .and_then(serde_json::Value::as_object)
        else {
            return Self::new();
        };

        let kind = schema
            .get("x-acp-kind")
            .or_else(|| schema.get("x-tool-kind"))
            .and_then(|val| serde_json::from_value::<ToolKind>(val.clone()).ok());

        let title = schema
            .get("x-acp-title")
            .or_else(|| schema.get("x-tool-title"))
            .and_then(serde_json::Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        let title_field = schema
            .get("x-acp-title-field")
            .or_else(|| schema.get("x-title-field"))
            .and_then(serde_json::Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        Self {
            kind,
            title,
            title_field,
        }
    }
}

/// Converts a stream of [`TimedEvent`]s into ACP [`SessionUpdate`]s.
#[derive(Debug, Default)]
pub struct SessionUpdateMapper {
    /// Arguments accumulated per tool call, keyed by the agent and call id.
    tool_arguments: HashMap<(AgentPath, String), String>,
    /// Optional per-tool display metadata keyed by tool name.
    tool_metadata: HashMap<String, ToolMetadata>,
}

impl SessionUpdateMapper {
    /// Builds a mapper pre-populated with optional display metadata extracted
    /// from `definitions`.
    #[must_use]
    pub fn from_definitions<'a>(definitions: impl IntoIterator<Item = &'a ToolDefinition>) -> Self {
        Self::default().with_definitions(definitions)
    }

    /// Registers display metadata extracted from each [`ToolDefinition`] in
    /// `definitions`.
    #[must_use]
    pub fn with_definitions<'a>(
        mut self,
        definitions: impl IntoIterator<Item = &'a ToolDefinition>,
    ) -> Self {
        for def in definitions {
            let meta = ToolMetadata::from_definition(def);
            if meta != ToolMetadata::default() {
                self.tool_metadata.insert(def.name().to_string(), meta);
            }
        }
        self
    }

    /// Attaches explicit [`ToolMetadata`] for `name`.
    #[must_use]
    pub fn with_tool_metadata(mut self, name: impl Into<String>, metadata: ToolMetadata) -> Self {
        self.register_tool_metadata(name, metadata);
        self
    }

    /// Inserts or replaces [`ToolMetadata`] for `name`.
    pub fn register_tool_metadata(&mut self, name: impl Into<String>, metadata: ToolMetadata) {
        self.tool_metadata.insert(name.into(), metadata);
    }

    fn kind_for(&self, name: &str) -> ToolKind {
        self.tool_metadata
            .get(name)
            .and_then(|meta| meta.kind)
            .unwrap_or(ToolKind::Other)
    }

    fn title_for(&self, agent: &AgentPath, name: &str, arguments: &str) -> String {
        let meta = self.tool_metadata.get(name);
        format_tool_title(agent, name, arguments, meta)
    }

    fn map_tool_started(
        &mut self,
        agent: &AgentPath,
        call: &str,
        name: &str,
        arguments: &serde_json::Value,
        origin: ToolOrigin,
    ) -> SessionUpdate {
        let (title_arguments, raw_input) = match origin {
            ToolOrigin::Hosted => (arguments.as_str().unwrap_or_default().to_string(), None),
            _ => (arguments.to_string(), Some(arguments.clone())),
        };
        self.tool_arguments
            .insert((agent.clone(), call.to_string()), title_arguments.clone());
        let title = match origin {
            ToolOrigin::Hosted if !title_arguments.is_empty() => {
                qualify_agent_title(agent, truncate_title(&title_arguments))
            }
            _ => self.title_for(agent, name, &title_arguments),
        };
        tool_call_pending(call_id(agent, call), title, self.kind_for(name), raw_input)
    }

    /// Convert a `TimedEvent` to an ACP `SessionUpdate`.
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
            } => Some(self.map_tool_started(agent, call.as_str(), name, arguments, *origin)),

            AgentEvent::ToolCallDelta {
                call,
                name,
                arguments_fragment,
                ..
            } => {
                let arguments_snapshot = {
                    let entry = self
                        .tool_arguments
                        .entry((agent.clone(), call.as_str().to_string()))
                        .or_default();
                    entry.push_str(arguments_fragment);
                    entry.clone()
                };
                Some(SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    status: Some(ToolCallStatus::Pending),
                    title: Some(self.title_for(agent, name, &arguments_snapshot)),
                    kind: Some(self.kind_for(name)),
                    raw_input: Some(serde_json::Value::String(arguments_snapshot)),
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
                    content: Some(vec![text_content(result.render_for_model())]),
                    raw_output: Some(tool_result_raw_output(result)),
                    ..update(call_id(agent, call.as_str()))
                }))
            }

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

fn decision_update(
    verdict: &ToolVerdict,
) -> Option<(ToolCallStatus, Option<Vec<ToolCallContent>>)> {
    match verdict {
        ToolVerdict::Allowed | ToolVerdict::Rewritten => Some((ToolCallStatus::InProgress, None)),
        ToolVerdict::Denied { reason } => Some((
            ToolCallStatus::Failed,
            Some(vec![text_content(reason.clone())]),
        )),
        _ => None,
    }
}

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

const fn text_content(text: String) -> ToolCallContent {
    ToolCallContent::Content {
        content: ContentBlock::Text(TextContent {
            text,
            annotations: None,
            meta: None,
        }),
    }
}

fn progress_text(progress: &Progress) -> String {
    if let Some(message) = progress.message() {
        return message.to_string();
    }
    progress.total().map_or_else(
        || progress.progress().to_string(),
        |total| format!("{} of {total}", progress.progress()),
    )
}

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
        Some((byte_index, _)) => text
            .get(..byte_index)
            .map_or_else(|| text.to_string(), |prefix| format!("{prefix}...")),
        None => text.to_string(),
    }
}

fn qualify_agent_title(agent: &AgentPath, title: String) -> String {
    if agent.is_main() {
        title
    } else {
        format!("[{agent}] {title}")
    }
}

/// Format a human-readable title for a tool call from optional [`ToolMetadata`],
/// falling back gracefully when no metadata is configured.
fn format_tool_title(
    agent: &AgentPath,
    name: &str,
    arguments: &str,
    metadata: Option<&ToolMetadata>,
) -> String {
    let dynamic_title = metadata
        .and_then(|meta| meta.title_field.as_deref())
        .and_then(|field| match serde_json::from_str(arguments) {
            Ok(serde_json::Value::Object(map)) => map
                .get(field)
                .and_then(serde_json::Value::as_str)
                .filter(|val| !val.is_empty())
                .map(truncate_title),
            _ => None,
        });

    let title = dynamic_title
        .or_else(|| metadata.and_then(|meta| meta.title.clone()))
        .unwrap_or_else(|| format!("Running {name}"));

    qualify_agent_title(agent, title)
}

#[cfg(test)]
#[path = "adapter_tests.rs"]
mod tests;
