//! Tool call updates, content, and diffs.
//!
//! [`ToolCallUpdate`] is the canonical v2 patch shape: every field besides
//! `tool_call_id` is a [`Patch`]. [`ToolCall`] is the materialized
//! snapshot type the reducer/client work against.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::content::ContentBlock;
use super::ids::{TerminalId, ToolCallId};
use super::open::{ExtIdent, Open, open_tags, open_union_serde};
use super::patch::{EmptyObject, Patch};

/// Categories of tool that can be invoked (bare string on the wire).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// Reading files or data.
    Read,
    /// Modifying files or content.
    Edit,
    /// Removing files or data.
    Delete,
    /// Moving or renaming files.
    Move,
    /// Searching for information.
    Search,
    /// Running commands or code.
    Execute,
    /// Internal reasoning or planning.
    Think,
    /// Retrieving external data.
    Fetch,
    /// Switching the current session mode.
    SwitchMode,
    /// Other tool types (`"other"`).
    #[default]
    Other,
}
open_tags!(ToolKind; None => ["read", "edit", "delete", "move", "search", "execute", "think", "fetch", "switch_mode", "other"]);

/// Execution status of a tool call (bare string on the wire).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallStatus {
    /// Not started yet.
    #[default]
    Pending,
    /// Currently running.
    InProgress,
    /// Paused awaiting something that is not user-facing.
    Paused,
    /// Completed successfully.
    Completed,
    /// Failed with an error.
    Failed,
    /// Cancelled before completion.
    Cancelled,
}
open_tags!(ToolCallStatus; None => ["pending", "in_progress", "paused", "completed", "failed", "cancelled"]);

/// A tool-call patch (`{"toolCallId", <patch fields>}`).
///
/// Fields are omitted to leave stored state, `null` to clear, concrete to
/// replace.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallUpdate {
    /// Unique tool call id within the session.
    pub tool_call_id: ToolCallId,
    /// Programmatic tool name.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub name: Patch<String>,
    /// Human-readable title.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub title: Patch<String>,
    /// Tool category.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub kind: Patch<Open<ToolKind>>,
    /// Execution status.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub status: Patch<Open<ToolCallStatus>>,
    /// Produced content (replaces, does not append — see
    /// [`ToolCallContentChunk`]).
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub content: Patch<Vec<ToolCallContent>>,
    /// Affected file locations.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub locations: Patch<Vec<Location>>,
    /// Raw tool input.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub raw_input: Patch<Value>,
    /// Raw tool output.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub raw_output: Patch<Value>,
    /// Per-call capabilities.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub capabilities: Patch<ToolCallCaps>,
    /// Extension metadata; `null` clears.
    #[serde(rename = "_meta", default, skip_serializing_if = "Patch::is_unchanged")]
    pub meta: Patch<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ToolCallUpdate {
    /// A bare upsert for `tool_call_id`.
    #[must_use]
    pub fn new(tool_call_id: impl Into<ToolCallId>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            ..Self::default()
        }
    }
}

/// A materialized snapshot of a tool call (no patch fields).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    /// Unique tool call id within the session.
    pub tool_call_id: ToolCallId,
    /// Programmatic tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Human-readable title.
    #[serde(default)]
    pub title: String,
    /// Tool category.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<Open<ToolKind>>,
    /// Execution status.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<Open<ToolCallStatus>>,
    /// Produced content.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<ToolCallContent>,
    /// Affected file locations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locations: Vec<Location>,
    /// Raw tool input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_input: Option<Value>,
    /// Raw tool output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<Value>,
    /// Per-call capabilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<ToolCallCaps>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Per-tool-call capabilities (`{"cancel": {}}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallCaps {
    /// `session/tool_call_cancel` is honoured when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancel: Option<EmptyObject>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A file location touched by a tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Location {
    /// Absolute file path.
    pub path: PathBuf,
    /// Optional line number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// An item of tool-call content (`{"type": ...}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallContent {
    /// Standard content block (`"type": "content"`).
    Content(Content),
    /// File modification shown as a diff (`"type": "diff"`).
    Diff(Diff),
    /// A display-only reference to an agent-owned terminal
    /// (`"type": "terminal"`).
    Terminal(TerminalRef),
    /// A `_`-prefixed extension payload, preserved verbatim.
    Custom(ExtIdent, Value),
    /// An unrecognized content type, preserved verbatim.
    Unknown(String, Value),
}

open_union_serde!(ToolCallContent, "type", [
    Content => "content",
    Diff => "diff",
    Terminal => "terminal",
]);

/// `{"type": "content", "content": ContentBlock}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Content {
    /// The content block.
    pub content: ContentBlock,
    /// Item-scoped extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `{"type": "terminal", "terminalId": ...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalRef {
    /// The terminal to display.
    pub terminal_id: TerminalId,
    /// Item-scoped extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// File changes produced by a tool call
/// (`{"type": "diff", "changes": [...], "patch": {...}?}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Diff {
    /// Structured file-level changes.
    pub changes: Vec<DiffChange>,
    /// Renderable patch text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch: Option<DiffPatch>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Renderable patch text (`{"format": "git_patch", "text": ...}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiffPatch {
    /// Patch format.
    pub format: Open<DiffPatchFormat>,
    /// Patch text.
    pub text: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Text patch format (bare string).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffPatchFormat {
    /// One or more `diff --git` sections.
    GitPatch,
}
open_tags!(DiffPatchFormat; None => ["git_patch"]);

/// Kind of file content in a diff change (bare string).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffFileType {
    /// Text content.
    Text,
    /// Binary or otherwise non-text content.
    Binary,
    /// Directory entry.
    Directory,
    /// Symbolic link.
    Symlink,
}
open_tags!(DiffFileType; None => ["text", "binary", "directory", "symlink"]);

/// One file-level change: `{"operation", path fields..., "fileType"?,
/// "mimeType"?}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiffChange {
    /// The file operation.
    #[serde(flatten)]
    pub operation: Open<DiffOperation>,
    /// File content kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_type: Option<Open<DiffFileType>>,
    /// MIME type of the file contents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The file operation inside a [`DiffChange`] (tagged `operation`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum DiffOperation {
    /// A file was added (`{"operation":"add","path"}`).
    Add(DiffPathChange),
    /// A file was deleted.
    Delete(DiffPathChange),
    /// A file was modified in place.
    Modify(DiffPathChange),
    /// A file was moved or renamed.
    Move(DiffPathPairChange),
    /// A file was copied.
    Copy(DiffPathPairChange),
}
open_tags!(DiffOperation; Some("operation") => ["add", "delete", "modify", "move", "copy"]);

/// Path payload for add/delete/modify (`{"path"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiffPathChange {
    /// Absolute path for the operation.
    pub path: PathBuf,
}

/// Path payload for move/copy (`{"oldPath","path"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiffPathPairChange {
    /// Absolute path before the operation.
    pub old_path: PathBuf,
    /// Absolute path after the operation.
    pub path: PathBuf,
}

/// A content chunk appended to a tool call
/// (`{"toolCallId", "content"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallContentChunk {
    /// The tool call receiving the content.
    pub tool_call_id: ToolCallId,
    /// One content item to append.
    pub content: ToolCallContent,
    /// Chunk-scoped extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn update_patch_fields_wire() {
        let raw = json!({"toolCallId": "t1", "title": null, "status": "in_progress", "capabilities": {"cancel": {}}});
        let u: ToolCallUpdate = serde_json::from_value(raw).expect("de");
        assert_eq!(u.title, Patch::Clear);
        assert!(matches!(
            u.status,
            Patch::Set(Open::Known(ToolCallStatus::InProgress))
        ));
        assert!(matches!(u.capabilities, Patch::Set(ref c) if c.cancel.is_some()));
        assert!(u.content.is_unchanged());
        let back = serde_json::to_value(&u).expect("ser");
        assert_eq!(back["title"], serde_json::Value::Null);
        assert_eq!(back["status"], "in_progress");
        assert!(back.get("content").is_none());
    }

    #[test]
    fn tool_call_content_open_union() {
        let raw = json!({"type": "diff", "changes": [{"operation": "modify", "path": "/a"}], "patch": {"format": "git_patch", "text": "@@"}});
        let c: ToolCallContent = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(c, ToolCallContent::Diff(_)));
        assert_eq!(serde_json::to_value(&c).expect("ser"), raw);

        let raw = json!({"type": "_graph", "n": 1});
        let c: ToolCallContent = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(c, ToolCallContent::Custom(_, _)));
        assert_eq!(serde_json::to_value(&c).expect("ser"), raw);
    }

    #[test]
    fn open_string_enums() {
        let k: Open<ToolKind> = serde_json::from_value(json!("switch_mode")).expect("de");
        assert_eq!(k, Open::Known(ToolKind::SwitchMode));
        let k: Open<ToolKind> = serde_json::from_value(json!("teleport")).expect("de");
        assert!(matches!(k, Open::Unknown(ref t, _) if t == "teleport"));
    }
}
