//! Agent-owned terminals.
//!
//! v2 terminals are owned by the agent and reported to the client through
//! [`TerminalUpdate`] patches and [`TerminalOutputChunk`] streams. The
//! client-side `terminal/*` requests of v1 are a different (client-owned)
//! lane and live in [`crate::wire::v1`].

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ids::TerminalId;
use super::patch::{EmptyObject, Patch};

/// A terminal upsert (`{"terminalId", <patch fields>}`).
///
/// Every field besides `terminal_id` uses patch semantics: omitted leaves
/// the stored value, `null` clears it, a concrete value replaces it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalUpdate {
    /// The terminal being created or patched.
    pub terminal_id: TerminalId,
    /// Process identifier, when known.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub pid: Patch<i64>,
    /// Shell command.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub command: Patch<String>,
    /// Arguments to `command`.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub args: Patch<Vec<String>>,
    /// Working directory.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub cwd: Patch<PathBuf>,
    /// Current output snapshot.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub output: Patch<TerminalOutput>,
    /// Exit status; set once when the process ends.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub exit_status: Patch<TerminalExitStatus>,
    /// Terminal size.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub size: Patch<TerminalSize>,
    /// Terminal capabilities.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub capabilities: Patch<TerminalCaps>,
    /// Extension metadata; `null` clears.
    #[serde(rename = "_meta", default, skip_serializing_if = "Patch::is_unchanged")]
    pub meta: Patch<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl TerminalUpdate {
    /// A bare upsert for `terminal_id`.
    #[must_use]
    pub fn new(terminal_id: impl Into<TerminalId>) -> Self {
        Self {
            terminal_id: terminal_id.into(),
            ..Self::default()
        }
    }
}

/// A terminal output snapshot (`{"data", "truncated"?, "byteCount"?}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalOutput {
    /// Output text.
    #[serde(default)]
    pub data: String,
    /// Whether earlier bytes were dropped from the snapshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncated: Option<bool>,
    /// Total bytes produced so far.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byte_count: Option<u64>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A chunk appended to a terminal's output
/// (`{"terminalId", "data", "truncated"?}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalOutputChunk {
    /// The terminal receiving output.
    pub terminal_id: TerminalId,
    /// Output data.
    pub data: String,
    /// Whether earlier output was dropped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncated: Option<bool>,
    /// Chunk-scoped extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// How a terminal's process ended
/// (`{"exitCode"?, "signal"?, "success"?}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalExitStatus {
    /// Process exit code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    /// Signal that terminated the process.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal: Option<String>,
    /// Whether the terminal task finished successfully (may differ from
    /// `exit_code` semantics).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub success: Option<bool>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Terminal display size (`{"rows", "cols"}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalSize {
    /// Row count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<u32>,
    /// Column count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cols: Option<u32>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Terminal capabilities (`{"cancel": {}}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalCaps {
    /// `session/terminal_cancel` is honoured when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancel: Option<EmptyObject>,
    /// Extension metadata.
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
    fn terminal_update_patch_fields() {
        let raw = json!({"terminalId": "t1", "command": "ls", "output": null, "exitStatus": {"exitCode": 0}});
        let u: TerminalUpdate = serde_json::from_value(raw).expect("de");
        assert!(matches!(u.command, Patch::Set(ref c) if c == "ls"));
        assert_eq!(u.output, Patch::Clear);
        assert!(matches!(u.exit_status, Patch::Set(ref s) if s.exit_code == Some(0)));
        let back = serde_json::to_value(&u).expect("ser");
        assert_eq!(back["output"], serde_json::Value::Null);
        assert!(back.get("cwd").is_none());
    }
}
