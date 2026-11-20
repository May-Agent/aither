//! Permission requests and outcomes.
//!
//! `session/request_permission` asks the client to authorize a described
//! action ([`PermissionSubject`], known `tool_call`/`command` shapes plus
//! custom/unknown). The client replies with a [`PermissionOutcome`] —
//! `selected` (with an `option_id`) or `cancelled`, preserving unknown
//! outcomes verbatim.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ids::{PermissionOptionId, SessionId, TerminalId, ToolCallId};
use super::open::{ExtIdent, Open, open_tags, open_union_serde};
use super::patch::Patch;
use super::tool::ToolCallUpdate;

/// `session/request_permission` params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionRequest {
    /// Target session.
    pub session_id: SessionId,
    /// Short human-readable description of the action.
    pub title: String,
    /// Longer explanation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// What permission is being asked about.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<Open<PermissionSubject>>,
    /// Options the user may pick.
    pub options: Vec<PermissionOption>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// What a permission request refers to (tagged `type`); wrapped in
/// [`Open`] at the field to preserve custom/unknown subjects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PermissionSubject {
    /// A tool call (`{"type":"tool_call","toolCall":<ToolCallUpdate>}`).
    ToolCall(Box<ToolCallSubject>),
    /// A shell command (`{"type":"command","command","cwd",...}`).
    Command(CommandSubject),
}
open_tags!(PermissionSubject; Some("type") => ["tool_call", "command"]);

/// `{"type":"tool_call","toolCall":<ToolCallUpdate>}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallSubject {
    /// The tool call requiring permission.
    pub tool_call: ToolCallUpdate,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `{"type":"command","command","cwd","toolCallId"?,"terminalId"?}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommandSubject {
    /// The command line.
    pub command: String,
    /// Working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// Related tool call, when the command runs inside one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<ToolCallId>,
    /// Related agent-owned terminal, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_id: Option<TerminalId>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// One option offered by a permission request
/// (`{"optionId","name","kind"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionOption {
    /// Option id echoed back by the `selected` outcome.
    pub option_id: PermissionOptionId,
    /// Display name.
    pub name: String,
    /// Behaviour the option implies.
    pub kind: Open<PermissionOptionKind>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Behaviour a [`PermissionOption`] grants (bare string on the wire).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionOptionKind {
    /// Allow this invocation only.
    AllowOnce,
    /// Allow this and equivalent future invocations.
    AllowAlways,
    /// Deny this invocation.
    RejectOnce,
    /// Deny this and equivalent future invocations.
    RejectAlways,
}
open_tags!(PermissionOptionKind; None => ["allow_once", "allow_always", "reject_once", "reject_always"]);

/// `session/request_permission` result (`{"outcome": {...}}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionResult {
    /// How the request resolved.
    pub outcome: PermissionOutcome,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// How a permission request resolved (tagged `outcome`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionOutcome {
    /// The user picked an option (`{"outcome":"selected","optionId"}`).
    Selected(SelectedOutcome),
    /// The request was cancelled (`{"outcome":"cancelled"}`).
    Cancelled(CancelledOutcome),
    /// `_`-prefixed extension outcome, preserved verbatim.
    Custom(ExtIdent, Value),
    /// Unrecognized outcome, preserved verbatim.
    Unknown(String, Value),
}

open_union_serde!(PermissionOutcome, "outcome", [
    Selected => "selected",
    Cancelled => "cancelled",
]);

/// `{"outcome":"selected","optionId"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectedOutcome {
    /// The chosen [`PermissionOption`]'s id.
    pub option_id: PermissionOptionId,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `{"outcome":"cancelled"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelledOutcome {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Patch-shaped field used by updates that may clear their permission
/// subject; re-exported for convenience.
pub type SubjectPatch = Patch<Open<PermissionSubject>>;

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn outcome_selected_cancelled_unknown() {
        let o: PermissionOutcome =
            serde_json::from_value(json!({"outcome": "selected", "optionId": "ok"})).expect("de");
        assert!(matches!(o, PermissionOutcome::Selected(ref s) if s.option_id.as_str() == "ok"));

        let o: PermissionOutcome =
            serde_json::from_value(json!({"outcome": "cancelled"})).expect("de");
        assert!(matches!(o, PermissionOutcome::Cancelled(_)));

        let raw = json!({"outcome": "_timeout", "secs": 5});
        let o: PermissionOutcome = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(o, PermissionOutcome::Custom(_, _)));
        assert_eq!(serde_json::to_value(&o).expect("ser"), raw);
    }

    #[test]
    fn result_nests_the_outcome() {
        let raw = json!({"outcome": {"outcome": "selected", "optionId": "ok"}});
        let r: PermissionResult = serde_json::from_value(raw.clone()).expect("de");
        assert!(
            matches!(r.outcome, PermissionOutcome::Selected(ref s) if s.option_id.as_str() == "ok")
        );
        assert_eq!(serde_json::to_value(&r).expect("ser"), raw);

        let raw = json!({"outcome": {"outcome": "cancelled"}});
        let r: PermissionResult = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(r.outcome, PermissionOutcome::Cancelled(_)));
        assert_eq!(serde_json::to_value(&r).expect("ser"), raw);
    }

    #[test]
    fn subject_open_union() {
        let raw = json!({"type": "tool_call", "toolCall": {"toolCallId": "t", "title": "rm -rf"}});
        let s: PermissionSubject = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(s, PermissionSubject::ToolCall(_)));
        assert_eq!(serde_json::to_value(&s).expect("ser"), raw);

        let raw = json!({"type": "command", "command": "make", "cwd": "/tmp"});
        let s: PermissionSubject = serde_json::from_value(raw).expect("de");
        assert!(matches!(s, PermissionSubject::Command(ref c) if c.command == "make"));
    }
}
