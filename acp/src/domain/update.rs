//! `session/update` notification payloads.
//!
//! [`SessionUpdate`] is the canonical v2 open union tagged
//! `sessionUpdate`. Known tags parse strictly; `_`-prefixed tags land in
//! [`SessionUpdate::Custom`] and all other unrecognized tags land in
//! [`SessionUpdate::Unknown`], each preserving the raw object so future
//! protocol updates round-trip uninterpreted.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::command::AvailableCommandsUpdate;
use super::config::ConfigOptionUpdate;
use super::content::{ContentBlock, ContentChunk, Message};
use super::ids::{CompactionId, MessageId, PlanId, SessionId, TerminalId, ToolCallId};
use super::open::{ExtIdent, Open, open_tags, open_union_serde};
use super::patch::Patch;
use super::session::{CurrentModeUpdate, SessionInfoUpdate};
use super::terminal::{TerminalOutputChunk, TerminalUpdate};
use super::tool::{ToolCallContentChunk, ToolCallUpdate};
use super::usage::UsageUpdate;

/// One `session/update` payload (`{"sessionUpdate":"<tag>",...}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionUpdate {
    /// A chunk appended to a user message.
    UserMessageChunk(ContentChunk),
    /// A chunk appended to an agent message.
    AgentMessageChunk(ContentChunk),
    /// A chunk appended to agent reasoning.
    AgentThoughtChunk(ContentChunk),
    /// Whole user-message upsert.
    UserMessage(Message),
    /// Whole agent-message upsert.
    AgentMessage(Message),
    /// Whole agent-reasoning upsert.
    AgentThought(Message),
    /// Session state snapshot.
    StateUpdate(StateUpdate),
    /// A tool call upsert.
    ToolCallUpdate(ToolCallUpdate),
    /// A content chunk appended to a tool call.
    ToolCallContentChunk(ToolCallContentChunk),
    /// A terminal upsert.
    TerminalUpdate(TerminalUpdate),
    /// A chunk appended to a terminal's output.
    TerminalOutputChunk(TerminalOutputChunk),
    /// A plan upsert.
    PlanUpdate(PlanUpdate),
    /// A subagent session upsert.
    SubagentUpdate(SubagentUpdate),
    /// A session-to-session message upsert.
    SessionMessage(SessionMessage),
    /// A chunk appended to a session message.
    SessionMessageChunk(SessionMessageChunk),
    /// The available-command list changed.
    AvailableCommandsUpdate(AvailableCommandsUpdate),
    /// The config-option list changed.
    ConfigOptionUpdate(ConfigOptionUpdate),
    /// Session title/updated-at patch.
    SessionInfoUpdate(SessionInfoUpdate),
    /// Context usage.
    UsageUpdate(UsageUpdate),
    /// Context compaction progress.
    CompactionUpdate(CompactionUpdate),
    /// A chunk of compaction summary.
    CompactionSummaryChunk(CompactionSummaryChunk),
    /// A transient notice.
    Notice(Notice),
    /// Replay stream end marker.
    ReplayComplete(ReplayComplete),
    /// v1 mode echo (`current_mode_update`); produced by the v1 codec,
    /// consumed by the reducer.
    CurrentModeUpdate(CurrentModeUpdate),
    /// `_`-prefixed extension update, preserved verbatim.
    Custom(ExtIdent, Value),
    /// Unrecognized update, preserved verbatim.
    Unknown(String, Value),
}

open_union_serde!(SessionUpdate, "sessionUpdate", [
    UserMessageChunk => "user_message_chunk",
    AgentMessageChunk => "agent_message_chunk",
    AgentThoughtChunk => "agent_thought_chunk",
    UserMessage => "user_message",
    AgentMessage => "agent_message",
    AgentThought => "agent_thought",
    StateUpdate => "state_update",
    ToolCallUpdate => "tool_call_update",
    ToolCallContentChunk => "tool_call_content_chunk",
    TerminalUpdate => "terminal_update",
    TerminalOutputChunk => "terminal_output_chunk",
    PlanUpdate => "plan_update",
    SubagentUpdate => "subagent_update",
    SessionMessage => "session_message",
    SessionMessageChunk => "session_message_chunk",
    AvailableCommandsUpdate => "available_commands_update",
    ConfigOptionUpdate => "config_option_update",
    SessionInfoUpdate => "session_info_update",
    UsageUpdate => "usage_update",
    CompactionUpdate => "compaction_update",
    CompactionSummaryChunk => "compaction_summary_chunk",
    Notice => "notice",
    ReplayComplete => "replay_complete",
    CurrentModeUpdate => "current_mode_update",
]);

/// Session state (`{"state", "stopReason"?, "background"?, "error"?}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateUpdate {
    /// Foreground state.
    pub state: Open<ForegroundState>,
    /// Why the last turn ended (set on `idle`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<Open<StopReason>>,
    /// Work still running after the turn returned.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub background: Vec<BackgroundItem>,
    /// A JSON-RPC error object (`{code,message,data?}`) describing the
    /// failure when the turn ended with `stopReason: "error"`. Kept as
    /// `Value` to keep `domain` independent of `crate::rpc`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Foreground work state (bare string on the wire).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForegroundState {
    /// The turn is actively working.
    Running,
    /// No work in flight.
    Idle,
    /// Work is blocked on user action.
    RequiresAction,
    /// The agent cannot report its state (`"unknown"`).
    Unknown,
}
open_tags!(ForegroundState; None => ["running", "idle", "requires_action", "unknown"]);

/// Why a turn ended (bare string on the wire).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The turn completed normally.
    EndTurn,
    /// Token limit reached.
    MaxTokens,
    /// Per-turn request limit reached.
    MaxTurnRequests,
    /// The agent refused to continue.
    Refusal,
    /// The client cancelled the turn.
    Cancelled,
    /// The turn failed (see [`StateUpdate::error`]).
    Error,
}
open_tags!(StopReason; None => ["end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled", "error"]);

/// Background work still running after the turn (tagged `type`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackgroundItem {
    /// `{"type":"terminal","terminalId"}`.
    Terminal(BackgroundTerminal),
    /// `{"type":"tool_call","toolCallId"}`.
    ToolCall(BackgroundToolCall),
    /// `{"type":"subagent","sessionId"}`.
    Subagent(BackgroundSubagent),
    /// `_`-prefixed extension item, preserved verbatim.
    Custom(ExtIdent, Value),
    /// Unrecognized item, preserved verbatim.
    Unknown(String, Value),
}

open_union_serde!(BackgroundItem, "type", [
    Terminal => "terminal",
    ToolCall => "tool_call",
    Subagent => "subagent",
]);

/// `{"type":"terminal","terminalId"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackgroundTerminal {
    /// The still-running terminal.
    pub terminal_id: TerminalId,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `{"type":"tool_call","toolCallId"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackgroundToolCall {
    /// The still-running tool call.
    pub tool_call_id: ToolCallId,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `{"type":"subagent","sessionId"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackgroundSubagent {
    /// The still-running subagent session.
    pub session_id: SessionId,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A plan upsert (`{"plan": <PlanUpdateContent>}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanUpdate {
    /// The plan payload.
    pub plan: Open<Plan>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The plan payload (tagged `type`; only `items` is known).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Plan {
    /// `{"type":"items","planId","entries"}`.
    Items(PlanItems),
}
open_tags!(Plan; Some("type") => ["items"]);

/// `{"type":"items","planId","entries"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanItems {
    /// The plan being replaced.
    pub plan_id: PlanId,
    /// The full current entry list.
    pub entries: Vec<PlanEntry>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// One plan entry (`{"content","priority","status"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanEntry {
    /// The work item.
    pub content: String,
    /// Priority.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<Open<PlanEntryPriority>>,
    /// Status.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<Open<PlanEntryStatus>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Plan entry priority (bare string).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanEntryPriority {
    /// High priority.
    High,
    /// Medium priority.
    Medium,
    /// Low priority.
    Low,
}
open_tags!(PlanEntryPriority; None => ["high", "medium", "low"]);

/// Plan entry status (bare string).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanEntryStatus {
    /// Not started.
    Pending,
    /// In progress.
    InProgress,
    /// Completed.
    Completed,
    /// Cancelled.
    Cancelled,
}
open_tags!(PlanEntryStatus; None => ["pending", "in_progress", "completed", "cancelled"]);

/// A subagent session upsert (`{"sessionId", <patch fields>}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubagentUpdate {
    /// The subagent's session id.
    pub session_id: SessionId,
    /// Display title.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub title: Patch<String>,
    /// Description.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub description: Patch<String>,
    /// Capabilities of the subagent session.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub capabilities: Patch<Value>,
    /// The subagent's own state.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub state: Patch<StateUpdate>,
    /// Working directory.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub cwd: Patch<PathBuf>,
    /// Additional working directories.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub additional_directories: Patch<Vec<PathBuf>>,
    /// Extension metadata; `null` clears.
    #[serde(rename = "_meta", default, skip_serializing_if = "Patch::is_unchanged")]
    pub meta: Patch<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A session-to-session message upsert
/// (`{"messageId","senderSessionId"?,...,"content":patch}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMessage {
    /// The message id.
    pub message_id: MessageId,
    /// Originating session.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub sender_session_id: Patch<SessionId>,
    /// Destination session.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub recipient_session_id: Patch<SessionId>,
    /// The upstream message this one derives from.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub sender_message_id: Patch<MessageId>,
    /// The message this one is being delivered as.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub recipient_message_id: Patch<MessageId>,
    /// Message content; patch semantics.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub content: Patch<Vec<ContentBlock>>,
    /// Extension metadata; `null` clears.
    #[serde(rename = "_meta", default, skip_serializing_if = "Patch::is_unchanged")]
    pub meta: Patch<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A chunk appended to a session message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMessageChunk {
    /// The message this chunk appends to.
    pub message_id: MessageId,
    /// The content item.
    pub content: ContentBlock,
    /// Chunk-scoped extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A transient notice (`{"severity","title","description"?}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Notice {
    /// Notice severity.
    pub severity: Open<NoticeSeverity>,
    /// Short title.
    pub title: String,
    /// Longer description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Notice severity (bare string).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoticeSeverity {
    /// Informational.
    Info,
    /// A warning.
    Warning,
    /// An error.
    Error,
}
open_tags!(NoticeSeverity; None => ["info", "warning", "error"]);

/// Context-compaction progress
/// (`{"compactionId","status"?, "summary"?, "error"?}` — patch fields).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionUpdate {
    /// The compaction being reported.
    pub compaction_id: CompactionId,
    /// Compaction status.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub status: Patch<Open<CompactionStatus>>,
    /// Compacted summary content.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub summary: Patch<Vec<ContentBlock>>,
    /// Failure message.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub error: Patch<String>,
    /// Extension metadata; `null` clears.
    #[serde(rename = "_meta", default, skip_serializing_if = "Patch::is_unchanged")]
    pub meta: Patch<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Compaction status (bare string).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionStatus {
    /// Compaction is running.
    InProgress,
    /// Compaction completed.
    Completed,
    /// Compaction failed.
    Failed,
}
open_tags!(CompactionStatus; None => ["in_progress", "completed", "failed"]);

/// A chunk of a compaction's summary (`{"compactionId","content"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactionSummaryChunk {
    /// The compaction this chunk belongs to.
    pub compaction_id: CompactionId,
    /// The content item.
    pub content: ContentBlock,
    /// Chunk-scoped extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Marks the end of a replay stream (`{"sessionUpdate":"replay_complete"}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayComplete {
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
    fn session_update_known_unknown_round_trip() {
        let raw = json!({"sessionUpdate": "agent_message_chunk", "messageId": "m1", "content": {"type": "text", "text": "hi"}});
        let u: SessionUpdate = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(u, SessionUpdate::AgentMessageChunk(_)));
        assert_eq!(serde_json::to_value(&u).expect("ser"), raw);

        let raw = json!({"sessionUpdate": "_x_custom", "n": 1});
        let u: SessionUpdate = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(u, SessionUpdate::Custom(ref i, _) if i.as_str() == "_x_custom"));
        assert_eq!(serde_json::to_value(&u).expect("ser"), raw);

        let raw = json!({"sessionUpdate": "session_hologram", "x": 2});
        let u: SessionUpdate = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(u, SessionUpdate::Unknown(ref t, _) if t == "session_hologram"));
        assert_eq!(serde_json::to_value(&u).expect("ser"), raw);

        // Malformed known payload is a hard error.
        assert!(
            serde_json::from_value::<SessionUpdate>(json!({"sessionUpdate": "tool_call_update"}))
                .is_err()
        );
    }

    #[test]
    fn state_update_and_plan() {
        let raw = json!({"sessionUpdate": "state_update", "state": "idle", "stopReason": "end_turn",
            "background": [{"type": "terminal", "terminalId": "t1"}]});
        let u: SessionUpdate = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(u, SessionUpdate::StateUpdate(ref s) if s.background.len() == 1));
        assert_eq!(serde_json::to_value(&u).expect("ser"), raw);

        let raw = json!({"sessionUpdate": "plan_update", "plan": {"type": "items", "planId": "p1",
            "entries": [{"content": "do", "priority": "high", "status": "pending"}]}});
        let u: SessionUpdate = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(u, SessionUpdate::PlanUpdate(_)));
        assert_eq!(serde_json::to_value(&u).expect("ser"), raw);
    }
}
