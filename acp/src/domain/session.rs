//! Session lifecycle: setup, resume/replay, listing, prompt, cancel.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::command::AvailableCommand;
use super::config::ConfigOption;
use super::content::ContentBlock;
use super::ids::{MessageId, SessionId};
use super::mcp::McpServer;
use super::open::Open;
use super::patch::Patch;
use super::update::SessionUpdate;
use super::usage::Usage;

/// Shared session-setup fields for `session/new`, `session/load`,
/// `session/resume`, and `session/fork`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSetupParams {
    /// Working directory.
    pub cwd: PathBuf,
    /// Additional working directories.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_directories: Vec<PathBuf>,
    /// MCP servers to connect.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp_servers: Vec<McpServer>,
    /// System prompt override, when supported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<Open<SystemPrompt>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// How a session's system prompt is set (tagged `type`; only `mode` is
/// known).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SystemPrompt {
    /// `{"type":"mode","mode"}` — select a system-prompt mode.
    Mode(ModeSystemPrompt),
}

impl super::open::Openable for SystemPrompt {
    const TAG: Option<&'static str> = Some("type");
    const KNOWN: &'static [&'static str] = &["mode"];
}

/// `{"type":"mode","mode"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModeSystemPrompt {
    /// Mode identifier.
    pub mode: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `session/new` params — setup fields only.
pub type NewSessionParams = SessionSetupParams;

/// `session/resume`/`session/load`/`session/fork` params: setup fields
/// plus the target `sessionId` and `replayFrom` cursor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResumeParams {
    /// Setup fields.
    #[serde(flatten)]
    pub setup: SessionSetupParams,
    /// Session to attach.
    pub session_id: SessionId,
    /// Replay cursor (`{"type":"start"}` to replay from the beginning;
    /// absent to attach without replay).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_from: Option<Open<ReplayCursor>>,
}

/// Where replay starts (tagged `type`; only `start` is known).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReplayCursor {
    /// `{"type":"start"}` — replay from the first replayable entry.
    Start(ReplayStart),
}

impl super::open::Openable for ReplayCursor {
    const TAG: Option<&'static str> = Some("type");
    const KNOWN: &'static [&'static str] = &["start"];
}

/// `{"type":"start"}` replay cursor payload.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayStart {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `session/fork` params (same shape as [`ResumeParams`]).
pub type ForkParams = ResumeParams;

/// Result of `session/new`/`session/load`/`session/resume`/`session/fork`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetupResult {
    /// The session id (echoed for resume/load/fork).
    pub session_id: SessionId,
    /// Configuration options available for the session.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config_options: Vec<ConfigOption>,
    /// Commands available in the session.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub available_commands: Vec<AvailableCommand>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `session/list` params.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionListParams {
    /// Filter by working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// Pagination cursor from a previous page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A page of `session/list` results.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionPage {
    /// Sessions in this page.
    #[serde(default)]
    pub sessions: Vec<SessionInfo>,
    /// Cursor for the next page, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Metadata for one listed session (`{"sessionId","cwd","title"?,"updatedAt"?}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfo {
    /// Session id.
    pub session_id: SessionId,
    /// Working directory.
    pub cwd: PathBuf,
    /// Optional title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// RFC 3339 last-updated time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `session/delete`/`session/close` params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionEndParams {
    /// Session to end.
    pub session_id: SessionId,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `session/cancel` params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelParams {
    /// Session whose turn should stop.
    pub session_id: SessionId,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `session/prompt` params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptParams {
    /// Target session.
    pub session_id: SessionId,
    /// Prompt content.
    pub prompt: Vec<ContentBlock>,
    /// Client-generated idempotency key (v2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// v2 `session/prompt` result: the turn was accepted and its user message
/// was assigned `message_id`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptAccepted {
    /// The id the agent gave the submitted user message.
    pub message_id: MessageId,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The v1-style turn-end prompt result (how v1 `session/prompt` resolves
/// through the codec): a stop reason plus optional usage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptFinished {
    /// Why the turn ended.
    pub stop_reason: Open<crate::domain::update::StopReason>,
    /// Token usage for the turn, when reported.
    pub usage: Option<Usage>,
    /// Extension metadata.
    pub meta: Option<Value>,
}

/// A `session/update` notification payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNotification {
    /// Session the update belongs to.
    pub session_id: SessionId,
    /// The update.
    pub update: SessionUpdate,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// v1 legacy mode state (`{"modes":{...}}` becomes a mode-category
/// [`ConfigOption`] through the codec); kept here for the domain
/// [`SessionUpdate::CurrentModeUpdate`] variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CurrentModeUpdate {
    /// The active mode's value id.
    pub current_mode_id: super::ids::ConfigValueId,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Session info patch (`{"sessionUpdate":"session_info_update",...}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfoUpdate {
    /// New title.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub title: Patch<String>,
    /// New updated-at timestamp.
    #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
    pub updated_at: Patch<String>,
    /// Extension metadata; `null` clears.
    #[serde(rename = "_meta", default, skip_serializing_if = "Patch::is_unchanged")]
    pub meta: Patch<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}
