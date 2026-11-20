//! ACP v2 wire shapes that differ from [`crate::domain`].
//!
//! Everything else on the v2 wire IS the domain model; only these
//! handshake/prompt envelopes live here.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::domain::{
    AgentCapabilities, AuthMethod, AuthMethodId, ClientCapabilities, Implementation, MessageId,
};

/// `initialize` request as sent on the v2 wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeRequest {
    /// Requested protocol version (`2` for v2).
    pub protocol_version: u16,
    /// Client capabilities (v2 shape).
    pub capabilities: ClientCapabilities,
    /// Client implementation info.
    pub info: Implementation,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `initialize` response on the v2 wire.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResponse {
    /// Negotiated protocol version.
    #[serde(default)]
    pub protocol_version: u16,
    /// Agent capabilities (v2 shape).
    pub capabilities: AgentCapabilities,
    /// Agent implementation info.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info: Option<Implementation>,
    /// Advertised authentication methods.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auth_methods: Vec<AuthMethod>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `auth/login` params on the v2 wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthLoginParams {
    /// Method to log in with.
    pub method_id: AuthMethodId,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `session/resume` params on the v2 wire (`replayFrom` spelled out).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionResumeParams {
    /// Setup fields shared with `session/new`.
    #[serde(flatten)]
    pub setup: crate::domain::SessionSetupParams,
    /// Session to resume.
    pub session_id: crate::domain::SessionId,
    /// Replay cursor (`{"type":"start"}` for full replay).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_from: Option<crate::domain::Open<crate::domain::ReplayCursor>>,
}

/// `session/prompt` result on the v2 wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptResult {
    /// Message id the agent assigned the submitted user message.
    pub message_id: MessageId,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}
