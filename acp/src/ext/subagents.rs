//! `subagent_spawned` / `subagent_state_update` session updates.
//!
//! A provider-neutral convention: an agent that delegates work to
//! sub-agent sessions advertises a `subagents` object under
//! `agentCapabilities.session` (vendor fields), then reports their
//! lifecycle through `session/update` notifications carrying
//! `subagent_spawned` and `subagent_state_update` update kinds. Neither
//! kind is a standard `sessionUpdate` tag, so on the domain model they
//! arrive as [`SessionUpdate::Unknown`] (or
//! [`SessionUpdate::Custom`] when `_`-prefixed) payloads —
//! [`try_from_update`] decodes them. (v2's native `subagent_update`
//! variant is a different, domain-modeled lane.)

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::domain::open::string_enum;
use crate::domain::{AgentCapabilities, SessionId, SessionUpdate};
use crate::ext::tagged_payload;

/// `SessionCaps.extra` key advertising subagent reporting.
pub const CAPABILITY: &str = "subagents";

/// The `subagent_spawned` update kind.
pub const SPAWNED_UPDATE: &str = "subagent_spawned";

/// The `subagent_state_update` update kind.
pub const STATE_UPDATE: &str = "subagent_state_update";

/// Whether the agent advertised subagent session reporting.
#[must_use]
pub fn advertised(caps: &AgentCapabilities) -> bool {
    caps.session
        .as_ref()
        .is_some_and(|session| session.extra.contains_key(CAPABILITY))
}

string_enum! {
    /// Terminal state of a subagent session.
    pub enum SubagentState {
        /// The subagent finished its task.
        Completed = "completed",
        /// The subagent failed.
        Failed = "failed",
        /// The subagent was cancelled.
        Cancelled = "cancelled",
        /// The subagent's connection dropped.
        Disconnected = "disconnected",
    }
}

/// Which control operations a spawned subagent accepts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubagentCapabilities {
    /// Whether/how the subagent can be cancelled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancel: Option<Value>,
    /// Whether/how the subagent can be closed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub close: Option<Value>,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A `subagent_spawned` update payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubagentSpawned {
    /// Session ID the agent assigned to the subagent.
    pub subagent_session_id: SessionId,
    /// Human-readable subagent name.
    pub name: String,
    /// The delegated task.
    pub task: String,
    /// Control operations the subagent accepts.
    #[serde(default)]
    pub capabilities: SubagentCapabilities,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A `subagent_state_update` update payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubagentStateUpdate {
    /// Session ID of the subagent that ended.
    pub subagent_session_id: SessionId,
    /// How the subagent ended.
    pub state: SubagentState,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A decoded subagent session update.
#[derive(Debug, Clone)]
pub enum SubagentUpdate {
    /// A subagent was spawned.
    Spawned(SubagentSpawned),
    /// A subagent reached a terminal state.
    State(SubagentStateUpdate),
}

/// Interpret a [`SessionUpdate`] as a subagent update.
///
/// Returns `None` for every update that is neither `subagent_spawned`
/// nor `subagent_state_update`, and for malformed payloads.
#[must_use]
pub fn try_from_update(update: &SessionUpdate) -> Option<SubagentUpdate> {
    let (kind, payload) = tagged_payload(update)?;
    match kind.as_str() {
        SPAWNED_UPDATE => serde_json::from_value(payload)
            .ok()
            .map(SubagentUpdate::Spawned),
        STATE_UPDATE => serde_json::from_value(payload)
            .ok()
            .map(SubagentUpdate::State),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The capability is a `subagents` key on `agentCapabilities.session`.
    #[test]
    fn advertised_capability() {
        let caps: AgentCapabilities = serde_json::from_value(json!({
            "session": {"subagents": {}},
        }))
        .expect("deserializes");
        assert!(advertised(&caps));
        assert!(!advertised(&AgentCapabilities::default()));
    }

    /// Spawned and terminal-state updates decode from `Unknown` payloads.
    #[test]
    fn update_parsing() {
        let update: SessionUpdate = serde_json::from_value(json!({
            "sessionUpdate": "subagent_spawned",
            "subagentSessionId": "sub-1",
            "name": "researcher",
            "task": "find docs",
            "capabilities": {"cancel": true},
            "vendor": "x",
        }))
        .expect("deserializes");
        assert!(matches!(update, SessionUpdate::Unknown(..)));
        let Some(SubagentUpdate::Spawned(spawned)) = try_from_update(&update) else {
            panic!("expected spawned");
        };
        assert_eq!(spawned.subagent_session_id.as_str(), "sub-1");
        assert_eq!(spawned.capabilities.cancel, Some(json!(true)));
        assert_eq!(spawned.extra["vendor"], "x");

        let update: SessionUpdate = serde_json::from_value(json!({
            "sessionUpdate": "subagent_state_update",
            "subagentSessionId": "sub-1",
            "state": "completed",
        }))
        .expect("deserializes");
        let Some(SubagentUpdate::State(state)) = try_from_update(&update) else {
            panic!("expected state");
        };
        assert_eq!(state.state, SubagentState::Completed);

        // Unrelated `Unknown` updates and standard variants return None.
        let update: SessionUpdate = serde_json::from_value(json!({
            "sessionUpdate": "vendor_thing",
        }))
        .expect("deserializes");
        assert!(try_from_update(&update).is_none());
        let update: SessionUpdate = serde_json::from_value(json!({
            "sessionUpdate": "session_info_update",
        }))
        .expect("deserializes");
        assert!(try_from_update(&update).is_none());
    }
}
