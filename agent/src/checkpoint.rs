//! Agent checkpoint bundles for restart-safe persistence.

use serde::{Deserialize, Serialize};

use crate::{ContextCheckpoint, ContextWindowSnapshot};

/// Checkpoint payload exported by the agent runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentCheckpoint {
    /// Non-persistent runtime context managed by the agent.
    pub context: ContextCheckpoint,
    /// Hash of the active tool surface for compatibility checks.
    pub tool_surface_hash: String,
    /// Structured snapshot of the current context window.
    pub context_window: ContextWindowSnapshot,
}
