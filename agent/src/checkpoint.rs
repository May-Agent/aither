//! Agent checkpoint bundles for restart-safe persistence.

use std::sync::Arc;

use arc_swap::ArcSwapOption;
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

/// The newest checkpoint of a run in progress, readable while the run is
/// being driven.
///
/// The run itself holds the agent, so an embedder cannot call
/// [`Agent::export_checkpoint`](crate::Agent::export_checkpoint) between its
/// events. The run instead records a checkpoint each time it emits
/// [`AgentEvent::Checkpoint`](crate::AgentEvent::Checkpoint), before the event
/// is queued; this handle, taken with
/// [`Run::checkpoints`](crate::Run::checkpoints) before the run is driven,
/// reads it from the event handler.
///
/// The checkpoint read is the one of that event or of a later one: the run
/// keeps progressing while the handler awaits, so it may have checkpointed
/// again. Either is a consistent state to restore.
#[derive(Debug, Clone, Default)]
pub struct Checkpoints(Arc<ArcSwapOption<AgentCheckpoint>>);

impl Checkpoints {
    /// The newest checkpoint the run recorded, or `None` before its first
    /// [`AgentEvent::Checkpoint`](crate::AgentEvent::Checkpoint).
    #[must_use]
    pub fn latest(&self) -> Option<AgentCheckpoint> {
        self.0.load_full().map(|checkpoint| (*checkpoint).clone())
    }

    pub(crate) fn record(&self, checkpoint: AgentCheckpoint) {
        self.0.store(Some(Arc::new(checkpoint)));
    }
}
