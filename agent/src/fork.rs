//! What a fork subagent inherits from its parent agent.
//!
//! A parent agent publishes the request it is about to send to a
//! [`ForkSource`]; an embedder's fork subagent reads the latest one and starts
//! from it with [`AgentBuilder::forked_from`](crate::AgentBuilder::forked_from),
//! so its first request is the parent's request followed by the fork prompt
//! and reads the parent's warm prompt cache.

use std::sync::{Arc, Mutex};

use aither_core::llm::model::{Parameters, Profile};
use aither_core::llm::tool::ToolDefinition;

use crate::context::{Context, ContextBudget, ContextPolicy, PromptCache};

/// The parent agent's request, as a fork subagent starts from it.
///
/// Everything a request is made of is kept as the parent assembled it: the
/// window (system blocks, handoff, conversation and live tail, thinking
/// blocks included), the tool definitions, the request parameters and the
/// prompt-cache identity.
#[derive(Debug, Clone)]
pub struct ForkSnapshot {
    /// The window the request was built from, with its live tail as the
    /// last message.
    pub(crate) context: Context,
    pub(crate) prompt_cache: PromptCache,
    pub(crate) budget: ContextBudget,
    /// The parent's model, whose name identifies it.
    pub(crate) profile: Profile,
    pub(crate) request_parameters: Parameters,
    /// The request-level effort the parent sends, so the fork's first
    /// request carries the same and reads the parent's cache.
    pub(crate) effort: crate::effort::EffortAnchor,
    pub(crate) context_policy: ContextPolicy,
    pub(crate) definitions: Vec<ToolDefinition>,
    /// Window usage the request was sent at.
    pub(crate) usage: f32,
}

impl ForkSnapshot {
    /// The name of the model the parent ran the request on.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.profile.name
    }

    /// The tool definitions the parent's request carried. The fork's own
    /// tools must present exactly these, or its first request diverges from
    /// the parent's cached prefix at the tool definitions.
    #[must_use]
    pub fn definitions(&self) -> &[ToolDefinition] {
        &self.definitions
    }

    /// The fraction of the context window the parent's request used.
    #[must_use]
    pub const fn usage(&self) -> f32 {
        self.usage
    }

    /// The fraction of the context window the parent pauses at.
    #[must_use]
    pub fn compaction_usage(&self) -> f32 {
        let window = f64::from(self.budget.window());
        if window == 0.0 {
            return 0.0;
        }
        let point = f64::from(
            u32::try_from(self.context_policy.pause_point(self.budget.window()))
                .unwrap_or(u32::MAX),
        );
        num_traits::ToPrimitive::to_f32(&(point / window)).unwrap_or(1.0)
    }

    /// Whether a fork may start from the request the parent sent: it is
    /// refused within a tenth of the parent's pause point.
    #[must_use]
    pub fn allows_fork(&self) -> bool {
        self.context_policy
            .allows_fork(self.budget.estimated_tokens(), self.budget.window())
    }

    /// The parent's context policy.
    #[must_use]
    pub const fn policy(&self) -> &ContextPolicy {
        &self.context_policy
    }
}

/// The latest request of a parent agent, shared with whatever forks it.
///
/// Create one, hand a clone to the parent's builder with
/// [`AgentBuilder::fork_source`](crate::AgentBuilder::fork_source) and keep
/// one: the agent publishes every request it sends, and the holder reads the
/// newest.
#[derive(Debug, Clone, Default)]
pub struct ForkSource {
    latest: Arc<Mutex<Option<Arc<ForkSnapshot>>>>,
}

impl ForkSource {
    /// Replaces the published request.
    pub(crate) fn publish(&self, snapshot: ForkSnapshot) {
        *self
            .latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::new(snapshot));
    }

    /// The request the parent sent last; `None` before its first.
    #[must_use]
    pub fn latest(&self) -> Option<Arc<ForkSnapshot>> {
        self.latest
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}
