//! Agent configuration.

use crate::context::ContextPolicy;

/// Configuration for agent behavior.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum number of agent loop iterations.
    pub max_iterations: usize,

    /// Thresholds governing compaction, pausing and reassembly.
    pub context_policy: ContextPolicy,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            context_policy: ContextPolicy::default(),
        }
    }
}

impl AgentConfig {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub const fn with_max_iterations(mut self, limit: usize) -> Self {
        self.max_iterations = limit;
        self
    }

    /// Sets the context management policy.
    #[must_use]
    pub const fn with_context_policy(mut self, policy: ContextPolicy) -> Self {
        self.context_policy = policy;
        self
    }
}
