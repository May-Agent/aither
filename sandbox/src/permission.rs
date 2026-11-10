//! Execution modes of the terminal.
//!
//! - `Sandboxed`: no approval needed; outbound network is available by
//!   default, filtered per-domain by the sandbox's network lists and its
//!   [`DomainApprover`](crate::DomainApprover) and audited by higher layers
//! - `Unsafe`: runs without the sandbox, so a [`ScriptPolicy`](crate::ScriptPolicy)
//!   must allow it or the approver must approve it, per script

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Permission mode for terminal execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TerminalMode {
    /// Sandboxed execution without extra host privileges.
    ///
    /// Outbound network is available by default, filtered per-domain through
    /// the configured network policy and always audited.
    #[default]
    Sandboxed,

    /// Unsafe execution without sandbox.
    /// Full system access. Requires per-script approval.
    Unsafe,
}

impl TerminalMode {
    /// Returns a human-readable description of this mode.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Sandboxed => "sandboxed",
            Self::Unsafe => "unsafe (no sandbox, full access)",
        }
    }
}
