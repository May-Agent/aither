//! Budgets for tests.
//!
//! Their numbers stand for nothing: a test that depends on a value states it.

use crate::{OutputPolicy, ShellLimits};

/// Shell limits for a test.
#[must_use]
pub fn limits() -> ShellLimits {
    crate::limits::fixture::limits()
}

/// An output policy for a test.
#[must_use]
pub fn output_policy() -> OutputPolicy {
    crate::output_policy::fixture::policy()
}

/// The monitor settings for a test.
#[must_use]
pub fn monitor_settings() -> crate::MonitorSettings {
    crate::monitor::fixture_settings()
}
