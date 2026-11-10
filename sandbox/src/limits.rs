//! The time and size budgets of a shell, set by its embedder.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::foreground_wait::ForegroundWait;
use crate::memory::ResponseWindow;
use crate::subagent::ImportBudget;

/// A duration adjustable while the shell runs. Every clone shares it.
#[derive(Debug, Clone)]
pub struct LiveDuration(Arc<AtomicU64>);

impl LiveDuration {
    /// A duration of `duration`.
    #[must_use]
    pub fn new(duration: Duration) -> Self {
        Self(Arc::new(AtomicU64::new(Self::nanos(duration))))
    }

    fn nanos(duration: Duration) -> u64 {
        u64::try_from(duration.as_nanos()).expect("a budget fits in u64 nanoseconds")
    }

    /// Changes the duration for every handle.
    pub fn set(&self, duration: Duration) {
        self.0.store(Self::nanos(duration), Ordering::Relaxed);
    }

    /// The duration now.
    #[must_use]
    pub fn get(&self) -> Duration {
        Duration::from_nanos(self.0.load(Ordering::Relaxed))
    }
}

/// The budgets a shell runs its scripts under.
///
/// There is no `Default`: the values are the embedder's, and each is a live
/// handle, so changing one reaches the shells already running.
#[derive(Debug, Clone)]
pub struct ShellLimits {
    /// How long a foreground call waits before a still-running script
    /// promotes to a job.
    pub foreground_wait: ForegroundWait,
    /// The largest `timeout` a script accepts.
    pub timeout_cap: LiveDuration,
    /// How long a job runs before it raises one [`ShellEvent::LongRunning`].
    ///
    /// [`ShellEvent::LongRunning`]: crate::ShellEvent::LongRunning
    pub long_job_alert: LiveDuration,
    /// The most bytes a subagent's sandbox may send the host while its
    /// commits are imported.
    pub subagent_import: ImportBudget,
    /// How long the agent has to act on a final memory warning.
    pub response_window: ResponseWindow,
}

impl ShellLimits {
    /// Limits holding the given values.
    #[must_use]
    pub fn new(
        foreground_wait: Duration,
        timeout_cap: Duration,
        long_job_alert: Duration,
        subagent_import_bytes: u64,
        response_window: Duration,
    ) -> Self {
        Self {
            foreground_wait: ForegroundWait::new(foreground_wait),
            timeout_cap: LiveDuration::new(timeout_cap),
            long_job_alert: LiveDuration::new(long_job_alert),
            subagent_import: ImportBudget::new(subagent_import_bytes),
            response_window: ResponseWindow::new(response_window),
        }
    }
}

#[cfg(any(test, feature = "test-budgets"))]
pub mod fixture {
    use std::time::Duration;

    use super::ShellLimits;

    /// Limits for tests, whose numbers stand for nothing.
    pub fn limits() -> ShellLimits {
        ShellLimits::new(
            Duration::from_secs(10),
            Duration::from_secs(120 * 60),
            Duration::from_secs(40 * 60),
            1024 * 1024 * 1024,
            Duration::from_secs(120),
        )
    }
}
