//! The foreground window of a terminal, adjustable while commands run.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// How long a `terminal` call waits in the foreground before a still-running
/// command promotes to a background task.
///
/// The window is a normal one plus an optional longer one that applies while
/// it is held. Every clone shares both, so a terminal, the terminals derived
/// from it and a host that lengthens the window while the machine is loaded
/// all see one setting; a change applies to the next foreground wait.
#[derive(Debug, Clone)]
pub struct ForegroundWait {
    shared: Arc<Shared>,
    scale: u32,
}

#[derive(Debug)]
struct Shared {
    normal_nanos: AtomicU64,
    /// The longer window in force, or `0` when the normal one applies.
    extended_nanos: AtomicU64,
}

fn nanos(window: Duration) -> u64 {
    u64::try_from(window.as_nanos()).expect("a foreground window fits in u64 nanoseconds")
}

impl ForegroundWait {
    /// A window of `normal` with no longer window in force.
    #[must_use]
    pub fn new(normal: Duration) -> Self {
        Self {
            shared: Arc::new(Shared {
                normal_nanos: AtomicU64::new(nanos(normal)),
                extended_nanos: AtomicU64::new(0),
            }),
            scale: 1,
        }
    }

    /// Derives a handle sharing the same live base and extended windows, scaled
    /// by `factor` (clamped to at least `1`).
    ///
    /// Live updates via [`Self::set_normal`], [`Self::extend`] and
    /// [`Self::restore`] on the parent handle apply immediately with `factor`
    /// multiplied in.
    #[must_use]
    pub fn scaled(&self, factor: u32) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            scale: self.scale.saturating_mul(factor.max(1)),
        }
    }

    /// Changes the unscaled normal window for every handle sharing this wait.
    pub fn set_normal(&self, normal: Duration) {
        self.shared
            .normal_nanos
            .store(nanos(normal), Ordering::Relaxed);
    }

    /// The normal window for this handle, whether or not a longer one is in force.
    #[must_use]
    pub fn normal(&self) -> Duration {
        Duration::from_nanos(self.shared.normal_nanos.load(Ordering::Relaxed))
            .saturating_mul(self.scale)
    }

    /// Puts a longer unscaled window in force until [`Self::restore`]. A zero
    /// `window` leaves the normal one in force.
    pub fn extend(&self, window: Duration) {
        self.shared
            .extended_nanos
            .store(nanos(window), Ordering::Relaxed);
    }

    /// Ends the longer window; the normal one applies again.
    pub fn restore(&self) {
        self.shared.extended_nanos.store(0, Ordering::Relaxed);
    }

    /// The window a foreground wait that starts now uses.
    #[must_use]
    pub fn get(&self) -> Duration {
        match self.shared.extended_nanos.load(Ordering::Relaxed) {
            0 => self.normal(),
            extended => Duration::from_nanos(extended).saturating_mul(self.scale),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_longer_window_applies_until_restored() {
        let wait = ForegroundWait::new(Duration::from_secs(10));
        let shared = wait.clone();
        assert_eq!(wait.get(), Duration::from_secs(10));
        shared.extend(Duration::from_secs(60));
        assert_eq!(wait.get(), Duration::from_secs(60));
        assert_eq!(wait.normal(), Duration::from_secs(10));
        shared.restore();
        assert_eq!(wait.get(), Duration::from_secs(10));
    }

    #[test]
    fn scaled_view_multiplies_live_normal_and_extended_windows() {
        let base = ForegroundWait::new(Duration::from_secs(10));
        let non_interactive = base.scaled(5);
        assert_eq!(base.get(), Duration::from_secs(10));
        assert_eq!(non_interactive.get(), Duration::from_secs(50));

        base.extend(Duration::from_secs(60));
        assert_eq!(base.get(), Duration::from_secs(60));
        assert_eq!(non_interactive.get(), Duration::from_secs(300));
        assert_eq!(non_interactive.normal(), Duration::from_secs(50));

        base.set_normal(Duration::from_secs(12));
        base.restore();
        assert_eq!(base.get(), Duration::from_secs(12));
        assert_eq!(non_interactive.get(), Duration::from_secs(60));
    }
}
