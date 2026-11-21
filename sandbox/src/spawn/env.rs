//! Environment added to a process at each launch.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use aither_core::llm::CallIdentity;

use crate::SandboxMode;

/// Monotonic identifier of one command launch within a shell.
///
/// Allocated before the OS process starts so pre-spawn policies, outage
/// tracking and IPC output appenders can refer to the launch before its OS PID
/// exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LaunchId(u64);

impl LaunchId {
    /// Allocates a fresh launch identifier.
    #[must_use]
    pub fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// Wraps a raw numeric launch identifier (for tests).
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Numeric value of this launch identifier.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for LaunchId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// What a launch is, as the shell that starts it knows it.
///
/// It names the launch the process runs as, the model call that asked for it
/// and the mode it runs in. A [`SpawnSpec`](crate::SpawnSpec) carries it, and
/// an [`EnvSource`] reads it to decide the environment of that one process.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct LaunchInfo {
    /// Identifier of the launch the process runs as, when started by a shell.
    pub launch: Option<LaunchId>,
    /// The tool call that started it, when a model call did.
    pub call: Option<CallIdentity>,
    /// Whether it runs inside the sandbox or outside it.
    pub mode: SandboxMode,
}

impl LaunchInfo {
    /// The launch `launch` in `mode`, started by `call`.
    #[must_use]
    pub const fn new(launch: LaunchId, call: Option<CallIdentity>, mode: SandboxMode) -> Self {
        Self {
            launch: Some(launch),
            call,
            mode,
        }
    }
}

/// Supplies environment variables resolved at each launch.
///
/// Implemented by embedders that need values decided at spawn time (secrets
/// read from a live store, the address of an IPC bridge, the id of the call
/// a command is run for) rather than when the sandbox is built.
/// `Vec<(String, String)>` implements it for static pairs. A source's values
/// override the sandbox's computed environment and are themselves overridden
/// by a [`SpawnSpec`](crate::SpawnSpec)'s `env`.
pub trait EnvSource: std::fmt::Debug + Send + Sync {
    /// `(name, value)` pairs injected into the environment of the process
    /// `launch` describes.
    fn env(&self, launch: &LaunchInfo) -> Vec<(String, String)>;
}

impl EnvSource for Vec<(String, String)> {
    fn env(&self, _launch: &LaunchInfo) -> Vec<(String, String)> {
        self.clone()
    }
}

impl<T: EnvSource + ?Sized> EnvSource for std::sync::Arc<T> {
    fn env(&self, launch: &LaunchInfo) -> Vec<(String, String)> {
        (**self).env(launch)
    }
}
