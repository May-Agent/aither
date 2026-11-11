//! Environment added to a process at each launch.

use aither_core::llm::CallIdentity;

use crate::SandboxMode;

/// What a launch is, as the shell that starts it knows it.
///
/// It names the job the process runs as, the model call that asked for it and
/// the mode it runs in. A
/// [`SpawnSpec`](crate::SpawnSpec) carries it, and an [`EnvSource`] reads it
/// to decide the environment of that one process.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct LaunchInfo {
    /// The id of the job the process runs as, when it runs as one.
    pub task_id: Option<String>,
    /// The tool call that started it, when a model call did.
    pub call: Option<CallIdentity>,
    /// Whether it runs inside the sandbox or outside it.
    pub mode: SandboxMode,
}

impl LaunchInfo {
    /// The launch of job `task_id` in `mode`, started by `call`.
    #[must_use]
    pub fn new(task_id: impl Into<String>, call: Option<CallIdentity>, mode: SandboxMode) -> Self {
        Self {
            task_id: Some(task_id.into()),
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
