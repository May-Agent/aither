//! Environment added to a process at each launch.

/// Supplies environment variables resolved at each launch.
///
/// Implemented by embedders that need values decided at spawn time (secrets
/// read from a live store, the address of an IPC bridge) rather than when
/// the sandbox is built. `Vec<(String, String)>` implements it for static
/// pairs. A source's values override the sandbox's computed environment and
/// are themselves overridden by a [`SpawnSpec`](crate::SpawnSpec)'s `env`.
pub trait EnvSource: std::fmt::Debug + Send + Sync {
    /// `(name, value)` pairs injected into the spawned process's environment.
    fn env(&self) -> Vec<(String, String)>;
}

impl EnvSource for Vec<(String, String)> {
    fn env(&self) -> Vec<(String, String)> {
        self.clone()
    }
}

impl<T: EnvSource + ?Sized> EnvSource for std::sync::Arc<T> {
    fn env(&self) -> Vec<(String, String)> {
        (**self).env()
    }
}
