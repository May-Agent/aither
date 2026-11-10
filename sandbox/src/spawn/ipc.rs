//! The commands an embedder lets a shell's scripts call back into.

use std::fmt;
use std::sync::Arc;

use super::LaunchInfo;

/// Supplies the [`heel::IpcRouter`] a script of a shell can call back into
/// the host's commands through.
///
/// A shell asks once per process it starts on a backend that can serve a
/// router (the host and a container; a server has no way back to this
/// machine), with what that launch is, so the commands it serves may depend
/// on the job, the tool call and the mode. A subagent's shell asks the same
/// provider as the shell that launched it.
pub trait IpcProvider: Send + Sync + 'static {
    /// The commands a process launched as `launch` may call, or `None` for a
    /// process with no way back to the host.
    fn router(&self, launch: &LaunchInfo) -> Option<heel::IpcRouter>;
}

/// Object-safe twin of [`IpcProvider`].
trait IpcProviderImpl: Send + Sync {
    fn router(&self, launch: &LaunchInfo) -> Option<heel::IpcRouter>;
}

impl<T: IpcProvider> IpcProviderImpl for T {
    fn router(&self, launch: &LaunchInfo) -> Option<heel::IpcRouter> {
        IpcProvider::router(self, launch)
    }
}

/// A type-erased [`IpcProvider`].
#[derive(Clone)]
pub struct AnyIpcProvider(Arc<dyn IpcProviderImpl>);

impl AnyIpcProvider {
    /// Erases `provider`.
    #[must_use]
    pub fn new(provider: impl IpcProvider) -> Self {
        Self(Arc::new(provider))
    }

    /// The commands a process launched as `launch` may call.
    #[must_use]
    pub fn router(&self, launch: &LaunchInfo) -> Option<heel::IpcRouter> {
        self.0.router(launch)
    }
}

impl fmt::Debug for AnyIpcProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyIpcProvider")
    }
}

/// A provider for tests: it serves one command, `probe`, and records every
/// launch it was asked about.
#[cfg(test)]
pub mod fixture {
    use std::sync::{Arc, Mutex};

    use super::{IpcProvider, LaunchInfo};

    /// What `probe` answers.
    pub const ANSWER: &str = "probe_ok";

    struct Probe;

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct ProbeArgs {}

    impl heel::IpcCommand for Probe {
        fn name(&self) -> std::borrow::Cow<'static, str> {
            "probe".into()
        }
        type Args = ProbeArgs;
        type Response = String;
        fn handle(
            &self,
            _args: ProbeArgs,
            _context: heel::IpcContext,
        ) -> impl Future<Output = Result<String, heel::IpcError>> {
            std::future::ready(Ok(ANSWER.into()))
        }
    }

    #[derive(Clone, Default)]
    pub struct Recording {
        pub launches: Arc<Mutex<Vec<LaunchInfo>>>,
    }

    impl IpcProvider for Recording {
        fn router(&self, launch: &LaunchInfo) -> Option<heel::IpcRouter> {
            self.launches.lock().unwrap().push(launch.clone());
            Some(heel::IpcRouter::new().register(Probe))
        }
    }
}
