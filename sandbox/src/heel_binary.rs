//! The verified `heel` binary a terminal's sandboxes carry.

use std::sync::Arc;

use async_lock::OnceCell;
use heel::HeelBinary;

use crate::{HostProgramPath, ToolRuntimeError};

/// A terminal's `heel` binary, the one its tool distribution pins, shared by
/// the terminal, its clones and its subagents' terminals.
///
/// Initializing the terminal's tools provides it: the pinned executable is
/// verified, which spawns `heel --version` off the executor, and checked
/// against the version of the `heel` library this crate links. A terminal
/// whose tools were never initialized has none, and the command that needs a
/// sandbox reports that.
#[derive(Clone, Default)]
pub struct SharedHeelBinary(Arc<OnceCell<HeelBinary>>);

impl SharedHeelBinary {
    /// Verifies the distribution's `heel` and shares it from now on.
    ///
    /// # Errors
    /// Fails with [`ToolRuntimeError::Heel`] when the executable is missing,
    /// does not run, or is not the linked library's version.
    pub(crate) async fn provide(&self, heel: &HostProgramPath) -> Result<(), ToolRuntimeError> {
        let path = heel.path().to_path_buf();
        self.get_with(move || HeelBinary::verify(path))
            .await
            .map(drop)
            .map_err(ToolRuntimeError::Heel)
    }

    /// The provided binary.
    pub(crate) async fn get(&self) -> Result<HeelBinary, ToolRuntimeError> {
        if let Some(binary) = self.0.get() {
            return Ok(binary.clone());
        }
        self.unprovided().await
    }

    /// Without initialized tools, production has no binary; tests take the
    /// one the repository's distribution pins, as an embedder would.
    #[cfg(not(any(test, feature = "test-heel")))]
    fn unprovided(&self) -> std::future::Ready<Result<HeelBinary, ToolRuntimeError>> {
        std::future::ready(Err(ToolRuntimeError::NotInitialized(
            "the heel binary: initialize the terminal's tools".into(),
        )))
    }

    #[cfg(any(test, feature = "test-heel"))]
    async fn unprovided(&self) -> Result<HeelBinary, ToolRuntimeError> {
        self.0
            .get_or_try_init(crate::test_heel::pinned)
            .await
            .cloned()
    }

    /// The verified binary, produced by `verify` the first time.
    async fn get_with(
        &self,
        verify: impl FnOnce() -> heel::Result<HeelBinary> + Send + 'static,
    ) -> heel::Result<HeelBinary> {
        self.0
            .get_or_try_init(|| blocking::unblock(verify))
            .await
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A verification that counts its runs and yields `pinned`.
    fn counted(
        calls: &Arc<AtomicUsize>,
        pinned: &HeelBinary,
    ) -> impl FnOnce() -> heel::Result<HeelBinary> + use<> {
        let (calls, pinned) = (calls.clone(), pinned.clone());
        move || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(pinned)
        }
    }

    #[tokio::test]
    async fn verifies_once_for_every_clone() {
        let pinned = crate::test_heel::pinned().await.unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let shared = SharedHeelBinary::default();
        let first = shared.get_with(counted(&calls, &pinned)).await.unwrap();
        let second = shared
            .clone()
            .get_with(counted(&calls, &pinned))
            .await
            .unwrap();
        let third = shared.get_with(counted(&calls, &pinned)).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(first, second);
        assert_eq!(first, third);
    }

    #[tokio::test]
    async fn concurrent_first_use_verifies_once() {
        let pinned = crate::test_heel::pinned().await.unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let shared = SharedHeelBinary::default();
        let (a, (b, c)) = futures_lite::future::zip(
            shared.get_with(counted(&calls, &pinned)),
            futures_lite::future::zip(
                shared.clone().get_with(counted(&calls, &pinned)),
                shared.clone().get_with(counted(&calls, &pinned)),
            ),
        )
        .await;
        a.unwrap();
        b.unwrap();
        c.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_failure_is_reported_and_not_cached() {
        let pinned = crate::test_heel::pinned().await.unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let shared = SharedHeelBinary::default();
        let failing = |calls: Arc<AtomicUsize>| {
            move || {
                calls.fetch_add(1, Ordering::SeqCst);
                HeelBinary::verify("/nonexistent/heel")
            }
        };
        let error = shared.get_with(failing(calls.clone())).await.unwrap_err();
        assert!(error.to_string().contains("the heel binary is missing"));
        shared.get_with(failing(calls.clone())).await.unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        shared.get_with(counted(&calls, &pinned)).await.unwrap();
        shared.get_with(counted(&calls, &pinned)).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn the_distributions_heel_is_provided_and_shared() {
        let distribution = crate::test_heel::distribution();
        let target = crate::ToolTarget::native().unwrap();
        let cache = tempfile::tempdir().unwrap();
        let heel = distribution
            .prepare_host_program(crate::HostProgram::Heel, &target, cache.path())
            .await
            .unwrap();
        let shared = SharedHeelBinary::default();
        shared.provide(&heel).await.unwrap();
        assert_eq!(shared.clone().get().await.unwrap().as_path(), heel.path());
    }
}
