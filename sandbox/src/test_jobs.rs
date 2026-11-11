//! Jobs placed where a test needs them.
//!
//! A shell starts its jobs itself, and only on the host. An embedder that
//! tests how it treats a job in a container or on a remote host places one
//! on a [`Jobs`] handle here, and reads and kills it through the ordinary
//! [`Jobs::list`], [`Jobs::get`] and [`Jobs::kill`]. Dependent crates reach
//! it through the `test-util` feature.

use std::path::PathBuf;

use aither_core::{AnyIdAllocator, IdLease, WordAllocator};
use async_channel::Sender;
use executor_core::{Executor, Task, tokio::TokioGlobal};

use crate::job_registry::{JobPlace, job_registry_channel};
use crate::output::OutputStore;
use crate::output_policy::fixture;
use crate::policy::SandboxMode;
use crate::test_wording::TestWording;
use crate::{Interpreter, Jobs, ShellEvents};

/// A [`Jobs`] handle of a shell that starts nothing: it holds the jobs
/// [`Jobs::place_for_test`] places, and writes the output files of its
/// store into `dir`.
///
/// Its registry runs on the Tokio runtime the caller is in.
///
/// # Errors
///
/// Returns an error if `dir` cannot be created.
pub async fn standalone_jobs(dir: impl Into<PathBuf>) -> std::io::Result<Jobs> {
    let (registry, service) = job_registry_channel(fixture::policy());
    TokioGlobal
        .spawn(async move { service.serve().await })
        .detach();
    let outputs = OutputStore::new(
        dir,
        AnyIdAllocator::new(WordAllocator::new(["output"])),
        fixture::policy(),
        TestWording::any(),
    )
    .await?;
    Ok(Jobs::new(registry, ShellEvents::new(), outputs))
}

impl Jobs {
    /// Holds a running job named by `task_id` that runs `script` at `place`
    /// as process `pid`, the process id of its place.
    ///
    /// With a `kill_switch`, [`Jobs::kill`] signals it instead of the
    /// process, as it does a container job, and the job ends killed. Without
    /// one, [`Jobs::kill`] signals `pid` on this machine, as it does the
    /// local process standing for a remote job.
    pub async fn place_for_test(
        &self,
        place: JobPlace,
        pid: u32,
        task_id: &IdLease,
        script: &str,
        kill_switch: Option<Sender<()>>,
    ) {
        self.registry
            .register_at(
                place,
                pid,
                task_id,
                "exec",
                script,
                Interpreter::Shell,
                SandboxMode::Sandboxed,
            )
            .await;
        if let Some(kill_switch) = kill_switch {
            self.registry.attach_kill_switch(task_id, kill_switch).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job_registry::JobStatus;
    use crate::test_ids::lease;

    #[tokio::test]
    async fn a_container_job_is_listed_and_killed_through_its_switch() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = standalone_jobs(dir.path()).await.unwrap();
        let (tx, rx) = async_channel::bounded(1);
        jobs.place_for_test(
            JobPlace::Container,
            4_000_000,
            &lease("box"),
            "run",
            Some(tx),
        )
        .await;

        let listed = jobs.list().await;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].place, JobPlace::Container);
        assert_eq!(listed[0].status, JobStatus::Running);

        assert!(jobs.get("box").await.expect("the job is held").kill().await);
        assert!(rx.try_recv().is_ok(), "the kill switch was pulled");
        assert!(matches!(
            jobs.list().await[0].status,
            JobStatus::Killed { .. }
        ));
    }

    #[tokio::test]
    async fn a_remote_job_without_a_switch_is_listed_at_its_place() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = standalone_jobs(dir.path()).await.unwrap();
        jobs.place_for_test(JobPlace::Remote, 4_000_001, &lease("far"), "run", None)
            .await;

        let listed = jobs.list().await;
        assert_eq!(listed[0].place, JobPlace::Remote);
        assert_eq!(listed[0].status, JobStatus::Running);
    }
}
