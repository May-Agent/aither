//! A shell's jobs: the handle that lists, reads, feeds, waits for and kills
//! them.

use std::sync::Arc;

use aither_core::llm::{CommandOutcome, CommandResult, CommandStream, Interruption};

use crate::LaunchId;
use crate::events::ShellEvents;
use crate::job_registry::{
    JobInfo, JobInputError, JobRead, JobRegistry, JobStatus, JobWatch, KillCause, WeakLaunchLease,
};
use crate::output::{MediaResolution, OutputFormat, OutputStore};
use crate::pipeline::{PipelineReadError, PipelineSnapshot, StageStreamSelector, StageUiDelta};
use crate::spawn::PtyResizeError;
use crate::wording::CommandKilled;

/// How a script ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunOutcome {
    /// It ran to its end.
    Exited {
        /// Its exit code.
        code: i32,
    },
    /// It was killed, and why.
    Killed(KillCause),
}

/// What a script printed and how it ended, as data: the embedder words it
/// where it hands a result to its model.
#[derive(Debug, Clone)]
pub struct ScriptResult {
    /// What the script wrote to its stdout.
    pub stdout: CommandStream,
    /// What it wrote to its stderr, when it wrote any.
    pub stderr: Option<CommandStream>,
    /// How it ended.
    pub outcome: RunOutcome,
    /// The sleep and offline intervals it ran through.
    pub interruptions: Vec<Interruption>,
}

impl ScriptResult {
    /// The result a tool returns for the script of process `pid`, a kill
    /// worded by the store's wording.
    pub(crate) fn into_command_result(self, pid: u32, store: &OutputStore) -> CommandResult {
        let outcome = match self.outcome {
            RunOutcome::Exited { code } => CommandOutcome::Exited { code },
            RunOutcome::Killed(cause) => CommandOutcome::Killed {
                pid,
                notice: store.wording().command_killed(&CommandKilled::of(&cause)),
            },
        };
        CommandResult {
            stdout: self.stdout,
            stderr: self.stderr,
            outcome,
            retained_pid: None,
            interruptions: self.interruptions,
            notice: None,
        }
    }
}

/// Why waiting for a job did not end in a result.
#[derive(Debug, thiserror::Error)]
pub enum JobWaitError {
    /// The job failed rather than exited.
    #[error("process {pid} failed: {error}")]
    Failed {
        /// Process ID of the job.
        pid: u32,
        /// What went wrong.
        error: String,
    },
    /// The job's output could not be stored.
    #[error("process {pid}'s output could not be stored: {source}")]
    Output {
        /// Process ID of the job.
        pid: u32,
        /// The store's error.
        source: std::io::Error,
    },
}

/// The jobs of one shell: every script that outlived its call, was started
/// in the background or watched by a monitor.
///
/// Cloning shares the shell's jobs.
#[derive(Debug, Clone)]
pub struct Jobs {
    pub(crate) registry: JobRegistry,
    events: ShellEvents,
    outputs: OutputStore,
}

impl Jobs {
    pub(crate) const fn new(
        registry: JobRegistry,
        events: ShellEvents,
        outputs: OutputStore,
    ) -> Self {
        Self {
            registry,
            events,
            outputs,
        }
    }

    /// Every job the shell holds, running or ended and not yet taken.
    pub async fn list(&self) -> Vec<JobInfo> {
        self.registry.list().await
    }

    /// The job with `pid`, when the shell holds it.
    ///
    /// The returned handle pins that specific job instance so later OS PID
    /// reuse never redirects this handle onto a different process.
    pub async fn get(&self, pid: u32) -> Option<Job> {
        let watch = self.registry.watch(pid).await?;
        Some(Job::new(self.clone(), watch))
    }

    /// A way to append to the output of `launch`; `None` when the shell holds
    /// no such launch.
    pub async fn output_appender(&self, launch: LaunchId) -> Option<JobOutputAppender> {
        Some(JobOutputAppender {
            registry: self.registry.clone(),
            launch: self.registry.launch_lease(launch).await?,
        })
    }

    /// Kills the job with `pid` along with its whole process tree; `false`
    /// when no running job has that PID.
    pub async fn kill(&self, pid: u32) -> bool {
        self.registry.kill_by_pid(pid).await
    }

    /// Kills every running job and returns how many it killed.
    pub async fn kill_all(&self) -> usize {
        self.registry.kill_running().await
    }

    /// Writes `bytes` to the PTY master of the running PTY job with `pid`.
    ///
    /// # Errors
    /// Returns [`JobInputError`] when `pid` is unknown, not a PTY job, not
    /// running, or its input channel closed.
    pub async fn pty_input(&self, pid: u32, bytes: Vec<u8>) -> Result<(), JobInputError> {
        self.registry.pty_input(pid, bytes).await
    }

    /// Resizes the running PTY job with `pid` to `cols` by `rows`.
    ///
    /// # Errors
    /// Returns [`PtyResizeError`] when the dimensions are zero, `pid` is
    /// unknown, not a PTY job, not running, or the backend resize fails.
    pub async fn resize(&self, pid: u32, cols: u16, rows: u16) -> Result<(), PtyResizeError> {
        self.registry.resize_by_pid(pid, cols, rows).await
    }

    /// The facts raised about the shell's jobs.
    #[must_use]
    pub fn events(&self) -> ShellEvents {
        self.events.clone()
    }

    /// Every pipeline currently visible for live UI streaming or post-completion
    /// stage inspection.
    ///
    /// Foreground pipelines that have not yet run for `pipeline_record_after`
    /// are excluded so fast pipelines never flash staged UI.
    pub async fn pipelines(&self) -> Vec<PipelineSnapshot> {
        self.registry.list_pipelines().await
    }

    /// Snapshot of the pipeline for `pid`, running or within its post-exit
    /// retention window.
    ///
    /// # Errors
    /// Returns [`PipelineReadError`] when `pid` is unknown, had no tapped
    /// pipeline, or expired after its retention window.
    pub async fn pipeline(&self, pid: u32) -> Result<PipelineSnapshot, PipelineReadError> {
        self.registry.pipeline_by_pid(pid).await
    }

    /// Reads the full native or terminal-rendered output of `step` for `pid`.
    ///
    /// When `raw` is `false`, terminal control sequences and `\r` overwrites
    /// are rendered to plain text; when `true`, the stage's native byte stream
    /// (with harness-injected SGR stripped and command-authored ANSI kept) is
    /// returned verbatim.
    ///
    /// # Errors
    /// Returns [`PipelineReadError`] when `pid` is unknown, expired, or `step`
    /// was not retained.
    pub async fn read_pipeline_stage(
        &self,
        pid: u32,
        step: usize,
        selector: StageStreamSelector,
        raw: bool,
    ) -> Result<Vec<u8>, PipelineReadError> {
        self.registry
            .read_pipeline_stage(pid, step, selector, raw)
            .await
    }

    /// Incremental terminal-rendered stage read for `terminal_read`, tracking
    /// unchanged polls per `(launch, step)`.
    ///
    /// # Errors
    /// Returns [`PipelineReadError`] when `pid` is unknown, expired, or `step`
    /// was not retained.
    pub async fn read_pipeline_stage_delta(
        &self,
        pid: u32,
        step: usize,
        cursor: usize,
        max_bytes: usize,
    ) -> Result<JobRead, PipelineReadError> {
        self.registry
            .read_pipeline_stage_delta(pid, step, cursor, max_bytes, true)
            .await
    }

    /// Incremental UI-colored stage read (preserving ANSI colors), optionally
    /// pinned to `launch` so historical cards never collide on OS PID reuse.
    ///
    /// # Errors
    /// Returns [`PipelineReadError`] when `pid`/`launch` is unknown, expired,
    /// or `step` was not retained.
    pub async fn read_pipeline_stage_ui(
        &self,
        pid: u32,
        launch: Option<LaunchId>,
        step: usize,
        cursor: usize,
        max_bytes: usize,
    ) -> Result<StageUiDelta, PipelineReadError> {
        self.registry
            .read_pipeline_stage_ui(pid, launch, step, cursor, max_bytes)
            .await
    }
}

/// Whether [`JobOutputAppender::append`] reached its job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum AppendOutcome {
    /// The bytes are in the job's output.
    Appended,
    /// The job ended, was forgotten or is another job than the appender's;
    /// the bytes went nowhere.
    JobEnded,
}

/// Appends to the output of one running job.
#[derive(Debug, Clone)]
pub struct JobOutputAppender {
    registry: JobRegistry,
    launch: WeakLaunchLease,
}

impl JobOutputAppender {
    /// Appends `bytes` to the job's output, returning once they are in it.
    pub async fn append(&self, bytes: &[u8]) -> AppendOutcome {
        let Some(launch) = self.launch.upgrade() else {
            return AppendOutcome::JobEnded;
        };
        if self
            .registry
            .append_task_stdout(&launch, bytes.to_vec())
            .await
        {
            AppendOutcome::Appended
        } else {
            AppendOutcome::JobEnded
        }
    }
}

/// One job of a shell.
///
/// Cloning shares the hold on that exact job instance.
#[derive(Debug, Clone)]
pub struct Job {
    jobs: Jobs,
    watch: Arc<JobWatch>,
}

impl Job {
    pub(crate) fn new(jobs: Jobs, watch: JobWatch) -> Self {
        Self {
            jobs,
            watch: Arc::new(watch),
        }
    }

    /// Process ID of the job inside its execution environment.
    #[must_use]
    pub fn pid(&self) -> u32 {
        self.watch.pid()
    }

    pub(crate) fn launch(&self) -> LaunchId {
        self.watch.launch()
    }

    /// The job as the shell holds it now.
    ///
    /// # Panics
    /// Panics when the shell no longer holds the job, which a handle's own
    /// hold on it prevents.
    pub async fn info(&self) -> JobInfo {
        self.jobs
            .registry
            .get_by_launch(self.watch.launch())
            .await
            .expect("the job this handle watches is held")
    }

    /// Waits for the job to end and returns what it printed and how it ended.
    ///
    /// # Errors
    /// Returns an error when the job failed or its output could not be stored.
    ///
    /// # Panics
    /// Panics when the shell no longer holds the job, which a handle's own
    /// hold on it prevents.
    pub async fn wait(&self) -> Result<ScriptResult, JobWaitError> {
        let launch = self.watch.launch();
        let pid = self.watch.pid();
        let ended = self
            .jobs
            .registry
            .wait_taking(launch)
            .await
            .expect("the job this handle watches is held");
        let outcome = match ended.status {
            JobStatus::Running => unreachable!("a wait ends once the job stopped running"),
            JobStatus::Completed { exit_code, .. } => RunOutcome::Exited { code: exit_code },
            JobStatus::Failed { error, .. } => {
                return Err(JobWaitError::Failed { pid, error });
            }
            JobStatus::Killed { cause, .. } => RunOutcome::Killed(cause),
        };
        let (stdout, stderr) = self
            .jobs
            .registry
            .terminal_output(launch)
            .await
            .expect("the job this handle watches is held");
        let outputs = &self.jobs.outputs;
        let output = |source| JobWaitError::Output { pid, source };
        let stdout = match &ended.output_path {
            Some(path) => outputs.stream_of_file(path).await,
            None => {
                outputs
                    .save(&stdout, OutputFormat::Text, MediaResolution::Auto)
                    .await
            }
        }
        .map_err(output)?;
        let stderr = outputs
            .save(&stderr, OutputFormat::Text, MediaResolution::Auto)
            .await
            .map_err(output)?;
        let stderr = (!stderr.is_empty()).then_some(stderr);
        Ok(ScriptResult {
            stdout,
            stderr,
            outcome,
            interruptions: Vec::new(),
        })
    }

    /// Reads the job's output past `cursor`, at most `max_bytes` of it.
    ///
    /// # Errors
    /// Returns an error when the shell does not hold the job.
    pub async fn read(&self, cursor: usize, max_bytes: usize) -> Result<JobRead, String> {
        self.jobs
            .registry
            .terminal_read_launch(self.watch.launch(), cursor, max_bytes)
            .await
    }

    /// Writes `bytes` to the job's stdin.
    ///
    /// # Errors
    /// Returns an error when the job is not running, is a PTY job, or takes no
    /// input.
    pub async fn input(&self, bytes: Vec<u8>) -> Result<(), JobInputError> {
        self.jobs
            .registry
            .terminal_input_launch(self.watch.launch(), self.watch.pid(), bytes)
            .await
    }

    /// Writes `bytes` to a PTY job's master stream.
    ///
    /// # Errors
    /// Returns [`JobInputError`] when the job is not a PTY job, is not
    /// running, or its input channel closed.
    pub async fn pty_input(&self, bytes: Vec<u8>) -> Result<(), JobInputError> {
        self.jobs
            .registry
            .pty_input_launch(self.watch.launch(), self.watch.pid(), bytes)
            .await
    }

    /// Resizes a PTY job's terminal window to `cols` by `rows`.
    ///
    /// # Errors
    /// Returns [`PtyResizeError`] when the dimensions are zero, the job is not
    /// a PTY job, is not running, or the backend resize fails.
    pub async fn resize(&self, cols: u16, rows: u16) -> Result<(), PtyResizeError> {
        self.jobs
            .registry
            .resize_launch(self.watch.launch(), self.watch.pid(), cols, rows)
            .await
    }

    /// Kills the job with its whole process tree; `false` when it is not
    /// running.
    pub async fn kill(&self) -> bool {
        self.jobs.registry.kill_launch(self.watch.launch()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job_registry::{LaunchLease, job_registry_channel};
    use crate::policy::SandboxMode;
    use executor_core::Executor;
    use executor_core::Task;
    use executor_core::tokio::TokioGlobal;

    fn spawn_registry() -> JobRegistry {
        let (registry, service) = job_registry_channel(crate::output_policy::fixture::policy());
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();
        registry
    }

    async fn appender(registry: &JobRegistry, launch: LaunchId) -> JobOutputAppender {
        JobOutputAppender {
            registry: registry.clone(),
            launch: registry
                .launch_lease(launch)
                .await
                .expect("the job is held"),
        }
    }

    #[tokio::test]
    async fn appended_lines_interleave_with_the_jobs_own_output() {
        let registry = spawn_registry();
        let river = LaunchLease::fresh();
        registry
            .register(10, &river, "exec", "run", SandboxMode::Sandboxed, None)
            .await;
        let appender = appender(&registry, river.id()).await;

        registry.append_stdout(&river, b"one\n".to_vec()).await;
        assert_eq!(
            appender.append(b"progress\n").await,
            AppendOutcome::Appended
        );
        registry.append_stdout(&river, b"two\n".to_vec()).await;

        let (stdout, _) = registry
            .terminal_output(river.id())
            .await
            .expect("job exists");
        assert_eq!(stdout, b"one\nprogress\ntwo\n");
        let JobRead::Delta(delta) = registry
            .terminal_read(10, 0, 1024)
            .await
            .expect("job exists")
        else {
            panic!("a first read is a delta");
        };
        assert_eq!(delta.bytes, b"one\nprogress\ntwo\n");
    }

    #[tokio::test]
    async fn a_launching_job_takes_appends_ahead_of_its_registration() {
        let registry = spawn_registry();
        let river = LaunchLease::fresh();
        registry.launching(&river).await;
        let appender = appender(&registry, river.id()).await;

        let append = tokio::spawn(async move { appender.append(b"early\n").await });
        tokio::task::yield_now().await;
        assert!(!append.is_finished());

        registry
            .register(11, &river, "exec", "run", SandboxMode::Sandboxed, None)
            .await;
        assert_eq!(append.await.unwrap(), AppendOutcome::Appended);
        registry.append_stdout(&river, b"late\n".to_vec()).await;
        let (stdout, _) = registry
            .terminal_output(river.id())
            .await
            .expect("job exists");
        assert_eq!(stdout, b"early\nlate\n");
    }

    #[tokio::test]
    async fn a_launch_that_never_registers_releases_its_appenders() {
        let registry = spawn_registry();
        let river = LaunchLease::fresh();
        registry.launching(&river).await;
        let appender = appender(&registry, river.id()).await;

        let waiting = {
            let appender = appender.clone();
            tokio::spawn(async move { appender.append(b"early\n").await })
        };
        tokio::task::yield_now().await;
        registry.abandon_launch(&river).await;

        assert_eq!(waiting.await.unwrap(), AppendOutcome::JobEnded);
        assert_eq!(appender.append(b"more\n").await, AppendOutcome::JobEnded);
        assert!(registry.launch_lease(river.id()).await.is_none());
    }

    #[tokio::test]
    async fn an_appender_never_reaches_a_later_job_under_a_reused_pid() {
        let registry = spawn_registry();
        let first = LaunchLease::fresh();
        registry
            .register(77, &first, "exec", "run", SandboxMode::Sandboxed, None)
            .await;
        let appender = appender(&registry, first.id()).await;
        registry.complete(&first, 0, None).await;
        drop(registry.owe_completion(first.id(), Ok(0)).await.unwrap());
        drop(first);
        assert!(registry.list().await.is_empty());

        let second = LaunchLease::fresh();
        registry
            .register(77, &second, "exec", "run", SandboxMode::Sandboxed, None)
            .await;
        assert_eq!(appender.append(b"stale\n").await, AppendOutcome::JobEnded);
        let (stdout, _) = registry
            .terminal_output(second.id())
            .await
            .expect("job exists");
        assert_eq!(stdout, [] as [u8; 0]);
    }
}
