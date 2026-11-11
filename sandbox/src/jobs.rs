//! A shell's jobs: the handle that lists, reads, feeds, waits for and kills
//! them.

use std::sync::Arc;

use aither_core::WeakIdLease;
use aither_core::llm::{CommandOutcome, CommandResult, CommandStream, Interruption};

use crate::events::ShellEvents;
use crate::job_registry::{
    JobInfo, JobInputError, JobRead, JobRegistry, JobStatus, JobWatch, KillCause,
};
use crate::output::{MediaResolution, OutputFormat, OutputStore};
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
    /// The result a tool returns for the script of task `task_id`, a kill
    /// worded by the store's wording.
    pub(crate) fn into_command_result(self, task_id: &str, store: &OutputStore) -> CommandResult {
        let outcome = match self.outcome {
            RunOutcome::Exited { code } => CommandOutcome::Exited { code },
            RunOutcome::Killed(cause) => CommandOutcome::Killed {
                task_id: task_id.to_owned(),
                notice: store.wording().command_killed(&CommandKilled::of(&cause)),
            },
        };
        CommandResult {
            stdout: self.stdout,
            stderr: self.stderr,
            outcome,
            interruptions: self.interruptions,
        }
    }
}

/// Why waiting for a job did not end in a result.
#[derive(Debug, thiserror::Error)]
pub enum JobWaitError {
    /// The job failed rather than exited.
    #[error("job {task_id} failed: {error}")]
    Failed {
        /// The job.
        task_id: String,
        /// What went wrong.
        error: String,
    },
    /// The job's output could not be stored.
    #[error("job {task_id}'s output could not be stored: {source}")]
    Output {
        /// The job.
        task_id: String,
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

    /// The job named `task_id`, when the shell holds it.
    ///
    /// The shell keeps the job, and its name, for as long as the handle
    /// lives, so the handle never addresses another job that is later given
    /// the same name.
    pub async fn get(&self, task_id: &str) -> Option<Job> {
        let watch = self.registry.watch(task_id).await?;
        Some(Job::new(self.clone(), task_id.to_owned(), watch))
    }

    /// A way to append to the output of the job named `task_id`; `None`
    /// when the shell holds no such job. The shell holds a job from before
    /// its process starts, so the `task_id` of a [`crate::LaunchInfo`] it
    /// handed out always names one while the process can run; what is
    /// appended before the job registers lands ahead of the process's own
    /// output.
    ///
    /// The appender refers to that job alone: it keeps neither the job nor
    /// its name, and never writes into a later job given the same name.
    pub async fn output_appender(&self, task_id: &str) -> Option<JobOutputAppender> {
        Some(JobOutputAppender {
            registry: self.registry.clone(),
            task_id: self.registry.task_lease(task_id).await?,
        })
    }

    /// Kills the job named `task_id` with its whole process tree; `false`
    /// when no running job has that name.
    pub async fn kill(&self, task_id: &str) -> bool {
        self.registry.kill_by_task_id(task_id).await
    }

    /// Kills every running job and returns how many it killed.
    pub async fn kill_all(&self) -> usize {
        self.registry.kill_running().await
    }

    /// The facts raised about the shell's jobs.
    #[must_use]
    pub fn events(&self) -> ShellEvents {
        self.events.clone()
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

/// Appends to the output of one running job, such as the progress an
/// embedder's command reports while the script that called it runs.
///
/// What it appends joins the job's own output in arrival order, so
/// [`Job::read`], the job's stored output and the shell's output events see
/// it. Cloning shares the reference to the job, which is weak.
#[derive(Debug, Clone)]
pub struct JobOutputAppender {
    registry: JobRegistry,
    task_id: WeakIdLease,
}

impl JobOutputAppender {
    /// Appends `bytes` to the job's output, returning once they are in it.
    pub async fn append(&self, bytes: &[u8]) -> AppendOutcome {
        // The lease is held until the registry answers, so its word cannot
        // name another job meanwhile.
        let Some(task_id) = self.task_id.upgrade() else {
            return AppendOutcome::JobEnded;
        };
        if self
            .registry
            .append_task_stdout(&task_id, bytes.to_vec())
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
/// Cloning shares the hold on the job.
#[derive(Debug, Clone)]
pub struct Job {
    task_id: String,
    jobs: Jobs,
    /// Keeps the job, and the word naming it, in the shell's registry.
    _watch: Arc<JobWatch>,
}

impl Job {
    pub(crate) fn new(jobs: Jobs, task_id: String, watch: JobWatch) -> Self {
        Self {
            task_id,
            jobs,
            _watch: Arc::new(watch),
        }
    }

    /// The word the job is named by.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// The job as the shell holds it now.
    ///
    /// # Panics
    /// Panics when the shell no longer holds the job, which a handle's own
    /// hold on it prevents.
    pub async fn info(&self) -> JobInfo {
        self.jobs
            .registry
            .get_by_task_id(&self.task_id)
            .await
            .expect("the job this handle watches is held")
    }

    /// Waits for the job to end and returns what it printed and how it
    /// ended. The wait takes the job's completion: a job read here owes its
    /// agent no turn, so the end it raises carries no
    /// [pending completion](crate::JobEnded::into_parts).
    ///
    /// # Errors
    /// Returns an error when the job failed or its output could not be
    /// stored.
    ///
    /// # Panics
    /// Panics when the shell no longer holds the job, which a handle's own
    /// hold on it prevents.
    pub async fn wait(&self) -> Result<ScriptResult, JobWaitError> {
        let ended = self
            .jobs
            .registry
            .wait_taking(&self.task_id)
            .await
            .expect("the job this handle watches is held");
        let outcome = match ended.status {
            JobStatus::Running => unreachable!("a wait ends once the job stopped running"),
            JobStatus::Completed { exit_code, .. } => RunOutcome::Exited { code: exit_code },
            JobStatus::Failed { error, .. } => {
                return Err(JobWaitError::Failed {
                    task_id: self.task_id.clone(),
                    error,
                });
            }
            JobStatus::Killed { cause, .. } => RunOutcome::Killed(cause),
        };
        let (stdout, stderr) = self
            .jobs
            .registry
            .terminal_output(&self.task_id)
            .await
            .expect("the job this handle watches is held");
        let outputs = &self.jobs.outputs;
        let output = |source| JobWaitError::Output {
            task_id: self.task_id.clone(),
            source,
        };
        // The output the job already stored is the job's; a new file is
        // written only for what it did not store.
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

    /// Reads the job's output past `cursor`, at most `max_bytes` of it; a
    /// read that would return what the job's last read returned is
    /// [`JobRead::Unchanged`].
    ///
    /// # Errors
    /// Returns an error when the shell does not hold the job.
    pub async fn read(&self, cursor: usize, max_bytes: usize) -> Result<JobRead, String> {
        self.jobs
            .registry
            .terminal_read(&self.task_id, cursor, max_bytes)
            .await
    }

    /// Writes `bytes` to the job's stdin.
    ///
    /// # Errors
    /// Returns an error when the job is not running or takes no input: one
    /// started in the foreground has no stdin pipe, promoted or not.
    pub async fn input(&self, bytes: Vec<u8>) -> Result<(), JobInputError> {
        self.jobs
            .registry
            .terminal_input(&self.task_id, bytes)
            .await
    }

    /// Kills the job with its whole process tree; `false` when it is not
    /// running.
    pub async fn kill(&self) -> bool {
        self.jobs.kill(&self.task_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job_registry::job_registry_channel;
    use crate::policy::SandboxMode;
    use crate::test_ids::lease;
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

    async fn appender(registry: &JobRegistry, task_id: &str) -> JobOutputAppender {
        JobOutputAppender {
            registry: registry.clone(),
            task_id: registry.task_lease(task_id).await.expect("the job is held"),
        }
    }

    #[tokio::test]
    async fn appended_lines_interleave_with_the_jobs_own_output() {
        let registry = spawn_registry();
        registry
            .register(
                1,
                &lease("river"),
                "exec",
                "run",
                SandboxMode::Sandboxed,
                None,
            )
            .await;
        let appender = appender(&registry, "river").await;

        registry
            .append_stdout(&lease("river"), b"one\n".to_vec())
            .await;
        assert_eq!(
            appender.append(b"progress\n").await,
            AppendOutcome::Appended
        );
        registry
            .append_stdout(&lease("river"), b"two\n".to_vec())
            .await;

        let (stdout, _) = registry.terminal_output("river").await.expect("job exists");
        assert_eq!(stdout, b"one\nprogress\ntwo\n");
        let JobRead::Delta(delta) = registry
            .terminal_read("river", 0, 1024)
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
        let river = lease("river");
        registry.launching(&river).await;
        let appender = appender(&registry, "river").await;

        let append = tokio::spawn(async move { appender.append(b"early\n").await });
        tokio::task::yield_now().await;
        assert!(!append.is_finished(), "the append waits for the job");

        registry
            .register(1, &river, "exec", "run", SandboxMode::Sandboxed, None)
            .await;
        assert_eq!(append.await.unwrap(), AppendOutcome::Appended);
        registry.append_stdout(&river, b"late\n".to_vec()).await;
        let (stdout, _) = registry.terminal_output("river").await.expect("job exists");
        assert_eq!(stdout, b"early\nlate\n");
    }

    #[tokio::test]
    async fn a_launch_that_never_registers_releases_its_appenders() {
        let registry = spawn_registry();
        let river = lease("river");
        registry.launching(&river).await;
        let appender = appender(&registry, "river").await;

        let waiting = {
            let appender = appender.clone();
            tokio::spawn(async move { appender.append(b"early\n").await })
        };
        tokio::task::yield_now().await;
        registry.abandon_launch(&river).await;

        assert_eq!(waiting.await.unwrap(), AppendOutcome::JobEnded);
        assert_eq!(appender.append(b"more\n").await, AppendOutcome::JobEnded);
        assert!(registry.task_lease("river").await.is_none());
    }

    #[tokio::test]
    async fn a_held_appender_does_nothing_once_the_job_ended() {
        let registry = spawn_registry();
        registry
            .register(
                1,
                &lease("river"),
                "exec",
                "run",
                SandboxMode::Sandboxed,
                None,
            )
            .await;
        let appender = appender(&registry, "river").await;
        registry.complete(&lease("river"), 0, None).await;

        assert_eq!(appender.append(b"late\n").await, AppendOutcome::JobEnded);
        let (stdout, _) = registry
            .terminal_output("river")
            .await
            .expect("job is held");
        assert_eq!(stdout, b"");
    }

    #[tokio::test]
    async fn an_appender_never_reaches_a_later_job_under_the_same_word() {
        let ids = aither_core::AnyIdAllocator::new(aither_core::WordAllocator::new(["river"]));
        let registry = spawn_registry();
        let first = ids.lease().unwrap();
        registry
            .register(1, &first, "exec", "run", SandboxMode::Sandboxed, None)
            .await;
        let appender = appender(&registry, "river").await;
        registry.complete(&first, 0, None).await;
        drop(registry.owe_completion("river", Ok(0)).await.unwrap());
        drop(first);
        assert!(registry.list().await.is_empty(), "the job is forgotten");

        let second = ids.lease().expect("nothing keeps river");
        registry
            .register(2, &second, "exec", "run", SandboxMode::Sandboxed, None)
            .await;
        assert_eq!(appender.append(b"stale\n").await, AppendOutcome::JobEnded);
        let (stdout, _) = registry.terminal_output("river").await.expect("job exists");
        assert!(stdout.is_empty(), "the stale appender reached no job");
    }
}
