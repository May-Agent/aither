//! A shell's jobs: the handle that lists, reads, feeds, waits for and kills
//! them.

use aither_core::llm::{CommandOutcome, CommandResult, CommandStream};

use crate::events::ShellEvents;
use crate::job_registry::{JobInfo, JobRead, JobRegistry, JobStatus};
use crate::output::{MediaResolution, OutputFormat, OutputStore};
use crate::wording::CommandKilled;

/// What a job that ended returns to its waiter.
pub type ScriptResult = CommandResult;

/// Why waiting for a job did not end in a result.
#[derive(Debug, thiserror::Error)]
pub enum JobWaitError {
    /// The shell no longer holds the job: its completion was taken and
    /// nothing watched it.
    #[error("the shell no longer holds job {0}")]
    Gone(String),
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
    registry: JobRegistry,
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
    pub async fn get(&self, task_id: &str) -> Option<Job> {
        self.registry.get_by_task_id(task_id).await?;
        Some(Job {
            task_id: task_id.to_owned(),
            jobs: self.clone(),
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

/// One job of a shell.
#[derive(Debug, Clone)]
pub struct Job {
    task_id: String,
    jobs: Jobs,
}

impl Job {
    /// The word the job is named by.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// The job as the shell holds it now; `None` once its completion was
    /// taken and nothing watches it.
    pub async fn info(&self) -> Option<JobInfo> {
        self.jobs.registry.get_by_task_id(&self.task_id).await
    }

    /// Waits for the job to end and returns what it printed and how it
    /// ended; a job killed ends [`CommandOutcome::Killed`].
    ///
    /// # Errors
    /// Returns an error when the shell no longer holds the job, the job
    /// failed, or its output could not be stored.
    pub async fn wait(&self) -> Result<ScriptResult, JobWaitError> {
        let gone = || JobWaitError::Gone(self.task_id.clone());
        let ended = self
            .jobs
            .registry
            .wait_ended(&self.task_id)
            .await
            .ok_or_else(gone)?;
        let outcome = match &ended.status {
            JobStatus::Running => unreachable!("a wait ends once the job stopped running"),
            JobStatus::Completed { exit_code, .. } => CommandOutcome::Exited { code: *exit_code },
            JobStatus::Failed { error, .. } => {
                return Err(JobWaitError::Failed {
                    task_id: self.task_id.clone(),
                    error: error.clone(),
                });
            }
            JobStatus::Killed { .. } => CommandOutcome::Killed {
                notice: self
                    .jobs
                    .outputs
                    .wording()
                    .command_killed(&CommandKilled::BeforeTimeout),
                task_id: self.task_id.clone(),
            },
        };
        let (stdout, stderr) = self
            .jobs
            .registry
            .terminal_output(ended.pid)
            .await
            .ok_or_else(gone)?;
        let store = |bytes: Vec<u8>| async move {
            self.jobs
                .outputs
                .save(&bytes, OutputFormat::Text, MediaResolution::Auto)
                .await
                .map_err(|source| JobWaitError::Output {
                    task_id: self.task_id.clone(),
                    source,
                })
        };
        let stdout = store(stdout).await?;
        let stderr = store(stderr).await?;
        let stderr = match &stderr {
            CommandStream::Inline { text } if text.is_empty() => None,
            _ => Some(stderr),
        };
        Ok(CommandResult {
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
    /// Returns an error when the job is not running or takes no input.
    pub async fn input(&self, bytes: Vec<u8>) -> Result<(), String> {
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
