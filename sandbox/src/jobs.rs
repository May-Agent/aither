//! A shell's jobs: the handle that lists, reads, feeds, waits for and kills
//! them.

use std::sync::Arc;

use aither_core::llm::{CommandOutcome, CommandResult, CommandStream, Interruption};

use crate::events::ShellEvents;
use crate::job_registry::{JobInfo, JobRead, JobRegistry, JobStatus, JobWatch, KillCause};
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
    ///
    /// The shell keeps the job, and its name, for as long as the handle
    /// lives, so the handle never addresses another job that is later given
    /// the same name.
    pub async fn get(&self, task_id: &str) -> Option<Job> {
        let watch = self.registry.watch(task_id).await?;
        Some(Job::new(self.clone(), task_id.to_owned(), watch))
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
            .terminal_output(ended.pid)
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
