//! The control tools of a shell's jobs: read, input and kill.
//!
//! The embedder names and describes each; the texts they return are its
//! wording's.

use std::borrow::Cow;

use aither_core::llm::ToolContext;
use aither_core::llm::{Tool, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::jobs::Jobs;
use crate::wording::{AnyShellWording, JobInputSent, JobKillAsked};

/// What an embedder calls one of its tools and how it describes it to the
/// model; neither has a default.
#[derive(Debug, Clone)]
pub struct ToolIdentity {
    /// The tool's name.
    pub name: Cow<'static, str>,
    /// The tool's description, as the model reads it.
    pub description: Cow<'static, str>,
}

/// The identities of the three job control tools.
#[derive(Debug, Clone)]
pub struct JobToolSettings {
    /// The tool that reads a job's output.
    pub read: ToolIdentity,
    /// The tool that writes to a job's stdin.
    pub input: ToolIdentity,
    /// The tool that kills a job.
    pub kill: ToolIdentity,
}

fn task_id(argument: &str) -> aither_core::Result<&str> {
    let task_id = argument.trim();
    if task_id.is_empty() {
        return Err(anyhow::anyhow!("task_id must not be empty"));
    }
    Ok(task_id)
}

/// Tool for killing a job.
#[derive(Debug, Clone)]
pub struct JobKillTool {
    identity: ToolIdentity,
    jobs: Jobs,
    wording: AnyShellWording,
}

impl JobKillTool {
    /// A kill tool for `jobs`, named and described by `identity`, whose
    /// reply is `wording`'s.
    #[must_use]
    pub const fn new(identity: ToolIdentity, jobs: Jobs, wording: AnyShellWording) -> Self {
        Self {
            identity,
            jobs,
            wording,
        }
    }
}

/// Kill a job by its task id.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct JobKillArgs {
    /// Task identifier returned when the job was backgrounded.
    pub task_id: String,
}

impl Tool for JobKillTool {
    fn name(&self) -> Cow<'static, str> {
        self.identity.name.clone()
    }

    fn description(&self) -> Cow<'static, str> {
        self.identity.description.clone()
    }

    type Arguments = JobKillArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let task_id = task_id(&args.task_id)?;
        let killed = self.jobs.kill(task_id).await;
        Ok(ToolResult::text(
            self.wording.job_kill(&JobKillAsked { task_id, killed }),
        ))
    }
}

/// Tool for writing to a job's stdin.
#[derive(Debug, Clone)]
pub struct JobInputTool {
    identity: ToolIdentity,
    jobs: Jobs,
    wording: AnyShellWording,
}

impl JobInputTool {
    /// An input tool for `jobs`, named and described by `identity`, whose
    /// reply is `wording`'s.
    #[must_use]
    pub const fn new(identity: ToolIdentity, jobs: Jobs, wording: AnyShellWording) -> Self {
        Self {
            identity,
            jobs,
            wording,
        }
    }
}

/// Send input to a job that is waiting on stdin.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct JobInputArgs {
    /// Task identifier returned when the job was backgrounded.
    pub task_id: String,
    /// Raw bytes encoded as UTF-8 text written to the job's stdin.
    pub input: String,
    /// Append a trailing newline before writing (default true).
    #[serde(default = "default_append_newline")]
    pub append_newline: bool,
}

const fn default_append_newline() -> bool {
    true
}

impl Tool for JobInputTool {
    fn name(&self) -> Cow<'static, str> {
        self.identity.name.clone()
    }

    fn description(&self) -> Cow<'static, str> {
        self.identity.description.clone()
    }

    type Arguments = JobInputArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let task_id = task_id(&args.task_id)?;
        let job = self
            .jobs
            .get(task_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("unknown task_id: {task_id}"))?;

        let mut bytes = args.input.into_bytes();
        if args.append_newline {
            bytes.push(b'\n');
        }
        job.input(bytes).await.map_err(anyhow::Error::msg)?;

        Ok(ToolResult::text(
            self.wording.job_input(&JobInputSent { task_id }),
        ))
    }
}

/// Tool for reading a job's output as it grows.
#[derive(Debug, Clone)]
pub struct JobReadTool {
    identity: ToolIdentity,
    jobs: Jobs,
    wording: AnyShellWording,
    output: crate::OutputPolicy,
}

impl JobReadTool {
    /// A read tool for `jobs`, named and described by `identity`, whose
    /// replies are `wording`'s and whose reads default to the `read_bytes`
    /// of `output`.
    #[must_use]
    pub const fn new(
        identity: ToolIdentity,
        jobs: Jobs,
        wording: AnyShellWording,
        output: crate::OutputPolicy,
    ) -> Self {
        Self {
            identity,
            jobs,
            wording,
            output,
        }
    }
}

/// Read incremental output from a job.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct JobReadArgs {
    /// Task identifier returned when the job was backgrounded.
    pub task_id: String,
    /// Read only bytes after this cursor offset.
    #[serde(default)]
    pub cursor: usize,
    /// Maximum bytes to return in this call.
    #[serde(default)]
    pub max_bytes: Option<usize>,
}

impl Tool for JobReadTool {
    fn name(&self) -> Cow<'static, str> {
        self.identity.name.clone()
    }

    fn description(&self) -> Cow<'static, str> {
        self.identity.description.clone()
    }

    type Arguments = JobReadArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let task_id = task_id(&args.task_id)?;
        let max_bytes = args
            .max_bytes
            .unwrap_or_else(|| self.output.get().read_bytes);
        if max_bytes == 0 {
            return Err(anyhow::anyhow!("max_bytes must be greater than 0"));
        }
        let job = self
            .jobs
            .get(task_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("unknown task_id: {task_id}"))?;
        let read = job
            .read(args.cursor, max_bytes)
            .await
            .map_err(anyhow::Error::msg)?;
        Ok(ToolResult::text(self.wording.job_read(&read)))
    }
}

#[cfg(test)]
mod tests {
    use crate::job_registry::{Delivery, JobDelta, JobRead, JobStatus};
    use crate::test_wording::TestWording;
    use crate::wording::ShellWording;

    fn delta(bytes: &str, cursor: usize, total_bytes: usize, status: JobStatus) -> JobRead {
        JobRead::Delta(JobDelta {
            task_id: "amber-forest-thunder-pearl".to_string(),
            bytes: bytes.as_bytes().to_vec(),
            cursor,
            total_bytes,
            status,
        })
    }

    #[test]
    fn a_read_without_more_buffered_states_the_next_cursor() {
        assert_eq!(
            TestWording.job_read(&delta("hello", 1005, 1005, JobStatus::Running)),
            "hello\nTask 'amber-forest-thunder-pearl' is running; read 5 new bytes (1005 buffered in total); next cursor is 1005.\n"
        );
    }

    #[test]
    fn a_read_with_more_buffered_says_how_to_continue() {
        assert_eq!(
            TestWording.job_read(&delta(
                "hello",
                5,
                1005,
                JobStatus::Completed {
                    exit_code: 0,
                    delivery: Delivery::Pending,
                }
            )),
            "hello\nTask 'amber-forest-thunder-pearl' completed with exit code 0; read 5 new bytes (1005 buffered in total); next cursor is 5.\nMore output is buffered: call terminal_read with cursor=5.\n"
        );
    }

    #[test]
    fn an_unchanged_read_is_data_the_wording_words() {
        let text = TestWording.job_read(&JobRead::Unchanged {
            task_id: "amber-forest-thunder-pearl".to_string(),
        });
        assert!(
            text.contains("'amber-forest-thunder-pearl' produced no new output"),
            "{text}"
        );
    }
}
