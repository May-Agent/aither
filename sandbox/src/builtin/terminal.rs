//! The control tools of a shell's jobs: read, input and kill.
//!
//! The embedder names and describes each; the texts they return are its
//! wording's.

use std::borrow::Cow;
use std::collections::BTreeMap;

use aither_core::llm::ToolContext;
use aither_core::llm::{Tool, ToolResult};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::job_registry::{JobRead, UnchangedRead};
use crate::jobs::Jobs;
use crate::wording::{AnyShellWording, CallRefusal, JobInputSent, JobKillAsked, Refused};

/// What an embedder calls one of its tools and how it describes it to the
/// model; neither has a default.
#[derive(Debug, Clone)]
pub struct ToolIdentity {
    /// The tool's name.
    pub name: Cow<'static, str>,
    /// The tool's description, as the model reads it.
    pub description: Cow<'static, str>,
    /// What the model reads about each argument the tool takes, by argument
    /// name. An argument the tool takes and this lacks crashes the tool's
    /// schema: the arguments carry no descriptions of their own.
    pub arguments: BTreeMap<&'static str, Cow<'static, str>>,
}

impl ToolIdentity {
    /// `schema` with every property described as this identity says.
    ///
    /// # Panics
    /// Panics when a property of the schema has no description here.
    #[must_use]
    pub(crate) fn describe(&self, schema: schemars::Schema) -> schemars::Schema {
        let mut value = schema.to_value();
        self.describe_properties(&mut value);
        schemars::Schema::try_from(value).expect("a described schema is a schema")
    }

    fn describe_properties(&self, value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(object) => {
                if let Some(serde_json::Value::Object(properties)) = object.get_mut("properties") {
                    for (name, property) in properties {
                        let description = self.arguments.get(name.as_str()).unwrap_or_else(|| {
                            panic!(
                                "the identity of the tool `{}` describes no argument `{name}`",
                                self.name
                            )
                        });
                        if let serde_json::Value::Object(property) = property {
                            property.insert(
                                "description".to_owned(),
                                serde_json::Value::String(description.to_string()),
                            );
                        }
                    }
                }
                for child in object.values_mut() {
                    self.describe_properties(child);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    self.describe_properties(item);
                }
            }
            _ => {}
        }
    }
}

/// The identities of the shell tool and of the three job control tools.
#[derive(Debug, Clone)]
pub struct ShellToolSettings {
    /// The tool that runs a script.
    pub shell: ToolIdentity,
    /// The tool that reads a job's output.
    pub read: ToolIdentity,
    /// The tool that writes to a job's stdin.
    pub input: ToolIdentity,
    /// The tool that kills a job.
    pub kill: ToolIdentity,
}

fn task_id<'a>(argument: &'a str, wording: &AnyShellWording) -> aither_core::Result<&'a str> {
    let task_id = argument.trim();
    if task_id.is_empty() {
        return Err(Refused::call(&**wording, CallRefusal::EmptyTaskId));
    }
    Ok(task_id)
}

fn unknown_task(task_id: &str, wording: &AnyShellWording) -> anyhow::Error {
    Refused::call(
        &**wording,
        CallRefusal::UnknownTask {
            task_id: task_id.to_owned(),
        },
    )
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

// The arguments of the kill tool, described by its `ToolIdentity`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[allow(missing_docs, reason = "the schema's descriptions are the embedder's")]
pub struct JobKillArgs {
    // The job to kill.
    pub task_id: String,
}

impl Tool for JobKillTool {
    fn name(&self) -> Cow<'static, str> {
        self.identity.name.clone()
    }

    fn description(&self) -> Cow<'static, str> {
        self.identity.description.clone()
    }

    fn arguments_schema(&self) -> schemars::Schema {
        self.identity.describe(schemars::schema_for!(JobKillArgs))
    }

    type Arguments = JobKillArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let task_id = task_id(&args.task_id, &self.wording)?;
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

// The arguments of the input tool, described by its `ToolIdentity`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[allow(missing_docs, reason = "the schema's descriptions are the embedder's")]
pub struct JobInputArgs {
    // The job to write to.
    pub task_id: String,
    // The text written to the job's stdin.
    pub input: String,
    // Whether a trailing newline is appended before writing.
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

    fn arguments_schema(&self) -> schemars::Schema {
        self.identity.describe(schemars::schema_for!(JobInputArgs))
    }

    type Arguments = JobInputArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let task_id = task_id(&args.task_id, &self.wording)?;
        let job = self
            .jobs
            .get(task_id)
            .await
            .ok_or_else(|| unknown_task(task_id, &self.wording))?;

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

// The arguments of the read tool, described by its `ToolIdentity`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[allow(missing_docs, reason = "the schema's descriptions are the embedder's")]
pub struct JobReadArgs {
    // The job to read.
    pub task_id: String,
    // The offset after which output is read.
    #[serde(default)]
    pub cursor: usize,
    // The most bytes returned.
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

    fn arguments_schema(&self) -> schemars::Schema {
        self.identity.describe(schemars::schema_for!(JobReadArgs))
    }

    type Arguments = JobReadArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let task_id = task_id(&args.task_id, &self.wording)?;
        let max_bytes = args
            .max_bytes
            .unwrap_or_else(|| self.output.get().read_bytes);
        if max_bytes == 0 {
            return Err(Refused::call(&*self.wording, CallRefusal::ZeroMaxBytes));
        }
        let job = self
            .jobs
            .get(task_id)
            .await
            .ok_or_else(|| unknown_task(task_id, &self.wording))?;
        let read = job
            .read(args.cursor, max_bytes)
            .await
            .map_err(anyhow::Error::msg)?;
        match read {
            JobRead::Delta(_) => Ok(ToolResult::text(self.wording.job_read(&read))),
            // A poll is a refusal the loop can tell from a read, worded as
            // the unchanged read.
            JobRead::Unchanged { task_id } => {
                Err(Refused::rendered(UnchangedRead { task_id }, |unchanged| {
                    self.wording.job_read(&JobRead::Unchanged {
                        task_id: unchanged.task_id.clone(),
                    })
                })
                .into())
            }
        }
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
