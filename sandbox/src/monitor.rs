//! `monitor` — watch a command or a file and deliver its new output as it
//! arrives.
//!
//! Following a running command or a growing log used to mean polling: sleep,
//! read, sleep, read. `monitor` runs the source as an ordinary background
//! job — same sandbox, same PID-only kills, `terminal_kill` reaches it — and
//! a pump raises its new output as [`ShellEvent::MonitorIncrement`] facts,
//! which the embedder merges into its agent's next turn boundary.

use std::borrow::Cow;
use std::fmt;
use std::time::{Duration, Instant};

use aither_core::llm::{AgentPath, Tool, ToolContext, ToolResult};
use executor_core::{Executor, Task};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::events::{EventSender, ShellEvent};
use crate::interpreter::Interpreter;
use crate::job_registry::{JobRegistry, JobStatus};
use crate::limits::LiveDuration;
use crate::output::MediaResolution;
use crate::policy::SandboxMode;
use crate::terminal::{Shell, ShellArgs};
use crate::wording::{MonitorStarted, Refused};

/// How long the watched job's own `timeout` bound outlasts the monitor's
/// deadline.
const JOB_MARGIN: Duration = Duration::from_secs(60);

/// What an embedder decides about monitoring: the tool's name and description
/// and the bounds of a watch. The bounds are live, so changing one reaches
/// the tools already running.
#[derive(Debug, Clone)]
pub struct MonitorSettings {
    /// The tool's name, description and argument descriptions.
    pub identity: crate::ToolIdentity,
    /// The longest a monitor may run.
    pub max_timeout: LiveDuration,
    /// The interval between output collections of a call that states none.
    pub default_interval: LiveDuration,
}

// The arguments of the monitor tool, described by its `ToolIdentity`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[allow(missing_docs, reason = "the schema's descriptions are the embedder's")]
pub struct MonitorArgs {
    // What to watch.
    pub source: MonitorSource,
    // Seconds between output collections.
    #[serde(default)]
    pub interval_secs: Option<u64>,
    // Only lines matching it are delivered.
    #[serde(default)]
    pub filter: Option<String>,
    // The first line matching it ends the monitor.
    #[serde(default)]
    pub stop_on: Option<String>,
    // Whether the watched process is killed when `stop_on` matches.
    #[serde(default)]
    pub kill_on_stop: bool,
    // Seconds without new output after which the run is reported stuck.
    #[serde(default)]
    pub stall_secs: Option<u64>,
    // The monitor's maximum lifetime in seconds.
    pub timeout_secs: u64,
}

// The `monitor` source: a command's source text or a file path.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
#[allow(missing_docs, reason = "the schema's descriptions are the embedder's")]
pub enum MonitorSource {
    // Source text run inside the sandbox.
    Command {
        // The source to run.
        command: String,
        // Its interpreter.
        #[serde(default)]
        interpreter: Interpreter,
    },
    // A file whose appended lines are followed inside the sandbox.
    File {
        // The path, resolved against the working directory when relative.
        file: String,
    },
}

/// New output a monitor collected since the last delivery.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MonitorIncrement {
    pub(crate) agent: AgentPath,
    pub(crate) task_id: String,
    pub(crate) lines: Vec<String>,
}

impl MonitorIncrement {
    /// The agent whose shell raised the event.
    #[must_use]
    pub const fn agent(&self) -> &AgentPath {
        &self.agent
    }

    /// Task id of the watched job.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// The new lines (filtered when the monitor has a `filter`).
    #[must_use]
    pub fn lines(&self) -> &[String] {
        &self.lines
    }
}

/// A monitor saw no new output for `stall_secs`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MonitorStalled {
    pub(crate) agent: AgentPath,
    pub(crate) task_id: String,
    pub(crate) stall_secs: u64,
}

impl MonitorStalled {
    /// The agent whose shell raised the event.
    #[must_use]
    pub const fn agent(&self) -> &AgentPath {
        &self.agent
    }

    /// Task id of the watched job.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// How long it has produced nothing.
    #[must_use]
    pub const fn stall_secs(&self) -> u64 {
        self.stall_secs
    }
}

/// A monitor ended, and why.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MonitorEnded {
    pub(crate) agent: AgentPath,
    pub(crate) task_id: String,
    pub(crate) end: MonitorEnd,
}

impl MonitorEnded {
    /// The agent whose shell raised the event.
    #[must_use]
    pub const fn agent(&self) -> &AgentPath {
        &self.agent
    }

    /// Task id of the watched job.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// What ended it.
    #[must_use]
    pub const fn end(&self) -> &MonitorEnd {
        &self.end
    }
}

/// What ended a monitor.
#[derive(Debug, Clone)]
pub enum MonitorEnd {
    /// The watched command exited.
    Exited {
        /// Its exit code.
        exit_code: i32,
    },
    /// The first line matching `stop_on` arrived.
    StopMatched {
        /// The matching line.
        line: String,
    },
    /// The monitor's `timeout_secs` passed.
    TimedOut,
    /// The watched job was killed (`terminal_kill` or the watchdog).
    Killed,
    /// The watched job failed.
    Failed {
        /// Its error.
        error: String,
    },
    /// The job left the registry with no status.
    Lost,
}

/// Why a `monitor` call was refused before anything started.
#[derive(Debug)]
pub enum MonitorRefusal {
    /// `timeout_secs` was 0 or above the cap.
    TimeoutOutOfRange {
        /// The cap in force.
        max: Duration,
    },
    /// `interval_secs` was 0.
    IntervalTooSmall,
    /// `filter` or `stop_on` is not a valid regex.
    BadRegex {
        /// Which argument failed to compile.
        argument: &'static str,
        /// The regex error.
        reason: String,
    },
    /// A file source the sandbox may not read.
    UnreadableFile {
        /// The path as given.
        file: String,
    },
    /// The monitored job would not start.
    StartFailed {
        /// The spawn error.
        reason: String,
    },
}

/// Everything a running monitor needs to know to watch and report.
struct Watch {
    task_id: String,
    /// Keeps the job, and its word, until the pump reports its end, so a
    /// completion the agent took first still leaves the pump its status.
    _job: crate::Job,
    registry: JobRegistry,
    alerts: EventSender,
    interval: Duration,
    deadline: Instant,
    stall: Option<Duration>,
    filter: Option<regex::Regex>,
    stop_on: Option<regex::Regex>,
    kill_on_stop: bool,
}

/// The native `monitor` tool: starts the watched source as a background job
/// and a pump that raises its increments as [`ShellEvent`]s.
pub struct MonitorTool<E> {
    terminal: Shell<E>,
    settings: MonitorSettings,
}

/// Settings for tests, whose values stand for nothing.
#[cfg(any(test, feature = "test-budgets"))]
pub(crate) fn fixture_settings() -> MonitorSettings {
    MonitorSettings {
        identity: crate::ToolIdentity {
            name: "monitor".into(),
            description: "Watches a source.".into(),
            arguments: crate::test_wording::argument_descriptions(),
        },
        max_timeout: LiveDuration::new(Duration::from_secs(7200)),
        default_interval: LiveDuration::new(Duration::from_secs(10)),
    }
}

impl<E> MonitorTool<E> {
    /// Creates the tool bound to `terminal`, the configured terminal whose
    /// jobs it watches.
    #[must_use]
    pub const fn new(terminal: Shell<E>, settings: MonitorSettings) -> Self {
        Self { terminal, settings }
    }
}

impl<E> std::fmt::Debug for MonitorTool<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MonitorTool")
    }
}

/// The literal `'` inside a shell single-quoted string.
fn single_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', r"'\''"))
}

impl<E: Executor + Clone + 'static> MonitorTool<E> {
    /// The seconds between collections a call asks for, the embedder's
    /// default when it states none.
    fn interval_secs(&self, args: &MonitorArgs) -> u64 {
        args.interval_secs
            .unwrap_or_else(|| self.settings.default_interval.get().as_secs())
    }

    /// The script the monitored job runs for `source` and its interpreter; an
    /// unreadable file is refused before anything spawns.
    fn job_script(&self, source: &MonitorSource) -> Result<(String, Interpreter), MonitorRefusal> {
        match source {
            MonitorSource::Command {
                command,
                interpreter,
            } => {
                if command.trim().is_empty() {
                    return Err(MonitorRefusal::StartFailed {
                        reason: "command must not be empty".to_string(),
                    });
                }
                Ok((command.clone(), *interpreter))
            }
            MonitorSource::File { file } => {
                let mut path = std::path::PathBuf::from(file);
                if path.is_relative() {
                    path = self.terminal.working_dir().join(path);
                }
                let readable = std::fs::metadata(&path).is_ok_and(|meta| meta.is_file())
                    && (path.starts_with(self.terminal.working_dir())
                        || self
                            .terminal
                            .grants()
                            .snapshot()
                            .readable
                            .iter()
                            .any(|dir| path.starts_with(dir)));
                if !readable {
                    return Err(MonitorRefusal::UnreadableFile { file: file.clone() });
                }
                // `tail -F` follows appended lines and survives rotation
                // (the name is reopened) and truncation.
                Ok((
                    format!("tail -Fn0 {}", single_quote(&path.to_string_lossy())),
                    Interpreter::Shell,
                ))
            }
        }
    }
}

impl<E> Tool for MonitorTool<E>
where
    E: Executor + Clone + 'static,
{
    fn name(&self) -> Cow<'static, str> {
        self.settings.identity.name.clone()
    }

    fn description(&self) -> Cow<'static, str> {
        self.settings.identity.description.clone()
    }

    fn arguments_schema(&self) -> schemars::Schema {
        self.settings
            .identity
            .describe(schemars::schema_for!(MonitorArgs))
    }

    type Arguments = MonitorArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let notices = self.terminal.wording();
        let interval_secs = self.interval_secs(&args);
        match self.start(args).await {
            Ok(task_id) => Ok(ToolResult::text(notices.monitor_started(&MonitorStarted {
                task_id: &task_id,
                interval_secs,
            }))),
            Err(refusal) => {
                Err(Refused::rendered(refusal, |refusal| notices.monitor_refusal(refusal)).into())
            }
        }
    }
}

impl<E: Executor + Clone + 'static> MonitorTool<E> {
    /// Starts the watched job and its pump; returns the job's task id.
    async fn start(&self, args: MonitorArgs) -> Result<String, MonitorRefusal> {
        // The pump owns the deadline and the job's own bound stays a margin
        // past it, so a monitor is accepted only when the shell accepts that
        // bound too.
        let max_timeout = self
            .settings
            .max_timeout
            .get()
            .min(self.terminal.timeout_cap().saturating_sub(JOB_MARGIN));
        if args.timeout_secs == 0 || args.timeout_secs > max_timeout.as_secs() {
            return Err(MonitorRefusal::TimeoutOutOfRange { max: max_timeout });
        }
        let interval_secs = self.interval_secs(&args);
        if interval_secs == 0 {
            return Err(MonitorRefusal::IntervalTooSmall);
        }
        let compile = |argument: &'static str, pattern: &Option<String>| {
            pattern
                .as_deref()
                .map(|pattern| {
                    regex::Regex::new(pattern).map_err(|error| MonitorRefusal::BadRegex {
                        argument,
                        reason: error.to_string(),
                    })
                })
                .transpose()
        };
        let filter = compile("filter", &args.filter)?;
        let stop_on = compile("stop_on", &args.stop_on)?;
        let (script, interpreter) = self.job_script(&args.source)?;
        let description = match &args.source {
            MonitorSource::Command { .. } => "monitored command".to_string(),
            MonitorSource::File { file } => format!("monitor file {file}"),
        };
        let job = self
            .terminal
            .run_background(ShellArgs {
                description,
                script,
                interpreter,
                sandbox: SandboxMode::Sandboxed,
                location: None,
                resolution: MediaResolution::Auto,
                max_lines: Some(1),
                raw: true,
                // The job's own watchdog stays a margin past the monitor's
                // deadline, so the pump's kill — not the watchdog's — ends a
                // timed-out run and the end reports `TimedOut`.
                timeout: args.timeout_secs + JOB_MARGIN.as_secs(),
                background: true,
            })
            .await
            .map_err(|error| MonitorRefusal::StartFailed {
                reason: error.to_string(),
            })?;
        let task_id = job.task_id().to_owned();
        let watch = Watch {
            _job: job,
            task_id: task_id.clone(),
            registry: self.terminal.job_registry(),
            alerts: self.terminal.event_sender(),
            interval: Duration::from_secs(interval_secs),
            deadline: Instant::now() + Duration::from_secs(args.timeout_secs),
            stall: args.stall_secs.map(Duration::from_secs),
            filter,
            stop_on,
            kill_on_stop: args.kill_on_stop,
        };
        self.terminal
            .executor()
            .spawn(async move { watch_task(watch).await })
            .detach();
        Ok(task_id)
    }
}

/// Trailing bytes of a line without its `\n` yet; kept across polls.
fn drain_lines(buffer: &mut Vec<u8>) -> Vec<String> {
    let Some(last_nl) = buffer.iter().rposition(|byte| *byte == b'\n') else {
        return Vec::new();
    };
    let chunk: Vec<u8> = buffer.drain(..=last_nl).collect();
    String::from_utf8_lossy(&chunk)
        .lines()
        .map(ToString::to_string)
        .collect()
}

async fn send(alerts: &EventSender, alert: ShellEvent) {
    let _ = alerts.send(alert).await;
}

#[allow(clippy::too_many_lines)]
async fn watch_task(watch: Watch) {
    let Watch {
        task_id,
        _job,
        registry,
        alerts,
        interval,
        deadline,
        stall,
        filter,
        stop_on,
        kill_on_stop,
    } = watch;
    let mut cursor = 0usize;
    let mut remainder = Vec::new();
    let mut last_output = Instant::now();
    let mut stalled = false;
    loop {
        async_io::Timer::after(
            interval
                .min(deadline.saturating_duration_since(Instant::now()) + Duration::from_secs(1)),
        )
        .await;
        let now = Instant::now();
        let mut end = match registry.monitor_read(&task_id, cursor, usize::MAX).await {
            Ok(delta) => {
                cursor = delta.cursor;
                remainder.extend_from_slice(&delta.bytes);
                let mut stop = None;
                let mut lines = Vec::new();
                for line in drain_lines(&mut remainder) {
                    if stop_on.as_ref().is_some_and(|re| re.is_match(&line)) {
                        stop = Some(line);
                        break;
                    }
                    if filter.as_ref().is_none_or(|re| re.is_match(&line)) {
                        lines.push(line);
                    }
                }
                if !lines.is_empty() {
                    last_output = now;
                    send(
                        &alerts,
                        ShellEvent::MonitorIncrement(MonitorIncrement {
                            agent: alerts.agent().clone(),
                            task_id: task_id.clone(),
                            lines,
                        }),
                    )
                    .await;
                }
                if stop.is_some() {
                    last_output = now;
                }
                if let Some(line) = stop {
                    if kill_on_stop {
                        registry.kill_by_task_id(&task_id).await;
                    }
                    Some(MonitorEnd::StopMatched { line })
                } else {
                    match delta.status {
                        JobStatus::Running => None,
                        JobStatus::Completed { exit_code, .. } => {
                            Some(MonitorEnd::Exited { exit_code })
                        }
                        JobStatus::Failed { error, .. } => Some(MonitorEnd::Failed { error }),
                        JobStatus::Killed { .. } => Some(MonitorEnd::Killed),
                    }
                }
            }
            Err(_) => Some(MonitorEnd::Lost),
        };
        if end.is_none() && now >= deadline {
            registry.kill_by_task_id(&task_id).await;
            end = Some(MonitorEnd::TimedOut);
        }
        if let Some(end) = end {
            send(
                &alerts,
                ShellEvent::MonitorEnded(MonitorEnded {
                    agent: alerts.agent().clone(),
                    task_id,
                    end,
                }),
            )
            .await;
            return;
        }
        if !stalled && stall.is_some_and(|stall| now.duration_since(last_output) >= stall) {
            stalled = true;
            send(
                &alerts,
                ShellEvent::MonitorStalled(MonitorStalled {
                    agent: alerts.agent().clone(),
                    task_id: task_id.clone(),
                    stall_secs: stall.map_or(0, |s| s.as_secs()),
                }),
            )
            .await;
        }
    }
}
