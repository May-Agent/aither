//! Terminal command tool implementation with heel sandboxing.
//!
//! Provides the main `terminal` tool that LLMs use to execute terminal work.
//! Despite the historical tool name, execution does not have to go through a
//! literal `terminal` process: simple commands may be executed directly, while shell
//! syntax falls back to a real shell when needed.
//! Commands run in a sandbox with configurable permission modes.
//!
//! Each `Shell` works in one shared working directory. All terminal
//! executions share this directory, but each execution gets a fresh sandbox
//! (new TTY/process). The ids the model addresses — task ids, stored output
//! names, subagent spaces — are leased from the terminal's
//! [`AnyIdAllocator`](aither_core::AnyIdAllocator).

use crate::foreground_wait::ForegroundWait;
use crate::memory::{
    AnyContainerMemory, AnyFootprintReader, AnyFootprintRegistry, FootprintReader,
    FootprintRegistry, JobMemory, ProbedContainerMemory, ResponseWindow, SystemFootprints,
};
use crate::outage::Outages;
use crate::spawn::Backend;
#[cfg(test)]
use crate::{AnyShellWording, OutputPolicy};
use crate::{CallRefusal, RunFailure, ShellWording};
use std::{
    borrow::Cow,
    io::Read,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use crate::spawn::container_ipc::ExecSetup;
use aither_core::llm::tool::{
    AnyApprover, Approval, ApprovalError, ApprovalKind, ApprovalRequest, Approver,
};
use aither_core::llm::{AgentPath, CallIdentity, ToolContext};
use aither_core::llm::{
    CommandOutcome, CommandResult, CommandStream, Interruption, IntoToolResult, StoredFile, Tool,
    ToolResult,
};
use aither_core::{AnyIdAllocator, IdLease, IdsExhausted};
use askama::Template;
use async_channel::{Receiver, Sender};
use executor_core::{Executor, Task};
#[cfg(target_os = "linux")]
use heel::supervisor::{TreeControl, TreeReport};
use heel::{CpuPriority, IoPriority, NetworkAuditLog, ProcessPriority};
/// On platforms without `heel supervise` a wait produces no report.
#[cfg(not(target_os = "linux"))]
pub struct TreeReport;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use shell_words::split as split_shell_words;
use tracing::{debug, info, warn};

use crate::builtin::ShellToolSettings;
use crate::events::{
    DescendantsKilled, EventSender, JobEnded, JobStarted, LongRunning, ShellEvent, ShellEvents,
    StdinBlocked, StdinWatchFailed,
};
use crate::jobs::Jobs;
#[cfg(all(unix, feature = "jobserver"))]
use crate::jobserver::{JobBroker, JobPool, job_broker_channel};
use crate::{
    Carry, ContainerPath, MountRoot, OutputTreatment, RemoteLaunch, SandboxMode, ScriptDecision,
    ScriptDir, ScriptOrigin, ScriptOutcome, ScriptPolicies, ScriptRefusal,
    interpreter::Interpreter,
    job_registry::{JobRegistry, JobStatus, job_registry_channel},
    limits::{LiveDuration, ShellLimits},
    output::{MediaResolution, OutputFormat, OutputStore, stored_path},
    process_tree::{JobProcess, JobTree},
    shell_session::{ContainerShellRuntime, ShellSessionRegistry},
    spawn::stdio::{read_chunks, write_chunks},
    spawn::{
        Descendants, EnvSource, GrantSnapshot, Grants, LocalSpawn, Pipe, Sandbox, SpawnSpec,
        Stdio as SpawnStdio, TreeLifetime,
    },
    ssh::{SshClient, SshEnvironment, SshServer},
    stdin_watch::{
        EpisodeDetector, FeedCounter, LocalStdinProbe, STDIN_WATCH_INTERVAL, StdinWatchSignal,
    },
    subagent::{
        AnySubagentSpaces, CommitLedger, CommittingWorktree, ImportBudget, ImportLimits,
        LaunchHistory, Launcher, PreparedSpace, ResolvedGrant, RunSpace, Spelling, SubagentGit,
        SubagentGrant, SubagentGrantError, SubagentSpaces, canonical as canonical_path,
    },
};

mod builder;
mod subagent_git;

pub use builder::{Given, Missing, ShellBuilder};

/// Opens a daily-rotated network audit log under `dir`, creating the directory if needed.
///
/// Keeps at most `max_files` rotated files. Attach the log with
/// [`Sandbox::with_audit_log`] to record every sandboxed
/// network policy decision.
///
/// # Errors
/// Returns an error when the directory cannot be created or the log cannot
/// be opened.
pub async fn open_network_audit_log(
    dir: impl AsRef<Path>,
    max_files: usize,
) -> Result<NetworkAuditLog, TerminalError> {
    let dir = dir.as_ref();
    async_fs::create_dir_all(dir).await?;
    NetworkAuditLog::rolling_daily(dir, max_files).map_err(|error| {
        TerminalError::SandboxSetup(format!("failed to open network audit log: {error}"))
    })
}

fn parse_direct_command(script: &str) -> Result<(String, Vec<String>), TerminalError> {
    let parts = split_shell_words(script)
        .map_err(|error| TerminalError::Execution(format!("failed to parse command: {error}")))?;
    let Some((program, args)) = parts.split_first() else {
        return Err(TerminalError::Execution(
            "command must not be empty".to_string(),
        ));
    };
    Ok((program.clone(), args.to_vec()))
}

enum TerminalLaunch {
    Direct { program: String, args: Vec<String> },
    Shell { program: String, args: Vec<String> },
}

fn script_requires_shell(script: &str) -> bool {
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    for ch in script.chars() {
        if escaped {
            escaped = false;
            continue;
        }

        match ch {
            '\\' if !in_single => escaped = true,
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '\n' | '\r' | '|' | '&' | ';' | '<' | '>' | '(' | ')' | '$' | '`' | '*' | '?' | '['
            | ']' | '{' | '}' | '!' | '~'
                if !in_single && !in_double =>
            {
                return true;
            }
            _ => {}
        }
    }

    false
}

fn looks_like_env_assignment(token: &str) -> bool {
    let Some((name, _value)) = token.split_once('=') else {
        return false;
    };
    !name.is_empty() && name.chars().all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn is_shell_builtin(program: &str) -> bool {
    matches!(
        program,
        "alias"
            | "bg"
            | "break"
            | "cd"
            | "command"
            | "continue"
            | "eval"
            | "exec"
            | "exit"
            | "export"
            | "fg"
            | "jobs"
            | "read"
            | "readonly"
            | "return"
            | "set"
            | "shift"
            | "source"
            | "times"
            | "trap"
            | "type"
            | "ulimit"
            | "umask"
            | "unalias"
            | "unset"
    )
}

/// The program and arguments that run `script` in a shell. Windows runs
/// `cmd.exe`; every other platform runs `managed_shell`, the managed bash.
pub fn shell_launch(managed_shell: &str, script: &str) -> (String, Vec<String>) {
    #[cfg(windows)]
    {
        let _ = managed_shell;
        let shell = std::env::var("COMSPEC").unwrap_or_else(|_| "cmd.exe".to_string());
        return (shell, vec!["/C".to_string(), script.to_string()]);
    }

    #[cfg(not(windows))]
    {
        (
            managed_shell.to_string(),
            crate::spawn::shell_arguments(script),
        )
    }
}

/// The shell script that runs `script`, written in `language`, for the job
/// `task_id`.
fn launch_script<'a>(
    interpreter: Interpreter,
    script: &'a str,
    task_id: &str,
) -> Result<Cow<'a, str>, TerminalError> {
    interpreter.launch_script(script, task_id).map_err(|error| {
        TerminalError::Execution(format!("failed to render the launch script: {error}"))
    })
}

/// Turns `script` into the process to launch. A shell launch with a `report`
/// file records the shell's final directory there when it exits; a direct
/// command cannot change directory, so it reports nothing.
fn build_terminal_launch(
    managed_shell: &str,
    script: &str,
    report: Option<&Path>,
) -> Result<TerminalLaunch, TerminalError> {
    let script = script.trim();
    if script.is_empty() {
        return Err(TerminalError::Execution(
            "command must not be empty".to_string(),
        ));
    }

    match direct_command(script)? {
        Some((program, args)) => Ok(TerminalLaunch::Direct { program, args }),
        None => shell_launch_reporting(managed_shell, script, report),
    }
}

/// The program and arguments of a script that runs without a shell, or
/// `None` when it needs one.
fn direct_command(script: &str) -> Result<Option<(String, Vec<String>)>, TerminalError> {
    if script_requires_shell(script) {
        return Ok(None);
    }
    let (program, args) = parse_direct_command(script)?;
    if looks_like_env_assignment(&program) || is_shell_builtin(&program) {
        return Ok(None);
    }
    Ok(Some((program, args)))
}

/// Whether a local launch of `script` runs in a shell, the only way a
/// command can change directory.
fn runs_in_shell(script: &str) -> bool {
    matches!(direct_command(script.trim()), Ok(None))
}

/// A command killed because its `timeout` bound passed, with the UTC times it
/// started and was killed at.
///
/// [`ShellWording::command_killed`](crate::ShellWording::command_killed)
/// renders the sentence the model reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeoutKill {
    /// The `timeout` the command was bounded by.
    pub timeout: Duration,
    /// When the command started.
    pub started: jiff::Timestamp,
    /// When the kill was issued.
    pub killed: jiff::Timestamp,
}

#[derive(Template)]
#[template(path = "carry_cwd.sh", escape = "none")]
struct CarryCwdTemplate<'a> {
    escaped_report: &'a str,
    escaped_script: &'a str,
}

fn shell_launch_reporting(
    managed_shell: &str,
    script: &str,
    report: Option<&Path>,
) -> Result<TerminalLaunch, TerminalError> {
    let Some(report) = report else {
        let (program, args) = shell_launch(managed_shell, script);
        return Ok(TerminalLaunch::Shell { program, args });
    };
    let script = carry_cwd_script(script, &report.to_string_lossy())?;
    let (program, args) = shell_launch(managed_shell, &script);
    Ok(TerminalLaunch::Shell { program, args })
}

/// `script`, run through `eval`, followed by a trailer that writes the
/// shell's final directory to `report`, a path as the shell sees it, and
/// exits with the script's status.
///
/// The report is not written from an `EXIT` trap: setting one makes a
/// non-interactive bash catch SIGTERM, and a child it has just forked still
/// carries that handler until it resets its signals, so a `kill` sent in that
/// window is swallowed and the process it names runs on. `eval` keeps the
/// trailer out of reach of the script's own syntax, such as an unterminated
/// here-document. A script that ends through `exit` writes no report.
fn carry_cwd_script(script: &str, report: &str) -> Result<String, TerminalError> {
    let escaped_report = shell_escape(report);
    let escaped_script = shell_escape(script);
    CarryCwdTemplate {
        escaped_report: &escaped_report,
        escaped_script: &escaped_script,
    }
    .render()
    .map_err(|error| {
        TerminalError::Execution(format!("failed to render directory report: {error}"))
    })
}

// The arguments of the shell tool. They carry no rustdoc: the schema's
// descriptions are the embedder's, supplied by the tool's `ToolIdentity`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[allow(missing_docs, reason = "the schema's descriptions are the embedder's")]
pub struct ShellArgs {
    // A short explanation of what the command does, shown instead of its source.
    pub description: String,

    // The source to run.
    pub script: String,

    // The interpreter of `script`.
    #[serde(default)]
    pub interpreter: Interpreter,

    // Whether the command runs inside the sandbox.
    #[serde(default)]
    pub sandbox: SandboxMode,

    // Where the command runs: the home environment, or a named SSH server.
    #[serde(default)]
    pub location: Option<String>,

    // The resolution media in the output is delivered at.
    #[serde(default)]
    pub resolution: MediaResolution,

    // The most output lines returned inline.
    #[serde(default)]
    pub max_lines: Option<u32>,

    // Whether the output is returned verbatim, without compression.
    #[serde(default)]
    pub raw: bool,

    // The command's maximum runtime in seconds.
    pub timeout: u64,

    // Whether the command is detached at once instead of waited for.
    #[serde(default)]
    pub background: bool,
}

/// Time bounds applied to every execution this terminal spawns.
#[derive(Debug, Clone)]
struct TerminalLimits {
    /// Foreground wait before a `terminal` call promotes to a background
    /// task.
    foreground_wait: ForegroundWait,
    /// Maximum accepted `timeout` for a command.
    timeout_cap: LiveDuration,
    /// Runtime after which a background job produces a long-running alert.
    long_job_alert: LiveDuration,
    /// Most bytes a subagent's sandbox may send while its commits are
    /// imported; shared with every terminal derived from this one, and
    /// adjustable at runtime.
    subagent_import: ImportBudget,
    /// How long the agent has to act on a final memory warning; adjustable
    /// at runtime and shared with every terminal derived from this one.
    response_window: ResponseWindow,
    /// Reads what the terminal's jobs hold of the machine's memory.
    footprints: AnyFootprintReader,
    /// Shares the total the agent's jobs hold with the machine's other
    /// agents; `None` when the embedder configured no registry.
    footprint_registry: Option<AnyFootprintRegistry>,
}

impl TerminalLimits {
    fn new(limits: ShellLimits) -> Self {
        Self {
            response_window: limits.response_window,
            footprints: AnyFootprintReader::new(SystemFootprints),
            footprint_registry: None,
            foreground_wait: limits.foreground_wait,
            timeout_cap: limits.timeout_cap,
            long_job_alert: limits.long_job_alert,
            subagent_import: limits.subagent_import,
        }
    }
}

/// Why the next foreground command does not start where the last one left
/// the shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryResetReason {
    /// The command failed, so the directory it ended in is not carried.
    Failed,
    /// The command ended through `exit` or a signal and reported no
    /// directory.
    Unreported,
    /// The directory it ended in lies outside the paths the terminal may
    /// read.
    OutsideReadable,
    /// The directory it ended in lies outside the container's workspace
    /// mount.
    OutsideWorkspace,
    /// The directory no longer exists.
    Removed,
}

/// A command that ran to completion.
#[derive(Debug, Clone)]
pub struct CompletedCommand {
    /// stdout output.
    pub stdout: CommandStream,

    /// stderr output (if non-empty).
    pub stderr: Option<CommandStream>,

    /// Exit code of the script.
    pub exit_code: i32,

    /// Sleep and offline intervals the command ran through.
    pub interruptions: Vec<Interruption>,
}

impl CompletedCommand {
    /// The command as a result: it exited with its code, or was killed for
    /// `killed`.
    fn into_script_result(self, killed: Option<crate::KillCause>) -> crate::ScriptResult {
        crate::ScriptResult {
            stdout: self.stdout,
            stderr: self.stderr,
            outcome: killed.map_or(
                crate::RunOutcome::Exited {
                    code: self.exit_code,
                },
                crate::RunOutcome::Killed,
            ),
            interruptions: self.interruptions,
        }
    }
}

#[derive(Clone)]
struct BackgroundStartup {
    tx: async_channel::Sender<Result<(), String>>,
    reported: Arc<AtomicBool>,
    /// Closed once the startup was reported or every clone is gone: nothing
    /// is sent on it.
    settled: async_channel::Sender<()>,
    settled_rx: Receiver<()>,
}

impl BackgroundStartup {
    fn new(tx: async_channel::Sender<Result<(), String>>) -> Self {
        let (settled, settled_rx) = async_channel::bounded(1);
        Self {
            tx,
            reported: Arc::new(AtomicBool::new(false)),
            settled,
            settled_rx,
        }
    }

    /// Resolves once the job registered, or failed to start.
    fn settled(&self) -> Receiver<()> {
        self.settled_rx.clone()
    }

    async fn report_ready(&self) {
        if self
            .reported
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.tx.send(Ok(())).await;
            self.settled.close();
        }
    }

    async fn report_failure(&self, error: &TerminalError) {
        if self
            .reported
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.tx.send(Err(error.detail().into_owned())).await;
            self.settled.close();
        }
    }
}

/// A script [`Shell::run`] runs to its end.
#[derive(Debug, Clone)]
pub struct Script {
    /// The script's source, in the shell's own interpreter.
    pub source: String,
    /// Where it starts: a host path, which for a container shell lies in its
    /// workspace mount.
    pub dir: PathBuf,
    /// The bound after which the script and its process tree are killed.
    pub timeout: Duration,
    /// Whether it runs inside the sandbox; an unsandboxed script is put to
    /// the shell's policies like any other.
    pub sandbox: SandboxMode,
}

impl Script {
    /// `source`, started in `dir` and killed with its process tree once
    /// `timeout` passes, in [`SandboxMode::Sandboxed`].
    #[must_use]
    pub fn new(source: impl Into<String>, dir: impl Into<PathBuf>, timeout: Duration) -> Self {
        Self {
            source: source.into(),
            dir: dir.into(),
            timeout,
            sandbox: SandboxMode::Sandboxed,
        }
    }

    /// The same script run in `sandbox`.
    #[must_use]
    pub const fn with_sandbox(mut self, sandbox: SandboxMode) -> Self {
        self.sandbox = sandbox;
        self
    }
}

/// Why a [`Shell::run`] did not run to an end.
///
/// The variants carry the facts; the `Display` texts are for logs and
/// diagnostics, and the model never reads them: the tool boundary words a
/// failure through [`ShellWording::run_failure`] and
/// [`ShellWording::call_refused`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ShellRunError {
    /// The timeout is zero or above the terminal's cap.
    #[error("timeout must be between 1 and {} seconds", .cap.as_secs())]
    Timeout {
        /// The longest timeout the terminal accepts.
        cap: Duration,
    },
    /// A command policy refused the script.
    #[error(transparent)]
    Refused(ScriptRefusal),
    /// The terminal runs in a container that declares no workspace mount,
    /// so no host directory maps into it.
    #[error("the container declares no workspace mount, so a run cannot start in a host directory")]
    NoWorkspaceMount,
    /// The working directory lies outside the container's workspace mount.
    #[error(transparent)]
    OutsideWorkspace(#[from] crate::MountRootError),
    /// The terminal could not run the script: no backend for its mode, a
    /// refused permission or a failed execution.
    #[error(transparent)]
    Terminal(#[from] TerminalError),
    /// Every model-facing id is leased, so the command has no task id.
    #[error(transparent)]
    Ids(#[from] IdsExhausted),
}

impl ShellRunError {
    /// The error a tool call answers with: a refusal stays typed so the agent
    /// loop recognises it, every other failure is worded by `wording`.
    pub(crate) fn into_tool_error(self, wording: &dyn ShellWording) -> anyhow::Error {
        let failure = match self {
            Self::Refused(refusal) => return anyhow::Error::from(refusal),
            Self::Timeout { cap } => {
                return crate::Refused::call(wording, CallRefusal::TimeoutOutOfRange { cap });
            }
            Self::NoWorkspaceMount => RunFailure::NoWorkspaceMount,
            Self::OutsideWorkspace(ref error) => RunFailure::OutsideWorkspace(error),
            Self::Terminal(ref error) => RunFailure::Terminal(error),
            Self::Ids(_) => RunFailure::IdsExhausted,
        };
        anyhow::Error::msg(wording.run_failure(&failure))
    }
}

/// Kills the running jobs of a registry when dropped.
///
/// `Drop` cannot await, so the kill runs as a task the terminal's executor
/// spawns.
struct KillJobsOnDrop {
    registry: JobRegistry,
    /// Spawns the kill of the registry's jobs on the terminal's executor.
    spawn_kill: Box<dyn Fn(JobRegistry) + Send + Sync>,
}

impl KillJobsOnDrop {
    fn new<E>(registry: JobRegistry, executor: E) -> Self
    where
        E: Executor + 'static,
    {
        Self {
            registry,
            spawn_kill: Box::new(move |registry| {
                executor
                    .spawn(async move {
                        registry.kill_running().await;
                    })
                    .detach();
            }),
        }
    }
}

impl Drop for KillJobsOnDrop {
    fn drop(&mut self) {
        (self.spawn_kill)(self.registry.clone());
    }
}

impl std::fmt::Debug for KillJobsOnDrop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KillJobsOnDrop").finish_non_exhaustive()
    }
}

/// What a subagent's terminal holds that the main agent's does not.
#[derive(Clone, Debug)]
struct SubagentAuthority {
    /// Kills the subagent's running jobs when the last handle of its terminal
    /// goes. Only a subagent's terminal holds one: the main agent's jobs
    /// outlive its terminal, so an embedder can rebuild it with its
    /// background commands still running.
    _jobs: Arc<KillJobsOnDrop>,
    /// Keeps the word naming its space leased while the terminal lives.
    _space_name: IdLease,
    /// The worktrees it may commit in, with their git data; it hands on no
    /// other commit grant.
    committing: Vec<CommittingWorktree>,
    /// Runs git in its sandbox, as it, for the host's view of its commits.
    git: SubagentGit,
    /// What the launcher's [`SubagentSpaces`] made in its space; `None` when
    /// the launcher has none.
    prepared: Option<PreparedSpace>,
}

/// A subagent launch whose grant is resolved: everything the launch still
/// needs from what the launching terminal handed out.
struct ResolvedLaunch {
    /// The launching terminal's working directory and grants, canonical.
    authority: Vec<PathBuf>,
    /// Where the subagent's space goes.
    spaces_root: PathBuf,
    /// The container workspace mount, for a container terminal.
    mount: Option<MountRoot>,
    granted: ResolvedGrant,
}

/// The main agent's working directory, carried from one foreground command
/// to the next. `None` inside means the terminal's own working directory.
#[derive(Clone, Default)]
struct CarriedDir(Arc<Mutex<Option<PathBuf>>>);

impl CarriedDir {
    fn get(&self) -> Option<PathBuf> {
        self.0.lock().expect("carried directory lock").clone()
    }

    fn set(&self, dir: Option<PathBuf>) {
        *self.0.lock().expect("carried directory lock") = dir;
    }

    /// Records `moved_to` when there is one and returns the directory carried
    /// afterwards, under one lock: the directory the next foreground script
    /// starts in is the one returned here, whatever scripts settle beside
    /// this one.
    fn settle(&self, moved_to: Option<PathBuf>) -> Option<PathBuf> {
        let mut carried = self.0.lock().expect("carried directory lock");
        if moved_to.is_some() {
            *carried = moved_to;
        }
        carried.clone()
    }
}

/// Where a command starts, and for a foreground command of the main agent,
/// the report of the directory it ended in.
#[derive(Clone)]
struct CwdCarry {
    /// Where the command starts.
    start: Option<PathBuf>,
    /// The file the command's shell writes its final directory to when it
    /// exits; `None` for a command that carries no directory, such as a
    /// script [`Shell::run`] runs.
    report: Option<PathBuf>,
    /// Set for a container command: the workspace mount that maps the host
    /// paths above into the container and back.
    container: Option<ContainerCarry>,
    /// Whether the command runs in a shell and so writes the report; a
    /// direct command cannot change directory and writes none.
    shell: bool,
    /// Whether the directory carried to this command no longer existed, so
    /// the command starts in the terminal's own directory instead.
    removed: bool,
}

/// How a container command spells its start and directory report.
#[derive(Clone)]
struct ContainerCarry {
    mount: MountRoot,
    /// Where the container sees the report file, when there is one.
    report: Option<ContainerPath>,
    /// Where the command starts inside the container.
    start: ContainerPath,
}

/// The terminal tool for executing scripts in a sandbox.
///
/// Creates a shared working directory that persists across all executions,
/// named by heel with four random words; that name is not one of the leased
/// model-facing ids. Each execution creates a fresh sandbox but shares the
/// working directory.
pub struct Shell<E> {
    /// Shared working directory, named by heel with four random words (e.g.,
    /// `amber-forest-thunder-pearl/`) rather than a leased id.
    working_dir: PathBuf,
    /// The SSH servers scripts may name as their `location`.
    sessions: ShellSessionRegistry,
    /// Approves what no call's context can: a script run with no tool call
    /// to ask through. Shared with the subagents' terminals.
    approver: Option<AnyApprover>,
    executor: E,
    /// Leases the words naming what the model addresses: task ids, stored
    /// outputs and subagent spaces. Shared with the output store and every
    /// subagent terminal launched from this one.
    ids: AnyIdAllocator,
    output_store: OutputStore,
    job_registry: JobRegistry,
    /// Where the facts about this shell's jobs are raised (cloned for each
    /// task that reports one).
    events_tx: EventSender,
    /// Where they are read.
    events_rx: ShellEvents,
    /// What the shell tool and the job control tools are named and
    /// described as.
    job_tools: ShellToolSettings,
    /// The interpreters a script may run under.
    interpreters: Arc<[Interpreter]>,
    /// Time bounds applied to spawned executions.
    limits: TerminalLimits,
    /// Number of still-running spawned executions associated with this tool.
    running_tasks: Arc<AtomicUsize>,
    /// The sandbox this shell's scripts run in: its grants, which every clone
    /// shares and commands snapshot at launch, environment source, network
    /// policy, priority, backend and `heel`. Each job runs in a copy of it
    /// with what the job adds.
    sandbox: Sandbox,
    /// Set on a subagent's terminal, never on the main agent's.
    subagent: Option<SubagentAuthority>,
    /// The working directory the next foreground command starts in; held on
    /// the main agent's terminal only, a subagent's commands always start in
    /// its own space.
    carried_dir: Option<CarriedDir>,
    /// Shared jobserver token broker, when configured.
    #[cfg(all(unix, feature = "jobserver"))]
    jobserver: Option<JobBroker>,
    /// Records the outages commands run through.
    outages: Outages,
    /// Prepares the space of every subagent launched from this terminal and,
    /// at any depth, from theirs.
    subagent_spaces: Option<AnySubagentSpaces>,
    /// The script policies every script of this shell is put to, bound to
    /// the state of the agent the shell serves.
    policies: ScriptPolicies,
    /// Chooses each script's output treatment, shared with every subagent
    /// shell derived from this one; without it every output is command
    /// output.
    treat_output: Option<crate::output_treatment::AnyTreatOutput>,
    /// The commands the embedder serves to this shell's scripts, shared with
    /// every subagent shell derived from this one.
    ipc: Option<crate::AnyIpcProvider>,
    /// The worktrees this shell's subagents were granted commit in so far,
    /// in grant order: the map the host's `git` shim reads. Shared with
    /// every subagent shell derived from this one.
    committed: Arc<Mutex<Vec<CommittingWorktree>>>,
    /// What this shell handed its subagents, so no grant exceeds an earlier
    /// one's claim; launches resolve one at a time.
    launches: Arc<async_lock::Mutex<LaunchHistory>>,
    /// The shell scripts run under when the sandbox runs on the host; `None`
    /// on a container, whose launch names it.
    host_launch: Option<crate::HostLaunch>,
}

impl<E: Clone> Clone for Shell<E> {
    fn clone(&self) -> Self {
        Self {
            working_dir: self.working_dir.clone(),
            sessions: self.sessions.clone(),
            approver: self.approver.clone(),
            executor: self.executor.clone(),
            ids: self.ids.clone(),
            output_store: self.output_store.clone(),
            job_registry: self.job_registry.clone(),
            events_tx: self.events_tx.clone(),
            events_rx: self.events_rx.clone(),
            job_tools: self.job_tools.clone(),
            interpreters: self.interpreters.clone(),
            limits: self.limits.clone(),
            running_tasks: self.running_tasks.clone(),
            sandbox: self.sandbox.clone(),
            subagent: self.subagent.clone(),
            carried_dir: self.carried_dir.clone(),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: self.jobserver.clone(),
            outages: self.outages.clone(),
            subagent_spaces: self.subagent_spaces.clone(),
            policies: self.policies.clone(),
            treat_output: self.treat_output.clone(),
            ipc: self.ipc.clone(),
            committed: self.committed.clone(),
            launches: self.launches.clone(),
            host_launch: self.host_launch.clone(),
        }
    }
}

impl<E: Executor + Clone + 'static> Shell<E> {
    /// Sets the approver that answers the approvals of scripts no tool call
    /// carries ([`Shell::run`] and [`Self::run_background`]), and
    /// of calls whose dispatcher attached none. Without one, such a script
    /// that needs approval does not run.
    #[must_use]
    pub fn with_approver(mut self, approver: impl Approver) -> Self {
        self.approver = Some(AnyApprover::new(approver));
        self
    }

    /// Serves the SSH servers of `sessions` as the locations scripts may
    /// name. Registries share their server set, so terminals given clones of
    /// one see the same servers.
    #[must_use]
    pub fn with_shell_sessions(mut self, sessions: ShellSessionRegistry) -> Self {
        self.sessions = sessions;
        self
    }

    /// Has `spaces` prepare the space of every subagent launched from this
    /// terminal and, at any depth, from theirs, before the subagent's
    /// terminal exists. A launch whose space cannot be prepared fails with
    /// the error `spaces` gives.
    #[must_use]
    pub fn with_subagent_spaces(mut self, spaces: impl SubagentSpaces) -> Self {
        self.subagent_spaces = Some(AnySubagentSpaces::new(spaces));
        self
    }

    /// Leases every word this terminal names something with for the model —
    /// a command's task id, a stored output's file name, a subagent's space —
    /// from `ids`, as do the subagent terminals launched from it and the
    /// `subagent` command bound to it. A terminal given none leases from its
    /// own [`AnyIdAllocator::default`].
    #[must_use]
    pub fn with_id_allocator(mut self, ids: AnyIdAllocator) -> Self {
        self.output_store = OutputStore::at(
            self.output_store.dir().to_path_buf(),
            ids.clone(),
            self.output_store.policy().clone(),
            self.output_store.wording().clone(),
        );
        self.ids = ids;
        self
    }

    /// Creates a shell whose scripts run in `sandbox`, working in its
    /// working directory itself.
    ///
    /// # Errors
    /// Returns an error when the working directory or output store cannot be
    /// created, or when the sandbox runs its processes on an SSH server: a
    /// server is a location a script names, not a shell's home.
    pub(crate) async fn from_parts(
        sandbox: Sandbox,
        executor: E,
        parts: builder::ShellParts,
    ) -> Result<Self, TerminalError> {
        match (sandbox.backend(), &parts.host_shell) {
            (Backend::Ssh { server, .. }, _) => {
                return Err(TerminalError::SandboxSetup(format!(
                    "the sandbox runs on the SSH server {}: a server is a location, not a shell's home",
                    server.name
                )));
            }
            (Backend::Host { .. }, None) => {
                return Err(TerminalError::SandboxSetup(
                    "a shell on the host needs its host shell: build it with `Shell::builder_on_host`"
                        .into(),
                ));
            }
            (Backend::Container { .. }, Some(_)) => {
                return Err(TerminalError::SandboxSetup(
                    "a host shell was given to a shell on a container, which runs the shell its launch names"
                        .into(),
                ));
            }
            (Backend::Host { .. }, Some(_)) | (Backend::Container { .. }, None) => {}
        }
        // Ensure working directory exists, then resolve it once: heel and the
        // directory report both use this spelling, and a symlink in the given
        // one is not necessarily granted inside the sandbox.
        async_fs::create_dir_all(sandbox.working_dir()).await?;
        let working_dir_path =
            dunce::simplified(&async_fs::canonicalize(sandbox.working_dir()).await?).to_path_buf();

        // Create outputs directory inside working dir
        let ids = AnyIdAllocator::default();
        let output_store = OutputStore::new(
            working_dir_path.join("outputs"),
            ids.clone(),
            parts.output,
            parts.wording,
        )
        .await?;

        // Start job registry service
        let (job_registry, job_registry_service) =
            job_registry_channel(output_store.policy().clone());
        executor
            .spawn(async move { job_registry_service.serve().await })
            .detach();

        let events_rx = parts.events.unwrap_or_default();
        let events_tx = events_rx.sender(AgentPath::main());
        let running_tasks = Arc::new(AtomicUsize::new(0));
        let sandbox = sandbox
            .with_working_dir(working_dir_path.clone())
            .with_scanner(job_registry.scanner());

        Ok(Self {
            working_dir: working_dir_path,
            sessions: ShellSessionRegistry::default(),
            approver: None,
            executor,
            ids,
            output_store,
            job_registry,
            events_tx,
            events_rx,
            job_tools: parts.tools,
            interpreters: parts.interpreters.into(),
            limits: TerminalLimits::new(parts.limits),
            running_tasks,
            sandbox,
            subagent: None,
            carried_dir: Some(CarriedDir::default()),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: None,
            outages: Outages::none(),
            subagent_spaces: None,
            policies: ScriptPolicies::new(parts.policies),
            treat_output: parts.treat_output,
            ipc: parts.ipc,
            committed: Arc::new(Mutex::new(Vec::new())),
            launches: Arc::default(),
            host_launch: parts.host_shell,
        })
    }

    /// A shell on the host in `working_dir` under the test `heel`, with
    /// shell and Python configured and nothing granted.
    #[cfg(test)]
    pub(crate) async fn new_exact(
        working_dir: impl AsRef<std::path::Path>,
        executor: E,
        limits: ShellLimits,
        output: OutputPolicy,
        wording: AnyShellWording,
        tools: ShellToolSettings,
    ) -> Result<Self, TerminalError> {
        let heel = crate::test_heel::binary().await.expect("a test heel");
        let sandbox = Sandbox::new(
            Backend::Host { heel },
            executor.clone(),
            working_dir.as_ref(),
        );
        Self::new_exact_on(sandbox, executor, limits, output, wording, tools).await
    }

    /// A shell running in `sandbox`, with shell and Python configured.
    #[cfg(test)]
    pub(crate) async fn new_exact_on(
        sandbox: Sandbox,
        executor: E,
        limits: ShellLimits,
        output: OutputPolicy,
        wording: AnyShellWording,
        tools: ShellToolSettings,
    ) -> Result<Self, TerminalError> {
        let host_shell = matches!(sandbox.backend(), Backend::Host { .. })
            .then(|| crate::HostLaunch::new(TEST_HOST_SHELL));
        Self::from_parts(
            sandbox,
            executor,
            builder::ShellParts {
                interpreters: vec![Interpreter::Shell, Interpreter::Python],
                limits,
                output,
                wording,
                tools,
                events: None,
                policies: Vec::new(),
                ipc: None,
                treat_output: None,
                host_shell,
            },
        )
        .await
    }

    /// The sandbox this shell's scripts run in, so an embedder that built the
    /// shell spawns its own processes in the same sandbox: the same live
    /// grants, environment source, network policy and backend.
    #[must_use]
    pub const fn sandbox(&self) -> &Sandbox {
        &self.sandbox
    }
}

/// The shell program the tests' host shells run scripts under.
#[cfg(test)]
const TEST_HOST_SHELL: &str = "/bin/bash";

#[cfg(test)]
impl<E> Shell<E>
where
    E: Executor + Clone + 'static,
{
    /// This shell with `policies` as its script policies, in place of the
    /// ones it was built with.
    fn with_test_policies(mut self, policies: Vec<crate::AnyScriptPolicy>) -> Self {
        self.policies = ScriptPolicies::new(policies);
        self
    }

    /// A subagent terminal with an empty grant.
    async fn subagent_without_grant(&self) -> Result<Self, SubagentGrantError> {
        self.subagent(SubagentGrant::new(), None, "worker").await
    }
}

impl<E> Shell<E>
where
    E: Executor + Clone + 'static,
{
    /// Shares a jobserver token pool across every command this tool runs.
    ///
    /// `pool` is the host-provided shared pool (a `fifo:` jobserver). Each
    /// command — foreground or background — gets its own fifo beneath
    /// `fifo_dir` (created if missing, broker-owned, kept out of command
    /// working directories), exported through
    /// `MAKEFLAGS=--jobserver-auth=fifo:PATH`. The broker moves tokens
    /// between the pool and each fifo on demand and returns every token
    /// once the command's whole process tree is gone: the fifo's read end
    /// reports EOF only when its last writer — including surviving
    /// grandchildren — has exited. Commands never see the shared pool
    /// directly, so killed commands cannot leak tokens.
    ///
    /// A container command's fifo is made inside the container, where its
    /// `MAKEFLAGS` names it, and brokered against the same pool by a relay
    /// run with the image's managed `python3`, with the same accounting. An
    /// SSH command runs on another machine, outside the pool: it takes no
    /// token and gets no fifo.
    ///
    /// # Errors
    /// Returns an error when the fifo directory cannot be created or the
    /// broker's readiness poller cannot be set up.
    #[cfg(all(unix, feature = "jobserver"))]
    pub fn with_jobserver_pool(
        mut self,
        pool: JobPool,
        fifo_dir: impl AsRef<Path>,
    ) -> Result<Self, TerminalError> {
        let (broker, service) = job_broker_channel(pool, fifo_dir)?;
        self.executor
            .spawn(async move { service.serve().await })
            .detach();
        self.jobserver = Some(broker);
        Ok(self)
    }

    /// A handle on the foreground window, shared with every terminal derived
    /// from this one: a longer window put in force through it applies to the
    /// next foreground wait of each of them.
    #[must_use]
    pub fn foreground_wait(&self) -> ForegroundWait {
        self.limits.foreground_wait.clone()
    }

    /// A handle on the response window, shared with every terminal derived
    /// from this one: a window set through it applies to the next final
    /// warning of each of them.
    #[must_use]
    pub fn response_window(&self) -> ResponseWindow {
        self.limits.response_window.clone()
    }

    /// Reads job footprints and the machine's physical memory through
    /// `reader` instead of the platform's own.
    #[must_use]
    pub fn with_footprint_reader(mut self, reader: impl FootprintReader) -> Self {
        self.limits.footprints = AnyFootprintReader::new(reader);
        self
    }

    /// Shares the memory this terminal's jobs hold with the machine's other
    /// agents through `registry`, so a final warning can name the agent that
    /// contributes the most when all agents together exceed physical memory.
    #[must_use]
    pub fn with_footprint_registry(mut self, registry: impl FootprintRegistry) -> Self {
        self.limits.footprint_registry = Some(AnyFootprintRegistry::new(registry));
        self
    }

    /// The memory the terminal's jobs hold, for a guard that warns about and
    /// kills a runaway job.
    #[must_use]
    ///
    /// The machine the embedder runs on is always judged. When the terminal
    /// runs in a container, the container's own memory limit is judged too,
    /// read inside the container. Jobs of a remote execution environment are
    /// never judged.
    pub fn job_memory(&self) -> JobMemory {
        let memory = JobMemory::new(
            self.job_registry(),
            self.limits.footprints.clone(),
            self.limits.footprint_registry.clone(),
            self.limits.response_window.clone(),
        );
        match self.container_probe() {
            Some(probe) => {
                memory.with_container(AnyContainerMemory::new(ProbedContainerMemory::new(probe)))
            }
            None => memory,
        }
    }

    /// A reader of the container the agent works in, when the terminal's
    /// home environment is a container; `None` when it is the local
    /// machine. A remote execution environment is never a home environment.
    #[must_use]
    pub fn container_probe(&self) -> Option<crate::ContainerProbe> {
        match self.sandbox.backend() {
            Backend::Container { runtime, .. } => Some(runtime.probe()),
            Backend::Host { .. } | Backend::Ssh { .. } => None,
        }
    }

    /// A handle on the most bytes a subagent's sandbox may send the host
    /// while its commits are imported, shared with every subagent this
    /// terminal launches: a change through it applies to the next import of
    /// each of them.
    #[must_use]
    pub fn subagent_import_budget(&self) -> ImportBudget {
        self.limits.subagent_import.clone()
    }

    /// Records the sleep and offline intervals `connectivity` reports during
    /// each command, and states them in the command's result and, for a
    /// background job, its completion alert.
    ///
    /// Terminals launched for subagents share the record. Only commands that
    /// start after this call are recorded.
    #[cfg(feature = "connectivity")]
    #[must_use]
    pub fn with_connectivity(mut self, connectivity: &aither_connectivity::Connectivity) -> Self {
        self.outages = Outages::watching(connectivity, &self.executor);
        self
    }

    /// The SSH servers scripts may name as their `location`, shared with
    /// child terminals.
    #[must_use]
    pub const fn shell_sessions(&self) -> &ShellSessionRegistry {
        &self.sessions
    }

    /// Returns the working directory path.
    pub const fn working_dir(&self) -> &PathBuf {
        &self.working_dir
    }

    /// Returns the outputs directory path.
    pub fn outputs_dir(&self) -> PathBuf {
        self.working_dir.join("outputs")
    }

    /// Returns the output store.
    pub const fn output_store(&self) -> &OutputStore {
        &self.output_store
    }

    /// The allocator this terminal leases its model-facing ids from.
    pub const fn id_allocator(&self) -> &AnyIdAllocator {
        &self.ids
    }

    /// The job registry handle.
    pub(crate) fn job_registry(&self) -> JobRegistry {
        self.job_registry.clone()
    }

    /// The largest `timeout` a script of this shell accepts now.
    pub(crate) fn timeout_cap(&self) -> Duration {
        self.limits.timeout_cap.get()
    }

    /// This terminal's jobs.
    #[must_use]
    pub fn jobs(&self) -> Jobs {
        Jobs::new(
            self.job_registry(),
            self.events(),
            self.output_store.clone(),
        )
    }

    /// What a watch task reports through; `monitor` clones it for every pump
    /// it spawns.
    pub(crate) fn event_sender(&self) -> EventSender {
        self.events_tx.clone()
    }

    /// The executor backing this terminal, for spawning watch tasks.
    pub const fn executor(&self) -> &E {
        &self.executor
    }

    /// Paths mounted readable for commands, besides the working directory.
    /// The shared runtime grant handle: adding a grant through it affects
    /// every command launched afterwards, on this terminal and its clones.
    pub fn grants(&self) -> Grants {
        self.sandbox.grants().clone()
    }

    ///
    /// The facts about this terminal's jobs. Every handle shares the queues,
    /// so an event reaches one reader.
    pub fn events(&self) -> ShellEvents {
        self.events_rx.clone()
    }

    /// The events as tests read them.
    #[cfg(test)]
    pub(crate) fn background_receiver(&self) -> ShellEvents {
        self.events()
    }
}

/// The directory, inside a launching terminal's working directory, that holds
/// the spaces of the subagents it launches.
const SUBAGENT_SPACES_DIR: &str = "subagent-spaces";

impl<E> Shell<E>
where
    E: Executor + Clone + 'static,
{
    /// Launches the shell of a subagent: it runs in a space of its own
    /// inside this shell's working directory and never exceeds this shell's
    /// authority.
    ///
    /// `grant` is everything the subagent may write beyond its space, and
    /// must lie inside what this shell can write. When this shell has
    /// [`SubagentSpaces`], they prepare the space first and are told
    /// `caller`, the directory the launch is asked for from (this shell's
    /// working directory when `None`). Grants are resolved one at a time, so
    /// what this shell handed out is recorded in one place. The subagent's
    /// shell shares this shell's approver, output store, events and
    /// policies' definitions, with state of its own, and its own jobs.
    ///
    /// `name` is the subagent's name beneath this shell's agent: every event
    /// its shell raises carries that [`AgentPath`], where the main shell's
    /// carry [`AgentPath::main`]. The subagent's running jobs are killed when
    /// the last handle of its shell is dropped, which a main shell never
    /// does.
    ///
    /// # Errors
    /// Returns an error when `grant` names a path this shell cannot write or
    /// a commit it cannot grant, when a container shell has no workspace
    /// mount, or when the subagent's space cannot be created or prepared.
    pub async fn subagent(
        &self,
        grant: SubagentGrant,
        caller: Option<&Path>,
        name: impl Into<String>,
    ) -> Result<Self, SubagentGrantError> {
        let resolved = {
            let mut history = self.launches.lock().await;
            self.resolve_launch(&mut history, grant).await?
        };
        self.launch_resolved(resolved, caller, name.into()).await
    }

    /// Resolves the grant of a subagent this terminal launches: it never
    /// exceeds what this terminal can write or commit. `history` records what
    /// the main agent's terminal handed out.
    ///
    /// # Errors
    /// Returns an error when `grant` names a path this terminal cannot write
    /// or a commit it cannot grant, when the grant cannot be resolved, or when
    /// a container session has no workspace mount.
    async fn resolve_launch(
        &self,
        history: &mut LaunchHistory,
        grant: SubagentGrant,
    ) -> Result<ResolvedLaunch, SubagentGrantError> {
        // Grants are compared by real path, so every path of the subagent's
        // view is canonical.
        let working_dir = canonical_path(&self.working_dir).await?;
        let mut authority = vec![working_dir.clone()];
        for path in &self.sandbox.grants().snapshot().writable {
            authority.push(canonical_path(path).await?);
        }
        let spaces_root = working_dir.join(SUBAGENT_SPACES_DIR);
        let mount = self.container_mount().await?;
        let granted = grant
            .resolve(
                Launcher {
                    writable: &authority,
                    committing: self
                        .subagent
                        .as_ref()
                        .map(|subagent| subagent.committing.as_slice()),
                    spaces_root: &spaces_root,
                    spelling: mount
                        .as_ref()
                        .map_or_else(Spelling::host, Spelling::through),
                },
                history,
            )
            .await?;
        {
            let mut committed = self.committed.lock().expect("committed worktrees");
            for worktree in &granted.committing {
                if !committed.contains(worktree) {
                    committed.push(worktree.clone());
                }
            }
        }
        Ok(ResolvedLaunch {
            authority,
            spaces_root,
            mount,
            granted,
        })
    }

    /// Creates the terminal tool of a subagent this terminal launches with
    /// the grant `resolved` holds.
    ///
    /// The subagent shares the approver, output store and SSH servers, binds
    /// the script policies to state of its own and has its own launch
    /// history and background task tracking. It works in a space of its own
    /// inside this terminal's working directory, which this terminal's
    /// [`SubagentSpaces`] fill first; everything this terminal can write is
    /// read-only to it, apart from what the grant hands over.
    ///
    /// # Errors
    /// Returns an error when a container session cannot express the
    /// subagent's paths, or when the subagent's space cannot be created or
    /// prepared.
    #[allow(clippy::too_many_lines)]
    async fn launch_resolved(
        &self,
        ResolvedLaunch {
            authority,
            spaces_root,
            mount,
            granted,
        }: ResolvedLaunch,
        caller: Option<&Path>,
        name: String,
    ) -> Result<Self, SubagentGrantError> {
        async_fs::create_dir_all(&spaces_root)
            .await
            .map_err(|source| SubagentGrantError::Space {
                path: spaces_root.clone(),
                source,
            })?;
        // The space is named by a leased word the subagent's terminal holds
        // for as long as it lives. A word a space of an earlier terminal
        // already has stays leased until a free one is found, so the
        // allocator hands out another.
        let mut taken = Vec::new();
        let (space_name, space, container) = loop {
            let name = self.ids.lease()?;
            let space = spaces_root.join(name.as_str());
            let container = mount
                .as_ref()
                .map(|mount| container_confinement_for(mount, &space, &granted.writable))
                .transpose()?;
            match async_fs::create_dir(&space).await {
                Ok(()) => break (name, space, container),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    taken.push(name);
                }
                Err(source) => {
                    return Err(SubagentGrantError::Space {
                        path: space,
                        source,
                    });
                }
            }
        };
        drop(taken);
        let prepared = match &self.subagent_spaces {
            Some(spaces) => {
                let launcher = &authority[0];
                let run = RunSpace {
                    space: &space,
                    launcher,
                    caller: caller.unwrap_or(launcher),
                };
                match prepare_space(spaces, run).await {
                    Ok(prepared) => Some(prepared),
                    Err(error) => {
                        // No run ever worked in the space, so it is no one's
                        // to keep.
                        if let Err(removal) = async_fs::remove_dir_all(&space).await {
                            warn!(
                                space = %space.display(),
                                %removal,
                                "cannot remove a subagent space whose preparation failed"
                            );
                        }
                        return Err(error);
                    }
                }
            }
            None => None,
        };

        // The subagent's jobs raise their facts on the parent's handle, as
        // the agent named beneath this shell's.
        let agent = self.events_tx.agent().child(name);
        let events_rx = self.events_rx.clone();
        let events_tx = events_rx.sender(agent);
        let running_tasks = Arc::new(AtomicUsize::new(0));
        let (job_registry, job_registry_service) =
            job_registry_channel(self.output_store.policy().clone());
        self.executor
            .spawn(async move { job_registry_service.serve().await })
            .detach();
        let sandbox = self
            .sandbox
            .derived(GrantSnapshot {
                readable: authority,
                writable: granted.writable.clone(),
                ..GrantSnapshot::default()
            })
            .with_working_dir(space.clone())
            .with_scanner(job_registry.scanner());
        let jobs = Arc::new(KillJobsOnDrop::new(
            job_registry.clone(),
            self.executor.clone(),
        ));
        let readable_paths = sandbox.grants().snapshot().readable.clone();
        let git = match (&container, mount, self.sandbox.backend()) {
            (Some(confinement), Some(mount), Backend::Container { runtime, launch }) => {
                SubagentGit::new(Arc::new(subagent_git::ContainerGit {
                    executor: self.executor.clone(),
                    launch: launch.clone(),
                    runtime: runtime.clone(),
                    confinement: confinement.clone(),
                    mount,
                    priority: self.sandbox.priority(),
                }))
            }
            _ => SubagentGit::new(Arc::new(subagent_git::LocalGit {
                executor: self.executor.clone(),
                working_dir: space.clone(),
                writable: granted.writable,
                readable: readable_paths,
                committing: granted.committing.clone(),
                priority: self.sandbox.priority(),
                heel: self
                    .sandbox
                    .host_heel()
                    .expect("a subagent outside a container runs on the host")
                    .clone(),
            })),
        };
        Ok(Self {
            working_dir: space,
            sessions: self.sessions.clone(),
            approver: self.approver.clone(),
            executor: self.executor.clone(),
            ids: self.ids.clone(),
            output_store: self.output_store.clone(),
            job_registry,
            events_tx,
            events_rx,
            job_tools: self.job_tools.clone(),
            interpreters: self.interpreters.clone(),
            limits: self.limits.clone(),
            running_tasks,
            sandbox,
            subagent: Some(SubagentAuthority {
                _jobs: jobs,
                _space_name: space_name,
                committing: granted.committing,
                git,
                prepared,
            }),
            carried_dir: None,
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: self.jobserver.clone(),
            outages: self.outages.clone(),
            subagent_spaces: self.subagent_spaces.clone(),
            policies: self.policies.for_agent(),
            treat_output: self.treat_output.clone(),
            ipc: self.ipc.clone(),
            committed: self.committed.clone(),
            launches: Arc::default(),
            host_launch: self.host_launch.clone(),
        })
    }

    /// What this terminal's launcher prepared in its space; `None` for the
    /// main agent's terminal and when the launcher has no
    /// [`SubagentSpaces`].
    #[must_use]
    pub fn prepared_space(&self) -> Option<PreparedSpace> {
        self.subagent
            .as_ref()
            .and_then(|subagent| subagent.prepared.clone())
    }

    /// The workspace mount of this terminal's home container, with its host
    /// side canonical, spelling every path of a subagent's view inside the
    /// container. `None` outside a container.
    async fn container_mount(&self) -> Result<Option<MountRoot>, SubagentGrantError> {
        let Backend::Container { runtime, .. } = self.sandbox.backend() else {
            return Ok(None);
        };
        let mount = runtime
            .workspace_mount()
            .ok_or(SubagentGrantError::NoContainerMount)?;
        // The mount is compared by real path, like everything it maps.
        Ok(Some(MountRoot::new(
            canonical_path(&mount.host_root).await?,
            mount.container_root.clone(),
        )))
    }
}

/// Has `spaces` prepare `run`'s space and confines what they name to it.
async fn prepare_space(
    spaces: &AnySubagentSpaces,
    run: RunSpace<'_>,
) -> Result<PreparedSpace, SubagentGrantError> {
    spaces
        .prepare(run)
        .await
        .map_err(|source| SubagentGrantError::Preparation {
            space: run.space.to_path_buf(),
            source,
        })?
        .confine(run.space)
        .await
}

/// The sandbox a subagent's commands run in inside a container: its space and
/// granted paths, spelled through the container's workspace mount.
fn container_confinement_for(
    mount: &MountRoot,
    space: &Path,
    writable: &[PathBuf],
) -> Result<ContainerConfinement, SubagentGrantError> {
    let map = |path: &Path| {
        mount
            .container_path(path)
            .map_err(|source| SubagentGrantError::OutsideContainerMount {
                path: path.to_path_buf(),
                source,
            })
    };
    Ok(ContainerConfinement {
        working_dir: map(space)?,
        grants: writable
            .iter()
            .map(|path| Ok((map(path)?, heel::Access::WRITE)))
            .collect::<Result<_, SubagentGrantError>>()?,
    })
}

impl<E> Shell<E>
where
    E: Executor + Clone + 'static,
{
    /// Lists the commits of the subagent this terminal serves, when it holds a
    /// commit grant.
    #[must_use]
    pub fn commit_ledger(&self) -> Option<CommitLedger> {
        self.subagent.as_ref().and_then(|subagent| {
            CommitLedger::new(
                &subagent.committing,
                subagent.git.clone(),
                ImportLimits {
                    bytes: self.limits.subagent_import.clone(),
                    time: self.limits.foreground_wait.get(),
                },
            )
        })
    }
}

/// The execution path a `terminal` call and a [`Shell::run`] share.
impl<E> Shell<E>
where
    E: Executor + Clone + 'static,
{
    /// The server `location` names, or `None` for the home environment.
    fn resolve_location(
        &self,
        location: Option<&str>,
    ) -> Result<Option<Arc<SshEnvironment>>, TerminalError> {
        let Some(name) = location.map(str::trim).filter(|name| !name.is_empty()) else {
            return Ok(None);
        };
        self.sessions
            .ssh_environments()
            .get(name)
            .map(Some)
            .ok_or_else(|| TerminalError::UnknownLocation(name.to_string()))
    }

    fn resolve_home_target(&self, mode: SandboxMode) -> Result<ExecutionTarget, TerminalError> {
        let backend = match self.sandbox.backend() {
            Backend::Host { .. } => ExecutionBackend::Local {
                shell: self
                    .host_launch
                    .as_ref()
                    .expect("a shell on the host is built with its host shell")
                    .shell()
                    .clone(),
            },
            Backend::Container {
                runtime,
                #[cfg(all(unix, feature = "jobserver"))]
                launch,
                ..
            } => {
                if mode == SandboxMode::Unsandboxed {
                    return Err(TerminalError::InvalidTarget(
                        "unsafe mode is only available in heel local runtime".into(),
                    ));
                }
                ExecutionBackend::Container {
                    runtime: runtime.clone(),
                    #[cfg(all(unix, feature = "jobserver"))]
                    launch: launch.clone(),
                }
            }
            Backend::Ssh { .. } => unreachable!("a shell is never built on an SSH server"),
        };
        Ok(ExecutionTarget { mode, backend })
    }

    /// A command on `environment` in `mode`. Default mode needs the server's
    /// `heel`.
    ///
    /// # Errors
    /// Fails when the server's first preparation fails; the next command
    /// prepares again.
    async fn resolve_ssh_target(
        &self,
        environment: &SshEnvironment,
        mode: SandboxMode,
    ) -> Result<ExecutionTarget, TerminalError> {
        let prepared = environment.prepared().await?;
        let server = environment.server();
        let heel = match (mode, &prepared.heel) {
            (SandboxMode::Sandboxed, Some(heel)) => Some(heel.clone()),
            (SandboxMode::Sandboxed, None) => {
                return Err(TerminalError::NoRemoteSandbox(server.name.clone()));
            }
            (SandboxMode::Unsandboxed, _) => None,
        };
        Ok(ExecutionTarget {
            mode,
            backend: ExecutionBackend::Ssh {
                client: self.sessions.ssh_environments().client().clone(),
                server: server.clone(),
                heel,
                launch: prepared.launch.clone(),
            },
        })
    }

    /// The worktrees the main agent's scripts may reach through the host's
    /// `git` shim, in grant order; a subagent's shell reaches none.
    fn committed_worktrees(&self) -> Vec<CommittingWorktree> {
        if self.subagent.is_some() {
            return Vec::new();
        }
        self.committed.lock().expect("committed worktrees").clone()
    }

    fn spawn_terminal_execution(&self, request: TerminalSpawnRequest) -> SpawnedTerminal {
        let TerminalSpawnRequest {
            target,
            task_id,
            execution_id,
            script,
            interpreter,
            expect,
            resolution,
            max_lines,
            stdout,
            background,
            timeout,
            carry,
            call,
        } = request;
        let ExecutionTarget { mode, backend } = target;
        let (result_tx, result_rx) = async_channel::bounded(1);
        let (taken_tx, taken_rx) = async_channel::bounded(1);
        let (startup_tx, startup_rx) = async_channel::bounded(1);
        // Every job gets a real stdin pipe: a foreground command that blocks
        // on it promotes to a background task the model can feed through
        // `terminal_input`, and a blocked episode raises an alert.
        let stdin_alerts = Some(StdinAlerts {
            tx: self.events_tx.clone(),
            task_id: task_id.to_string(),
            script: script.clone(),
        });
        let background_startup = BackgroundStartup::new(startup_tx);
        self.spawn_kill_watchdog(&task_id, timeout, background_startup.settled());
        self.spawn_long_job_alert(&task_id);

        let info = crate::LaunchInfo::new(task_id.to_string(), call.clone(), mode);
        let lifetime = if background {
            JobLifetime::Background
        } else {
            JobLifetime::Foreground {
                alerts: self.events_tx.clone(),
                task_id: task_id.to_string(),
                call,
            }
        };
        let running_tasks = self.running_tasks.clone();
        let completed_tx = self.events_tx.clone();
        let outages = self.outages.clone();
        outages.begin(&task_id);

        running_tasks.fetch_add(1, Ordering::AcqRel);
        self.executor
            .spawn(execute_terminal_task(SpawnedTerminalTask {
                sandbox: self.sandbox.clone(),
                executor: self.executor.clone(),
                policies: self.policies.clone(),
                ipc: self.ipc.clone(),
                committed: self.committed_worktrees(),
                job_registry: self.job_registry.clone(),
                task_id,
                info,
                execution_id,
                background,
                timeout,
                carry,
                carried_dir: self.carried_dir.clone(),
                script,
                interpreter,
                mode,
                backend,
                stdin_alerts,
                lifetime,
                background_startup,
                expect,
                resolution,
                outputs: self.output_store.clone(),
                max_lines,
                stdout,
                result_tx,
                taken: taken_rx,
                completed_tx,
                running_tasks,
                #[cfg(all(unix, feature = "jobserver"))]
                jobserver: self.jobserver.clone(),
                outages,
            }))
            .detach();

        SpawnedTerminal {
            result: result_rx,
            startup: startup_rx,
            taken: taken_tx,
        }
    }
}

/// The model-facing tool of a [`Shell`]: the embedder's name and description
/// over the shell's execution, with an argument schema that lists exactly the
/// interpreters the shell was configured with.
#[derive(Debug, Clone)]
pub struct ShellTool<E> {
    shell: Shell<E>,
    identity: crate::builtin::ToolIdentity,
}

impl<E: Executor + Clone + 'static> Tool for ShellTool<E> {
    fn name(&self) -> Cow<'static, str> {
        self.identity.name.clone()
    }

    fn description(&self) -> Cow<'static, str> {
        self.identity.description.clone()
    }

    fn arguments_schema(&self) -> schemars::Schema {
        let mut schema = schemars::schema_for!(ShellArgs);
        let interpreters = self
            .shell
            .interpreters
            .iter()
            .map(|interpreter| {
                serde_json::to_value(interpreter).expect("an interpreter serializes as a name")
            })
            .collect::<Vec<_>>();
        let definition = schema
            .get_mut("$defs")
            .and_then(serde_json::Value::as_object_mut)
            .and_then(|definitions| definitions.get_mut("Interpreter"))
            .expect("the schema of ShellArgs defines its interpreter");
        *definition = serde_json::json!({
            "type": "string",
            "enum": interpreters,
        });
        self.identity.describe(schema)
    }

    type Arguments = ShellArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        arguments: Self::Arguments,
        cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        self.shell.call(arguments, cx).await
    }
}

impl<E: Executor + Clone + 'static> Shell<E> {
    /// Runs `script` to its end, through the foreground window and past it,
    /// and returns its output and how it ended; a script its timeout killed
    /// ends [`CommandOutcome::Killed`].
    ///
    /// The run is under the shell's whole policy: its grants, network
    /// policy, backend, policies, priority, environment and jobserver. It
    /// starts where `script` says and leaves the directory the shell carries
    /// between calls alone. It is a job of the shell, listed and killable
    /// like any other, but never announced as one: its end is told to the
    /// caller alone.
    ///
    /// # Errors
    /// Returns an error when the timeout is out of range, a policy refuses
    /// the script, the directory does not map into the shell's container, or
    /// the execution fails.
    pub async fn run(&self, script: Script) -> Result<crate::ScriptResult, ShellRunError> {
        let mut shell = self.clone();
        shell.carried_dir = None;
        shell.run_to_end(script).await
    }

    /// The model-facing tool of this shell, named and described as the
    /// embedder's [`ShellToolSettings`] say.
    #[must_use]
    pub fn tool(&self) -> ShellTool<E> {
        ShellTool {
            shell: self.clone(),
            identity: self.job_tools.shell.clone(),
        }
    }

    /// Runs one `terminal` call: the script of `arguments` in its
    /// interpreter, in the foreground window or detached.
    ///
    /// # Errors
    /// Returns an error when the arguments are out of range, the interpreter
    /// is not one the shell is configured with, a policy refuses the script
    /// or the execution fails.
    pub async fn call(
        &self,
        arguments: ShellArgs,
        cx: ToolContext,
    ) -> aither_core::Result<ToolResult> {
        match self.execute(arguments, cx).await? {
            Execution::Finished(result) => result.into_tool_result(),
            Execution::Detached(detached) => detached.into_command_result().into_tool_result(),
        }
    }

    /// Runs one call to the point it answers: its result, or the job it
    /// detached.
    async fn execute(
        &self,
        arguments: ShellArgs,
        mut cx: ToolContext,
    ) -> aither_core::Result<Execution> {
        if !self.interpreters.contains(&arguments.interpreter) {
            return Err(crate::Refused::call(
                &**self.wording(),
                CallRefusal::InterpreterNotConfigured {
                    interpreter: arguments.interpreter,
                },
            ));
        }
        let cap = self.limits.timeout_cap.get();
        let timeout = Duration::from_secs(arguments.timeout);
        if arguments.timeout == 0 || timeout > cap {
            return Err(crate::Refused::call(
                &**self.wording(),
                CallRefusal::TimeoutOutOfRange { cap },
            ));
        }
        let prepared = self.prepare_execution(&arguments, &mut cx).await?;
        // A detached job cannot report a directory carry: its final
        // directory is unknowable when the call returns.
        let carry = if arguments.background {
            None
        } else {
            self.cwd_carry(&prepared, &arguments.script, arguments.interpreter)
        };
        let spawned = self.spawn_terminal_execution(TerminalSpawnRequest {
            target: prepared.target,
            task_id: prepared.task_id.clone(),
            execution_id: prepared.execution_id,
            script: arguments.script,
            interpreter: arguments.interpreter,
            expect: OutputFormat::Auto,
            resolution: arguments.resolution,
            max_lines: prepared.max_lines,
            stdout: StdoutTreatment::of_call(arguments.raw, prepared.treatment),
            background: arguments.background,
            timeout,
            carry: carry.clone(),
            call: cx.identity().cloned(),
        });
        // Registration precedes `report_ready` on every backend, so once
        // startup returns the job is killable through the registry and the
        // foreground clock only measures the command's own runtime.
        wait_for_background_startup(&spawned.startup, &**self.wording()).await?;
        if arguments.background {
            return detach_job(
                &self.job_registry,
                &prepared.outputs,
                &prepared.task_id,
                prepared.max_lines,
                &self.events_tx,
            )
            .await
            .map(Execution::Detached);
        }
        let outcome =
            wait_for_foreground_result(&spawned.result, self.limits.foreground_wait.get()).await;
        // A result that raced in at the window's edge counts as a
        // foreground finish.
        let outcome = outcome.or_else(|| spawned.result.try_recv().ok());
        match outcome {
            Some(Ok(result)) => {
                let killed = self
                    .job_registry
                    .get_by_task_id_settled(&prepared.task_id)
                    .await
                    .and_then(|job| job.status.kill_cause());
                let command = result
                    .into_script_result(killed)
                    .into_command_result(&prepared.task_id, &self.output_store);
                spawned.took_result().await;
                Ok(Execution::Finished(command))
            }
            Some(Err(detail)) => {
                spawned.took_result().await;
                Err(failed_run(&**self.wording(), &detail))
            }
            // The command outlived its foreground window: promote it to a
            // background task.
            None => detach_job(
                &self.job_registry,
                &prepared.outputs,
                &prepared.task_id,
                prepared.max_lines,
                &self.events_tx,
            )
            .await
            .map(Execution::Detached),
        }
    }
}

/// What a call answered.
enum Execution {
    /// The script ended inside the foreground window.
    Finished(CommandResult),
    /// The script became a job.
    Detached(Detached),
}

/// A job a call detached, with the output file it was redirected to.
struct Detached {
    task_id: String,
    redirect: BackgroundRedirect,
    /// Holds the job until the caller took it over.
    watch: crate::job_registry::JobWatch,
}

impl Detached {
    /// The result the model reads for the job's start.
    fn into_command_result(self) -> CommandResult {
        let Self {
            task_id, redirect, ..
        } = self;
        let stdout = if redirect.truncated {
            CommandStream::Stored {
                file: redirect.file.clone(),
                preview: redirect.preview,
                truncated: true,
            }
        } else {
            CommandStream::Inline {
                text: redirect.preview,
            }
        };
        CommandResult {
            stdout,
            stderr: None,
            interruptions: Vec::new(),
            outcome: CommandOutcome::Running {
                task_id,
                output: redirect.file,
            },
        }
    }
}

impl<E: Executor + Clone + 'static> Shell<E> {
    /// The directory report of a foreground command on the main agent's
    /// terminal; `None` for a subagent, for an ssh command and for a container
    /// whose workspace mount is not declared.
    fn cwd_carry(
        &self,
        prepared: &ExecutionPreparation,
        script: &str,
        interpreter: Interpreter,
    ) -> Option<CwdCarry> {
        let carried = self.carried_dir.as_ref()?;
        let mount = match &prepared.target.backend {
            ExecutionBackend::Local { .. } if !cfg!(windows) => None,
            ExecutionBackend::Container { runtime, .. } => Some(runtime.workspace_mount()?.clone()),
            _ => return None,
        };
        // Where admission found the script starts, as the policies saw it;
        // a directory removed since it was recorded no longer carries.
        let (start, removed) = match (&prepared.carried_start, carried.get()) {
            (Some(dir), _) => (Some(dir.clone()), false),
            (None, Some(_)) => {
                carried.set(None);
                (None, true)
            }
            (None, None) => (None, false),
        };
        let report = self.working_dir.join(format!(".cwd-{}", prepared.task_id));
        let container = match mount {
            None => None,
            Some(mount) => {
                // The report must lie in the mounted workspace for the host
                // to read it back.
                let container_report = mount.container_path(&report).ok()?;
                let container_start = start
                    .as_ref()
                    .and_then(|start| mount.container_path(start).ok())
                    .unwrap_or_else(|| {
                        ContainerPath::new(CONTAINER_EXEC_WORKING_DIR)
                            .expect("the exec directory is absolute")
                    });
                Some(ContainerCarry {
                    mount,
                    report: Some(container_report),
                    start: container_start,
                })
            }
        };
        // A Python script runs from a shell wrapper, so its report is written
        // like a shell script's.
        let shell =
            container.is_some() || interpreter == Interpreter::Python || runs_in_shell(script);
        Some(CwdCarry {
            start,
            report: Some(report),
            container,
            shell,
            removed,
        })
    }

    /// Makes the directory a finished foreground command ended in the start
    /// of the next one, and says what became of the carried directory. In
    /// production the task consumes the report (see
    /// [`execute_script_standalone`]); tests drive the same logic here.
    #[cfg(test)]
    async fn record_carried_dir(&self, carry: &CwdCarry, succeeded: bool) -> Settled {
        let ctx = CarryContext {
            carried_dir: self.carried_dir.as_ref(),
            working_dir: &self.working_dir,
            grants: self.sandbox.grants().snapshot(),
        };
        settle_cwd(carry, take_cwd_report(carry).await, succeeded, &ctx).await
    }
}

/// A host directory of a container script as the container names it: through
/// the workspace mount when `host` lies in it, otherwise the directory the
/// container starts in.
fn container_script_dir(mount: Option<&MountRoot>, host: Option<PathBuf>, own: &Path) -> ScriptDir {
    let inside = host
        .as_ref()
        .zip(mount)
        .and_then(|(host, mount)| mount.container_path(host).ok())
        .unwrap_or_else(|| {
            ContainerPath::new(CONTAINER_EXEC_WORKING_DIR).expect("the exec directory is absolute")
        });
    ScriptDir::Container {
        inside,
        host: host.unwrap_or_else(|| own.to_path_buf()),
    }
}

/// How the backend of a script names the directory the next foreground
/// script starts in.
#[derive(Clone)]
enum DirSpelling {
    Host,
    Container(Option<MountRoot>),
    Remote(String),
}

impl DirSpelling {
    fn of(backend: &ExecutionBackend) -> Self {
        match backend {
            ExecutionBackend::Local { .. } => Self::Host,
            ExecutionBackend::Container { runtime, .. } => {
                Self::Container(runtime.workspace_mount().cloned())
            }
            ExecutionBackend::Ssh { server, .. } => Self::Remote(server.name.clone()),
        }
    }

    /// `host`, the directory carried as settled under the lock of the
    /// carried directory, or the terminal's own directory `own` when none is.
    fn dir(&self, host: Option<PathBuf>, own: &Path) -> ScriptDir {
        match self {
            Self::Host => ScriptDir::Host(host.unwrap_or_else(|| own.to_path_buf())),
            Self::Container(mount) => container_script_dir(mount.as_ref(), host, own),
            Self::Remote(server) => ScriptDir::Remote {
                server: server.clone(),
            },
        }
    }
}

/// The directory `carried` holds, while it still exists: where the next
/// foreground script starts.
async fn carried_start(carried: Option<&CarriedDir>) -> Option<PathBuf> {
    let dir = carried?.get()?;
    async_fs::metadata(&dir)
        .await
        .is_ok_and(|meta| meta.is_dir())
        .then_some(dir)
}

/// Shared state needed to settle a finished command's directory report,
/// whether the command finished inside its foreground call or promoted
/// and ended as a background task.
struct CarryContext<'a> {
    carried_dir: Option<&'a CarriedDir>,
    working_dir: &'a Path,
    grants: std::sync::Arc<GrantSnapshot>,
}

/// What a finished command did to the carried directory: the fact for the
/// policies, and the directory now carried, read in the same critical
/// section that recorded it.
struct Settled {
    carry: Carry,
    carried: Option<PathBuf>,
}

/// Settles a finished command's directory report.
///
/// `report` is what [`take_cwd_report`] read: `None` when the shell wrote
/// none.
async fn settle_cwd(
    carry: &CwdCarry,
    report: Option<String>,
    succeeded: bool,
    ctx: &CarryContext<'_>,
) -> Settled {
    // A subagent's terminal carries nothing, and a run reports no directory.
    let Some(carried) = ctx.carried_dir.filter(|_| carry.report.is_some()) else {
        return Settled {
            carry: Carry::NotCarried,
            carried: ctx.carried_dir.and_then(CarriedDir::get),
        };
    };
    let (moved_to, outcome) = match report {
        Some(reported) => apply_cwd_report(carry, &reported, succeeded, ctx).await,
        // A direct command cannot change directory and writes no report.
        None if carry.shell => (None, Carry::Reset(DirectoryResetReason::Unreported)),
        None => (None, Carry::Unchanged),
    };
    let outcome = match outcome {
        Carry::Unchanged if carry.removed => Carry::Reset(DirectoryResetReason::Removed),
        other => other,
    };
    Settled {
        carry: outcome,
        carried: carried.settle(moved_to),
    }
}

/// Reads and removes a command's directory report, returning the directory
/// the shell ended in when it reported one. The file is consumed either
/// way so a promoted command never leaves it behind.
async fn take_cwd_report(carry: &CwdCarry) -> Option<String> {
    let report = carry.report.as_ref()?;
    let reported = async_fs::read_to_string(report).await;
    let _ = async_fs::remove_file(report).await;
    // A direct command writes no report; a shell that ended through
    // `exit` or a signal writes none either.
    reported
        .ok()
        .map(|reported| reported.trim_end_matches('\n').to_owned())
        .filter(|reported| !reported.is_empty())
}

/// The directory a finished command ended in as the directory the next one
/// starts in, when the command succeeded and the directory lies inside the
/// paths this terminal may read, and what became of the carried directory.
/// When the next command does not start where the shell ended, the answer
/// says why.
async fn apply_cwd_report(
    carry: &CwdCarry,
    reported: &str,
    succeeded: bool,
    ctx: &CarryContext<'_>,
) -> (Option<PathBuf>, Carry) {
    if !succeeded {
        let carry = if left_in_start(carry, reported, ctx.working_dir).await {
            Carry::Unchanged
        } else {
            Carry::Reset(DirectoryResetReason::Failed)
        };
        return (None, carry);
    }
    if let Some(container) = &carry.container {
        // A container is an unconfined main agent: what bounds its
        // directory is the workspace mount.
        return match ContainerPath::new(reported)
            .map_err(drop)
            .and_then(|dir| container.mount.host_path(&dir).map_err(drop))
        {
            Ok(dir) => {
                let moved = reported != container.start.as_str();
                (
                    Some(dir),
                    if moved {
                        Carry::Moved
                    } else {
                        Carry::Unchanged
                    },
                )
            }
            Err(()) => (None, Carry::Reset(DirectoryResetReason::OutsideWorkspace)),
        };
    }
    let Ok(dir) = canonical_path(Path::new(reported)).await else {
        return (None, Carry::Reset(DirectoryResetReason::Removed));
    };
    for grant in std::iter::once(ctx.working_dir)
        .chain(ctx.grants.writable.iter().map(PathBuf::as_path))
        .chain(ctx.grants.readable.iter().map(PathBuf::as_path))
    {
        if canonical_path(grant)
            .await
            .is_ok_and(|grant| dir.starts_with(grant))
        {
            let start = match &carry.start {
                Some(start) => Some(start.clone()),
                None => canonical_path(ctx.working_dir).await.ok(),
            };
            let moved = start.as_ref() != Some(&dir);
            return (
                Some(dir),
                if moved {
                    Carry::Moved
                } else {
                    Carry::Unchanged
                },
            );
        }
    }
    (None, Carry::Reset(DirectoryResetReason::OutsideReadable))
}

/// Whether the shell ended in the directory the command started in.
async fn left_in_start(carry: &CwdCarry, reported: &str, working_dir: &Path) -> bool {
    if let Some(container) = &carry.container {
        return reported == container.start.as_str();
    }
    let start = carry.start.as_deref().unwrap_or(working_dir);
    match (
        canonical_path(Path::new(reported)).await,
        canonical_path(start).await,
    ) {
        (Ok(reported), Ok(start)) => reported == start,
        _ => false,
    }
}

impl<E: Executor + Clone + 'static> Shell<E> {
    /// Runs `script` as a detached background job bounded by `timeout`, for
    /// embedders that drive the terminal directly rather than through the
    /// `terminal` tool schema, and returns the job. Shares [`Self::call`]'s
    /// `background: true` path, so the job is announced as one.
    ///
    /// # Errors
    ///
    /// Fails the way a `terminal` call does: an invalid `timeout`, a
    /// refused permission, a blocked command or a spawn failure.
    pub async fn run_background(&self, arguments: ShellArgs) -> aither_core::Result<crate::Job> {
        let execution = self
            .execute(
                ShellArgs {
                    background: true,
                    ..arguments
                },
                ToolContext::new(),
            )
            .await?;
        match execution {
            Execution::Detached(detached) => Ok(crate::Job::new(
                self.jobs(),
                detached.task_id,
                detached.watch,
            )),
            Execution::Finished(_) => unreachable!("a background call detaches its job"),
        }
    }
}

impl<E> Shell<E>
where
    E: Executor + Clone + 'static,
{
    /// The terminal's own directory as a command sees it.
    async fn own_dir(&self) -> PathBuf {
        canonical_path(&self.working_dir)
            .await
            .unwrap_or_else(|_| self.working_dir.clone())
    }

    /// `host`, or the terminal's own directory `own` when the script starts
    /// in neither a carried nor a named one, as the home backend names it.
    fn home_script_dir(&self, host: Option<PathBuf>, own: &Path) -> ScriptDir {
        match self.sandbox.backend() {
            Backend::Container { runtime, .. } => {
                container_script_dir(runtime.workspace_mount(), host, own)
            }
            Backend::Host { .. } | Backend::Ssh { .. } => {
                ScriptDir::Host(host.unwrap_or_else(|| own.to_path_buf()))
            }
        }
    }

    /// What renders this terminal's refusals and notices: its output store's.
    pub(crate) fn wording(&self) -> &crate::AnyShellWording {
        self.output_store.wording()
    }

    /// Shared front half of every `terminal` call: argument validation, then
    /// [`Self::admit_command`].
    async fn prepare_execution(
        &self,
        arguments: &ShellArgs,
        cx: &mut ToolContext,
    ) -> anyhow::Result<ExecutionPreparation> {
        if arguments.description.trim().is_empty() {
            return Err(crate::Refused::call(
                &**self.wording(),
                CallRefusal::EmptyDescription,
            ));
        }
        let origin = if arguments.background {
            ScriptOrigin::Background
        } else {
            ScriptOrigin::Foreground
        };
        let call = cx.identity().cloned();
        self.admit_command(
            Admission {
                script: &arguments.script,
                interpreter: arguments.interpreter,
                mode: arguments.sandbox,
                location: arguments.location.as_deref(),
                max_lines: arguments.max_lines,
                origin,
                call,
                run_dir: None,
            },
            Some(cx),
        )
        .await
        .map_err(|error| error.into_tool_error(&**self.wording()))
    }

    /// The terminal's policy for a command, whoever runs it — the one entry
    /// every foreground, background, subagent and run command on every
    /// backend passes through: the script policies, target resolution and
    /// permission; then the spawned job's ids.
    ///
    /// Every script the policies were asked about is reported to them once
    /// it ends; one admission fails is reported here, unrun.
    async fn admit_command(
        &self,
        admission: Admission<'_>,
        cx: Option<&mut ToolContext>,
    ) -> Result<ExecutionPreparation, ShellRunError> {
        // Leased for as long as the command can be addressed: the registry
        // keeps a clone once the job is registered.
        let task_id = self.ids.lease()?;
        let environment = self.resolve_location(admission.location)?;
        // Decided once, here: the policies see the directory the command
        // then starts in, and so does its launch.
        let carried_start = if admission.origin == ScriptOrigin::Foreground {
            carried_start(self.carried_dir.as_ref()).await
        } else {
            None
        };
        let own = self.own_dir().await;
        let start = match &environment {
            Some(environment) => ScriptDir::Remote {
                server: environment.server().name.clone(),
            },
            None => self.home_script_dir(
                admission
                    .run_dir
                    .map(Path::to_path_buf)
                    .or_else(|| carried_start.clone()),
                &own,
            ),
        };
        let checked = self
            .check_script(&task_id, &admission, &start)
            .await
            .map_err(ShellRunError::Refused);
        let (clearance, treatment) = match checked {
            Ok(checked) => checked,
            Err(refusal) => {
                self.end_unrun(&task_id).await;
                return Err(refusal);
            }
        };
        let admitted = self
            .admit_checked(&admission, environment.as_deref(), clearance, cx)
            .await;
        if admitted.is_err() {
            self.end_unrun(&task_id).await;
        }
        let (target, max_lines) = admitted?;
        Ok(ExecutionPreparation {
            target,
            execution_id: format!("exec-{task_id}"),
            task_id,
            outputs: self.output_store.clone(),
            max_lines,
            treatment,
            carried_start,
        })
    }

    /// Tells the policies a script they were asked about never started, and
    /// where the next foreground script starts.
    async fn end_unrun(&self, task_id: &str) {
        let policies = &self.policies;
        if policies.is_empty() {
            return;
        }
        let own = self.own_dir().await;
        let end = self.home_script_dir(carried_start(self.carried_dir.as_ref()).await, &own);
        policies
            .ended(task_id, ScriptOutcome::NotRun, &end, Carry::NotCarried)
            .await;
    }

    /// Puts the script to the shell's script policies and its output
    /// treatment, the shell parsed once for all of them, and folds the
    /// policies' decisions into what admission does next.
    async fn check_script(
        &self,
        task_id: &str,
        admission: &Admission<'_>,
        start: &ScriptDir,
    ) -> Result<(Clearance, OutputTreatment), ScriptRefusal> {
        let policies = &self.policies;
        if policies.is_empty() && self.treat_output.is_none() {
            return Ok((Clearance::Pass, OutputTreatment::default()));
        }
        let shell = (admission.interpreter == Interpreter::Shell)
            .then(|| crate::ShellScript::parse(admission.script));
        let check = crate::ScriptCheck {
            task_id,
            script: admission.script,
            interpreter: admission.interpreter,
            shell: shell.as_ref().map(Result::as_ref),
            origin: admission.origin,
            sandbox: admission.mode,
            location: admission.location(),
            start,
            call: admission.call(),
        };
        let treatment = self
            .treat_output
            .as_ref()
            .map_or_else(OutputTreatment::default, |treat| treat.treatment(&check));
        let clearance = match policies.check(&check).await {
            ScriptDecision::Pass => Clearance::Pass,
            ScriptDecision::Allow => Clearance::Allow,
            ScriptDecision::Ask(request) => Clearance::Ask(request),
            ScriptDecision::Refuse(refusal) => return Err(refusal),
        };
        Ok((clearance, treatment))
    }

    /// The rest of admission once the policies did not refuse the script.
    async fn admit_checked(
        &self,
        admission: &Admission<'_>,
        environment: Option<&SshEnvironment>,
        clearance: Clearance,
        cx: Option<&mut ToolContext>,
    ) -> Result<(ExecutionTarget, usize), ShellRunError> {
        let mode = admission.mode;
        let target = match environment {
            None => {
                let target = self.resolve_home_target(mode)?;
                self.clear(admission, clearance, cx).await?;
                target
            }
            Some(environment) => {
                self.clear(admission, clearance, cx).await?;
                self.resolve_ssh_target(environment, mode).await?
            }
        };
        Ok((
            target,
            self.output_store.policy().lines(admission.max_lines),
        ))
    }

    /// Settles whether the script may run, once the policies did not refuse
    /// it: a script that crosses the sandbox boundary, unsandboxed or on a
    /// server, runs only after an [`Allow`](ScriptDecision::Allow) or an
    /// approved [`Ask`](ScriptDecision::Ask); one the policies only passed
    /// is asked about.
    async fn clear(
        &self,
        admission: &Admission<'_>,
        clearance: Clearance,
        cx: Option<&mut ToolContext>,
    ) -> Result<(), TerminalError> {
        let request = match clearance {
            Clearance::Ask(request) => request,
            Clearance::Pass if admission.crosses_boundary() => admission.boundary_request(),
            Clearance::Allow | Clearance::Pass => return Ok(()),
        };
        self.approve(request, admission.call(), cx).await
    }

    /// Puts `request` to the approver of the call that carries the script, or
    /// to the terminal's own when the script has no call to ask through:
    /// a run, a background script started directly, or a call whose
    /// dispatcher attached none.
    async fn approve(
        &self,
        request: ApprovalRequest,
        call: Option<&CallIdentity>,
        cx: Option<&mut ToolContext>,
    ) -> Result<(), TerminalError> {
        let kind = request.kind().clone();
        let answer = match cx {
            Some(cx) => match cx.approve(request.clone()).await {
                Err(ApprovalError::NoApprover { .. }) => None,
                answer => Some(answer),
            },
            None => None,
        };
        let answer = if let Some(answer) = answer {
            answer
        } else {
            let Some(approver) = &self.approver else {
                return Err(TerminalError::NoApprover(kind));
            };
            let request = match call {
                Some(call) => request.with_call(call.clone()),
                None => request,
            };
            approver
                .approve(request)
                .await
                .map_err(|failure| ApprovalError::Failed { kind, failure })
        };
        match answer? {
            Approval::Approved => Ok(()),
            Approval::Denied { reason } => Err(TerminalError::Denied(reason)),
        }
    }

    /// Runs `run` to its end on this terminal; see [`Shell::run`].
    async fn run_to_end(&self, run: Script) -> Result<crate::ScriptResult, ShellRunError> {
        let cap = self.limits.timeout_cap.get();
        if run.timeout.is_zero() || run.timeout > cap {
            return Err(ShellRunError::Timeout { cap });
        }
        // A run starts in a host directory, so it resolves the terminal's
        // home backend, never an SSH server.
        let prepared = self
            .admit_command(
                Admission {
                    script: &run.source,
                    interpreter: Interpreter::Shell,
                    mode: run.sandbox,
                    location: None,
                    max_lines: None,
                    origin: ScriptOrigin::Run,
                    call: None,
                    run_dir: Some(&run.dir),
                },
                None,
            )
            .await?;
        let start = match run_start(&prepared.target, run.dir) {
            Ok(start) => start,
            Err(error) => {
                self.end_unrun(&prepared.task_id).await;
                return Err(error);
            }
        };
        let spawned = self.spawn_terminal_execution(TerminalSpawnRequest {
            target: prepared.target,
            task_id: prepared.task_id.clone(),
            execution_id: prepared.execution_id,
            script: run.source,
            interpreter: Interpreter::Shell,
            expect: OutputFormat::Text,
            resolution: MediaResolution::Auto,
            max_lines: prepared.max_lines,
            stdout: StdoutTreatment::Cleaned(prepared.treatment),
            background: false,
            timeout: run.timeout,
            carry: Some(start),
            call: None,
        });
        match spawned.startup.recv().await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => return Err(TerminalError::Execution(error).into()),
            Err(_) => return Err(TerminalError::TaskLost.into()),
        }
        // The run is never announced as a background task, so its end is
        // told to the caller alone, however long it takes.
        let result = spawned
            .result
            .recv()
            .await
            .map_err(|_| TerminalError::TaskLost);
        let killed = self
            .job_registry
            .get_by_task_id_settled(&prepared.task_id)
            .await
            .and_then(|job| job.status.kill_cause());
        spawned.took_result().await;
        let result = result?.map_err(TerminalError::Execution)?;
        Ok(result.into_script_result(killed))
    }

    /// Kills the job through the registry once its timeout passes, the job
    /// then reading [`KillCause::Timeout`](crate::KillCause::Timeout).
    ///
    /// A bound that passes before the job registered is enforced the moment
    /// `registered` closes, which is when the job's startup was reported.
    fn spawn_kill_watchdog(&self, task_id: &IdLease, timeout: Duration, registered: Receiver<()>) {
        let job_registry = self.job_registry.clone();
        let started = jiff::Timestamp::now();
        // Weak, so a job that ended gives its word back before the timer
        // fires; a word that went back may name another job by then.
        let task_id = task_id.downgrade();
        self.executor
            .spawn(async move {
                async_io::Timer::after(timeout).await;
                // Nothing is ever sent: the channel closes once the startup
                // was reported or its flow is gone.
                let _ = registered.recv().await;
                let Some(task_id) = task_id.upgrade() else {
                    return;
                };
                if let Some(job) = job_registry.get_by_task_id(&task_id).await
                    && matches!(job.status, JobStatus::Running)
                {
                    job_registry
                        .kill_timed_out(
                            job.pid,
                            TimeoutKill {
                                timeout,
                                started,
                                killed: jiff::Timestamp::now(),
                            },
                        )
                        .await;
                }
            })
            .detach();
    }

    /// Raises one [`LongRunning`] when the job crosses the configured
    /// long-running threshold.
    fn spawn_long_job_alert(&self, task_id: &IdLease) {
        let job_registry = self.job_registry.clone();
        let alert_tx = self.events_tx.clone();
        // Weak for the same reason as the watchdog's.
        let task_id = task_id.downgrade();
        let alert_after = self.limits.long_job_alert.get();
        self.executor
            .spawn(async move {
                async_io::Timer::after(alert_after).await;
                let Some(task_id) = task_id.upgrade() else {
                    return;
                };
                if let Some(job) = job_registry.get_by_task_id(&task_id).await
                    && matches!(job.status, JobStatus::Running)
                {
                    let _ = alert_tx
                        .send(ShellEvent::LongRunning(LongRunning {
                            agent: alert_tx.agent().clone(),
                            task_id: task_id.to_string(),
                            script: job.script,
                            after: job.started_at.elapsed(),
                        }))
                        .await;
                }
            })
            .detach();
    }
}

/// What [`Shell::admit_command`] admits: a script and how it came.
struct Admission<'a> {
    script: &'a str,
    interpreter: Interpreter,
    mode: SandboxMode,
    /// The SSH server the command names; `None` or empty for home.
    location: Option<&'a str>,
    max_lines: Option<u32>,
    origin: ScriptOrigin,
    /// The tool call that carries the script and the agent that made it;
    /// `None` for a script no call carries.
    call: Option<CallIdentity>,
    /// The directory a run names; `None` for a `terminal` call.
    run_dir: Option<&'a Path>,
}

impl<'a> Admission<'a> {
    /// The tool call that carries the script and the agent that made it.
    const fn call(&self) -> Option<&CallIdentity> {
        self.call.as_ref()
    }

    /// The SSH server the command names, when it names one.
    fn location(&self) -> Option<&'a str> {
        self.location.map(str::trim).filter(|name| !name.is_empty())
    }

    /// Whether the script runs outside the sandbox, unsandboxed or on a
    /// server.
    fn crosses_boundary(&self) -> bool {
        self.mode == SandboxMode::Unsandboxed || self.location().is_some()
    }

    /// What the approver is asked about a script the policies only passed.
    fn boundary_request(&self) -> ApprovalRequest {
        let (kind, reason) = if self.mode == SandboxMode::Unsandboxed {
            ("unsandboxed", "the script runs outside the sandbox")
        } else {
            ("remote", "the script runs on a remote server")
        };
        ApprovalRequest::new(ApprovalKind::new(kind), self.script).with_reason(reason)
    }
}

/// What the policies' decisions leave admission to do, with a refusal
/// already answered.
enum Clearance {
    /// No policy decided: a script inside the sandbox runs, one that crosses
    /// its boundary is asked about.
    Pass,
    /// A policy approved the script.
    Allow,
    /// A policy asks the approver first.
    Ask(ApprovalRequest),
}

struct ExecutionPreparation {
    target: ExecutionTarget,
    task_id: IdLease,
    execution_id: String,
    outputs: OutputStore,
    max_lines: usize,
    /// How the script's output is cleaned, as the shell's
    /// [`TreatOutput`](crate::TreatOutput) chose.
    treatment: OutputTreatment,
    /// The carried directory a foreground command starts in, when the
    /// terminal carries one and it still exists.
    carried_start: Option<PathBuf>,
}

/// What the output pipeline does to a foreground script's stdout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StdoutTreatment {
    /// The call asked for the bytes as written (`raw`): no stage applies,
    /// not even rendering.
    Raw,
    /// The pipeline runs under the script's output treatment.
    Cleaned(OutputTreatment),
}

impl StdoutTreatment {
    /// The treatment of a `terminal` call's stdout: `raw` wins over the
    /// shell's choice.
    const fn of_call(raw: bool, treatment: OutputTreatment) -> Self {
        if raw {
            Self::Raw
        } else {
            Self::Cleaned(treatment)
        }
    }
}

/// Where a run starts: `dir` on the host, mapped through the workspace
/// mount for a container command. A run carries no directory, so it writes
/// no report.
fn run_start(target: &ExecutionTarget, dir: PathBuf) -> Result<CwdCarry, ShellRunError> {
    let container = match &target.backend {
        ExecutionBackend::Local { .. } => None,
        ExecutionBackend::Container { runtime, .. } => {
            let mount = runtime
                .workspace_mount()
                .cloned()
                .ok_or(ShellRunError::NoWorkspaceMount)?;
            let start = mount.container_path(&dir)?;
            Some(ContainerCarry {
                mount,
                report: None,
                start,
            })
        }
        ExecutionBackend::Ssh { .. } => unreachable!("a run resolves the home backend"),
    };
    Ok(CwdCarry {
        start: Some(dir),
        report: None,
        container,
        shell: false,
        removed: false,
    })
}

async fn wait_for_background_startup(
    startup_rx: &Receiver<Result<(), String>>,
    wording: &dyn ShellWording,
) -> anyhow::Result<()> {
    match startup_rx.recv().await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(detail)) => Err(failed_run(wording, &detail)),
        Err(_) => Err(anyhow::Error::msg(
            wording.run_failure(&RunFailure::Terminal(&TerminalError::TaskLost)),
        )),
    }
}

/// The error of a run whose execution failed with `detail`, worded.
fn failed_run(wording: &dyn ShellWording, detail: &str) -> anyhow::Error {
    anyhow::Error::msg(wording.run_failure(&RunFailure::Failed { detail }))
}

/// Races the result channel against the foreground limit.
///
/// Returns `None` when the limit expired (or the channel was dropped, which
/// the caller treats the same way — the job is finished or killed).
async fn wait_for_foreground_result(
    result_rx: &Receiver<Result<CompletedCommand, String>>,
    limit: Duration,
) -> Option<Result<CompletedCommand, String>> {
    futures_lite::future::or(async { result_rx.recv().await.ok() }, async {
        async_io::Timer::after(limit).await;
        result_rx.try_recv().ok()
    })
    .await
}

/// Detaches the running task `task_id` as a job: its output goes to a file
/// and the shell raises [`ShellEvent::JobStarted`].
async fn detach_job(
    job_registry: &JobRegistry,
    outputs: &OutputStore,
    task_id: &str,
    max_lines: usize,
    started_tx: &EventSender,
) -> aither_core::Result<Detached> {
    // The call still owns the job's completion, so the job is registered; the
    // hold keeps it past a consumer that takes the completion at once.
    let watch = job_registry
        .watch(task_id)
        .await
        .expect("a job whose call has not returned is registered");
    let redirect =
        start_background_output_redirect(job_registry, outputs, task_id, max_lines).await?;
    let _ = started_tx
        .send(ShellEvent::JobStarted(JobStarted {
            agent: started_tx.agent().clone(),
            task_id: task_id.to_string(),
            output_preview: redirect.preview.clone(),
            output_file: redirect.file.clone(),
        }))
        .await;
    Ok(Detached {
        task_id: task_id.to_string(),
        redirect,
        watch,
    })
}

/// Output file a background job was redirected to, with what it holds so far.
struct BackgroundRedirect {
    file: StoredFile,
    /// First lines already written; empty when nothing was.
    preview: String,
    truncated: bool,
}

async fn start_background_output_redirect(
    job_registry: &JobRegistry,
    outputs: &OutputStore,
    task_id: &str,
    max_lines: usize,
) -> Result<BackgroundRedirect, anyhow::Error> {
    let (url, output_path) = outputs.save_raw(&[]).await?;
    let file = StoredFile {
        url,
        path: output_path.display().to_string(),
    };
    let snapshot = job_registry
        .start_output_redirect(task_id, output_path)
        .await
        .map_err(anyhow::Error::msg)?;
    let (preview, truncated) = preview_first_lines(&snapshot, max_lines);
    Ok(BackgroundRedirect {
        file,
        preview: background_preview(&preview),
        truncated,
    })
}

/// The preview of what a background job has written so far: empty when it
/// wrote only whitespace, otherwise the text with its trailing whitespace cut
/// and the first line's indentation kept.
fn background_preview(preview: &str) -> String {
    if preview.trim().is_empty() {
        String::new()
    } else {
        preview.trim_end().to_string()
    }
}

struct StandaloneExecution<E> {
    /// The shell's sandbox: its directory, live grants, environment source,
    /// network policy, priority and home backend.
    sandbox: Sandbox,
    executor: E,
    /// The commands the job's processes may call back into the host with.
    ipc: Option<crate::AnyIpcProvider>,
    /// The worktrees the host's `git` shim maps for the command.
    committed: Vec<CommittingWorktree>,
    job_registry: JobRegistry,
    task_id: IdLease,
    /// What the job's processes are launched as.
    info: crate::LaunchInfo,
    execution_id: String,
    background: bool,
    carry: Option<CwdCarry>,
    carried_dir: Option<CarriedDir>,
    script: String,
    interpreter: Interpreter,
    mode: SandboxMode,
    backend: ExecutionBackend,
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
    background_startup: BackgroundStartup,
    #[cfg(all(unix, feature = "jobserver"))]
    jobserver: Option<JobBroker>,
    expect: OutputFormat,
    resolution: MediaResolution,
    outputs: OutputStore,
    max_lines: usize,
    stdout: StdoutTreatment,
}

struct StandaloneStart<'a, E> {
    sandbox: &'a Sandbox,
    executor: E,
    ipc: Option<&'a crate::AnyIpcProvider>,
    committed: &'a [CommittingWorktree],
    job_registry: &'a JobRegistry,
    task_id: &'a IdLease,
    info: crate::LaunchInfo,
    execution_id: &'a str,
    carry: Option<&'a CwdCarry>,
    script: &'a str,
    interpreter: Interpreter,
    mode: SandboxMode,
    backend: &'a ExecutionBackend,
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
    background_startup: &'a BackgroundStartup,
    #[cfg(all(unix, feature = "jobserver"))]
    jobserver: &'a Option<JobBroker>,
}

struct LocalBackgroundExecution<'a, E> {
    /// The shell program scripts run under.
    shell: &'a str,
    sandbox: &'a Sandbox,
    executor: E,
    ipc: Option<&'a crate::AnyIpcProvider>,
    committed: &'a [CommittingWorktree],
    task_id: &'a IdLease,
    info: crate::LaunchInfo,
    execution_id: &'a str,
    carry: Option<&'a CwdCarry>,
    script: &'a str,
    interpreter: Interpreter,
    mode: SandboxMode,
    job_registry: &'a JobRegistry,
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
    background_startup: &'a BackgroundStartup,
    #[cfg(all(unix, feature = "jobserver"))]
    job_fifo: Option<crate::jobserver::CommandFifo>,
}

struct ContainerJobExecution<'a, E> {
    executor: E,
    sandbox: &'a Sandbox,
    ipc: Option<&'a crate::AnyIpcProvider>,
    committed: &'a [CommittingWorktree],
    task_id: &'a IdLease,
    info: crate::LaunchInfo,
    execution_id: &'a str,
    script: &'a str,
    interpreter: Interpreter,
    mode: SandboxMode,
    container_runtime: &'a ContainerShellRuntime,
    carry: Option<&'a CwdCarry>,
    job_registry: &'a JobRegistry,
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
    background_startup: &'a BackgroundStartup,
    /// The command's jobserver fifo inside the container, held until its
    /// exec has ended.
    #[cfg(all(unix, feature = "jobserver"))]
    job_fifo: Option<crate::container_jobserver::ContainerJobFifo>,
}

struct SshBackgroundExecution<'a, E> {
    executor: E,
    sandbox: &'a Sandbox,
    launch: &'a RemoteLaunch,
    task_id: &'a IdLease,
    info: crate::LaunchInfo,
    execution_id: &'a str,
    script: &'a str,
    interpreter: Interpreter,
    mode: SandboxMode,
    client: &'a SshClient,
    server: &'a SshServer,
    heel: Option<&'a str>,
    job_registry: &'a JobRegistry,
    background_startup: &'a BackgroundStartup,
}

struct CompletedProcessOutput<'a> {
    job_registry: &'a JobRegistry,
    pid: u32,
    output: std::process::Output,
    background_output: bool,
    outputs: &'a OutputStore,
    expect: OutputFormat,
    resolution: MediaResolution,
    max_lines: usize,
    stdout: StdoutTreatment,
}

/// Where a command runs, with the programs the embedder prepared for exactly
/// that environment.
enum ExecutionBackend {
    /// The local sandbox.
    Local {
        /// The shell program scripts run under.
        shell: Arc<str>,
    },
    /// A container, whose programs its backend's launch names.
    Container {
        runtime: ContainerShellRuntime,
        /// Read only to open a command's jobserver fifo in the container.
        #[cfg(all(unix, feature = "jobserver"))]
        launch: crate::ContainerLaunch,
    },
    /// A remote host over SSH.
    Ssh {
        client: SshClient,
        server: SshServer,
        /// The server's `heel`, which a default-mode command runs under; an
        /// `unsafe` command has none.
        heel: Option<String>,
        launch: RemoteLaunch,
    },
}

struct ExecutionTarget {
    mode: SandboxMode,
    backend: ExecutionBackend,
}

struct TerminalSpawnRequest {
    target: ExecutionTarget,
    task_id: IdLease,
    execution_id: String,
    script: String,
    interpreter: Interpreter,
    expect: OutputFormat,
    resolution: MediaResolution,
    max_lines: usize,
    stdout: StdoutTreatment,
    background: bool,
    timeout: Duration,
    carry: Option<CwdCarry>,
    /// The `terminal` call this execution belongs to.
    call: Option<CallIdentity>,
}

struct SpawnedTerminal {
    result: Receiver<Result<CompletedCommand, String>>,
    startup: Receiver<Result<(), String>>,
    /// Tells the job's flow the call returned its result itself; dropped
    /// unsent, the job is a background task whose completion the agent takes.
    taken: Sender<Sender<()>>,
}

impl SpawnedTerminal {
    /// The call returns the job's result itself: nothing addresses the job
    /// once it does, so its flow takes the completion on the agent's behalf.
    /// Returns once the flow ended and the registry forgot the job, so the
    /// task id is back with the allocator when the call returns.
    async fn took_result(self) {
        let (done_tx, done_rx) = async_channel::bounded(1);
        // A flow that ended without a result no longer listens.
        if self.taken.send(done_tx).await.is_ok() {
            let _ = done_rx.recv().await;
        }
    }
}

struct SpawnedTerminalTask<E> {
    sandbox: Sandbox,
    executor: E,
    policies: ScriptPolicies,
    ipc: Option<crate::AnyIpcProvider>,
    committed: Vec<CommittingWorktree>,
    job_registry: JobRegistry,
    task_id: IdLease,
    /// What the job's processes are launched as.
    info: crate::LaunchInfo,
    execution_id: String,
    background: bool,
    /// The bound the job runs under.
    timeout: Duration,
    carry: Option<CwdCarry>,
    carried_dir: Option<CarriedDir>,
    script: String,
    interpreter: Interpreter,
    mode: SandboxMode,
    backend: ExecutionBackend,
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
    background_startup: BackgroundStartup,
    expect: OutputFormat,
    resolution: MediaResolution,
    outputs: OutputStore,
    max_lines: usize,
    stdout: StdoutTreatment,
    result_tx: Sender<Result<CompletedCommand, String>>,
    /// Whether the call took the result: see [`SpawnedTerminal::took_result`].
    taken: Receiver<Sender<()>>,
    completed_tx: EventSender,
    running_tasks: Arc<AtomicUsize>,
    #[cfg(all(unix, feature = "jobserver"))]
    jobserver: Option<JobBroker>,
    outages: Outages,
}

#[allow(clippy::too_many_lines)]
async fn execute_terminal_task<E>(task: SpawnedTerminalTask<E>)
where
    E: Executor + Clone + 'static,
{
    let SpawnedTerminalTask {
        sandbox,
        executor,
        policies,
        ipc,
        committed,
        job_registry,
        task_id,
        info,
        execution_id,
        background,
        timeout,
        carry,
        carried_dir,
        script,
        interpreter,
        mode,
        backend,
        stdin_alerts,
        lifetime,
        background_startup,
        expect,
        resolution,
        outputs,
        max_lines,
        stdout,
        result_tx,
        taken,
        completed_tx,
        running_tasks,
        #[cfg(all(unix, feature = "jobserver"))]
        jobserver,
        outages,
    } = task;
    // Before the job registers, so the registry reports the bound it runs under.
    job_registry.bound_by_timeout(&task_id, timeout).await;
    // Before the process starts, so a command it calls over IPC can already
    // append to this job's output.
    job_registry.launching(&task_id).await;
    let status_registry = job_registry.clone();
    // What the policies are told the next script starts in, spelled as this
    // script's backend names a directory.
    let end_view = (!policies.is_empty()).then(|| {
        (
            DirSpelling::of(&backend),
            carried_dir.clone(),
            sandbox.working_dir().to_path_buf(),
        )
    });
    let result = execute_script_standalone(StandaloneExecution {
        sandbox,
        executor,
        ipc,
        committed,
        job_registry,
        task_id: task_id.clone(),
        info,
        execution_id,
        background,
        carry,
        carried_dir,
        script: script.clone(),
        interpreter,
        mode,
        backend,
        stdin_alerts,
        lifetime,
        background_startup,
        #[cfg(all(unix, feature = "jobserver"))]
        jobserver,
        expect,
        resolution,
        outputs,
        max_lines,
        stdout,
    })
    .await;
    // A process that never started leaves no job to register.
    status_registry.abandon_launch(&task_id).await;
    let interruptions = outages.end(&task_id).await;
    let (result, settled) = match result {
        Ok((command, settled)) => (Ok(command), Some(settled)),
        Err(error) => (Err(error), None),
    };
    let result = result.map(|command| CompletedCommand {
        interruptions,
        ..command
    });
    // Ahead of the result, so what a policy records of this end holds for
    // the agent's next call.
    let outcome = result
        .as_ref()
        .map_or(ScriptOutcome::Failed, |command| ScriptOutcome::Exited {
            code: command.exit_code,
        });
    if let Some((spelling, carried, working_dir)) = &end_view {
        // A settled script already read the carried directory under the
        // lock that recorded it; one that settled nothing reads it now.
        let (carry, host) = match settled {
            Some(Settled { carry, carried }) => (carry, carried),
            None => (
                Carry::NotCarried,
                carried.as_ref().and_then(CarriedDir::get),
            ),
        };
        let own = canonical_path(working_dir)
            .await
            .unwrap_or_else(|_| working_dir.clone());
        let end = spelling.dir(host, &own);
        policies.ended(&task_id, outcome, &end, carry).await;
    }

    let quick_result = match &result {
        Ok(ok) => Ok(ok.clone()),
        Err(err) => Err(err.detail().into_owned()),
    };
    let _ = result_tx.send(quick_result).await;
    // The call answers at once, taking the result or leaving it to a
    // background task.
    let foreground = taken.recv().await.ok();
    // A kill's signal can end the process before the kill records its
    // outcome, so the registry settles the ending once no kill is in flight.
    let ending = match &result {
        Ok(command) => Ok(command.exit_code),
        Err(error) => Err(error.detail().into_owned()),
    };
    let owed = status_registry.owe_completion(&task_id, ending).await;
    let kill = owed.as_ref().and_then(|owed| owed.kill);
    // The job as it ended, read before a foreground result takes it.
    let info = status_registry.get_by_task_id(&task_id).await;
    let pending = owed.and_then(|owed| owed.pending);
    // A result the call returned leaves nothing for the agent to take: the
    // completion is taken here, so the registry forgets the job, and the end
    // is no fact to raise: the call already told it.
    let pending = match foreground {
        Some(_) => {
            if let Some(pending) = pending {
                pending.take().await;
            }
            None
        }
        None => pending,
    };
    // Only a job, one promoted past the foreground window or started in the
    // background, reports its end. A command the shell never registered, one
    // that failed to start, has no job to report.
    if foreground.is_none()
        && let Some(info) = info
    {
        let timeout_kill = match kill {
            Some(crate::KillCause::Timeout(kill)) => Some(kill),
            _ => None,
        };
        let _ = completed_tx
            .send(ShellEvent::JobEnded(Box::new(JobEnded {
                agent: completed_tx.agent().clone(),
                info,
                result,
                timeout_kill,
                pending,
            })))
            .await;
    }
    running_tasks.fetch_sub(1, Ordering::AcqRel);
    // The flow's own hold on the word ends before the call returns.
    drop(task_id);
    if let Some(done) = foreground {
        let _ = done.send(()).await;
    }
}

/// Standalone script execution that can be spawned in a background task.
#[allow(clippy::too_many_lines)]
async fn execute_script_standalone<E>(
    request: StandaloneExecution<E>,
) -> Result<(CompletedCommand, Settled), TerminalError>
where
    E: Executor + Clone + 'static,
{
    let StandaloneExecution {
        sandbox,
        executor,
        ipc,
        committed,
        job_registry,
        task_id,
        info,
        execution_id,
        background,
        carry,
        carried_dir,
        script,
        interpreter,
        mode,
        backend,
        stdin_alerts,
        lifetime,
        background_startup,
        #[cfg(all(unix, feature = "jobserver"))]
        jobserver,
        expect,
        resolution,
        outputs,
        max_lines,
        stdout,
    } = request;

    info!(
        script_len = script.len(),
        ?mode,
        "executing background terminal command"
    );
    debug!(script = %script, "script content");
    let start_result = start_standalone_execution(StandaloneStart {
        sandbox: &sandbox,
        executor,
        ipc: ipc.as_ref(),
        committed: &committed,
        job_registry: &job_registry,
        task_id: &task_id,
        info,
        execution_id: &execution_id,
        carry: carry.as_ref(),
        script: &script,
        interpreter,
        mode,
        backend: &backend,
        stdin_alerts,
        lifetime,
        background_startup: &background_startup,
        #[cfg(all(unix, feature = "jobserver"))]
        jobserver: &jobserver,
    })
    .await;
    let (pid, output) = match start_result {
        Ok(started) => started,
        Err(error) => {
            background_startup.report_failure(&error).await;
            return Err(error);
        }
    };

    let result = save_completed_process_output(CompletedProcessOutput {
        job_registry: &job_registry,
        pid,
        output,
        background_output: background,
        outputs: &outputs,
        expect,
        resolution,
        max_lines,
        stdout,
    })
    .await?;
    let exit_code = result.exit_code;

    debug!(exit_code, "background script completed");

    let output_path = stored_path(&result.stdout);
    job_registry.complete(pid, exit_code, output_path).await;

    // The command's directory report arrives with its end, whether the call
    // answered in the foreground window or the command promoted: the task
    // is its sole consumer, so the next command starts where the shell
    // ended and the report file never lingers.
    let settled = match &carry {
        Some(carry) => {
            let ctx = CarryContext {
                carried_dir: carried_dir.as_ref(),
                working_dir: sandbox.working_dir(),
                grants: sandbox.grants().snapshot(),
            };
            settle_cwd(carry, take_cwd_report(carry).await, exit_code == 0, &ctx).await
        }
        // Nothing carries: a background or subagent script, a run, or a
        // script on a server. The next script starts where it would have.
        None => Settled {
            carry: Carry::NotCarried,
            carried: carried_dir.as_ref().and_then(CarriedDir::get),
        },
    };

    Ok((result, settled))
}

#[allow(clippy::too_many_lines)]
async fn start_standalone_execution<E>(
    request: StandaloneStart<'_, E>,
) -> Result<(u32, std::process::Output), TerminalError>
where
    E: Executor + Clone + 'static,
{
    let shell = match request.backend {
        ExecutionBackend::Local { shell } => shell.as_ref(),
        ExecutionBackend::Container {
            runtime,
            #[cfg(all(unix, feature = "jobserver"))]
            launch,
            ..
        } => {
            // The command's fifo lives in the container's kernel, brokered
            // by a relay run with the `python3` the launch's programs hold.
            #[cfg(all(unix, feature = "jobserver"))]
            let job_fifo = match request.jobserver {
                Some(broker) => Some(
                    crate::container_jobserver::open_container_job_fifo(
                        broker,
                        runtime,
                        &launch.bin_dir().join("python3"),
                        request.executor.clone(),
                    )
                    .await?,
                ),
                None => None,
            };
            return execute_container_job(ContainerJobExecution {
                executor: request.executor,
                sandbox: request.sandbox,
                ipc: request.ipc,
                committed: request.committed,
                task_id: request.task_id,
                info: request.info,
                execution_id: request.execution_id,
                script: request.script,
                interpreter: request.interpreter,
                mode: request.mode,
                container_runtime: runtime,
                carry: request.carry,
                job_registry: request.job_registry,
                stdin_alerts: request.stdin_alerts,
                lifetime: request.lifetime,
                background_startup: request.background_startup,
                #[cfg(all(unix, feature = "jobserver"))]
                job_fifo,
            })
            .await;
        }
        ExecutionBackend::Ssh {
            client,
            server,
            heel,
            launch,
        } => {
            // The pool bounds the jobs of this machine; the server is another
            // machine, so its commands take no token and get no fifo.
            return execute_ssh_background(SshBackgroundExecution {
                executor: request.executor,
                sandbox: request.sandbox,
                launch,
                task_id: request.task_id,
                info: request.info,
                execution_id: request.execution_id,
                script: request.script,
                interpreter: request.interpreter,
                mode: request.mode,
                client,
                server,
                heel: heel.as_deref(),
                job_registry: request.job_registry,
                background_startup: request.background_startup,
            })
            .await;
        }
    };
    #[cfg(all(unix, feature = "jobserver"))]
    let job_fifo = match request.jobserver {
        Some(broker) => Some(broker.open_fifo().await?),
        None => None,
    };
    let local = LocalBackgroundExecution {
        shell,
        sandbox: request.sandbox,
        executor: request.executor,
        ipc: request.ipc,
        committed: request.committed,
        task_id: request.task_id,
        info: request.info,
        execution_id: request.execution_id,
        carry: request.carry,
        script: request.script,
        interpreter: request.interpreter,
        mode: request.mode,
        job_registry: request.job_registry,
        stdin_alerts: request.stdin_alerts,
        lifetime: request.lifetime,
        background_startup: request.background_startup,
        #[cfg(all(unix, feature = "jobserver"))]
        job_fifo,
    };
    // Unsafe mode bypasses the proxy entirely, so its connections are not
    // audited at this layer.
    execute_local_background(local).await
}

async fn save_completed_process_output(
    request: CompletedProcessOutput<'_>,
) -> Result<CompletedCommand, TerminalError> {
    let CompletedProcessOutput {
        job_registry,
        pid,
        output,
        background_output,
        outputs,
        expect,
        resolution,
        max_lines,
        stdout,
    } = request;
    let stdout = save_completed_stdout(
        job_registry,
        pid,
        &output.stdout,
        CompletedStdoutFormat {
            background_output,
            outputs,
            expect,
            resolution,
            max_lines,
            stdout,
        },
    )
    .await?;
    let stderr = save_completed_stderr(job_registry, pid, outputs, &output.stderr).await?;
    let exit_code = output.status.code().unwrap_or(-1);

    Ok(CompletedCommand {
        stdout,
        stderr,
        exit_code,
        interruptions: Vec::new(),
    })
}

struct CompletedStdoutFormat<'a> {
    background_output: bool,
    outputs: &'a OutputStore,
    expect: OutputFormat,
    resolution: MediaResolution,
    max_lines: usize,
    stdout: StdoutTreatment,
}

async fn save_completed_stdout(
    job_registry: &JobRegistry,
    pid: u32,
    stdout: &[u8],
    format: CompletedStdoutFormat<'_>,
) -> Result<CommandStream, TerminalError> {
    let byte_limit = Some(format.outputs.policy().get().inline_bytes);
    let data_to_save = if format.background_output {
        Cow::Borrowed(stdout)
    } else {
        compressed_stdout_data(format.outputs, stdout, format.expect, format.stdout).await?
    };

    match format
        .outputs
        .save_text_with_line_limit(
            data_to_save.as_ref(),
            format.expect,
            format.resolution,
            format.max_lines,
            byte_limit,
        )
        .await
    {
        Ok(entry) => Ok(entry),
        Err(err) => {
            job_registry.fail(pid, &err.to_string(), None).await;
            Err(TerminalError::Io(err))
        }
    }
}

/// Runs the output cleaning pipeline over a completed command's stdout,
/// under the script's output treatment.
///
/// [`StdoutTreatment::Raw`] bypasses it entirely, as do non-text formats and
/// output that is not valid UTF-8, so the model sees exactly the bytes the
/// command wrote.
async fn compressed_stdout_data<'a>(
    outputs: &OutputStore,
    stdout: &'a [u8],
    expect: OutputFormat,
    treatment: StdoutTreatment,
) -> Result<Cow<'a, [u8]>, TerminalError> {
    let is_text = matches!(expect, OutputFormat::Text | OutputFormat::Auto);
    let StdoutTreatment::Cleaned(treatment) = treatment else {
        return Ok(Cow::Borrowed(stdout));
    };
    if !is_text || stdout.is_empty() {
        return Ok(Cow::Borrowed(stdout));
    }
    let Ok(text) = std::str::from_utf8(stdout) else {
        return Ok(Cow::Borrowed(stdout));
    };

    let cleaned = crate::output_compress::compress_text(outputs, text, treatment)
        .await
        .map_err(TerminalError::Io)?;
    Ok(cleaned.map_or(Cow::Borrowed(stdout), |cleaned| {
        Cow::Owned(cleaned.into_bytes())
    }))
}

async fn save_completed_stderr(
    job_registry: &JobRegistry,
    pid: u32,
    outputs: &OutputStore,
    stderr: &[u8],
) -> Result<Option<CommandStream>, TerminalError> {
    if stderr.is_empty() {
        return Ok(None);
    }
    match outputs
        .save_with_limit(
            stderr,
            OutputFormat::Text,
            MediaResolution::Auto,
            Some(outputs.policy().get().inline_bytes),
        )
        .await
    {
        Ok(entry) => Ok(Some(entry)),
        Err(err) => {
            job_registry.fail(pid, &err.to_string(), None).await;
            Err(TerminalError::Io(err))
        }
    }
}

/// The script a container's command runs ahead of its own: the worktree map
/// as exports. Empty without a map — subagent containers never see it.
#[cfg(unix)]
fn container_git_preamble(
    committed: &[CommittingWorktree],
    mount: Option<&MountRoot>,
) -> Result<String, TerminalError> {
    use std::fmt::Write as _;
    let mut environment = Vec::new();
    // A program inside the container compares its working directory against
    // the map: every path it holds must be spelled the way the container's
    // workspace mount resolves it.
    let spell = |path: &Path| -> Result<String, TerminalError> {
        let mount = mount.ok_or_else(|| {
            TerminalError::SandboxSetup(format!(
                "a committing worktree {} has no workspace mount to map through",
                path.display()
            ))
        })?;
        let host_root =
            std::fs::canonicalize(&mount.host_root).unwrap_or_else(|_| mount.host_root.clone());
        MountRoot::new(host_root, mount.container_root.clone())
            .container_path(path)
            .map(|path| path.to_string())
            .map_err(|source| {
                TerminalError::SandboxSetup(format!(
                    "{} is outside the container workspace mount: {source}",
                    path.display()
                ))
            })
    };
    committed_git_environment(committed, &mut environment, spell)?;
    let mut preamble = String::new();
    for (name, value) in &environment {
        writeln!(preamble, "export {name}={}", shell_words::quote(value))
            .expect("a string cannot fail a write");
    }
    Ok(preamble)
}

#[cfg(not(unix))]
fn container_git_preamble(
    _committed: &[CommittingWorktree],
    _mount: Option<&MountRoot>,
) -> Result<String, TerminalError> {
    Ok(String::new())
}

/// Exports the worktree map of a main shell's commands, each path spelled
/// the way `spell` resolves it for the command's context: a host path
/// locally, a container path inside a container.
///
/// The map only describes paths the host resolved when the grant was made;
/// a program of the embedder's that runs `git` in a committing worktree
/// reads the worktree's host-owned repository entry from it, so nothing a
/// subagent wrote decides what program git runs. A subagent shell holds
/// none.
fn committed_git_environment(
    committed: &[CommittingWorktree],
    environment: &mut Vec<(String, String)>,
    spell: impl Fn(&Path) -> Result<String, TerminalError>,
) -> Result<(), TerminalError> {
    if committed.is_empty() {
        return Ok(());
    }
    environment.push((
        "AITHER_GIT_WORKTREES_COUNT".to_string(),
        committed.len().to_string(),
    ));
    for (index, worktree) in committed.iter().enumerate() {
        environment.push((
            format!("AITHER_GIT_WORKTREE_PATH_{index}"),
            spell(worktree.worktree())?,
        ));
        environment.push((
            format!("AITHER_GIT_WORKTREE_GITDIR_{index}"),
            spell(worktree.git_dir())?,
        ));
    }
    Ok(())
}

/// The grant a command's jobserver fifo needs: it writes tokens back to it.
#[cfg(all(unix, feature = "jobserver"))]
pub fn job_fifo_grant(path: &Path) -> heel::Result<heel::Grant> {
    Ok(heel::Grant::new(
        crate::spawn::resolved(path)?,
        heel::Access::WRITE,
    ))
}

/// Runs a command on the host: sandboxed through the terminal's [`Sandbox`],
/// or with host access for an approved `unsafe` one.
async fn execute_local_background<E: Executor + Clone + 'static>(
    context: LocalBackgroundExecution<'_, E>,
) -> Result<(u32, std::process::Output), TerminalError> {
    // What this one command adds to the sandbox: its jobserver variables and
    // fifo grant, and the worktree map of a main shell.
    let mut environment = Vec::new();
    #[cfg_attr(not(all(unix, feature = "jobserver")), allow(unused_mut))]
    let mut grants = Vec::new();
    #[cfg(all(unix, feature = "jobserver"))]
    if let Some(fifo) = &context.job_fifo {
        grants.push(
            job_fifo_grant(fifo.path())
                .map_err(|error| TerminalError::SandboxSetup(error.to_string()))?,
        );
        environment.extend(fifo.environment());
    }
    committed_git_environment(context.committed, &mut environment, |path| {
        Ok(path.display().to_string())
    })?;
    let sandbox = match context.mode {
        SandboxMode::Sandboxed => context.sandbox.clone(),
        SandboxMode::Unsandboxed => context.sandbox.clone().unconfined(),
    };

    let launch_script = launch_script(context.interpreter, context.script, context.task_id)?;
    let launch = build_terminal_launch(
        context.shell,
        &launch_script,
        context.carry.and_then(|carry| carry.report.as_deref()),
    )?;
    let (program, args) = match launch {
        TerminalLaunch::Direct { program, args } | TerminalLaunch::Shell { program, args } => {
            (program, args)
        }
    };

    let mut spec = SpawnSpec::new(program)
        .args(args)
        .env(environment)
        .launch_info(context.info.clone())
        .stdio(SpawnStdio {
            // A background job's stdin is watched; any other reads EOF.
            stdin: if context.stdin_alerts.is_some() {
                Pipe::Piped
            } else {
                Pipe::Null
            },
            ..SpawnStdio::default()
        });
    spec = spec
        .grants(grants.into_iter().map(crate::SpawnGrant::Host))
        .ipc(context.ipc.and_then(|ipc| ipc.router(&context.info)));
    if let Some(start) = context.carry.and_then(|carry| carry.start.as_deref()) {
        spec = spec.cwd(start);
    }
    let spec = spec.lifetime(match context.lifetime {
        JobLifetime::Foreground { .. } => TreeLifetime::Foreground,
        JobLifetime::Background => TreeLifetime::Background,
    });
    let LocalSpawn {
        child,
        tree,
        sandbox: _sandbox,
    } = sandbox.spawn_local(spec).await?;

    let pid = child.id();
    context
        .job_registry
        .register_with_tree(
            pid,
            context.task_id,
            context.execution_id,
            context.script,
            context.interpreter,
            context.mode,
            None,
            Some(tree.clone()),
            crate::JobPlace::Host,
        )
        .await;
    context.background_startup.report_ready().await;

    let output = collect_local_process_output(
        child,
        context.executor,
        context.job_registry,
        pid,
        LocalJob {
            tree,
            lifetime: context.lifetime,
            stdin_alerts: context.stdin_alerts,
        },
        "missing stdout pipe for local process",
        "missing stderr pipe for local process",
    )
    .await?;

    Ok((pid, output))
}

/// How long a command's descendants may outlive its top-level shell.
#[derive(Clone)]
enum JobLifetime {
    /// None: the survivors are killed and reported on `alerts`, as those of
    /// the command `task_id` that the tool call `call` ran.
    Foreground {
        alerts: EventSender,
        task_id: String,
        call: Option<CallIdentity>,
    },
    /// All of them: the job runs until its whole tree has exited.
    Background,
}

impl JobLifetime {
    /// Reports `killed` as the descendants a foreground command left; a
    /// background job's are never killed, so it reports nothing.
    async fn report_descendants_killed(&self, killed: Vec<JobProcess>) {
        let Self::Foreground {
            alerts,
            task_id,
            call,
        } = self
        else {
            return;
        };
        let _ = alerts
            .send(ShellEvent::DescendantsKilled(DescendantsKilled {
                agent: alerts.agent().clone(),
                task_id: task_id.clone(),
                call: call.clone(),
                processes: killed,
            }))
            .await;
    }
}

/// Where a background job's stdin alerts go.
///
/// Its presence is what makes a command a background job: a command without
/// one reads EOF from a null stdin.
#[derive(Clone)]
struct StdinAlerts {
    tx: EventSender,
    task_id: String,
    script: String,
}

impl StdinAlerts {
    /// Raises an alert for each signal a stdin watch sends, until the watch
    /// ends.
    async fn relay(self, signals: async_channel::Receiver<StdinWatchSignal>) {
        while let Ok(signal) = signals.recv().await {
            match signal {
                StdinWatchSignal::Blocked => self.raise().await,
                StdinWatchSignal::Failed(error) => self.raise_failure(error).await,
            }
        }
    }

    async fn raise(&self) {
        debug!(task_id = %self.task_id, "background job is blocked on stdin");
        let _ = self
            .tx
            .send(ShellEvent::StdinBlocked(StdinBlocked {
                agent: self.tx.agent().clone(),
                task_id: self.task_id.clone(),
                script: self.script.clone(),
            }))
            .await;
    }

    /// Tells the agent this job's stdin watch died: no blocked-on-stdin
    /// alert can come for it anymore.
    async fn raise_failure(&self, error: String) {
        warn!(task_id = %self.task_id, error = %error, "stdin watch failed");
        let _ = self
            .tx
            .send(ShellEvent::StdinWatchFailed(StdinWatchFailed {
                agent: self.tx.agent().clone(),
                task_id: self.task_id.clone(),
                script: self.script.clone(),
                error,
            }))
            .await;
    }
}

/// Watches one local background job's stdin pipe for a waiting reader.
pub struct LocalStdinWatch {
    probe: LocalStdinProbe,
    feed: FeedCounter,
    detector: EpisodeDetector,
    signals: async_channel::Sender<StdinWatchSignal>,
}

impl LocalStdinWatch {
    /// A watch of the pipe `probe` looks at that `feed` counts the writes to,
    /// reporting on `signals`.
    pub(crate) fn new(
        probe: LocalStdinProbe,
        feed: FeedCounter,
        signals: async_channel::Sender<StdinWatchSignal>,
    ) -> Self {
        Self {
            probe,
            feed,
            detector: EpisodeDetector::default(),
            signals,
        }
    }

    /// Probes the pipe every [`STDIN_WATCH_INTERVAL`] and signals one
    /// [`StdinWatchSignal::Blocked`] per blocking episode, for as long as the
    /// future is polled.
    async fn run(mut self) -> std::convert::Infallible {
        loop {
            async_io::Timer::after(STDIN_WATCH_INTERVAL).await;
            let Self {
                probe,
                feed,
                mut detector,
                signals,
            } = self;
            let fed = feed.count();
            let (probe, observed) = blocking::unblock(move || {
                let observed = probe.probe();
                (probe, observed)
            })
            .await;
            match observed {
                Ok(observed) => {
                    if detector.observe(fed, observed) {
                        let _ = signals.send(StdinWatchSignal::Blocked).await;
                    }
                }
                Err(error) => {
                    // Detection cannot run: surface it once, then park — the
                    // job's exit still ends the race in
                    // `wait_for_local_process_exit`.
                    let _ = signals
                        .send(StdinWatchSignal::Failed(error.to_string()))
                        .await;
                    std::future::pending::<()>().await;
                }
            }
            self = Self {
                probe,
                feed,
                detector,
                signals,
            };
        }
    }
}

async fn collect_local_process_output<E: Executor + Clone + 'static>(
    mut child: LocalChild,
    executor: E,
    job_registry: &JobRegistry,
    pid: u32,
    job: LocalJob,
    missing_stdout_message: &'static str,
    missing_stderr_message: &'static str,
) -> Result<std::process::Output, TerminalError> {
    let LocalJob {
        tree,
        lifetime,
        stdin_alerts,
    } = job;
    let mut stdin_watch = None;
    if let Some(stdin) = child.take_stdin() {
        let feed = FeedCounter::default();
        if let Some(alerts) = stdin_alerts {
            let probe = LocalStdinProbe::new(pid, &stdin)
                .map_err(|error| TerminalError::Execution(format!("stdin probe: {error}")))?;
            let (signal_tx, signal_rx) = async_channel::unbounded();
            executor.spawn(alerts.relay(signal_rx)).detach();
            stdin_watch = Some(LocalStdinWatch::new(probe, feed.clone(), signal_tx));
        }
        let input_tx = write_chunks(stdin, feed);
        job_registry.attach_terminal_input(pid, input_tx).await;
    }

    let stdout = child
        .take_stdout()
        .ok_or_else(|| TerminalError::Execution(missing_stdout_message.into()))?;
    let stderr = child
        .take_stderr()
        .ok_or_else(|| TerminalError::Execution(missing_stderr_message.into()))?;

    let (chunk_tx, chunk_rx) = async_channel::unbounded();
    spawn_terminal_reader(stdout, chunk_tx.clone(), false);
    spawn_terminal_reader(stderr, chunk_tx.clone(), true);
    drop(chunk_tx);
    executor
        .spawn(drain_terminal_chunks(job_registry.clone(), pid, chunk_rx))
        .detach();

    // The root stays unreaped until its tree is gone, so its pid and process
    // group stay reserved while the job's descendants are tracked by them.
    let exited = match wait_for_local_process_exit(&mut child, stdin_watch).await {
        Ok(exited) => exited,
        Err(error) => {
            tree.finish().await;
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            return Err(error);
        }
    };
    let descendants = end_descendants(&tree, &lifetime).await;
    // The task ends first: afterwards the root's pid may belong to anyone.
    tree.finish().await;
    let (status, report) = match exited {
        Some(status) => (Ok(status), None),
        None => match child.wait_reported().await {
            Ok((status, report)) => (Ok(status), report),
            Err(error) => (Err(TerminalError::Execution(error.to_string())), None),
        },
    };
    let _ = &report;
    #[cfg(target_os = "linux")]
    if let Some(report) = report {
        let killed: Vec<JobProcess> = report
            .killed
            .into_iter()
            .map(|process| JobProcess {
                pid: process.pid,
                name: process.name,
            })
            .collect();
        if !killed.is_empty() {
            info!(
                killed = killed.len(),
                "the supervisor killed the finished command's descendants"
            );
            lifetime.report_descendants_killed(killed).await;
        }
    }
    let status = match (descendants, status) {
        (Ok(()), Ok(status)) => status,
        (Err(error), _) | (Ok(()), Err(error)) => {
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            job_registry.fail(pid, &error.detail(), None).await;
            return Err(error);
        }
    };
    wait_for_terminal_stream_close(job_registry, pid).await;
    let (stdout, stderr) = job_registry.terminal_output(pid).await.ok_or_else(|| {
        TerminalError::Execution(format!("missing terminal output for pid {pid}"))
    })?;
    let output = std::process::Output {
        status,
        stdout,
        stderr,
    };

    Ok(output)
}

/// A local job's process tree and how long its descendants may live.
struct LocalJob {
    tree: JobTree,
    lifetime: JobLifetime,
    stdin_alerts: Option<StdinAlerts>,
}

/// The teardown budget a job's supervisor gives its kills: the time to
/// freeze the tree, deliver `SIGKILL` and see every killed process exit.
/// Generous relative to the few milliseconds a `SIGKILL` exit takes, since a
/// too-short budget reports survivors that are merely still exiting.
#[cfg(target_os = "linux")]
pub const SUPERVISED_REAP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// A spawned local job: the plain `heel` child on platforms without a
/// supervisor, or a `heel supervise`-supervised one on Linux, where the
/// supervisor is the command's parent and owns its whole tree.
pub enum LocalChild {
    /// Everywhere but Linux: the sandboxed process itself.
    #[cfg_attr(target_os = "linux", allow(dead_code))]
    Plain(heel::Child),
    /// Linux, foreground lifetime: `wait` yields the command's status and
    /// the teardown's report of what it killed.
    #[cfg(target_os = "linux")]
    Foreground(heel::supervisor::SupervisedChild<heel::supervisor::Foreground>),
    /// Linux, background lifetime: `wait` ends only once the last descendant
    /// of the command has exited.
    #[cfg(target_os = "linux")]
    Background(heel::supervisor::SupervisedChild<heel::supervisor::Background>),
}

impl LocalChild {
    /// The spawned process's id: the supervisor's on Linux.
    pub(crate) fn id(&self) -> u32 {
        match self {
            Self::Plain(child) => child.id(),
            #[cfg(target_os = "linux")]
            Self::Foreground(child) => child.id(),
            #[cfg(target_os = "linux")]
            Self::Background(child) => child.id(),
        }
    }

    pub(crate) fn take_stdin(&mut self) -> Option<std::fs::File> {
        match self {
            Self::Plain(child) => child.take_stdin(),
            #[cfg(target_os = "linux")]
            Self::Foreground(child) => child.take_stdin(),
            #[cfg(target_os = "linux")]
            Self::Background(child) => child.take_stdin(),
        }
    }

    pub(crate) fn take_stdout(&mut self) -> Option<std::fs::File> {
        match self {
            Self::Plain(child) => child.take_stdout(),
            #[cfg(target_os = "linux")]
            Self::Foreground(child) => child.take_stdout(),
            #[cfg(target_os = "linux")]
            Self::Background(child) => child.take_stdout(),
        }
    }

    pub(crate) fn take_stderr(&mut self) -> Option<std::fs::File> {
        match self {
            Self::Plain(child) => child.take_stderr(),
            #[cfg(target_os = "linux")]
            Self::Foreground(child) => child.take_stderr(),
            #[cfg(target_os = "linux")]
            Self::Background(child) => child.take_stderr(),
        }
    }

    /// A handle to the supervisor owning the job's tree.
    #[cfg(target_os = "linux")]
    pub(crate) fn tree_control(&self) -> TreeControl {
        match self {
            Self::Foreground(child) => child.tree_control(),
            #[cfg(target_os = "linux")]
            Self::Background(child) => child.tree_control(),
            Self::Plain(_) => unreachable!("a supervised spawn returns no plain child"),
        }
    }

    /// Waits the job out, yielding the command's status and, for a
    /// supervised foreground job, the supervisor's teardown report: what it
    /// found still running under the command and killed.
    pub(crate) async fn wait_reported(
        &mut self,
    ) -> Result<(std::process::ExitStatus, Option<TreeReport>), heel::Error> {
        match self {
            Self::Plain(child) => Ok((child.wait().await?, None)),
            #[cfg(target_os = "linux")]
            Self::Foreground(child) => {
                let exit = child.wait().await?;
                Ok((exit.status, Some(exit.report)))
            }
            #[cfg(target_os = "linux")]
            Self::Background(child) => Ok((child.wait().await?, None)),
        }
    }
}

/// Applies the rule that a job's descendants live exactly as long as the job,
/// after its top-level shell exited.
///
/// A foreground command's surviving descendants are killed and reported in one
/// [`ShellEvent::DescendantsKilled`]; a background job stays running until every
/// descendant has exited, which its timeout, `terminal_kill` and the registry's
/// kills end by killing the tree.
async fn end_descendants(tree: &JobTree, lifetime: &JobLifetime) -> Result<(), TerminalError> {
    match lifetime {
        JobLifetime::Foreground { .. } => {
            let killed = tree
                .kill_descendants()
                .await
                .map_err(|error| TerminalError::Execution(error.to_string()))?;
            if !killed.is_empty() {
                info!(
                    killed = killed.len(),
                    "killed descendants of a finished foreground command"
                );
                lifetime.report_descendants_killed(killed).await;
            }
            Ok(())
        }
        JobLifetime::Background => tree
            .wait_descendants_exit()
            .await
            .map_err(|error| TerminalError::Execution(error.to_string())),
    }
}

/// Waits for the child to exit, raising stdin alerts meanwhile.
///
/// On Unix the child is only observed to have exited and stays unreaped (the
/// answer is `None`); the caller waits for it once its tree is gone. Windows
/// has no such state and answers the exit status.
#[cfg_attr(unix, allow(clippy::needless_pass_by_ref_mut))]
pub async fn wait_for_local_process_exit(
    child: &mut LocalChild,
    stdin_watch: Option<LocalStdinWatch>,
) -> Result<Option<std::process::ExitStatus>, TerminalError> {
    #[cfg(unix)]
    let exit = async {
        crate::process_tree::wait_for_exit(child.id())
            .await
            .map(|()| None)
            .map_err(|error| TerminalError::Execution(error.to_string()))
    };
    #[cfg(not(unix))]
    let exit = async {
        child
            .wait_reported()
            .await
            .map(|(status, _)| Some(status))
            .map_err(|error| TerminalError::Execution(error.to_string()))
    };
    let Some(stdin_watch) = stdin_watch else {
        return exit.await;
    };
    futures_lite::future::or(exit, async { match stdin_watch.run().await {} }).await
}

/// A container command's script as the shell hands it to the sandbox. A main
/// terminal's worktree map travels inside it as exports. A command that
/// carries its directory reports the directory it ended in.
fn container_job_script(
    committed: &[CommittingWorktree],
    workspace: Option<&MountRoot>,
    report: Option<&ContainerPath>,
    script: &str,
) -> Result<String, TerminalError> {
    let preamble = container_git_preamble(committed, workspace)?;
    let script = if preamble.is_empty() {
        script.to_string()
    } else {
        format!("{preamble}{script}")
    };
    match report {
        Some(report) => carry_cwd_script(&script, report.as_str()),
        None => Ok(script),
    }
}

/// Runs a command in a container through a [`Sandbox`] on the container's
/// backend, and answers what the job produced.
async fn execute_container_job<E: Executor + Clone + 'static>(
    context: ContainerJobExecution<'_, E>,
) -> Result<(u32, std::process::Output), TerminalError> {
    let result = spawn_container_job(&context).await;
    // The exec has ended, killed or not: the relay stops feeding the fifo and
    // returns its tokens once the job's last process holding it is gone.
    #[cfg(all(unix, feature = "jobserver"))]
    drop(context.job_fifo);
    result
}

async fn spawn_container_job<E: Executor + Clone + 'static>(
    context: &ContainerJobExecution<'_, E>,
) -> Result<(u32, std::process::Output), TerminalError> {
    // The spec's variables are exported after the sandbox's source's, so they
    // win over a source variable of the same name, as they do for a local
    // command.
    #[cfg_attr(not(all(unix, feature = "jobserver")), allow(unused_mut))]
    let mut environment = vec![(
        crate::memory::JOB_MARKER_ENV.to_string(),
        context.task_id.to_string(),
    )];
    #[cfg_attr(not(all(unix, feature = "jobserver")), allow(unused_mut))]
    let mut grants = Vec::new();
    #[cfg(all(unix, feature = "jobserver"))]
    if let Some(fifo) = &context.job_fifo {
        environment.extend(fifo.environment());
        // The build tools write tokens back to the fifo.
        grants.push(crate::SpawnGrant::Container {
            path: fifo.path().clone(),
            access: heel::Access::WRITE,
        });
    }
    let runtime = context.container_runtime;
    let sandbox = context.sandbox.clone();

    let carry = context.carry;
    let script = container_job_script(
        context.committed,
        runtime.workspace_mount(),
        carry
            .and_then(|carry| carry.container.as_ref())
            .and_then(|container| container.report.as_ref()),
        &launch_script(context.interpreter, context.script, context.task_id)?,
    )?;
    let foreground = matches!(context.lifetime, JobLifetime::Foreground { .. });
    let mut spec = SpawnSpec::script(script)
        .env(environment)
        .launch_info(context.info.clone())
        .grants(grants)
        .ipc(context.ipc.and_then(|ipc| ipc.router(&context.info)))
        .lifetime(if foreground {
            TreeLifetime::Foreground
        } else {
            TreeLifetime::Background
        })
        // A foreground command gets no input channel: its stdin reads EOF.
        .stdio(SpawnStdio {
            stdin: if foreground { Pipe::Null } else { Pipe::Piped },
            ..SpawnStdio::default()
        });
    if let Some(carry) = carry.filter(|carry| carry.container.is_some())
        && let Some(start) = &carry.start
    {
        spec = spec.cwd(start);
    }
    let child = sandbox.spawn(spec).await?;
    run_spawned_job(
        child,
        SpawnedJob {
            executor: context.executor.clone(),
            job_registry: context.job_registry,
            background_startup: context.background_startup,
            place: crate::JobPlace::Container,
            task_id: context.task_id,
            execution_id: context.execution_id,
            script: context.script,
            interpreter: context.interpreter,
            mode: context.mode,
            stdin_alerts: context.stdin_alerts.clone(),
            lifetime: context.lifetime.clone(),
        },
    )
    .await
}

/// The status of a process a kill ended, for a backend that has none to
/// report: ended by `SIGKILL`.
fn killed_exit_status() -> std::process::ExitStatus {
    #[cfg(unix)]
    {
        std::os::unix::process::ExitStatusExt::from_raw(libc::SIGKILL)
    }
    #[cfg(windows)]
    {
        std::os::windows::process::ExitStatusExt::from_raw(1)
    }
}

/// Moves the chunks of a spawned process's stream to `tx` until the stream
/// ends, then says it closed.
async fn forward_stream(
    stream: async_channel::Receiver<Vec<u8>>,
    tx: async_channel::Sender<TerminalChunk>,
    is_stderr: bool,
) {
    while let Ok(bytes) = stream.recv().await {
        let chunk = if is_stderr {
            TerminalChunk::Stderr(bytes)
        } else {
            TerminalChunk::Stdout(bytes)
        };
        if tx.send(chunk).await.is_err() {
            return;
        }
    }
    let _ = tx
        .send(if is_stderr {
            TerminalChunk::StderrClosed
        } else {
            TerminalChunk::StdoutClosed
        })
        .await;
}

/// Runs a command on an SSH server through a [`Sandbox`] on the server's
/// backend: a default-mode command runs the managed shell under the server's
/// `heel`, an `unsafe` one runs it directly.
async fn execute_ssh_background<E: Executor + Clone + 'static>(
    context: SshBackgroundExecution<'_, E>,
) -> Result<(u32, std::process::Output), TerminalError> {
    // Only what the spec names travels to a server: the sandbox's own
    // environment, source and grants never leave this machine.
    let sandbox = context
        .sandbox
        .clone()
        .with_backend(crate::Backend::Ssh {
            server: context.server.clone(),
            heel: context.heel.map(str::to_owned),
            launch: context.launch.clone(),
        })
        .with_ssh_client(context.client.clone());
    let sandbox = match context.mode {
        SandboxMode::Sandboxed => sandbox,
        SandboxMode::Unsandboxed => sandbox.unconfined(),
    };
    let script = launch_script(context.interpreter, context.script, context.task_id)?;
    // No `.ipc(..)`: a server has no way back to this machine, and its
    // backend refuses a router with `SpawnError::Unsupported`.
    let child = sandbox
        .spawn(SpawnSpec::script(script).launch_info(context.info.clone()))
        .await?;
    run_spawned_job(
        child,
        SpawnedJob {
            executor: context.executor,
            job_registry: context.job_registry,
            background_startup: context.background_startup,
            // The local `ssh` process stands for a job that runs on the
            // remote host, whose memory is never judged.
            place: crate::JobPlace::Remote,
            task_id: context.task_id,
            execution_id: context.execution_id,
            script: context.script,
            interpreter: context.interpreter,
            mode: context.mode,
            stdin_alerts: None,
            lifetime: JobLifetime::Background,
        },
    )
    .await
}

/// A job whose process a [`Sandbox`] spawned, and where its results go.
struct SpawnedJob<'a, E> {
    executor: E,
    job_registry: &'a JobRegistry,
    background_startup: &'a BackgroundStartup,
    place: crate::JobPlace,
    task_id: &'a IdLease,
    execution_id: &'a str,
    script: &'a str,
    interpreter: Interpreter,
    mode: SandboxMode,
    /// Where a blocked stdin is alerted, for a child that watches its stdin.
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
}

/// Registers a spawned child as a job, streams its output into the registry
/// and answers what it produced when it ended.
async fn run_spawned_job<E: Executor + Clone + 'static>(
    mut child: crate::SandboxedChild,
    job: SpawnedJob<'_, E>,
) -> Result<(u32, std::process::Output), TerminalError> {
    let stdout = child
        .take_stdout()
        .ok_or_else(|| TerminalError::Execution("missing stdout pipe for the process".into()))?;
    let stderr = child
        .take_stderr()
        .ok_or_else(|| TerminalError::Execution("missing stderr pipe for the process".into()))?;
    let stdin = child.take_stdin();
    let stdin_blocked = child.take_stdin_blocked();
    let child = Arc::new(child);

    let pid = child.id();
    job.job_registry
        .register_at(
            job.place,
            pid,
            job.task_id,
            job.execution_id,
            job.script,
            job.interpreter,
            job.mode,
        )
        .await;
    if let Some(stdin) = stdin {
        job.job_registry.attach_terminal_input(pid, stdin).await;
    }
    if let (Some(signals), Some(alerts)) = (stdin_blocked, job.stdin_alerts) {
        job.executor.spawn(alerts.relay(signals)).detach();
    }
    // A kill of the job ends the whole process tree, wherever it runs.
    let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
    job.job_registry.attach_kill_switch(pid, kill_tx).await;
    let killed = Arc::clone(&child);
    job.executor
        .spawn(async move {
            if kill_rx.recv().await.is_ok()
                && let Err(error) = killed.kill().await
            {
                warn!(%error, "ending the job failed");
            }
        })
        .detach();
    job.background_startup.report_ready().await;

    let (chunk_tx, chunk_rx) = async_channel::unbounded();
    for (stream, is_stderr) in [(stdout, false), (stderr, true)] {
        job.executor
            .spawn(forward_stream(stream, chunk_tx.clone(), is_stderr))
            .detach();
    }
    drop(chunk_tx);
    job.executor
        .spawn(drain_terminal_chunks(
            job.job_registry.clone(),
            pid,
            chunk_rx,
        ))
        .detach();

    let status = match child.wait().await {
        crate::ChildExit::Exited(status) => status,
        // A kill ended it, and the backend has no status to report: what it
        // wrote before is the job's output, and the registry knows the cause.
        crate::ChildExit::Killed => killed_exit_status(),
        crate::ChildExit::Failed(error) => {
            job.job_registry.close_stdout(pid).await;
            job.job_registry.close_stderr(pid).await;
            job.job_registry.fail(pid, &error, None).await;
            return Err(TerminalError::Execution(error));
        }
    };
    if matches!(job.lifetime, JobLifetime::Foreground { .. })
        && let Descendants::Killed(killed) = child.descendants_killed().await
        && !killed.is_empty()
    {
        info!(
            killed = killed.len(),
            "killed descendants of a finished foreground command"
        );
        job.lifetime.report_descendants_killed(killed).await;
    }
    wait_for_terminal_stream_close(job.job_registry, pid).await;
    let (stdout, stderr) = job.job_registry.terminal_output(pid).await.ok_or_else(|| {
        TerminalError::Execution(format!("missing terminal output for pid {pid}"))
    })?;
    Ok((
        pid,
        std::process::Output {
            status,
            stdout,
            stderr,
        },
    ))
}

/// Prepends a source's env pairs to `script` as `export` statements, for
/// backends (container exec) whose channel accepts only a script — never an
/// env channel. Names are validated so a provider can't inject shell syntax.
pub fn inject_env_exports(
    script: String,
    env_source: Option<&Arc<dyn EnvSource>>,
    launch: &crate::LaunchInfo,
) -> Result<String, TerminalError> {
    let Some(source) = env_source else {
        return Ok(script);
    };
    let mut rendered = String::new();
    for (name, value) in source.env(launch) {
        if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            || name.chars().next().is_some_and(|c| c.is_ascii_digit())
        {
            return Err(TerminalError::Execution(format!(
                "invalid environment variable name for container exec: {name}"
            )));
        }
        rendered.push_str("export ");
        rendered.push_str(&name);
        rendered.push('=');
        rendered.push_str(&shell_escape(&value));
        rendered.push('\n');
    }
    rendered.push_str(&script);
    Ok(rendered)
}

pub fn shell_escape(value: &str) -> String {
    let escaped = value.replace('\'', "'\"'\"'");
    let mut output = String::with_capacity(escaped.len() + 2);
    output.push('\'');
    output.push_str(&escaped);
    output.push('\'');
    output
}

#[derive(Template)]
#[template(path = "container_setup.sh", escape = "none")]
struct ContainerSetupTemplate<'a> {
    setup: &'a str,
    script: &'a str,
}

#[derive(Template)]
#[template(path = "container_start_dir.sh", escape = "none")]
struct ContainerStartDirTemplate<'a> {
    escaped_directory: &'a str,
    script: &'a str,
}

/// `script` entered in `directory` first: a confined command's sandbox is
/// rooted at its own directory, and the directory the command was asked to
/// start in is entered inside it.
pub fn start_in_container_directory(
    script: &str,
    directory: &ContainerPath,
) -> Result<String, TerminalError> {
    ContainerStartDirTemplate {
        escaped_directory: &shell_escape(directory.as_str()),
        script,
    }
    .render()
    .map_err(|error| {
        TerminalError::Execution(format!(
            "failed to render the container start directory: {error}"
        ))
    })
}

/// Working directory container executions run their scripts in.
pub const CONTAINER_EXEC_WORKING_DIR: &str = "/workspace";

/// The sandbox a subagent's command runs in inside a container, as container
/// paths: the main agent's commands run unconfined because the container is its
/// boundary, but a subagent's keep the per-command sandbox they have on the
/// host — write access to its own space and its granted paths only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContainerConfinement {
    pub(crate) working_dir: ContainerPath,
    /// What the sandbox may reach beyond the working directory.
    pub(crate) grants: Vec<(ContainerPath, heel::Access)>,
}

/// Where a container command runs: the mounted tools, the exec's working
/// directory and, for a subagent, its sandbox.
#[derive(Clone, Copy)]
struct ContainerRunScope<'a> {
    tools_dir: &'a ContainerPath,
    working_dir: &'a str,
    confinement: Option<&'a ContainerConfinement>,
}

#[derive(Template)]
#[template(path = "container_heel_run.sh", escape = "none")]
struct ContainerHeelRunTemplate<'a> {
    escaped_working_dir: &'a str,
    escaped_tools_dir: &'a str,
    /// Set for a subagent: the isolation level and grants replacing the
    /// container-wide permissive one.
    escaped_grants: Option<Vec<String>>,
    /// The embedder's shell words, expanded by the exec's shell.
    sandbox_args: &'a [String],
    nice: Option<u8>,
    io_priority: Option<&'static str>,
    escaped_script: &'a str,
}

/// Renders the command that runs `script` inside the container at the
/// configured scheduling priority.
///
/// The priority goes through heel's own sandbox configuration: the container
/// image's `heel` binary runs the script under `heel run`, so the settings are
/// the same flags the CLI takes everywhere — `nice`/`ionice` are never
/// reimplemented here. The command resolves the image's `heel` itself and
/// fails when the image lacks one; `working_dir` is the exact directory the
/// execution runs in, matching the directory the container exec applies.
///
/// The container is already the isolation boundary, so the invocation asks for
/// `permissive` isolation: the script keeps the filesystem permissions it had
/// as the container caller. The ruleset still exec-denies everything outside
/// the system directories, so `--executable` grants the mounted tools
/// directory read-execute — without it every prepared tool fails with
/// Permission denied. `PATH` and `HOME` are forwarded, so the script sees the
/// environment a setup prepared. Networking keeps heel's deny-all default.
/// `sandbox_args` are the embedder's own words for the command line.
fn heel_run_container_command(
    script: &str,
    priority: ProcessPriority,
    sandbox_args: &[String],
    scope: ContainerRunScope<'_>,
) -> Result<String, TerminalError> {
    let ContainerRunScope {
        tools_dir,
        working_dir,
        confinement,
    } = scope;
    let nice = match priority.cpu() {
        CpuPriority::Nice(level) => Some(level.get()),
        CpuPriority::Normal => None,
    };
    let io_priority = match priority.io() {
        IoPriority::Utility => Some("utility"),
        IoPriority::Background => Some("background"),
        IoPriority::Normal => None,
    };
    // A subagent runs in its own space with heel's default isolation (that space
    // writable, the rest readable) plus its granted paths.
    let working_dir =
        confinement.map_or(working_dir, |confinement| confinement.working_dir.as_str());
    let escaped_grants = confinement.map(|confinement| {
        confinement
            .grants
            .iter()
            .map(|(path, access)| shell_escape(&format!("{path}={access}")))
            .collect()
    });
    let escaped_working_dir = shell_escape(working_dir);
    let escaped_tools_dir = shell_escape(tools_dir.as_str());
    let escaped_script = shell_escape(script);
    ContainerHeelRunTemplate {
        escaped_working_dir: &escaped_working_dir,
        escaped_tools_dir: &escaped_tools_dir,
        escaped_grants,
        sandbox_args,
        nice,
        io_priority,
        escaped_script: &escaped_script,
    }
    .render()
    .map_err(|error| {
        TerminalError::Execution(format!("failed to render container heel command: {error}"))
    })
}

fn prioritized_container_script(
    script: &str,
    priority: ProcessPriority,
    tools_dir: &ContainerPath,
    working_dir: &str,
    confinement: Option<&ContainerConfinement>,
) -> Result<String, TerminalError> {
    if priority.is_normal() && confinement.is_none() {
        return Ok(script.to_string());
    }
    heel_run_container_command(
        script,
        priority,
        &[],
        ContainerRunScope {
            tools_dir,
            working_dir,
            confinement,
        },
    )
}

/// The script a container exec runs for `script`: `setup` first, outside the
/// sandbox, and then `script` in the sandbox the priority and confinement ask
/// for, with the setup's sandbox arguments.
pub fn wrap_container_script(
    script: &str,
    setup: Option<&ExecSetup>,
    priority: ProcessPriority,
    tools_dir: &ContainerPath,
    working_dir: &str,
    confinement: Option<&ContainerConfinement>,
) -> Result<String, TerminalError> {
    let Some(setup) = setup else {
        return prioritized_container_script(script, priority, tools_dir, working_dir, confinement);
    };
    let script = if priority.is_normal() && confinement.is_none() {
        script.to_string()
    } else {
        heel_run_container_command(
            script,
            priority,
            &setup.sandbox_args,
            ContainerRunScope {
                tools_dir,
                working_dir,
                confinement,
            },
        )?
    };
    ContainerSetupTemplate {
        setup: &setup.script,
        script: &script,
    }
    .render()
    .map_err(|error| TerminalError::Execution(format!("failed to render container setup: {error}")))
}

fn preview_first_lines(data: &[u8], max_lines: usize) -> (String, bool) {
    if data.is_empty() || max_lines == 0 {
        return (String::new(), !data.is_empty() && max_lines == 0);
    }

    let text = String::from_utf8_lossy(data);
    let mut preview = String::new();
    let mut total_lines = 0usize;

    for line in text.lines() {
        total_lines += 1;
        if total_lines <= max_lines {
            if !preview.is_empty() {
                preview.push('\n');
            }
            preview.push_str(line);
        }
    }

    let truncated = total_lines > max_lines;
    (preview, truncated)
}

enum TerminalChunk {
    Stdout(Vec<u8>),
    Stderr(Vec<u8>),
    StdoutClosed,
    StderrClosed,
}

fn spawn_terminal_reader<R>(reader: R, tx: async_channel::Sender<TerminalChunk>, is_stderr: bool)
where
    R: Read + Send + 'static,
{
    read_chunks(reader, move |chunk| {
        let chunk = match (chunk, is_stderr) {
            (Some(bytes), false) => TerminalChunk::Stdout(bytes),
            (Some(bytes), true) => TerminalChunk::Stderr(bytes),
            (None, false) => TerminalChunk::StdoutClosed,
            (None, true) => TerminalChunk::StderrClosed,
        };
        let _ = tx.send_blocking(chunk);
    });
}

async fn drain_terminal_chunks(
    job_registry: JobRegistry,
    pid: u32,
    rx: async_channel::Receiver<TerminalChunk>,
) {
    while let Ok(chunk) = rx.recv().await {
        match chunk {
            TerminalChunk::Stdout(bytes) => job_registry.append_stdout(pid, bytes).await,
            TerminalChunk::Stderr(bytes) => job_registry.append_stderr(pid, bytes).await,
            TerminalChunk::StdoutClosed => job_registry.close_stdout(pid).await,
            TerminalChunk::StderrClosed => job_registry.close_stderr(pid).await,
        }
    }
}

async fn wait_for_terminal_stream_close(job_registry: &JobRegistry, pid: u32) {
    while !job_registry.terminal_streams_closed(pid).await {
        async_io::Timer::after(Duration::from_millis(10)).await;
    }
}

/// Errors that can occur during terminal execution.
///
/// The variants carry the facts; the `Display` texts are for logs and
/// diagnostics, and the model never reads them: it reads what
/// [`ShellWording::run_failure`] makes of the error. The `String` of
/// [`Execution`](Self::Execution) and [`SandboxSetup`](Self::SandboxSetup) is
/// the detail the failing backend reported, not a sentence of the shell's.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TerminalError {
    /// The task that runs the script ended before it registered or answered.
    #[error("the task running the script ended without answering")]
    TaskLost,
    /// Execution environment or mode is not configured.
    #[error("invalid terminal execution target: {0}")]
    InvalidTarget(String),
    /// `location` names no configured SSH server.
    #[error("no SSH server named `{0}` is configured")]
    UnknownLocation(String),
    /// A default-mode command names a server that has no `heel` to sandbox it.
    #[error("default mode runs under `heel`, which `{0}` does not have")]
    NoRemoteSandbox(String),
    /// The server's preparation failed; the next command naming it prepares
    /// it again.
    #[error(transparent)]
    SshPrepare(#[from] crate::ssh::SshPrepareError),
    /// The approver denied the script; the model reads the reason it gave.
    #[error("{0}")]
    Denied(String),

    /// The script needs approval, and neither its call nor the terminal has
    /// an approver to give it.
    #[error(
        "approval for `{0}` is required to run this script, but no approver is configured for this terminal"
    )]
    NoApprover(ApprovalKind),

    /// Asking for approval failed or was cancelled.
    #[error(transparent)]
    Approval(#[from] ApprovalError),

    /// Sandbox setup failed.
    #[error("sandbox setup failed: {0}")]
    SandboxSetup(String),

    /// Script execution failed.
    #[error("execution failed: {0}")]
    Execution(String),

    /// IO error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Jobserver pool or broker error.
    #[cfg(all(unix, feature = "jobserver"))]
    #[error(transparent)]
    Jobserver(#[from] crate::jobserver::JobserverError),
}

impl TerminalError {
    /// What the failure reports, without the frame its `Display` puts around
    /// it: the text a job's registry status carries, which the embedder
    /// words inside its own.
    #[must_use]
    pub fn detail(&self) -> Cow<'_, str> {
        match self {
            Self::InvalidTarget(detail)
            | Self::UnknownLocation(detail)
            | Self::NoRemoteSandbox(detail)
            | Self::Denied(detail)
            | Self::SandboxSetup(detail)
            | Self::Execution(detail) => Cow::Borrowed(detail),
            other => Cow::Owned(other.to_string()),
        }
    }
}

impl From<crate::SpawnError> for TerminalError {
    fn from(error: crate::SpawnError) -> Self {
        match error {
            crate::SpawnError::Setup(message) => Self::SandboxSetup(message),
            crate::SpawnError::Start(message) => Self::Execution(message),
            error @ crate::SpawnError::Unsupported { .. } => Self::Execution(error.to_string()),
        }
    }
}

impl<E> std::fmt::Debug for Shell<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shell")
            .field("working_dir", &self.working_dir)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use crate::ContainerLaunch;
    use std::io::Write;
    use std::time::Duration;

    use super::*;
    use crate::GitDirAccess;
    use crate::JobInputTool;
    use crate::ShellWording;
    #[cfg(unix)]
    use crate::builtin::JobInputArgs;
    use crate::output::empty_stream;
    use crate::shell_session::{ContainerExecOutcome, ExecLifetime};
    use crate::test_wording::TestWording;
    #[cfg(unix)]
    use heel::DenyAll;
    use heel::NiceLevel;

    #[cfg(all(unix, feature = "connectivity"))]
    mod outage;

    #[cfg(unix)]
    mod ssh;

    /// The tool that writes to the stdin of `jobs`' jobs.
    fn input_tool(jobs: Jobs) -> JobInputTool {
        JobInputTool::new(
            crate::test_wording::job_tools().input,
            jobs,
            TestWording::any(),
        )
    }

    /// A stdin whose drop reports that the writer thread holding it ended.
    struct DropSignal(async_channel::Sender<()>);

    impl Write for DropSignal {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Drop for DropSignal {
        fn drop(&mut self) {
            let _ = self.0.send_blocking(());
        }
    }

    /// A finished job's stdin writer thread exits instead of blocking on its
    /// channel forever.
    #[tokio::test]
    async fn finished_job_ends_its_stdin_writer_thread() {
        use executor_core::{Executor as _, Task as _};
        let (registry, service) =
            crate::job_registry::job_registry_channel(crate::output_policy::fixture::policy());
        executor_core::tokio::TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();
        registry
            .register(
                9001,
                &crate::test_ids::lease("writer"),
                "exec-writer",
                "cat",
                SandboxMode::Sandboxed,
                None,
            )
            .await;
        let (ended_tx, ended_rx) = async_channel::bounded::<()>(1);
        let input_tx = write_chunks(DropSignal(ended_tx), FeedCounter::default());
        registry.attach_terminal_input(9001, input_tx).await;
        registry
            .terminal_input("writer", b"hi\n".to_vec())
            .await
            .expect("a running job takes input");
        assert!(ended_rx.is_empty(), "the writer lives while the job runs");

        registry.complete(9001, 0, None).await;

        tokio::time::timeout(Duration::from_secs(10), ended_rx.recv())
            .await
            .expect("the writer thread must end once the job finishes")
            .expect("the writer reports its end");
    }

    /// The programs a test container's launch names.
    fn test_container_launch() -> ContainerLaunch {
        ContainerLaunch::new(
            ContainerPath::root().join("opt/tools/bin"),
            ContainerPath::root().join("opt/tools/bin/bash"),
        )
    }

    /// A sandbox on `runtime`, working in `directory`.
    fn container_sandbox(runtime: &ContainerShellRuntime, directory: &Path) -> Sandbox {
        Sandbox::new(
            Backend::Container {
                runtime: runtime.clone(),
                launch: test_container_launch(),
            },
            executor_core::tokio::TokioGlobal,
            directory,
        )
    }

    async fn fresh_terminal(dir: &Path) -> Shell<executor_core::tokio::TokioGlobal> {
        Shell::new_exact(
            dir,
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
            crate::test_wording::job_tools(),
        )
        .await
        .expect("terminal tool should initialize")
    }

    /// A detached call answered as the model reads it.
    trait CallDetached {
        async fn call_detached(&self, arguments: ShellArgs) -> aither_core::Result<ToolResult>;
    }

    impl<E: Executor + Clone + 'static> CallDetached for Shell<E> {
        async fn call_detached(&self, arguments: ShellArgs) -> aither_core::Result<ToolResult> {
            self.call(
                ShellArgs {
                    background: true,
                    ..arguments
                },
                ToolContext::new(),
            )
            .await
        }
    }

    fn render_terminal_result(result: &ToolResult) -> String {
        result.render_for_model()
    }

    fn started_task_id(receiver: &ShellEvents) -> String {
        let started = receiver.take_started();
        assert_eq!(started.len(), 1, "expected one started-task event");
        started[0].task_id.clone()
    }

    #[cfg(unix)]
    #[test]
    fn shell_launch_runs_the_managed_bash_without_startup_files() {
        let script = "printf '%s\\n' hello | cat";
        let (program, args) = shell_launch("/tools/bin/bash", script);

        assert_eq!(program, "/tools/bin/bash");
        assert_eq!(args[..3], ["--noprofile", "--norc", "-c"]);
        assert!(args[3].ends_with(script));
        assert_eq!(args.len(), 4);
    }

    #[test]
    fn host_programs_run_without_a_shell() {
        for name in ["grep", "sed", "awk", "find"] {
            assert_eq!(
                direct_command(&format!("{name} x")).unwrap(),
                Some((name.to_string(), vec!["x".to_string()]))
            );
        }
    }

    #[test]
    fn inject_env_exports_prepends_valid_pairs() {
        let source: Option<Arc<dyn EnvSource>> = Some(Arc::new(vec![
            ("MY_KEY".to_string(), "it's".to_string()),
            ("OTHER".to_string(), "x".to_string()),
        ]));
        let rendered = inject_env_exports(
            "echo hi".to_string(),
            source.as_ref(),
            &crate::LaunchInfo::default(),
        )
        .expect("valid env names should render");
        assert_eq!(
            rendered,
            "export MY_KEY='it'\"'\"'s'\nexport OTHER='x'\necho hi"
        );
    }

    #[test]
    fn inject_env_exports_rejects_shell_syntax_names() {
        let source: Option<Arc<dyn EnvSource>> =
            Some(Arc::new(vec![("BAD;name".to_string(), "x".to_string())]));
        let launch = crate::LaunchInfo::default();
        assert!(inject_env_exports("echo hi".to_string(), source.as_ref(), &launch).is_err());
        assert_eq!(
            inject_env_exports("echo hi".to_string(), None, &launch)
                .expect("no source passes through"),
            "echo hi"
        );
    }

    fn output_text(output: &CommandStream) -> Option<&str> {
        match output {
            CommandStream::Inline { text } => Some(text.as_str()),
            CommandStream::Stored { preview, .. } => Some(preview.as_str()),
            CommandStream::Image { .. } => None,
        }
    }

    async fn wait_for_completed_task(receiver: &ShellEvents, task_id: &str) -> JobEnded {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(task) = receiver
                    .take_completed()
                    .into_iter()
                    .find(|task| task.info.task_id == task_id)
                {
                    return task;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("background task should complete after terminal input")
    }

    #[tokio::test]
    async fn test_terminal_args_defaults() {
        let args: ShellArgs = serde_json::from_str(
            r#"{"description":"print a greeting","script":"echo hello","timeout":30}"#,
        )
        .unwrap();
        assert_eq!(args.resolution, MediaResolution::Auto);
        assert_eq!(args.sandbox, SandboxMode::Sandboxed);
        assert!(!args.background);
    }

    #[test]
    fn background_preview_keeps_leading_indentation() {
        assert_eq!(
            background_preview("  col a  col b\nrow\n\n"),
            "  col a  col b\nrow"
        );
        assert_eq!(background_preview(" \n\t\n"), "");
    }

    fn inline(text: &str) -> CommandStream {
        CommandStream::Inline {
            text: text.to_string(),
        }
    }

    fn completed(
        stdout: CommandStream,
        stderr: Option<CommandStream>,
        exit_code: i32,
    ) -> CommandResult {
        CommandResult {
            stdout,
            stderr,
            outcome: CommandOutcome::Exited { code: exit_code },
            interruptions: Vec::new(),
        }
    }

    #[test]
    fn completed_command_exit_zero_is_the_bare_output() {
        let result = completed(inline("done"), None, 0)
            .into_tool_result()
            .unwrap();
        assert_eq!(result.render_for_model(), "done\n");
        assert_eq!(
            result.command().map(|command| &command.outcome),
            Some(&CommandOutcome::Exited { code: 0 })
        );
    }

    #[test]
    fn completed_command_without_any_output_says_so() {
        assert_eq!(
            completed(empty_stream(), None, 0).render_for_model(),
            "(no output)\n"
        );
    }

    #[test]
    fn completed_command_non_zero_exit_with_stderr() {
        let result = completed(inline("partial"), Some(inline("oops")), 2);
        assert_eq!(
            result.render_for_model(),
            "partial\nstderr:\noops\nExit code: 2.\n"
        );
    }

    #[test]
    fn completed_command_stderr_only_has_no_contradicting_placeholder() {
        assert_eq!(
            completed(empty_stream(), Some(inline("oops")), 2).render_for_model(),
            "stderr:\noops\nExit code: 2.\n"
        );
    }

    #[tokio::test]
    async fn test_terminal_args_parsing() {
        let args: ShellArgs = serde_json::from_str(
            r#"{"description":"sleep in background","script":"sleep 5","timeout":30,"background":true}"#,
        )
        .unwrap();
        assert_eq!(args.timeout, 30);
        assert_eq!(args.script, "sleep 5");
        assert!(args.background);

        let missing_timeout = serde_json::from_str::<ShellArgs>(
            r#"{"description":"sleep in background","script":"sleep 5"}"#,
        )
        .unwrap_err();
        assert!(missing_timeout.to_string().contains("timeout"));

        let missing_description =
            serde_json::from_str::<ShellArgs>(r#"{"script":"echo hello","timeout":1}"#)
                .unwrap_err();
        assert!(missing_description.to_string().contains("description"));
    }

    #[test]
    fn terminal_args_schema_lists_required_fields() {
        let schema = serde_json::to_value(schemars::schema_for!(ShellArgs)).unwrap();
        let properties = schema["properties"].as_object().expect("object schema");
        for field in [
            "description",
            "script",
            "timeout",
            "max_lines",
            "background",
        ] {
            assert!(
                properties.contains_key(field),
                "{field} missing from schema {schema}"
            );
        }
        let required = schema["required"].as_array().expect("required list");
        for field in ["description", "script", "timeout"] {
            assert!(
                required.iter().any(|name| name == field),
                "{field} not required in {schema}"
            );
        }
        assert!(
            !required.iter().any(|name| name == "background"),
            "background must not be required in {schema}"
        );
    }

    /// The lifetime of a foreground command whose facts go to `alerts`.
    fn foreground_lifetime(alerts: EventSender) -> JobLifetime {
        JobLifetime::Foreground {
            alerts,
            task_id: "probe".to_string(),
            call: None,
        }
    }

    /// The tools bin directory a container execution passes to the wrapper.
    fn container_tools_dir() -> ContainerPath {
        test_container_launch().bin_dir().clone()
    }

    /// `script` wrapped with the IPC setup for `commands` at `host:port`.
    fn wrap_with_ipc(
        script: &str,
        commands: &[String],
        endpoint: (&str, u16),
        priority: ProcessPriority,
    ) -> Result<String, TerminalError> {
        let setup = crate::spawn::container_ipc::exec_setup(commands, endpoint.0, endpoint.1)?;
        wrap_container_script(
            script,
            Some(&setup),
            priority,
            &container_tools_dir(),
            CONTAINER_EXEC_WORKING_DIR,
            None,
        )
    }

    #[test]
    fn wrap_container_script_refreshes_command_hash_table() {
        let wrapped = wrap_with_ipc(
            "websearch \"gold price\"",
            &[String::from("websearch")],
            ("host.docker.internal", 9000),
            ProcessPriority::default(),
        )
        .expect("wrap script");
        assert!(wrapped.contains("hash -r;"));
        assert!(wrapped.contains("AITHER_IPC_DIR"));
        assert!(wrapped.contains("HEEL_IPC_ENDPOINT"));
    }

    #[test]
    fn wrap_container_script_preserves_user_script_semantics() {
        let script = "echo '$5,040' && subagent art_direction --prompt-file tmp/prompt.md";
        let wrapped = wrap_with_ipc(
            script,
            &[String::from("subagent")],
            ("host.docker.internal", 9000),
            ProcessPriority::default(),
        )
        .expect("wrap script");
        assert!(wrapped.contains(script));
        assert!(!wrapped.contains("set -euo pipefail"));
    }

    #[test]
    fn wrap_container_script_rejects_invalid_ipc_command_names() {
        let error = wrap_with_ipc(
            "echo ok",
            &[String::from("bad name")],
            ("host.docker.internal", 9000),
            ProcessPriority::default(),
        )
        .expect_err("invalid command name must fail");
        assert!(
            error
                .to_string()
                .contains("cannot be a shim in a container")
        );
    }

    #[test]
    fn wrap_container_script_renders_expected_endpoint() {
        let wrapped = wrap_with_ipc(
            "environment --query deployment",
            &[String::from("environment"), String::from("session")],
            ("host.containers.internal", 43123),
            ProcessPriority::default(),
        )
        .expect("wrap script");
        assert!(wrapped.contains("tcp://host.containers.internal:43123"));
        assert!(wrapped.contains("for cmd in 'environment' 'session'; do"));
    }

    /// Approves every request, and records what it was asked.
    #[derive(Clone, Default)]
    struct Approvals {
        asked: Arc<Mutex<Vec<ApprovalRequest>>>,
        /// Denies with this reason instead of approving.
        deny: Option<&'static str>,
    }

    impl Approvals {
        fn denying(reason: &'static str) -> Self {
            Self {
                deny: Some(reason),
                ..Self::default()
            }
        }

        fn asked(&self) -> Vec<ApprovalRequest> {
            self.asked.lock().unwrap().clone()
        }
    }

    impl Approver for Approvals {
        fn approve(
            &self,
            request: ApprovalRequest,
        ) -> impl Future<Output = Result<Approval, aither_core::llm::tool::ApproverFailure>> + Send
        {
            self.asked.lock().unwrap().push(request);
            std::future::ready(Ok(self.deny.map_or(Approval::Approved, |reason| {
                Approval::Denied {
                    reason: reason.to_string(),
                }
            })))
        }
    }

    /// Approves everything it is asked.
    struct ApproveAll;

    impl Approver for ApproveAll {
        fn approve(
            &self,
            _request: ApprovalRequest,
        ) -> impl Future<Output = Result<Approval, aither_core::llm::tool::ApproverFailure>> + Send
        {
            std::future::ready(Ok(Approval::Approved))
        }
    }

    /// Decides the same way on every script.
    struct Decide(ScriptDecision);

    impl crate::ScriptPolicy for Decide {
        type State = ();

        fn check(
            &self,
            _script: &crate::ScriptCheck<'_>,
            (): &(),
        ) -> impl Future<Output = ScriptDecision> + Send {
            std::future::ready(self.0.clone())
        }
    }

    /// A local terminal in `dir` whose script policies are `decisions`,
    /// asking `approver` for what needs approval.
    #[cfg(unix)]
    async fn approval_tool(
        dir: &Path,
        decisions: impl IntoIterator<Item = ScriptDecision>,
        approver: Option<Approvals>,
    ) -> Shell<executor_core::tokio::TokioGlobal> {
        let tool = fresh_terminal(dir).await.with_test_policies(
            decisions
                .into_iter()
                .map(|decision| crate::AnyScriptPolicy::new(Decide(decision)))
                .collect(),
        );
        match approver {
            Some(approver) => tool.with_approver(approver),
            None => tool,
        }
    }

    #[cfg(unix)]
    fn unsandboxed_args(script: &str) -> ShellArgs {
        ShellArgs {
            sandbox: SandboxMode::Unsandboxed,
            ..terminal_args("run outside the sandbox", script)
        }
    }

    /// A script that runs outside the sandbox and that no policy decided is
    /// asked about; a script inside the sandbox is not.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_unsandboxed_script_with_only_passes_is_asked() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let approvals = Approvals::default();
        let tool = approval_tool(dir.path(), [ScriptDecision::Pass], Some(approvals.clone())).await;

        tool.call(terminal_args("inside", "true"), ToolContext::new())
            .await
            .expect("a sandboxed script with only passes runs");
        assert_eq!(approvals.asked(), []);

        tool.call(unsandboxed_args("true"), ToolContext::new())
            .await
            .expect("the approver approved");
        let asked = approvals.asked();
        assert_eq!(asked.len(), 1);
        assert_eq!(asked[0].kind().as_str(), "unsandboxed");
        assert_eq!(asked[0].subject(), "true");
    }

    /// An `Allow` runs a script that crosses the boundary without asking;
    /// a denied `Ask` keeps it from running and its reason reaches the
    /// caller.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_allow_runs_and_a_denied_ask_does_not() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let approvals = Approvals::default();
        let allowed =
            approval_tool(dir.path(), [ScriptDecision::Allow], Some(approvals.clone())).await;
        allowed
            .call(unsandboxed_args("true"), ToolContext::new())
            .await
            .expect("an allowed script runs");
        assert_eq!(approvals.asked(), [], "an allow asks nobody");

        let dir = tempfile::tempdir().expect("temp dir should be created");
        let marker = dir.path().join("ran");
        let denying = Approvals::denying("not today");
        let ask = ScriptDecision::Ask(ApprovalRequest::new(ApprovalKind::new("test"), "touch ran"));
        let asking = approval_tool(
            dir.path(),
            [ScriptDecision::Allow, ask],
            Some(denying.clone()),
        )
        .await;
        let error = asking
            .call(
                terminal_args("touch the marker", &format!("touch {}", marker.display())),
                ToolContext::new(),
            )
            .await
            .expect_err("the approver denied the script");
        assert!(error.to_string().contains("not today"), "{error}");
        assert_eq!(denying.asked().len(), 1);
        assert!(!marker.exists(), "a denied script does not run");
    }

    /// The approver of the call that carries the script is asked before the
    /// terminal's own, and a call without one falls back to the terminal's;
    /// with neither the script does not run.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_ask_goes_to_the_calls_approver_then_the_terminals() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let terminals = Approvals::default();
        let tool = approval_tool(dir.path(), [], Some(terminals.clone())).await;
        let calls = Approvals::denying("the call's approver decides");
        let error = tool
            .call(
                unsandboxed_args("true"),
                ToolContext::new().with_approver(AnyApprover::new(calls.clone())),
            )
            .await
            .expect_err("the call's approver denied the script");
        assert!(
            error.to_string().contains("the call's approver decides"),
            "{error}"
        );
        assert_eq!((calls.asked().len(), terminals.asked().len()), (1, 0));

        tool.run_background(ShellArgs {
            background: true,
            ..unsandboxed_args("true")
        })
        .await
        .expect("a call with no approver falls back to the terminal's");
        assert_eq!(terminals.asked().len(), 1);

        let dir = tempfile::tempdir().expect("temp dir should be created");
        let bare = approval_tool(dir.path(), [], None).await;
        let error = bare
            .call(unsandboxed_args("true"), ToolContext::new())
            .await
            .expect_err("nobody can approve the script");
        assert!(error.to_string().contains("no approver"), "{error}");
    }

    #[test]
    fn sandbox_is_sandboxed_or_unsandboxed_and_location_replaces_the_ssh_arguments() {
        for (text, mode) in [
            ("\"sandboxed\"", SandboxMode::Sandboxed),
            ("\"unsandboxed\"", SandboxMode::Unsandboxed),
        ] {
            assert_eq!(serde_json::from_str::<SandboxMode>(text).unwrap(), mode);
        }
        for removed in ["\"default\"", "\"unsafe\"", "\"ssh\""] {
            assert!(serde_json::from_str::<SandboxMode>(removed).is_err());
        }
        let args: ShellArgs = serde_json::from_value(serde_json::json!({
            "description": "list the build host",
            "script": "ls",
            "timeout": 30,
            "location": "build",
        }))
        .unwrap();
        assert_eq!(args.location.as_deref(), Some("build"));
        assert_eq!(args.sandbox, SandboxMode::Sandboxed);
        let schema = serde_json::to_value(schemars::schema_for!(ShellArgs)).unwrap();
        assert!(schema["properties"].get("ssh_server_id").is_none());
        assert!(schema["properties"].get("location").is_some());
    }

    #[tokio::test]
    async fn child_terminal_isolates_jobs() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = fresh_terminal(dir.path()).await;
        tool.job_registry()
            .register(
                424_242,
                &crate::test_ids::lease("parent"),
                "parent-execution",
                "sleep 30",
                SandboxMode::Sandboxed,
                None,
            )
            .await;

        let child = tool
            .subagent_without_grant()
            .await
            .expect("an empty grant is always given");
        assert!(child.job_registry().list().await.is_empty());
        assert_eq!(tool.job_registry().list().await.len(), 1);
    }

    /// A subagent's events name its path and the main shell's the root, on
    /// the one handle the two share.
    #[cfg(unix)]
    #[tokio::test]
    async fn events_name_the_agent_whose_shell_raised_them() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = fresh_terminal(dir.path()).await;
        let events = tool.background_receiver();
        let child = tool
            .subagent_without_grant()
            .await
            .expect("an empty grant is always given");
        let grandchild = child
            .subagent(SubagentGrant::new(), None, "reviewer")
            .await
            .expect("an empty grant is always given");

        for shell in [&tool, &child, &grandchild] {
            shell
                .run_background(background_args("a job", "sleep 60"))
                .await
                .expect("the job starts");
        }
        let started = events.take_started();
        let paths: Vec<_> = started.iter().map(|event| event.agent().clone()).collect();
        assert_eq!(
            paths,
            [
                AgentPath::main(),
                AgentPath::main().child("worker"),
                AgentPath::main().child("worker").child("reviewer"),
            ]
        );

        for shell in [&tool, &child, &grandchild] {
            shell.jobs().kill_all().await;
        }
        let mut ended: Vec<_> = Vec::new();
        for _ in 0..3 {
            ended.push(ShellEvent::JobEnded(Box::new(
                events.recv_completed().await,
            )));
        }
        let mut paths: Vec<_> = ended
            .iter()
            .map(|event| event.agent().to_string())
            .collect();
        paths.sort();
        assert_eq!(paths, ["main", "worker", "worker/reviewer"]);
    }

    /// The last handle of a subagent's shell going kills its running jobs.
    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_the_last_handle_of_a_subagent_shell_ends_its_job() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = fresh_terminal(dir.path()).await;
        let child = tool
            .subagent_without_grant()
            .await
            .expect("an empty grant is always given");
        let job = child
            .run_background(background_args("a job", "sleep 60"))
            .await
            .expect("the job starts");

        // A clone keeps the jobs: only the last handle's drop kills them.
        let clone = child.clone();
        drop(child);
        async_io::Timer::after(Duration::from_millis(300)).await;
        assert!(matches!(job.info().await.status, crate::JobStatus::Running));
        drop(clone);

        let result = tokio::time::timeout(Duration::from_secs(30), job.wait())
            .await
            .expect("the job ends once its shell is gone")
            .expect("the job reports its end");
        assert!(
            !matches!(result.outcome, crate::RunOutcome::Exited { code: 0 }),
            "the job was killed, not run to its end: {result:?}"
        );
    }

    /// A main shell's jobs outlive its handles, so an embedder can rebuild it
    /// with its background commands running.
    #[cfg(unix)]
    #[tokio::test]
    async fn dropping_a_main_shell_leaves_its_job_running() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = fresh_terminal(dir.path()).await;
        let events = tool.background_receiver();
        let jobs = tool.jobs();
        tool.run_background(background_args("a job", "sleep 60"))
            .await
            .expect("the job starts");
        let task_id = started_task_id(&events);
        drop(tool);

        async_io::Timer::after(Duration::from_millis(500)).await;
        let listed = jobs.list().await;
        assert!(
            matches!(
                listed.as_slice(),
                [job] if job.task_id == task_id && matches!(job.status, crate::JobStatus::Running)
            ),
            "the job keeps running: {listed:?}"
        );
        jobs.kill_all().await;
    }

    /// Background terminal arguments for `script`.
    #[cfg(unix)]
    fn background_args(description: &str, script: &str) -> ShellArgs {
        ShellArgs {
            description: description.to_string(),
            script: script.to_string(),
            interpreter: Interpreter::Shell,
            sandbox: SandboxMode::Sandboxed,
            location: None,
            resolution: MediaResolution::Auto,
            max_lines: Some(50),
            raw: false,
            timeout: 30,
            background: true,
        }
    }

    /// A command's task id, the file its background output goes to and a
    /// subagent's space are each named by a word leased from the terminal's
    /// allocator, which the subagent's terminal shares; the stored output and
    /// the space give their words back once the terminals holding them drop.
    #[cfg(unix)]
    #[tokio::test]
    async fn model_facing_names_are_leased_from_the_terminal_allocator() {
        let ids = crate::test_ids::RecordingIds::default();
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let terminal = fresh_terminal(dir.path())
            .await
            .with_id_allocator(ids.allocator());

        let receiver = terminal.background_receiver();
        terminal
            .run_background(background_args("print a line", "echo leased"))
            .await
            .expect("terminal call should succeed");
        let [started] = <[_; 1]>::try_from(receiver.take_started()).expect("one started task");
        assert_eq!(ids.leased(), ["alfa", "bravo"]);
        assert_eq!(started.task_id, "alfa");
        assert_eq!(started.output_file.url, "outputs/bravo.txt");
        receiver.recv_completed().await;

        let child = terminal
            .subagent_without_grant()
            .await
            .expect("an empty grant is always given");
        assert_eq!(ids.leased(), ["alfa", "bravo", "charlie"]);
        assert_eq!(
            child
                .working_dir()
                .file_name()
                .and_then(|name| name.to_str()),
            Some("charlie")
        );
        // The child's terminal leases from the parent's allocator.
        let from_child = child.id_allocator().lease().expect("a free word");
        assert_eq!(&*from_child, "delta");
        drop(from_child);

        drop((child, terminal, receiver));
        for _ in 0..500 {
            let released = ids.released();
            if released.iter().any(|word| word == "bravo")
                && released.iter().any(|word| word == "charlie")
            {
                return;
            }
            async_io::Timer::after(Duration::from_millis(10)).await;
        }
        panic!(
            "the output file and the space keep their words: {:?} released",
            ids.released()
        );
    }

    /// A terminal leasing its model-facing names from `ids`.
    #[cfg(unix)]
    async fn terminal_leasing_from(
        dir: &Path,
        ids: AnyIdAllocator,
    ) -> Shell<executor_core::tokio::TokioGlobal> {
        fresh_terminal(dir).await.with_id_allocator(ids)
    }

    /// A foreground command gives its task id back by the time its call
    /// returns, so commands run one after another never exhaust the words.
    #[cfg(unix)]
    #[tokio::test]
    async fn foreground_commands_one_after_another_never_exhaust_the_ids() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let ids = AnyIdAllocator::new(aither_core::WordAllocator::new(["river"]));
        let terminal = terminal_leasing_from(dir.path(), ids.clone()).await;
        for run in 0..4 {
            let result = terminal
                .call(
                    ShellArgs {
                        background: false,
                        ..background_args("finish at once", "exit 0")
                    },
                    ToolContext::new(),
                )
                .await
                .unwrap_or_else(|error| panic!("run {run} should get the one word: {error}"));
            assert!(!result.is_error(), "{result:?}");
        }
        assert!(
            terminal.job_registry().list().await.is_empty(),
            "a returned result leaves nothing to address"
        );
        assert_eq!(&*ids.lease().expect("river is free"), "river");
    }

    /// A background job keeps its task id while the agent can still address
    /// it, and gives it back once its completion is delivered; after that,
    /// the id answers as an unknown task and can be leased again.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_finished_background_job_gives_its_id_back_once_its_completion_is_delivered() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let ids = AnyIdAllocator::new(aither_core::WordAllocator::new(["river", "stone"]));
        let terminal = terminal_leasing_from(dir.path(), ids.clone()).await;
        let receiver = terminal.background_receiver();
        let jobs = terminal.job_registry();
        terminal
            .run_background(background_args("print a line", "echo done"))
            .await
            .expect("terminal call should succeed");
        let [started] = <[_; 1]>::try_from(receiver.take_started()).expect("one started task");
        assert_eq!(started.task_id, "river");
        assert_eq!(started.output_file.url, "outputs/stone.txt");

        let completed = tokio::time::timeout(Duration::from_secs(30), receiver.recv_completed())
            .await
            .expect("the job should deliver its completion");
        let pending = jobs
            .get_by_task_id("river")
            .await
            .expect("an undelivered completion keeps the job");
        assert!(pending.status.completion_pending(), "{pending:?}");
        assert!(
            ids.lease().is_err(),
            "the job and its output hold both words"
        );

        drop(completed);
        assert!(jobs.list().await.is_empty(), "a delivered job is forgotten");
        assert!(jobs.terminal_read("river", 0, 1024).await.is_err());
        assert!(!jobs.kill_by_task_id("river").await);
        // The job's flow lets go of the word as it ends.
        for _ in 0..500 {
            if let Ok(again) = ids.lease() {
                assert_eq!(&*again, "river");
                return;
            }
            async_io::Timer::after(Duration::from_millis(10)).await;
        }
        panic!("a delivered job keeps its word");
    }

    // These four drive a real sandbox running `/bin/pwd`. `heel` reports
    // `UnsupportedPlatform` off Unix, and the binary they run does not exist
    // there either, so there is nothing for them to exercise on Windows.
    #[cfg(unix)]
    #[tokio::test]
    async fn sandboxed_command_output_executes_pwd() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let config = heel::SandboxConfig::builder()
            .network(DenyAll)
            .working_dir(
                std::fs::canonicalize(dir.path()).unwrap_or_else(|_| dir.path().to_path_buf()),
            )
            .security(heel::SecurityConfig::interactive())
            .build();
        let sandbox =
            heel::Sandbox::with_config_and_executor(config, executor_core::tokio::TokioGlobal)
                .await
                .expect("sandbox should initialize");

        let output = sandbox
            .command("/bin/pwd")
            .output()
            .await
            .expect("sandbox command output should succeed");
        eprintln!("PWD OUTPUT STATUS: {:?}", output.status);
        eprintln!(
            "PWD OUTPUT STDOUT: {:?}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert_eq!(output.status.code(), Some(0));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn sandboxed_terminal_executes_pwd() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = fresh_terminal(dir.path()).await;
        let job_registry = tool.job_registry();
        let (startup_tx, _startup_rx) = async_channel::bounded(1);
        let result = execute_script_standalone(StandaloneExecution {
            ipc: None,
            sandbox: tool.sandbox.clone(),
            executor: executor_core::tokio::TokioGlobal,
            committed: Vec::new(),
            job_registry,
            task_id: crate::test_ids::lease("probe"),
            info: crate::LaunchInfo::default(),
            execution_id: "exec-test".to_string(),
            background: false,
            carry: None,
            carried_dir: None,
            script: "/bin/pwd".to_string(),
            interpreter: Interpreter::Shell,
            mode: SandboxMode::Sandboxed,
            backend: ExecutionBackend::Local {
                shell: Arc::from(TEST_HOST_SHELL),
            },
            stdin_alerts: None,
            lifetime: foreground_lifetime(ShellEvents::new().sender(AgentPath::main())),
            background_startup: BackgroundStartup::new(startup_tx),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: None,
            expect: OutputFormat::Text,
            resolution: MediaResolution::Auto,
            outputs: tool.output_store().clone(),
            max_lines: 50,
            stdout: StdoutTreatment::Cleaned(OutputTreatment::Command),
        })
        .await;
        match result {
            Ok((ok, _settled)) => {
                eprintln!("PWD RESULT: {ok:?}");
                assert_eq!(ok.exit_code, 0, "unexpected terminal result: {ok:?}");
            }
            Err(error) => panic!("sandboxed pwd should succeed, got error: {error}"),
        }
    }

    /// A sandboxed terminal command is confined by the live grants: a
    /// writable path takes a write and a readable-only one does not.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_sandboxed_command_is_confined_by_the_live_grants() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let writable = tempfile::tempdir().unwrap();
        let readable = tempfile::tempdir().unwrap();
        let tool = fresh_terminal(dir.path()).await;
        let working_dir = tool.working_dir().clone();
        let writable_paths = vec![writable.path().to_path_buf()];
        let readable_paths = vec![readable.path().to_path_buf()];

        let written = writable.path().join("written");
        let refused = readable.path().join("refused");
        let script = format!(
            "touch {} && echo write-ok; touch {} && echo read-write-ok; true",
            written.display(),
            refused.display(),
        );
        let job_registry = tool.job_registry();
        let (startup_tx, _startup_rx) = async_channel::bounded(1);
        let result = execute_script_standalone(StandaloneExecution {
            ipc: None,
            sandbox: tool
                .sandbox
                .clone()
                .with_working_dir(working_dir)
                .with_grants(test_grants(
                    GitDirAccess::ReadWrite,
                    writable_paths,
                    readable_paths,
                    Vec::new(),
                )),
            executor: executor_core::tokio::TokioGlobal,
            committed: Vec::new(),
            job_registry,
            task_id: crate::test_ids::lease("policy"),
            info: crate::LaunchInfo::default(),
            execution_id: "exec-policy".to_string(),
            background: false,
            carry: None,
            carried_dir: None,
            script,
            interpreter: Interpreter::Shell,
            mode: SandboxMode::Sandboxed,
            backend: ExecutionBackend::Local {
                shell: Arc::from(TEST_HOST_SHELL),
            },
            stdin_alerts: None,
            lifetime: foreground_lifetime(ShellEvents::new().sender(AgentPath::main())),
            background_startup: BackgroundStartup::new(startup_tx),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: None,
            expect: OutputFormat::Text,
            resolution: MediaResolution::Auto,
            outputs: tool.output_store().clone(),
            max_lines: 50,
            stdout: StdoutTreatment::Cleaned(OutputTreatment::Command),
        })
        .await
        .expect("the sandboxed script runs");
        let stdout = output_text(&result.0.stdout).expect("inline stdout");
        let mut lines = stdout.lines();
        assert_eq!(lines.next(), Some("write-ok"));
        assert_eq!(lines.next(), None, "a readable-only path stays read-only");
        assert!(written.exists() && !refused.exists());
    }

    /// The grants of a session whose git directory access is `git_dir`: the
    /// session's own for `ReadWrite`, a child's view for `ReadOnly`.
    #[cfg(unix)]
    fn test_grants(
        git_dir: GitDirAccess,
        writable: Vec<PathBuf>,
        readable: Vec<PathBuf>,
        read_denied: Vec<PathBuf>,
    ) -> Grants {
        match git_dir {
            GitDirAccess::ReadOnly => Grants::new().derived(crate::GrantSnapshot {
                readable,
                writable,
                read_denied,
                git_dir,
            }),
            GitDirAccess::ReadWrite => {
                let grants = Grants::new();
                for path in writable {
                    grants.add_writable(path);
                }
                for path in readable {
                    grants.add_readable(path);
                }
                for path in read_denied {
                    grants.deny_read(path);
                }
                grants
            }
        }
    }

    /// Runs `script` in a sandboxed session with the given grants and returns
    /// its standard output. Runs the distribution's pinned heel.
    #[cfg(unix)]
    async fn sandboxed_stdout(
        writable_paths: Vec<PathBuf>,
        readable_paths: Vec<PathBuf>,
        script: String,
    ) -> String {
        sandboxed_stdout_with(
            GitDirAccess::ReadWrite,
            writable_paths,
            readable_paths,
            Vec::new(),
            script,
        )
        .await
    }

    /// [`sandboxed_stdout`] for a terminal whose grants give a writable
    /// worktree's git directory `git_dir` access.
    #[cfg(unix)]
    async fn sandboxed_stdout_with(
        git_dir: GitDirAccess,
        writable_paths: Vec<PathBuf>,
        readable_paths: Vec<PathBuf>,
        read_denied_paths: Vec<PathBuf>,
        script: String,
    ) -> String {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = fresh_terminal(dir.path()).await;
        let (startup_tx, _startup_rx) = async_channel::bounded(1);
        let result = execute_script_standalone(StandaloneExecution {
            ipc: None,
            // The embedder's persistence denylist, the session's own
            // directory exempt.
            sandbox: tool
                .sandbox
                .clone()
                .with_write_denylist(heel::WriteDenylist::persistence().except(
                    std::fs::canonicalize(dir.path()).expect("the session directory exists"),
                ))
                .with_grants(test_grants(
                    git_dir,
                    writable_paths,
                    readable_paths,
                    read_denied_paths,
                )),
            executor: executor_core::tokio::TokioGlobal,
            committed: Vec::new(),
            job_registry: tool.job_registry(),
            task_id: crate::test_ids::lease("grants"),
            info: crate::LaunchInfo::default(),
            execution_id: "exec-grants".to_string(),
            background: false,
            carry: None,
            carried_dir: None,
            script,
            interpreter: Interpreter::Shell,
            mode: SandboxMode::Sandboxed,
            backend: ExecutionBackend::Local {
                shell: Arc::from(TEST_HOST_SHELL),
            },
            stdin_alerts: None,
            lifetime: foreground_lifetime(ShellEvents::new().sender(AgentPath::main())),
            background_startup: BackgroundStartup::new(startup_tx),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: None,
            expect: OutputFormat::Text,
            resolution: MediaResolution::Auto,
            outputs: tool.output_store().clone(),
            max_lines: 200,
            stdout: StdoutTreatment::Cleaned(OutputTreatment::Command),
        })
        .await
        .expect("the sandboxed script runs");
        output_text(&result.0.stdout)
            .expect("inline stdout")
            .to_string()
    }

    /// A sandboxed command reads an ordinary file outside every granted path.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn sandboxed_commands_read_outside_the_granted_paths() {
        let outside = tempfile::tempdir().unwrap();
        let file = std::fs::canonicalize(outside.path())
            .unwrap()
            .join("ordinary.txt");
        std::fs::write(&file, "ordinary-content").unwrap();
        let stdout = sandboxed_stdout(
            Vec::new(),
            Vec::new(),
            format!("cat {} 2>&1; true", file.display()),
        )
        .await;
        assert!(stdout.contains("ordinary-content"), "{stdout}");
    }

    /// The credential table applies by absolute path although the sandbox
    /// sets `HOME` to the session directory. Checks the credential stores
    /// that exist in the real home.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn sandboxed_commands_cannot_read_home_credentials() {
        let home = PathBuf::from(std::env::var_os("HOME").expect("HOME is set"));
        let stores: Vec<PathBuf> = [".ssh", ".config/gh"]
            .into_iter()
            .map(|store| home.join(store))
            .filter(|store| store.is_dir())
            .collect();
        if stores.is_empty() {
            eprintln!("no ~/.ssh or ~/.config/gh in this home; nothing to deny");
            return;
        }
        for store in stores {
            assert!(
                std::fs::read_dir(&store).is_ok(),
                "the host reads {}",
                store.display()
            );
            let stdout = sandboxed_stdout(
                Vec::new(),
                Vec::new(),
                format!("ls {} 2>&1; echo rc=$?", store.display()),
            )
            .await;
            assert!(
                stdout.contains("rc=1") || stdout.contains("rc=2"),
                "{} must be unreadable: {stdout}",
                store.display()
            );
            assert!(
                stdout.to_lowercase().contains("not permitted")
                    || stdout.to_lowercase().contains("denied"),
                "{} must be refused by the sandbox: {stdout}",
                store.display()
            );
        }
    }

    /// A path the embedder adds is denied, while a sibling stays readable.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn sandboxed_commands_cannot_read_embedder_denied_paths() {
        let fake_home = tempfile::tempdir().unwrap();
        let fake_home = std::fs::canonicalize(fake_home.path()).unwrap();
        let secret = fake_home.join("vault");
        std::fs::create_dir(&secret).unwrap();
        std::fs::write(secret.join("key"), "vault-secret").unwrap();
        std::fs::write(fake_home.join("notes"), "plain-notes").unwrap();
        let stdout = sandboxed_stdout_with(
            GitDirAccess::ReadWrite,
            Vec::new(),
            Vec::new(),
            vec![secret.clone()],
            format!(
                "cat {}/key 2>&1; echo rc=$?; cat {}/notes 2>&1",
                secret.display(),
                fake_home.display()
            ),
        )
        .await;
        assert!(!stdout.contains("vault-secret"), "{stdout}");
        assert!(!stdout.contains("rc=0"), "{stdout}");
        assert!(stdout.contains("plain-notes"), "{stdout}");
    }

    /// Runs the host's `git` in `dir`; the caller's environment keeps the
    /// user's configuration out.
    #[cfg(unix)]
    fn host_git(dir: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .current_dir(dir)
            .args(["-c", "user.name=t", "-c", "user.email=t@t"])
            .args(args)
            .status()
            .expect("git runs");
        assert!(status.success(), "git {args:?} in {}", dir.display());
    }

    /// A repository `main` with a linked worktree `linked`, under a fresh
    /// canonical temporary directory.
    #[cfg(unix)]
    fn repository_with_linked_worktree() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let base = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(base.path()).unwrap();
        let main = root.join("main");
        let linked = root.join("linked");
        std::fs::create_dir(&main).unwrap();
        host_git(&main, &["init", "-q", "-b", "main"]);
        host_git(&main, &["commit", "-q", "--allow-empty", "-m", "root"]);
        host_git(&main, &["worktree", "add", "-q", "../linked"]);
        (base, main, linked)
    }

    /// Under strict isolation git works in a granted linked worktree: its
    /// repository is outside the worktree, and the session grants it after
    /// git's back-links confirm the pointer.
    #[cfg(unix)]
    #[tokio::test]
    async fn git_commits_in_a_granted_linked_worktree() {
        let (_base, main, linked) = repository_with_linked_worktree();
        let script = format!(
            "cd {linked} && git status --short 2>&1 && echo changed > file && git add file 2>&1 \
             && git -c user.name=t -c user.email=t@t commit -q -m inside 2>&1 && echo committed; true",
            linked = linked.display(),
        );
        let stdout = sandboxed_stdout(vec![linked.clone()], Vec::new(), script).await;
        assert_eq!(stdout.lines().last(), Some("committed"), "{stdout}");
        let subject = std::process::Command::new("git")
            .current_dir(&main)
            .args(["log", "-1", "--format=%s", "linked"])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&subject.stdout).trim(), "inside");
    }

    /// A subagent's terminal holding a plain worktree grant reads the main
    /// repository's git data and cannot write it: `git commit` fails and the
    /// branch stays where it was. The session's own terminal commits (above).
    #[cfg(unix)]
    #[tokio::test]
    async fn a_derived_terminal_reads_but_cannot_commit_in_a_plain_worktree_grant() {
        let (_base, main, linked) = repository_with_linked_worktree();
        let tip = |branch: &str| {
            let output = std::process::Command::new("git")
                .current_dir(&main)
                .args(["rev-parse", branch])
                .output()
                .unwrap();
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        };
        let before = tip("linked");
        let script = format!(
            "cd {linked} && git status --short >/dev/null 2>&1 && echo status-ok; \
             git log -1 --format=%s 2>&1 | head -n 1; \
             echo changed > file; git add file 2>&1; \
             git -c user.name=t -c user.email=t@t commit -q -m inside 2>&1 && echo committed; true",
            linked = linked.display(),
        );
        let stdout = sandboxed_stdout_with(
            GitDirAccess::ReadOnly,
            vec![linked.clone()],
            Vec::new(),
            Vec::new(),
            script,
        )
        .await;
        assert!(stdout.lines().any(|line| line == "status-ok"), "{stdout}");
        assert!(stdout.lines().any(|line| line == "root"), "{stdout}");
        assert!(!stdout.contains("committed"), "{stdout}");
        assert_eq!(tip("linked"), before, "{stdout}");
    }

    /// The grants a subagent's terminal derives give worktree git data read
    /// access; the session's own give it write access.
    #[test]
    fn only_derived_grants_read_worktree_git_data() {
        let parent = Grants::new();
        assert_eq!(parent.snapshot().git_dir, GitDirAccess::ReadWrite);
        let derived = parent.derived(crate::GrantSnapshot {
            writable: vec![PathBuf::from("/w")],
            ..Default::default()
        });
        assert_eq!(derived.snapshot().git_dir, GitDirAccess::ReadOnly);
    }

    /// In a read-write repository the persistence paths stay unwritable, in a
    /// linked worktree's common git directory too, while git still commits.
    #[cfg(unix)]
    #[tokio::test]
    async fn git_persistence_paths_are_unwritable_in_granted_repositories() {
        let (_base, main, linked) = repository_with_linked_worktree();
        let attempt = |repository: &Path, work: &Path| {
            format!(
                "cd {work} && echo '#!/bin/sh' > {hook} 2>/dev/null && echo hook-written; \
                 echo '[alias]' >> {config} 2>/dev/null && echo config-written; \
                 echo changed > file && git add file 2>&1 \
                 && git -c user.name=t -c user.email=t@t commit -q -m inside 2>&1 && echo committed; true",
                work = work.display(),
                hook = repository.join(".git/hooks/pre-commit").display(),
                config = repository.join(".git/config").display(),
            )
        };
        for (repository, work, writable) in [
            (&main, &main, vec![main.clone()]),
            (&main, &linked, vec![linked.clone()]),
        ] {
            let stdout = sandboxed_stdout(writable, Vec::new(), attempt(repository, work)).await;
            assert_eq!(stdout.lines().last(), Some("committed"), "{stdout}");
            assert!(!stdout.contains("written"), "{stdout}");
            assert!(!main.join(".git/hooks/pre-commit").exists());
        }
    }

    /// The persistence table forbids creating a `.git` in a granted
    /// directory: a new repository there is refused, one in the session
    /// directory (see the committing subagent's clone test) is not.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_new_repository_in_a_granted_workdir_is_refused() {
        let base = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(base.path()).unwrap();
        let script = format!(
            "cd {root} && if git init fresh >/dev/null 2>&1; then echo init-succeeded; fi; true",
            root = root.display()
        );
        let stdout = sandboxed_stdout(vec![root.clone()], Vec::new(), script).await;
        assert!(!stdout.contains("init-succeeded"), "{stdout}");
        assert!(!root.join("fresh/.git/config").exists(), "{stdout}");
    }

    /// A readable linked worktree brings its repository with read access:
    /// git reads, and nothing is written.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_readable_linked_worktree_reads_its_repository() {
        let (_base, _main, linked) = repository_with_linked_worktree();
        let script = format!(
            "cd {linked} && git log -1 --format=%s 2>&1; true",
            linked = linked.display(),
        );
        let stdout = sandboxed_stdout(Vec::new(), vec![linked.clone()], script).await;
        assert_eq!(stdout.lines().last(), Some("root"), "{stdout}");
    }

    /// A `.git` file anyone can write names whatever it likes: without
    /// matching back-links it adds no grant, and git there gets nothing of
    /// the repository it points at.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_forged_git_pointer_adds_no_grant() {
        let (base, main, linked) = repository_with_linked_worktree();
        let root = std::fs::canonicalize(base.path()).unwrap();
        let entry = main.join(".git/worktrees/linked");
        let forged = root.join("forged");
        std::fs::create_dir(&forged).unwrap();
        // The real entry of another worktree: its `gitdir` names `linked`,
        // not `forged`.
        std::fs::write(
            forged.join(".git"),
            format!("gitdir: {}\n", entry.display()),
        )
        .unwrap();
        // A pointer straight at the repository, and one at nothing.
        let direct = root.join("direct");
        std::fs::create_dir(&direct).unwrap();
        std::fs::write(
            direct.join(".git"),
            format!("gitdir: {}\n", main.join(".git").display()),
        )
        .unwrap();
        let writable = vec![forged.clone(), direct.clone()];
        for path in &writable {
            assert!(
                crate::subagent::confirmed_common_dir(path).await.is_none(),
                "{} confirmed",
                path.display()
            );
        }
        assert_eq!(
            crate::subagent::confirmed_common_dir(&linked).await,
            Some(main.join(".git")),
        );
        assert_eq!(
            crate::spawn::host::linked_worktree_grants(&writable, &[], GitDirAccess::ReadWrite)
                .await
                .unwrap()
                .len(),
            0
        );

        // Reads are open under the read denylist policy, so the grant is
        // probed by what it would make writable: the repository, directly and
        // through git run in the forged worktree.
        let planted = main.join(".git/forged_probe");
        let script = format!(
            "echo x > {planted} 2>&1; git -C {forged} update-ref refs/heads/forged_probe HEAD 2>&1; \
             git -C {direct} update-ref refs/heads/direct_probe HEAD 2>&1; true",
            planted = planted.display(),
            forged = forged.display(),
            direct = direct.display(),
        );
        sandboxed_stdout(writable, Vec::new(), script).await;
        assert!(!planted.exists(), "the repository became writable");
        for branch in ["forged_probe", "direct_probe"] {
            assert!(
                !main.join(".git/refs/heads").join(branch).exists(),
                "git wrote {branch} into the repository through a forged pointer"
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn failed_command_reports_nonzero_exit_code() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = fresh_terminal(dir.path()).await;

        let result = tool
            .call(
                ShellArgs {
                    description: "exit with non-zero status".to_string(),
                    script: "exit 42".to_string(),
                    interpreter: Interpreter::Shell,
                    sandbox: SandboxMode::Sandboxed,
                    location: None,
                    resolution: MediaResolution::Auto,
                    max_lines: Some(50),
                    raw: false,
                    timeout: 30,
                    background: false,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal call should not be promoted to transport error on non-zero exit");

        assert!(
            !result.is_error(),
            "non-zero exit must not produce ToolResult::Error, got: {result:?}"
        );
        assert_eq!(
            render_terminal_result(&result),
            "(no output)\nExit code: 42.\n"
        );
        assert_eq!(
            result.command().map(|command| &command.outcome),
            Some(&CommandOutcome::Exited { code: 42 })
        );
    }

    /// What a host reading the registry can tell: whether the session still
    /// owes its agent a turn.
    fn host_sees_owed_turn(jobs: &[crate::JobInfo]) -> bool {
        jobs.iter()
            .any(|job| job.status == JobStatus::Running || job.status.completion_pending())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_detached_job_reads_pending_until_its_completion_is_taken() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = fresh_terminal(dir.path()).await;
        let receiver = tool.background_receiver();
        let jobs = tool.job_registry();

        tool.run_background(ShellArgs {
            description: "finish at once".to_string(),
            script: "exit 3".to_string(),
            interpreter: Interpreter::Shell,
            sandbox: SandboxMode::Sandboxed,
            location: None,
            resolution: MediaResolution::Auto,
            max_lines: Some(50),
            raw: false,
            timeout: 30,
            background: true,
        })
        .await
        .expect("terminal call should succeed");
        let task_id = started_task_id(&receiver);

        // The agent's receiver is the gate: the completion it receives is
        // held, not yet taken, while the host reads the registry.
        let completed = tokio::time::timeout(Duration::from_secs(30), receiver.recv_completed())
            .await
            .expect("the job should deliver its completion");
        assert_eq!(completed.info.task_id, task_id);
        let held = jobs.list().await;
        assert!(
            held.iter().all(|job| job.status != JobStatus::Running),
            "the job has finished: {held:?}"
        );
        assert!(
            host_sees_owed_turn(&held),
            "a finished job whose completion the agent has not taken reads as done: {held:?}"
        );

        // The agent takes the completion; the registry serves the take
        // before the list that follows it, and forgets the job.
        drop(completed);
        let taken = jobs.list().await;
        assert!(taken.is_empty(), "{taken:?}");
        assert!(!host_sees_owed_turn(&taken), "{taken:?}");
        assert!(
            jobs.get_by_task_id(&task_id).await.is_none(),
            "a delivered job's id is unknown"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn background_terminal_accepts_terminal_input() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = fresh_terminal(dir.path()).await;
        let background_receiver = tool.background_receiver();
        let input_tool = input_tool(tool.jobs());

        let result = tool
            .call_detached(ShellArgs {
                description: "wait for terminal input and echo it back".to_string(),
                script: "printf 'name? '; read name; printf 'hello %s\\n' \"$name\"".to_string(),
                interpreter: Interpreter::Shell,
                sandbox: SandboxMode::Sandboxed,
                location: None,
                resolution: MediaResolution::Auto,
                max_lines: Some(50),
                raw: false,
                timeout: 30,
                background: true,
            })
            .await
            .expect("terminal call should succeed");

        let rendered = render_terminal_result(&result);
        assert!(
            rendered.contains("Started in background as task '"),
            "{rendered}"
        );
        let task_id = started_task_id(&tool.background_receiver());

        input_tool
            .call(
                JobInputArgs {
                    task_id: task_id.clone(),
                    input: "lexo".to_string(),
                    append_newline: true,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal_input should succeed");

        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        let terminal_result = completed
            .result
            .expect("background task should complete successfully after stdin input");
        let stdout =
            output_text(&terminal_result.stdout).expect("completed task should include stdout");
        assert!(
            stdout.contains("hello lexo"),
            "stdin input should reach the terminal process, got: {stdout}"
        );
    }

    /// A job is read, waited for and killed through its handle, and an
    /// unchanged read is data.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_job_handle_reads_waits_and_kills() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = fresh_terminal(dir.path()).await;
        let events = tool.background_receiver();
        let jobs = tool.jobs();

        let args = |script: &str| ShellArgs {
            background: true,
            timeout: 60,
            ..terminal_args("a job", script)
        };
        tool.run_background(args("printf done; exit 3"))
            .await
            .expect("the job starts");
        let finished = jobs
            .get(&started_task_id(&events))
            .await
            .expect("the shell holds the job");
        let result = finished.wait().await.expect("the job ends");
        assert_eq!(result.outcome, crate::RunOutcome::Exited { code: 3 });
        assert_eq!(output_text(&result.stdout), Some("done"));

        tool.run_background(args("sleep 60"))
            .await
            .expect("the job starts");
        let sleeping = jobs
            .get(&started_task_id(&events))
            .await
            .expect("the shell holds the job");
        let first = sleeping.read(0, 1024).await.expect("the job reads");
        let cursor = first.into_delta().expect("a first read is a delta").cursor;
        assert_eq!(
            sleeping.read(cursor, 1024).await,
            Ok(crate::JobRead::Unchanged {
                task_id: sleeping.task_id().to_owned()
            }),
            "a repeat read is a poll"
        );
        // The read tool refuses a poll with a typed reason the loop can
        // tell from a read.
        let read_tool = crate::JobReadTool::new(
            crate::test_wording::job_tools().read,
            jobs.clone(),
            TestWording::any(),
            crate::output_policy::fixture::policy(),
        );
        let refusal = read_tool
            .call(
                crate::JobReadArgs {
                    task_id: sleeping.task_id().to_owned(),
                    cursor,
                    max_bytes: None,
                },
                ToolContext::new(),
            )
            .await
            .expect_err("a repeat read is refused");
        let refusal = refusal
            .downcast_ref::<crate::Refused<crate::UnchangedRead>>()
            .expect("the refusal is typed");
        assert_eq!(refusal.reason().task_id, sleeping.task_id());
        assert!(
            refusal.text().contains("produced no new output"),
            "{refusal}"
        );
        assert_eq!(jobs.kill_all().await, 1);
        let killed = sleeping.wait().await.expect("the killed job ends");
        assert_eq!(
            killed.outcome,
            crate::RunOutcome::Killed(crate::KillCause::User)
        );
        assert!(
            jobs.list()
                .await
                .iter()
                .all(|job| !matches!(job.status, JobStatus::Running))
        );
    }

    /// A result the call returned, and a run, are no job's end: nothing is
    /// raised for them, so a caller that never reads events piles up none.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_foreground_result_and_a_run_raise_no_job_end() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;
        let events = tool.background_receiver();
        tool.call(terminal_args("a call", "printf done"), ToolContext::new())
            .await
            .expect("the call returns its result");
        let start = std::fs::canonicalize(dir.path()).expect("the directory resolves");
        tool.run(Script::new("printf done", start, Duration::from_secs(30)))
            .await
            .expect("the run ends");
        assert!(events.is_empty(), "no event was raised");
    }

    /// A job read through its handle owes its agent no turn, and the handle
    /// keeps the job after its completion was taken.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_waited_job_owes_its_agent_nothing_and_its_handle_keeps_it() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;
        let events = tool.background_receiver();
        let jobs = tool.jobs();
        let args = ShellArgs {
            background: true,
            ..terminal_args("a job", "sleep 1; printf done")
        };
        tool.run_background(args).await.expect("the job starts");
        let job = jobs
            .get(&started_task_id(&events))
            .await
            .expect("the shell holds the job");
        let result = job.wait().await.expect("the job ends");
        assert_eq!(result.outcome, crate::RunOutcome::Exited { code: 0 });
        let ended = events.recv_completed().await;
        assert!(ended.pending.is_none(), "a waiter took the completion");
        drop(ended);
        // The handle holds the job, which still reads and waits.
        let again = job.wait().await.expect("the held job waits again");
        assert_eq!(output_text(&again.stdout), Some("done"));
        assert_eq!(jobs.list().await.len(), 1);
        drop(job);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn killing_background_terminal_kills_descendant_processes() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = fresh_terminal(dir.path()).await;
        let child_pid_path = dir.path().join("child.pid");

        let result = tool
            .call_detached(ShellArgs {
                description: "start a descendant process".to_string(),
                script: "sleep 30 & echo $! > child.pid; wait".to_string(),
                interpreter: Interpreter::Shell,
                sandbox: SandboxMode::Sandboxed,
                location: None,
                resolution: MediaResolution::Auto,
                max_lines: Some(50),
                raw: false,
                timeout: 30,
                background: true,
            })
            .await
            .expect("terminal call should succeed");
        let rendered = render_terminal_result(&result);
        assert!(
            rendered.contains("Started in background as task '"),
            "{rendered}"
        );

        let mut child_pid = None;
        for _ in 0..100 {
            // `echo $! > child.pid` creates the file before writing to it,
            // so a successful read can still yield empty or partial
            // content — poll until the pid actually parses.
            let parsed = match async_fs::read_to_string(&child_pid_path).await {
                Ok(value) => value.trim().parse::<i32>().ok(),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => panic!("failed to read child pid: {error}"),
            };
            if let Some(pid) = parsed {
                child_pid = Some(pid);
                break;
            }
            async_io::Timer::after(std::time::Duration::from_millis(20)).await;
        }
        let child_pid = child_pid.expect("descendant pid should be written before timeout");

        assert_eq!(tool.job_registry().kill_running().await, 1);

        for _ in 0..100 {
            let exists = unsafe { libc::kill(child_pid, 0) } == 0;
            if !exists {
                return;
            }
            async_io::Timer::after(std::time::Duration::from_millis(20)).await;
        }
        panic!("descendant process {child_pid} survived terminal cancellation");
    }

    /// A foreground command blocked on stdin sees no EOF: every job owns a
    /// live stdin pipe, so at the foreground window the call promotes it to
    /// a background task the model feeds through `terminal_input`.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn foreground_command_blocked_on_stdin_promotes_at_the_window() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(1),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;
        let background_receiver = tool.background_receiver();

        let result = tool
            .call(
                terminal_args("read all of stdin", "cat"),
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");

        let command = result.command().expect("terminal returns a command result");
        let CommandOutcome::Running { task_id, .. } = &command.outcome else {
            panic!(
                "cat blocked on stdin must promote to a task, got {:?}",
                command.outcome
            );
        };
        assert_eq!(started_task_id(&background_receiver), *task_id);

        assert!(
            tool.job_registry().kill_by_task_id(task_id).await,
            "the promoted task must still be running"
        );
    }

    #[cfg(unix)]
    fn terminal_args(description: &str, script: &str) -> ShellArgs {
        ShellArgs {
            description: description.to_string(),
            script: script.to_string(),
            interpreter: Interpreter::Shell,
            sandbox: SandboxMode::Sandboxed,
            location: None,
            resolution: MediaResolution::Auto,
            max_lines: Some(50),
            raw: false,
            timeout: 30,
            background: false,
        }
    }

    /// Starts `script` as a background job and returns its task id.
    #[cfg(unix)]
    async fn start_background_job(
        tool: &Shell<executor_core::tokio::TokioGlobal>,
        script: &str,
    ) -> String {
        let job = tool
            .run_background(ShellArgs {
                background: true,
                ..terminal_args("background job", script)
            })
            .await
            .expect("terminal should start the job");
        job.task_id().to_owned()
    }

    /// Python terminal arguments for `code`.
    #[cfg(unix)]
    fn python_args(code: &str) -> ShellArgs {
        ShellArgs {
            interpreter: Interpreter::Python,
            ..terminal_args("run python", code)
        }
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn python_runs_and_a_raised_exception_carries_file_line_numbers() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(20),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await;

        let ok = tool
            .call(
                python_args("import sys\nprint(sys.version_info[0] * 6)"),
                ToolContext::new(),
            )
            .await
            .expect("python should run");
        let ok = ok.command().expect("a command result");
        assert_eq!(ok.outcome, CommandOutcome::Exited { code: 0 });
        assert_eq!(output_text(&ok.stdout).map(str::trim), Some("18"));

        let failed = tool
            .call(
                python_args("x = 1\nraise ValueError('boom')"),
                ToolContext::new(),
            )
            .await
            .expect("a raising script is still a result");
        let failed = failed.command().expect("a command result");
        assert_eq!(failed.outcome, CommandOutcome::Exited { code: 1 });
        let stderr = failed
            .stderr
            .as_ref()
            .and_then(output_text)
            .expect("the traceback is on stderr");
        assert!(stderr.contains("line 2"), "{stderr}");
        assert!(stderr.contains("ValueError: boom"), "{stderr}");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn python_background_job_is_read_and_killed_like_a_shell_job() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await;
        let receiver = tool.background_receiver();
        let job = tool
            .run_background(python_args(
                "import time\nprint('ready', flush=True)\ntime.sleep(60)",
            ))
            .await
            .expect("python should start detached");
        let task_id = job.task_id().to_owned();
        assert_eq!(started_task_id(&receiver), task_id);

        let jobs = tool.job_registry().list().await;
        let job = jobs
            .iter()
            .find(|job| job.task_id == task_id)
            .expect("the job is recorded");
        assert_eq!(job.interpreter, Interpreter::Python);
        assert!(job.script.contains("time.sleep(60)"), "{}", job.script);

        let handle = tool
            .jobs()
            .get(&task_id)
            .await
            .expect("the shell holds the job");
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            let read = handle.read(0, 4096).await.expect("the job reads");
            if read
                .into_delta()
                .is_some_and(|delta| String::from_utf8_lossy(&delta.bytes).contains("ready"))
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the job never printed"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        drop(handle);

        assert!(tool.job_registry().kill_by_task_id(&task_id).await);
        wait_for_completed_task(&receiver, &task_id).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unsafe_python_asks_for_approval_with_its_code() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let refusing = Approvals::denying("no");
        let tool = approval_tool(dir.path(), [], Some(refusing.clone())).await;

        let code = "import os\nprint(os.getcwd())";
        let refused = tool
            .call(
                ShellArgs {
                    sandbox: SandboxMode::Unsandboxed,
                    ..python_args(code)
                },
                ToolContext::new(),
            )
            .await;
        assert!(refused.is_err(), "the refused approval stops the run");
        let asked = refusing.asked();
        assert_eq!(asked.len(), 1);
        assert_eq!(asked[0].subject(), code);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn omitting_language_runs_shell() {
        let args: ShellArgs = serde_json::from_value(serde_json::json!({
            "description": "print",
            "script": "echo $((6 * 7))",
            "timeout": 30,
        }))
        .expect("language is optional");
        assert_eq!(args.interpreter, Interpreter::Shell);

        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(20),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await;
        let result = tool
            .call(args, ToolContext::new())
            .await
            .expect("shell should run");
        let command = result.command().expect("a command result");
        assert_eq!(output_text(&command.stdout).map(str::trim), Some("42"));
    }

    /// Collects the alerts that arrive within `window`.
    #[cfg(unix)]
    async fn alerts_within(receiver: &ShellEvents, window: Duration) -> Vec<ShellEvent> {
        let deadline = std::time::Instant::now() + window;
        let mut alerts = Vec::new();
        while std::time::Instant::now() < deadline {
            alerts.extend(receiver.take_job_alerts());
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        alerts
    }

    /// A foreground container command's stdin reads EOF at once: only a
    /// background job's stdin is a pipe the agent feeds.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn container_foreground_command_reads_eof_from_stdin() {
        /// A container executor that asserts stdin is at EOF.
        struct ForegroundStdinPipeExec;

        impl crate::ContainerExec for ForegroundStdinPipeExec {
            fn exec(
                &self,
                _container_id: &str,
                _script: &str,
                _working_dir: &str,
                _kill_rx: Receiver<()>,
                _stdin_rx: Receiver<Vec<u8>>,
                _lifetime: ExecLifetime,
            ) -> impl std::future::Future<Output = Result<ContainerExecOutcome, String>> + Send
            {
                std::future::ready(Err("a job streams its output".to_string()))
            }

            async fn exec_streamed(
                &self,
                _container_id: &str,
                _script: &str,
                _working_dir: &str,
                _kill_rx: Receiver<()>,
                stdin_rx: Receiver<Vec<u8>>,
                _stdout: Sender<Vec<u8>>,
                lifetime: ExecLifetime,
            ) -> Result<ContainerExecOutcome, String> {
                assert!(
                    matches!(lifetime, ExecLifetime::Foreground),
                    "a foreground command's descendants end with it"
                );
                let eof = futures_lite::future::or(async { Some(stdin_rx.recv().await) }, async {
                    async_io::Timer::after(Duration::from_secs(5)).await;
                    None
                })
                .await
                .expect("a foreground exec's stdin must be EOF at once, not left open");
                assert!(eof.is_err(), "a foreground exec's stdin must carry no data");
                Ok(ContainerExecOutcome::Completed {
                    output: std::process::Output {
                        status: std::os::unix::process::ExitStatusExt::from_raw(0),
                        stdout: Vec::new(),
                        stderr: Vec::new(),
                    },
                    killed: Vec::new(),
                })
            }
        }

        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (job_registry, job_registry_service) =
            job_registry_channel(crate::output_policy::fixture::policy());
        tokio::spawn(async move { job_registry_service.serve().await });
        let runtime = ContainerShellRuntime::new(
            "aither-test-foreground-stdin",
            crate::HostRoute::new(
                "host.docker.internal",
                std::net::IpAddr::from([127, 0, 0, 1]),
            )
            .expect("loopback is routable"),
            crate::AnyContainerExec::new(Arc::new(ForegroundStdinPipeExec)),
        );
        let (startup_tx, _startup_rx) = async_channel::bounded(1);
        let alert_tx = ShellEvents::new().sender(AgentPath::main());
        let (_pid, output) = execute_container_job(ContainerJobExecution {
            executor: executor_core::tokio::TokioGlobal,
            sandbox: &container_sandbox(&runtime, dir.path()),
            ipc: None,
            committed: &[],
            task_id: &crate::test_ids::lease("probe"),
            info: crate::LaunchInfo::default(),
            execution_id: "exec-test",
            script: "cat",
            interpreter: Interpreter::Shell,
            mode: SandboxMode::Sandboxed,
            container_runtime: &runtime,
            job_registry: &job_registry,
            stdin_alerts: Some(StdinAlerts {
                tx: alert_tx.clone(),
                task_id: "probe".to_string(),
                script: "cat".to_string(),
            }),
            lifetime: foreground_lifetime(alert_tx),
            background_startup: &BackgroundStartup::new(startup_tx),
            carry: None,
            #[cfg(all(unix, feature = "jobserver"))]
            job_fifo: None,
        })
        .await
        .expect("a foreground container exec should complete");
        assert_eq!(output.status.code(), Some(0));
    }

    /// A container command's exec carries the commands the shell's provider
    /// serves for that launch. (No container runs here: the exec is a fake
    /// that records the script it was given, so the serving of the answer is
    /// covered by `container_ipc_bridge_round_trips_a_command_over_tcp`.)
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_container_command_is_given_the_providers_router_for_its_launch() {
        struct RecordingExec(Sender<String>);

        impl crate::ContainerExec for RecordingExec {
            fn exec(
                &self,
                _container_id: &str,
                _script: &str,
                _working_dir: &str,
                _kill_rx: Receiver<()>,
                _stdin_rx: Receiver<Vec<u8>>,
                _lifetime: ExecLifetime,
            ) -> impl std::future::Future<Output = Result<ContainerExecOutcome, String>> + Send
            {
                std::future::ready(Err("a job streams its output".to_string()))
            }

            async fn exec_streamed(
                &self,
                _container_id: &str,
                script: &str,
                _working_dir: &str,
                _kill_rx: Receiver<()>,
                _stdin_rx: Receiver<Vec<u8>>,
                _stdout: Sender<Vec<u8>>,
                _lifetime: ExecLifetime,
            ) -> Result<ContainerExecOutcome, String> {
                self.0.send(script.to_owned()).await.unwrap();
                Ok(ContainerExecOutcome::Completed {
                    output: std::process::Output {
                        status: std::os::unix::process::ExitStatusExt::from_raw(0),
                        stdout: Vec::new(),
                        stderr: Vec::new(),
                    },
                    killed: Vec::new(),
                })
            }
        }

        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (job_registry, job_registry_service) =
            job_registry_channel(crate::output_policy::fixture::policy());
        tokio::spawn(async move { job_registry_service.serve().await });
        let (scripts_tx, scripts) = async_channel::unbounded();
        let runtime = ContainerShellRuntime::new(
            "aither-test-ipc-provider",
            crate::HostRoute::new(
                "host.docker.internal",
                std::net::IpAddr::from([127, 0, 0, 1]),
            )
            .expect("loopback is routable"),
            crate::AnyContainerExec::new(Arc::new(RecordingExec(scripts_tx))),
        );
        let provider = crate::spawn::ipc_fixture::Recording::default();
        let erased = crate::AnyIpcProvider::new(provider.clone());
        let info = crate::LaunchInfo::new("probe", None, SandboxMode::Sandboxed);
        let (startup_tx, _startup_rx) = async_channel::bounded(1);
        let alert_tx = ShellEvents::new().sender(AgentPath::main());
        execute_container_job(ContainerJobExecution {
            executor: executor_core::tokio::TokioGlobal,
            sandbox: &container_sandbox(&runtime, dir.path()),
            ipc: Some(&erased),
            committed: &[],
            task_id: &crate::test_ids::lease("probe"),
            info: info.clone(),
            execution_id: "exec-test",
            script: "probe",
            interpreter: Interpreter::Shell,
            mode: SandboxMode::Sandboxed,
            container_runtime: &runtime,
            job_registry: &job_registry,
            stdin_alerts: None,
            lifetime: foreground_lifetime(alert_tx),
            background_startup: &BackgroundStartup::new(startup_tx),
            carry: None,
            #[cfg(all(unix, feature = "jobserver"))]
            job_fifo: None,
        })
        .await
        .expect("the exec completes");
        let script = scripts.recv().await.expect("the exec ran a script");
        assert!(script.contains("for cmd in 'probe'; do"), "{script}");
        let launches = provider.launches.lock().unwrap();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].task_id.as_deref(), Some("probe"));
        assert_eq!(launches[0].mode, SandboxMode::Sandboxed);
    }

    /// With a pool configured, a container command's script exports the
    /// jobserver variables naming its own fifo inside the container, and the
    /// fifo's relay is told once the command's exec ends.
    /// Captures each command's script; plays a jobserver relay that reports
    /// ready and records what the host tells it.
    #[cfg(all(unix, feature = "jobserver"))]
    struct CapturingExec {
        scripts: Sender<String>,
        relay_heard: Sender<Vec<u8>>,
        /// How many execs have started: the first is the jobserver relay.
        started: std::sync::atomic::AtomicUsize,
    }

    #[cfg(all(unix, feature = "jobserver"))]
    impl crate::ContainerExec for CapturingExec {
        fn exec(
            &self,
            _container_id: &str,
            _script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _lifetime: ExecLifetime,
        ) -> impl std::future::Future<Output = Result<ContainerExecOutcome, String>> + Send
        {
            std::future::ready(Err("a job streams its output".to_string()))
        }

        async fn exec_streamed(
            &self,
            _container_id: &str,
            script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            stdin_rx: Receiver<Vec<u8>>,
            stdout: Sender<Vec<u8>>,
            _lifetime: ExecLifetime,
        ) -> Result<ContainerExecOutcome, String> {
            if self
                .started
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                > 0
            {
                self.scripts.send(script.to_string()).await.unwrap();
                return Ok(ContainerExecOutcome::Completed {
                    output: std::process::Output {
                        status: std::os::unix::process::ExitStatusExt::from_raw(0),
                        stdout: Vec::new(),
                        stderr: Vec::new(),
                    },
                    killed: Vec::new(),
                });
            }
            stdout.send(b"R".to_vec()).await.unwrap();
            while let Ok(message) = stdin_rx.recv().await {
                let finished = message == b"F";
                self.relay_heard.send(message).await.unwrap();
                if finished {
                    break;
                }
            }
            Err("the relay exits".to_string())
        }
    }

    #[cfg(all(unix, feature = "jobserver"))]
    #[tokio::test(flavor = "multi_thread")]
    async fn container_command_environment_names_its_own_fifo() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let pool = crate::jobserver::JobPool::create_fifo(dir.path().join("pool.fifo"), 2)
            .expect("pool should be created");
        let (broker, service) =
            crate::jobserver::job_broker_channel(pool, dir.path().join("commands"))
                .expect("broker should start");
        tokio::spawn(async move { service.serve().await });
        let (job_registry, job_registry_service) =
            job_registry_channel(crate::output_policy::fixture::policy());
        tokio::spawn(async move { job_registry_service.serve().await });
        let (scripts_tx, scripts) = async_channel::unbounded();
        let (heard_tx, heard) = async_channel::unbounded();
        let runtime = ContainerShellRuntime::new(
            "aither-test-jobserver",
            crate::HostRoute::new(
                "host.docker.internal",
                std::net::IpAddr::from([127, 0, 0, 1]),
            )
            .expect("loopback is routable"),
            crate::AnyContainerExec::new(Arc::new(CapturingExec {
                scripts: scripts_tx,
                relay_heard: heard_tx,
                started: std::sync::atomic::AtomicUsize::new(0),
            })),
        );
        let job_fifo = crate::container_jobserver::open_container_job_fifo(
            &broker,
            &runtime,
            &test_container_launch().bin_dir().join("python3"),
            executor_core::tokio::TokioGlobal,
        )
        .await
        .expect("the relay reports ready");
        let fifo_path = job_fifo.path().clone();
        let (startup_tx, _startup_rx) = async_channel::bounded(1);
        let alert_tx = ShellEvents::new().sender(AgentPath::main());
        execute_container_job(ContainerJobExecution {
            executor: executor_core::tokio::TokioGlobal,
            sandbox: &container_sandbox(&runtime, dir.path()),
            ipc: None,
            committed: &[],
            task_id: &crate::test_ids::lease("probe"),
            info: crate::LaunchInfo::default(),
            execution_id: "exec-test",
            script: "make",
            interpreter: Interpreter::Shell,
            mode: SandboxMode::Sandboxed,
            container_runtime: &runtime,
            job_registry: &job_registry,
            stdin_alerts: None,
            lifetime: foreground_lifetime(alert_tx),
            background_startup: &BackgroundStartup::new(startup_tx),
            carry: None,
            job_fifo: Some(job_fifo),
        })
        .await
        .expect("the container exec completes");

        let script = scripts.recv().await.expect("the command ran");
        for name in ["MAKEFLAGS", "CARGO_MAKEFLAGS", "MFLAGS"] {
            assert!(
                script.contains(&format!("export {name}=")) && script.contains(fifo_path.as_str()),
                "{name} names the container fifo: {script}"
            );
        }
        assert!(!script.contains(dir.path().to_str().unwrap()));
        // The relay asked for nothing, so all it hears is the finish.
        assert_eq!(heard.recv().await.unwrap(), b"F");
    }

    /// A job whose `reader` script blocks on stdin yields exactly one alert
    /// until it is fed, and finishes once it is.
    #[cfg(unix)]
    async fn stdin_reader_alerts_once_and_resumes_when_fed(reader: &str) {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await;
        let background_receiver = tool.background_receiver();
        let task_id = start_background_job(&tool, reader).await;

        let alerts = alerts_within(&background_receiver, Duration::from_secs(8)).await;
        assert!(
            matches!(
                alerts.as_slice(),
                [ShellEvent::StdinBlocked(job)] if job.task_id == task_id
            ),
            "a blocked reader must yield exactly one stdin alert: {alerts:?}"
        );

        input_tool(tool.jobs())
            .call(
                JobInputArgs {
                    task_id: task_id.clone(),
                    input: "lexo".to_string(),
                    append_newline: true,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal_input should succeed");
        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        let terminal_result = completed.result.expect("the fed job should complete");
        let stdout =
            output_text(&terminal_result.stdout).expect("completed task should include stdout");
        assert!(stdout.contains("hello lexo"), "got: {stdout}");
        assert!(
            background_receiver.take_job_alerts().is_empty(),
            "a fed job that finished must not alert again"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_job_blocked_on_stdin_yields_one_alert() {
        stdin_reader_alerts_once_and_resumes_when_fed("read name; printf 'hello %s\\n' \"$name\"")
            .await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_grandchild_blocked_on_stdin_yields_one_alert() {
        stdin_reader_alerts_once_and_resumes_when_fed(
            "( ( read name; printf 'hello %s\\n' \"$name\" ); true )",
        )
        .await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_job_that_never_reads_stdin_yields_no_alert() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await;
        let background_receiver = tool.background_receiver();
        let task_id = start_background_job(&tool, "sleep 3; echo finished").await;

        let alerts = alerts_within(&background_receiver, Duration::from_secs(4)).await;
        assert!(alerts.is_empty(), "no alert expected: {alerts:?}");
        wait_for_completed_task(&background_receiver, &task_id).await;
    }

    /// The nice value `level` steps below this test process lands at, capped
    /// at the kernel's lowest priority — the same computation heel applies.
    #[cfg(unix)]
    fn expected_lowered_nice(level: u8) -> i64 {
        // SAFETY: reads the calling process's own nice value.
        let current = unsafe { libc::getpriority(libc::PRIO_PROCESS as _, 0) };
        i64::from(current + i32::from(level)).min(19)
    }

    #[cfg(unix)]
    fn test_nice_priority() -> ProcessPriority {
        ProcessPriority::new(
            CpuPriority::Nice(NiceLevel::new(10).expect("nice level 10 is in range")),
            IoPriority::Normal,
        )
    }

    /// A terminal with the given shortened limits.
    #[cfg(unix)]
    async fn configured_test_terminal(
        dir: &Path,
        foreground_wait: Duration,
        timeout_cap: Duration,
        alert_after: Duration,
    ) -> Shell<executor_core::tokio::TokioGlobal> {
        let tool = fresh_terminal(dir).await;
        tool.limits.foreground_wait.set_normal(foreground_wait);
        tool.limits.timeout_cap.set(timeout_cap);
        tool.limits.long_job_alert.set(alert_after);
        tool
    }

    /// A foreground command still running at the window promotes to a
    /// background task instead of dying; its `timeout` then kills it with
    /// its descendants.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn foreground_command_still_running_at_the_window_promotes_and_times_out_with_descendants()
     {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(1),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;
        let background_receiver = tool.background_receiver();
        let child_pid_path = dir.path().join("child.pid");

        let result = tool
            .call(
                ShellArgs {
                    description: "start a descendant process".to_string(),
                    script: "sleep 30 & echo $! > child.pid; wait".to_string(),
                    interpreter: Interpreter::Shell,
                    sandbox: SandboxMode::Sandboxed,
                    location: None,
                    resolution: MediaResolution::Auto,
                    max_lines: Some(50),
                    raw: false,
                    timeout: 3,
                    background: false,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");
        let rendered = render_terminal_result(&result);
        assert!(
            rendered.contains("Started in background as task '"),
            "the still-running command must promote: {rendered}"
        );
        let task_id = started_task_id(&background_receiver);

        let child_pid = async_fs::read_to_string(&child_pid_path)
            .await
            .expect("descendant pid file should exist")
            .trim()
            .parse::<i32>()
            .expect("descendant pid should parse");
        for _ in 0..500 {
            // SAFETY: signal 0 probes process existence.
            if unsafe { libc::kill(child_pid, 0) } != 0 {
                break;
            }
            async_io::Timer::after(std::time::Duration::from_millis(20)).await;
        }
        assert!(
            unsafe { libc::kill(child_pid, 0) } != 0,
            "descendant process {child_pid} survived the timeout kill"
        );

        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        let info = tool
            .job_registry()
            .get_by_task_id(&task_id)
            .await
            .expect("the finished job stays in the registry");
        assert!(
            matches!(
                info.status,
                JobStatus::Killed {
                    cause: crate::KillCause::Timeout(_),
                    delivery: crate::Delivery::Pending
                }
            ),
            "the promoted job should be killed at its timeout: {completed:?}"
        );
    }

    /// A longer foreground window put in force through the shared handle
    /// keeps a command that outlasts the normal window in the foreground, and
    /// the normal window applies again once it is restored.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_longer_foreground_window_holds_until_restored() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(1),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;
        let wait = tool.foreground_wait();
        let run = || async {
            let result = tool
                .call(
                    ShellArgs {
                        description: "outlast the normal window".to_string(),
                        script: "sleep 2 & wait; echo finished".to_string(),
                        interpreter: Interpreter::Shell,
                        sandbox: SandboxMode::Sandboxed,
                        location: None,
                        resolution: MediaResolution::Auto,
                        max_lines: Some(50),
                        raw: false,
                        timeout: 30,
                        background: false,
                    },
                    ToolContext::new(),
                )
                .await
                .expect("terminal call should succeed");
            render_terminal_result(&result)
        };

        wait.extend(Duration::from_secs(20));
        let during = run().await;
        assert!(
            during.contains("finished") && !during.contains("Started in background"),
            "the longer window keeps the command in the foreground: {during}"
        );

        wait.restore();
        let after = run().await;
        assert!(
            after.contains("Started in background as task '"),
            "the normal window applies again: {after}"
        );
    }

    /// What a test leaves running after its shell exits.
    ///
    /// `Plain` is a `sleep` in the shell's process group. `Setsid` is a helper
    /// program that calls `setsid` and stays alive, so it also leaves the
    /// group and the session. The helper is built here because macOS ships no
    /// `setsid` binary, and because the kernel withholds the environment of
    /// Apple's own binaries (`sleep`, `perl`), which is what finds a `setsid`
    /// descendant once its parent is gone.
    #[cfg(unix)]
    #[derive(Clone, Copy, Debug)]
    enum Descendant {
        Plain,
        Setsid,
    }

    /// The name of the `setsid` helper program, and so of its process.
    #[cfg(unix)]
    const SETSID_HELPER: &str = "setsid-helper";

    /// The directory the sandbox may execute programs from: the shell's own
    /// working directory.
    #[cfg(unix)]
    fn exec_dir<E>(tool: &Shell<E>) -> PathBuf {
        tool.working_dir.clone()
    }

    /// Builds the `setsid` helper into `dir` and returns its path. The sandbox
    /// executes only what lies under a granted directory, so `dir` is the
    /// shell's [`exec_dir`].
    #[cfg(unix)]
    fn build_setsid_helper(dir: &Path) -> PathBuf {
        let source = dir.join("setsid_helper.rs");
        std::fs::write(
            &source,
            "extern \"C\" { fn setsid() -> i32; }\n\
             fn main() {\n\
                 let seconds: u64 = std::env::args().nth(1).expect(\"seconds\").parse().expect(\"seconds\");\n\
                 // SAFETY: `setsid` takes no arguments.\n\
                 if unsafe { setsid() } < 0 { std::process::exit(1); }\n\
                 std::thread::sleep(std::time::Duration::from_secs(seconds));\n\
             }\n",
        )
        .expect("helper source should be written");
        let binary = dir.join(SETSID_HELPER);
        let status = std::process::Command::new("rustc")
            .arg("--edition=2021")
            .arg("-o")
            .arg(&binary)
            .arg(&source)
            .status()
            .expect("rustc should run");
        assert!(status.success(), "the setsid helper should compile");
        binary
    }

    #[cfg(unix)]
    impl Descendant {
        /// The name the descendant's process has once it ran `exec`.
        const fn process_name(self) -> &'static str {
            match self {
                Self::Plain => "sleep",
                Self::Setsid => SETSID_HELPER,
            }
        }

        /// A command that starts the descendant for `seconds` detached from
        /// the shell's output, records its pid in `child.pid` and returns once
        /// the child exists.
        fn script(self, exec_dir: &Path, seconds: u32) -> String {
            // The foreground `sleep` guard refuses a plain `sleep 0.3`, so
            // the settle wait is a `kill -0` check on the recorded pid.
            match self {
                Self::Plain => format!(
                    "sleep {seconds} > /dev/null 2>&1 & echo $! > child.pid; until kill -0 \"$(cat child.pid)\" 2>/dev/null; do :; done"
                ),
                Self::Setsid => format!(
                    "'{}' {seconds} > /dev/null 2>&1 & echo $! > child.pid; until kill -0 \"$(cat child.pid)\" 2>/dev/null; do :; done",
                    build_setsid_helper(exec_dir).display()
                ),
            }
        }
    }

    /// The pid `child.pid` in `dir` holds, once the shell wrote it.
    #[cfg(unix)]
    async fn recorded_child_pid(dir: &Path) -> i32 {
        for _ in 0..250 {
            // The redirect creates the file before `echo` writes it, so an
            // empty read only means the pid is not there yet.
            if let Ok(text) = async_fs::read_to_string(dir.join("child.pid")).await
                && let Ok(pid) = text.trim().parse()
            {
                return pid;
            }
            async_io::Timer::after(Duration::from_millis(20)).await;
        }
        panic!("the descendant never recorded its pid");
    }

    /// Whether `pid` is a live process.
    #[cfg(unix)]
    fn process_exists(pid: i32) -> bool {
        // SAFETY: signal 0 probes process existence.
        unsafe { libc::kill(pid, 0) == 0 }
    }

    /// Fails unless `pid` is gone within a few seconds.
    #[cfg(unix)]
    async fn assert_process_gone(pid: i32, context: &str) {
        for _ in 0..250 {
            if !process_exists(pid) {
                return;
            }
            async_io::Timer::after(Duration::from_millis(20)).await;
        }
        panic!("process {pid} survived: {context}");
    }

    #[cfg(unix)]
    async fn foreground_command_leaving_a_descendant_kills_it_and_alerts_once(
        descendant: Descendant,
    ) {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;
        let receiver = tool.background_receiver();

        let result = tool
            .call(
                terminal_args(
                    "leave a descendant",
                    &descendant.script(&exec_dir(&tool), 300),
                ),
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");
        let command = result.command().expect("terminal returns a command result");
        assert!(
            matches!(command.outcome, CommandOutcome::Exited { code: 0 }),
            "{:?}",
            command.outcome
        );

        let child = recorded_child_pid(tool.working_dir()).await;
        assert_process_gone(
            child,
            &format!("{descendant:?} descendant of a foreground command"),
        )
        .await;
        let alerts = receiver.take_job_alerts();
        match alerts.as_slice() {
            [ShellEvent::DescendantsKilled(killed)] => {
                assert_ne!(killed.task_id(), "");
                assert!(killed.call().is_none(), "the call carried no identity");
                // The name is read after the freeze, once the child exec'd.
                assert_eq!(
                    killed
                        .processes
                        .iter()
                        .map(|process| i64::from(process.pid))
                        .collect::<Vec<_>>(),
                    [i64::from(child)]
                );
                assert_eq!(killed.processes[0].name, descendant.process_name());
            }
            other => panic!("expected one descendants-killed alert, got {other:?}"),
        }
        assert!(
            receiver.take_job_alerts().is_empty(),
            "the same event must not alert twice"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn foreground_command_leaving_a_descendant_kills_it_and_alerts_once_plain() {
        foreground_command_leaving_a_descendant_kills_it_and_alerts_once(Descendant::Plain).await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn foreground_command_leaving_a_setsid_descendant_kills_it_and_alerts_once() {
        foreground_command_leaving_a_descendant_kills_it_and_alerts_once(Descendant::Setsid).await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn foreground_background_jobs_joined_by_wait_run_without_an_alert() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;
        let receiver = tool.background_receiver();

        let result = tool
            .call(
                terminal_args(
                    "run two jobs and join them",
                    "(sleep 1; echo one > one.txt) & (sleep 1; echo two > two.txt) & wait",
                ),
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");
        let command = result.command().expect("terminal returns a command result");
        assert!(
            matches!(command.outcome, CommandOutcome::Exited { code: 0 }),
            "{:?}",
            command.outcome
        );
        for name in ["one.txt", "two.txt"] {
            assert!(
                tool.working_dir().join(name).exists(),
                "{name} must be written before the command returns"
            );
        }
        assert!(
            receiver.take_job_alerts().is_empty(),
            "a joined job is no survivor"
        );
    }

    /// Starts the background `script`, waits until its shell has exited with the
    /// descendant still alive, and returns the task id and the descendant's pid.
    #[cfg(unix)]
    async fn background_job_whose_shell_exited(
        tool: &Shell<executor_core::tokio::TokioGlobal>,
        script: &str,
        timeout: u64,
    ) -> (String, i32) {
        let job = tool
            .run_background(ShellArgs {
                background: true,
                timeout,
                ..terminal_args("background job", script)
            })
            .await
            .expect("terminal should start the background job");
        let task_id = job.task_id().to_owned();
        let child = recorded_child_pid(tool.working_dir()).await;
        // Past the shell's exit and several of the tree's checks.
        tokio::time::sleep(Duration::from_millis(1200)).await;
        assert!(
            process_exists(child),
            "the descendant must outlive the shell"
        );
        (task_id, child)
    }

    #[cfg(unix)]
    async fn background_job_stays_running_until_its_descendant_exits(descendant: Descendant) {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;

        let (task_id, child) =
            background_job_whose_shell_exited(&tool, &descendant.script(&exec_dir(&tool), 3), 30)
                .await;
        let info = tool
            .job_registry()
            .get_by_task_id(&task_id)
            .await
            .expect("the job is registered");
        assert_eq!(
            info.status,
            JobStatus::Running,
            "the job runs while its descendant does"
        );
        assert!(
            tool.job_registry()
                .format_running_jobs()
                .await
                .contains(&task_id),
            "the running-jobs context lists the job"
        );

        assert_process_gone(child, "the descendant's own exit").await;
        let receiver = tool.background_receiver();
        let completed = wait_for_completed_task(&receiver, &task_id).await;
        assert!(completed.result.is_ok(), "{completed:?}");
        let info = tool
            .job_registry()
            .get_by_task_id(&task_id)
            .await
            .expect("the finished job stays in the registry");
        assert_eq!(
            info.status,
            JobStatus::Completed {
                exit_code: 0,
                delivery: crate::Delivery::Pending
            }
        );
        assert!(
            tool.job_registry().format_running_jobs().await.is_empty(),
            "a finished job is no longer listed"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_job_stays_running_until_its_plain_descendant_exits() {
        background_job_stays_running_until_its_descendant_exits(Descendant::Plain).await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_job_stays_running_until_its_setsid_descendant_exits() {
        background_job_stays_running_until_its_descendant_exits(Descendant::Setsid).await;
    }

    #[cfg(unix)]
    async fn killing_a_background_job_kills_its_descendant(descendant: Descendant) {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;

        let (task_id, child) =
            background_job_whose_shell_exited(&tool, &descendant.script(&exec_dir(&tool), 300), 30)
                .await;
        assert!(
            tool.job_registry().kill_by_task_id(&task_id).await,
            "a job with a live descendant is killable"
        );
        assert_process_gone(
            child,
            &format!("terminal_kill of a job with a {descendant:?} descendant"),
        )
        .await;
        let info = tool
            .job_registry()
            .get_by_task_id(&task_id)
            .await
            .expect("the killed job stays in the registry");
        assert!(
            matches!(info.status, JobStatus::Killed { .. }),
            "{:?}",
            info.status
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn killing_a_background_job_kills_its_plain_descendant() {
        killing_a_background_job_kills_its_descendant(Descendant::Plain).await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn killing_a_background_job_kills_its_setsid_descendant() {
        killing_a_background_job_kills_its_descendant(Descendant::Setsid).await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_timeout_kills_a_descendant_that_outlived_the_shell() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;

        // The timeout fires while the shell is long gone: the job is only its
        // descendant by then.
        let (task_id, child) = background_job_whose_shell_exited(
            &tool,
            &Descendant::Setsid.script(&exec_dir(&tool), 300),
            2,
        )
        .await;
        assert_process_gone(child, "the timeout of a job whose shell exited").await;
        let receiver = tool.background_receiver();
        let completed = wait_for_completed_task(&receiver, &task_id).await;
        let info = tool
            .job_registry()
            .get_by_task_id(&task_id)
            .await
            .expect("the finished job stays in the registry");
        assert!(
            matches!(
                info.status,
                JobStatus::Killed {
                    cause: crate::KillCause::Timeout(_),
                    delivery: crate::Delivery::Pending
                }
            ),
            "{completed:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagents_cleanup_kills_a_descendant_that_outlived_the_shell() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;

        let (_task_id, child) = background_job_whose_shell_exited(
            &tool,
            &Descendant::Plain.script(&exec_dir(&tool), 300),
            30,
        )
        .await;
        // What `kill_running_terminal_jobs` and the subagent run's cleanup call.
        assert_eq!(tool.job_registry().kill_running().await, 1);
        assert_process_gone(child, "the cleanup of a subagent's job").await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn terminal_rejects_timeout_above_cap_and_kills_at_timeout() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(2),
            Duration::from_secs(60),
        )
        .await;
        let background_receiver = tool.background_receiver();
        let args = |timeout: u64| ShellArgs {
            description: "sleep in the background".to_string(),
            script: "sleep 30".to_string(),
            interpreter: Interpreter::Shell,
            sandbox: SandboxMode::Sandboxed,
            location: None,
            resolution: MediaResolution::Auto,
            max_lines: Some(50),
            raw: false,
            timeout,
            background: true,
        };

        assert!(
            tool.call(args(3), ToolContext::new()).await.is_err(),
            "a timeout above the cap must be rejected"
        );

        let result = tool
            .call(args(1), ToolContext::new())
            .await
            .expect("a timeout within the cap should start the job");
        let rendered = render_terminal_result(&result);
        assert!(
            rendered.contains("Started in background as task '"),
            "{rendered}"
        );
        let task_id = started_task_id(&background_receiver);

        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        let info = tool
            .job_registry()
            .get_by_task_id(&task_id)
            .await
            .expect("the finished job stays in the registry");
        assert!(
            matches!(
                info.status,
                JobStatus::Killed {
                    cause: crate::KillCause::Timeout(_),
                    delivery: crate::Delivery::Pending
                }
            ),
            "the job should have been killed at its timeout: {completed:?}"
        );
    }

    #[test]
    fn the_timeout_notice_states_both_utc_times() {
        let kill = TimeoutKill {
            timeout: Duration::from_secs(90),
            started: "2026-11-04T12:00:00Z".parse().expect("valid timestamp"),
            killed: "2026-11-04T12:01:31Z".parse().expect("valid timestamp"),
        };
        assert_eq!(
            TestWording.command_killed(&crate::CommandKilled::TimedOut(&kill)),
            "Its 90s timeout bound passed; it started at 2026-11-04 12:00:00 UTC \
             and was killed at 2026-11-04 12:01:31 UTC."
        );
    }

    #[cfg(unix)]
    fn spinning_args(timeout: u64, background: bool) -> ShellArgs {
        ShellArgs {
            description: "spin past the timeout".to_string(),
            script: if background {
                "while :; do sleep 0.1; done"
            } else {
                // A foreground `sleep` is refused; a niced spin leaves the
                // other tests their CPU.
                "nice -n 19 sh -c 'while :; do :; done'"
            }
            .to_string(),
            interpreter: Interpreter::Shell,
            sandbox: SandboxMode::Sandboxed,
            location: None,
            resolution: MediaResolution::Auto,
            max_lines: Some(50),
            raw: false,
            timeout,
            background,
        }
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_background_job_killed_at_its_timeout_reports_its_utc_times() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;
        let background_receiver = tool.background_receiver();
        tool.call(spinning_args(1, true), ToolContext::new())
            .await
            .expect("the job starts");
        let task_id = started_task_id(&background_receiver);
        let completed = wait_for_completed_task(&background_receiver, &task_id).await;

        let kill = completed
            .timeout_kill
            .expect("a job killed at its timeout carries the times");
        assert_eq!(
            completed.info.status.kill_cause(),
            Some(crate::KillCause::Timeout(kill))
        );
        assert_eq!(kill.timeout, Duration::from_secs(1));
        assert!(
            kill.killed.duration_since(kill.started).as_secs_f64() >= 1.0,
            "{kill:?}"
        );
        assert!(
            kill.killed.duration_since(kill.started).as_secs() < 10,
            "{kill:?}"
        );
        assert!(
            TestWording
                .command_killed(&crate::CommandKilled::TimedOut(&kill))
                .starts_with("Its 1s timeout bound passed; it started at ")
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_job_killed_by_hand_is_not_a_timeout_kill() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;
        let background_receiver = tool.background_receiver();
        tool.call(spinning_args(30, true), ToolContext::new())
            .await
            .expect("the job starts");
        let task_id = started_task_id(&background_receiver);
        assert!(tool.job_registry().kill_by_task_id(&task_id).await);
        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        assert_eq!(
            completed.info.status.kill_cause(),
            Some(crate::KillCause::User)
        );
        assert_eq!(completed.timeout_kill, None);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_foreground_command_killed_at_its_timeout_reports_its_utc_times() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(20),
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
        .await;
        let before = jiff::Timestamp::now();
        let result = tool
            .call(spinning_args(1, false), ToolContext::new())
            .await
            .expect("a killed command still returns a result");
        let rendered = render_terminal_result(&result);
        let utc = |marker: &str| {
            let text = rendered
                .split(marker)
                .nth(1)
                .and_then(|rest| rest.split(" UTC").next())
                .unwrap_or_else(|| panic!("{marker:?} missing: {rendered}"));
            jiff::civil::DateTime::strptime("%Y-%m-%d %H:%M:%S", text)
                .expect("a UTC date and time")
                .to_zoned(jiff::tz::TimeZone::UTC)
                .expect("UTC has no gaps")
                .timestamp()
        };
        assert!(
            rendered.contains("Its 1s timeout bound passed; it started at "),
            "{rendered}"
        );
        let started = utc("it started at ");
        let killed = utc("was killed at ");
        assert!(started.as_second() >= before.as_second(), "{rendered}");
        assert!(killed.as_second() - started.as_second() >= 1, "{rendered}");
        assert!(killed.as_second() - started.as_second() < 10, "{rendered}");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn background_job_past_the_alert_threshold_yields_one_alert() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_millis(300),
        )
        .await;
        let background_receiver = tool.background_receiver();

        let _result = tool
            .run_background(ShellArgs {
                description: "sleep in the background".to_string(),
                script: "sleep 30".to_string(),
                interpreter: Interpreter::Shell,
                sandbox: SandboxMode::Sandboxed,
                location: None,
                resolution: MediaResolution::Auto,
                max_lines: Some(50),
                raw: false,
                timeout: 30,
                background: true,
            })
            .await
            .expect("terminal should start the job");
        let task_id = started_task_id(&background_receiver);

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let alerts = loop {
            let alerts = background_receiver.take_job_alerts();
            if !alerts.is_empty() {
                break alerts;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "long-running alert did not fire"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        assert_eq!(alerts.len(), 1);
        assert!(
            matches!(&alerts[0], ShellEvent::LongRunning(job) if job.task_id == task_id),
            "expected the long-running alert for the job: {alerts:?}"
        );
        assert!(
            background_receiver.take_job_alerts().is_empty(),
            "the same job must not alert twice"
        );

        assert!(
            tool.job_registry().kill_by_task_id(&task_id).await,
            "test cleanup should kill the job"
        );
    }

    /// A terminal tool with `priority` applied.
    #[cfg(unix)]
    async fn prioritized_terminal_tool(
        dir: &Path,
        priority: ProcessPriority,
    ) -> Shell<executor_core::tokio::TokioGlobal> {
        let sandbox = Sandbox::new(
            Backend::Host {
                heel: crate::test_heel::binary().await.expect("a test heel"),
            },
            executor_core::tokio::TokioGlobal,
            dir,
        )
        .with_priority(priority);
        Shell::new_exact_on(
            sandbox,
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
            crate::test_wording::job_tools(),
        )
        .await
        .expect("terminal tool should initialize")
    }

    /// A probe script made only of `sh` builtins: a strict sandbox executes
    /// nothing but the launched shell, the prepared tools and granted paths,
    /// so it reports its own `$$` into `pid` and waits for the host's `go`
    /// marker instead of running `nice`.
    #[cfg(unix)]
    fn pid_probe_args() -> ShellArgs {
        ShellArgs {
            description: "report the process pid and wait".to_string(),
            script: concat!(
                "printf '%s\\n' $$ > \"$PWD/pid\"; ",
                "while [ ! -f \"$PWD/go\" ]; do :; done"
            )
            .to_string(),
            interpreter: Interpreter::Shell,
            sandbox: SandboxMode::Sandboxed,
            location: None,
            resolution: MediaResolution::Auto,
            max_lines: Some(50),
            raw: false,
            timeout: 30,
            background: false,
        }
    }

    /// Reads the pid the sandboxed probe wrote, retrying while the sandbox
    /// starts.
    #[cfg(unix)]
    async fn probe_pid(dir: &Path) -> u32 {
        let path = dir.join("pid");
        for _ in 0..150 {
            if let Ok(text) = std::fs::read_to_string(&path)
                && let Ok(pid) = text.trim().parse()
            {
                return pid;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("the sandboxed probe did not write {}", path.display());
    }

    /// The `pid`'s nice level, read from the host while the probe is alive.
    #[cfg(unix)]
    fn probe_nice(pid: u32) -> i64 {
        // SAFETY: queries the kernel for the named process's priority.
        i64::from(unsafe { libc::getpriority(libc::PRIO_PROCESS as _, pid as _) })
    }

    /// Releases the sandboxed probe by creating the `go` marker it polls for.
    #[cfg(unix)]
    fn release_probe(dir: &Path) {
        std::fs::File::create(dir.join("go")).expect("create the probe marker");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn foreground_command_runs_with_configured_nice() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = prioritized_terminal_tool(dir.path(), test_nice_priority()).await;

        let call = tool.call(pid_probe_args(), ToolContext::new());
        let probe = async {
            let pid = probe_pid(dir.path()).await;
            assert_eq!(
                probe_nice(pid),
                expected_lowered_nice(10),
                "foreground command should run at the configured nice level"
            );
            release_probe(dir.path());
        };
        let (result, ()) = futures::join!(call, probe);
        let rendered = render_terminal_result(&result.expect("terminal call should succeed"));
        assert!(!rendered.contains("Exit code:"), "{rendered}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn persistent_session_runs_with_configured_nice() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = prioritized_terminal_tool(dir.path(), test_nice_priority()).await;
        let background_receiver = tool.background_receiver();
        let input_tool = input_tool(tool.jobs());

        // `exec` keeps `$$` — the pid the probe wrote — on the interactive
        // shell the session then drives.
        let result = tool
            .call(
                ShellArgs {
                    description: "start a persistent shell".to_string(),
                    script: "printf '%s\\n' $$ > \"$PWD/pid\"; exec sh".to_string(),
                    interpreter: Interpreter::Shell,
                    sandbox: SandboxMode::Sandboxed,
                    location: None,
                    resolution: MediaResolution::Auto,
                    max_lines: Some(50),
                    raw: false,
                    timeout: 30,
                    background: true,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");
        let rendered = render_terminal_result(&result);
        assert!(
            rendered.contains("Started in background as task '"),
            "{rendered}"
        );
        let task_id = started_task_id(&tool.background_receiver());

        let pid = probe_pid(dir.path()).await;
        assert_eq!(
            probe_nice(pid),
            expected_lowered_nice(10),
            "persistent session should run at the configured nice level"
        );

        input_tool
            .call(
                JobInputArgs {
                    task_id: task_id.clone(),
                    input: "exit".to_string(),
                    append_newline: true,
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal_input should succeed");

        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        let terminal_result = completed
            .result
            .expect("persistent shell should exit cleanly after input");
        assert_eq!(terminal_result.exit_code, 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn background_job_runs_with_configured_nice() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = prioritized_terminal_tool(dir.path(), test_nice_priority()).await;
        let background_receiver = tool.background_receiver();

        let result = tool
            .call(
                ShellArgs {
                    background: true,
                    ..pid_probe_args()
                },
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");
        let rendered = render_terminal_result(&result);
        let mut started = tool.background_receiver().take_started();
        assert_eq!(started.len(), 1, "expected one started-task event");
        let started = started.remove(0);
        let task_id = started.task_id;
        assert_eq!(
            rendered,
            format!(
                "Started in background as task '{task_id}'; output is being written to {} (absolute path: {}).\n",
                started.output_file.url, started.output_file.path
            )
        );
        assert_eq!(
            result.command().map(|command| &command.outcome),
            Some(&CommandOutcome::Running {
                task_id: task_id.clone(),
                output: started.output_file.clone(),
            })
        );

        let pid = probe_pid(dir.path()).await;
        assert_eq!(
            probe_nice(pid),
            expected_lowered_nice(10),
            "background job should run at the configured nice level"
        );
        release_probe(dir.path());

        let completed = wait_for_completed_task(&background_receiver, &task_id).await;
        let terminal_result = completed
            .result
            .expect("background job should complete successfully");
        assert_eq!(terminal_result.exit_code, 0);
    }

    #[test]
    fn wrap_container_script_applies_configured_priority() {
        let priority = ProcessPriority::new(
            CpuPriority::Nice(NiceLevel::new(5).expect("nice level 5 is in range")),
            IoPriority::Background,
        );
        let tools_dir = container_tools_dir();
        let executable = format!("--executable {}", shell_escape(tools_dir.as_str()));
        let setup = crate::spawn::container_ipc::exec_setup(
            &[String::from("websearch")],
            "host.docker.internal",
            9000,
        )
        .expect("ipc setup");
        let wrapped = wrap_container_script(
            "echo ok",
            Some(&setup),
            priority,
            &tools_dir,
            "/workspace",
            None,
        )
        .expect("wrap script");
        assert!(wrapped.contains(
            "\"$AITHER_HEEL_BIN\" run --isolation permissive --working-dir '/workspace' \
             --env-passthrough PATH --env-passthrough HOME"
        ));
        assert!(wrapped.contains(&executable));
        assert!(wrapped.contains(
            "--ipc-endpoint 'tcp://host.docker.internal:9000' --grant \"$AITHER_IPC_DIR=rx\" \
             --nice 5 --io-priority background -- sh -c 'echo ok'"
        ));
        assert!(wrapped.contains("HEEL_IPC_ENDPOINT"));

        let bare = wrap_container_script("echo ok", None, priority, &tools_dir, "/workspace", None)
            .expect("wrap script");
        assert!(bare.contains("AITHER_HEEL_BIN=\"$(command -v heel || true)\""));
        assert!(bare.contains(
            "\"$AITHER_HEEL_BIN\" run --isolation permissive --working-dir '/workspace' \
             --env-passthrough PATH --env-passthrough HOME"
        ));
        assert!(bare.contains(&executable));
        assert!(bare.contains("--nice 5 --io-priority background -- sh -c 'echo ok'"));

        let normal = wrap_container_script(
            "echo ok",
            None,
            ProcessPriority::default(),
            &tools_dir,
            "/workspace",
            None,
        )
        .expect("wrap script");
        assert_eq!(normal, "echo ok");
    }

    #[test]
    fn wrap_container_script_prioritizes_each_half_independently() {
        let io_only = wrap_container_script(
            "echo ok",
            None,
            ProcessPriority::new(CpuPriority::Normal, IoPriority::Utility),
            &container_tools_dir(),
            "/workspace",
            None,
        )
        .expect("wrap script");
        assert!(io_only.contains(" --io-priority utility -- sh -c 'echo ok'"));
        assert!(!io_only.contains("--nice"));

        let cpu_only = wrap_container_script(
            "echo ok",
            None,
            ProcessPriority::new(
                CpuPriority::Nice(NiceLevel::new(9).expect("nice level 9 is in range")),
                IoPriority::Normal,
            ),
            &container_tools_dir(),
            "/workspace",
            None,
        )
        .expect("wrap script");
        assert!(cpu_only.contains(" --nice 9 -- sh -c 'echo ok'"));
        assert!(!cpu_only.contains("--io-priority"));
    }

    /// An IPC command that answers with the marker it was invoked with, so a
    /// request that actually reached the live handler is unmistakable.
    struct ProbeEchoCommand;

    #[derive(Debug, Deserialize)]
    struct ProbeEchoArgs {
        marker: String,
    }

    impl heel::IpcCommand for ProbeEchoCommand {
        fn name(&self) -> Cow<'static, str> {
            "probe_echo".into()
        }

        type Args = ProbeEchoArgs;
        type Response = String;

        fn handle(
            &self,
            args: Self::Args,
            _context: heel::IpcContext,
        ) -> impl std::future::Future<Output = Result<Self::Response, heel::IpcError>> + Send
        {
            std::future::ready(Ok(format!("marker: {}", args.marker)))
        }
    }

    /// A container executor for runtimes whose container is never reached.
    struct UnreachableContainerExec;

    impl crate::ContainerExec for UnreachableContainerExec {
        fn exec(
            &self,
            _container_id: &str,
            _script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _lifetime: ExecLifetime,
        ) -> impl std::future::Future<Output = Result<crate::ContainerExecOutcome, String>> + Send
        {
            std::future::ready(Err("the container is never executed in".to_string()))
        }

        fn exec_streamed(
            &self,
            _container_id: &str,
            _script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _stdout: Sender<Vec<u8>>,
            _lifetime: ExecLifetime,
        ) -> impl std::future::Future<Output = Result<crate::ContainerExecOutcome, String>> + Send
        {
            std::future::ready(Err("the container is never executed in".to_string()))
        }
    }

    /// Host side of the container IPC transport on every platform: a client
    /// speaking the bridge's framing over loopback TCP reaches the embedder's
    /// live handler and gets its answer back.
    #[tokio::test(flavor = "multi_thread")]
    async fn container_ipc_bridge_round_trips_a_command_over_tcp() {
        use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
        let router = heel::IpcRouter::new().register(ProbeEchoCommand);
        let loopback = std::net::IpAddr::from([127, 0, 0, 1]);
        let runtime = crate::ContainerShellRuntime::new(
            "aither-test-ipc",
            crate::HostRoute::new("host.docker.internal", loopback).expect("loopback is routable"),
            crate::AnyContainerExec::new(Arc::new(UnreachableContainerExec)),
        );
        let executor = Arc::new(executor_core::AnyExecutor::new(
            executor_core::tokio::TokioGlobal,
        ));
        let bridge = crate::spawn::container_ipc::ContainerIpc::serve(&executor, router, &runtime)
            .expect("bridge should start on loopback");

        let mut stream = async_io::Async::<std::net::TcpStream>::connect(
            std::net::SocketAddr::new(loopback, bridge.port()),
        )
        .await
        .expect("client should connect to the bridge");

        let params = rmp_serde::to_vec_named(&serde_json::json!({ "marker": "pong" }))
            .expect("params should encode");
        let request = heel::ipc::IpcRequest::encode("probe_echo", Path::new("/workspace"), &params)
            .expect("request should encode");
        stream
            .write_all(&request)
            .await
            .expect("request frame should be written");

        let mut length_bytes = [0_u8; 4];
        stream
            .read_exact(&mut length_bytes)
            .await
            .expect("response length should be read");
        let mut response = vec![0_u8; u32::from_be_bytes(length_bytes) as usize];
        stream
            .read_exact(&mut response)
            .await
            .expect("response body should be read");

        let (success, payload) = response
            .split_first()
            .expect("response carries a success flag");
        assert_eq!(*success, 1, "bridge should report success");
        let rendered: String =
            rmp_serde::from_slice(payload).expect("response should decode as msgpack");
        assert_eq!(
            rendered, "marker: pong",
            "probe_echo should answer with the CLI-rendered payload, as `heel ipc` prints it"
        );

        drop(stream);
        bridge.stop().await;
    }

    // ---- subagent authority -------------------------------------------------

    type ConfiguredTestTool = Shell<executor_core::tokio::TokioGlobal>;

    /// Runs `script` in `tool`'s sandbox and returns its exit code.
    #[cfg(unix)]
    async fn exit_code_of(tool: &ConfiguredTestTool, script: &str) -> i32 {
        let result = tool
            .call(terminal_args("authority probe", script), ToolContext::new())
            .await
            .expect("terminal call should succeed");
        match result.command().map(|command| &command.outcome) {
            Some(CommandOutcome::Exited { code }) => *code,
            other => panic!("the probe should run to completion, got {other:?}"),
        }
    }

    #[cfg(unix)]
    async fn authority_terminal(dir: &Path) -> ConfiguredTestTool {
        configured_test_terminal(
            dir,
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await
    }

    /// An environment source that names the launch it is asked about.
    #[derive(Debug)]
    struct LaunchEnv;

    impl EnvSource for LaunchEnv {
        fn env(&self, launch: &crate::LaunchInfo) -> Vec<(String, String)> {
            vec![
                (
                    "PROBE_TASK".to_string(),
                    launch.task_id.clone().unwrap_or_default(),
                ),
                (
                    "PROBE_CALL".to_string(),
                    launch
                        .call
                        .as_ref()
                        .map(|call| call.call().to_string())
                        .unwrap_or_default(),
                ),
                ("PROBE_MODE".to_string(), format!("{:?}", launch.mode)),
            ]
        }
    }

    /// The environment source is told which job, call and mode the process
    /// it supplies is launched for.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn the_env_source_is_told_the_launch_it_supplies() {
        use aither_core::llm::{AgentIdentity, ToolCallId};
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let heel = crate::test_heel::binary().await.expect("a test heel");
        let sandbox = Sandbox::new(
            Backend::Host { heel },
            executor_core::tokio::TokioGlobal,
            dir.path(),
        )
        .with_env_source(LaunchEnv);
        let tool = Shell::new_exact_on(
            sandbox,
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
            crate::test_wording::job_tools(),
        )
        .await
        .expect("terminal tool should initialize");
        let identity = CallIdentity::new(ToolCallId::new("call_probe"), AgentIdentity::Main);

        let result = tool
            .call(
                terminal_args(
                    "print the launch",
                    "printenv PROBE_CALL; printenv PROBE_MODE",
                ),
                ToolContext::new().with_identity(identity),
            )
            .await
            .expect("terminal call should succeed");
        let text = render_terminal_result(&result);
        assert!(text.contains("call_probe\nSandboxed"), "{text}");
    }

    /// A command that reports one line of progress into the job that runs it.
    struct ProgressCommand {
        jobs: crate::Jobs,
        task_id: String,
    }

    #[derive(Debug, Deserialize)]
    struct NoArgs {}

    impl heel::IpcCommand for ProgressCommand {
        fn name(&self) -> Cow<'static, str> {
            "progress".into()
        }

        type Args = NoArgs;
        type Response = String;

        fn handle(
            &self,
            _args: Self::Args,
            _context: heel::IpcContext,
        ) -> impl std::future::Future<Output = Result<Self::Response, heel::IpcError>> + Send
        {
            let jobs = self.jobs.clone();
            let task_id = self.task_id.clone();
            async move {
                let appender = jobs
                    .output_appender(&task_id)
                    .await
                    .expect("the job is held before its process can call");
                assert_eq!(
                    appender.append(b"working\n").await,
                    crate::AppendOutcome::Appended
                );
                Ok("done".to_owned())
            }
        }
    }

    struct ProgressProvider(crate::Jobs);

    impl crate::IpcProvider for ProgressProvider {
        fn router(&self, launch: &crate::LaunchInfo) -> Option<heel::IpcRouter> {
            Some(heel::IpcRouter::new().register(ProgressCommand {
                jobs: self.0.clone(),
                task_id: launch.task_id.clone().expect("a shell names its launches"),
            }))
        }
    }

    /// A script that calls an embedder's command at its first instruction
    /// already finds its job: the appender is handed out and its line lands
    /// in the job's output, between the script's own.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_command_called_at_once_appends_to_the_job_that_ran_it() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let mut tool = fresh_terminal(dir.path()).await;
        tool.ipc = Some(crate::AnyIpcProvider::new(ProgressProvider(tool.jobs())));

        let result = tool
            .call(
                terminal_args("report progress", "progress >/dev/null; echo end"),
                ToolContext::new(),
            )
            .await
            .expect("terminal call should succeed");
        let text = render_terminal_result(&result);
        assert!(text.contains("working\nend"), "{text}");
    }

    /// A script the sandboxed command writes into its working directory and
    /// chmods executes — writable space also carries EXEC.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_script_written_into_the_working_dir_executes() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = authority_terminal(dir.path()).await;
        assert_eq!(
            exit_code_of(
                &tool,
                "echo '#!/bin/sh\necho ok' > run.sh && chmod +x run.sh && ./run.sh",
            )
            .await,
            0,
            "a script the command writes and chmods runs"
        );
    }

    /// Launches a subagent of `parent` the way its launcher does, with a
    /// launch history of its own.
    async fn launch(
        parent: &ConfiguredTestTool,
        grant: SubagentGrant,
    ) -> Result<ConfiguredTestTool, SubagentGrantError> {
        let resolved = parent
            .resolve_launch(&mut LaunchHistory::default(), grant)
            .await?;
        parent
            .launch_resolved(resolved, None, "worker".to_owned())
            .await
    }

    /// What a [`CloneSpaces`] was asked to prepare, as `(space, launcher,
    /// caller)`.
    #[cfg(unix)]
    type SeenSpace = (PathBuf, PathBuf, PathBuf);

    /// How a test's [`SubagentSpaces`] fills a space.
    #[cfg(unix)]
    #[derive(Clone, Copy)]
    enum Filling {
        /// Clones the source repository into `repo` and names it.
        Clone,
        /// Writes a file into the space, then fails.
        Fail,
        /// Names the launcher's working directory as the working copy.
        Outside,
    }

    #[cfg(unix)]
    #[derive(Debug, thiserror::Error)]
    #[error("the session has no clone to give")]
    struct NoClone;

    /// Prepares each space from `source`, the launcher's repository, and
    /// records what it was asked.
    #[cfg(unix)]
    struct CloneSpaces {
        source: PathBuf,
        filling: Filling,
        seen: async_channel::Sender<SeenSpace>,
    }

    #[cfg(unix)]
    impl SubagentSpaces for CloneSpaces {
        type Error = NoClone;

        async fn prepare(&self, run: RunSpace<'_>) -> Result<PreparedSpace, NoClone> {
            self.seen
                .send((
                    run.space.to_path_buf(),
                    run.launcher.to_path_buf(),
                    run.caller.to_path_buf(),
                ))
                .await
                .expect("the test records each space");
            let clone = run.space.join("repo");
            match self.filling {
                Filling::Clone => {
                    let (source, target) = (self.source.clone(), clone.clone());
                    blocking::unblock(move || {
                        crate::test_git::git(
                            &source,
                            &["clone", "-q", "--local", ".", target.to_str().unwrap()],
                        )
                    })
                    .await;
                    Ok(PreparedSpace::new().with_working_copy(clone))
                }
                Filling::Fail => {
                    async_fs::write(run.space.join("half-done"), "x")
                        .await
                        .unwrap();
                    Err(NoClone)
                }
                Filling::Outside => Ok(PreparedSpace::new().with_working_copy(run.launcher)),
            }
        }
    }

    /// A terminal in `dir` whose subagents' spaces `filling` prepares from
    /// the repository `dir/repo`, with the receiver of what it is asked.
    #[cfg(unix)]
    async fn spaced_terminal(
        dir: &Path,
        filling: Filling,
    ) -> (ConfiguredTestTool, async_channel::Receiver<SeenSpace>) {
        let source = dir.join("repo");
        std::fs::create_dir_all(&source).unwrap();
        crate::test_git::git(&source, &["init", "-q", "--initial-branch=main"]);
        crate::test_git::git(&source, &["commit", "-q", "--allow-empty", "-m", "base"]);
        let (seen, seen_rx) = async_channel::unbounded();
        let tool = configured_test_terminal(
            dir,
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await
        .with_subagent_spaces(CloneSpaces {
            source,
            filling,
            seen,
        });
        (tool, seen_rx)
    }

    /// A subagent works in the clone its launcher's spaces prepared: it
    /// commits there, cannot push into the launcher's repository, and its
    /// work reaches the launcher only when the launcher fetches it. The space
    /// outlives the subagent's terminal for that, and the launcher can
    /// remove it; a subagent's own launches are prepared by the same spaces.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagent_works_in_the_clone_its_launchers_spaces_prepared() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let root = dunce::canonicalize(dir.path()).unwrap();
        let (parent, seen) = spaced_terminal(dir.path(), Filling::Clone).await;
        let caller = dir.path().join("repo");
        let child = parent
            .subagent(SubagentGrant::new(), Some(&caller), "worker")
            .await
            .expect("the space is prepared");

        let (space, launcher, asked_from) = seen.try_recv().expect("the spaces were asked");
        assert_eq!(&space, child.working_dir(), "the run works in the space");
        assert!(space.starts_with(root.join(SUBAGENT_SPACES_DIR)));
        assert_eq!(launcher, root, "the launcher is the main agent's terminal");
        assert_eq!(asked_from, caller, "the caller's directory is passed on");
        let clone = space.join("repo");
        assert_eq!(
            child.prepared_space().unwrap().working_copies(),
            std::slice::from_ref(&clone)
        );
        assert!(parent.prepared_space().is_none());

        let git = "git -c user.name=t -c user.email=t@t -c commit.gpgSign=false";
        assert_eq!(
            exit_code_of(
                &child,
                &format!(
                    "cd '{}' && echo work > file && {git} add file && {git} commit -q -m work",
                    clone.display()
                ),
            )
            .await,
            0,
            "the subagent commits in its clone"
        );
        assert_ne!(
            exit_code_of(
                &child,
                &format!(
                    "cd '{}' && {git} push -q origin HEAD:refs/heads/pushed",
                    clone.display()
                ),
            )
            .await,
            0,
            "the launcher's repository is read-only to the subagent"
        );
        assert!(!crate::test_git::succeeds(
            &caller,
            &["rev-parse", "--verify", "-q", "refs/heads/pushed"]
        ));

        let nested = child
            .subagent(SubagentGrant::new(), None, "worker")
            .await
            .expect("a nested space is prepared");
        let (nested_space, nested_launcher, nested_caller) = seen.try_recv().unwrap();
        assert_eq!(&nested_space, nested.working_dir());
        assert_eq!(nested_launcher, space, "a subagent launches from its space");
        assert_eq!(nested_caller, space, "the caller defaults to the launcher");
        drop((nested, child));

        assert!(clone.join(".git").is_dir(), "the space outlives the run");
        assert_eq!(
            exit_code_of(
                &parent,
                &format!(
                    "cd repo && {git} fetch -q '{}' HEAD && {git} merge -q --ff-only FETCH_HEAD",
                    clone.display()
                ),
            )
            .await,
            0,
            "the launcher merges the run's work"
        );
        assert_eq!(
            crate::test_git::git(&caller, &["log", "-1", "--format=%s"]),
            "work"
        );
        assert_eq!(
            exit_code_of(&parent, &format!("rm -rf '{}'", space.display())).await,
            0,
            "the launcher owns the space"
        );
        assert!(!space.exists());
    }

    /// A space the launcher's spaces cannot prepare fails the launch with
    /// their error, and is removed with what they left in it.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_space_that_cannot_be_prepared_fails_the_launch_and_is_removed() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (parent, seen) = spaced_terminal(dir.path(), Filling::Fail).await;
        let error = launch(&parent, SubagentGrant::new())
            .await
            .expect_err("the launch fails");
        let (asked, _, _) = seen.try_recv().unwrap();
        let SubagentGrantError::Preparation { space, source } = &error else {
            panic!("the launch fails with the preparation error: {error}");
        };
        assert_eq!(space, &asked);
        assert!(source.downcast_ref::<NoClone>().is_some(), "{error}");
        assert!(
            error
                .to_string()
                .contains("the session has no clone to give")
        );
        assert!(!space.exists(), "the failed space is removed");
    }

    /// A working copy named outside the space fails the launch, and the space
    /// is removed.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_working_copy_outside_the_space_fails_the_launch() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (parent, seen) = spaced_terminal(dir.path(), Filling::Outside).await;
        let error = launch(&parent, SubagentGrant::new())
            .await
            .expect_err("the launch fails");
        let (asked, launcher, _) = seen.try_recv().unwrap();
        let SubagentGrantError::PreparedOutsideSpace { path, space } = &error else {
            panic!("the launch fails on the working copy: {error}");
        };
        assert_eq!((path, space), (&launcher, &asked));
        assert!(!asked.exists(), "the refused space is removed");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn subagent_cannot_write_the_session_root_but_writes_its_space_and_granted_worktrees() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let parent = authority_terminal(dir.path()).await;
        let granted = dir.path().join("granted");
        let ungranted = dir.path().join("ungranted");
        std::fs::create_dir_all(&granted).unwrap();
        std::fs::create_dir_all(&ungranted).unwrap();
        let child = launch(&parent, SubagentGrant::new().with_worktree(&granted))
            .await
            .expect("a writable directory can be granted");

        let touch = |path: &Path| format!("touch '{}'", path.display());
        let root_file = dir.path().join("root.txt");
        assert_eq!(
            exit_code_of(&parent, &touch(&root_file)).await,
            0,
            "the main agent writes its own space"
        );
        std::fs::remove_file(&root_file).unwrap();

        assert_ne!(
            exit_code_of(&child, &touch(&root_file)).await,
            0,
            "the session root is read-only to a subagent"
        );
        assert!(!root_file.exists());
        assert_ne!(
            exit_code_of(&child, &touch(&ungranted.join("x"))).await,
            0,
            "what the parent can write but did not grant is read-only"
        );
        assert_eq!(
            exit_code_of(&child, &touch(&child.working_dir().join("own.txt"))).await,
            0,
            "a subagent writes its own space"
        );
        assert!(
            child
                .working_dir()
                .starts_with(dunce::canonicalize(dir.path()).unwrap())
        );
        assert_ne!(child.working_dir(), parent.working_dir());
        assert_eq!(
            exit_code_of(&child, &touch(&granted.join("g.txt"))).await,
            0,
            "a subagent writes the worktrees it was granted"
        );
    }

    /// Runs `script` in `tool` and returns its trimmed standard output.
    #[cfg(unix)]
    async fn stdout_of(tool: &ConfiguredTestTool, script: &str) -> String {
        let result = tool
            .call(terminal_args("carry probe", script), ToolContext::new())
            .await
            .expect("terminal call should succeed");
        render_terminal_result(&result).trim().to_string()
    }

    /// The directory `pwd` prints for the next command.
    #[cfg(unix)]
    async fn pwd_of(tool: &ConfiguredTestTool) -> PathBuf {
        PathBuf::from(stdout_of(tool, "pwd").await)
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_successful_cd_carries_to_the_next_foreground_command() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = authority_terminal(dir.path()).await;
        let root = dunce::canonicalize(dir.path()).unwrap();
        assert_eq!(pwd_of(&tool).await, root);

        assert_eq!(exit_code_of(&tool, "mkdir sub && cd sub").await, 0);
        assert_eq!(pwd_of(&tool).await, root.join("sub"));
        // Relative to the carried directory, and a later `cd` carries on.
        assert_eq!(exit_code_of(&tool, "mkdir -p a/b && cd a/b").await, 0);
        assert_eq!(pwd_of(&tool).await, root.join("sub/a/b"));
        // A direct command starts in the carried directory too.
        assert_eq!(
            PathBuf::from(stdout_of(&tool, "/bin/pwd").await),
            root.join("sub/a/b")
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_failure_an_outside_cd_and_a_background_cd_leave_the_directory() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = authority_terminal(dir.path()).await;
        let root = dunce::canonicalize(dir.path()).unwrap();
        stdout_of(&tool, "mkdir sub other && cd sub").await;
        let sub = root.join("sub");
        assert_eq!(pwd_of(&tool).await, sub);

        assert_eq!(
            exit_code_of(&tool, "cd ../other; exit 3").await,
            3,
            "the command fails"
        );
        assert_eq!(pwd_of(&tool).await, sub, "a failed command carries nothing");

        // Readable to the sandbox, yet in none of the terminal's grants.
        assert_eq!(
            exit_code_of(&tool, "cd /usr").await,
            0,
            "the cd itself succeeds"
        );
        assert_eq!(
            pwd_of(&tool).await,
            sub,
            "a directory outside the readable paths carries nothing"
        );

        start_background_job(&tool, "cd ../other && pwd").await;
        for _ in 0..500 {
            if tool.running_tasks.load(Ordering::Acquire) == 0 {
                break;
            }
            async_io::Timer::after(Duration::from_millis(20)).await;
        }
        assert_eq!(tool.running_tasks.load(Ordering::Acquire), 0);
        assert_eq!(pwd_of(&tool).await, sub, "a background cd carries nothing");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn an_exported_variable_does_not_reach_the_next_command() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = authority_terminal(dir.path()).await;
        assert_eq!(
            stdout_of(
                &tool,
                "export CARRY_PROBE=set; mkdir sub; cd sub; echo $CARRY_PROBE"
            )
            .await,
            "set"
        );
        assert_eq!(
            stdout_of(&tool, "printenv CARRY_PROBE; echo rc=$?").await,
            "rc=1"
        );
        assert_eq!(
            pwd_of(&tool).await,
            dunce::canonicalize(dir.path()).unwrap().join("sub")
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_symlinked_working_directory_is_resolved_for_the_report() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let real = dunce::canonicalize(dir.path()).unwrap().join("real");
        std::fs::create_dir(&real).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let tool = authority_terminal(&link).await;
        assert_eq!(tool.working_dir(), &real);
        assert_eq!(exit_code_of(&tool, "mkdir sub && cd sub").await, 0);
        assert_eq!(pwd_of(&tool).await, real.join("sub"));
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagents_cd_does_not_carry_over() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let parent = authority_terminal(dir.path()).await;
        let child = launch(&parent, SubagentGrant::new())
            .await
            .expect("a subagent without grants launches");
        let space = child.working_dir().clone();

        assert_eq!(exit_code_of(&child, "mkdir sub && cd sub").await, 0);
        assert_eq!(
            pwd_of(&child).await,
            space,
            "the subagent starts in its space"
        );
        // Nor does it disturb the main agent's directory.
        assert_eq!(
            pwd_of(&parent).await,
            dunce::canonicalize(dir.path()).unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn container_directory_report_is_written_by_the_rendered_script() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let report = dir.path().join("it's report");
        let script = carry_cwd_script("mkdir sub && cd sub", &report.to_string_lossy())
            .expect("the trailer should render");
        let status = std::process::Command::new("sh")
            .args(["-c", &script])
            .current_dir(dir.path())
            .status()
            .expect("sh should run");
        assert!(status.success());
        let written = std::fs::read_to_string(&report).expect("the trailer writes the report");
        assert_eq!(
            PathBuf::from(written.trim_end()),
            dunce::canonicalize(dir.path().join("sub")).unwrap()
        );
        // The script's own exit status survives the trailer, whether the
        // script ends through `exit` or its last command fails.
        for (script, code) in [("exit 7", 7), ("cd sub; (exit 5)", 5)] {
            let status = std::process::Command::new("sh")
                .args([
                    "-c",
                    &carry_cwd_script(script, &report.to_string_lossy()).unwrap(),
                ])
                .current_dir(dir.path())
                .status()
                .expect("sh should run");
            assert_eq!(status.code(), Some(code), "{script}");
        }
        // The script's own syntax cannot swallow the trailer: an unterminated
        // here-document holding a quote still ends before the report.
        std::fs::remove_file(&report).expect("the report exists");
        let output = std::process::Command::new("sh")
            .args([
                "-c",
                &carry_cwd_script("cd sub; cat <<EOF\nit's", &report.to_string_lossy()).unwrap(),
            ])
            .current_dir(dir.path())
            .output()
            .expect("sh should run");
        assert!(output.status.success());
        // bash ends the unterminated body with a newline, dash does not.
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim_end(), "it's");
        let written = std::fs::read_to_string(&report).expect("the trailer writes the report");
        assert_eq!(
            PathBuf::from(written.trim_end()),
            dunce::canonicalize(dir.path().join("sub")).unwrap()
        );
    }

    /// Reports how each script ended, to a channel the test holds.
    struct RecordCarries(async_channel::Sender<(ScriptDir, Carry)>);

    impl crate::ScriptPolicy for RecordCarries {
        type State = ();

        fn check(
            &self,
            _script: &crate::ScriptCheck<'_>,
            (): &(),
        ) -> impl Future<Output = ScriptDecision> + Send {
            std::future::ready(ScriptDecision::Pass)
        }

        async fn ended(&self, end: &crate::ScriptEnd<'_>, (): &()) {
            self.0
                .send((end.end.clone(), end.carry))
                .await
                .expect("the test holds the receiver");
        }
    }

    /// The main agent's terminal of [`authority_terminal`], with the end of
    /// every script reported on the returned channel.
    #[cfg(unix)]
    async fn carry_terminal(
        dir: &Path,
    ) -> (
        ConfiguredTestTool,
        async_channel::Receiver<(ScriptDir, Carry)>,
    ) {
        let (ends_tx, ends) = async_channel::unbounded();
        let tool = configured_test_terminal(
            dir,
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await
        .with_test_policies(vec![crate::AnyScriptPolicy::new(RecordCarries(ends_tx))]);
        (tool, ends)
    }

    /// The ends reported since the last call.
    #[cfg(unix)]
    fn carries(ends: &async_channel::Receiver<(ScriptDir, Carry)>) -> Vec<(ScriptDir, Carry)> {
        std::iter::from_fn(|| ends.try_recv().ok()).collect()
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_carried_directory_is_moved_or_unchanged() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (tool, ends) = carry_terminal(dir.path()).await;
        let sub = ScriptDir::Host(dunce::canonicalize(dir.path()).unwrap().join("sub"));

        assert_eq!(exit_code_of(&tool, "mkdir sub && cd sub").await, 0);
        assert_eq!(exit_code_of(&tool, "true").await, 0);
        assert_eq!(exit_code_of(&tool, "pwd").await, 0);
        // A failure that stayed in its directory, and a direct command.
        assert_ne!(exit_code_of(&tool, "false").await, 0);
        assert_ne!(exit_code_of(&tool, "ls /nonexistent-directory").await, 0);
        assert_eq!(
            carries(&ends),
            [
                (sub.clone(), Carry::Moved),
                (sub.clone(), Carry::Unchanged),
                (sub.clone(), Carry::Unchanged),
                (sub.clone(), Carry::Unchanged),
                (sub, Carry::Unchanged),
            ]
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_failed_command_resets_to_the_directory_commands_run_in() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (tool, ends) = carry_terminal(dir.path()).await;
        let root = dunce::canonicalize(dir.path()).unwrap();

        assert_ne!(exit_code_of(&tool, "mkdir sub && cd sub && false").await, 0);
        assert_eq!(
            carries(&ends),
            [(
                ScriptDir::Host(root.clone()),
                Carry::Reset(DirectoryResetReason::Failed)
            )]
        );
        assert_eq!(pwd_of(&tool).await, root);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn exit_without_a_report_resets() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (tool, ends) = carry_terminal(dir.path()).await;
        let root = dunce::canonicalize(dir.path()).unwrap();
        let sub = ScriptDir::Host(root.join("sub"));

        assert_eq!(exit_code_of(&tool, "mkdir sub && cd sub").await, 0);
        assert_eq!(exit_code_of(&tool, "cd ..; exit 0").await, 0);
        assert_eq!(
            carries(&ends),
            [
                (sub.clone(), Carry::Moved),
                (sub, Carry::Reset(DirectoryResetReason::Unreported)),
            ]
        );
        assert_eq!(pwd_of(&tool).await, root.join("sub"));
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_directory_outside_the_readable_paths_resets() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (tool, ends) = carry_terminal(dir.path()).await;
        let root = dunce::canonicalize(dir.path()).unwrap();

        // Readable to the sandbox, yet in none of the terminal's grants.
        assert_eq!(exit_code_of(&tool, "cd /usr").await, 0);
        assert_eq!(
            carries(&ends),
            [(
                ScriptDir::Host(root.clone()),
                Carry::Reset(DirectoryResetReason::OutsideReadable)
            )]
        );
        assert_eq!(pwd_of(&tool).await, root);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_removed_carried_directory_resets() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (tool, ends) = carry_terminal(dir.path()).await;
        let root = dunce::canonicalize(dir.path()).unwrap();

        assert_eq!(exit_code_of(&tool, "mkdir gone && cd gone").await, 0);
        assert_eq!(
            carries(&ends),
            [(ScriptDir::Host(root.join("gone")), Carry::Moved)]
        );
        std::fs::remove_dir(root.join("gone")).unwrap();
        assert_eq!(pwd_of(&tool).await, root);
        assert_eq!(
            carries(&ends),
            [(
                ScriptDir::Host(root.clone()),
                Carry::Reset(DirectoryResetReason::Removed)
            )]
        );
        assert_eq!(pwd_of(&tool).await, root);
        assert_eq!(carries(&ends), [(ScriptDir::Host(root), Carry::Unchanged)]);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_container_directory_report_maps_through_the_workspace_mount() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = authority_terminal(dir.path()).await;
        let mount = MountRoot::new(
            tool.working_dir(),
            ContainerPath::new("/workspace").expect("absolute"),
        );
        let report = tool.working_dir().join(".cwd-container");
        let carry = CwdCarry {
            start: None,
            report: Some(report.clone()),
            container: Some(ContainerCarry {
                mount: mount.clone(),
                report: Some(
                    mount
                        .container_path(&report)
                        .expect("report is in the mount"),
                ),
                start: ContainerPath::new("/workspace").unwrap(),
            }),
            shell: true,
            removed: false,
        };
        assert_eq!(
            carry
                .container
                .as_ref()
                .unwrap()
                .report
                .as_ref()
                .unwrap()
                .as_str(),
            format!(
                "/workspace/{}",
                report.file_name().unwrap().to_string_lossy()
            )
        );
        let carried = tool.carried_dir.as_ref().unwrap();

        // A directory inside the mount, spelled as the container sees it.
        std::fs::write(&report, "/workspace/sub/dir\n").unwrap();
        let settled = tool.record_carried_dir(&carry, true).await;
        assert_eq!(carried.get(), Some(tool.working_dir().join("sub/dir")));
        assert!(!report.exists(), "the report is consumed");
        assert_eq!(settled.carry, Carry::Moved);
        assert_eq!(settled.carried, carried.get());

        // Outside the mount, after a failure and an empty report: unchanged,
        // and each says why the next command starts where the last did.
        std::fs::write(&report, "/etc\n").unwrap();
        let settled = tool.record_carried_dir(&carry, true).await;
        assert_eq!(carried.get(), Some(tool.working_dir().join("sub/dir")));
        assert_eq!(
            settled.carry,
            Carry::Reset(DirectoryResetReason::OutsideWorkspace)
        );
        std::fs::write(&report, "/workspace/other\n").unwrap();
        let settled = tool.record_carried_dir(&carry, false).await;
        assert_eq!(carried.get(), Some(tool.working_dir().join("sub/dir")));
        assert_eq!(settled.carry, Carry::Reset(DirectoryResetReason::Failed));
        let settled = tool.record_carried_dir(&carry, true).await;
        assert_eq!(
            settled.carry,
            Carry::Reset(DirectoryResetReason::Unreported)
        );
        // A failure that stayed where it started changes nothing.
        std::fs::write(&report, "/workspace\n").unwrap();
        let settled = tool.record_carried_dir(&carry, false).await;
        assert_eq!(settled.carry, Carry::Unchanged);

        // The mapping back to the container for the next start.
        assert_eq!(
            mount
                .container_path(&tool.working_dir().join("sub/dir"))
                .unwrap()
                .as_str(),
            "/workspace/sub/dir"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_grant_outside_the_launchers_authority_is_refused() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let outside = tempfile::tempdir().expect("temp dir should be created");
        let parent = authority_terminal(dir.path()).await;
        let error = launch(&parent, SubagentGrant::new().with_worktree(outside.path()))
            .await
            .expect_err("a launcher cannot hand on what it cannot write");
        assert!(
            matches!(error, SubagentGrantError::OutsideAuthority(_)),
            "{error}"
        );
    }

    #[cfg(unix)]
    fn commit_script(worktree: &Path, subject: &str) -> String {
        format!(
            "git -c user.name=agent -c user.email=agent@example.com -c commit.gpgSign=false -C '{}' commit --allow-empty -m '{subject}'",
            worktree.display()
        )
    }

    /// A repository with a worktree one can commit in, and the paths the
    /// commit grant is about, all canonical.
    #[cfg(unix)]
    struct CommitSetup {
        dir: tempfile::TempDir,
        worktree: PathBuf,
        /// The main repository's git directory.
        common: PathBuf,
        /// The worktree's entry in it.
        entry: PathBuf,
        /// The subagent's repository.
        subagent: PathBuf,
        parent: ConfiguredTestTool,
    }

    #[cfg(unix)]
    impl CommitSetup {
        async fn new(namespace: &str) -> Self {
            let dir = tempfile::tempdir().expect("temp dir should be created");
            let worktree =
                crate::subagent::tests::repository_with_linked_worktree(dir.path(), namespace);
            let parent = authority_terminal(dir.path()).await;
            let common = dunce::canonicalize(dir.path().join("repository/.git")).unwrap();
            let entry = common.join("worktrees").join(namespace);
            Self {
                worktree: dunce::canonicalize(worktree).unwrap(),
                subagent: entry.join("subagent"),
                common,
                entry,
                parent,
                dir,
            }
        }

        async fn committing_child(&self, launcher: &ConfiguredTestTool) -> ConfiguredTestTool {
            launch(
                launcher,
                SubagentGrant::new().with_committing_worktree(&self.worktree),
            )
            .await
            .expect("the worktree can be granted with commit")
        }

        fn main_has(&self, id: &str) -> bool {
            crate::test_git::command()
                .args(["cat-file", "-e", id])
                .current_dir(&self.common)
                .output()
                .unwrap()
                .status
                .success()
        }

        fn main(&self, args: &[&str]) -> String {
            crate::subagent::tests::git_in(&self.common, args)
        }
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_sandboxed_commit_lands_in_the_subagents_repository_and_the_main_one_is_read_only() {
        let setup = CommitSetup::new("one").await;
        let child = setup.committing_child(&setup.parent).await;
        let ledger = child.commit_ledger().expect("a commit grant has a ledger");
        let mark = ledger.mark().await.expect("the namespace can be read");
        let before = setup.main(&["rev-parse", "refs/heads/subagent/one/work"]);

        assert_eq!(
            exit_code_of(&child, &commit_script(&setup.worktree, "authorized")).await,
            0,
            "a commit grant lets the subagent commit"
        );
        assert_eq!(
            setup.main(&["rev-parse", "refs/heads/subagent/one/work"]),
            before,
            "the commit is not in the main repository yet"
        );

        let commits = ledger.since(&mark).await.expect("commits can be listed");
        assert_eq!(commits.len(), 1, "{commits:?}");
        assert_eq!(commits[0].subject, "authorized");
        assert_eq!(commits[0].worktree, setup.worktree);
        assert!(setup.main_has(&commits[0].commit));
        assert_eq!(
            setup.main(&["rev-parse", "refs/heads/subagent/one/work"]),
            commits[0].commit
        );
        setup.main(&["fsck", "--strict", "--no-dangling"]);
    }

    /// Nothing of the main repository's git directory is writable to a
    /// committing subagent: each attempt on the main repository fails, and the
    /// same command shape on the subagent's own repository, its positive
    /// control, succeeds.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    #[allow(clippy::too_many_lines)]
    async fn a_committing_subagent_can_change_nothing_the_main_repository_reads() {
        let setup = CommitSetup::new("two").await;
        let child = setup.committing_child(&setup.parent).await;

        // A fan-out directory known to be absent, and a loose object known to
        // exist, in the main object store.
        let absent = setup.common.join("objects/ff");
        let _ = std::fs::remove_dir(&absent);
        assert!(!absent.exists());
        let loose_dir = std::fs::read_dir(setup.common.join("objects"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.len() == 2 && name != "ff")
            })
            .expect("the initial commit left loose objects");
        let loose = std::fs::read_dir(&loose_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let loose_name = loose.file_name().unwrap().to_str().unwrap().to_string();
        let loose_dir_name = loose_dir.file_name().unwrap().to_str().unwrap().to_string();
        let loose_bytes = std::fs::read(&loose).unwrap();

        // The subagent's own repository, for the controls.
        let own = &setup.subagent;
        std::fs::create_dir_all(own.join("objects").join(&loose_dir_name)).unwrap();
        std::fs::write(
            own.join("objects").join(&loose_dir_name).join(&loose_name),
            b"own",
        )
        .unwrap();

        let main_objects = setup.common.join("objects");
        let own_objects = own.join("objects");
        let attempts: Vec<(&str, String, String)> = vec![
            (
                "create a fan-out directory",
                format!("mkdir '{}'", main_objects.join("ff").display()),
                format!("mkdir '{}'", own_objects.join("ff").display()),
            ),
            (
                "create an object in an existing fan-out directory",
                format!(
                    "touch '{}'",
                    main_objects.join(&loose_dir_name).join("new").display()
                ),
                format!(
                    "touch '{}'",
                    own_objects.join(&loose_dir_name).join("new").display()
                ),
            ),
            (
                "overwrite an existing loose object",
                format!("echo evil > '{}'", loose.display()),
                format!(
                    "echo evil > '{}'",
                    own_objects
                        .join(&loose_dir_name)
                        .join(&loose_name)
                        .display()
                ),
            ),
            (
                "replace an existing loose object by renaming over it",
                format!(
                    "echo evil > '{}/stage' && mv -f '{}/stage' '{}'",
                    own_objects.display(),
                    own_objects.display(),
                    loose.display()
                ),
                format!(
                    "echo evil > '{}/stage' && mv -f '{}/stage' '{}'",
                    own_objects.display(),
                    own_objects.display(),
                    own_objects
                        .join(&loose_dir_name)
                        .join(&loose_name)
                        .display()
                ),
            ),
            (
                "remove an existing loose object",
                format!("rm -f '{}'", loose.display()),
                format!(
                    "rm -f '{}'",
                    own_objects
                        .join(&loose_dir_name)
                        .join(&loose_name)
                        .display()
                ),
            ),
            (
                "create a pack",
                format!(
                    "touch '{}'",
                    main_objects.join("pack/pack-x.pack").display()
                ),
                format!("touch '{}'", own_objects.join("pack/pack-x.pack").display()),
            ),
            (
                "write an alternates file",
                format!(
                    "echo /tmp > '{}'",
                    main_objects.join("info/alternates").display()
                ),
                format!(
                    "echo /tmp > '{}'",
                    own_objects.join("info/scratch").display()
                ),
            ),
            (
                "move the worktree's branch",
                format!(
                    "echo {} > '{}'",
                    "a".repeat(40),
                    setup.common.join("refs/heads/subagent/two/work").display()
                ),
                format!(
                    "echo {} > '{}'",
                    "a".repeat(40),
                    own.join("refs/heads/subagent/two/scratch").display()
                ),
            ),
            (
                "rewrite the entry's HEAD",
                format!("echo evil >> '{}'", setup.entry.join("HEAD").display()),
                format!("echo evil >> '{}'", own.join("scratch-head").display()),
            ),
            (
                "rewrite the entry's index",
                format!("echo evil >> '{}'", setup.entry.join("index").display()),
                format!("echo evil >> '{}'", own.join("scratch-index").display()),
            ),
            (
                "rewrite the repository's configuration",
                format!("echo evil >> '{}'", setup.common.join("config").display()),
                format!("echo evil >> '{}'", own.join("scratch-config").display()),
            ),
        ];
        for (what, denied, control) in attempts {
            assert_ne!(
                exit_code_of(&child, &denied).await,
                0,
                "must not {what} in the main repository"
            );
            assert_eq!(
                exit_code_of(&child, &control).await,
                0,
                "the same command succeeds in the subagent's own repository: {what}"
            );
        }
        assert!(!absent.exists(), "no fan-out directory appeared");
        assert_eq!(
            std::fs::read(&loose).unwrap(),
            loose_bytes,
            "the object is as it was"
        );
        assert!(
            !std::fs::read_to_string(setup.common.join("config"))
                .unwrap()
                .contains("evil")
        );
        setup.main(&["fsck", "--strict", "--no-dangling"]);
    }

    /// Git in a repository the sandbox can read keeps working next to the
    /// redirected worktree: nothing is set process-wide.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn git_in_another_repository_works_inside_a_committing_subagents_sandbox() {
        let setup = CommitSetup::new("three").await;
        let other = setup.dir.path().join("other");
        std::fs::create_dir_all(&other).unwrap();
        crate::subagent::tests::git_in(&other, &["init", "--initial-branch=main"]);
        crate::subagent::tests::git_in(&other, &["commit", "--allow-empty", "-m", "elsewhere"]);
        let child = setup.committing_child(&setup.parent).await;

        assert!(
            stdout_of(
                &child,
                &format!("git -C '{}' log --oneline", other.display())
            )
            .await
            .contains("elsewhere")
        );
        let clone = child.working_dir().join("clone");
        assert_eq!(
            exit_code_of(
                &child,
                &format!("git clone '{}' '{}'", other.display(), clone.display())
            )
            .await,
            0
        );
        assert!(clone.join(".git").exists());
    }

    /// A subagent a committing subagent launches with the commit grant it
    /// holds works in the same repository: it sees its parent's commits and
    /// the parent's ledger reports both.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_committing_child_shares_its_parents_repository() {
        let setup = CommitSetup::new("four").await;
        let parent_subagent = setup.committing_child(&setup.parent).await;
        let parent_ledger = parent_subagent.commit_ledger().unwrap();
        let parent_mark = parent_ledger.mark().await.unwrap();
        assert_eq!(
            exit_code_of(
                &parent_subagent,
                &commit_script(&setup.worktree, "parent work")
            )
            .await,
            0
        );

        let child = setup.committing_child(&parent_subagent).await;
        let child_ledger = child.commit_ledger().unwrap();
        let child_mark = child_ledger.mark().await.unwrap();
        assert!(
            stdout_of(
                &child,
                &format!("git -C '{}' log --oneline", setup.worktree.display())
            )
            .await
            .contains("parent work"),
            "the child sees its parent's commit"
        );
        assert_eq!(
            exit_code_of(&child, &commit_script(&setup.worktree, "child work")).await,
            0
        );

        let child_commits = child_ledger.since(&child_mark).await.unwrap();
        assert_eq!(
            child_commits
                .iter()
                .map(|commit| commit.subject.as_str())
                .collect::<Vec<_>>(),
            ["child work"]
        );
        let parent_commits = parent_ledger.since(&parent_mark).await.unwrap();
        assert_eq!(
            parent_commits
                .iter()
                .map(|commit| commit.subject.as_str())
                .collect::<Vec<_>>(),
            ["parent work", "child work"]
        );
        for commit in &parent_commits {
            assert!(setup.main_has(&commit.commit));
        }
        setup.main(&["fsck", "--strict", "--no-dangling"]);
    }

    /// The import asks a sandbox that cannot read another repository for that
    /// repository's objects: they cannot reach the main one.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn objects_the_subagent_cannot_read_never_reach_the_main_repository() {
        let setup = CommitSetup::new("five").await;
        let child = setup.committing_child(&setup.parent).await;
        let outside = tempfile::tempdir().expect("temp dir should be created");
        std::fs::write(outside.path().join("secret.txt"), "secret").unwrap();
        crate::subagent::tests::git_in(outside.path(), &["init", "--initial-branch=main"]);
        crate::subagent::tests::git_in(outside.path(), &["add", "secret.txt"]);
        crate::subagent::tests::git_in(outside.path(), &["commit", "-m", "secret commit"]);
        let secret = crate::subagent::tests::git_in(outside.path(), &["rev-parse", "HEAD"]);
        // Reads are open except the denylist, so the other repository is
        // unreadable to the subagent only by an embedder denial.
        child
            .grants()
            .deny_read(std::fs::canonicalize(outside.path()).unwrap());
        assert_ne!(
            exit_code_of(
                &child,
                &format!("cat '{}'", outside.path().join("secret.txt").display())
            )
            .await,
            0,
            "the control: the sandbox cannot read the other repository"
        );

        let ledger = child.commit_ledger().unwrap();
        let mark = ledger.mark().await.unwrap();
        // The subagent's repository names a commit only the other one has,
        // and reaches its objects through symlinks, as its own: a process
        // that could read the other repository would pack them (the pack
        // leaves out only objects borrowed from an alternate).
        for entry in std::fs::read_dir(outside.path().join(".git/objects")).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name();
            if name.len() == 2 {
                std::os::unix::fs::symlink(entry.path(), setup.subagent.join("objects").join(name))
                    .unwrap();
            }
        }
        std::fs::write(
            setup.subagent.join("refs/heads/subagent/five/stolen"),
            format!("{secret}\n"),
        )
        .unwrap();
        assert_eq!(
            crate::subagent::tests::git_in(
                &setup.subagent,
                &["--git-dir", ".", "cat-file", "-t", &secret]
            ),
            "commit",
            "the control: a git that could read the other repository packs its commit"
        );
        let error = ledger
            .since(&mark)
            .await
            .expect_err("the sandbox cannot read the objects it is asked to pack");
        assert!(
            matches!(error, crate::CommitLedgerError::Export { .. }),
            "the sandboxed git is what refuses: {error}"
        );
        assert!(!setup.main_has(&secret), "the secret commit never arrives");
        assert!(
            !setup
                .common
                .join("refs/heads/subagent/five/stolen")
                .exists(),
            "and no ref names it"
        );
    }

    /// The import budget is the terminal's operator setting: 1 GiB unless set,
    /// and a change through its handle reaches the subagents already running.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn the_import_budget_is_a_setting_adjustable_while_subagents_run() {
        let setup = CommitSetup::new("budget").await;
        let budget = setup.parent.subagent_import_budget();
        assert_eq!(budget.get(), 1024 * 1024 * 1024);
        let child = setup.committing_child(&setup.parent).await;
        let ledger = child.commit_ledger().unwrap();
        let mark = ledger.mark().await.unwrap();
        assert_eq!(
            exit_code_of(
                &child,
                &commit_script(&setup.worktree, "heavier than 1 byte")
            )
            .await,
            0
        );

        budget.set(1);
        let error = ledger
            .since(&mark)
            .await
            .expect_err("the setting changed after the subagent launched applies to it");
        assert!(
            matches!(
                error,
                crate::CommitLedgerError::Export { .. } | crate::CommitLedgerError::Rejected { .. }
            ),
            "{error}"
        );
        budget.set(0);
        let listed = ledger.since(&mark).await.expect("zero bounds nothing");
        assert_eq!(listed.len(), 1);
    }

    /// A commit the main agent makes between runs is visible to the next run,
    /// and what the subagent had not yet shared survives the refresh.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn the_run_start_refresh_shows_new_main_commits_and_keeps_unimported_work() {
        let setup = CommitSetup::new("six").await;
        let child = setup.committing_child(&setup.parent).await;
        let ledger = child.commit_ledger().unwrap();
        ledger.mark().await.unwrap();
        assert_eq!(
            exit_code_of(&child, &commit_script(&setup.worktree, "unimported")).await,
            0
        );

        crate::subagent::tests::git_in(
            &dunce::canonicalize(setup.dir.path().join("repository")).unwrap(),
            &["commit", "--allow-empty", "-m", "main moved on"],
        );
        let log = |branch: &str| {
            format!(
                "git -C '{}' log --format=%s {branch}",
                setup.worktree.display()
            )
        };
        assert!(
            !stdout_of(&child, &log("main"))
                .await
                .contains("main moved on"),
            "the snapshot is the one from before"
        );

        ledger.mark().await.unwrap();
        assert!(
            stdout_of(&child, &log("main"))
                .await
                .contains("main moved on"),
            "the next run starts with the main agent's new commit"
        );
        assert!(
            stdout_of(&child, &log("subagent/six/work"))
                .await
                .contains("unimported"),
            "the subagent's own work survives the refresh"
        );
    }

    /// A container backend spells the repository's remote and runs the
    /// refresh, as the subagent, through the confined streamed exec.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_container_commit_grant_refreshes_through_the_confined_streamed_exec() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let worktree = crate::subagent::tests::repository_with_linked_worktree(dir.path(), "seven");
        let (scripts, parent) = recording_container_terminal(dir.path(), Some(dir.path())).await;
        let child = launch(
            &parent,
            SubagentGrant::new().with_committing_worktree(&worktree),
        )
        .await
        .expect("a worktree inside the workspace can be granted commit");

        let subagent = dunce::canonicalize(dir.path())
            .unwrap()
            .join("repository/.git/worktrees/seven/subagent");
        let config = std::fs::read_to_string(subagent.join("config")).unwrap();
        assert!(
            config.contains("url = \"/workspace/repository/.git\""),
            "the remote is spelled as the container sees it: {config}"
        );
        assert!(
            config.contains("hooksPath = \"../repository/.git/hooks\""),
            "{config}"
        );

        // The recording exec prints nothing, so the import that follows the
        // refresh finds no report; only the refresh is of interest here.
        let _ = child.commit_ledger().unwrap().mark().await;
        let scripts = scripts.lock().unwrap().clone();
        let refresh = scripts
            .iter()
            .find(|script| script.contains("fetch"))
            .expect("the refresh ran in the container");
        assert!(refresh.contains("run --working-dir"), "{refresh}");
        assert!(
            refresh.contains("--grant '/workspace/repository/.git/worktrees/seven/subagent=rw'"),
            "{refresh}"
        );
        assert!(
            refresh.contains("git --git-dir /workspace/repository/.git/worktrees/seven/subagent"),
            "{refresh}"
        );
        assert!(refresh.contains("main-repository"), "{refresh}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_commit_grant_needs_a_linked_worktree_whose_repository_the_launcher_can_write() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let repository = crate::subagent::tests::repository_with_linked_worktree(dir.path(), "two")
            .parent()
            .unwrap()
            .join("repository");
        let plain = dir.path().join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        let parent = authority_terminal(dir.path()).await;

        let checkout = launch(
            &parent,
            SubagentGrant::new().with_committing_worktree(&repository),
        )
        .await
        .expect_err("a checkout holds its git data inside the grant");
        assert!(
            matches!(checkout, SubagentGrantError::RepositoryCheckout(_)),
            "{checkout}"
        );
        let not_git = launch(
            &parent,
            SubagentGrant::new().with_committing_worktree(&plain),
        )
        .await
        .expect_err("a plain directory has nothing to commit in");
        assert!(
            matches!(not_git, SubagentGrantError::NotAWorktree(_)),
            "{not_git}"
        );

        // A worktree inside the launcher's space whose repository is not.
        let elsewhere = tempfile::tempdir().expect("temp dir should be created");
        let foreign =
            crate::subagent::tests::repository_with_linked_worktree(elsewhere.path(), "three");
        let inside = dir.path().join("three");
        std::fs::rename(&foreign, &inside).unwrap();
        crate::subagent::tests::git_in(
            &elsewhere.path().join("repository"),
            &["worktree", "repair", inside.to_str().unwrap()],
        );
        let outside = launch(
            &parent,
            SubagentGrant::new().with_committing_worktree(&inside),
        )
        .await
        .expect_err("the launcher cannot hand on git data it cannot write");
        assert!(
            matches!(outside, SubagentGrantError::OutsideAuthority(_)),
            "{outside}"
        );
    }

    /// A subagent hands on only what it holds: commit where it was granted
    /// commit, and paths inside its own authority.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_nested_launch_never_exceeds_its_launcher() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let worktree = crate::subagent::tests::repository_with_linked_worktree(dir.path(), "four");
        let sibling = dir.path().join("sibling");
        std::fs::create_dir_all(&sibling).unwrap();
        let parent = authority_terminal(dir.path()).await;

        let writer = launch(&parent, SubagentGrant::new().with_worktree(&worktree))
            .await
            .unwrap();
        let escalated = launch(
            &writer,
            SubagentGrant::new().with_committing_worktree(&worktree),
        )
        .await
        .expect_err("a subagent without commit cannot grant it");
        assert!(
            matches!(escalated, SubagentGrantError::CommitOutsideAuthority(_)),
            "{escalated}"
        );
        let sideways = launch(&writer, SubagentGrant::new().with_worktree(&sibling))
            .await
            .expect_err("a subagent cannot grant what it cannot write");
        assert!(
            matches!(sideways, SubagentGrantError::OutsideAuthority(_)),
            "{sideways}"
        );

        let committer = launch(
            &parent,
            SubagentGrant::new().with_committing_worktree(&worktree),
        )
        .await
        .unwrap();
        let handed_on = launch(
            &committer,
            SubagentGrant::new().with_committing_worktree(&worktree),
        )
        .await
        .expect("a subagent hands on the commit grant it holds");
        assert_eq!(
            handed_on.grants().snapshot().writable,
            committer.grants().snapshot().writable
        );
        assert!(handed_on.working_dir().starts_with(committer.working_dir()));
    }

    /// A terminal that approves unsafe scripts.
    #[cfg(unix)]
    async fn unsafe_terminal(dir: &Path) -> Shell<executor_core::tokio::TokioGlobal> {
        fresh_terminal(dir).await.with_approver(ApproveAll)
    }

    /// Runs an approved unsafe `script` to its end and renders the result.
    #[cfg(unix)]
    async fn run_unsafe(tool: &Shell<executor_core::tokio::TokioGlobal>, script: &str) -> String {
        let result = tool
            .call(
                ShellArgs {
                    sandbox: SandboxMode::Unsandboxed,
                    ..terminal_args("unsafe probe", script)
                },
                ToolContext::new(),
            )
            .await
            .expect("the unsafe call succeeds");
        render_terminal_result(&result)
    }

    /// An approved unsafe script reads a file its sandboxed sibling cannot:
    /// host access grants the invoking user's whole filesystem, not a grant
    /// list.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_unsafe_script_reads_a_file_outside_every_grant() {
        let session = tempfile::tempdir().expect("temp dir should be created");
        let home = tempfile::Builder::new()
            .prefix("aither-unsafe-home-")
            .tempdir_in(std::env::var_os("HOME").expect("HOME is set"))
            .expect("a temp dir under the real home");
        let secret = home.path().join("secret.txt");
        std::fs::write(&secret, "token-value").expect("secret written");
        let tool = unsafe_terminal(session.path()).await;

        let result = run_unsafe(&tool, &format!("cat {}", secret.display())).await;

        assert!(
            result.contains("token-value"),
            "the unsafe run read the file: {result}"
        );
    }

    /// Host access also writes wherever the invoking user may write.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_unsafe_script_writes_a_file_outside_every_grant() {
        let session = tempfile::tempdir().expect("temp dir should be created");
        let outside = tempfile::tempdir().expect("temp dir should be created");
        let probe = outside.path().join("written.txt");
        let tool = unsafe_terminal(session.path()).await;

        let result = run_unsafe(&tool, &format!("printf %s written > {}", probe.display())).await;

        assert!(
            !result.contains("Exit code"),
            "the unsafe run failed: {result}"
        );
        assert_eq!(
            std::fs::read_to_string(&probe).expect("written file reads back"),
            "written"
        );
    }

    /// Confined profiles deny exec in the shared temp directories; host access
    /// runs what a user put there.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_unsafe_script_executes_a_program_written_to_the_temp_directory() {
        let session = tempfile::tempdir().expect("temp dir should be created");
        let outside = tempfile::tempdir().expect("temp dir should be created");
        let probe = outside.path().join("probe.sh");
        std::fs::write(&probe, "#!/bin/sh\necho executed\n").expect("probe written");
        let tool = unsafe_terminal(session.path()).await;

        let result = run_unsafe(&tool, &format!("chmod +x {0} && {0}", probe.display())).await;

        assert!(
            result.contains("executed"),
            "the unsafe run ran the probe: {result}"
        );
    }

    /// No proxy stands between an unsafe script and the network: a loopback
    /// listener the test binds accepts its connection directly.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_unsafe_script_connects_to_loopback_without_a_proxy() {
        let session = tempfile::tempdir().expect("temp dir should be created");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("local addr").port();
        let (accepted, got) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = accepted.send(listener.accept());
        });
        let tool = unsafe_terminal(session.path()).await;

        let result = run_unsafe(
            &tool,
            &format!("/bin/bash -c 'exec 3<>/dev/tcp/127.0.0.1/{port}'"),
        )
        .await;

        assert!(
            !result.contains("Exit code"),
            "the unsafe run failed: {result}"
        );
        got.recv_timeout(Duration::from_secs(10))
            .expect("the listener accepted the connection")
            .expect("accept succeeded");
    }

    #[tokio::test]
    async fn a_subagent_of_a_subagent_works_inside_its_launchers_space() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let parent = fresh_terminal(dir.path()).await;
        let child = launch(&parent, SubagentGrant::new()).await.unwrap();
        let grandchild = launch(&child, SubagentGrant::new()).await.unwrap();
        assert!(grandchild.working_dir().starts_with(child.working_dir()));
    }

    // ---- terminal runner ----------------------------------------------------

    /// Runs a script on `terminal` and answers with how the script ended and
    /// what it printed.
    #[cfg(unix)]
    async fn run_check_probe(
        terminal: &ConfiguredTestTool,
        dir: &Path,
        script: &str,
        timeout: u64,
    ) -> String {
        let result = terminal
            .run(Script::new(script, dir, Duration::from_secs(timeout)))
            .await
            .expect("the run ends");
        match (&result.outcome, &result.stdout) {
            (crate::RunOutcome::Exited { code }, CommandStream::Inline { text }) => {
                format!("exit={code} stdout={text}")
            }
            (crate::RunOutcome::Killed(_), _) => "killed".to_string(),
            other => panic!("a run ends exited with inline output or killed: {other:?}"),
        }
    }

    /// A run executes a script on the shell, under its policy as it stands at
    /// the run: a grant made after the registry was bound included.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_run_executes_a_script_under_the_shells_policy() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let clone = tempfile::tempdir().expect("clone dir should be created");
        let read_only = tempfile::tempdir().expect("read-only dir should be created");
        let terminal = fresh_terminal(dir.path()).await;
        terminal.grants().add_readable(read_only.path());
        terminal.grants().add_writable(clone.path());

        let clone_dir = std::fs::canonicalize(clone.path()).unwrap();
        let answer = run_check_probe(&terminal, &clone_dir, "pwd && touch checked", 30).await;
        assert!(answer.starts_with("exit=0 "), "{answer}");
        assert!(
            answer.contains(clone_dir.to_str().unwrap()),
            "the run starts in the directory it names: {answer}"
        );
        assert!(clone_dir.join("checked").exists());

        let read_only_dir = std::fs::canonicalize(read_only.path()).unwrap();
        let answer = run_check_probe(&terminal, &read_only_dir, "touch denied", 30).await;
        assert!(
            !answer.starts_with("exit=0 "),
            "a read-only grant stays read-only: {answer}"
        );
        assert!(!read_only_dir.join("denied").exists());

        let answer = run_check_probe(&terminal, &clone_dir, "sleep 30", 1).await;
        assert_eq!(answer, "killed", "the timeout kills the run");

        let unsafe_run = terminal
            .run(
                Script::new("true", &clone_dir, Duration::from_secs(30))
                    .with_sandbox(SandboxMode::Unsandboxed),
            )
            .await;
        assert!(
            matches!(
                unsafe_run,
                Err(ShellRunError::Terminal(TerminalError::NoApprover(_)))
            ),
            "an unsafe run is put to an approver, and this terminal has none: {unsafe_run:?}"
        );
    }

    /// Records the script every container exec runs.
    #[cfg(unix)]
    struct RecordingExec {
        scripts: Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[cfg(unix)]
    impl crate::ContainerExec for RecordingExec {
        fn exec(
            &self,
            _container_id: &str,
            script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _lifetime: ExecLifetime,
        ) -> impl std::future::Future<Output = Result<ContainerExecOutcome, String>> + Send
        {
            self.scripts.lock().unwrap().push(script.to_string());
            std::future::ready(Ok(ContainerExecOutcome::Completed {
                output: std::process::Output {
                    status: std::os::unix::process::ExitStatusExt::from_raw(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                },
                killed: Vec::new(),
            }))
        }

        fn exec_streamed(
            &self,
            _container_id: &str,
            script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _stdout: Sender<Vec<u8>>,
            _lifetime: ExecLifetime,
        ) -> impl std::future::Future<Output = Result<ContainerExecOutcome, String>> + Send
        {
            self.scripts.lock().unwrap().push(script.to_string());
            std::future::ready(Ok(ContainerExecOutcome::Completed {
                output: std::process::Output {
                    status: std::os::unix::process::ExitStatusExt::from_raw(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                },
                killed: Vec::new(),
            }))
        }
    }

    /// A terminal whose home backend is a recording container, which mounts
    /// `workspace` as `/workspace` when one is given.
    #[cfg(unix)]
    async fn recording_container_terminal(
        dir: &Path,
        workspace: Option<&Path>,
    ) -> (Arc<std::sync::Mutex<Vec<String>>>, ConfiguredTestTool) {
        let scripts = Arc::new(std::sync::Mutex::new(Vec::new()));
        let runtime = ContainerShellRuntime::new(
            "aither-test-confinement",
            crate::HostRoute::new(
                "host.docker.internal",
                std::net::IpAddr::from([127, 0, 0, 1]),
            )
            .expect("loopback is routable"),
            crate::AnyContainerExec::new(Arc::new(RecordingExec {
                scripts: scripts.clone(),
            })),
        );
        let runtime = match workspace {
            Some(workspace) => runtime.with_workspace_mount(MountRoot::new(
                workspace,
                ContainerPath::new(CONTAINER_EXEC_WORKING_DIR).expect("the workspace is absolute"),
            )),
            None => runtime,
        };
        let sandbox = container_sandbox(&runtime, dir);
        let parent = Shell::new_exact_on(
            sandbox,
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
            crate::test_wording::job_tools(),
        )
        .await
        .expect("terminal tool should initialize");
        (scripts, parent)
    }

    /// A container terminal's jobs are the container's: they carry the
    /// marker its memory reader finds them by, are registered as running
    /// there, and the terminal judges the container's memory too.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn container_jobs_are_marked_and_placed_in_the_container() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (scripts, parent) = recording_container_terminal(dir.path(), None).await;
        assert!(parent.container_probe().is_some());
        assert!(parent.job_memory().has_container());

        // A background job, whose completion is held so the registry still
        // keeps it to be read.
        let receiver = parent.background_receiver();
        parent
            .run_background(background_args("mark", "echo marked-command"))
            .await
            .expect("terminal call should succeed");
        let task_id = started_task_id(&receiver);
        let _completed = wait_for_completed_task(&receiver, &task_id).await;

        let scripts = scripts.lock().unwrap().clone();
        let script = scripts
            .iter()
            .find(|script| script.contains("marked-command"))
            .expect("the command ran");
        let jobs = parent.job_registry().list().await;
        let job = jobs
            .iter()
            .find(|job| job.script == "echo marked-command")
            .expect("the job is registered");
        assert_eq!(job.place, crate::JobPlace::Container);
        // The sandbox exports it ahead of the command, which the managed
        // shell's command line quotes.
        assert!(
            script.contains(&format!("export {}=", crate::memory::JOB_MARKER_ENV))
                && script.contains(job.task_id.as_str()),
            "{script}"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_local_terminal_judges_no_container() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await;
        assert!(tool.container_probe().is_none());
        assert!(!tool.job_memory().has_container());
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn container_session_runs_the_main_agent_unconfined_and_a_subagent_confined() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let granted = dir.path().join("granted");
        std::fs::create_dir_all(&granted).unwrap();
        let (scripts, parent) = recording_container_terminal(dir.path(), Some(dir.path())).await;
        let child = launch(&parent, SubagentGrant::new().with_worktree(&granted))
            .await
            .expect("a directory inside the workspace can be granted");

        assert_eq!(exit_code_of(&parent, "echo main-command").await, 0);
        assert_eq!(exit_code_of(&child, "echo subagent-command").await, 0);

        let scripts = scripts.lock().unwrap().clone();
        let main = scripts
            .iter()
            .find(|script| script.contains("main-command"))
            .expect("the main agent's command ran");
        let sub = scripts
            .iter()
            .find(|script| script.contains("subagent-command"))
            .expect("the subagent's command ran");
        assert!(
            !main.contains(" run --"),
            "the container is the main agent's boundary: {main}"
        );
        assert!(
            sub.contains("\"$AITHER_HEEL_BIN\" run --working-dir"),
            "{sub}"
        );
        assert!(!sub.contains("--isolation permissive"), "{sub}");
        let root = dunce::canonicalize(dir.path()).unwrap();
        let space = child
            .working_dir()
            .strip_prefix(&root)
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            sub.contains(&format!("--working-dir '/workspace/{space}'")),
            "{sub}"
        );
        let granted_name = granted.file_name().unwrap().to_str().unwrap();
        assert!(
            sub.contains(&format!("--grant '/workspace/{granted_name}=rw'")),
            "{sub}"
        );
    }

    /// The shim inside a container compares its working directory against
    /// the worktree map: the map must spell the paths the mount resolves —
    /// host spellings would leave the worktree unprotected, the `git` shim
    /// unable to hit it.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_containers_git_map_spells_the_worktree_and_gitdir_the_container_way() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let worktree =
            crate::subagent::tests::repository_with_linked_worktree(dir.path(), "mapped");
        let (scripts, parent) = recording_container_terminal(dir.path(), Some(dir.path())).await;
        let _child = launch(
            &parent,
            SubagentGrant::new().with_committing_worktree(&worktree),
        )
        .await
        .expect("the worktree can be granted with commit");

        assert_eq!(exit_code_of(&parent, "echo mapped-worktree").await, 0);

        let script = scripts
            .lock()
            .unwrap()
            .iter()
            .find(|script| script.contains("mapped-worktree"))
            .expect("the main agent's command ran")
            .clone();
        assert!(
            script.contains("export AITHER_GIT_WORKTREE_PATH_0=/workspace/mapped"),
            "{script}"
        );
        assert!(
            script.contains(
                "export AITHER_GIT_WORKTREE_GITDIR_0=/workspace/repository/.git/worktrees/mapped"
            ),
            "{script}"
        );
        let host = dunce::canonicalize(&worktree)
            .unwrap()
            .display()
            .to_string();
        assert!(
            !script.contains(&host),
            "the shim sees no host spelling: {script}"
        );
    }

    /// A subagent whose paths the container cannot express is refused when
    /// it is launched, not when it first runs a command.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_container_subagent_outside_the_workspace_mount_is_refused_at_launch() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let mounted = dir.path().join("mounted");
        let granted = dir.path().join("granted");
        std::fs::create_dir_all(&mounted).unwrap();
        std::fs::create_dir_all(&granted).unwrap();

        let (_scripts, parent) = recording_container_terminal(dir.path(), Some(&mounted)).await;
        let error = launch(&parent, SubagentGrant::new().with_worktree(&granted))
            .await
            .expect_err("the subagent's paths are not inside the mount");
        assert!(
            matches!(error, SubagentGrantError::OutsideContainerMount { .. }),
            "{error}"
        );

        let (_scripts, unmounted) = recording_container_terminal(dir.path(), None).await;
        let error = launch(&unmounted, SubagentGrant::new())
            .await
            .expect_err("a container without a declared mount cannot confine a subagent");
        assert!(
            matches!(error, SubagentGrantError::NoContainerMount),
            "{error}"
        );
    }

    /// A local terminal in `dir` under `policies`, for the script policy
    /// tests.
    #[cfg(unix)]
    async fn policy_tool(
        dir: &Path,
        policies: Vec<crate::AnyScriptPolicy>,
    ) -> Shell<executor_core::tokio::TokioGlobal> {
        fresh_terminal(dir).await.with_test_policies(policies)
    }

    /// Refuses every shell script that runs `program`, and reports how each
    /// script it was asked about ended.
    struct RefuseProgram {
        program: &'static str,
        ends: async_channel::Sender<crate::ScriptOutcome>,
    }

    impl RefuseProgram {
        fn refusal(&self) -> String {
            format!(
                "`{}` does not run here; kill the PID you started.",
                self.program
            )
        }
    }

    impl crate::ScriptPolicy for RefuseProgram {
        type State = ();

        fn check(
            &self,
            script: &crate::ScriptCheck<'_>,
            _state: &(),
        ) -> impl Future<Output = ScriptDecision> + Send {
            let runs = matches!(
                script.shell,
                Some(Ok(shell)) if shell.commands().iter().any(|command| command.program() == self.program)
            );
            std::future::ready(if runs {
                ScriptDecision::Refuse(ScriptRefusal::new(self.refusal()))
            } else {
                ScriptDecision::Pass
            })
        }

        async fn ended(&self, end: &crate::ScriptEnd<'_>, _state: &()) {
            self.ends
                .send(end.outcome)
                .await
                .expect("the test holds the receiver");
        }
    }

    /// Refuses a script identical to the agent's last one that exited 0;
    /// a script that failed is not remembered.
    struct RefuseRepeat;

    #[derive(Default)]
    struct Repeats {
        /// Task id → script, for the scripts allowed and not yet ended.
        pending: std::sync::Mutex<std::collections::HashMap<String, String>>,
        last_success: std::sync::Mutex<Option<String>>,
    }

    impl crate::ScriptPolicy for RefuseRepeat {
        type State = Repeats;

        fn check(
            &self,
            script: &crate::ScriptCheck<'_>,
            state: &Repeats,
        ) -> impl Future<Output = ScriptDecision> + Send {
            let decision = if state.last_success.lock().unwrap().as_deref() == Some(script.script) {
                ScriptDecision::Refuse(ScriptRefusal::new(
                    "It would print what it printed last time.",
                ))
            } else {
                state
                    .pending
                    .lock()
                    .unwrap()
                    .insert(script.task_id.to_string(), script.script.to_string());
                ScriptDecision::Pass
            };
            std::future::ready(decision)
        }

        fn ended(
            &self,
            end: &crate::ScriptEnd<'_>,
            state: &Repeats,
        ) -> impl Future<Output = ()> + Send {
            let script = state.pending.lock().unwrap().remove(end.task_id);
            if end.outcome == (crate::ScriptOutcome::Exited { code: 0 }) {
                *state.last_success.lock().unwrap() = script;
            }
            std::future::ready(())
        }
    }

    /// A policy's refusal is what the caller reads, before any part of the
    /// script runs, in the foreground and the background alike; the policy
    /// is told the script never ran.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_policy_refusal_reaches_the_caller_before_the_script_runs() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (ends, ended) = async_channel::unbounded();
        let policy = RefuseProgram {
            program: "pkill",
            ends,
        };
        let text = policy.refusal();
        let tool = policy_tool(dir.path(), vec![crate::AnyScriptPolicy::new(policy)]).await;
        let marker = dir.path().join("ran");
        let script = format!("touch {}; sudo env pkill anything", marker.display());

        let error = tool
            .call(
                terminal_args("name-based kill", &script),
                ToolContext::new(),
            )
            .await
            .expect_err("the policy refuses the script");
        assert_eq!(error.to_string(), text);
        let refusal = error
            .downcast_ref::<ScriptRefusal>()
            .expect("the refusal stays typed for the agent loop");
        assert_eq!(refusal.text(), text);
        let error = tool
            .run_background(terminal_args("name-based kill", &script))
            .await
            .expect_err("the policy refuses the background script");
        assert_eq!(error.to_string(), text);
        assert!(!marker.exists(), "no part of a refused script runs");
        assert_eq!(ended.try_recv(), Ok(crate::ScriptOutcome::NotRun));
        assert_eq!(ended.try_recv(), Ok(crate::ScriptOutcome::NotRun));

        let result = tool
            .call(
                terminal_args("mention a killer", "echo pkill; exit 3"),
                ToolContext::new(),
            )
            .await
            .expect("a script that only names the program runs");
        assert!(render_terminal_result(&result).contains("pkill"));
        assert_eq!(
            ended.try_recv(),
            Ok(crate::ScriptOutcome::Exited { code: 3 }),
            "the policy learns how the script ended before its result returns"
        );
    }

    /// Records the directory each script started in and the one it ended in.
    struct RecordDirs {
        starts: async_channel::Sender<ScriptDir>,
        ends: async_channel::Sender<(ScriptDir, Carry)>,
    }

    impl crate::ScriptPolicy for RecordDirs {
        type State = ();

        async fn check(&self, script: &crate::ScriptCheck<'_>, (): &()) -> ScriptDecision {
            self.starts
                .send(script.start.clone())
                .await
                .expect("the test holds the receiver");
            ScriptDecision::Pass
        }

        async fn ended(&self, end: &crate::ScriptEnd<'_>, (): &()) {
            self.ends
                .send((end.end.clone(), end.carry))
                .await
                .expect("the test holds the receiver");
        }
    }

    /// A policy sees where each script starts: after a script that ran `cd`,
    /// the next starts in that directory, and the end directory of a script
    /// is the start directory of the next.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_policy_sees_the_directory_a_script_starts_and_ends_in() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (starts_tx, starts) = async_channel::unbounded();
        let (ends_tx, ends) = async_channel::unbounded();
        let policy = RecordDirs {
            starts: starts_tx,
            ends: ends_tx,
        };
        let tool = policy_tool(dir.path(), vec![crate::AnyScriptPolicy::new(policy)]).await;
        let root = dunce::canonicalize(dir.path()).unwrap();

        for (description, script) in [
            ("enter a directory", "mkdir sub && cd sub"),
            ("stay there", "true"),
            ("leave it", "cd .."),
            ("stay at the root", "true"),
        ] {
            tool.call(terminal_args(description, script), ToolContext::new())
                .await
                .expect("the script runs");
        }

        let starts: Vec<_> = std::iter::from_fn(|| starts.try_recv().ok()).collect();
        let ends: Vec<_> = std::iter::from_fn(|| ends.try_recv().ok()).collect();
        let sub = root.join("sub");
        let host = |dir: &Path| ScriptDir::Host(dir.to_path_buf());
        assert_eq!(starts, [host(&root), host(&sub), host(&sub), host(&root)]);
        assert_eq!(
            ends,
            [
                (host(&sub), Carry::Moved),
                (host(&sub), Carry::Unchanged),
                (host(&root), Carry::Moved),
                (host(&root), Carry::Unchanged),
            ]
        );
        for ((end, _), next) in ends.iter().zip(&starts[1..]) {
            assert_eq!(end, next, "a script starts where the last ended");
        }
    }

    /// `Carry` reports each of its cases: a script that moved the shell, left
    /// it where it started, failed after moving it, exited without a report,
    /// and one that runs in the background.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_policy_is_told_what_each_script_did_to_the_carried_directory() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (starts_tx, _starts) = async_channel::unbounded();
        let (ends_tx, ends) = async_channel::unbounded();
        let policy = RecordDirs {
            starts: starts_tx,
            ends: ends_tx,
        };
        let tool = policy_tool(dir.path(), vec![crate::AnyScriptPolicy::new(policy)]).await;
        let root = dunce::canonicalize(dir.path()).unwrap();
        let sub = ScriptDir::Host(root.join("sub"));

        for script in ["mkdir sub && cd sub", "true", "cd ..; false", "exit 0"] {
            tool.call(terminal_args("carry", script), ToolContext::new())
                .await
                .expect("the script runs");
        }
        // The call returns before a background script ends.
        tool.run_background(terminal_args("detach", "sleep 1"))
            .await
            .expect("the background script starts");
        assert!(
            ends.len() == 4,
            "a background script's end is told after its call returned"
        );
        let carries: Vec<_> = std::iter::from_fn(|| ends.try_recv().ok()).collect();
        assert_eq!(
            carries,
            [
                (sub.clone(), Carry::Moved),
                (sub.clone(), Carry::Unchanged),
                (sub.clone(), Carry::Reset(DirectoryResetReason::Failed)),
                (sub.clone(), Carry::Reset(DirectoryResetReason::Unreported)),
            ]
        );
        let background = tokio::time::timeout(Duration::from_secs(20), ends.recv())
            .await
            .expect("the background script ends")
            .expect("the test holds the sender");
        assert_eq!(background, (sub, Carry::NotCarried));
    }

    /// Scripts started together start in the directory that was carried when
    /// each was admitted, and each reports an end directory that is a
    /// directory some script of the group settled, which the next script
    /// then starts in.
    #[cfg(unix)]
    #[tokio::test]
    async fn parallel_scripts_start_alike_and_the_next_starts_where_a_last_ended() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (starts_tx, starts) = async_channel::unbounded();
        let (ends_tx, ends) = async_channel::unbounded();
        let policy = RecordDirs {
            starts: starts_tx,
            ends: ends_tx,
        };
        let tool = policy_tool(dir.path(), vec![crate::AnyScriptPolicy::new(policy)]).await;
        let root = dunce::canonicalize(dir.path()).unwrap();

        let calls = ["a", "b", "c", "d"].map(|name| {
            let tool = &tool;
            async move {
                tool.call(
                    terminal_args("enter", &format!("mkdir -p {name} && cd {name}")),
                    ToolContext::new(),
                )
                .await
                .expect("the script runs");
            }
        });
        futures::future::join_all(calls).await;
        let starts: Vec<_> = std::iter::from_fn(|| starts.try_recv().ok()).collect();
        assert_eq!(starts, vec![ScriptDir::Host(root.clone()); 4]);
        let ends: Vec<_> = std::iter::from_fn(|| ends.try_recv().ok()).collect();
        assert_eq!(ends.len(), 4);
        let settled = ["a", "b", "c", "d"].map(|name| ScriptDir::Host(root.join(name)));
        for (end, carry) in &ends {
            assert!(settled.contains(end), "{end:?}");
            assert_eq!(*carry, Carry::Moved);
        }

        tool.call(terminal_args("next", "true"), ToolContext::new())
            .await
            .expect("the script runs");
        let next = starts_after(&tool).await;
        assert!(ends.iter().any(|(end, _)| *end == next));
    }

    /// The directory the terminal's next foreground script starts in.
    #[cfg(unix)]
    async fn starts_after(tool: &Shell<executor_core::tokio::TokioGlobal>) -> ScriptDir {
        carried_start(tool.carried_dir.as_ref()).await.map_or_else(
            || ScriptDir::Host(tool.working_dir.clone()),
            ScriptDir::Host,
        )
    }

    /// Two policies that disagree resolve to the strictest decision: a
    /// refusal outranks an approval, wherever they stand in the registry.
    #[cfg(unix)]
    #[tokio::test]
    async fn policies_that_disagree_resolve_to_the_strictest_decision() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = approval_tool(
            dir.path(),
            [
                ScriptDecision::Refuse(ScriptRefusal::new("refused by the second")),
                ScriptDecision::Allow,
            ],
            None,
        )
        .await;
        let error = tool
            .call(terminal_args("disagree", "true"), ToolContext::new())
            .await
            .expect_err("the refusal outranks the approval");
        assert_eq!(error.to_string(), "refused by the second");
    }

    /// With no policy the sandbox refuses nothing: `grep` and a foreground
    /// `sleep` run.
    #[cfg(unix)]
    #[tokio::test]
    async fn with_no_policy_grep_and_a_foreground_sleep_run() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = policy_tool(dir.path(), Vec::new()).await;
        let result = tool
            .call(
                terminal_args(
                    "grep and sleep",
                    "printf 'needle\\n' | grep needle; sleep 1; echo slept",
                ),
                ToolContext::new(),
            )
            .await
            .expect("nothing is refused");
        let rendered = render_terminal_result(&result);
        assert!(rendered.contains("needle"), "{rendered}");
        assert!(rendered.contains("slept"), "{rendered}");
    }

    /// A policy keeps state for its agent across calls, and learns of each
    /// end before the call returns: a repeat of a script that succeeded is
    /// refused, a failed one runs again.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_policy_state_follows_each_script_to_its_end() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = policy_tool(dir.path(), vec![crate::AnyScriptPolicy::new(RefuseRepeat)]).await;
        assert_eq!(exit_code_of(&tool, "exit 3").await, 3);
        assert_eq!(
            exit_code_of(&tool, "exit 3").await,
            3,
            "a failed script is not remembered, so it runs again"
        );
        assert_eq!(exit_code_of(&tool, "echo once").await, 0);
        let error = tool
            .call(terminal_args("again", "echo once"), ToolContext::new())
            .await
            .expect_err("a repeat of a successful script is refused");
        assert!(error.downcast_ref::<ScriptRefusal>().is_some(), "{error}");
        assert_eq!(exit_code_of(&tool, "echo twice").await, 0);
    }

    use crate::monitor::{MonitorArgs, MonitorEnd, MonitorSource, MonitorTool};

    /// Job alerts the monitor produces until `ended` or `duration` passes.
    async fn monitor_alerts(receiver: &ShellEvents, duration: Duration) -> Vec<ShellEvent> {
        let deadline = std::time::Instant::now() + duration;
        let mut alerts = Vec::new();
        while let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) {
            let Some(alert) = receiver.recv_timeout(remaining).await else {
                break;
            };
            if matches!(alert, ShellEvent::JobEnded(_) | ShellEvent::JobStarted(_)) {
                continue;
            }
            let ended = matches!(alert, ShellEvent::MonitorEnded(_));
            alerts.push(alert);
            if ended {
                break;
            }
        }
        alerts
    }

    fn monitor_command(command: &str, timeout_secs: u64) -> MonitorArgs {
        MonitorArgs {
            source: MonitorSource::Command {
                command: command.to_string(),
                interpreter: Interpreter::Shell,
            },
            interval_secs: Some(1),
            filter: None,
            stop_on: None,
            kill_on_stop: false,
            stall_secs: None,
            timeout_secs,
        }
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn monitored_command_delivers_increments_and_its_exit() {
        let dir = tempfile::tempdir().unwrap();
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(7200),
            Duration::from_secs(600),
        )
        .await;
        let monitor = MonitorTool::new(tool.clone(), crate::monitor::fixture_settings());
        monitor
            .call(
                monitor_command("echo a; sleep 2; echo b", 30),
                ToolContext::new(),
            )
            .await
            .expect("monitor starts");

        let alerts = monitor_alerts(&tool.background_receiver(), Duration::from_secs(20)).await;
        let increments: Vec<&[String]> = alerts
            .iter()
            .filter_map(|alert| match alert {
                ShellEvent::MonitorIncrement(increment) => Some(increment.lines.as_slice()),
                _ => None,
            })
            .collect();
        assert!(
            increments.iter().any(|inc| *inc == ["a".to_string()]),
            "the first line is delivered as it arrives: {alerts:?}"
        );
        assert!(
            increments.iter().any(|inc| *inc == ["b".to_string()]),
            "the later line is delivered in a following increment: {alerts:?}"
        );
        assert!(
            alerts.iter().any(|alert| matches!(
                alert,
                ShellEvent::MonitorEnded(ended)
                    if matches!(ended.end, MonitorEnd::Exited { exit_code: 0 })
            )),
            "the monitor reports the command's exit: {alerts:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn monitor_filter_keeps_matching_lines_and_stop_on_ends() {
        let dir = tempfile::tempdir().unwrap();
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(7200),
            Duration::from_secs(600),
        )
        .await;
        let monitor = MonitorTool::new(tool.clone(), crate::monitor::fixture_settings());
        let mut args = monitor_command(
            "printf 'keep one\\ndrop\\nkeep two\\nlast\\n'; sleep 60",
            30,
        );
        args.filter = Some("^keep".to_string());
        args.stop_on = Some("two".to_string());
        monitor
            .call(args, ToolContext::new())
            .await
            .expect("monitor starts");

        let alerts = monitor_alerts(&tool.background_receiver(), Duration::from_secs(20)).await;
        let lines: Vec<&String> = alerts
            .iter()
            .filter_map(|alert| match alert {
                ShellEvent::MonitorIncrement(increment) => Some(&increment.lines),
                _ => None,
            })
            .flatten()
            .collect();
        assert!(
            lines.iter().all(|line| line.starts_with("keep")),
            "filter drops non-matching lines: {lines:?}"
        );
        assert!(
            !lines.iter().any(|line| *line == "last"),
            "lines after the stop match are not delivered: {lines:?}"
        );
        assert!(
            alerts.iter().any(|alert| matches!(
                alert,
                ShellEvent::MonitorEnded(ended)
                    if matches!(&ended.end, MonitorEnd::StopMatched { line } if line == "keep two")
            )),
            "the first stop_on match ends the monitor: {alerts:?}"
        );
        tool.job_registry().kill_running().await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn monitor_rejects_a_timeout_over_the_cap_and_times_out_at_the_deadline() {
        let dir = tempfile::tempdir().unwrap();
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(7200),
            Duration::from_secs(600),
        )
        .await;
        let monitor = MonitorTool::new(tool.clone(), crate::monitor::fixture_settings());
        let refused = monitor
            .call(monitor_command("sleep 60", 7201), ToolContext::new())
            .await;
        assert!(refused.is_err(), "timeout_secs over the cap is refused");

        monitor
            .call(monitor_command("sleep 60", 2), ToolContext::new())
            .await
            .expect("monitor starts");
        let alerts = monitor_alerts(&tool.background_receiver(), Duration::from_secs(15)).await;
        assert!(
            alerts.iter().any(|alert| matches!(
                alert,
                ShellEvent::MonitorEnded(ended) if matches!(ended.end, MonitorEnd::TimedOut)
            )),
            "the deadline kills the run and reports a timeout: {alerts:?}"
        );
    }

    /// The watched job's bound stays a margin past the monitor's deadline, so
    /// the largest monitor is the shell's cap less that margin, and one past
    /// it is refused up front instead of failing to start the job.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn the_largest_monitor_leaves_the_jobs_margin_inside_the_shells_cap() {
        let dir = tempfile::tempdir().unwrap();
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(7200),
            Duration::from_secs(600),
        )
        .await;
        let monitor = MonitorTool::new(tool.clone(), crate::monitor::fixture_settings());
        monitor
            .call(monitor_command("sleep 60", 7140), ToolContext::new())
            .await
            .expect("a monitor whose job bound fits the cap starts");
        let refused = monitor
            .call(monitor_command("sleep 60", 7141), ToolContext::new())
            .await;
        assert!(refused.is_err(), "its job's bound would pass the cap");
        tool.jobs().kill_all().await;
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn monitor_reports_a_stall_once() {
        let dir = tempfile::tempdir().unwrap();
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(7200),
            Duration::from_secs(600),
        )
        .await;
        let monitor = MonitorTool::new(tool.clone(), crate::monitor::fixture_settings());
        let mut args = monitor_command("sleep 4; echo done", 30);
        args.stall_secs = Some(1);
        monitor
            .call(args, ToolContext::new())
            .await
            .expect("monitor starts");

        let alerts = monitor_alerts(&tool.background_receiver(), Duration::from_secs(15)).await;
        let stalls = alerts
            .iter()
            .filter(|alert| matches!(alert, ShellEvent::MonitorStalled(_)))
            .count();
        assert_eq!(stalls, 1, "a stall is reported once: {alerts:?}");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn monitored_file_delivers_appends_and_refuses_an_unreadable_file() {
        let dir = tempfile::tempdir().unwrap();
        let tool = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(7200),
            Duration::from_secs(600),
        )
        .await;
        let monitor = MonitorTool::new(tool.clone(), crate::monitor::fixture_settings());
        let unreadable = monitor
            .call(
                MonitorArgs {
                    source: MonitorSource::File {
                        file: "/etc/sudoers".to_string(),
                    },
                    ..monitor_command("true", 30)
                },
                ToolContext::new(),
            )
            .await;
        assert!(
            unreadable.is_err(),
            "a file the sandbox may not read is refused"
        );

        let path = tool.working_dir().join("followed.log");
        std::fs::write(&path, "existing\n").unwrap();
        monitor
            .call(
                MonitorArgs {
                    source: MonitorSource::File {
                        file: "followed.log".to_string(),
                    },
                    ..monitor_command("true", 30)
                },
                ToolContext::new(),
            )
            .await
            .expect("monitor starts");
        async_io::Timer::after(Duration::from_millis(1500)).await;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        std::io::Write::write_all(&mut file, b"appended\n").unwrap();
        std::fs::rename(&path, tool.working_dir().join("followed.log.1")).unwrap();
        std::fs::write(&path, "rotated\n").unwrap();
        async_io::Timer::after(Duration::from_millis(500)).await;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        std::io::Write::write_all(&mut file, b"after\n").unwrap();

        let alerts = monitor_alerts(&tool.background_receiver(), Duration::from_secs(15)).await;
        let lines: Vec<&String> = alerts
            .iter()
            .filter_map(|alert| match alert {
                ShellEvent::MonitorIncrement(increment) => Some(&increment.lines),
                _ => None,
            })
            .flatten()
            .collect();
        assert!(
            !lines.iter().any(|line| *line == "existing"),
            "content written before the monitor started is not delivered: {lines:?}"
        );
        for expected in ["appended", "rotated", "after"] {
            assert!(
                lines.iter().any(|line| *line == expected),
                "appended line {expected:?} is delivered, across rotation: {lines:?}"
            );
        }
        tool.job_registry().kill_running().await;
    }

    // ---- runtime grants -----------------------------------------------------

    /// The done-when case: a writable grant added at runtime reaches every
    /// command launched afterwards, while a command already running keeps
    /// the grants it launched with.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_runtime_writable_grant_reaches_new_commands_not_running_ones() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = authority_terminal(dir.path()).await;
        let outside = tempfile::tempdir().expect("outside dir should be created");
        let late_target = outside.path().join("late.txt");
        // Launched before the grant; its write happens after the grant is
        // added but its grant snapshot was taken at launch.
        let task_id = start_background_job(
            &tool,
            &format!("sleep 2; echo x > '{}'", late_target.display()),
        )
        .await;
        let receiver = tool.background_receiver();

        assert_ne!(
            exit_code_of(
                &tool,
                &format!("echo x > '{}'", outside.path().join("denied.txt").display())
            )
            .await,
            0,
            "a write outside the sandbox is denied before the grant"
        );
        tool.grants().add_writable(outside.path());
        assert_eq!(
            exit_code_of(
                &tool,
                &format!(
                    "echo x > '{}'",
                    outside.path().join("granted.txt").display()
                )
            )
            .await,
            0,
            "the command after the grant writes the approved dir"
        );

        let completed = wait_for_completed_task(&receiver, &task_id).await;
        let _ = completed.result;
        assert!(
            !late_target.exists(),
            "a command launched before the grant keeps its launch grants"
        );
    }

    /// A grant taken back no longer reaches the next command: the write the
    /// grant allowed is denied again.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_removed_grant_is_gone_from_the_next_command() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = authority_terminal(dir.path()).await;
        let outside = tempfile::tempdir().expect("outside dir should be created");
        tool.grants().add_writable(outside.path());
        assert_eq!(
            exit_code_of(
                &tool,
                &format!("touch '{}'", outside.path().join("granted.txt").display())
            )
            .await,
            0,
            "the granted dir is writable"
        );
        tool.grants().revoke(outside.path());
        assert_ne!(
            exit_code_of(
                &tool,
                &format!("touch '{}'", outside.path().join("revoked.txt").display())
            )
            .await,
            0,
            "the command after the removal is denied the write"
        );
    }

    /// A writable grant deleted mid-session no longer fails admission, and
    /// the alert fires when it was the directory the shell had carried.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_deleted_writable_grant_does_not_block_the_next_command() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (tool, ends) = carry_terminal(dir.path()).await;
        let root = dunce::canonicalize(dir.path()).unwrap();
        let outside = tempfile::tempdir().expect("outside dir should be created");
        let granted = dunce::canonicalize(outside.path()).unwrap();
        tool.grants().add_writable(&granted);
        assert_eq!(
            exit_code_of(&tool, &format!("cd '{}'", granted.display())).await,
            0
        );
        assert_eq!(
            carries(&ends),
            [(ScriptDir::Host(granted.clone()), Carry::Moved)]
        );

        std::fs::remove_dir_all(&granted).unwrap();
        assert_eq!(exit_code_of(&tool, "echo test").await, 0);
        assert_eq!(
            carries(&ends),
            [(
                ScriptDir::Host(root.clone()),
                Carry::Reset(DirectoryResetReason::Removed)
            )]
        );
        assert_eq!(pwd_of(&tool).await, root);
    }

    /// Cloned terminals share one grant store: a grant added through the
    /// original's handle reaches the clone's next command.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_cloned_terminal_sees_runtime_grants() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = authority_terminal(dir.path()).await;
        let clone = tool.clone();
        let outside = tempfile::tempdir().expect("outside dir should be created");
        tool.grants().add_writable(outside.path());
        assert_eq!(
            exit_code_of(
                &clone,
                &format!("touch '{}'", outside.path().join("shared.txt").display())
            )
            .await,
            0,
            "the clone's commands carry grants added through the original"
        );
    }

    /// A subagent's derived grants follow its parent's readable list live:
    /// a readable grant added to the parent after the subagent launches is
    /// readable inside it; a writable grant on the parent never becomes
    /// writable in the subagent (it lands on the subagent's readable side).
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagent_sees_runtime_readable_but_never_writable_parent_grants() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let parent = authority_terminal(dir.path()).await;
        let readable_dir = tempfile::tempdir().expect("readable dir");
        let writable_dir = tempfile::tempdir().expect("writable dir");
        std::fs::write(readable_dir.path().join("r.txt"), "data").unwrap();
        std::fs::write(writable_dir.path().join("w.txt"), "data").unwrap();

        let child = launch(&parent, SubagentGrant::new())
            .await
            .expect("subagent launches");
        parent.grants().add_readable(readable_dir.path());
        parent.grants().add_writable(writable_dir.path());

        assert_eq!(
            exit_code_of(
                &child,
                &format!("cat '{}'", readable_dir.path().join("r.txt").display())
            )
            .await,
            0,
            "a runtime readable grant on the parent reaches the subagent"
        );
        let denied_dir = tempfile::tempdir().expect("denied dir");
        std::fs::write(denied_dir.path().join("d.txt"), "data").unwrap();
        parent
            .grants()
            .deny_read(std::fs::canonicalize(denied_dir.path()).unwrap());
        assert_ne!(
            exit_code_of(
                &child,
                &format!("cat '{}'", denied_dir.path().join("d.txt").display())
            )
            .await,
            0,
            "a read denial added to the parent after launch reaches the subagent"
        );
        assert_ne!(
            exit_code_of(
                &child,
                &format!("touch '{}'", writable_dir.path().join("x.txt").display())
            )
            .await,
            0,
            "a parent writable grant is never writable inside the subagent"
        );
    }

    /// A shell launched before the grant keeps the grants it started
    /// with — its sandbox was fixed when it spawned — while every new
    /// command picks the grant up (`add_writable` documents this).
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_running_shell_keeps_the_grants_it_started_with() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool = authority_terminal(dir.path()).await;
        let outside = tempfile::tempdir().expect("outside dir should be created");
        let shell_target = outside.path().join("from_shell.txt");

        // A shell session launched before the grant: its write lands after
        // the grant was added, but inside its launch-time sandbox.
        let task_id = start_background_job(
            &tool,
            &format!("sleep 2; touch '{}'", shell_target.display()),
        )
        .await;
        let receiver = tool.background_receiver();
        tool.grants().add_writable(outside.path());
        let fresh_target = outside.path().join("fresh.txt");
        assert_eq!(
            exit_code_of(&tool, &format!("touch '{}'", fresh_target.display())).await,
            0,
            "a fresh command picks up the grant"
        );
        let _ = wait_for_completed_task(&receiver, &task_id).await;
        assert!(
            !shell_target.exists(),
            "the shell launched before the grant is still sandboxed without it"
        );
    }
}
