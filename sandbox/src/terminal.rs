//! Terminal command tool implementation with heel sandboxing.
//!
//! Provides the main `terminal` tool that LLMs use to execute terminal work.
//! Despite the historical tool name, execution does not have to go through a
//! literal `terminal` process: simple commands may be executed directly, while shell
//! syntax falls back to a real shell when needed.
//! Commands run in a sandbox with configurable permission modes.
//!
//! Each `TerminalTool` works in one shared working directory. All terminal
//! executions share this directory, but each execution gets a fresh sandbox
//! (new TTY/process). The ids the model addresses — task ids, stored output
//! names, subagent spaces — are leased from the terminal's
//! [`AnyIdAllocator`](aither_core::AnyIdAllocator).

use crate::foreground_wait::ForegroundWait;
use crate::heel_binary::SharedHeelBinary;
use crate::memory::{
    AnyContainerMemory, AnyFootprintReader, AnyFootprintRegistry, FootprintReader,
    FootprintRegistry, JobMemory, ProbedContainerMemory, ResponseWindow, SystemFootprints,
};
use crate::network_policy::CompiledNetworkPolicy;
use crate::outage::Outages;
use std::{
    borrow::Cow,
    future::Future,
    io::Read,
    net::{SocketAddr, TcpListener as StdTcpListener, TcpStream as StdTcpStream},
    path::{Path, PathBuf},
    pin::Pin,
    process::Stdio,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use aither_core::llm::tool::{
    AnyApprover, Approval, ApprovalError, ApprovalKind, ApprovalRequest, Approver,
};
use aither_core::llm::{CallIdentity, ToolContext};
use aither_core::llm::{
    CommandOutcome, CommandResult, CommandStream, Interruption, IntoToolResult, StoredFile, Tool,
    ToolResult,
};
use aither_core::{AnyIdAllocator, IdLease, IdsExhausted};
use askama::Template;
use async_channel::{Receiver, Sender};
use async_io::Async;
use executor_core::{Executor, Task};
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
#[cfg(target_os = "linux")]
use heel::supervisor::{TreeControl, TreeReport};
use heel::{
    Access, CpuPriority, Grant, IoPriority, IpcRouter, NetworkAuditLog, ProcessPriority, WorkingDir,
};
/// On platforms without `heel supervise` a wait produces no report.
#[cfg(not(target_os = "linux"))]
pub struct TreeReport;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use shell_words::split as split_shell_words;
use tracing::{debug, info, warn};

#[cfg(all(unix, feature = "jobserver"))]
use crate::jobserver::{JobBroker, JobPool, job_broker_channel};
use crate::{
    Carry, ContainerPath, MountRoot, PreparedTools, SandboxMode, ScriptDecision, ScriptDir,
    ScriptOrigin, ScriptOutcome, ScriptRefusal, ToolDistribution, ToolRuntimeError,
    builtin::builtin_router,
    command::{
        AnyCommandGate, AnyCommandObserver, CommandBlocked, CommandRole, GateDecision, ToolRegistry,
    },
    interpreter::Interpreter,
    job_registry::{JobRegistry, JobStatus, job_registry_channel},
    limits::{LiveDuration, ShellLimits},
    output::{MediaResolution, OutputFormat, OutputStore, stored_path},
    output_policy::OutputPolicy,
    process_tree::{JobProcess, JobTree},
    shell_session::{
        ContainerExecOutcome, ContainerShellRuntime, ExecLifetime, ShellBackend,
        ShellRuntimeAvailability, ShellSessionRegistry,
    },
    spawn::stdio::{read_chunks, write_chunks},
    spawn::{
        AnyDomainApprover, DomainApprover, EnvSource, GrantSnapshot, Grants, LocalSpawn, Pipe,
        Sandbox, SpawnSpec, Stdio as SpawnStdio, TreeLifetime,
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
    wording::AnyShellWording,
};
use crate::events::{
    EventSender, JobEnded, JobStarted, LongRunning, ShellEvent, ShellEvents, StdinBlocked,
    StdinWatchFailed, event_channel,
};

mod subagent_git;

pub fn erase_terminal_tool<T>(tool: T) -> crate::command::DynTerminalToolEntry
where
    T: Tool + 'static,
{
    use crate::command::{DynTerminalToolEntry, DynToolHandler};
    use aither_core::llm::tool::ToolDefinition;

    let definition = ToolDefinition::new(&tool);
    let tool = Arc::new(tool);
    let handler: DynToolHandler = Arc::new(move |args: &str, cx: ToolContext| {
        let tool = tool.clone();
        let args = args.to_string();
        Box::pin(async move {
            let parsed = match serde_json::from_str::<T::Arguments>(&args) {
                Ok(parsed) => parsed,
                Err(error) => return ToolResult::error(format!("Parse error: {error}")),
            };
            match tool.call(parsed, cx).await {
                Ok(output) => output
                    .into_tool_result()
                    .unwrap_or_else(|error| ToolResult::error(format!("Error: {error}"))),
                Err(error) => ToolResult::error(format!("Error: {error}")),
            }
        })
    });

    DynTerminalToolEntry {
        definition,
        handler,
    }
}

/// What a terminal's sandboxed network connections are checked against: the
/// lists, the approver for what they leave open, and the audit log.
#[derive(Clone, Default)]
struct NetworkSettings {
    lists: Option<CompiledNetworkPolicy>,
    approver: Option<AnyDomainApprover>,
    audit: Option<NetworkAuditLog>,
}

/// Opens a daily-rotated network audit log under `dir`, creating the directory if needed.
///
/// Keeps at most `max_files` rotated files. Attach the log with
/// [`TerminalTool::with_network_audit_log`] to record every sandboxed
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
/// [`SandboxNotices::command_killed`](crate::SandboxNotices::command_killed)
/// renders the sentence the model reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeoutKill {
    /// The `timeout` the command was bounded by.
    pub timeout: Duration,
    /// When the command started.
    pub started: jiff::Timestamp,
    /// When the kill was issued.
    pub killed: jiff::Timestamp,
}

/// What the kill watchdog of one job records: when the job started, and when
/// the watchdog issued the kill, if it did.
#[derive(Debug)]
struct TimeoutWatch {
    timeout: Duration,
    started: jiff::Timestamp,
    killed: std::sync::OnceLock<jiff::Timestamp>,
}

impl TimeoutWatch {
    fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            started: jiff::Timestamp::now(),
            killed: std::sync::OnceLock::new(),
        }
    }

    /// The kill, once the watchdog has issued one.
    fn kill(&self) -> Option<TimeoutKill> {
        self.killed.get().map(|killed| TimeoutKill {
            timeout: self.timeout,
            started: self.started,
            killed: *killed,
        })
    }
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

/// Execute terminal commands in a sandboxed environment.
///
/// The primary interface to all system capabilities. Scripts run in a sandbox
/// with full read access to the host filesystem but writes contained to a
/// dedicated working directory.
///
/// ## Output Handling
///
/// By default, stdout is **compressed** before being returned. Compression
/// is semantics-preserving: the meaning is identical to the raw output, but
/// the representation may differ. Specifically:
///
/// - **JSON** outputs (arrays of objects or single objects) are automatically
///   converted to **TSV** with dot-notation flattened column headers, but
///   only when the TSV is smaller than the original JSON.
/// - **Source code** outputs (detected via content analysis) are folded
///   using syntax-aware block folding with line numbers. The full raw
///   output is saved to a file whose URL is included in the result.
/// - **Empty lines** and **invisible/control characters** are stripped.
///
/// If you need the exact verbatim output (e.g., for checksums, binary
/// protocols, or diffing), either set `raw: true`, pipe through further
/// processing in the script, or redirect to a file within the script.
///
/// Large outputs are automatically saved to file to manage context. When this
/// happens, you receive the file path and can process it using standard Unix
/// tools (head, tail, grep, less) or pipe through `ask` for summarization.
///
/// ## Built-in Commands
///
/// IPC-backed commands are configured by the host application and are exposed
/// inside the terminal runtime as ordinary CLI commands. Inspect the
/// application-provided command list and each command's `--help` output instead
/// of assuming a fixed built-in command inventory.
///
/// ## Execution Modes
///
/// Each call selects its own runtime mode and starts a fresh shell; only a
/// successful `cd` carries over to the next call. `timeout` is required:
/// it is the job's maximum runtime in seconds — the job and its process
/// tree are killed when it passes — and it may be at most 120 minutes.
///
/// By default the call waits in the foreground for up to 10 seconds. A
/// command still running then **promotes to a background task**: the call
/// returns the task id and the output captured so far, and the job keeps
/// running under its `timeout` bound. Manage it with `terminal_read`
/// (incremental output), `terminal_input` (stdin) and `terminal_kill`.
/// Set `background: true` to detach immediately instead of waiting.
///
/// `language` is `shell` by default; `python` runs `script` as Python source,
/// without a shell wrapper.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ShellArgs {
    /// Short human-readable explanation of what this command is doing.
    /// This is shown in the UI instead of echoing the shell source.
    pub description: String,

    /// The terminal command payload to execute.
    pub script: String,

    /// The language of `script`.
    /// - "shell" (default): shell source
    /// - "python": Python source, run from a file so tracebacks carry line numbers
    #[serde(default)]
    pub interpreter: Interpreter,

    /// How the command runs.
    /// - "sandboxed" (default): inside the sandbox
    /// - "unsandboxed": without the sandbox, after approval
    #[serde(default)]
    pub sandbox: SandboxMode,

    /// Where the command runs: omitted or empty for the home environment,
    /// otherwise the name of a configured SSH server.
    #[serde(default)]
    pub location: Option<String>,

    /// Requested delivery resolution for media returned to the model.
    /// - "auto" (default): preserve the source resolution
    /// - "low": downscale to a 512px bounding box
    /// - "medium": downscale to a 1024px bounding box
    /// - "high": downscale to a 2048px bounding box
    /// - "native": preserve original media bytes without resizing
    #[serde(default)]
    pub resolution: MediaResolution,

    /// Maximum number of output lines to include inline.
    /// For foreground executions, this is the inline line budget
    /// before output offloads to file.
    /// For executions detached with `background: true` or promoted past
    /// the foreground window, only the first `max_lines` are returned
    /// immediately, while full output is redirected to file.
    /// Clamped to the ceiling of the output policy; omitted, the policy's
    /// default.
    #[serde(default)]
    pub max_lines: Option<u32>,

    /// When true, skip all output compression and return the verbatim
    /// stdout bytes. Use this when you need exact byte-level fidelity
    /// (checksums, binary protocols, diff inputs). Default: false.
    #[serde(default)]
    pub raw: bool,

    /// Maximum runtime of the command in seconds. Required. The command and
    /// its process tree are killed when it passes; the value may be at most
    /// 7200 (120 minutes).
    pub timeout: u64,

    /// Start the command detached and return immediately with its task id
    /// instead of waiting in the foreground. Default: false — the call
    /// waits up to the foreground window and promotes a still-running
    /// command to a background task.
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

/// The most processes a [`DescendantsKilled`] alert names.
const KILLED_PROCESSES_NAMED: usize = 8;

/// Processes a foreground command left running when its shell exited, killed
/// with it.
///
/// Sent once per command, at the moment they were killed.
#[derive(Debug, Clone)]
pub struct DescendantsKilled {
    /// The first processes killed, at most eight.
    pub processes: Vec<JobProcess>,
    /// How many more were killed beyond those named.
    pub omitted: usize,
}

impl DescendantsKilled {
    fn new(mut killed: Vec<JobProcess>) -> Self {
        let omitted = killed.len().saturating_sub(KILLED_PROCESSES_NAMED);
        killed.truncate(KILLED_PROCESSES_NAMED);
        Self {
            processes: killed,
            omitted,
        }
    }
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

impl From<CompletedCommand> for CommandResult {
    fn from(command: CompletedCommand) -> Self {
        Self {
            stdout: command.stdout,
            stderr: command.stderr,
            outcome: CommandOutcome::Exited {
                code: command.exit_code,
            },
            interruptions: command.interruptions,
        }
    }
}

#[derive(Clone)]
struct BackgroundStartup {
    tx: async_channel::Sender<Result<(), String>>,
    reported: Arc<AtomicBool>,
}

impl BackgroundStartup {
    fn new(tx: async_channel::Sender<Result<(), String>>) -> Self {
        Self {
            tx,
            reported: Arc::new(AtomicBool::new(false)),
        }
    }

    async fn report_ready(&self) {
        if self
            .reported
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.tx.send(Ok(())).await;
        }
    }

    async fn report_failure(&self, error: &TerminalError) {
        if self
            .reported
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.tx.send(Err(error.to_string())).await;
        }
    }
}

/// Launches subagents from one terminal: every subagent it creates runs with
/// at most that terminal's authority.
///
/// Each terminal that serves a registry owns one, handed to its terminal-bound
/// commands through its [`TerminalBinding`].
#[derive(Clone, Debug)]
pub struct TerminalToolFactory {
    tx: Sender<TerminalToolRequest>,
}

/// One request to a terminal's launcher: the grant the subagent launches with
/// and where the finished tool goes.
#[derive(Debug)]
struct TerminalToolRequest {
    grant: SubagentGrant,
    observer: Option<AnyCommandObserver>,
    gate: Option<AnyCommandGate>,
    caller: Option<PathBuf>,
    reply: Sender<Result<crate::command::DynTerminalTool, SubagentGrantError>>,
}

/// Errors that can occur when requesting a subagent terminal tool.
#[derive(Debug, thiserror::Error)]
pub enum TerminalToolFactoryError {
    /// The launching terminal is gone.
    #[error("terminal tool factory is not available")]
    Unavailable,
    /// The factory failed to return a tool.
    #[error("terminal tool factory failed to return a tool")]
    NoResponse,
    /// The subagent's grant cannot be given.
    #[error(transparent)]
    Grant(#[from] SubagentGrantError),
}

impl TerminalToolFactory {
    /// Requests a new subagent terminal tool from the terminal this launcher
    /// belongs to.
    ///
    /// The tool serves the subagent registry and runs in a space of its own
    /// inside the launching terminal's working directory; `grant` is
    /// everything it may write beyond that space, and must lie inside what the
    /// launching terminal can write. The commands it calls are told to
    /// `observer`, and every shell command it runs is put to `gate`. When the
    /// launching terminal has [`SubagentSpaces`], they prepare the space
    /// first and are told `caller`, the directory the launch is asked for
    /// from (the launching terminal's working directory when `None`).
    ///
    /// # Errors
    /// Returns an error when the launching terminal is gone or does not
    /// respond, when the grant cannot be given, or when the space cannot be
    /// prepared.
    pub async fn create(
        &self,
        grant: SubagentGrant,
        observer: Option<AnyCommandObserver>,
        gate: Option<AnyCommandGate>,
        caller: Option<&Path>,
    ) -> Result<crate::command::DynTerminalTool, TerminalToolFactoryError> {
        let (reply, reply_rx) = async_channel::bounded(1);
        self.tx
            .send(TerminalToolRequest {
                grant,
                observer,
                gate,
                caller: caller.map(Path::to_path_buf),
                reply,
            })
            .await
            .map_err(|_| TerminalToolFactoryError::Unavailable)?;
        Ok(reply_rx
            .recv()
            .await
            .map_err(|_| TerminalToolFactoryError::NoResponse)??)
    }
}

/// Serves a terminal's launch requests: their grants are resolved one at a
/// time, so the grants handed out from it are recorded in one place, and
/// each launch then goes on, its space prepared, beside the others.
///
/// `source` is the terminal with its registry not yet bound: it holds no
/// handle to its own launcher, so the loop ends once every terminal-bound
/// command holding the launcher is gone.
async fn serve_launches<E>(
    requests: Receiver<TerminalToolRequest>,
    source: TerminalTool<E, Launching>,
) where
    E: Executor + Clone + 'static,
{
    let mut history = LaunchHistory::default();
    while let Ok(TerminalToolRequest {
        grant,
        observer,
        gate,
        caller,
        reply,
    }) = requests.recv().await
    {
        let resolved = match source.resolve_launch(&mut history, grant).await {
            Ok(resolved) => resolved,
            Err(error) => {
                if reply.send(Err(error)).await.is_err() {
                    warn!("terminal tool factory response channel dropped");
                }
                continue;
            }
        };
        let source = source.clone();
        source
            .executor
            .clone()
            .spawn(async move {
                let tool = source
                    .launch_resolved(resolved, observer, gate, caller.as_deref())
                    .await
                    .map(TerminalTool::to_dyn);
                if reply.send(tool).await.is_err() {
                    warn!("terminal tool factory response channel dropped");
                }
            })
            .detach();
    }
}

/// What a terminal hands the commands bound to it: where it works, the
/// launcher for the subagents it starts and the runner for the scripts it
/// runs for them.
#[derive(Clone, Debug)]
pub struct TerminalBinding {
    launcher: TerminalToolFactory,
    runner: TerminalRunner,
    working_dir: PathBuf,
    role: CommandRole,
    observer: Option<AnyCommandObserver>,
    executor: Arc<executor_core::AnyExecutor>,
    ids: AnyIdAllocator,
}

impl TerminalBinding {
    /// The allocator the terminal leases its model-facing ids from, for a
    /// command bound to it that names things for the model too.
    #[must_use]
    pub const fn id_allocator(&self) -> &AnyIdAllocator {
        &self.ids
    }

    /// The executor the terminal runs its tasks on, for work that must
    /// outlive the call that started it.
    #[must_use]
    pub fn executor(&self) -> &executor_core::AnyExecutor {
        &self.executor
    }

    /// Launches subagents from this terminal, within its authority.
    #[must_use]
    pub const fn launcher(&self) -> &TerminalToolFactory {
        &self.launcher
    }

    /// Runs scripts on this terminal, under its policy.
    #[must_use]
    pub const fn runner(&self) -> &TerminalRunner {
        &self.runner
    }

    /// The terminal's working directory: the main agent's sandbox, or a
    /// subagent's own space.
    #[must_use]
    pub fn working_dir(&self) -> &Path {
        &self.working_dir
    }

    /// The role the terminal's command registry serves: the main agent's, or
    /// a subagent's at any depth.
    #[must_use]
    pub const fn role(&self) -> CommandRole {
        self.role
    }

    /// The observer told about the commands this terminal's registry serves.
    #[must_use]
    pub const fn observer(&self) -> Option<&AnyCommandObserver> {
        self.observer.as_ref()
    }
}

/// A script a [`TerminalRunner`] runs to its end.
#[derive(Debug, Clone)]
pub struct TerminalScript {
    script: String,
    working_dir: PathBuf,
    timeout: Duration,
    mode: SandboxMode,
}

impl TerminalScript {
    /// `script`, started in `working_dir` on the host and killed with its
    /// process tree once `timeout` passes, in [`SandboxMode::Sandboxed`].
    /// For a container terminal, `working_dir` lies in its workspace mount.
    #[must_use]
    pub fn new(
        script: impl Into<String>,
        working_dir: impl Into<PathBuf>,
        timeout: Duration,
    ) -> Self {
        Self {
            script: script.into(),
            working_dir: working_dir.into(),
            timeout,
            mode: SandboxMode::Sandboxed,
        }
    }

    /// Runs the script in `mode`; an unsafe one is put to the terminal's
    /// permission handler like any unsafe command.
    #[must_use]
    pub const fn with_mode(mut self, mode: SandboxMode) -> Self {
        self.mode = mode;
        self
    }
}

/// Why a [`TerminalRunner`] run did not run to an end.
#[derive(Debug, thiserror::Error)]
pub enum TerminalRunError {
    /// The terminal that served the command is gone.
    #[error("the terminal serving this command is gone")]
    TerminalGone,
    /// The timeout is zero or above the terminal's cap.
    #[error("timeout must be between 1 and {} seconds", .cap.as_secs())]
    Timeout {
        /// The longest timeout the terminal accepts.
        cap: Duration,
    },
    /// A command policy refused the script.
    #[error(transparent)]
    Refused(ScriptRefusal),
    /// The terminal's command gate blocked the script.
    #[error(transparent)]
    Blocked(crate::Refused<CommandBlocked>),
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

/// Runs scripts on the terminal that serves a terminal-bound command.
///
/// A run is under that terminal's whole policy: its grants, network policy,
/// backend (local sandbox or container), command gate, permission handler,
/// process priority, environment and jobserver. Each run is a job of the
/// terminal, listed and killable like any other, but never announced as a
/// background task: its end is told to the caller alone.
///
/// Cloning shares the terminal.
#[derive(Clone)]
pub struct TerminalRunner(Arc<dyn TerminalRunnerImpl>);

impl TerminalRunner {
    /// Runs `script` to its end, through the foreground window and past it,
    /// and returns its exit status and output; a run its timeout killed ends
    /// [`CommandOutcome::Killed`].
    ///
    /// # Errors
    /// Returns an error when the terminal is gone, the timeout is out of
    /// range, the terminal's policy refuses the script, the working
    /// directory does not map into the terminal's container, or the
    /// execution fails.
    pub async fn run(&self, script: TerminalScript) -> Result<CommandResult, TerminalRunError> {
        self.0.run(script).await
    }
}

impl std::fmt::Debug for TerminalRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerminalRunner").finish_non_exhaustive()
    }
}

/// The object-safe twin behind [`TerminalRunner`].
trait TerminalRunnerImpl: Send + Sync {
    fn run(
        &self,
        script: TerminalScript,
    ) -> Pin<Box<dyn Future<Output = Result<CommandResult, TerminalRunError>> + Send + '_>>;
}

/// A terminal's runner: the terminal, as configured when its registry was
/// bound, and that bound registry.
///
/// The registry is held weakly: it holds the commands, which hold this
/// runner, so the terminal's copy here holds no registry of its own and the
/// runner ends with the terminal.
struct BoundRunner<E> {
    terminal: TerminalTool<E, Unconfigured>,
    registry: std::sync::Weak<ToolRegistry>,
}

impl<E> TerminalRunnerImpl for BoundRunner<E>
where
    E: Executor + Clone + 'static,
{
    fn run(
        &self,
        script: TerminalScript,
    ) -> Pin<Box<dyn Future<Output = Result<CommandResult, TerminalRunError>> + Send + '_>> {
        Box::pin(async move {
            let registry = self
                .registry
                .upgrade()
                .ok_or(TerminalRunError::TerminalGone)?;
            self.terminal
                .clone()
                .map_state(|Unconfigured| Running { registry })
                .run_to_end(script)
                .await
        })
    }
}

/// The registry state of the copy of a terminal one run executes on: the
/// terminal's bound registry, held for the run.
#[derive(Clone, Debug)]
struct Running {
    registry: Arc<ToolRegistry>,
}

impl RegistryState for Running {
    fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }
}

/// The terminal tool for executing scripts in a sandbox.
///
/// Creates a shared working directory that persists across all executions,
/// named by heel with four random words; that name is not one of the leased
/// model-facing ids. Each execution creates a fresh sandbox but shares the
/// working directory.
///
/// The executor type `E` determines how async tasks are spawned for the IPC server.
/// Use `TokioExecutor` when running in a tokio runtime.
/// Marker type for a terminal tool without a configured registry.
#[derive(Clone, Debug)]
pub struct Unconfigured;

/// Marker type for a terminal tool with a configured registry, bound to the
/// terminal.
#[derive(Clone, Debug)]
pub struct Configured {
    registry: Arc<ToolRegistry>,
    binding: TerminalBinding,
}

/// The registry state of the copy of a terminal its launch loop holds: the
/// registry is not bound, so the copy holds no handle to its own launcher.
#[derive(Clone, Debug)]
pub struct Launching {
    registry: Arc<ToolRegistry>,
}

/// A registry state a terminal can launch subagents from.
///
/// Sealed: declared `pub` only so it may bound the `TerminalTool` impl that
/// launches subagents; it lives in this private module and is not
/// re-exported.
pub trait RegistryState: Clone + Send + Sync {
    /// The registry the terminal serves; a subagent's registry derives from it.
    fn registry(&self) -> &Arc<ToolRegistry>;
}

impl RegistryState for Configured {
    fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }
}

impl RegistryState for Launching {
    fn registry(&self) -> &Arc<ToolRegistry> {
        &self.registry
    }
}

/// What a subagent's terminal holds that the main agent's does not.
#[derive(Clone, Debug)]
struct SubagentAuthority {
    /// Keeps the word naming its space leased while the terminal lives.
    _space_name: IdLease,
    /// The worktrees it may commit in, with their git data; it hands on no
    /// other commit grant.
    committing: Vec<CommittingWorktree>,
    /// Runs git in its sandbox, as it, for the host's view of its commits.
    git: SubagentGit,
    /// Where its commands run when its home backend is a container: its
    /// space and granted paths, mapped through the container's workspace
    /// mount when it was launched.
    container: Option<ContainerConfinement>,
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
    /// script a [`TerminalRunner`] runs.
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
pub struct TerminalTool<E, State = Unconfigured> {
    /// Shared working directory, named by heel with four random words (e.g.,
    /// `amber-forest-thunder-pearl/`) rather than a leased id.
    working_dir: PathBuf,
    /// Runtime registry for container and ssh execution metadata.
    shell_sessions: ShellSessionRegistry,
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
    /// Time bounds applied to spawned executions.
    limits: TerminalLimits,
    /// Number of still-running spawned executions associated with this tool.
    running_tasks: Arc<AtomicUsize>,
    /// Filesystem grants, shared by every clone; commands snapshot them at
    /// launch.
    grants: Grants,
    /// Set on a subagent's terminal, never on the main agent's.
    subagent: Option<SubagentAuthority>,
    /// The working directory the next foreground command starts in; held on
    /// the main agent's terminal only, a subagent's commands always start in
    /// its own space.
    carried_dir: Option<CarriedDir>,
    /// Extra environment variables resolved for each spawned execution.
    env_source: Option<Arc<dyn EnvSource>>,
    /// Rolling audit log recording every sandboxed network policy decision.
    network: NetworkSettings,
    /// Scheduling priority applied to every sandboxed execution.
    priority: ProcessPriority,
    /// Shared jobserver token broker, when configured.
    #[cfg(all(unix, feature = "jobserver"))]
    jobserver: Option<JobBroker>,
    /// The verified `heel` binary every sandbox this tool creates carries,
    /// shared by its clones and its subagents' tools.
    heel: SharedHeelBinary,
    /// Records the outages commands run through.
    outages: Outages,
    /// Prepares the space of every subagent launched from this terminal and,
    /// at any depth, from theirs.
    subagent_spaces: Option<AnySubagentSpaces>,
    /// Tool registry state.
    registry: State,
}

impl<E: Clone, State: Clone> Clone for TerminalTool<E, State> {
    fn clone(&self) -> Self {
        Self {
            working_dir: self.working_dir.clone(),
            shell_sessions: self.shell_sessions.clone(),
            approver: self.approver.clone(),
            executor: self.executor.clone(),
            ids: self.ids.clone(),
            output_store: self.output_store.clone(),
            job_registry: self.job_registry.clone(),
            events_tx: self.events_tx.clone(),
            events_rx: self.events_rx.clone(),
            limits: self.limits.clone(),
            running_tasks: self.running_tasks.clone(),
            grants: self.grants.clone(),
            subagent: self.subagent.clone(),
            carried_dir: self.carried_dir.clone(),
            env_source: self.env_source.clone(),
            network: self.network.clone(),
            priority: self.priority,
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: self.jobserver.clone(),
            heel: self.heel.clone(),
            outages: self.outages.clone(),
            subagent_spaces: self.subagent_spaces.clone(),
            registry: self.registry.clone(),
        }
    }
}

impl<E: Executor + Clone + 'static> TerminalTool<E, Unconfigured> {
    /// Sets the approver that answers the approvals of scripts no tool call
    /// carries ([`TerminalRunner::run`] and [`Self::run_background`]), and
    /// of calls whose dispatcher attached none. Without one, such a script
    /// that needs approval does not run.
    #[must_use]
    pub fn with_approver(mut self, approver: impl Approver) -> Self {
        self.approver = Some(AnyApprover::new(approver));
        self
    }

    /// Injects the runtime configuration used by `terminal` execution mode resolution.
    ///
    /// The registry is configuration to prepare with [`Self::initialize_tools`];
    /// a registry that is already prepared is installed with
    /// [`Self::adopt_prepared_tools`], which checks it against this terminal.
    ///
    /// # Panics
    /// Panics if `sessions` is already prepared.
    #[must_use]
    pub fn with_shell_sessions(mut self, sessions: ShellSessionRegistry) -> Self {
        assert!(
            sessions.prepared_tools.is_none(),
            "a prepared shell session registry is installed with adopt_prepared_tools"
        );
        self.shell_sessions = sessions;
        self
    }

    /// Sets dynamic runtime availability for terminal execution.
    #[must_use]
    pub fn with_shell_runtime_availability(
        mut self,
        availability: crate::shell_session::ShellRuntimeAvailability,
    ) -> Self {
        self.shell_sessions = self.shell_sessions.with_availability(availability);
        self
    }

    /// Adds container runtime metadata for container-backed terminal execution.
    #[must_use]
    pub fn with_container_runtime(mut self, runtime: ContainerShellRuntime) -> Self {
        self.shell_sessions = self.shell_sessions.with_container_runtime(runtime);
        self
    }

    /// Records every sandboxed network policy decision — allowed or denied —
    /// into the given rolling audit log.
    #[must_use]
    pub fn with_network_audit_log(mut self, log: NetworkAuditLog) -> Self {
        self.network.audit = Some(log);
        self
    }

    /// Sets the network lists a sandboxed connection is checked against
    /// first: a host they allow or deny is decided without the approver.
    #[must_use]
    pub fn with_network_policy(mut self, policy: CompiledNetworkPolicy) -> Self {
        self.network.lists = Some(policy);
        self
    }

    /// Decides the connections the network lists leave open. Without an
    /// approver every such connection is denied.
    #[must_use]
    pub fn with_domain_approver(mut self, approver: impl DomainApprover + 'static) -> Self {
        self.network.approver = Some(AnyDomainApprover::new(approver));
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

    /// Creates a new terminal tool with an executor.
    ///
    /// Creates a working directory under the specified parent directory, which
    /// heel names with four random words (not a leased id). The executor is
    /// used to spawn async tasks for the IPC server.
    ///
    /// # Errors
    /// Returns an error when the working directory or output store cannot be created.
    pub async fn new_in(
        parent_dir: impl AsRef<std::path::Path>,
        executor: E,
        limits: ShellLimits,
        output: OutputPolicy,
        wording: AnyShellWording,
    ) -> Result<Self, TerminalError> {
        let parent_dir = parent_dir.as_ref();

        // Ensure parent directory exists
        async_fs::create_dir_all(parent_dir).await?;

        // heel names the working directory with four random words; it is
        // not a leased model-facing id.
        let working_dir = WorkingDir::random_in(parent_dir).map_err(|e| {
            TerminalError::SandboxSetup(format!("failed to create working dir: {e}"))
        })?;
        // Resolved once, as in `new_exact`.
        let working_dir_path =
            dunce::simplified(&async_fs::canonicalize(working_dir.path()).await?).to_path_buf();

        info!(working_dir = %working_dir_path.display(), "created terminal tool working directory");

        // Create outputs directory inside working dir
        let ids = AnyIdAllocator::default();
        let output_store = OutputStore::new(
            working_dir_path.join("outputs"),
            ids.clone(),
            output,
            wording,
        )
        .await?;

        // Start job registry service
        let (job_registry, job_registry_service) =
            job_registry_channel(output_store.policy().clone());
        executor
            .spawn(async move { job_registry_service.serve().await })
            .detach();

        // Create channel for background task completion (unbounded to not block spawned tasks)
        let (events_tx, events_rx) = event_channel();
        let running_tasks = Arc::new(AtomicUsize::new(0));

        Ok(Self {
            working_dir: working_dir_path,
            shell_sessions: ShellSessionRegistry::new(ShellRuntimeAvailability::default()),
            approver: None,
            executor,
            ids,
            output_store,
            job_registry,
            events_tx,
            events_rx,
            limits: TerminalLimits::new(limits),
            running_tasks,
            grants: Grants::new(),
            subagent: None,
            carried_dir: Some(CarriedDir::default()),
            env_source: None,
            network: NetworkSettings::default(),
            priority: ProcessPriority::default(),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: None,
            heel: SharedHeelBinary::default(),
            outages: Outages::none(),
            subagent_spaces: None,
            registry: Unconfigured,
        })
    }

    /// Creates a new terminal tool with an executor,
    /// using the provided directory directly as the working directory.
    ///
    /// Unlike `new_in` which creates a random subdirectory, this method
    /// uses the exact path provided. Use this when you want explicit control
    /// over the working directory location.
    ///
    /// # Errors
    /// Returns an error when the working directory or output store cannot be created.
    pub async fn new_exact(
        working_dir: impl AsRef<std::path::Path>,
        executor: E,
        limits: ShellLimits,
        output: OutputPolicy,
        wording: AnyShellWording,
    ) -> Result<Self, TerminalError> {
        // Ensure working directory exists, then resolve it once: heel and the
        // directory report both use this spelling, and a symlink in the given
        // one is not necessarily granted inside the sandbox.
        async_fs::create_dir_all(working_dir.as_ref()).await?;
        let working_dir_path =
            dunce::simplified(&async_fs::canonicalize(working_dir.as_ref()).await?).to_path_buf();

        // Create outputs directory inside working dir
        let ids = AnyIdAllocator::default();
        let output_store = OutputStore::new(
            working_dir_path.join("outputs"),
            ids.clone(),
            output,
            wording,
        )
        .await?;

        // Start job registry service
        let (job_registry, job_registry_service) =
            job_registry_channel(output_store.policy().clone());
        executor
            .spawn(async move { job_registry_service.serve().await })
            .detach();

        // Create channel for background task completion (unbounded to not block spawned tasks)
        let (events_tx, events_rx) = event_channel();
        let running_tasks = Arc::new(AtomicUsize::new(0));

        Ok(Self {
            working_dir: working_dir_path,
            shell_sessions: ShellSessionRegistry::new(ShellRuntimeAvailability::default()),
            approver: None,
            executor,
            ids,
            output_store,
            job_registry,
            events_tx,
            events_rx,
            limits: TerminalLimits::new(limits),
            running_tasks,
            grants: Grants::new(),
            subagent: None,
            carried_dir: Some(CarriedDir::default()),
            env_source: None,
            network: NetworkSettings::default(),
            priority: ProcessPriority::default(),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: None,
            heel: SharedHeelBinary::default(),
            outages: Outages::none(),
            subagent_spaces: None,
            registry: Unconfigured,
        })
    }
}

impl<E, State> TerminalTool<E, State> {
    /// Moves every part into a terminal with another registry state.
    fn map_state<Next>(self, state: impl FnOnce(State) -> Next) -> TerminalTool<E, Next> {
        TerminalTool {
            working_dir: self.working_dir,
            shell_sessions: self.shell_sessions,
            approver: self.approver,
            executor: self.executor,
            ids: self.ids,
            output_store: self.output_store,
            job_registry: self.job_registry,
            events_tx: self.events_tx,
            events_rx: self.events_rx,
            limits: self.limits,
            running_tasks: self.running_tasks,
            grants: self.grants,
            subagent: self.subagent,
            carried_dir: self.carried_dir,
            env_source: self.env_source,
            network: self.network,
            priority: self.priority,
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: self.jobserver,
            heel: self.heel,
            outages: self.outages,
            subagent_spaces: self.subagent_spaces,
            registry: state(self.registry),
        }
    }
}

impl<E> TerminalTool<E, Unconfigured>
where
    E: Executor + Clone + 'static,
{
    /// Attaches a tool registry to this terminal tool, enabling IPC command
    /// dispatch.
    ///
    /// The terminal gets its own launcher for the subagents it starts, and
    /// the registry's terminal-bound commands are built for it.
    #[must_use]
    pub fn with_registry(self, registry: Arc<ToolRegistry>) -> TerminalTool<E, Configured> {
        let registry = Arc::new(
            Arc::unwrap_or_clone(registry)
                .with_terminal_controls(self.job_registry(), self.output_store.clone()),
        );
        self.map_state(|Unconfigured| Launching { registry })
            .into_served()
    }
}

#[cfg(test)]
impl<E> TerminalTool<E, Configured>
where
    E: Executor + Clone + 'static,
{
    /// A subagent terminal with an empty grant, told to no observer or gate.
    async fn subagent_without_grant(&self) -> Result<Self, SubagentGrantError> {
        self.subagent(&mut LaunchHistory::default(), SubagentGrant::new(), None)
            .await
    }
}

impl<E> TerminalTool<E, Launching>
where
    E: Executor + Clone + 'static,
{
    /// Starts this terminal's launcher and binds its registry to it and to
    /// its runner.
    fn into_served(self) -> TerminalTool<E, Configured> {
        let (launcher, requests) = async_channel::unbounded();
        // A run starts where it is told and leaves the main agent's carried
        // directory alone.
        let mut runs_on = self.clone().map_state(|_| Unconfigured);
        runs_on.carried_dir = None;
        let mut binding = None;
        let registry = Arc::new_cyclic(|bound| {
            let served = TerminalBinding {
                launcher: TerminalToolFactory { tx: launcher },
                runner: TerminalRunner(Arc::new(BoundRunner {
                    terminal: runs_on,
                    registry: bound.clone(),
                })),
                working_dir: self.working_dir.clone(),
                role: self.registry.registry.role(),
                observer: self.registry.registry.observer().cloned(),
                executor: Arc::new(executor_core::AnyExecutor::new(self.executor.clone())),
                ids: self.ids.clone(),
            };
            let registry = ToolRegistry::clone(&self.registry.registry).bind(&served);
            binding = Some(served);
            registry
        });
        let binding = binding.expect("binding the registry builds the binding");
        let source = self.clone();
        self.executor
            .spawn(async move { serve_launches(requests, source).await })
            .detach();
        self.map_state(|Launching { .. }| Configured { registry, binding })
    }
}

impl<E, State> TerminalTool<E, State>
where
    E: Executor + Clone + 'static,
    State: Clone,
{
    /// Adds additional writable paths to the sandbox configuration.
    ///
    /// These paths will be writable in sandboxed and network modes.
    #[must_use]
    pub fn with_writable_paths(self, paths: impl IntoIterator<Item = impl Into<PathBuf>>) -> Self {
        for path in paths {
            self.grants.add_writable(path.into());
        }
        self
    }

    /// Adds additional readable (but not writable) paths to the sandbox configuration.
    ///
    /// These paths will be readable in all sandbox modes, even in strict
    /// filesystem mode where reads outside the sandbox are normally denied.
    #[must_use]
    pub fn with_readable_paths(self, paths: impl IntoIterator<Item = impl Into<PathBuf>>) -> Self {
        for path in paths {
            self.grants.add_readable(path.into());
        }
        self
    }

    /// Denies sandboxed local commands reading these absolute paths and
    /// everything beneath them, in addition to heel's credential table
    /// (`~/.ssh`, `~/.config/gh`, keychains and other credential stores).
    /// Every other path on the host is readable; writes stay limited to the
    /// granted paths.
    #[must_use]
    pub fn with_read_denied_paths(
        self,
        paths: impl IntoIterator<Item = impl Into<PathBuf>>,
    ) -> Self {
        for path in paths {
            self.grants.deny_read(path.into());
        }
        self
    }

    /// Injects extra environment variables into every command this tool
    /// spawns in the home environment.
    ///
    /// The source is resolved per execution, so values it returns reflect
    /// the moment each command launches. Injection applies to the local and
    /// container backends; SSH targets are deliberately excluded so
    /// host-side values (for example secrets) never reach remote machines.
    /// Caller-provided names override the session's computed environment.
    #[must_use]
    pub fn with_env_source(mut self, source: impl EnvSource + 'static) -> Self {
        self.env_source = Some(Arc::new(source));
        self
    }

    /// Injects a fixed set of environment variables into every spawned
    /// command; a convenience for [`Self::with_env_source`] over static pairs.
    #[must_use]
    pub fn with_envs(self, envs: impl IntoIterator<Item = (String, String)>) -> Self {
        self.with_env_source(envs.into_iter().collect::<Vec<_>>())
    }

    /// Lowers the scheduling priority of the commands this tool runs.
    ///
    /// The priority is applied to every sandbox the tool creates — one-shot
    /// commands, persistent shell sessions and background jobs — and is
    /// inherited by everything they spawn. Container executions run the script
    /// through the image's own `heel run`, so the image needs a `heel` binary
    /// when a non-normal priority is configured. SSH executions are not
    /// affected: the remote host runs its own `heel`, managed outside this
    /// configuration.
    #[must_use]
    pub const fn with_process_priority(mut self, priority: ProcessPriority) -> Self {
        self.priority = priority;
        self
    }

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

    /// Sets how long a `terminal` call waits in the foreground before a
    /// still-running command promotes to a background task. Defaults to
    /// [`DEFAULT_FOREGROUND_WAIT`].
    #[must_use]
    pub fn with_foreground_wait(self, wait: Duration) -> Self {
        self.limits.foreground_wait.set_normal(wait);
        self
    }

    /// A handle on the foreground window, shared with every terminal derived
    /// from this one: a longer window put in force through it applies to the
    /// next foreground wait of each of them.
    #[must_use]
    pub fn foreground_wait(&self) -> ForegroundWait {
        self.limits.foreground_wait.clone()
    }

    /// Sets how long the agent has to act on a final memory warning before
    /// the job it names is killed. Defaults to [`DEFAULT_RESPONSE_WINDOW`].
    #[must_use]
    pub fn with_response_window(self, window: Duration) -> Self {
        self.limits.response_window.set(window);
        self
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
        if self.shell_sessions.resolve_local_backend() != Ok(ShellBackend::Container) {
            return None;
        }
        self.shell_sessions
            .container_runtime()
            .map(ContainerShellRuntime::probe)
    }

    /// Sets the largest `timeout` a command accepts.
    /// Defaults to [`DEFAULT_TIMEOUT_CAP`].
    #[must_use]
    pub fn with_timeout_cap(self, cap: Duration) -> Self {
        self.limits.timeout_cap.set(cap);
        self
    }

    /// Sets the most bytes a subagent's sandbox may send the host while the
    /// commits it made are imported into the main repository; a pack or answer
    /// that exceeds it fails the import. `0` leaves the bytes unbounded.
    /// Defaults to [`DEFAULT_SUBAGENT_IMPORT_LIMIT`]. Each piece of an import's
    /// sandboxed work is bounded in time by the foreground limit.
    ///
    /// The setting can be changed while the terminal and the subagents it
    /// launches run, through [`Self::subagent_import_budget`].
    #[must_use]
    pub fn with_subagent_import_limit(self, bytes: u64) -> Self {
        self.limits.subagent_import.set(bytes);
        self
    }

    /// A handle on the setting [`Self::with_subagent_import_limit`] sets,
    /// shared with every subagent this terminal launches: a change through it
    /// applies to the next import of each of them.
    #[must_use]
    pub fn subagent_import_budget(&self) -> ImportBudget {
        self.limits.subagent_import.clone()
    }

    /// Sets how long a background job may run before it produces one system
    /// alert naming it. Defaults to [`DEFAULT_LONG_JOB_ALERT_AFTER`].
    #[must_use]
    pub fn with_long_job_alert_after(self, after: Duration) -> Self {
        self.limits.long_job_alert.set(after);
        self
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

    /// Initializes tools before agent construction or command execution.
    ///
    /// # Errors
    /// Rejects preparation failures and writable grants that expose managed tools.
    pub async fn initialize_tools(
        mut self,
        distribution: &ToolDistribution,
        cache_root: &Path,
    ) -> Result<Self, ToolRuntimeError>
    where
        State: Send,
    {
        self.shell_sessions = self
            .shell_sessions
            .initialize_tools(distribution, cache_root)
            .await?;
        if let Some(tools) = self.local_home_tools()? {
            tools.ensure_readonly(self.writable_grants().iter()).await?;
            if let Some(pinned) = tools.heel() {
                self.heel.provide(pinned).await?;
            }
            let heel = self.heel.get().await?;
            tools
                .verify_local_versions(self.executor.clone(), heel)
                .await?;
        }
        Ok(self)
    }

    /// Adopts a registry that [`Self::initialize_tools`] already prepared for
    /// another terminal.
    ///
    /// Only the checks that depend on this terminal run again: its working
    /// directory and writable paths must not expose the managed tool
    /// directory. Artifact downloads, integrity hashing and version probes
    /// are not repeated. The registry's SSH servers come with it, shared:
    /// a server already approved or prepared is not asked about or prepared
    /// again.
    ///
    /// # Errors
    /// Rejects a registry that is not prepared, one whose home backend or
    /// container changed after preparation, and writable grants that expose
    /// the managed tools.
    pub async fn adopt_prepared_tools(
        mut self,
        sessions: ShellSessionRegistry,
    ) -> Result<Self, ToolRuntimeError>
    where
        State: Send,
    {
        sessions.prepared_catalog()?;
        self.shell_sessions = sessions;
        if let Some(tools) = self.local_home_tools()? {
            tools.ensure_readonly(self.writable_grants().iter()).await?;
            if let Some(pinned) = tools.heel() {
                self.heel.provide(pinned).await?;
            }
        }
        Ok(self)
    }

    /// Prepared tools of the local home backend, when this terminal runs on it.
    fn local_home_tools(&self) -> Result<Option<&PreparedTools<PathBuf>>, ToolRuntimeError> {
        match self.shell_sessions.resolve_local_backend() {
            Ok(ShellBackend::Local) => self.shell_sessions.local_tools().map(Some),
            _ => Ok(None),
        }
    }

    /// The paths this terminal grants write access to, which must not expose
    /// the managed tools.
    fn writable_grants(&self) -> Vec<PathBuf> {
        let grants = self.grants.snapshot();
        std::iter::once(self.working_dir.clone())
            .chain(grants.writable.iter().cloned())
            .collect()
    }

    /// Initialized session configuration shared with child terminals.
    #[must_use]
    pub const fn shell_sessions(&self) -> &ShellSessionRegistry {
        &self.shell_sessions
    }

    /// Operational context from successfully initialized environments.
    ///
    /// # Errors
    /// Fails when no tools are prepared or a backend configuration has changed.
    pub fn prepared_tools_context(&self) -> Result<String, ToolRuntimeError> {
        self.shell_sessions.prepared_catalog()?.render()
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

    /// Returns the job registry handle.
    pub fn job_registry(&self) -> JobRegistry {
        self.job_registry.clone()
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
        self.grants.clone()
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

impl<E, State> TerminalTool<E, State>
where
    E: Executor + Clone + 'static,
    State: RegistryState,
{
    /// Creates the terminal tool of a subagent this terminal launches, as
    /// [`Self::resolve_launch`] and [`Self::launch_resolved`] do one after
    /// the other.
    #[cfg(test)]
    pub(crate) async fn subagent(
        &self,
        history: &mut LaunchHistory,
        grant: SubagentGrant,
        caller: Option<&Path>,
    ) -> Result<TerminalTool<E, Configured>, SubagentGrantError> {
        let resolved = self.resolve_launch(history, grant).await?;
        self.launch_resolved(resolved, None, None, caller).await
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
        for path in &self.grants.snapshot().writable {
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
        self.registry
            .registry()
            .record_committed(&granted.committing);
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
    /// The subagent shares the permission handler, output store and prepared
    /// tools, serves the subagent registry (so nothing reserved for the main
    /// agent is reachable, however deep subagents nest) with its
    /// terminal-bound commands built for it, and has its own launcher and
    /// background task tracking. It works in a space of its own inside this
    /// terminal's working directory, which this terminal's
    /// [`SubagentSpaces`] fill first; everything this terminal can write is
    /// read-only to it, apart from what the grant hands over. The commands
    /// it calls are told to `observer`, and every shell command it runs is
    /// put to `gate`.
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
        observer: Option<AnyCommandObserver>,
        gate: Option<AnyCommandGate>,
        caller: Option<&Path>,
    ) -> Result<TerminalTool<E, Configured>, SubagentGrantError> {
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

        let (events_tx, events_rx) = event_channel();
        let running_tasks = Arc::new(AtomicUsize::new(0));
        let (job_registry, job_registry_service) =
            job_registry_channel(self.output_store.policy().clone());
        self.executor
            .spawn(async move { job_registry_service.serve().await })
            .detach();
        let grants = self.grants.derived(GrantSnapshot {
            readable: authority,
            writable: granted.writable.clone(),
            ..GrantSnapshot::default()
        });
        let readable_paths = grants.snapshot().readable.clone();
        let git = match (&container, mount) {
            (Some(confinement), Some(mount)) => {
                SubagentGit::new(Arc::new(subagent_git::ContainerGit {
                    executor: self.executor.clone(),
                    shell_sessions: self.shell_sessions.clone(),
                    confinement: confinement.clone(),
                    mount,
                    priority: self.priority,
                }))
            }
            _ => SubagentGit::new(Arc::new(subagent_git::LocalGit {
                executor: self.executor.clone(),
                working_dir: space.clone(),
                writable: granted.writable,
                readable: readable_paths,
                committing: granted.committing.clone(),
                priority: self.priority,
                heel: self.heel.clone(),
            })),
        };
        let registry = Arc::new(
            self.registry
                .registry()
                .for_subagent(observer, gate)
                .with_terminal_controls(job_registry.clone(), self.output_store.clone()),
        );
        let launching = TerminalTool {
            working_dir: space,
            shell_sessions: self.shell_sessions.clone(),
            approver: self.approver.clone(),
            executor: self.executor.clone(),
            ids: self.ids.clone(),
            output_store: self.output_store.clone(),
            job_registry,
            events_tx,
            events_rx,
            limits: self.limits.clone(),
            running_tasks,
            grants,
            subagent: Some(SubagentAuthority {
                _space_name: space_name,
                committing: granted.committing,
                git,
                container,
                prepared,
            }),
            carried_dir: None,
            env_source: self.env_source.clone(),
            network: self.network.clone(),
            priority: self.priority,
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: self.jobserver.clone(),
            heel: self.heel.clone(),
            outages: self.outages.clone(),
            subagent_spaces: self.subagent_spaces.clone(),
            registry: Launching { registry },
        };
        Ok(launching.into_served())
    }

    /// What this terminal's launcher prepared in its space; `None` for the
    /// main agent's terminal and when the launcher has no
    /// [`SubagentSpaces`].
    fn prepared_space(&self) -> Option<PreparedSpace> {
        self.subagent
            .as_ref()
            .and_then(|subagent| subagent.prepared.clone())
    }

    /// The workspace mount of this terminal's home container, with its host
    /// side canonical, spelling every path of a subagent's view inside the
    /// container. `None` outside a container.
    async fn container_mount(&self) -> Result<Option<MountRoot>, SubagentGrantError> {
        if self.shell_sessions.resolve_local_backend() != Ok(ShellBackend::Container) {
            return Ok(None);
        }
        let mount = self
            .shell_sessions
            .container_runtime()
            .and_then(ContainerShellRuntime::workspace_mount)
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
        writable: writable
            .iter()
            .map(|path| map(path))
            .collect::<Result<_, _>>()?,
    })
}

impl<E> TerminalTool<E, Configured>
where
    E: Executor + Clone + 'static,
{
    /// Returns a receiver for the `terminal` calls in which a command
    /// registered with
    /// [`ToolRegistryBuilder::kept_through_compaction`](crate::ToolRegistryBuilder::kept_through_compaction)
    /// succeeded.
    #[must_use]
    pub fn kept_receiver(&self) -> crate::command::KeptExchangeReceiver {
        self.registry().kept_receiver()
    }

    /// What this terminal hands the commands bound to it: its working
    /// directory and the launcher for the subagents it starts.
    #[must_use]
    pub const fn binding(&self) -> &TerminalBinding {
        &self.registry.binding
    }

    /// Lists the commits of the subagent this terminal serves, when it holds a
    /// commit grant.
    fn commit_ledger(&self) -> Option<CommitLedger> {
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

    /// Converts this `TerminalTool` into a type-erased `DynTerminalTool`.
    ///
    /// This is useful for creating child terminal tools for subagents where
    /// the concrete type cannot be known at compile time.
    ///
    /// # Panics
    /// Panics when execution environments have not completed tool initialization.
    pub fn to_dyn(self) -> crate::command::DynTerminalTool {
        use crate::command::DynTerminalTool;

        let events = self.events();
        let kept_receiver = self.kept_receiver();
        let job_registry = self.job_registry();
        let working_dir = self.working_dir().clone();
        let command_names = self.registry().available_tool_names();
        let commit_ledger = self.commit_ledger();
        let prepared_space = self.prepared_space();
        let role = self.registry().role();
        let wording = self.output_store.wording().clone();
        let output = self.output_store.policy().clone();
        let prepared_tools = self
            .shell_sessions
            .prepared_catalog()
            .expect("terminal tools must be initialized before exposing child capabilities")
            .clone();
        let entries = vec![erase_terminal_tool(self)];

        DynTerminalTool {
            entries,
            events,
            kept_receiver,
            job_registry,
            working_dir,
            command_names,
            prepared_tools,
            commit_ledger,
            prepared_space,
            role,
            wording,
            output,
        }
    }
}

/// The execution path every terminal state that serves a registry shares:
/// a `terminal` call and a [`TerminalRunner`] run go through it alike.
impl<E, S> TerminalTool<E, S>
where
    E: Executor + Clone + 'static,
    S: RegistryState,
{
    fn registry(&self) -> &Arc<ToolRegistry> {
        self.registry.registry()
    }

    /// The server `location` names, or `None` for the home environment.
    fn resolve_location(
        &self,
        location: Option<&str>,
    ) -> Result<Option<Arc<SshEnvironment>>, TerminalError> {
        let Some(name) = location.map(str::trim).filter(|name| !name.is_empty()) else {
            return Ok(None);
        };
        self.shell_sessions
            .ssh_environments()
            .get(name)
            .map(Some)
            .ok_or_else(|| TerminalError::UnknownLocation(name.to_string()))
    }

    fn resolve_home_target(&self, mode: SandboxMode) -> Result<ExecutionTarget, TerminalError> {
        let backend = self
            .shell_sessions
            .resolve_local_backend()
            .map_err(TerminalError::InvalidTarget)?;
        if mode == SandboxMode::Unsandboxed && backend != ShellBackend::Local {
            return Err(TerminalError::InvalidTarget(
                "unsafe mode is only available in heel local runtime".into(),
            ));
        }
        let backend = match backend {
            ShellBackend::Local => {
                ExecutionBackend::Local(self.shell_sessions.local_tools()?.clone())
            }
            ShellBackend::Container => ExecutionBackend::Container {
                runtime: self
                    .shell_sessions
                    .container_runtime()
                    .cloned()
                    .ok_or(ToolRuntimeError::MissingContainerRuntime)?,
                tools: self.shell_sessions.container_tools()?.clone(),
                confinement: self
                    .subagent
                    .as_ref()
                    .and_then(|subagent| subagent.container.clone()),
            },
        };
        Ok(ExecutionTarget { mode, backend })
    }

    /// A command on `environment` in `mode`, the server prepared first when
    /// this is its first command. Default mode needs the server's `heel`.
    async fn resolve_ssh_target(
        &self,
        environment: &SshEnvironment,
        mode: SandboxMode,
    ) -> Result<ExecutionTarget, TerminalError> {
        let prepared = self.shell_sessions.ssh_tools(environment).await?;
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
                client: self.shell_sessions.ssh_environments().client().clone(),
                server: server.clone(),
                heel,
                tools: prepared.tools.clone(),
            },
        })
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
            raw,
            background,
            timeout,
            carry,
            call,
        } = request;
        let ExecutionTarget { mode, backend } = target;
        // Every IPC channel this execution opens, local router or container
        // bridge, is served from this one registry, so each command its script
        // runs, a detached job's included, is told the id of this call, and
        // only the commands that answer in its mode are there.
        let registry = self
            .registry()
            .in_mode(mode)
            .for_job(crate::command::JobOutput::new(
                self.job_registry.clone(),
                &task_id,
            ));
        let registry = Arc::new(match call {
            Some(call) => registry.for_call(call),
            None => registry,
        });
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
        let timeout_watch = self.spawn_kill_watchdog(&task_id, timeout);
        self.spawn_long_job_alert(&task_id);

        let lifetime = if background {
            JobLifetime::Background
        } else {
            JobLifetime::Foreground {
                alerts: self.events_tx.clone(),
            }
        };
        let background_startup = BackgroundStartup::new(startup_tx);
        let running_tasks = self.running_tasks.clone();
        let completed_tx = self.events_tx.clone();
        let outages = self.outages.clone();
        outages.begin(&task_id);

        running_tasks.fetch_add(1, Ordering::AcqRel);
        self.executor
            .spawn(execute_terminal_task(SpawnedTerminalTask {
                working_dir: self.working_dir.clone(),
                grants: self.grants.clone(),
                env_source: self.env_source.clone(),
                executor: self.executor.clone(),
                registry,
                job_registry: self.job_registry.clone(),
                task_id,
                execution_id,
                background,
                timeout_watch: timeout_watch.clone(),
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
                raw,
                result_tx,
                taken: taken_rx,
                completed_tx,
                running_tasks,
                network: self.network.clone(),
                priority: self.priority,
                #[cfg(all(unix, feature = "jobserver"))]
                jobserver: self.jobserver.clone(),
                heel: self.heel.clone(),
                outages,
            }))
            .detach();

        SpawnedTerminal {
            result: result_rx,
            startup: startup_rx,
            timeout_watch,
            taken: taken_tx,
        }
    }
}

impl<E: Executor + Clone + 'static> Tool for TerminalTool<E, Configured> {
    fn name(&self) -> Cow<'static, str> {
        "terminal".into()
    }

    type Arguments = ShellArgs;
    type Res = ToolResult;

    async fn call(
        &self,
        arguments: Self::Arguments,
        mut cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let cap = self.limits.timeout_cap.get();
        let timeout = Duration::from_secs(arguments.timeout);
        if arguments.timeout == 0 || timeout > cap {
            return Err(anyhow::anyhow!(
                "terminal timeout must be between 1 and {} seconds",
                cap.as_secs()
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
            raw: arguments.raw,
            background: arguments.background,
            timeout,
            carry: carry.clone(),
            call: cx.identity().cloned(),
        });
        // Registration precedes `report_ready` on every backend, so once
        // startup returns the job is killable through the registry and the
        // foreground clock only measures the command's own runtime.
        wait_for_background_startup(&spawned.startup).await?;
        if arguments.background {
            return background_terminal_result(
                &self.job_registry,
                &prepared.outputs,
                &prepared.task_id,
                prepared.max_lines,
                &self.events_tx,
            )
            .await;
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
                    .get_by_task_id(&prepared.task_id)
                    .await
                    .is_some_and(|job| matches!(job.status, JobStatus::Killed { .. }));
                let command = if killed {
                    CommandResult {
                        stdout: result.stdout,
                        stderr: result.stderr,
                        interruptions: result.interruptions,
                        outcome: CommandOutcome::Killed {
                            task_id: prepared.task_id.to_string(),
                            notice: self.kill_notice(spawned.timeout_watch.kill().as_ref()),
                        },
                    }
                } else {
                    CommandResult::from(result)
                };
                spawned.took_result().await;
                command.into_tool_result()
            }
            Some(Err(err)) => {
                spawned.took_result().await;
                Err(anyhow::anyhow!(err))
            }
            // The command outlived its foreground window: promote it to a
            // background task.
            None => {
                background_terminal_result(
                    &self.job_registry,
                    &prepared.outputs,
                    &prepared.task_id,
                    prepared.max_lines,
                    &self.events_tx,
                )
                .await
            }
        }
    }
}

impl<E: Executor + Clone + 'static> TerminalTool<E, Configured> {
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
            ExecutionBackend::Local(_) if !cfg!(windows) => None,
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
            grants: self.grants.snapshot(),
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
            ExecutionBackend::Local(_) => Self::Host,
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

impl<E: Executor + Clone + 'static> TerminalTool<E, Configured> {
    /// Runs `script` as a detached background job bounded by `timeout`, for
    /// embedders that drive the terminal directly rather than through the
    /// `terminal` tool schema. Shares [`Self::call`]'s `background: true`
    /// path.
    ///
    /// # Errors
    ///
    /// Fails the way a `terminal` call does: an invalid `timeout`, a
    /// refused permission, a blocked command or a spawn failure.
    pub async fn run_background(&self, arguments: ShellArgs) -> aither_core::Result<ToolResult> {
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

impl<E, S> TerminalTool<E, S>
where
    E: Executor + Clone + 'static,
    S: RegistryState,
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
        match self.shell_sessions.resolve_local_backend() {
            Ok(ShellBackend::Container) => container_script_dir(
                self.shell_sessions
                    .container_runtime()
                    .and_then(|runtime| runtime.workspace_mount()),
                host,
                own,
            ),
            _ => ScriptDir::Host(host.unwrap_or_else(|| own.to_path_buf())),
        }
    }

    /// What renders this terminal's refusals and notices: its output store's.
    pub(crate) fn wording(&self) -> &crate::AnyShellWording {
        self.output_store.wording()
    }

    /// Why a foreground command that was killed was killed, as the model
    /// reads it in the command's result.
    fn kill_notice(&self, kill: Option<&TimeoutKill>) -> String {
        let killed = kill.map_or(crate::CommandKilled::BeforeTimeout, |kill| {
            crate::CommandKilled::TimedOut(kill)
        });
        self.wording().command_killed(&killed)
    }

    /// Shared front half of every `terminal` call: argument validation, then
    /// [`Self::admit_command`].
    async fn prepare_execution(
        &self,
        arguments: &ShellArgs,
        cx: &mut ToolContext,
    ) -> anyhow::Result<ExecutionPreparation> {
        if arguments.description.trim().is_empty() {
            return Err(anyhow::anyhow!("terminal description must not be empty"));
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
        .map_err(|error| match error {
            // Kept unwrapped, so the agent loop recognises the refusal.
            TerminalRunError::Refused(refusal) => anyhow::Error::from(refusal),
            other => anyhow::Error::from(other),
        })
    }

    /// The terminal's policy for a command, whoever runs it — the one entry
    /// every foreground, background, subagent and run command on every
    /// backend passes through: the command policies, the command gate,
    /// target resolution, the read-only tools check and permission; then
    /// the spawned job's ids.
    ///
    /// Every script the policies were asked about is reported to them once
    /// it ends; one admission fails is reported here, unrun.
    async fn admit_command(
        &self,
        admission: Admission<'_>,
        cx: Option<&mut ToolContext>,
    ) -> Result<ExecutionPreparation, TerminalRunError> {
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
        let clearance = self
            .check_policies(&task_id, &admission, &start)
            .await
            .map_err(TerminalRunError::Refused);
        let clearance = match clearance {
            Ok(clearance) => clearance,
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
            carried_start,
        })
    }

    /// Tells the policies a script they were asked about never started, and
    /// where the next foreground script starts.
    async fn end_unrun(&self, task_id: &str) {
        let policies = self.registry.registry().policies();
        if policies.is_empty() {
            return;
        }
        let own = self.own_dir().await;
        let end = self.home_script_dir(carried_start(self.carried_dir.as_ref()).await, &own);
        policies
            .ended(task_id, ScriptOutcome::NotRun, &end, Carry::NotCarried)
            .await;
    }

    /// Puts the script to the registry's script policies, the shell parsed
    /// once for all of them, and folds their decisions into what admission
    /// does next.
    async fn check_policies(
        &self,
        task_id: &str,
        admission: &Admission<'_>,
        start: &ScriptDir,
    ) -> Result<Clearance, ScriptRefusal> {
        let policies = self.registry.registry().policies();
        if policies.is_empty() {
            return Ok(Clearance::Pass);
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
        match policies.check(&check).await {
            ScriptDecision::Pass => Ok(Clearance::Pass),
            ScriptDecision::Allow => Ok(Clearance::Allow),
            ScriptDecision::Ask(request) => Ok(Clearance::Ask(request)),
            ScriptDecision::Refuse(refusal) => Err(refusal),
        }
    }

    /// The rest of admission once the policies did not refuse the script.
    async fn admit_checked(
        &self,
        admission: &Admission<'_>,
        environment: Option<&SshEnvironment>,
        clearance: Clearance,
        cx: Option<&mut ToolContext>,
    ) -> Result<(ExecutionTarget, usize), TerminalRunError> {
        // Ahead of target resolution and permission, so a blocked command
        // never asks the user for anything. The decision carries no grant.
        // The gate reads shell syntax, so Python source skips it.
        if admission.interpreter == Interpreter::Shell
            && let Some(gate) = self.registry.registry().gate()
            && let GateDecision::Block { reason } = gate.check(admission.script).await
        {
            return Err(TerminalRunError::Blocked(crate::Refused::rendered(
                CommandBlocked { reason },
                |blocked| self.wording().command_blocked(blocked),
            )));
        }
        let mode = admission.mode;
        let target = match environment {
            None => {
                let target = self.resolve_home_target(mode)?;
                if let ExecutionBackend::Local(tools) = &target.backend {
                    tools
                        .ensure_readonly(
                            std::iter::once(&self.working_dir)
                                .chain(self.grants.snapshot().writable.iter()),
                        )
                        .await
                        .map_err(TerminalError::from)?;
                }
                self.clear(admission, clearance, cx).await?;
                target
            }
            // Boxed: a server's first command carries its preparation,
            // which would otherwise inflate every command's future.
            Some(environment) => {
                Box::pin(self.admit_on_server(environment, admission, clearance, cx)).await?
            }
        };
        Ok((
            target,
            self.output_store.policy().lines(admission.max_lines),
        ))
    }

    /// Admission of a command on a server. The question is put before the
    /// server is prepared, which may take a while on its first command.
    async fn admit_on_server(
        &self,
        environment: &SshEnvironment,
        admission: &Admission<'_>,
        clearance: Clearance,
        cx: Option<&mut ToolContext>,
    ) -> Result<ExecutionTarget, TerminalError> {
        self.clear(admission, clearance, cx).await?;
        self.resolve_ssh_target(environment, admission.mode).await
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

    /// Runs `run` to its end on this terminal; see [`TerminalRunner::run`].
    async fn run_to_end(&self, run: TerminalScript) -> Result<CommandResult, TerminalRunError> {
        let cap = self.limits.timeout_cap.get();
        if run.timeout.is_zero() || run.timeout > cap {
            return Err(TerminalRunError::Timeout { cap });
        }
        // A run starts in a host directory, so it resolves the terminal's
        // home backend, never an SSH server.
        let prepared = self
            .admit_command(
                Admission {
                    script: &run.script,
                    interpreter: Interpreter::Shell,
                    mode: run.mode,
                    location: None,
                    max_lines: None,
                    origin: ScriptOrigin::Run,
                    call: None,
                    run_dir: Some(&run.working_dir),
                },
                None,
            )
            .await?;
        let start = match run_start(&prepared.target, run.working_dir) {
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
            script: run.script,
            interpreter: Interpreter::Shell,
            expect: OutputFormat::Text,
            resolution: MediaResolution::Auto,
            max_lines: prepared.max_lines,
            raw: false,
            background: false,
            timeout: run.timeout,
            carry: Some(start),
            call: None,
        });
        match spawned.startup.recv().await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => return Err(TerminalError::Execution(error).into()),
            Err(_) => {
                return Err(TerminalError::Execution(
                    "the run ended before it was registered".to_string(),
                )
                .into());
            }
        }
        // The run is never announced as a background task, so its end is
        // told to the caller alone, however long it takes.
        let result =
            spawned.result.recv().await.map_err(|_| {
                TerminalError::Execution("the run ended without a result".to_string())
            });
        let killed = self
            .job_registry
            .get_by_task_id_settled(&prepared.task_id)
            .await
            .is_some_and(|job| matches!(job.status, JobStatus::Killed { .. }));
        let timeout_kill = spawned.timeout_watch.kill();
        spawned.took_result().await;
        let result = result?.map_err(TerminalError::Execution)?;
        if !killed {
            return Ok(CommandResult::from(result));
        }
        Ok(CommandResult {
            stdout: result.stdout,
            stderr: result.stderr,
            interruptions: result.interruptions,
            outcome: CommandOutcome::Killed {
                notice: self.kill_notice(timeout_kill.as_ref()),
                task_id: prepared.task_id.to_string(),
            },
        })
    }

    /// Kills the job through the registry once its timeout passes.
    ///
    /// The returned watch holds the job's start time and, once the watchdog
    /// issues the kill, the time it did.
    fn spawn_kill_watchdog(&self, task_id: &IdLease, timeout: Duration) -> Arc<TimeoutWatch> {
        let watch = Arc::new(TimeoutWatch::new(timeout));
        let job_registry = self.job_registry.clone();
        // Weak, so a job that ended gives its word back before the timer
        // fires; a word that went back may name another job by then.
        let task_id = task_id.downgrade();
        let recorded = watch.clone();
        self.executor
            .spawn(async move {
                async_io::Timer::after(timeout).await;
                let Some(task_id) = task_id.upgrade() else {
                    return;
                };
                if let Some(job) = job_registry.get_by_task_id(&task_id).await
                    && matches!(job.status, JobStatus::Running)
                {
                    // Recorded before the kill, so a result that races the
                    // kill's return already finds it.
                    let _ = recorded.killed.set(jiff::Timestamp::now());
                    job_registry.kill(job.pid).await;
                }
            })
            .detach();
        watch
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

/// What [`TerminalTool::admit_command`] admits: a script and how it came.
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
    /// The carried directory a foreground command starts in, when the
    /// terminal carries one and it still exists.
    carried_start: Option<PathBuf>,
}

/// Where a run starts: `dir` on the host, mapped through the workspace
/// mount for a container command. A run carries no directory, so it writes
/// no report.
fn run_start(target: &ExecutionTarget, dir: PathBuf) -> Result<CwdCarry, TerminalRunError> {
    let container = match &target.backend {
        ExecutionBackend::Local(_) => None,
        ExecutionBackend::Container { runtime, .. } => {
            let mount = runtime
                .workspace_mount()
                .cloned()
                .ok_or(TerminalRunError::NoWorkspaceMount)?;
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
) -> anyhow::Result<()> {
    match startup_rx.recv().await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(anyhow::anyhow!(err)),
        Err(_) => Err(anyhow::anyhow!(
            "background startup channel dropped before registration"
        )),
    }
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

async fn background_terminal_result(
    job_registry: &JobRegistry,
    outputs: &OutputStore,
    task_id: &str,
    max_lines: usize,
    started_tx: &EventSender,
) -> aither_core::Result<ToolResult> {
    let redirect =
        start_background_output_redirect(job_registry, outputs, task_id, max_lines).await?;
    let _ = started_tx
        .send(ShellEvent::JobStarted(JobStarted {
            task_id: task_id.to_string(),
            output_preview: redirect.preview.clone(),
            output_file: redirect.file.clone(),
        }))
        .await;
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
            task_id: task_id.to_string(),
            output: redirect.file,
        },
    }
    .into_tool_result()
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
    working_dir: PathBuf,
    grants: Grants,
    env_source: Option<Arc<dyn EnvSource>>,
    executor: E,
    registry: Arc<ToolRegistry>,
    job_registry: JobRegistry,
    task_id: IdLease,
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
    network: NetworkSettings,
    priority: ProcessPriority,
    #[cfg(all(unix, feature = "jobserver"))]
    jobserver: Option<JobBroker>,
    heel: SharedHeelBinary,
    expect: OutputFormat,
    resolution: MediaResolution,
    outputs: OutputStore,
    max_lines: usize,
    raw: bool,
}

struct StandaloneStart<'a, E> {
    working_dir: &'a Path,
    grants: &'a Grants,
    env_source: &'a Option<Arc<dyn EnvSource>>,
    executor: E,
    registry: &'a Arc<ToolRegistry>,
    job_registry: &'a JobRegistry,
    task_id: &'a IdLease,
    execution_id: &'a str,
    carry: Option<&'a CwdCarry>,
    script: &'a str,
    interpreter: Interpreter,
    mode: SandboxMode,
    backend: &'a ExecutionBackend,
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
    background_startup: &'a BackgroundStartup,
    network: NetworkSettings,
    priority: ProcessPriority,
    #[cfg(all(unix, feature = "jobserver"))]
    jobserver: &'a Option<JobBroker>,
    heel: &'a SharedHeelBinary,
}

struct LocalBackgroundExecution<'a, E> {
    prepared_tools: &'a PreparedTools<PathBuf>,
    working_dir: &'a Path,
    grants: &'a Grants,
    env_source: &'a Option<Arc<dyn EnvSource>>,
    executor: E,
    registry: &'a Arc<ToolRegistry>,
    task_id: &'a IdLease,
    execution_id: &'a str,
    carry: Option<&'a CwdCarry>,
    script: &'a str,
    interpreter: Interpreter,
    mode: SandboxMode,
    job_registry: &'a JobRegistry,
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
    background_startup: &'a BackgroundStartup,
    priority: ProcessPriority,
    network: NetworkSettings,
    #[cfg(all(unix, feature = "jobserver"))]
    job_fifo: Option<crate::jobserver::CommandFifo>,
    heel: &'a SharedHeelBinary,
}

struct ContainerBackgroundExecution<'a, E> {
    executor: E,
    env_source: &'a Option<Arc<dyn EnvSource>>,
    registry: &'a Arc<ToolRegistry>,
    task_id: &'a IdLease,
    execution_id: &'a str,
    script: &'a str,
    interpreter: Interpreter,
    mode: SandboxMode,
    container_runtime: &'a ContainerShellRuntime,
    prepared_tools: &'a PreparedTools<ContainerPath>,
    confinement: Option<&'a ContainerConfinement>,
    carry: Option<&'a ContainerCarry>,
    ipc_commands: &'a [String],
    job_registry: &'a JobRegistry,
    stdin_alerts: Option<StdinAlerts>,
    lifetime: JobLifetime,
    background_startup: &'a BackgroundStartup,
    priority: ProcessPriority,
    /// The command's jobserver fifo inside the container, held until its
    /// exec has ended.
    #[cfg(all(unix, feature = "jobserver"))]
    job_fifo: Option<crate::container_jobserver::ContainerJobFifo>,
}

struct SshBackgroundExecution<'a> {
    prepared_tools: &'a PreparedTools<PathBuf>,
    task_id: &'a IdLease,
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
    raw: bool,
}

/// Where a command runs, with the tools prepared for exactly that environment.
enum ExecutionBackend {
    /// The local sandbox, whose tools live at a host path.
    Local(PreparedTools<PathBuf>),
    /// A container, whose tools are mounted at a container path.
    Container {
        runtime: ContainerShellRuntime,
        tools: PreparedTools<ContainerPath>,
        /// The sandbox a subagent's command runs in inside the container; the
        /// main agent's commands run unconfined.
        confinement: Option<ContainerConfinement>,
    },
    /// A remote host over SSH.
    Ssh {
        client: SshClient,
        server: SshServer,
        /// The server's `heel`, which a default-mode command runs under; an
        /// `unsafe` command has none.
        heel: Option<String>,
        tools: PreparedTools<PathBuf>,
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
    raw: bool,
    background: bool,
    timeout: Duration,
    carry: Option<CwdCarry>,
    /// The `terminal` call this execution belongs to.
    call: Option<CallIdentity>,
}

struct SpawnedTerminal {
    result: Receiver<Result<CompletedCommand, String>>,
    startup: Receiver<Result<(), String>>,
    timeout_watch: Arc<TimeoutWatch>,
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
    working_dir: PathBuf,
    grants: Grants,
    executor: E,
    registry: Arc<ToolRegistry>,
    job_registry: JobRegistry,
    task_id: IdLease,
    execution_id: String,
    background: bool,
    timeout_watch: Arc<TimeoutWatch>,
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
    raw: bool,
    result_tx: Sender<Result<CompletedCommand, String>>,
    /// Whether the call took the result: see [`SpawnedTerminal::took_result`].
    taken: Receiver<Sender<()>>,
    completed_tx: EventSender,
    running_tasks: Arc<AtomicUsize>,
    env_source: Option<Arc<dyn EnvSource>>,
    network: NetworkSettings,
    priority: ProcessPriority,
    #[cfg(all(unix, feature = "jobserver"))]
    jobserver: Option<JobBroker>,
    heel: SharedHeelBinary,
    outages: Outages,
}

#[allow(clippy::too_many_lines)]
async fn execute_terminal_task<E>(task: SpawnedTerminalTask<E>)
where
    E: Executor + Clone + 'static,
{
    let SpawnedTerminalTask {
        working_dir,
        grants,
        env_source,
        executor,
        registry,
        job_registry,
        task_id,
        execution_id,
        background,
        timeout_watch,
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
        raw,
        result_tx,
        taken,
        completed_tx,
        running_tasks,
        network,
        priority,
        #[cfg(all(unix, feature = "jobserver"))]
        jobserver,
        heel,
        outages,
    } = task;
    // Before the job registers, so the registry reports the bound it runs under.
    job_registry
        .bound_by_timeout(&task_id, timeout_watch.timeout)
        .await;
    let status_registry = job_registry.clone();
    let policies = registry.policies().clone();
    // What the policies are told the next script starts in, spelled as this
    // script's backend names a directory.
    let end_view = (!policies.is_empty()).then(|| {
        (
            DirSpelling::of(&backend),
            carried_dir.clone(),
            working_dir.clone(),
        )
    });
    let result = execute_script_standalone(StandaloneExecution {
        working_dir,
        grants,
        env_source,
        executor,
        registry,
        job_registry,
        task_id: task_id.clone(),
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
        network,
        priority,
        #[cfg(all(unix, feature = "jobserver"))]
        jobserver,
        heel,
        expect,
        resolution,
        outputs,
        max_lines,
        raw,
    })
    .await;
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
        Err(err) => Err(err.to_string()),
    };
    let _ = result_tx.send(quick_result).await;
    // The call answers at once, taking the result or leaving it to a
    // background task.
    let foreground = taken.recv().await.ok();
    // A kill's signal can end the process before the kill records its
    // outcome, so the registry settles the ending once no kill is in flight.
    let ending = match &result {
        Ok(command) => Ok(command.exit_code),
        Err(error) => Err(error.to_string()),
    };
    let owed = status_registry.owe_completion(&task_id, ending).await;
    let kill = owed.as_ref().and_then(|owed| owed.kill);
    let mut pending = owed.map(|owed| owed.pending);
    // A result the call returned leaves nothing for the agent to take: the
    // completion is taken here, so the registry forgets the job.
    if foreground.is_some()
        && let Some(pending) = pending.take()
    {
        pending.take().await;
    }
    // Every job reports completion; the agent reinjects the result only for
    // tasks it saw promoted or detached (`take_started`), so a plain
    // foreground result is not delivered twice.
    let _ = completed_tx
        .send(ShellEvent::JobEnded(Box::new(JobEnded {
            task_id: task_id.to_string(),
            script,
            result,
            kill,
            timeout_kill: timeout_watch.kill().filter(|_| kill.is_some()),
            pending,
        })))
        .await;
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
        working_dir,
        grants,
        env_source,
        executor,
        registry,
        job_registry,
        task_id,
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
        network,
        priority,
        #[cfg(all(unix, feature = "jobserver"))]
        jobserver,
        heel,
        expect,
        resolution,
        outputs,
        max_lines,
        raw,
    } = request;

    info!(
        script_len = script.len(),
        ?mode,
        "executing background terminal command"
    );
    debug!(script = %script, "script content");
    let start_result = start_standalone_execution(StandaloneStart {
        working_dir: &working_dir,
        grants: &grants,
        env_source: &env_source,
        executor,
        registry: &registry,
        job_registry: &job_registry,
        task_id: &task_id,
        execution_id: &execution_id,
        carry: carry.as_ref(),
        script: &script,
        interpreter,
        mode,
        backend: &backend,
        stdin_alerts,
        lifetime,
        background_startup: &background_startup,
        network,
        priority,
        #[cfg(all(unix, feature = "jobserver"))]
        jobserver: &jobserver,
        heel: &heel,
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
        raw,
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
                working_dir: &working_dir,
                grants: grants.snapshot(),
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
    let prepared_tools = match request.backend {
        ExecutionBackend::Local(tools) => tools,
        ExecutionBackend::Container {
            runtime,
            tools,
            confinement,
        } => {
            // The command's fifo lives in the container's kernel, brokered
            // by a relay run with the image's managed `python3`.
            #[cfg(all(unix, feature = "jobserver"))]
            let job_fifo = match request.jobserver {
                Some(broker) => Some(
                    crate::container_jobserver::open_container_job_fifo(
                        broker,
                        runtime,
                        &tools.bin_dir().join("python3"),
                        request.executor.clone(),
                    )
                    .await?,
                ),
                None => None,
            };
            let ipc_commands = request.registry.registered_tool_names();
            return execute_container_background(ContainerBackgroundExecution {
                executor: request.executor,
                env_source: request.env_source,
                registry: request.registry,
                task_id: request.task_id,
                execution_id: request.execution_id,
                script: request.script,
                interpreter: request.interpreter,
                mode: request.mode,
                container_runtime: runtime,
                prepared_tools: tools,
                confinement: confinement.as_ref(),
                carry: request.carry.and_then(|carry| carry.container.as_ref()),
                ipc_commands: &ipc_commands,
                job_registry: request.job_registry,
                stdin_alerts: request.stdin_alerts,
                lifetime: request.lifetime,
                background_startup: request.background_startup,
                priority: request.priority,
                #[cfg(all(unix, feature = "jobserver"))]
                job_fifo,
            })
            .await;
        }
        ExecutionBackend::Ssh {
            client,
            server,
            heel,
            tools,
        } => {
            // The pool bounds the jobs of this machine; the server is another
            // machine, so its commands take no token and get no fifo.
            return execute_ssh_background(SshBackgroundExecution {
                prepared_tools: tools,
                task_id: request.task_id,
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
        prepared_tools,
        working_dir: request.working_dir,
        grants: request.grants,
        env_source: request.env_source,
        executor: request.executor,
        registry: request.registry,
        task_id: request.task_id,
        execution_id: request.execution_id,
        carry: request.carry,
        script: request.script,
        interpreter: request.interpreter,
        mode: request.mode,
        job_registry: request.job_registry,
        stdin_alerts: request.stdin_alerts,
        lifetime: request.lifetime,
        background_startup: request.background_startup,
        priority: request.priority,
        network: request.network,
        #[cfg(all(unix, feature = "jobserver"))]
        job_fifo,
        heel: request.heel,
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
        raw,
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
            raw,
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
    raw: bool,
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
        compressed_stdout_data(format.outputs, stdout, format.expect, format.raw).await?
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

/// Runs the output cleaning pipeline over a completed command's stdout.
///
/// `raw: true` bypasses it entirely, as do non-text formats and output that
/// is not valid UTF-8, so the model sees exactly the bytes the command wrote.
async fn compressed_stdout_data<'a>(
    outputs: &OutputStore,
    stdout: &'a [u8],
    expect: OutputFormat,
    raw: bool,
) -> Result<Cow<'a, [u8]>, TerminalError> {
    let is_text = matches!(expect, OutputFormat::Text | OutputFormat::Auto);
    if raw || !is_text || stdout.is_empty() {
        return Ok(Cow::Borrowed(stdout));
    }
    let Ok(text) = std::str::from_utf8(stdout) else {
        return Ok(Cow::Borrowed(stdout));
    };

    let cleaned = crate::output_compress::compress_text(outputs, text)
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

/// The `git` shim every sandboxed command's PATH resolves first: a git
/// run in a committing worktree reads the worktree's host-owned entry
/// instead of whatever `.git` its subagent wrote.
#[cfg(unix)]
const GIT_SHIM: &str = include_str!("../templates/git_shim");

/// Writes the `git` shim into the prepared tools directory once: the
/// commands' PATH sees it ahead of the system's `git`.
#[cfg(unix)]
async fn write_git_shim(tools: &PreparedTools<PathBuf>) -> Result<(), TerminalError> {
    use std::os::unix::fs::PermissionsExt as _;
    let shim = tools.bin_dir().join("git");
    match async_fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&shim)
        .await
    {
        Ok(mut file) => {
            file.write_all(GIT_SHIM.as_bytes()).await.map_err(|error| {
                TerminalError::Execution(format!("writing {}: {error}", shim.display()))
            })?;
            file.flush().await.map_err(|error| {
                TerminalError::Execution(format!("writing {}: {error}", shim.display()))
            })?;
            async_fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755))
                .await
                .map_err(|error| {
                    TerminalError::Execution(format!("writing {}: {error}", shim.display()))
                })
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(TerminalError::Execution(format!(
            "writing {}: {error}",
            shim.display()
        ))),
    }
}

/// Windows commands run no `sh`: no shim, and the worktrees' git map is
/// never handed out either.
#[cfg(not(unix))]
async fn write_git_shim(_tools: &PreparedTools<PathBuf>) -> Result<(), TerminalError> {
    Ok(())
}

/// The script a container's command runs ahead of its own: the `git`
/// shim written into a writable directory it puts first on `PATH`, and
/// the worktree map as exports. Empty without a map — subagent
/// containers never see it.
#[cfg(unix)]
fn container_git_preamble(
    registry: &ToolRegistry,
    mount: Option<&MountRoot>,
) -> Result<String, TerminalError> {
    use std::fmt::Write as _;
    let mut environment = Vec::new();
    // The shim inside the container compares its working directory against
    // the map: every path it holds must be spelled the way the container's
    // workspace mount resolves it, or the worktree goes unprotected.
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
    committed_git_environment(registry, &mut environment, spell)?;
    if environment.is_empty() {
        return Ok(String::new());
    }
    let mut preamble =
        "aither_shim=${TMPDIR:-/tmp}/aither-shim-$$\nmkdir -p \"$aither_shim\"\n".to_string();
    preamble.push_str("cat > \"$aither_shim/git\" <<'AITHER_GIT_SHIM'\n");
    preamble.push_str(GIT_SHIM);
    preamble.push_str(
        "AITHER_GIT_SHIM\nchmod +x \"$aither_shim/git\"\nPATH=\"$aither_shim:$PATH\"\nexport PATH\n",
    );
    for (name, value) in &environment {
        writeln!(preamble, "export {name}={}", shell_words::quote(value))
            .expect("a string cannot fail a write");
    }
    Ok(preamble)
}

#[cfg(not(unix))]
fn container_git_preamble(
    _registry: &ToolRegistry,
    _mount: Option<&MountRoot>,
) -> Result<String, TerminalError> {
    Ok(String::new())
}

/// Hands the git shim of a main terminal's commands the worktree map,
/// each path spelled the way `spell` resolves it for the command's
/// context — a host path locally, a container path inside a container.
///
/// A command's `git` in a committing worktree resolves the worktree's
/// host-owned repository entry through `GIT_DIR`, so nothing a subagent
/// wrote — a `.git` file, the repository's configuration, hooks,
/// fsmonitor or filter programs — decides what program git runs. The
/// map only describes paths the host resolved when the grant was made.
/// Subagent terminals keep their own repositories' view.
fn committed_git_environment(
    registry: &ToolRegistry,
    environment: &mut Vec<(String, String)>,
    spell: impl Fn(&Path) -> Result<String, TerminalError>,
) -> Result<(), TerminalError> {
    if registry.role() != crate::command::CommandRole::Main {
        return Ok(());
    }
    let committed = registry.committed_worktrees();
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

use crate::spawn::resolved;
use heel::ResolvedPath;

/// The grant a command's jobserver fifo needs: it writes tokens back to it.
#[cfg(all(unix, feature = "jobserver"))]
pub fn job_fifo_grant(path: &Path) -> heel::Result<Grant> {
    Ok(Grant::new(resolved(path)?, Access::WRITE))
}

/// The grants the terminal's tools need in a session sandbox: the managed
/// tools directory, executable; the shared `uv` directory, which `uv` writes
/// and the interpreters it installs execute; and the host toolchain homes.
/// They are the terminal's toolchain specifics, handed to the sandbox as
/// fixed grants.
pub fn tool_grants(tools: &PreparedTools<PathBuf>) -> heel::Result<Vec<Grant>> {
    std::iter::once(Ok(Grant::new(resolved(tools.bin_dir())?, Access::EXEC)))
        .chain(
            tools
                .uv_dir()
                .map(|path| Ok(Grant::new(resolved(path)?, Access::WRITE | Access::EXEC))),
        )
        .chain(toolchain_grants(resolved))
        .collect()
}

/// Host toolchain homes a local sandbox reaches: (env var, resolved
/// existing dir). `CARGO_HOME` and `RUSTUP_HOME` fall back to their
/// `~/.cargo` / `~/.rustup` defaults; only directories that exist yield, so
/// a toolchain absent from the host is never granted.
fn host_toolchain_homes() -> impl Iterator<Item = (&'static str, PathBuf)> {
    let home = std::env::home_dir;
    [
        std::env::var_os("CARGO_HOME")
            .map(PathBuf::from)
            .or_else(|| home().map(|home| home.join(".cargo")))
            .map(|path| ("CARGO_HOME", path)),
        std::env::var_os("RUSTUP_HOME")
            .map(PathBuf::from)
            .or_else(|| home().map(|home| home.join(".rustup")))
            .map(|path| ("RUSTUP_HOME", path)),
    ]
    .into_iter()
    .flatten()
    .filter(|(_, path)| path.is_dir())
}

/// The grants that let a command run what the session's `PATH` lists: each
/// existing absolute directory of the `PATH` in `environment`, read and
/// execute, never write. They come from the `PATH` itself, so a tool in any
/// directory the user put there runs without a list of known toolchains.
pub fn path_directory_grants(environment: &[(String, String)]) -> heel::Result<Vec<Grant>> {
    let Some((_, path)) = environment.iter().find(|(name, _)| name == "PATH") else {
        return Ok(Vec::new());
    };
    let mut seen = std::collections::HashSet::new();
    let mut grants = Vec::new();
    for directory in std::env::split_paths(path) {
        if !directory.is_absolute() || !directory.is_dir() {
            continue;
        }
        let resolved = resolved(&directory)?;
        if seen.insert(resolved.as_path().to_path_buf()) {
            grants.push(Grant::new(resolved, Access::READ | Access::EXEC));
        }
    }
    Ok(grants)
}

/// The paths a session sandbox must not write: heel's persistence table
/// (git hooks and config, shell startup files, `PATH` directories and the
/// rest), where a write runs as code outside the sandbox later.
///
/// Three places are exempt, because the table forbids creating a `.git` at all
/// and the commands that need to would otherwise be impossible: cargo's git
/// cache, `$CARGO_HOME/git`, where cargo clones git dependencies and keeps
/// their `.git/hooks`, which it creates and removes itself and no user runs;
/// the session directory `working_dir`, where `git clone` and `git init`
/// make the session's own repositories; and the prepared runtime's `uv`
/// cache (`UV_CACHE_DIR`), whose buckets `uv` marks with a `.git` entry and
/// where no user runs git. Nothing else of a toolchain home is exempt.
pub fn session_write_denylist(
    working_dir: &Path,
    tools: &PreparedTools<PathBuf>,
) -> heel::WriteDenylist {
    let resolved = |path: &Path| std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let denylist = heel::WriteDenylist::persistence().except(resolved(working_dir));
    let denylist = match host_toolchain_homes().find(|(name, _)| *name == "CARGO_HOME") {
        Some((_, cargo_home)) => denylist.except(resolved(&cargo_home).join("git")),
        None => denylist,
    };
    match tools.uv_environment() {
        Some(uv) => denylist.except(resolved(&uv.cache)),
        None => denylist,
    }
}

/// The toolchain directories a sandboxed build needs: package-manager
/// installs, compilers, and their caches, each rwx since builds both run
/// the binaries and write the caches. Credential files inside them stay
/// denied by the security tables, which emit after grants.
fn toolchain_grants(
    resolved: impl Fn(&Path) -> heel::Result<ResolvedPath>,
) -> impl Iterator<Item = heel::Result<Grant>> {
    host_toolchain_homes()
        .map(move |(_, path)| Ok(Grant::new(resolved(&path)?, Access::WRITE | Access::EXEC)))
}

/// Runs a command on the host: sandboxed through the terminal's [`Sandbox`],
/// or with host access for an approved `unsafe` one.
async fn execute_local_background<E: Executor + Clone + 'static>(
    context: LocalBackgroundExecution<'_, E>,
) -> Result<(u32, std::process::Output), TerminalError> {
    let (sandbox, additions) = local_sandbox(&context).await?;
    let sandbox = match context.mode {
        SandboxMode::Sandboxed => sandbox,
        SandboxMode::Unsandboxed => sandbox.unconfined(),
    };
    // The sandboxed IPC router offers the built-in commands and every
    // command of the mode; the unsafe one the `ipc` gateway.
    let ipc = match context.mode {
        SandboxMode::Sandboxed => create_ipc_router(context.registry),
        SandboxMode::Unsandboxed => create_ipc_gateway_router(context.registry),
    };

    let launch_script = launch_script(context.interpreter, context.script, context.task_id)?;
    let launch = build_terminal_launch(
        &context.prepared_tools.shell()?,
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
        .env(additions.environment)
        .stdio(SpawnStdio {
            // A background job's stdin is watched; any other reads EOF.
            stdin: if context.stdin_alerts.is_some() {
                Pipe::Piped
            } else {
                Pipe::Null
            },
            ..SpawnStdio::default()
        });
    spec.extra_grants = additions.grants;
    spec.ipc = ipc;
    if let Some(start) = context.carry.and_then(|carry| carry.start.as_deref()) {
        spec = spec.cwd(start);
    }
    let lifetime = match context.lifetime {
        JobLifetime::Foreground { .. } => TreeLifetime::Foreground,
        JobLifetime::Background => TreeLifetime::Background,
    };
    let LocalSpawn {
        child,
        tree,
        sandbox: _sandbox,
    } = sandbox.spawn_local(spec, lifetime).await?;

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
    /// None: the survivors are killed and reported on `alerts`.
    Foreground { alerts: EventSender },
    /// All of them: the job runs until its whole tree has exited.
    Background,
}

/// Where a background job's stdin alerts go.
///
/// Its presence is what makes a command a background job: a command without
/// one reads EOF from a null stdin.
struct StdinAlerts {
    tx: EventSender,
    task_id: String,
    script: String,
}

impl StdinAlerts {
    async fn raise(&self) {
        debug!(task_id = %self.task_id, "background job is blocked on stdin");
        let _ = self
            .tx
            .send(ShellEvent::StdinBlocked(StdinBlocked {
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
    alerts: StdinAlerts,
}

impl LocalStdinWatch {
    /// Probes the pipe every [`STDIN_WATCH_INTERVAL`] and raises one alert per
    /// blocking episode, for as long as the future is polled.
    async fn run(mut self) -> std::convert::Infallible {
        loop {
            async_io::Timer::after(STDIN_WATCH_INTERVAL).await;
            let Self {
                probe,
                feed,
                mut detector,
                alerts,
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
                        alerts.raise().await;
                    }
                }
                Err(error) => {
                    // Detection cannot run: surface it once, then park — the
                    // job's exit still ends the race in
                    // `wait_for_local_process_exit`.
                    alerts.raise_failure(error.to_string()).await;
                    std::future::pending::<()>().await;
                }
            }
            self = Self {
                probe,
                feed,
                detector,
                alerts,
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
            stdin_watch = Some(LocalStdinWatch {
                probe,
                feed: feed.clone(),
                detector: EpisodeDetector::default(),
                alerts,
            });
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
            if let JobLifetime::Foreground { alerts } = &lifetime {
                let _ = alerts
                    .send(ShellEvent::DescendantsKilled(DescendantsKilled::new(
                        killed,
                    )))
                    .await;
            }
        }
    }
    let status = match (descendants, status) {
        (Ok(()), Ok(status)) => status,
        (Err(error), _) | (Ok(()), Err(error)) => {
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            job_registry.fail(pid, &error.to_string(), None).await;
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
        JobLifetime::Foreground { alerts } => {
            let killed = tree
                .kill_descendants()
                .await
                .map_err(|error| TerminalError::Execution(error.to_string()))?;
            if !killed.is_empty() {
                info!(
                    killed = killed.len(),
                    "killed descendants of a finished foreground command"
                );
                let _ = alerts
                    .send(ShellEvent::DescendantsKilled(DescendantsKilled::new(
                        killed,
                    )))
                    .await;
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

/// A container command's full script. A main terminal's git shim and
/// worktree map travel inside it: the container's tools mount is
/// read-only, so the command writes the shim where its own `PATH` can
/// put it first.
fn container_exec_script(
    registry: &ToolRegistry,
    container_runtime: &ContainerShellRuntime,
    prepared_tools: &PreparedTools<ContainerPath>,
    carry: Option<&ContainerCarry>,
    script: &str,
) -> Result<String, TerminalError> {
    let preamble = container_git_preamble(registry, container_runtime.workspace_mount())?;
    let script = if preamble.is_empty() {
        script.to_string()
    } else {
        format!("{preamble}{script}")
    };
    let launch = prepared_tools.container_launch();
    let script = launch
        .wrap_script(&script)
        .map_err(ToolRuntimeError::from)?;
    let script = match carry.and_then(|carry| carry.report.as_ref()) {
        Some(report) => carry_cwd_script(&script, report.as_str())?,
        None => script,
    };
    Ok(launch
        .shell_command(&script)
        .map_err(ToolRuntimeError::from)?)
}

#[allow(clippy::too_many_lines)]
async fn execute_container_background<E: Executor + Clone + 'static>(
    context: ContainerBackgroundExecution<'_, E>,
) -> Result<(u32, std::process::Output), TerminalError> {
    let container_runtime = context.container_runtime;
    let exec = container_runtime.exec();

    // Use lower 32 bits of a UUID as a synthetic PID for job tracking.
    let pid = u32::from_le_bytes(
        uuid::Uuid::new_v4().as_u128().to_le_bytes()[..4]
            .try_into()
            .expect("uuid byte slice has four bytes"),
    );
    let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
    let (input_tx, input_rx) = async_channel::unbounded::<Vec<u8>>();
    context
        .job_registry
        .register_at(
            crate::JobPlace::Container,
            pid,
            context.task_id,
            context.execution_id,
            context.script,
            context.interpreter,
            context.mode,
        )
        .await;
    let script = container_exec_script(
        context.registry,
        context.container_runtime,
        context.prepared_tools,
        context.carry,
        &launch_script(context.interpreter, context.script, context.task_id)?,
    )?;
    let script = inject_env_exports(script, context.env_source.as_ref())?;
    // Exported ahead of the embedder's variables, which win as they do for
    // a local command.
    #[cfg(all(unix, feature = "jobserver"))]
    let script = match &context.job_fifo {
        Some(fifo) => {
            let environment: Arc<dyn EnvSource> = Arc::new(fifo.environment());
            inject_env_exports(script, Some(&environment))?
        }
        None => script,
    };
    #[cfg(all(unix, feature = "jobserver"))]
    let confinement = job_confinement(
        context.confinement,
        context
            .job_fifo
            .as_ref()
            .map(crate::container_jobserver::ContainerJobFifo::path),
    );
    #[cfg(not(all(unix, feature = "jobserver")))]
    let confinement = context.confinement.cloned();
    let script = mark_container_job(script, context.task_id)?;
    // A foreground command gets no input channel: the dropped sender is the
    // EOF its stdin reads. StdinAlerts may ride along either way; they are
    // the background job's blocked-reader wiring.
    let exec_lifetime = if let JobLifetime::Foreground { .. } = context.lifetime {
        drop(input_tx);
        ExecLifetime::Foreground
    } else {
        context
            .job_registry
            .attach_terminal_input(pid, input_tx)
            .await;
        let (signal_tx, signal_rx) = async_channel::unbounded::<StdinWatchSignal>();
        if let Some(alerts) = context.stdin_alerts {
            context
                .executor
                .clone()
                .spawn(async move {
                    while let Ok(signal) = signal_rx.recv().await {
                        match signal {
                            StdinWatchSignal::Blocked => alerts.raise().await,
                            StdinWatchSignal::Failed(error) => {
                                alerts.raise_failure(error).await;
                            }
                        }
                    }
                })
                .detach();
        }
        ExecLifetime::Background {
            stdin_blocked: signal_tx,
        }
    };
    context.job_registry.attach_kill_switch(pid, kill_tx).await;
    context.background_startup.report_ready().await;

    let ipc_bridge = if context.ipc_commands.is_empty() {
        None
    } else {
        Some(start_container_ipc_bridge(
            context.executor,
            Arc::clone(context.registry),
            container_runtime.workspace_mount().cloned(),
            container_runtime.host_route().address(),
        )?)
    };
    let working_dir = context
        .carry
        .map_or(CONTAINER_EXEC_WORKING_DIR, |carry| carry.start.as_str());
    let wrapped_script = wrap_container_script(
        &script,
        context.ipc_commands,
        ipc_bridge.as_ref().map(|bridge| ContainerIpcEndpoint {
            host: container_runtime.host_route().name(),
            port: bridge.port(),
        }),
        context.priority,
        context.prepared_tools.bin_dir(),
        working_dir,
        confinement.as_ref(),
    )?;

    let execution = exec
        .exec_boxed(
            container_runtime.container_id(),
            &wrapped_script,
            working_dir,
            kill_rx,
            input_rx,
            exec_lifetime,
        )
        .await;

    if let Some(bridge) = ipc_bridge {
        bridge.stop().await;
    }
    // The exec has ended, killed or not: the relay stops feeding the fifo and
    // returns its tokens once the job's last process holding it is gone.
    #[cfg(all(unix, feature = "jobserver"))]
    drop(context.job_fifo);

    record_container_exec_outcome(context.job_registry, pid, &context.lifetime, execution).await
}

/// Records a finished container exec in the job's streams and folds it into
/// the call's result.
async fn record_container_exec_outcome(
    job_registry: &JobRegistry,
    pid: u32,
    lifetime: &JobLifetime,
    execution: Result<ContainerExecOutcome, String>,
) -> Result<(u32, std::process::Output), TerminalError> {
    match execution {
        Ok(ContainerExecOutcome::Completed { output, killed }) => {
            if let JobLifetime::Foreground { alerts } = lifetime
                && !killed.is_empty()
            {
                info!(
                    killed = killed.len(),
                    "killed processes of a finished foreground container command"
                );
                let _ = alerts
                    .send(ShellEvent::DescendantsKilled(DescendantsKilled::new(
                        killed,
                    )))
                    .await;
            }
            if !output.stdout.is_empty() {
                job_registry.append_stdout(pid, output.stdout.clone()).await;
            }
            if !output.stderr.is_empty() {
                job_registry.append_stderr(pid, output.stderr.clone()).await;
            }
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            Ok((pid, output))
        }
        Ok(ContainerExecOutcome::Killed) => {
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            Err(TerminalError::Execution("container job killed".to_string()))
        }
        Err(err) => {
            job_registry.fail(pid, &err, None).await;
            job_registry.close_stdout(pid).await;
            job_registry.close_stderr(pid).await;
            Err(TerminalError::Execution(err))
        }
    }
}

async fn execute_ssh_background(
    context: SshBackgroundExecution<'_>,
) -> Result<(u32, std::process::Output), TerminalError> {
    // A default-mode command runs the managed shell under the server's
    // `heel`; an `unsafe` one runs it directly.
    let remote_cmd = crate::spawn::remote_command(
        &context.prepared_tools.remote_launch()?,
        context.heel,
        &launch_script(context.interpreter, context.script, context.task_id)?,
    )
    .map_err(ToolRuntimeError::from)?;

    let mut command = context.client.command(context.server, &remote_cmd);
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut command, 0);

    // The pipes are set on the async command: converting a std command
    // resets them, and spawning would then inherit this process's stdio.
    let child = async_process::Command::from(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| TerminalError::Execution(e.to_string()))?;

    let pid = child.id();
    // The local `ssh` process stands for a job that runs on the remote host,
    // whose memory is never judged.
    context
        .job_registry
        .register_at(
            crate::JobPlace::Remote,
            pid,
            context.task_id,
            context.execution_id,
            context.script,
            context.interpreter,
            context.mode,
        )
        .await;
    context.background_startup.report_ready().await;

    match child.output().await {
        Ok(output) => {
            if !output.stdout.is_empty() {
                context
                    .job_registry
                    .append_stdout(pid, output.stdout.clone())
                    .await;
            }
            if !output.stderr.is_empty() {
                context
                    .job_registry
                    .append_stderr(pid, output.stderr.clone())
                    .await;
            }
            context.job_registry.close_stdout(pid).await;
            context.job_registry.close_stderr(pid).await;
            Ok((pid, output))
        }
        Err(err) => {
            context.job_registry.fail(pid, &err.to_string(), None).await;
            context.job_registry.close_stdout(pid).await;
            context.job_registry.close_stderr(pid).await;
            Err(TerminalError::Execution(err.to_string()))
        }
    }
}

/// Prepends a source's env pairs to `script` as `export` statements, for
/// backends (container exec) whose channel accepts only a script — never an
/// env channel. Names are validated so a provider can't inject shell syntax.
pub fn inject_env_exports(
    script: String,
    env_source: Option<&Arc<dyn EnvSource>>,
) -> Result<String, TerminalError> {
    let Some(source) = env_source else {
        return Ok(script);
    };
    let mut rendered = String::new();
    for (name, value) in source.env() {
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

/// Marks every process of a container job with its task ID, the marker the
/// container's memory reader finds the job's processes by: each process the
/// job starts inherits it.
fn mark_container_job(script: String, task_id: &str) -> Result<String, TerminalError> {
    let marker: Arc<dyn EnvSource> = Arc::new(vec![(
        crate::memory::JOB_MARKER_ENV.to_string(),
        task_id.to_string(),
    )]);
    inject_env_exports(script, Some(&marker))
}

fn shell_escape(value: &str) -> String {
    let escaped = value.replace('\'', "'\"'\"'");
    let mut output = String::with_capacity(escaped.len() + 2);
    output.push('\'');
    output.push_str(&escaped);
    output.push('\'');
    output
}

/// What a session sandbox carries for the terminal's tools: the grants and
/// environment of its toolchains and the write denylist. The embedder's side
/// of the sandbox: [`Sandbox`] takes them as fixed grants, environment and
/// denylist.
struct SessionBase {
    grants: Vec<Grant>,
    environment: Vec<(String, String)>,
    write_denylist: heel::WriteDenylist,
}

/// Builds the session base of a session whose directory is `working_dir`:
/// the session's `HOME`, temporary and cache directories, `PATH` led by the
/// managed tools directory, `uv` and toolchain variables, and the grants that
/// let a command run what the `PATH` lists and the tools it needs.
///
/// Creates the session's `tmp`, `.cache` and `.config` directories, which
/// its environment points at.
async fn session_base(
    working_dir: &Path,
    tools: &PreparedTools<PathBuf>,
) -> Result<SessionBase, TerminalError> {
    let environment = prepare_session_runtime_environment(working_dir, tools).await?;
    let setup = |error: heel::Error| TerminalError::SandboxSetup(error.to_string());
    // The `PATH` directories lead, so a writable grant over one of them is
    // never narrowed by them.
    let mut grants = path_directory_grants(&environment).map_err(setup)?;
    grants.extend(tool_grants(tools).map_err(setup)?);
    Ok(SessionBase {
        grants,
        environment,
        write_denylist: session_write_denylist(working_dir, tools),
    })
}

/// What one local command adds to the sandbox for itself.
struct CommandAdditions {
    /// Its jobserver variables and the git shim's committed-grant variables.
    environment: Vec<(String, String)>,
    /// Its jobserver fifo, which its build tools write tokens back to.
    grants: Vec<Grant>,
}

/// Builds the [`Sandbox`] a local command spawns in, and what the command
/// adds to it: the session base, the live grants, the network lists with
/// their approver and audit log, and the embedder's environment source.
async fn local_sandbox<E: Executor + Clone + 'static>(
    context: &LocalBackgroundExecution<'_, E>,
) -> Result<(Sandbox, CommandAdditions), TerminalError> {
    write_git_shim(context.prepared_tools).await?;
    let session = session_base(context.working_dir, context.prepared_tools).await?;
    let heel = context
        .heel
        .get()
        .await
        .map_err(|e| TerminalError::SandboxSetup(e.to_string()))?;
    let mut additions = CommandAdditions {
        environment: Vec::new(),
        grants: Vec::new(),
    };
    #[cfg(all(unix, feature = "jobserver"))]
    if let Some(fifo) = &context.job_fifo {
        additions.grants.push(
            job_fifo_grant(fifo.path())
                .map_err(|error| TerminalError::SandboxSetup(error.to_string()))?,
        );
        additions.environment.extend(fifo.environment());
    }
    committed_git_environment(context.registry, &mut additions.environment, |path| {
        Ok(path.display().to_string())
    })?;

    let mut sandbox = Sandbox::new(
        crate::Backend::Host { heel },
        context.executor.clone(),
        context.working_dir,
    )
    .with_grants(context.grants.clone())
    .with_fixed_grants(session.grants)
    .with_environment(session.environment)
    .with_write_denylist(session.write_denylist)
    .with_priority(context.priority)
    .with_scanner(context.job_registry.scanner());
    if let Some(source) = context.env_source {
        sandbox = sandbox.with_env_source(source.clone());
    }
    if let Some(lists) = &context.network.lists {
        sandbox = sandbox.with_network_policy(lists.clone());
    }
    if let Some(approver) = &context.network.approver {
        sandbox = sandbox.with_domain_approver(approver.clone());
    }
    if let Some(log) = &context.network.audit {
        sandbox = sandbox.with_audit_log(log.clone());
    }
    Ok((sandbox, additions))
}

pub async fn prepare_session_runtime_environment(
    working_dir: &Path,
    tools: &PreparedTools<PathBuf>,
) -> Result<Vec<(String, String)>, TerminalError> {
    let session_tmp = working_dir.join("tmp");
    let session_cache = working_dir.join(".cache");
    let bun_cache = session_cache.join("bun");
    let session_config = working_dir.join(".config");
    let playwright_cache = session_cache.join("ms-playwright");
    let python_path = working_dir.join("skills").join("python");

    for directory in [
        &session_tmp,
        &session_cache,
        &bun_cache,
        &session_config,
        &playwright_cache,
    ] {
        async_fs::create_dir_all(directory).await?;
    }

    let tmp = path_environment_value("TMPDIR", &session_tmp)?;
    let cache = path_environment_value("XDG_CACHE_HOME", &session_cache)?;
    let bun = path_environment_value("BUN_INSTALL_CACHE_DIR", &bun_cache)?;
    let config = path_environment_value("XDG_CONFIG_HOME", &session_config)?;
    let playwright = path_environment_value("PLAYWRIGHT_BROWSERS_PATH", &playwright_cache)?;
    let home = path_environment_value("HOME", working_dir)?;
    let python = std::env::var_os("PYTHONPATH").map_or_else(
        || Ok::<_, TerminalError>(python_path.clone().into_os_string()),
        |existing| {
            std::env::join_paths(
                std::iter::once(python_path.clone()).chain(std::env::split_paths(&existing)),
            )
            .map_err(|error| {
                TerminalError::Execution(format!("failed to construct session PYTHONPATH: {error}"))
            })
        },
    )?;
    let python = python.into_string().map_err(|value| {
        TerminalError::Execution(format!(
            "session PYTHONPATH is not valid UTF-8: {}",
            value.to_string_lossy()
        ))
    })?;

    // The managed tools directory leads; the host PATH follows so host
    // toolchains (cargo, rustup, anything else) stay reachable. The heel
    // sandbox path is only a fallback for a host without a PATH.
    let host_path = std::env::var_os("PATH")
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| heel::default_sandbox_path().into());
    let path = tools
        .prepend_path(Some(host_path.as_os_str()))?
        .into_string()
        .map_err(|_| ToolRuntimeError::InvalidCache(tools.bin_dir().clone()))?;
    // Point cargo and rustup at the same host homes the grants cover, so
    // the remapped HOME does not strand them under the working dir.
    let mut env = vec![
        ("PATH".to_string(), path),
        ("TMPDIR".to_string(), tmp.clone()),
        ("TMP".to_string(), tmp.clone()),
        ("TEMP".to_string(), tmp),
        ("HOME".to_string(), home),
        ("XDG_CACHE_HOME".to_string(), cache),
        ("XDG_CONFIG_HOME".to_string(), config),
        ("BUN_INSTALL_CACHE_DIR".to_string(), bun),
        ("PLAYWRIGHT_BROWSERS_PATH".to_string(), playwright),
        ("PYTHONPATH".to_string(), python),
    ];
    // `uv` keeps its cache and the Pythons it installs for projects in one
    // directory shared by every session, not under the remapped HOME.
    if let Some(uv) = tools.uv_environment() {
        env.push((
            "UV_CACHE_DIR".to_string(),
            path_environment_value("UV_CACHE_DIR", &uv.cache)?,
        ));
        env.push((
            "UV_PYTHON_INSTALL_DIR".to_string(),
            path_environment_value("UV_PYTHON_INSTALL_DIR", &uv.python_installs)?,
        ));
        env.push((
            "UV_PYTHON_PREFERENCE".to_string(),
            crate::tool_runtime::UV_PYTHON_PREFERENCE.to_string(),
        ));
    }
    for (name, home) in host_toolchain_homes() {
        env.push((name.to_string(), path_environment_value(name, &home)?));
    }
    Ok(env)
}

fn path_environment_value(name: &str, path: &Path) -> Result<String, TerminalError> {
    path.to_str().map(str::to_string).ok_or_else(|| {
        TerminalError::Execution(format!(
            "session environment path for {name} is not valid UTF-8: {}",
            path.display()
        ))
    })
}

#[derive(Template)]
#[template(path = "container_ipc_wrapper.sh", escape = "none")]
struct ContainerIpcWrapperTemplate<'a> {
    escaped_commands: &'a str,
    ipc_host: &'a str,
    ipc_port: u16,
    script: &'a str,
}

#[derive(Clone, Copy)]
pub struct ContainerIpcEndpoint<'a> {
    host: &'a str,
    port: u16,
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
    pub(crate) writable: Vec<ContainerPath>,
}

/// A container command's sandbox: a subagent's confinement plus write access
/// to the command's jobserver fifo, which its build tools write tokens back
/// to. An unconfined command already reaches the fifo.
#[cfg(all(unix, feature = "jobserver"))]
fn job_confinement(
    confinement: Option<&ContainerConfinement>,
    job_fifo: Option<&ContainerPath>,
) -> Option<ContainerConfinement> {
    let mut confinement = confinement?.clone();
    confinement.writable.extend(job_fifo.cloned());
    Some(confinement)
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
    heel_bin: &'a str,
    resolve_heel: bool,
    escaped_working_dir: &'a str,
    escaped_tools_dir: &'a str,
    /// Set for a subagent: the isolation level and grants replacing the
    /// container-wide permissive one.
    escaped_grants: Option<Vec<String>>,
    escaped_ipc_endpoint: Option<String>,
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
/// reimplemented here. `heel_bin` is the shell expression that resolves the
/// image's `heel` (`$MAY_IPC_BIN` once the IPC wrapper has resolved it, or
/// `$MAY_HEEL_BIN` the template resolves itself when `resolve_heel` is set)
/// and `working_dir` is the exact directory the execution runs in, matching
/// the directory the container exec applies.
///
/// The container is already the isolation boundary, so the invocation asks for
/// `permissive` isolation: the script keeps the filesystem permissions it had
/// as the container caller. The ruleset still exec-denies everything outside
/// the system directories, so `--executable` grants the mounted tools
/// directory read-execute — without it every prepared tool fails with
/// Permission denied. `PATH` and `HOME` are forwarded so the generated
/// `heel ipc` shims still resolve and the script sees its usual environment.
/// Networking keeps heel's deny-all default, and no rule is bent to pass the
/// IPC transport: `--ipc-endpoint` has heel relay the sandbox's local IPC
/// socket to the bridge endpoint from outside the sandbox, so the sandboxed
/// `heel ipc` never needs TCP access of its own.
fn heel_run_container_command(
    script: &str,
    priority: ProcessPriority,
    heel_bin: &str,
    resolve_heel: bool,
    ipc_endpoint: Option<ContainerIpcEndpoint<'_>>,
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
            .writable
            .iter()
            .map(|path| shell_escape(&format!("{path}=rw")))
            .collect()
    });
    let escaped_working_dir = shell_escape(working_dir);
    let escaped_tools_dir = shell_escape(tools_dir.as_str());
    let escaped_ipc_endpoint = ipc_endpoint
        .map(|endpoint| format!("tcp://{}:{}", endpoint.host, endpoint.port))
        .map(|endpoint| shell_escape(&endpoint));
    let escaped_script = shell_escape(script);
    ContainerHeelRunTemplate {
        heel_bin,
        resolve_heel,
        escaped_working_dir: &escaped_working_dir,
        escaped_tools_dir: &escaped_tools_dir,
        escaped_grants,
        escaped_ipc_endpoint,
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
    // Outside the IPC wrapper no `heel` binary has been resolved yet; the
    // template resolves it the same way the wrapper does and fails when the
    // image lacks one.
    heel_run_container_command(
        script,
        priority,
        "\"$MAY_HEEL_BIN\"",
        true,
        None,
        ContainerRunScope {
            tools_dir,
            working_dir,
            confinement,
        },
    )
}

pub fn wrap_container_script(
    script: &str,
    ipc_commands: &[String],
    ipc_endpoint: Option<ContainerIpcEndpoint<'_>>,
    priority: ProcessPriority,
    tools_dir: &ContainerPath,
    working_dir: &str,
    confinement: Option<&ContainerConfinement>,
) -> Result<String, TerminalError> {
    if ipc_commands.is_empty() {
        return prioritized_container_script(script, priority, tools_dir, working_dir, confinement);
    }

    let endpoint = ipc_endpoint.ok_or_else(|| {
        TerminalError::Execution(
            "missing container IPC endpoint for wrapped script execution".to_string(),
        )
    })?;

    for name in ipc_commands {
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(TerminalError::Execution(format!(
                "invalid ipc command name for container wrapper: {name}"
            )));
        }
    }

    let escaped_commands = ipc_commands
        .iter()
        .map(|name| shell_escape(name))
        .collect::<Vec<_>>()
        .join(" ");
    let script = if priority.is_normal() && confinement.is_none() {
        script.to_string()
    } else {
        // The generated shims are written under the shared temp directory,
        // which stays execute-denied even at `permissive` isolation: the
        // render below grants them read-and-execute so the sandboxed `PATH`
        // resolves them — the wrapper writes them before `heel run`, so the
        // sandbox itself never writes the directory.
        heel_run_container_command(
            script,
            priority,
            "\"$MAY_IPC_BIN\"",
            false,
            Some(endpoint),
            ContainerRunScope {
                tools_dir,
                working_dir,
                confinement,
            },
        )?
    };
    ContainerIpcWrapperTemplate {
        escaped_commands: &escaped_commands,
        ipc_host: endpoint.host,
        ipc_port: endpoint.port,
        script: &script,
    }
    .render()
    .map_err(|error| {
        TerminalError::Execution(format!("failed to render container IPC wrapper: {error}"))
    })
}

struct ContainerIpcBridge {
    shutdown_tx: Sender<()>,
    port: u16,
}

impl ContainerIpcBridge {
    const fn port(&self) -> u16 {
        self.port
    }

    async fn stop(self) {
        tracing::debug!("stopping container IPC bridge");
        let _ = self.shutdown_tx.send(()).await;
    }
}

fn start_container_ipc_bridge<E: Executor + Clone + 'static>(
    executor: E,
    registry: Arc<ToolRegistry>,
    workspace: Option<MountRoot>,
    bind: std::net::IpAddr,
) -> Result<ContainerIpcBridge, TerminalError> {
    // The bridge executes host IPC commands for whoever connects, so it binds
    // only the address the container route names — never a wildcard.
    let listener = StdTcpListener::bind(SocketAddr::new(bind, 0)).map_err(|e| {
        TerminalError::Execution(format!("failed to bind container IPC tcp port: {e}"))
    })?;
    let local_addr = listener.local_addr().map_err(|e| {
        TerminalError::Execution(format!("failed to resolve container IPC tcp endpoint: {e}"))
    })?;
    tracing::debug!(
        bind = %local_addr,
        "starting container IPC bridge"
    );
    listener.set_nonblocking(true).map_err(|e| {
        TerminalError::Execution(format!(
            "failed to set IPC tcp listener nonblocking mode: {e}"
        ))
    })?;
    let listener = Async::new(listener).map_err(|e| {
        TerminalError::Execution(format!(
            "failed to register IPC tcp listener with async reactor: {e}"
        ))
    })?;

    let (shutdown_tx, shutdown_rx) = async_channel::bounded::<()>(1);
    let bridge_executor = executor.clone();
    executor
        .spawn(async move {
            tracing::debug!(bind = %local_addr, "container IPC bridge listening");
            if let Err(error) = run_container_ipc_bridge(
                listener,
                registry,
                workspace,
                shutdown_rx,
                bridge_executor,
            )
            .await
            {
                tracing::debug!(error = %error, "container IPC bridge stopped with error");
            }
            tracing::debug!(bind = %local_addr, "container IPC bridge stopped");
        })
        .detach();

    Ok(ContainerIpcBridge {
        shutdown_tx,
        port: local_addr.port(),
    })
}

enum ContainerIpcBridgeEvent {
    Accept(std::io::Result<(Async<StdTcpStream>, std::net::SocketAddr)>),
    Shutdown,
}

async fn run_container_ipc_bridge<E: Executor + Clone + 'static>(
    listener: Async<StdTcpListener>,
    registry: Arc<ToolRegistry>,
    workspace: Option<MountRoot>,
    shutdown_rx: Receiver<()>,
    executor: E,
) -> Result<(), String> {
    loop {
        let event = futures_lite::future::or(
            async { ContainerIpcBridgeEvent::Accept(listener.accept().await) },
            async {
                let _ = shutdown_rx.recv().await;
                ContainerIpcBridgeEvent::Shutdown
            },
        )
        .await;

        match event {
            ContainerIpcBridgeEvent::Shutdown => break,
            ContainerIpcBridgeEvent::Accept(Ok((stream, _addr))) => {
                tracing::debug!("container IPC bridge accepted connection");
                let registry = registry.clone();
                let workspace = workspace.clone();
                executor
                    .spawn(async move {
                        if let Err(error) =
                            handle_container_ipc_connection(stream, registry, workspace).await
                        {
                            tracing::debug!(error = %error, "container IPC connection failed");
                        }
                    })
                    .detach();
            }
            ContainerIpcBridgeEvent::Accept(Err(error))
                if error.kind() == std::io::ErrorKind::Interrupted => {}
            ContainerIpcBridgeEvent::Accept(Err(error)) => {
                return Err(format!("container IPC accept failed: {error}"));
            }
        }
    }
    Ok(())
}

async fn handle_container_ipc_connection(
    mut stream: Async<StdTcpStream>,
    registry: Arc<ToolRegistry>,
    workspace: Option<MountRoot>,
) -> Result<(), String> {
    loop {
        let mut length_bytes = [0_u8; 4];
        match stream.read_exact(&mut length_bytes).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(format!("failed to read IPC request length: {error}")),
        }
        let body_length = u32::from_be_bytes(length_bytes) as usize;
        if body_length == 0 || body_length > 16 * 1024 * 1024 {
            return Err(format!("invalid IPC request length: {body_length}"));
        }

        let mut body = vec![0_u8; body_length];
        stream
            .read_exact(&mut body)
            .await
            .map_err(|e| format!("failed to read IPC request body: {e}"))?;

        let request = match heel::ipc::IpcRequest::decode(&body) {
            Ok(request) if !request.method.is_empty() => request,
            Ok(_) => {
                write_container_ipc_error(&mut stream, "empty IPC method name").await?;
                continue;
            }
            Err(error) => {
                write_container_ipc_error(&mut stream, &error.to_string()).await?;
                continue;
            }
        };
        let cli_args = decode_container_ipc_args(&request.params)?;
        // The caller reports its directory as the container sees it; the
        // host resolves relative paths in the directory that is mounted at.
        // A directory outside the workspace mount has no host spelling.
        let working_dir = workspace.as_ref().and_then(|workspace| {
            let container_dir = ContainerPath::new(request.working_dir.to_str()?).ok()?;
            workspace.host_path(&container_dir).ok()
        });
        let envelope = registry
            .query_tool_handler_from(&request.method, &cli_args, working_dir.as_deref())
            .await;
        match crate::command::envelope_to_cli(envelope) {
            Ok(rendered) => write_container_ipc_success(&mut stream, &rendered).await?,
            Err(error) => {
                write_container_ipc_error(&mut stream, &error.to_string()).await?;
            }
        }
    }
}

fn decode_container_ipc_args(params: &[u8]) -> Result<Vec<String>, String> {
    let parsed: serde_json::Value =
        rmp_serde::from_slice(params).map_err(|e| format!("invalid IPC params: {e}"))?;
    let args = parsed
        .as_object()
        .ok_or_else(|| "container IPC params must be a map".to_string())?;
    let args = args
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<std::collections::HashMap<_, _>>();
    Ok(crate::command::flatten_args_to_cli(&args))
}

async fn write_container_ipc_success(
    stream: &mut Async<StdTcpStream>,
    rendered: &str,
) -> Result<(), String> {
    let payload = rmp_serde::to_vec_named(rendered)
        .map_err(|e| format!("failed to encode IPC success payload: {e}"))?;
    write_container_ipc_response(stream, true, &payload).await
}

async fn write_container_ipc_error(
    stream: &mut Async<StdTcpStream>,
    message: &str,
) -> Result<(), String> {
    let payload = rmp_serde::to_vec_named(&message.to_string())
        .map_err(|e| format!("failed to encode IPC error payload: {e}"))?;
    write_container_ipc_response(stream, false, &payload).await
}

async fn write_container_ipc_response(
    stream: &mut Async<StdTcpStream>,
    success: bool,
    payload: &[u8],
) -> Result<(), String> {
    let body_length = 1usize
        .checked_add(payload.len())
        .ok_or_else(|| "IPC response payload length overflow".to_string())?;
    let response_length = u32::try_from(body_length)
        .map_err(|_| format!("IPC response body too large: {body_length} bytes"))?;

    stream
        .write_all(&response_length.to_be_bytes())
        .await
        .map_err(|e| format!("failed to write IPC response length: {e}"))?;
    stream
        .write_all(&[u8::from(success)])
        .await
        .map_err(|e| format!("failed to write IPC success flag: {e}"))?;
    stream
        .write_all(payload)
        .await
        .map_err(|e| format!("failed to write IPC payload: {e}"))?;
    stream
        .flush()
        .await
        .map_err(|e| format!("failed to flush IPC payload: {e}"))
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

/// Creates a sandboxed execution's IPC router: the built-in commands and
/// every command `registry` has in the execution's mode.
fn create_ipc_router(registry: &Arc<ToolRegistry>) -> Option<IpcRouter> {
    let mut router = builtin_router();

    let tool_names = registry.registered_tool_names();
    tracing::info!(tools = ?tool_names, "Creating IPC router with registered tools");
    for name in tool_names {
        router = crate::register_tool_command(router, Arc::clone(registry), &name);
    }

    let has_methods = router.methods().next().is_some();
    has_methods.then_some(router)
}

/// Creates an unsafe execution's IPC router: the `ipc` gateway and every
/// command `registry` has in unsafe mode. A command registered for
/// sandboxed modes alone gets no shim, so the name resolves on the host's
/// `PATH`.
fn create_ipc_gateway_router(registry: &Arc<ToolRegistry>) -> Option<IpcRouter> {
    let mut router = crate::register_ipc_gateway_command(IpcRouter::new(), Arc::clone(registry));

    for name in registry.registered_tool_names() {
        router = crate::register_tool_command(router, Arc::clone(registry), &name);
    }

    let has_methods = router.methods().next().is_some();
    has_methods.then_some(router)
}

/// Errors that can occur during terminal execution.
#[derive(Debug, thiserror::Error)]
pub enum TerminalError {
    /// Execution environment or mode is not configured.
    #[error("invalid terminal execution target: {0}")]
    InvalidTarget(String),
    /// `location` names no configured SSH server.
    #[error("no SSH server named `{0}` is configured")]
    UnknownLocation(String),
    /// A default-mode command names a server that has no `heel` to sandbox it.
    #[error("default mode runs under `heel`, which `{0}` does not have")]
    NoRemoteSandbox(String),
    /// Required tool runtime is unavailable or unsafe to expose.
    #[error(transparent)]
    Tools(#[from] ToolRuntimeError),
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

impl From<crate::SpawnError> for TerminalError {
    fn from(error: crate::SpawnError) -> Self {
        match error {
            crate::SpawnError::Setup(message) => Self::SandboxSetup(message),
            crate::SpawnError::Start(message) => Self::Execution(message),
        }
    }
}

impl<E, State> std::fmt::Debug for TerminalTool<E, State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerminalTool")
            .field("working_dir", &self.working_dir)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::time::Duration;

    use super::*;
    use crate::GitDirAccess;
    #[cfg(unix)]
    use crate::builtin::{InputTerminalArgs, InputTerminalTool};
    use crate::output::empty_stream;
    use crate::test_wording::TestWording;
    use crate::{ShellWording, ToolRegistryBuilder};
    #[cfg(unix)]
    use heel::DenyAll;
    use heel::NiceLevel;

    #[cfg(all(unix, feature = "connectivity"))]
    mod outage;

    #[cfg(unix)]
    mod progress;

    #[cfg(unix)]
    mod ssh;

    /// An output store in `dir`, which exists, leasing from the default
    /// allocator.
    fn test_outputs(dir: impl Into<PathBuf>) -> OutputStore {
        let policy = crate::output_policy::fixture::policy();
        OutputStore::testing(dir.into(), AnyIdAllocator::default(), policy)
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

    /// Toolchain directories named by `*_HOME` variables become rwx grants;
    /// ones that do not exist are skipped.
    #[test]
    fn toolchain_grants_cover_existing_dirs_rwx() {
        let cargo = tempfile::tempdir().unwrap();
        let rustup = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("CARGO_HOME", cargo.path());
            std::env::set_var("RUSTUP_HOME", rustup.path().join("missing"));
        }
        let grants: Vec<Grant> = toolchain_grants(|path| {
            ResolvedPath::new(std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()))
        })
        .map(|grant| grant.expect("grant"))
        .collect();
        assert_eq!(grants.len(), 1, "only the existing dir is granted");
        let access = grants[0].access();
        assert!(access.can_write());
        assert!(access.can_execute());
        unsafe {
            std::env::remove_var("CARGO_HOME");
            std::env::remove_var("RUSTUP_HOME");
        }
    }

    async fn prepared_test_registry() -> (
        crate::tool_runtime::tests::UnlockingTempDir,
        ShellSessionRegistry,
    ) {
        let cache = crate::tool_runtime::tests::unlocking_tempdir();
        let tools = crate::tool_runtime::tests::bundle()
            .prepare_local(cache.path())
            .await
            .unwrap();
        let registry = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: true,
            container: false,
        })
        .with_test_tools(tools);
        (cache, registry)
    }

    /// A container executor for registries whose container is never reached.
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
            std::future::ready(Err("adoption never executes in the container".to_string()))
        }

        fn exec_streamed(
            &self,
            _container_id: &str,
            _script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _stdout: Sender<Vec<u8>>,
        ) -> impl std::future::Future<Output = Result<crate::ContainerExecOutcome, String>> + Send
        {
            std::future::ready(Err("adoption never executes in the container".to_string()))
        }
    }

    fn test_container_runtime(container_id: &str) -> ContainerShellRuntime {
        ContainerShellRuntime::new(
            container_id,
            crate::HostRoute::new(
                "host.docker.internal",
                std::net::IpAddr::from([127, 0, 0, 1]),
            )
            .expect("loopback is routable"),
            crate::AnyContainerExec::new(Arc::new(UnreachableContainerExec)),
        )
    }

    fn build_server() -> crate::SshServer {
        crate::SshServer {
            name: "build".to_string(),
            target: "builder@build.internal".to_string(),
        }
    }

    /// A local home, not yet prepared.
    fn local_registry() -> ShellSessionRegistry {
        ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: true,
            container: false,
        })
    }

    async fn fresh_terminal(dir: &Path) -> TerminalTool<executor_core::tokio::TokioGlobal> {
        TerminalTool::new_exact(
            dir,
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
    }

    /// A trusted manifest serving one script, which reports `tool 1.0.0` and
    /// lists the coreutils aliases, for every managed tool at version 1.0.0.
    #[cfg(unix)]
    async fn script_distribution(directory: &Path) -> ToolDistribution {
        // Invoked through a coreutils alias it runs the host's command of
        // that name, so the scripts of these tests still find `pwd` and `cat`.
        let commands = crate::ManagedTool::ALL.map(crate::ManagedTool::command);
        // As `uv`, `python install <request>` installs a copy of the script as
        // the interpreter, which reports the Python version below.
        let script = format!(
            "#!/bin/sh\n\
             case \"${{0##*/}}\" in bash)\n\
               case \"$1\" in --version) printf '%s\\n' 'GNU bash, version 1.0.0(1)-release (fixture)'; exit 0;; esac\n\
               AITHER_FIXTURE_BASH=1.0.0; export AITHER_FIXTURE_BASH; exec /bin/bash \"$@\";; esac\n\
             case \"${{0##*/}}\" in python*) printf '%s\\n' 'Python 3.13.1'; exit 0;; esac\n\
             case \"${{0##*/}}\" in {}) ;; *) PATH=/usr/bin:/bin exec \"${{0##*/}}\" \"$@\";; esac\n\
             case \"$1\" in --list) printf '%s\\n' {}; exit 0;; esac\n\
             case \"$1\" in python) PATH=/usr/bin:/bin; for request; do :; done; \
             dir=\"$UV_PYTHON_INSTALL_DIR/$request\"; mkdir -p \"$dir/bin\"; \
             cp \"$0\" \"$dir/bin/python3.13\"; chmod 755 \"$dir/bin/python3.13\"; \
             ln -s python3.13 \"$dir/bin/python3\"; exit 0;; esac\n\
             printf '%s\\n' 'tool 1.0.0'\n",
            commands.join("|"),
            crate::tool_runtime::COREUTILS_ALIASES.join(" ")
        )
        .into_bytes();
        let script = script.as_slice();
        let path = directory.join("baseline-tool");
        async_fs::write(&path, script).await.unwrap();
        let artifacts = crate::tool_runtime::tests::native_tools()
            .iter()
            .map(|&tool| crate::ToolArtifact {
                tool,
                version: "1.0.0".into(),
                target: crate::ToolTarget::native().unwrap(),
                url: crate::ToolArtifactLocation::Download(
                    url::Url::from_file_path(&path).unwrap(),
                ),
                sha256: hex::encode(crate::tool_runtime::digest(script)),
                size: script.len() as u64,
                encoding: crate::ToolArtifactFormat::Executable,
            })
            .collect::<Vec<_>>();
        let python = crate::tool_runtime::tests::python_pins_for(
            crate::tool_runtime::digest(script),
            "3.13.1",
        );
        ToolDistribution::from_json(
            &serde_json::to_vec(&serde_json::json!({
                "artifacts": artifacts,
                "host_programs": crate::test_heel::host_programs(),
                "python": python,
            }))
            .unwrap(),
        )
        .unwrap()
    }

    /// A script runs through the managed bash, not the host's `sh`: the
    /// fixture bash marks its environment with the pinned version, and the
    /// script's shell is a real bash.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn terminal_script_runs_in_the_managed_bash() {
        let tools_cache = tempfile::tempdir().unwrap();
        let tools = crate::tool_runtime::tests::bundle()
            .prepare_local(tools_cache.path())
            .await
            .unwrap();
        let prepared = local_registry().with_test_tools(tools);
        let directory = tempfile::tempdir().unwrap();
        let terminal = fresh_terminal(directory.path())
            .await
            .adopt_prepared_tools(prepared)
            .await
            .expect("a prepared registry should be adopted");
        let registry = Arc::new(ToolRegistryBuilder::new().build(terminal.output_store().clone()));
        let terminal = terminal.with_registry(registry);
        let result = terminal
            .call(
                ShellArgs {
                    description: "report the shell".to_string(),
                    script: "[[ -n $BASH_VERSION ]] && echo \"managed=$AITHER_FIXTURE_BASH bash\""
                        .to_string(),
                    interpreter: Interpreter::Shell,
                    sandbox: SandboxMode::Sandboxed,
                    location: None,
                    resolution: MediaResolution::Auto,
                    max_lines: Some(50),
                    raw: true,
                    timeout: 30,
                    background: false,
                },
                ToolContext::new(),
            )
            .await
            .expect("the script should run");
        let stdout = render_terminal_result(&result);
        assert!(!stdout.contains("Exit code:"), "{stdout}");
        assert!(stdout.contains("managed=1.0.0 bash"), "{stdout}");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn adopted_registry_runs_baseline_tools_without_version_probes() {
        // Tools pinned to a version their commands do not report: the version
        // probe of full preparation rejects them.
        let tools_cache = crate::tool_runtime::tests::unlocking_tempdir();
        let tools = crate::tool_runtime::tests::bundle_pinned("2.0.0")
            .prepare_local(tools_cache.path())
            .await
            .unwrap();
        assert!(matches!(
            tools
                .verify_local_versions(
                    executor_core::tokio::TokioGlobal,
                    crate::test_heel::pinned().await.unwrap(),
                )
                .await,
            Err(ToolRuntimeError::VersionMismatch(_))
        ));
        let prepared = local_registry().with_test_tools(tools);

        let second = tempfile::tempdir().unwrap();
        let terminal = fresh_terminal(second.path())
            .await
            .adopt_prepared_tools(prepared)
            .await
            .expect("a prepared registry should be adopted without probing");

        let registry = Arc::new(ToolRegistryBuilder::new().build(terminal.output_store().clone()));
        let terminal = terminal.with_registry(registry);
        let result = terminal
            .call(
                ShellArgs {
                    description: "run every baseline command".to_string(),
                    script: "apply_patch --version && rg --version && ast-grep --version && jq --version"
                        .to_string(),
                    interpreter: Interpreter::Shell,
                    sandbox: SandboxMode::Sandboxed,
                    location: None,
                    resolution: MediaResolution::Auto,
                    max_lines: Some(50),
                    raw: true,
                    timeout: 30,
                    background: false,
                },
                ToolContext::new(),
            )
            .await
            .expect("baseline commands should run");
        let stdout = render_terminal_result(&result);
        assert!(!stdout.contains("Exit code:"), "{stdout}");
        for command in ["apply_patch", "rg", "ast-grep", "jq"] {
            assert!(
                stdout.contains(&format!("{command} 1.0.0")),
                "{command} should run from the adopted tools; stdout: {stdout}"
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_container_home_still_gives_the_host_runnable_tools() {
        let container = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: false,
            container: true,
        })
        .with_container_runtime(test_container_runtime("aither-host-tools"))
        .with_test_container_tools(
            crate::tool_runtime::tests::container_tools(&crate::tool_runtime::tests::image_tools()),
            "aither-host-tools".to_string(),
        );
        assert!(
            matches!(
                container.local_tools(),
                Err(ToolRuntimeError::NotInitialized(_))
            ),
            "the container's own tools are not host-runnable"
        );

        let artifacts = tempfile::tempdir().unwrap();
        let distribution = script_distribution(artifacts.path()).await;
        let cache = crate::tool_runtime::tests::unlocking_tempdir();
        let host = container
            .host_tools(&distribution, cache.path())
            .await
            .expect("the distribution prepares the host target");
        let runtimes = std::fs::canonicalize(cache.path().join("runtimes")).unwrap();
        assert!(host.bin_dir().starts_with(runtimes));
        for command in ["apply_patch", "rg", "ast-grep", "jq"] {
            let output = std::process::Command::new(host.bin_dir().join(command))
                .output()
                .expect("the host runs the prepared tool");
            assert_eq!(
                String::from_utf8_lossy(&output.stdout),
                "tool 1.0.0\n",
                "{command}"
            );
        }
    }

    #[tokio::test]
    async fn adoption_rejects_a_registry_changed_after_preparation() {
        let tools_cache = crate::tool_runtime::tests::unlocking_tempdir();
        let tools = crate::tool_runtime::tests::bundle()
            .prepare_local(tools_cache.path())
            .await
            .unwrap();
        let prepared = local_registry()
            .with_test_tools(tools.clone())
            .with_test_ssh(build_server(), None, tools);
        let staging = crate::SshServer {
            name: "staging".to_string(),
            target: "deploy@staging.internal".to_string(),
        };
        let container = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: false,
            container: true,
        })
        .with_container_runtime(test_container_runtime("aither-tools-a"))
        .with_test_container_tools(
            crate::tool_runtime::tests::container_tools(&crate::tool_runtime::tests::image_tools()),
            "aither-tools-a".to_string(),
        );

        let dir = tempfile::tempdir().unwrap();
        let adopt = |sessions: ShellSessionRegistry| {
            let dir = dir.path().to_path_buf();
            async move {
                fresh_terminal(&dir)
                    .await
                    .adopt_prepared_tools(sessions)
                    .await
            }
        };

        assert!(adopt(prepared.clone()).await.is_ok());
        assert!(adopt(container.clone()).await.is_ok());
        assert!(matches!(
            adopt(local_registry()).await,
            Err(ToolRuntimeError::NotInitialized(_))
        ));
        // The server set is live: changing it changes nothing preparation
        // is bound to.
        prepared.add_ssh_server(staging).unwrap();
        assert!(prepared.remove_ssh_server("build").is_some());
        assert!(adopt(prepared.clone()).await.is_ok());
        assert!(matches!(
            adopt(container.with_container_runtime(test_container_runtime("aither-tools-b"))).await,
            Err(ToolRuntimeError::EnvironmentChanged(
                crate::PreparedEnvironment::Container
            ))
        ));
        assert!(matches!(
            adopt(prepared.with_availability(ShellRuntimeAvailability {
                local: true,
                container: true,
            }))
            .await,
            Err(ToolRuntimeError::EnvironmentChanged(
                crate::PreparedEnvironment::HomeBackend
            ))
        ));
    }

    #[tokio::test]
    async fn adoption_rechecks_this_terminals_writable_paths() {
        let (cache, prepared) = prepared_test_registry().await;
        let dir = tempfile::tempdir().unwrap();
        let exposed = fresh_terminal(dir.path())
            .await
            .with_writable_paths([cache.path()])
            .adopt_prepared_tools(prepared)
            .await;
        assert!(matches!(
            exposed,
            Err(ToolRuntimeError::WritableToolDirectory(_))
        ));
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
        let rendered = inject_env_exports("echo hi".to_string(), source.as_ref())
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
        assert!(inject_env_exports("echo hi".to_string(), source.as_ref()).is_err());
        assert_eq!(
            inject_env_exports("echo hi".to_string(), None).expect("no source passes through"),
            "echo hi"
        );
    }

    #[tokio::test]
    async fn session_runtime_environment_is_prepared_without_wrapping_the_script() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let cache = crate::tool_runtime::tests::unlocking_tempdir();
        let ready = crate::tool_runtime::tests::bundle()
            .prepare_local(cache.path())
            .await
            .unwrap();
        let environment = prepare_session_runtime_environment(dir.path(), &ready)
            .await
            .expect("session runtime environment should be prepared");
        let value = |name: &str| {
            environment
                .iter()
                .find_map(|(key, value)| (key == name).then_some(value.as_str()))
                .unwrap_or_else(|| panic!("missing session environment variable {name}"))
        };

        assert_eq!(value("HOME"), dir.path().to_str().expect("UTF-8 temp path"));
        assert_eq!(
            value("TMPDIR"),
            dir.path().join("tmp").to_str().expect("UTF-8 temp path")
        );
        assert_eq!(value("TMP"), value("TMPDIR"));
        assert_eq!(value("TEMP"), value("TMPDIR"));
        assert!(dir.path().join("tmp").is_dir());
        assert!(dir.path().join(".cache/bun").is_dir());
        assert!(dir.path().join(".cache/ms-playwright").is_dir());
        assert!(dir.path().join(".config").is_dir());
        // `uv` keeps its state in the shared directory, not in the session's HOME.
        if let Some(uv) = ready.uv_dir() {
            assert_eq!(
                value("UV_CACHE_DIR"),
                uv.join("cache").to_str().expect("UTF-8 temp path")
            );
            assert_eq!(
                value("UV_PYTHON_INSTALL_DIR"),
                uv.join("python").to_str().expect("UTF-8 temp path")
            );
            assert_eq!(value("UV_PYTHON_PREFERENCE"), "system");
            assert!(!uv.starts_with(dir.path()));
        }
    }

    /// The shared `uv` directory is writable and executable beside the
    /// read-only tools directory, and nothing else of the cache is.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_shared_uv_directory_is_a_writable_grant_apart_from_the_tools() {
        let cache = crate::tool_runtime::tests::unlocking_tempdir();
        let ready = crate::tool_runtime::tests::bundle()
            .prepare_local(cache.path())
            .await
            .unwrap();
        let uv = ready.uv_dir().expect("the test host has a managed uv");
        let grants = tool_grants(&ready).expect("grants build");
        let access_of = |path: &Path| {
            let path = std::fs::canonicalize(path).unwrap();
            grants
                .iter()
                .find(|grant| grant.resolved_path().as_path() == path)
                .map(heel::Grant::access)
        };
        assert_eq!(access_of(uv), Some(Access::WRITE | Access::EXEC));
        assert_eq!(access_of(ready.bin_dir()), Some(Access::EXEC));
        // Neither grant exposes the other.
        assert!(!ready.bin_dir().starts_with(uv) && !uv.starts_with(ready.bin_dir()));
    }

    #[tokio::test]
    async fn session_runtime_environment_sets_toolchain_homes_and_host_path() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let cache = crate::tool_runtime::tests::unlocking_tempdir();
        let cargo_home = tempfile::tempdir().unwrap();
        let rustup_home = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("CARGO_HOME", cargo_home.path());
            std::env::set_var("RUSTUP_HOME", rustup_home.path());
        }
        let ready = crate::tool_runtime::tests::bundle()
            .prepare_local(cache.path())
            .await
            .unwrap();
        let environment = prepare_session_runtime_environment(dir.path(), &ready)
            .await
            .expect("session runtime environment should be prepared");
        unsafe {
            std::env::remove_var("CARGO_HOME");
            std::env::remove_var("RUSTUP_HOME");
        }
        let value = |name: &str| {
            environment
                .iter()
                .find_map(|(key, value)| (key == name).then_some(value.as_str()))
                .unwrap_or_else(|| panic!("missing session environment variable {name}"))
        };

        assert_eq!(
            value("CARGO_HOME"),
            cargo_home.path().to_str().expect("UTF-8 temp path")
        );
        assert_eq!(
            value("RUSTUP_HOME"),
            rustup_home.path().to_str().expect("UTF-8 temp path")
        );
        let host_path = std::env::var("PATH").expect("host PATH");
        let path = value("PATH");
        assert!(
            path.starts_with(ready.bin_dir().to_str().expect("UTF-8 tools dir")),
            "the managed tools dir leads PATH: {path}"
        );
        assert!(
            path.contains(&host_path),
            "the host PATH follows the tools dir: {path}"
        );
    }

    fn output_text(output: &CommandStream) -> Option<&str> {
        match output {
            CommandStream::Inline { text } => Some(text.as_str()),
            CommandStream::Stored { preview, .. } => Some(preview.as_str()),
            CommandStream::Image { .. } => None,
        }
    }

    async fn wait_for_completed_task(
        receiver: &ShellEvents,
        task_id: &str,
    ) -> JobEnded {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Some(task) = receiver
                    .take_completed()
                    .into_iter()
                    .find(|task| task.task_id == task_id)
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
        CompletedCommand {
            stdout,
            stderr,
            exit_code,
            interruptions: Vec::new(),
        }
        .into()
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

    /// The tools bin directory a container execution passes to the wrapper.
    fn container_tools_dir() -> ContainerPath {
        crate::tool_runtime::tests::container_tools(&crate::tool_runtime::tests::image_tools())
            .bin_dir()
            .clone()
    }

    #[test]
    fn wrap_container_script_refreshes_command_hash_table() {
        let wrapped = wrap_container_script(
            "websearch \"gold price\"",
            &[String::from("websearch")],
            Some(ContainerIpcEndpoint {
                host: "host.docker.internal",
                port: 9000,
            }),
            ProcessPriority::default(),
            &container_tools_dir(),
            CONTAINER_EXEC_WORKING_DIR,
            None,
        )
        .expect("wrap script");
        assert!(wrapped.contains("hash -r;"));
        assert!(wrapped.contains("MAY_IPC_DIR"));
        assert!(wrapped.contains("HEEL_IPC_ENDPOINT"));
    }

    #[test]
    fn wrap_container_script_preserves_user_script_semantics() {
        let script = "echo '$5,040' && subagent art_direction --prompt-file tmp/prompt.md";
        let wrapped = wrap_container_script(
            script,
            &[String::from("subagent")],
            Some(ContainerIpcEndpoint {
                host: "host.docker.internal",
                port: 9000,
            }),
            ProcessPriority::default(),
            &container_tools_dir(),
            CONTAINER_EXEC_WORKING_DIR,
            None,
        )
        .expect("wrap script");
        assert!(wrapped.contains(script));
        assert!(!wrapped.contains("set -euo pipefail"));
    }

    #[test]
    fn wrap_container_script_rejects_invalid_ipc_command_names() {
        let error = wrap_container_script(
            "echo ok",
            &[String::from("bad name")],
            Some(ContainerIpcEndpoint {
                host: "host.docker.internal",
                port: 9000,
            }),
            ProcessPriority::default(),
            &container_tools_dir(),
            CONTAINER_EXEC_WORKING_DIR,
            None,
        )
        .expect_err("invalid command name must fail");
        assert!(
            error
                .to_string()
                .contains("invalid ipc command name for container wrapper")
        );
    }

    #[test]
    fn wrap_container_script_renders_expected_endpoint() {
        let wrapped = wrap_container_script(
            "environment --query deployment",
            &[String::from("environment"), String::from("session")],
            Some(ContainerIpcEndpoint {
                host: "host.containers.internal",
                port: 43123,
            }),
            ProcessPriority::default(),
            &container_tools_dir(),
            CONTAINER_EXEC_WORKING_DIR,
            None,
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
    ) -> (
        crate::tool_runtime::tests::UnlockingTempDir,
        TerminalTool<executor_core::tokio::TokioGlobal, Configured>,
    ) {
        let (cache, shell_sessions) = prepared_test_registry().await;
        let tool = TerminalTool::new_exact(
            dir,
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize");
        let tool = match approver {
            Some(approver) => tool.with_approver(approver),
            None => tool,
        }
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
            local: true,
            container: false,
        });
        let mut registry = ToolRegistryBuilder::new();
        for decision in decisions {
            registry.script_policy(Decide(decision));
        }
        let registry = Arc::new(registry.build(tool.output_store().clone()));
        (cache, tool.with_registry(registry))
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
        let (_cache, tool) =
            approval_tool(dir.path(), [ScriptDecision::Pass], Some(approvals.clone())).await;

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
        let (_cache, allowed) =
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
        let (_cache, asking) = approval_tool(
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
        let (_cache, tool) = approval_tool(dir.path(), [], Some(terminals.clone())).await;
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
        let (_cache, bare) = approval_tool(dir.path(), [], None).await;
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
    async fn uninitialized_terminal_rejects_execution_and_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let tool = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .unwrap();
        assert!(matches!(
            tool.prepared_tools_context(),
            Err(ToolRuntimeError::NotInitialized(_))
        ));
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        let tool = tool.with_registry(registry);
        assert!(matches!(
            tool.resolve_home_target(SandboxMode::Sandboxed),
            Err(TerminalError::Tools(ToolRuntimeError::NotInitialized(_)))
        ));
    }

    #[tokio::test]
    async fn child_terminal_exposes_controls_with_isolated_jobs() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
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

        let parent_registry =
            Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        let parent = tool.with_registry(parent_registry);
        let child = parent
            .subagent_without_grant()
            .await
            .expect("an empty grant is always given");
        assert_eq!(
            parent.prepared_tools_context().unwrap(),
            child.prepared_tools_context().unwrap()
        );
        assert!(child.job_registry().list().await.is_empty());
        for name in ["terminal_kill", "terminal_input", "terminal_read"] {
            assert!(child.registry().is_tool_configured(name));
        }
        let args = vec!["--task-id".to_string(), "parent".to_string()];
        let parent_output = serde_json::to_value(
            parent
                .registry()
                .query_tool_handler("terminal_read", &args)
                .await,
        )
        .unwrap();
        let child_output = serde_json::to_value(
            child
                .registry()
                .query_tool_handler("terminal_read", &args)
                .await,
        )
        .unwrap();
        assert!(parent_output.to_string().contains("running"));
        assert!(!child_output.to_string().contains("running"));
        let names = child
            .to_dyn()
            .into_entries()
            .into_iter()
            .map(|entry| entry.definition.name().to_string())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["terminal"]);
    }

    /// Picks one of the entries of its terminal's working directory, and
    /// names those entries when a call does not parse.
    struct PickTool {
        dir: PathBuf,
    }

    /// Pick an entry.
    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct PickArgs {
        /// The entry.
        entry: String,
    }

    impl aither_core::llm::Tool for PickTool {
        fn name(&self) -> std::borrow::Cow<'static, str> {
            "pick".into()
        }
        type Arguments = PickArgs;
        type Res = aither_core::llm::ToolResult;

        fn call(
            &self,
            args: Self::Arguments,
            _cx: aither_core::llm::ToolContext,
        ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
            std::future::ready(Ok(aither_core::llm::ToolResult::text(args.entry)))
        }
    }

    impl crate::CommandUsage for PickTool {
        async fn usage(&self) -> String {
            use futures_lite::StreamExt as _;
            let mut entries: Vec<String> = async_fs::read_dir(&self.dir)
                .await
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|name| name.starts_with("entry-"))
                .collect()
                .await;
            entries.sort();
            format!("entries: {}", entries.join(", "))
        }
    }

    #[tokio::test]
    async fn a_main_exclusive_command_names_its_choices_and_does_not_exist_for_a_subagent() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let mut builder = ToolRegistryBuilder::new();
        builder.main_exclusive(|main| {
            main.configure_terminal_bound_usage_tool("pick", |binding| PickTool {
                dir: binding.working_dir().to_path_buf(),
            });
        });
        let parent = tool.with_registry(Arc::new(
            builder.build(test_outputs(dir.path().join("outputs"))),
        ));

        // What the usage names is read when the call fails, not when the
        // terminal bound the command.
        std::fs::write(dir.path().join("entry-b"), "").unwrap();
        std::fs::write(dir.path().join("entry-a"), "").unwrap();
        let bare = parent.registry().query_tool_handler("pick", &[]).await;
        assert!(!bare.ok);
        assert_eq!(
            bare.error.as_deref(),
            Some(
                "missing required argument: entry\n\nentries: entry-a, entry-b\n\nRun `pick --help` for usage information."
            )
        );
        let picked = parent
            .registry()
            .query_tool_handler("pick", &["entry-a".to_string()])
            .await;
        assert_eq!(picked.payload.unwrap().render_for_cli(), "entry-a");

        let child = parent
            .subagent_without_grant()
            .await
            .expect("an empty grant is always given");
        let registry = child.registry();
        assert!(!registry.is_tool_configured("pick"));
        assert!(!registry.registered_tool_names().contains(&"pick".into()));
        assert!(registry.tool_help("pick").is_none());
        let refused = registry
            .query_tool_handler("pick", &["entry-a".to_string()])
            .await;
        assert_eq!(
            refused.error.as_deref(),
            Some("unknown command 'pick'. Run 'help' to see available commands.")
        );
        assert!(!child.to_dyn().has_command("pick"));
    }

    async fn query_control(
        registry: &ToolRegistry,
        name: &str,
        args: &[&str],
    ) -> serde_json::Value {
        let args = args.iter().map(ToString::to_string).collect::<Vec<_>>();
        serde_json::to_value(registry.query_tool_handler(name, &args).await)
            .expect("command envelope should serialize")
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

    /// A parent terminal serving a registry and a child terminal launched from
    /// it without a grant. The directories live as long as the guards.
    #[cfg(unix)]
    async fn parent_and_child_terminals() -> (
        (
            tempfile::TempDir,
            crate::tool_runtime::tests::UnlockingTempDir,
        ),
        ConfiguredTestTool,
        ConfiguredTestTool,
    ) {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
            local: true,
            container: false,
        });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        let parent = tool.with_registry(registry);
        let child = parent
            .subagent_without_grant()
            .await
            .expect("an empty grant is always given");
        ((dir, cache), parent, child)
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
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .with_id_allocator(ids.allocator())
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
            local: true,
            container: false,
        });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        let terminal = tool.with_registry(registry);

        let receiver = terminal.background_receiver();
        terminal
            .run_background(background_args("print a line", "echo leased"))
            .await
            .expect("terminal call should succeed");
        let [started] = <[_; 1]>::try_from(receiver.take_started()).expect("one started task");
        assert_eq!(ids.leased(), ["alfa", "bravo"]);
        assert_eq!(started.task_id, "alfa");
        assert_eq!(started.output_file.url, "outputs/bravo.txt");
        receiver.recv_completed().await.expect("the job completes");

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
        // The child's terminal and the commands bound to the parent lease
        // from the same allocator.
        let from_child = child.id_allocator().lease().expect("a free word");
        let from_binding = terminal
            .binding()
            .id_allocator()
            .lease()
            .expect("a free word");
        assert_eq!((&*from_child, &*from_binding), ("delta", "echo"));
        drop((from_child, from_binding));

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

    /// A configured terminal leasing its model-facing names from `ids`.
    #[cfg(unix)]
    async fn terminal_leasing_from(
        dir: &Path,
        ids: AnyIdAllocator,
    ) -> (
        crate::tool_runtime::tests::UnlockingTempDir,
        TerminalTool<executor_core::tokio::TokioGlobal, Configured>,
    ) {
        let (cache, shell_sessions) = prepared_test_registry().await;
        let tool = TerminalTool::new_exact(
            dir,
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .with_id_allocator(ids)
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
            local: true,
            container: false,
        });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        (cache, tool.with_registry(registry))
    }

    /// A foreground command gives its task id back by the time its call
    /// returns, so commands run one after another never exhaust the words.
    #[cfg(unix)]
    #[tokio::test]
    async fn foreground_commands_one_after_another_never_exhaust_the_ids() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let ids = AnyIdAllocator::new(aither_core::WordAllocator::new(["river"]));
        let (_cache, terminal) = terminal_leasing_from(dir.path(), ids.clone()).await;
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
        let (_cache, terminal) = terminal_leasing_from(dir.path(), ids.clone()).await;
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
            .expect("the job should deliver its completion")
            .expect("the completion channel stays open");
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
        assert!(matches!(
            jobs.terminal_read("river", 0, 1024).await,
            Err(crate::TerminalReadError::Failed(_))
        ));
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

    /// A child terminal's `terminal_input` and `terminal_read` reach its own
    /// job.
    #[cfg(unix)]
    #[tokio::test]
    async fn configured_child_terminal_input_and_read_address_child_jobs() {
        let (_dirs, _parent, child) = parent_and_child_terminals().await;

        // The echo job exits once its stdin line arrives, so its completion
        // event is the signal that its output exists; the read after it is the
        // first read of that task and cannot be refused as unchanged.
        let receiver = child.background_receiver();
        let result = child
            .run_background(background_args(
                "echo one stdin line",
                "read line; printf 'got %s\\n' \"$line\"",
            ))
            .await
            .expect("terminal call should succeed");
        let rendered = render_terminal_result(&result);
        assert!(
            rendered.contains("Started in background as task '"),
            "{rendered}"
        );
        let echo_id = started_task_id(&receiver);

        let input = query_control(
            child.registry(),
            "terminal_input",
            &["--task-id", &echo_id, "--input", "ping"],
        )
        .await;
        assert_eq!(input["ok"], true, "child terminal_input failed: {input}");

        // Held, so the job is still addressable while it is read.
        let _completed = wait_for_completed_task(&receiver, &echo_id).await;
        let echoed =
            query_control(child.registry(), "terminal_read", &["--task-id", &echo_id]).await;
        assert_eq!(echoed["ok"], true, "child terminal_read failed: {echoed}");
        assert!(echoed.to_string().contains("got ping"), "{echoed}");
    }

    /// A child terminal's job is invisible to the parent's controls and
    /// stopped by the child's own `terminal_kill`.
    #[cfg(unix)]
    #[tokio::test]
    async fn configured_child_terminal_kill_addresses_child_jobs() {
        let (_dirs, parent, child) = parent_and_child_terminals().await;
        let result = child
            .run_background(background_args("sleep", "sleep 30"))
            .await
            .expect("terminal call should succeed");
        let rendered = render_terminal_result(&result);
        assert!(
            rendered.contains("Started in background as task '"),
            "{rendered}"
        );
        let task_id = started_task_id(&child.background_receiver());
        let running =
            query_control(child.registry(), "terminal_read", &["--task-id", &task_id]).await;
        assert_eq!(running["ok"], true, "child terminal_read failed: {running}");
        assert!(running.to_string().contains("running"), "{running}");

        let parent_read =
            query_control(parent.registry(), "terminal_read", &["--task-id", &task_id]).await;
        assert_eq!(parent_read["ok"], false, "{parent_read}");
        assert!(
            parent_read["error"]
                .as_str()
                .is_some_and(|error| error.contains("unknown task_id")),
            "parent terminal_read must not see the child's task: {parent_read}"
        );
        let parent_kill =
            query_control(parent.registry(), "terminal_kill", &["--task-id", &task_id]).await;
        assert!(
            parent_kill
                .to_string()
                .contains("No running background task"),
            "parent terminal_kill must not reach the child's task: {parent_kill}"
        );

        let kill = query_control(child.registry(), "terminal_kill", &["--task-id", &task_id]).await;
        assert!(
            kill.to_string().contains("terminated"),
            "child terminal_kill should stop its task: {kill}"
        );
        let stopped =
            query_control(child.registry(), "terminal_read", &["--task-id", &task_id]).await;
        assert!(
            stopped.to_string().contains("was killed"),
            "child task should be reported killed: {stopped}"
        );
    }

    /// Under strict isolation a command in a `PATH` directory runs, the
    /// directory is not writable, and a directory off `PATH` stays unreadable.
    #[cfg(unix)]
    #[tokio::test]
    async fn path_directories_are_readable_and_executable_but_not_writable() {
        use std::os::unix::fs::PermissionsExt as _;
        let base = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(base.path()).unwrap();
        let on_path = root.join("on-path");
        let off_path = root.join("off-path");
        let missing = root.join("missing");
        for dir in [&on_path, &off_path] {
            std::fs::create_dir(dir).unwrap();
        }
        let tool = on_path.join("hello-tool");
        std::fs::write(&tool, "#!/bin/sh\necho hello-from-path\n").unwrap();
        std::fs::set_permissions(&tool, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::write(off_path.join("secret"), "secret\n").unwrap();

        let path = std::env::join_paths([&on_path, &missing, Path::new("relative")]).unwrap();
        let environment = vec![("PATH".to_string(), path.into_string().unwrap())];
        let grants = path_directory_grants(&environment).unwrap();
        assert_eq!(grants.len(), 1, "only the existing absolute directory");
        assert_eq!(grants[0].access(), Access::READ | Access::EXEC);

        let session = tempfile::tempdir().unwrap();
        let config = heel::SandboxConfig::builder()
            .network(DenyAll)
            .working_dir(std::fs::canonicalize(session.path()).unwrap())
            .grants(grants)
            .security(heel::SecurityConfig::interactive())
            .build();
        let sandbox =
            heel::Sandbox::with_config_and_executor(config, executor_core::tokio::TokioGlobal)
                .await
                .expect("sandbox should initialize");
        let script = format!(
            "{tool}; echo x > {on}/planted 2>/dev/null && echo planted; cat {off}/secret 2>/dev/null; true",
            tool = tool.display(),
            on = on_path.display(),
            off = off_path.display(),
        );
        let output = sandbox
            .command("/bin/sh")
            .arg("-c")
            .arg(script)
            .output()
            .await
            .expect("the sandboxed script runs");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(stdout.trim(), "hello-from-path", "{stdout}");
        assert!(!on_path.join("planted").exists());
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
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        let job_registry = tool.job_registry();
        let (startup_tx, _startup_rx) = async_channel::bounded(1);
        let result = execute_script_standalone(StandaloneExecution {
            working_dir: tool.working_dir().clone(),
            grants: Grants::new(),
            env_source: None,
            executor: executor_core::tokio::TokioGlobal,
            registry,
            job_registry,
            task_id: crate::test_ids::lease("probe"),
            execution_id: "exec-test".to_string(),
            background: false,
            carry: None,
            carried_dir: None,
            script: "/bin/pwd".to_string(),
            interpreter: Interpreter::Shell,
            mode: SandboxMode::Sandboxed,
            backend: ExecutionBackend::Local(tool.shell_sessions.local_tools().unwrap().clone()),
            stdin_alerts: None,
            lifetime: JobLifetime::Foreground {
                alerts: event_channel().0,
            },
            background_startup: BackgroundStartup::new(startup_tx),
            network: NetworkSettings::default(),
            priority: ProcessPriority::default(),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: None,
            heel: SharedHeelBinary::default(),
            expect: OutputFormat::Text,
            resolution: MediaResolution::Auto,
            outputs: tool.output_store().clone(),
            max_lines: 50,
            raw: false,
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

    /// The session base is what the terminal runs a command under: the
    /// grants name the tools and the `uv` directory with the access the
    /// terminal gives them, and a sandboxed terminal command sees exactly the
    /// environment the base lists and is confined by the live grants.
    #[cfg(unix)]
    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn session_base_is_what_the_terminal_runs_commands_under() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let writable = tempfile::tempdir().unwrap();
        let readable = tempfile::tempdir().unwrap();
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tools = shell_sessions.local_tools().unwrap().clone();
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let working_dir = tool.working_dir().clone();
        let writable_paths = vec![writable.path().to_path_buf()];
        let readable_paths = vec![readable.path().to_path_buf()];

        let base = session_base(&working_dir, &tools)
            .await
            .expect("session base builds");
        let access_of = |path: &Path| {
            let path = std::fs::canonicalize(path).unwrap();
            base.grants
                .iter()
                .find(|grant| grant.resolved_path().as_path() == path)
                .map(heel::Grant::access)
        };
        assert_eq!(access_of(tools.bin_dir()), Some(Access::EXEC));
        if let Some(uv) = tools.uv_dir() {
            assert_eq!(access_of(uv), Some(Access::WRITE | Access::EXEC));
        }

        let names = ["HOME", "TMPDIR", "XDG_CACHE_HOME", "PATH", "PYTHONPATH"];
        let written = writable.path().join("written");
        let refused = readable.path().join("refused");
        let prints = names
            .iter()
            .map(|name| format!("echo \"${name}\""))
            .collect::<Vec<_>>()
            .join("; ");
        let script = format!(
            "{prints}; touch {} && echo write-ok; touch {} && echo read-write-ok; true",
            written.display(),
            refused.display(),
        );
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        let job_registry = tool.job_registry();
        let (startup_tx, _startup_rx) = async_channel::bounded(1);
        let result = execute_script_standalone(StandaloneExecution {
            working_dir,
            grants: test_grants(
                GitDirAccess::ReadWrite,
                writable_paths,
                readable_paths,
                Vec::new(),
            ),
            env_source: None,
            executor: executor_core::tokio::TokioGlobal,
            registry,
            job_registry,
            task_id: crate::test_ids::lease("policy"),
            execution_id: "exec-policy".to_string(),
            background: false,
            carry: None,
            carried_dir: None,
            script,
            interpreter: Interpreter::Shell,
            mode: SandboxMode::Sandboxed,
            backend: ExecutionBackend::Local(tools),
            stdin_alerts: None,
            lifetime: JobLifetime::Foreground {
                alerts: event_channel().0,
            },
            background_startup: BackgroundStartup::new(startup_tx),
            network: NetworkSettings::default(),
            priority: ProcessPriority::default(),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: None,
            heel: SharedHeelBinary::default(),
            expect: OutputFormat::Text,
            resolution: MediaResolution::Auto,
            outputs: tool.output_store().clone(),
            max_lines: 50,
            raw: false,
        })
        .await
        .expect("the sandboxed script runs");
        let stdout = output_text(&result.0.stdout).expect("inline stdout");
        let mut lines = stdout.lines();
        for name in names {
            let expected = base
                .environment
                .iter()
                .rev()
                .find_map(|(key, value)| (key == name).then_some(value.as_str()))
                .unwrap_or_else(|| panic!("the policy lists {name}"));
            assert_eq!(lines.next(), Some(expected), "{name} in the command");
        }
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
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tools = shell_sessions.local_tools().unwrap().clone();
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        let (startup_tx, _startup_rx) = async_channel::bounded(1);
        let result = execute_script_standalone(StandaloneExecution {
            working_dir: tool.working_dir().clone(),
            grants: test_grants(git_dir, writable_paths, readable_paths, read_denied_paths),
            env_source: None,
            executor: executor_core::tokio::TokioGlobal,
            registry,
            job_registry: tool.job_registry(),
            task_id: crate::test_ids::lease("grants"),
            execution_id: "exec-grants".to_string(),
            background: false,
            carry: None,
            carried_dir: None,
            script,
            interpreter: Interpreter::Shell,
            mode: SandboxMode::Sandboxed,
            backend: ExecutionBackend::Local(tools),
            stdin_alerts: None,
            lifetime: JobLifetime::Foreground {
                alerts: event_channel().0,
            },
            background_startup: BackgroundStartup::new(startup_tx),
            network: NetworkSettings::default(),
            priority: ProcessPriority::default(),
            #[cfg(all(unix, feature = "jobserver"))]
            jobserver: None,
            heel: SharedHeelBinary::default(),
            expect: OutputFormat::Text,
            resolution: MediaResolution::Auto,
            outputs: tool.output_store().clone(),
            max_lines: 200,
            raw: false,
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

    /// `uv` marks its cache buckets with a `.git` entry: the prepared
    /// runtime's cache takes that, while a granted repository still refuses
    /// git's config. Judged by outcome, since platforms differ on whether the
    /// bare `.git` directory itself is refused.
    #[cfg(unix)]
    #[tokio::test]
    async fn uv_cache_may_hold_git_entries_and_a_granted_repository_may_not() {
        let base = tempfile::tempdir().unwrap();
        let granted = std::fs::canonicalize(base.path()).unwrap().join("repo");
        std::fs::create_dir(&granted).unwrap();
        let script = format!(
            "mkdir -p \"$UV_CACHE_DIR/x\" && mkdir \"$UV_CACHE_DIR/x/.git\" 2>&1 && echo cache-written; \
             mkdir -p {repo}/.git 2>/dev/null; touch {repo}/.git/config 2>/dev/null && echo repo-written; true",
            repo = granted.display(),
        );
        let stdout = sandboxed_stdout(vec![granted.clone()], Vec::new(), script).await;
        assert!(
            stdout.lines().any(|line| line == "cache-written"),
            "{stdout}"
        );
        assert!(!stdout.contains("repo-written"), "{stdout}");
        assert!(!granted.join(".git/config").exists(), "{stdout}");
    }

    /// Cargo's git cache keeps the `.git/hooks` rule off, and nothing else of
    /// a toolchain home is exempt.
    #[cfg(unix)]
    #[tokio::test]
    async fn cargo_git_cache_is_the_only_toolchain_home_denylist_exception() {
        let (_base, main, _linked) = repository_with_linked_worktree();
        let session = tempfile::tempdir().unwrap();
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tools = shell_sessions.local_tools().unwrap().clone();
        let denylist = session_write_denylist(session.path(), &tools);
        // The cache directory appears with the first `uv` run; its parent
        // is canonical from preparation.
        let uv_cache = tools.uv_environment().unwrap().cache;
        let cargo_git = host_toolchain_homes()
            .find(|(name, _)| *name == "CARGO_HOME")
            .map(|(_, home)| std::fs::canonicalize(home).unwrap().join("git"))
            .expect("the test host has a cargo home");
        assert_eq!(
            denylist.exempt(),
            [
                std::fs::canonicalize(session.path()).unwrap(),
                cargo_git.clone(),
                uv_cache
            ]
        );

        let checkout = cargo_git.join(format!(
            "checkouts/aither-denylist-test-{}/rev",
            std::process::id()
        ));
        std::fs::create_dir_all(&checkout).unwrap();
        host_git(&checkout, &["init", "-q"]);
        let script = format!(
            "echo x > {cache}/hooks/update 2>&1 && echo cache-written; \
             echo x > {repo}/.git/hooks/update 2>/dev/null && echo repo-written; true",
            cache = checkout.join(".git").display(),
            repo = main.display(),
        );
        let stdout = sandboxed_stdout(vec![main.clone()], Vec::new(), script).await;
        std::fs::remove_dir_all(checkout.parent().unwrap()).unwrap();
        assert_eq!(stdout.lines().last(), Some("cache-written"), "{stdout}");
        assert!(!stdout.contains("repo-written"), "{stdout}");
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
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
            local: true,
            container: false,
        });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        let tool = tool.with_registry(registry);

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
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
            local: true,
            container: false,
        });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        let tool = tool.with_registry(registry);
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
            .expect("the job should deliver its completion")
            .expect("the completion channel stays open");
        assert_eq!(completed.task_id, task_id);
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
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
            local: true,
            container: false,
        });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        let tool = tool.with_registry(registry);
        let background_receiver = tool.background_receiver();
        let input_tool = InputTerminalTool::new(tool.job_registry());

        let result = tool
            .run_background(ShellArgs {
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
                InputTerminalArgs {
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

    #[cfg(unix)]
    #[tokio::test]
    async fn killing_background_terminal_kills_descendant_processes() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
            local: true,
            container: false,
        });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        let tool = tool.with_registry(registry);
        let child_pid_path = dir.path().join("child.pid");

        let result = tool
            .run_background(ShellArgs {
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
        let (_cache, tool) = configured_test_terminal(
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
        tool: &TerminalTool<executor_core::tokio::TokioGlobal, Configured>,
        script: &str,
    ) -> String {
        let result = tool
            .run_background(ShellArgs {
                background: true,
                ..terminal_args("background job", script)
            })
            .await
            .expect("terminal should start the job");
        match result.command().map(|command| &command.outcome) {
            Some(CommandOutcome::Running { task_id, .. }) => task_id.clone(),
            other => panic!("background job needs a task id, got {other:?}"),
        }
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
        let (_cache, tool) = configured_test_terminal(
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
        let (_cache, tool) = configured_test_terminal(
            dir.path(),
            Duration::from_secs(10),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await;
        let receiver = tool.background_receiver();
        let result = tool
            .run_background(python_args(
                "import time\nprint('ready', flush=True)\ntime.sleep(60)",
            ))
            .await
            .expect("python should start detached");
        let task_id = match result.command().map(|command| &command.outcome) {
            Some(CommandOutcome::Running { task_id, .. }) => task_id.clone(),
            other => panic!("a background job needs a task id, got {other:?}"),
        };
        assert_eq!(started_task_id(&receiver), task_id);

        let jobs = tool.job_registry().list().await;
        let job = jobs
            .iter()
            .find(|job| job.task_id == task_id)
            .expect("the job is recorded");
        assert_eq!(job.interpreter, Interpreter::Python);
        assert!(job.script.contains("time.sleep(60)"), "{}", job.script);

        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            let read =
                query_control(tool.registry(), "terminal_read", &["--task-id", &task_id]).await;
            if read.to_string().contains("ready") {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the job never printed: {read}"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        assert!(tool.job_registry().kill_by_task_id(&task_id).await);
        wait_for_completed_task(&receiver, &task_id).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unsafe_python_asks_for_approval_with_its_code() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let refusing = Approvals::denying("no");
        let (_cache, tool) = approval_tool(dir.path(), [], Some(refusing.clone())).await;

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
        let (_cache, tool) = configured_test_terminal(
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

    /// A foreground container command gets a live stdin pipe like every
    /// other job: no EOF shortcut. The exec's stdin channel stays open and
    /// its blocked watch channel is wired, so a reader can be fed through
    /// `terminal_input` once the call promotes.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn container_foreground_command_gets_a_live_stdin_pipe() {
        /// A container executor that asserts stdin is a watched live pipe.
        struct ForegroundStdinPipeExec;

        impl crate::ContainerExec for ForegroundStdinPipeExec {
            async fn exec(
                &self,
                _container_id: &str,
                _script: &str,
                _working_dir: &str,
                _kill_rx: Receiver<()>,
                stdin_rx: Receiver<Vec<u8>>,
                _lifetime: ExecLifetime,
            ) -> Result<ContainerExecOutcome, String> {
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

            fn exec_streamed(
                &self,
                _container_id: &str,
                _script: &str,
                _working_dir: &str,
                _kill_rx: Receiver<()>,
                _stdin_rx: Receiver<Vec<u8>>,
                _stdout: Sender<Vec<u8>>,
            ) -> impl std::future::Future<Output = Result<ContainerExecOutcome, String>> + Send
            {
                std::future::ready(Err("a foreground command does not stream".to_string()))
            }
        }

        let dir = tempfile::tempdir().expect("temp dir should be created");
        let registry = Arc::new(ToolRegistryBuilder::new().build(test_outputs(dir.path())));
        let (job_registry, job_registry_service) =
            job_registry_channel(crate::output_policy::fixture::policy());
        tokio::spawn(async move { job_registry_service.serve().await });
        let container_tools =
            crate::tool_runtime::tests::container_tools(&crate::tool_runtime::tests::image_tools());
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
        let (alert_tx, _alert_rx) = event_channel();
        let (_pid, output) = execute_container_background(ContainerBackgroundExecution {
            executor: executor_core::tokio::TokioGlobal,
            env_source: &None,
            registry: &registry,
            task_id: &crate::test_ids::lease("probe"),
            execution_id: "exec-test",
            script: "cat",
            interpreter: Interpreter::Shell,
            mode: SandboxMode::Sandboxed,
            container_runtime: &runtime,
            prepared_tools: &container_tools,
            ipc_commands: &[],
            job_registry: &job_registry,
            stdin_alerts: Some(StdinAlerts {
                tx: alert_tx.clone(),
                task_id: "probe".to_string(),
                script: "cat".to_string(),
            }),
            lifetime: JobLifetime::Foreground { alerts: alert_tx },
            background_startup: &BackgroundStartup::new(startup_tx),
            priority: ProcessPriority::default(),
            confinement: None,
            carry: None,
            #[cfg(all(unix, feature = "jobserver"))]
            job_fifo: None,
        })
        .await
        .expect("a foreground container exec should complete");
        assert_eq!(output.status.code(), Some(0));
    }

    /// A subagent's command in a container gets write access to its
    /// jobserver fifo on top of its confinement; an unconfined command gets
    /// no grants at all.
    #[cfg(all(unix, feature = "jobserver"))]
    #[test]
    fn container_job_fifo_is_granted_to_a_confined_command() {
        let fifo = ContainerPath::new("/tmp/aither-jobserver-1.fifo").unwrap();
        let confinement = ContainerConfinement {
            working_dir: ContainerPath::new("/workspace/.subagents/a").unwrap(),
            writable: vec![ContainerPath::new("/workspace/.subagents/a").unwrap()],
        };
        assert_eq!(job_confinement(None, Some(&fifo)), None);
        let confined = job_confinement(Some(&confinement), Some(&fifo)).unwrap();
        let tools_dir = container_tools_dir();
        let wrapped = wrap_container_script(
            "make",
            &[],
            None,
            ProcessPriority::default(),
            &tools_dir,
            "/workspace",
            Some(&confined),
        )
        .unwrap();
        assert!(
            wrapped.contains(
                "--grant '/workspace/.subagents/a=rw' --grant '/tmp/aither-jobserver-1.fifo=rw'"
            ),
            "{wrapped}"
        );
        assert_eq!(
            job_confinement(Some(&confinement), None).as_ref(),
            Some(&confinement)
        );
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
    }

    #[cfg(all(unix, feature = "jobserver"))]
    impl crate::ContainerExec for CapturingExec {
        async fn exec(
            &self,
            _container_id: &str,
            script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _lifetime: ExecLifetime,
        ) -> Result<ContainerExecOutcome, String> {
            self.scripts.send(script.to_string()).await.unwrap();
            Ok(ContainerExecOutcome::Completed {
                output: std::process::Output {
                    status: std::os::unix::process::ExitStatusExt::from_raw(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                },
                killed: Vec::new(),
            })
        }

        async fn exec_streamed(
            &self,
            _container_id: &str,
            _script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            stdin_rx: Receiver<Vec<u8>>,
            stdout: Sender<Vec<u8>>,
        ) -> Result<ContainerExecOutcome, String> {
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
        let registry = Arc::new(ToolRegistryBuilder::new().build(test_outputs(dir.path())));
        let (job_registry, job_registry_service) =
            job_registry_channel(crate::output_policy::fixture::policy());
        tokio::spawn(async move { job_registry_service.serve().await });
        let container_tools =
            crate::tool_runtime::tests::container_tools(&crate::tool_runtime::tests::image_tools());
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
            })),
        );
        let job_fifo = crate::container_jobserver::open_container_job_fifo(
            &broker,
            &runtime,
            &container_tools.bin_dir().join("python3"),
            executor_core::tokio::TokioGlobal,
        )
        .await
        .expect("the relay reports ready");
        let fifo_path = job_fifo.path().clone();
        let (startup_tx, _startup_rx) = async_channel::bounded(1);
        let (alert_tx, _alert_rx) = event_channel();
        execute_container_background(ContainerBackgroundExecution {
            executor: executor_core::tokio::TokioGlobal,
            env_source: &None,
            registry: &registry,
            task_id: &crate::test_ids::lease("probe"),
            execution_id: "exec-test",
            script: "make",
            interpreter: Interpreter::Shell,
            mode: SandboxMode::Sandboxed,
            container_runtime: &runtime,
            prepared_tools: &container_tools,
            ipc_commands: &[],
            job_registry: &job_registry,
            stdin_alerts: None,
            lifetime: JobLifetime::Foreground { alerts: alert_tx },
            background_startup: &BackgroundStartup::new(startup_tx),
            priority: ProcessPriority::default(),
            confinement: None,
            carry: None,
            job_fifo: Some(job_fifo),
        })
        .await
        .expect("the container exec completes");

        let script = scripts.recv().await.expect("the command ran");
        let flags = format!("'-j --jobserver-auth=fifo:{fifo_path}'");
        for name in ["MAKEFLAGS", "CARGO_MAKEFLAGS", "MFLAGS"] {
            assert!(
                script.contains(&format!("export {name}={flags}\n")),
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
        let (_cache, tool) = configured_test_terminal(
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

        InputTerminalTool::new(tool.job_registry())
            .call(
                InputTerminalArgs {
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
        let (_cache, tool) = configured_test_terminal(
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

    /// A configured terminal tool with prepared local tools and the given
    /// shortened limits. Returns the tool and the cache owning its tools.
    #[cfg(unix)]
    async fn configured_test_terminal(
        dir: &Path,
        foreground_wait: Duration,
        timeout_cap: Duration,
        alert_after: Duration,
    ) -> (
        crate::tool_runtime::tests::UnlockingTempDir,
        TerminalTool<executor_core::tokio::TokioGlobal, Configured>,
    ) {
        let (cache, tool) =
            unconfigured_test_terminal(dir, foreground_wait, timeout_cap, alert_after).await;
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        (cache, tool.with_registry(registry))
    }

    /// [`configured_test_terminal`] before its registry is bound.
    async fn unconfigured_test_terminal(
        dir: &Path,
        foreground_wait: Duration,
        timeout_cap: Duration,
        alert_after: Duration,
    ) -> (
        crate::tool_runtime::tests::UnlockingTempDir,
        TerminalTool<executor_core::tokio::TokioGlobal, Unconfigured>,
    ) {
        let (cache, shell_sessions) = prepared_test_registry().await;
        let tool = TerminalTool::new_exact(
            dir,
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_foreground_wait(foreground_wait)
        .with_timeout_cap(timeout_cap)
        .with_long_job_alert_after(alert_after);
        (cache, tool)
    }

    /// A foreground command still running at the window promotes to a
    /// background task instead of dying; its `timeout` then kills it with
    /// its descendants.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn foreground_command_still_running_at_the_window_promotes_and_times_out_with_descendants()
     {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
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
        assert_eq!(
            info.status,
            JobStatus::Killed {
                cause: crate::KillCause::User,
                delivery: crate::Delivery::Pending
            },
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
        let (_cache, tool) = configured_test_terminal(
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

    /// The directory the sandbox may execute programs from.
    #[cfg(unix)]
    fn exec_dir<E>(tool: &TerminalTool<E, Configured>) -> PathBuf {
        tool.shell_sessions
            .local_tools()
            .expect("the terminal has local tools")
            .bin_dir()
            .clone()
    }

    /// Builds the `setsid` helper into `dir` and returns its path. The sandbox
    /// executes only what lies under a granted directory, so `dir` is the
    /// prepared tools' [`exec_dir`].
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
        let (_cache, tool) = configured_test_terminal(
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
                assert_eq!(killed.omitted, 0);
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
        let (_cache, tool) = configured_test_terminal(
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
        tool: &TerminalTool<executor_core::tokio::TokioGlobal, Configured>,
        script: &str,
        timeout: u64,
    ) -> (String, i32) {
        let result = tool
            .run_background(ShellArgs {
                background: true,
                timeout,
                ..terminal_args("background job", script)
            })
            .await
            .expect("terminal should start the background job");
        let task_id = match result.command().map(|command| &command.outcome) {
            Some(CommandOutcome::Running { task_id, .. }) => task_id.clone(),
            other => panic!("background job needs a task id, got {other:?}"),
        };
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
        let (_cache, tool) = configured_test_terminal(
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
        let (_cache, tool) = configured_test_terminal(
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
        let (_cache, tool) = configured_test_terminal(
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
        assert_eq!(
            info.status,
            JobStatus::Killed {
                cause: crate::KillCause::User,
                delivery: crate::Delivery::Pending
            },
            "{completed:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagents_cleanup_kills_a_descendant_that_outlived_the_shell() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
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
        let (_cache, tool) = configured_test_terminal(
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
        assert_eq!(
            info.status,
            JobStatus::Killed {
                cause: crate::KillCause::User,
                delivery: crate::Delivery::Pending
            },
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
        let (_cache, tool) = configured_test_terminal(
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

        assert_eq!(completed.kill, Some(crate::KillCause::User));
        let kill = completed
            .timeout_kill
            .expect("a job killed at its timeout carries the times");
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
        let (_cache, tool) = configured_test_terminal(
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
        assert_eq!(completed.kill, Some(crate::KillCause::User));
        assert_eq!(completed.timeout_kill, None);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_foreground_command_killed_at_its_timeout_reports_its_utc_times() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
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
        let (_cache, tool) = configured_test_terminal(
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

    /// A terminal tool with prepared local tools and `priority` applied. The
    /// returned `TempDir` owns the prepared tools' cache; it must outlive the
    /// tool.
    #[cfg(unix)]
    async fn prioritized_terminal_tool(
        dir: &Path,
        priority: ProcessPriority,
    ) -> (
        crate::tool_runtime::tests::UnlockingTempDir,
        TerminalTool<executor_core::tokio::TokioGlobal, Configured>,
    ) {
        let (cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir,
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_process_priority(priority);
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        (cache, tool.with_registry(registry))
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
        let (_tools_cache, tool) =
            prioritized_terminal_tool(dir.path(), test_nice_priority()).await;

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
        let (_tools_cache, tool) =
            prioritized_terminal_tool(dir.path(), test_nice_priority()).await;
        let background_receiver = tool.background_receiver();
        let input_tool = InputTerminalTool::new(tool.job_registry());

        // `exec` keeps `$$` — the pid the probe wrote — on the interactive
        // shell the session then drives.
        let result = tool
            .run_background(ShellArgs {
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
            })
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
                InputTerminalArgs {
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
        let (_tools_cache, tool) =
            prioritized_terminal_tool(dir.path(), test_nice_priority()).await;
        let background_receiver = tool.background_receiver();

        let result = tool
            .run_background(ShellArgs {
                background: true,
                ..pid_probe_args()
            })
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
        let wrapped = wrap_container_script(
            "echo ok",
            &[String::from("websearch")],
            Some(ContainerIpcEndpoint {
                host: "host.docker.internal",
                port: 9000,
            }),
            priority,
            &tools_dir,
            "/workspace",
            None,
        )
        .expect("wrap script");
        assert!(wrapped.contains(
            "\"$MAY_IPC_BIN\" run --isolation permissive --working-dir '/workspace' \
             --env-passthrough PATH --env-passthrough HOME"
        ));
        assert!(wrapped.contains(&executable));
        assert!(wrapped.contains(
            "--ipc-endpoint 'tcp://host.docker.internal:9000' --grant \"$MAY_IPC_DIR=rx\" \
             --nice 5 --io-priority background -- sh -c 'echo ok'"
        ));
        assert!(wrapped.contains("HEEL_IPC_ENDPOINT"));

        let bare = wrap_container_script(
            "echo ok",
            &[],
            None,
            priority,
            &tools_dir,
            "/workspace",
            None,
        )
        .expect("wrap script");
        assert!(bare.contains("MAY_HEEL_BIN=\"$(command -v heel || true)\""));
        assert!(bare.contains(
            "\"$MAY_HEEL_BIN\" run --isolation permissive --working-dir '/workspace' \
             --env-passthrough PATH --env-passthrough HOME"
        ));
        assert!(bare.contains(&executable));
        assert!(bare.contains("--nice 5 --io-priority background -- sh -c 'echo ok'"));

        let normal = wrap_container_script(
            "echo ok",
            &[],
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
            &[],
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
            &[],
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

    /// Probe tool for the container acceptance test: it answers an IPC call
    /// with the marker it was invoked with, so a request that actually reached
    /// the live handler is unmistakable.
    #[derive(Debug, Deserialize, JsonSchema)]
    struct ProbeEchoArgs {
        /// Marker echoed back to the caller.
        marker: String,
    }

    struct ProbeEchoTool;

    impl Tool for ProbeEchoTool {
        fn name(&self) -> Cow<'static, str> {
            "probe_echo".into()
        }

        type Arguments = ProbeEchoArgs;
        type Res = ToolResult;

        fn call(
            &self,
            arguments: Self::Arguments,
            _cx: ToolContext,
        ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
            std::future::ready(Ok(ToolResult::text(format!(
                "marker: {}",
                arguments.marker
            ))))
        }
    }

    /// Host side of the container IPC transport on every platform: a client
    /// speaking the bridge's framing over loopback TCP reaches the registry's
    /// live handler and gets its envelope back.
    #[tokio::test(flavor = "multi_thread")]
    async fn container_ipc_bridge_round_trips_a_command_over_tcp() {
        let outputs = tempfile::tempdir().expect("outputs dir should be created");
        let mut builder = ToolRegistryBuilder::new();
        builder.configure_tool(ProbeEchoTool);
        let registry = Arc::new(builder.build(test_outputs(outputs.path())));
        let loopback = std::net::IpAddr::from([127, 0, 0, 1]);
        let bridge =
            start_container_ipc_bridge(executor_core::tokio::TokioGlobal, registry, None, loopback)
                .expect("bridge should start on loopback");

        let mut stream = Async::<StdTcpStream>::connect(SocketAddr::new(loopback, bridge.port()))
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

    /// Scratch `ubuntu:24.04` container used by the acceptance test; removed
    /// on drop.
    #[cfg(all(unix, aither_container_acceptance))]
    struct TestDockerContainer(String);

    #[cfg(all(unix, aither_container_acceptance))]
    impl TestDockerContainer {
        /// Starts a container with `heel_bin` mounted at `/usr/local/bin/heel`,
        /// the staged tool bundle mounted read-only where the image carries
        /// it, and `host.docker.internal` resolving to the docker host gateway,
        /// and prepares the `/workspace` directory container executions run
        /// in. Returns `None` when docker cannot start the container.
        fn start(heel_bin: &Path, tools: &crate::MountSpec) -> Option<Self> {
            let name = format!("aither-test-{}", uuid::Uuid::new_v4().simple());
            let mount = format!("{}:/usr/local/bin/heel:ro", heel_bin.display());
            let tools_mount = format!("{}:{}:ro", tools.host_path.display(), tools.container_path);
            let output = std::process::Command::new("docker")
                .args([
                    "run",
                    "-d",
                    "--name",
                    &name,
                    "--add-host=host.docker.internal:host-gateway",
                    "-v",
                    &mount,
                    "-v",
                    &tools_mount,
                    "ubuntu:24.04",
                    "tail",
                    "-f",
                    "/dev/null",
                ])
                .output()
                .ok()?;
            if !output.status.success() {
                return None;
            }
            let prepared = std::process::Command::new("docker")
                .args(["exec", &name, "mkdir", "-p", CONTAINER_EXEC_WORKING_DIR])
                .output()
                .is_ok_and(|output| output.status.success());
            if !prepared {
                let _ = std::process::Command::new("docker")
                    .args(["rm", "-f", &name])
                    .output();
                return None;
            }
            Some(Self(name))
        }
    }

    #[cfg(all(unix, aither_container_acceptance))]
    impl Drop for TestDockerContainer {
        fn drop(&mut self) {
            let _ = std::process::Command::new("docker")
                .args(["rm", "-f", &self.0])
                .output();
        }
    }

    /// The docker `bridge` network's gateway as the host owns it — the
    /// address `host.docker.internal` resolves to inside the container via
    /// `--add-host …:host-gateway`, so it is the address the IPC bridge binds.
    #[cfg(all(unix, aither_container_acceptance))]
    async fn container_bridge_address(
        client: &bollard::Docker,
        container_id: &str,
    ) -> std::net::IpAddr {
        client
            .inspect_container(
                container_id,
                None::<bollard::query_parameters::InspectContainerOptions>,
            )
            .await
            .expect("container acceptance should inspect the container")
            .network_settings
            .and_then(|settings| settings.networks)
            .and_then(|mut networks| networks.remove("bridge"))
            .and_then(|endpoint| endpoint.gateway)
            .expect("container acceptance requires the bridge network gateway")
            .parse::<std::net::IpAddr>()
            .expect("the bridge gateway should parse as an IP address")
    }

    /// A Linux `heel` binary for the container under test: `HEEL_BIN` wins,
    /// then a `heel` already on `PATH`.
    #[cfg(all(unix, aither_container_acceptance))]
    fn container_test_heel_binary() -> Option<PathBuf> {
        std::env::var_os("HEEL_BIN")
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .or_else(|| {
                std::process::Command::new("sh")
                    .args(["-c", "command -v heel"])
                    .output()
                    .ok()
                    .filter(|output| output.status.success())
                    .and_then(|output| String::from_utf8(output.stdout).ok())
                    .map(|resolved| PathBuf::from(resolved.trim()))
            })
            .filter(|path| path.is_file())
    }

    /// End-to-end container path: `TerminalTool` execs the wrapped script in a
    /// real container through `BollardContainerExec`, and the script fires a
    /// real `heel ipc` request at the live host bridge.
    ///
    /// The wrapped `heel run` carries the configured priority through heel's
    /// own sandbox configuration, and `--ipc-endpoint` relays the nested
    /// sandbox's local IPC socket to the host bridge, so `probe_echo` answers
    /// from the live handler rather than failing at the transport.
    ///
    /// Compiled only with `--cfg aither_container_acceptance`: the test needs
    /// a real docker daemon and a Linux `heel` binary, which the dedicated CI
    /// job provides and the feature matrices do not. A missing daemon or
    /// binary fails the test — it is never skipped silently.
    /// Starts the acceptance container with the test bundle staged as an
    /// image build stages it and mounted read-only where the image carries
    /// it, so the wrapped script runs a prepared command from that directory.
    /// The returned directory holds the staged bundle and must outlive the
    /// container.
    #[cfg(all(unix, aither_container_acceptance))]
    async fn start_acceptance_container() -> (
        TestDockerContainer,
        crate::tool_runtime::ImageTools,
        crate::tool_runtime::tests::UnlockingTempDir,
    ) {
        let docker_ready = std::process::Command::new("docker")
            .arg("info")
            .output()
            .is_ok_and(|output| output.status.success());
        assert!(
            docker_ready,
            "container acceptance requires a docker daemon"
        );
        let heel_bin = container_test_heel_binary()
            .expect("container acceptance requires a Linux heel binary (set HEEL_BIN)");
        let image = crate::tool_runtime::tests::image_tools();
        let bundle = crate::tool_runtime::tests::bundle();
        let tools_cache = crate::tool_runtime::tests::unlocking_tempdir();
        // Stage the executables under the image's cache key — exactly what
        // `stage_image_tools` produces — and mount the directory where the
        // image carries it.
        let prepared = bundle
            .prepare_local_sync(tools_cache.path(), &image.cache_key().unwrap(), None)
            .expect("test bundle should stage for the image");
        let staged = prepared
            .bin_dir()
            .parent()
            .expect("prepared binaries have a parent directory")
            .to_path_buf();
        let tools_mount =
            crate::MountSpec::read_only(&staged, image.container_directory().unwrap());
        let container = TestDockerContainer::start(&heel_bin, &tools_mount)
            .expect("container acceptance could not start ubuntu:24.04");
        (container, image, tools_cache)
    }

    /// The shell sessions of a terminal whose home is the acceptance
    /// container, reaching the host IPC bridge through the docker gateway.
    #[cfg(all(unix, aither_container_acceptance))]
    async fn acceptance_shell_sessions(
        container: &TestDockerContainer,
        image: &crate::tool_runtime::ImageTools,
        workspace: &Path,
    ) -> ShellSessionRegistry {
        let client = Arc::new(
            bollard::Docker::connect_with_local_defaults().expect("docker client should connect"),
        );
        // The bridge binds the container network's gateway as the host owns
        // it (docker0), because native Linux Docker resolves
        // `host.docker.internal` (via `--add-host …:host-gateway`) to exactly
        // that address.
        let bind = container_bridge_address(&client, &container.0).await;
        println!("ipc bridge bind: {bind}");
        let exec = crate::AnyContainerExec::new(Arc::new(crate::BollardContainerExec::new(client)));
        let runtime = ContainerShellRuntime::new(
            container.0.clone(),
            crate::HostRoute::new("host.docker.internal", bind).expect("the gateway is routable"),
            exec,
        )
        .with_workspace_mount(MountRoot::new(
            workspace,
            ContainerPath::new(CONTAINER_EXEC_WORKING_DIR).expect("the workspace is absolute"),
        ));

        // The container backend needs its own prepared tools: the registry
        // entry names the container and its `/opt/aither/tools` bin dir, which
        // the wrapped script puts on `PATH` and heel must grant execute on.
        let container_tools = crate::tool_runtime::tests::container_tools(image);
        ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: false,
            container: true,
        })
        .with_container_runtime(runtime)
        .with_test_container_tools(container_tools, container.0.clone())
    }

    /// The streamed exec of the real docker backend delivers a command's
    /// stdout in order and whole across chunks, reports its exit status, and
    /// kills the command when its kill switch is dropped.
    #[cfg(all(unix, aither_container_acceptance))]
    #[tokio::test]
    async fn container_streamed_exec_streams_stdout_reports_status_and_kills_on_drop() {
        use crate::ContainerExec as _;
        let (container, _bundle, _tools_cache) = start_acceptance_container().await;
        let client = Arc::new(
            bollard::Docker::connect_with_local_defaults().expect("docker client should connect"),
        );
        let exec = crate::BollardContainerExec::new(client);

        let (_kill_tx, kill_rx) = async_channel::bounded::<()>(1);
        let (stdin_tx, stdin_rx) = async_channel::bounded::<Vec<u8>>(1);
        drop(stdin_tx);
        let (out_tx, out_rx) = async_channel::bounded::<Vec<u8>>(8);
        let streaming = exec.exec_streamed(
            &container.0,
            "head -c 300000 /dev/zero | tr '\\0' 'x'; exit 3",
            "/",
            kill_rx,
            stdin_rx,
            out_tx,
        );
        let collect = async {
            let mut total = 0;
            while let Ok(chunk) = out_rx.recv().await {
                assert!(chunk.iter().all(|byte| *byte == b'x'));
                total += chunk.len();
            }
            total
        };
        let (outcome, total) = futures::join!(streaming, collect);
        assert_eq!(total, 300_000, "every byte arrives");
        match outcome.expect("the exec runs") {
            ContainerExecOutcome::Completed { output, .. } => {
                assert_eq!(output.status.code(), Some(3));
            }
            ContainerExecOutcome::Killed => panic!("nothing killed it"),
        }

        // A reader that goes away ends the command.
        let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
        let (stdin_tx, stdin_rx) = async_channel::bounded::<Vec<u8>>(1);
        drop(stdin_tx);
        let (out_tx, out_rx) = async_channel::bounded::<Vec<u8>>(1);
        drop(out_rx);
        let outcome = exec
            .exec_streamed(&container.0, "yes", "/", kill_rx, stdin_rx, out_tx)
            .await;
        drop(kill_tx);
        assert!(
            matches!(outcome, Ok(ContainerExecOutcome::Killed) | Err(_)),
            "a gone reader stops the command: {outcome:?}"
        );
    }

    /// In a real container the main agent's command runs unconfined, while a
    /// subagent's runs under its per-command sandbox: it writes its own space
    /// and nothing else of the workspace.
    #[cfg(all(unix, aither_container_acceptance))]
    #[tokio::test]
    async fn container_subagent_is_confined_while_the_main_agent_is_not() {
        let (container, bundle, _tools_cache) = start_acceptance_container().await;
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let shell_sessions = acceptance_shell_sessions(&container, &bundle, dir.path()).await;
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let parent = tool.with_registry(Arc::new(
            ToolRegistryBuilder::new().build(test_outputs(dir.path())),
        ));
        let child = parent
            .subagent_without_grant()
            .await
            .expect("an empty grant is always given");
        // The acceptance container does not really mount the host workspace,
        // so the subagent's space is created inside it too.
        let root = dunce::canonicalize(dir.path()).expect("workspace root should resolve");
        let space = child
            .working_dir()
            .strip_prefix(&root)
            .expect("the space is inside the workspace");
        let container_space = format!("{CONTAINER_EXEC_WORKING_DIR}/{}", space.display());
        let made = std::process::Command::new("docker")
            .args(["exec", &container.0, "mkdir", "-p", &container_space])
            .output()
            .expect("docker exec should run");
        assert!(made.status.success(), "{made:?}");

        let host_workspace_write =
            |name: &str| format!("touch {CONTAINER_EXEC_WORKING_DIR}/{name}");
        assert_eq!(
            exit_code_of(&parent, &host_workspace_write("main.txt")).await,
            0,
            "the main agent is unconfined in its container"
        );
        assert_ne!(
            exit_code_of(&child, &host_workspace_write("sub.txt")).await,
            0,
            "a subagent cannot write the workspace root"
        );
        assert_eq!(
            exit_code_of(&child, &format!("touch {container_space}/own.txt")).await,
            0,
            "a subagent writes its own space"
        );
    }

    #[cfg(all(unix, aither_container_acceptance))]
    #[tokio::test]
    async fn container_command_runs_with_configured_nice_and_reaches_ipc() {
        let (container, bundle, _tools_cache) = start_acceptance_container().await;

        let shell_sessions = acceptance_shell_sessions(&container, &bundle, Path::new("/")).await;

        let dir = tempfile::tempdir().expect("temp dir should be created");
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_process_priority(ProcessPriority::new(
            CpuPriority::Nice(NiceLevel::new(5).expect("nice level 5 is in range")),
            IoPriority::Normal,
        ));
        let mut builder = ToolRegistryBuilder::new();
        builder.configure_tool(ProbeEchoTool);
        let registry = Arc::new(builder.build(tool.output_store().clone()));
        let tool = tool.with_registry(registry);

        let result = tool
            .call(
                ShellArgs {
                    description: "probe container priority and IPC transport".to_string(),
                    script: concat!(
                        "printf 'nice=%s\\n' \"$(nice)\"; ",
                        "jq --version; ",
                        "probe_echo --marker pong; ",
                        "printf 'probe_exit=%s\\n' \"$?\""
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
                },
                ToolContext::new(),
            )
            .await
            .expect("container terminal call should succeed");
        let stdout = render_terminal_result(&result);
        assert!(
            !stdout.contains("Exit code:"),
            "container command failed; stdout: {stdout}"
        );
        // The CI job runs the test with --nocapture so the container's output
        // — the applied nice level and the live probe reply — lands in the
        // job log.
        println!("container script output:\n{stdout}");
        assert!(
            stdout.contains("nice=5"),
            "container command should run at the configured nice level; stdout: {stdout}"
        );
        assert!(
            stdout.contains("jq 1.0.0"),
            "a prepared command should run from the read-only tools mount; stdout: {stdout}"
        );
        assert!(
            stdout.contains("pong"),
            "probe_echo should answer from the live handler; stdout: {stdout}"
        );
        assert!(
            stdout.contains("probe_exit=0"),
            "heel ipc should exit 0 through the relayed endpoint; stdout: {stdout}"
        );
    }

    // ---- subagent authority -------------------------------------------------

    type ConfiguredTestTool = TerminalTool<executor_core::tokio::TokioGlobal, Configured>;

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
    async fn authority_terminal(
        dir: &Path,
    ) -> (
        crate::tool_runtime::tests::UnlockingTempDir,
        ConfiguredTestTool,
    ) {
        configured_test_terminal(
            dir,
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await
    }

    /// A script the sandboxed command writes into its working directory and
    /// chmods executes — writable space also carries EXEC.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_script_written_into_the_working_dir_executes() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = authority_terminal(dir.path()).await;
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

    /// The host Rust toolchain works end to end inside the local sandbox:
    /// cargo runs, and `cargo test` executes the binary it just built under
    /// the writable working dir.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn the_host_rust_toolchain_builds_and_tests_a_crate() {
        assert!(
            std::process::Command::new("cargo")
                .arg("--version")
                .output()
                .is_ok_and(|output| output.status.success()),
            "cargo must be on the host PATH for this test"
        );
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = authority_terminal(dir.path()).await;

        assert_eq!(
            exit_code_of(&tool, "cargo --version").await,
            0,
            "cargo is reachable inside the sandbox"
        );

        let krate = dir.path().join("probe");
        std::fs::create_dir_all(krate.join("src")).unwrap();
        std::fs::write(
            krate.join("Cargo.toml"),
            "[package]\nname = \"probe\"\nversion = \"0.0.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        std::fs::write(
            krate.join("src/lib.rs"),
            "#[test]\nfn it_runs() { assert_eq!(1 + 1, 2); }\n",
        )
        .unwrap();
        let script = format!(
            "cd '{}' && CARGO_TARGET_DIR='{}' cargo test",
            krate.display(),
            krate.join("target").display()
        );
        assert_eq!(
            exit_code_of(&tool, &script).await,
            0,
            "cargo test executes the binary it built under the working dir"
        );
    }

    /// Credential files inside the granted toolchain homes stay unreadable.
    #[cfg(all(unix, target_os = "macos"))]
    #[tokio::test(flavor = "multi_thread")]
    async fn cargo_credentials_inside_the_granted_home_stay_unreadable() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = authority_terminal(dir.path()).await;
        let code = exit_code_of(&tool, "cat ~/.cargo/credentials.toml").await;
        assert_ne!(
            code, 0,
            "a credential file under ~/.cargo stays denied inside the sandbox"
        );
    }
    /// Launches a subagent of `parent` the way its launcher does, with a
    /// launch history of its own.
    async fn launch(
        parent: &ConfiguredTestTool,
        grant: SubagentGrant,
    ) -> Result<ConfiguredTestTool, SubagentGrantError> {
        parent
            .subagent(&mut LaunchHistory::default(), grant, None)
            .await
    }

    /// What a [`CloneSpaces`] was asked to prepare, as `(space, launcher,
    /// caller)`.
    type SeenSpace = (PathBuf, PathBuf, PathBuf);

    /// How a test's [`SubagentSpaces`] fills a space.
    #[derive(Clone, Copy)]
    enum Filling {
        /// Clones the source repository into `repo` and names it.
        Clone,
        /// Writes a file into the space, then fails.
        Fail,
        /// Names the launcher's working directory as the working copy.
        Outside,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("the session has no clone to give")]
    struct NoClone;

    /// Prepares each space from `source`, the launcher's repository, and
    /// records what it was asked.
    struct CloneSpaces {
        source: PathBuf,
        filling: Filling,
        seen: async_channel::Sender<SeenSpace>,
    }

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
    async fn spaced_terminal(
        dir: &Path,
        filling: Filling,
    ) -> (
        crate::tool_runtime::tests::UnlockingTempDir,
        ConfiguredTestTool,
        async_channel::Receiver<SeenSpace>,
    ) {
        let source = dir.join("repo");
        std::fs::create_dir_all(&source).unwrap();
        crate::test_git::git(&source, &["init", "-q", "--initial-branch=main"]);
        crate::test_git::git(&source, &["commit", "-q", "--allow-empty", "-m", "base"]);
        let (seen, seen_rx) = async_channel::unbounded();
        let (cache, tool) = unconfigured_test_terminal(
            dir,
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await;
        let tool = tool.with_subagent_spaces(CloneSpaces {
            source,
            filling,
            seen,
        });
        let registry = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
        (cache, tool.with_registry(registry), seen_rx)
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
        let (_cache, parent, seen) = spaced_terminal(dir.path(), Filling::Clone).await;
        let caller = dir.path().join("repo");
        let child = parent
            .subagent(
                &mut LaunchHistory::default(),
                SubagentGrant::new(),
                Some(&caller),
            )
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
        let dyn_child = child.clone().to_dyn();
        assert_eq!(
            dyn_child.prepared_space().unwrap().working_copies(),
            std::slice::from_ref(&clone),
            "the bundle a subagent's agent gets names its working copies"
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
            .subagent(&mut LaunchHistory::default(), SubagentGrant::new(), None)
            .await
            .expect("a nested space is prepared");
        let (nested_space, nested_launcher, nested_caller) = seen.try_recv().unwrap();
        assert_eq!(&nested_space, nested.working_dir());
        assert_eq!(nested_launcher, space, "a subagent launches from its space");
        assert_eq!(nested_caller, space, "the caller defaults to the launcher");
        drop((nested, dyn_child, child));

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
        let (_cache, parent, seen) = spaced_terminal(dir.path(), Filling::Fail).await;
        let error = parent
            .binding()
            .launcher()
            .create(SubagentGrant::new(), None, None, None)
            .await
            .expect_err("the launch fails");
        let (asked, _, _) = seen.try_recv().unwrap();
        let TerminalToolFactoryError::Grant(SubagentGrantError::Preparation { space, source }) =
            &error
        else {
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
        let (_cache, parent, seen) = spaced_terminal(dir.path(), Filling::Outside).await;
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
        let (_cache, parent) = authority_terminal(dir.path()).await;
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
        let (_cache, tool) = authority_terminal(dir.path()).await;
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
        let (_cache, tool) = authority_terminal(dir.path()).await;
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
        let (_cache, tool) = authority_terminal(dir.path()).await;
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

        let (_cache, tool) = authority_terminal(&link).await;
        assert_eq!(tool.working_dir(), &real);
        assert_eq!(exit_code_of(&tool, "mkdir sub && cd sub").await, 0);
        assert_eq!(pwd_of(&tool).await, real.join("sub"));
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagents_cd_does_not_carry_over() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, parent) = authority_terminal(dir.path()).await;
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
        crate::tool_runtime::tests::UnlockingTempDir,
        ConfiguredTestTool,
        async_channel::Receiver<(ScriptDir, Carry)>,
    ) {
        let (cache, tool) = unconfigured_test_terminal(
            dir,
            Duration::from_secs(30),
            Duration::from_secs(60),
            Duration::from_secs(600),
        )
        .await;
        let (ends_tx, ends) = async_channel::unbounded();
        let mut registry = ToolRegistryBuilder::new();
        registry.script_policy(RecordCarries(ends_tx));
        let registry = Arc::new(registry.build(tool.output_store().clone()));
        (cache, tool.with_registry(registry), ends)
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
        let (_cache, tool, ends) = carry_terminal(dir.path()).await;
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
        let (_cache, tool, ends) = carry_terminal(dir.path()).await;
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
        let (_cache, tool, ends) = carry_terminal(dir.path()).await;
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
        let (_cache, tool, ends) = carry_terminal(dir.path()).await;
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
        let (_cache, tool, ends) = carry_terminal(dir.path()).await;
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
        let (_cache, tool) = authority_terminal(dir.path()).await;
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
        let (_cache, parent) = authority_terminal(dir.path()).await;
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
        _cache: crate::tool_runtime::tests::UnlockingTempDir,
    }

    #[cfg(unix)]
    impl CommitSetup {
        async fn new(namespace: &str) -> Self {
            let dir = tempfile::tempdir().expect("temp dir should be created");
            let worktree =
                crate::subagent::tests::repository_with_linked_worktree(dir.path(), namespace);
            let (cache, parent) = authority_terminal(dir.path()).await;
            let common = dunce::canonicalize(dir.path().join("repository/.git")).unwrap();
            let entry = common.join("worktrees").join(namespace);
            Self {
                worktree: dunce::canonicalize(worktree).unwrap(),
                subagent: entry.join("subagent"),
                common,
                entry,
                parent,
                dir,
                _cache: cache,
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

    /// The main agent's `git` in a committing worktree resolves the
    /// worktree's host-owned repository entry through the terminal's shim:
    /// no program the subagent's repository config names — fsmonitor,
    /// hooks, filters — and no `.git` file it replaced decides what runs.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn the_main_agents_git_in_a_committing_worktree_runs_nothing_the_subagent_chose() {
        use std::fmt::Write as _;
        use std::os::unix::fs::PermissionsExt as _;
        let setup = CommitSetup::new("guarded").await;
        let _child = setup.committing_child(&setup.parent).await;

        // A program only subagent-chosen configuration can reach.
        let marker = setup.dir.path().join("subagent-ran");
        let program = setup.dir.path().join("ran-by-git.sh");
        std::fs::write(
            &program,
            format!("#!/bin/sh\nprintf ran >> '{}'\n", marker.display()),
        )
        .unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        let hooks = setup.dir.path().join("evil-hooks");
        std::fs::create_dir(&hooks).unwrap();
        let pre_commit = hooks.join("pre-commit");
        std::fs::write(
            &pre_commit,
            format!("#!/bin/sh\nprintf hook >> '{}'\n", marker.display()),
        )
        .unwrap();
        std::fs::set_permissions(&pre_commit, std::fs::Permissions::from_mode(0o755)).unwrap();

        // Its repository's configuration names the program everywhere a
        // program can be named; the worktree's attributes bind the filter.
        let mut config = std::fs::read_to_string(setup.subagent.join("config")).unwrap();
        write!(
            config,
            "\n[core]\n\tfsmonitor = {}\n\thooksPath = {}\n[filter \"pwn\"]\n\tclean = {}\n\trequired = true\n",
            program.display(),
            hooks.display(),
            program.display()
        )
        .unwrap();
        std::fs::write(setup.subagent.join("config"), config).unwrap();
        std::fs::write(setup.worktree.join(".gitattributes"), "* filter=pwn\n").unwrap();

        let worktree = setup.worktree.display().to_string();
        for script in [
            format!("git -C '{worktree}' status"),
            format!("cd '{worktree}' && git status"),
            format!(
                "git -C '{worktree}' add -A && \
                 git -C '{worktree}' -c user.name=a -c user.email=a@a -c commit.gpgSign=false \
                 commit --allow-empty -m x"
            ),
        ] {
            assert_eq!(
                exit_code_of(&setup.parent, &script).await,
                0,
                "the ordinary git command should keep working: {script}"
            );
        }
        assert!(
            !marker.exists(),
            "no program of the subagent's choosing ran through the worktree's repository"
        );

        // A replaced `.git` file resolves nothing either: the shim acts on
        // the worktree's path, not on what its `.git` says.
        std::fs::write(
            setup.worktree.join(".git"),
            format!("gitdir: {}\n", setup.subagent.display()),
        )
        .unwrap();
        assert_eq!(
            exit_code_of(&setup.parent, &format!("git -C '{worktree}' status")).await,
            0
        );
        assert!(
            !marker.exists(),
            "a replaced .git file chose no program either"
        );

        // Git anywhere else is untouched.
        let repository = dunce::canonicalize(setup.dir.path().join("repository")).unwrap();
        assert_eq!(
            exit_code_of(
                &setup.parent,
                &format!("git -C '{}' log --oneline -1", repository.display())
            )
            .await,
            0,
            "git in unrelated repositories still reaches them"
        );
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
        let (_cache, parent) = authority_terminal(dir.path()).await;

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
        let (_cache, parent) = authority_terminal(dir.path()).await;

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

    #[derive(Debug, Deserialize, JsonSchema)]
    struct LaunchProbeArgs {
        /// A directory to grant the launched subagent.
        #[serde(default)]
        worktree: Option<String>,
    }

    /// Launches a subagent from the terminal it is bound to and reports
    /// where it works.
    struct LaunchProbe {
        binding: TerminalBinding,
    }

    impl Tool for LaunchProbe {
        fn name(&self) -> Cow<'static, str> {
            "launch_probe".into()
        }

        type Arguments = LaunchProbeArgs;
        type Res = ToolResult;

        async fn call(
            &self,
            arguments: Self::Arguments,
            _cx: ToolContext,
        ) -> aither_core::Result<Self::Res> {
            let grant = arguments
                .worktree
                .into_iter()
                .fold(SubagentGrant::new(), SubagentGrant::with_worktree);
            let launched = self
                .binding
                .launcher()
                .create(grant, None, None, None)
                .await?;
            Ok(ToolResult::Text {
                text: launched.working_dir().display().to_string(),
            })
        }
    }

    /// A subagent that runs the launching command launches from its own
    /// terminal: the child lives in its space and gets nothing it lacks.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_subagent_running_the_launch_command_launches_with_its_own_authority() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let granted = dir.path().join("granted");
        let sibling = dir.path().join("sibling");
        std::fs::create_dir_all(&granted).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();
        let tool = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let mut builder = ToolRegistryBuilder::new();
        builder.configure_terminal_bound_tool("launch_probe", |binding: &TerminalBinding| {
            LaunchProbe {
                binding: binding.clone(),
            }
        });
        let parent = tool.with_registry(Arc::new(
            builder.build(test_outputs(dir.path().join("outputs"))),
        ));

        // The main agent launches the subagent through the command.
        let sibling_arg = sibling.to_str().unwrap();
        let from_main = query_control(
            parent.registry(),
            "launch_probe",
            &["--worktree", sibling_arg],
        )
        .await;
        assert_eq!(from_main["ok"], true, "{from_main}");

        let child = launch(&parent, SubagentGrant::new().with_worktree(&granted))
            .await
            .unwrap();
        let refused = query_control(
            child.registry(),
            "launch_probe",
            &["--worktree", sibling_arg],
        )
        .await;
        assert_eq!(refused["ok"], false, "{refused}");
        assert!(
            refused["error"]
                .as_str()
                .is_some_and(|error| error.contains("is not writable by the agent launching")),
            "{refused}"
        );
        let nested = query_control(
            child.registry(),
            "launch_probe",
            &["--worktree", granted.to_str().unwrap()],
        )
        .await;
        assert_eq!(nested["ok"], true, "{nested}");
        let grandchild = nested["payload"].to_string();
        let child_space = child.working_dir().display().to_string();
        assert!(
            grandchild.contains(&child_space),
            "the grandchild works inside its caller's space: {grandchild} vs {child_space}"
        );
    }

    /// Stands in for a host program inside sandboxed modes: answers with a
    /// refusal naming itself.
    #[cfg(unix)]
    struct SandboxedUname;

    #[cfg(unix)]
    #[derive(Debug, Deserialize, JsonSchema)]
    struct SandboxedUnameArgs {}

    #[cfg(unix)]
    impl Tool for SandboxedUname {
        fn name(&self) -> Cow<'static, str> {
            "uname".into()
        }

        type Arguments = SandboxedUnameArgs;
        type Res = ToolResult;

        fn call(
            &self,
            _arguments: Self::Arguments,
            _cx: ToolContext,
        ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
            std::future::ready(Ok(ToolResult::text("sandboxed uname refused")))
        }
    }

    /// A command registered for sandboxed modes answers a sandboxed script,
    /// while an unsafe script resolves the same name on the host's `PATH`.
    #[cfg(unix)]
    #[tokio::test]
    async fn sandboxed_only_command_leaves_unsafe_scripts_the_host_program() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .with_approver(ApproveAll)
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let mut builder = ToolRegistryBuilder::new();
        builder.sandboxed_only(|sandboxed| sandboxed.configure_tool(SandboxedUname));
        let tool = tool.with_registry(Arc::new(
            builder.build(test_outputs(dir.path().join("outputs"))),
        ));

        let sandboxed = tool
            .call(
                terminal_args("sandboxed uname", "uname"),
                ToolContext::new(),
            )
            .await
            .expect("sandboxed call should succeed");
        let sandboxed = render_terminal_result(&sandboxed);
        assert!(
            sandboxed.contains("sandboxed uname refused"),
            "a sandboxed script reaches the command: {sandboxed}"
        );

        let unsafe_run = tool
            .call(
                ShellArgs {
                    sandbox: SandboxMode::Unsandboxed,
                    ..terminal_args("host uname", "uname -s; ipc uname")
                },
                ToolContext::new(),
            )
            .await
            .expect("unsafe call should succeed");
        let unsafe_run = render_terminal_result(&unsafe_run);
        let host = std::process::Command::new("uname")
            .arg("-s")
            .output()
            .expect("the host runs uname");
        let host = String::from_utf8(host.stdout).expect("uname prints UTF-8");
        assert!(
            unsafe_run.contains(host.trim()),
            "an unsafe script runs the host program: {unsafe_run}"
        );
        assert!(
            !unsafe_run.contains("sandboxed uname refused"),
            "an unsafe script never reaches the command, not even through `ipc`: {unsafe_run}"
        );
    }

    /// A terminal that approves unsafe scripts, with the managed tools adopted.
    #[cfg(unix)]
    async fn unsafe_terminal(
        dir: &Path,
        shell_sessions: ShellSessionRegistry,
    ) -> TerminalTool<executor_core::tokio::TokioGlobal, Configured> {
        let tool = TerminalTool::new_exact(
            dir,
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .with_approver(ApproveAll)
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        tool.with_registry(Arc::new(
            ToolRegistryBuilder::new().build(test_outputs(dir.join("outputs"))),
        ))
    }

    /// Runs an approved unsafe `script` to its end and renders the result.
    #[cfg(unix)]
    async fn run_unsafe(
        tool: &TerminalTool<executor_core::tokio::TokioGlobal, Configured>,
        script: &str,
    ) -> String {
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
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool = unsafe_terminal(session.path(), shell_sessions).await;

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
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool = unsafe_terminal(session.path(), shell_sessions).await;

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
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool = unsafe_terminal(session.path(), shell_sessions).await;

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
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool = unsafe_terminal(session.path(), shell_sessions).await;

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
    async fn a_subagent_of_a_subagent_gets_the_subagent_registry() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let mut builder = ToolRegistryBuilder::new();
        builder.main_only(|reserved| reserved.configure_tool(ProbeEchoTool));
        let parent = tool.with_registry(Arc::new(
            builder.build(test_outputs(dir.path().join("outputs"))),
        ));
        assert_eq!(parent.registry().role(), crate::CommandRole::Main);

        let child = launch(&parent, SubagentGrant::new()).await.unwrap();
        let grandchild = launch(&child, SubagentGrant::new()).await.unwrap();
        for terminal in [&child, &grandchild] {
            assert_eq!(terminal.registry().role(), crate::CommandRole::Subagent);
            let refused =
                query_control(terminal.registry(), "probe_echo", &["--marker", "x"]).await;
            assert_eq!(refused["ok"], false, "{refused}");
            assert!(
                refused["error"]
                    .as_str()
                    .is_some_and(|error| error.contains("reserved for the main agent")),
                "{refused}"
            );
        }
        assert!(grandchild.working_dir().starts_with(child.working_dir()));
        let served = parent.clone().to_dyn();
        assert!(served.has_command("probe_echo"));
        assert_eq!(served.role(), crate::CommandRole::Main);
        assert_eq!(parent.binding().role(), crate::CommandRole::Main);
        assert_eq!(child.binding().role(), crate::CommandRole::Subagent);
        let child = child.to_dyn();
        assert!(!child.has_command("probe_echo"));
        assert_eq!(child.role(), crate::CommandRole::Subagent);
    }

    // ---- terminal runner ----------------------------------------------------

    /// Runs a script on the terminal serving it and answers with how the
    /// script ended and what it printed.
    #[cfg(unix)]
    struct CheckProbe {
        runner: TerminalRunner,
    }

    #[cfg(unix)]
    #[derive(Debug, Deserialize, JsonSchema)]
    struct CheckProbeArgs {
        /// The directory the script starts in.
        dir: String,
        /// The script.
        script: String,
        /// Seconds before the script is killed.
        timeout: u64,
    }

    #[cfg(unix)]
    impl Tool for CheckProbe {
        fn name(&self) -> Cow<'static, str> {
            "check_probe".into()
        }

        type Arguments = CheckProbeArgs;
        type Res = ToolResult;

        async fn call(
            &self,
            arguments: Self::Arguments,
            _cx: ToolContext,
        ) -> aither_core::Result<Self::Res> {
            let result = self
                .runner
                .run(TerminalScript::new(
                    arguments.script,
                    arguments.dir,
                    Duration::from_secs(arguments.timeout),
                ))
                .await?;
            let answer = match (&result.outcome, &result.stdout) {
                (CommandOutcome::Exited { code }, CommandStream::Inline { text }) => {
                    format!("exit={code} stdout={text}")
                }
                (CommandOutcome::Killed { .. }, _) => "killed".to_string(),
                other => panic!("a run ends exited with inline output or killed: {other:?}"),
            };
            Ok(ToolResult::text(answer))
        }
    }

    /// Calls `check_probe` on `terminal` and returns its answer.
    #[cfg(unix)]
    async fn run_check_probe(
        terminal: &ConfiguredTestTool,
        dir: &Path,
        script: &str,
        timeout: u64,
    ) -> String {
        let timeout = timeout.to_string();
        let answer = query_control(
            terminal.registry(),
            "check_probe",
            &[
                "--dir",
                dir.to_str().unwrap(),
                "--script",
                script,
                "--timeout",
                &timeout,
            ],
        )
        .await;
        assert_eq!(answer["ok"], true, "{answer}");
        answer["payload"]["content"].as_str().unwrap().to_string()
    }

    /// A terminal-bound command runs scripts on the terminal serving it,
    /// under that terminal's policy as it stands at the run — a grant made
    /// after the registry was bound included — and the runner ends with the
    /// terminal.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_terminal_bound_command_runs_a_script_under_its_terminals_policy() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let clone = tempfile::tempdir().expect("clone dir should be created");
        let read_only = tempfile::tempdir().expect("read-only dir should be created");
        let (_cache, shell_sessions) = prepared_test_registry().await;
        let tool = TerminalTool::new_exact(
            dir.path(),
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_readable_paths([read_only.path()]);
        let mut builder = ToolRegistryBuilder::new();
        builder.configure_terminal_bound_tool("check_probe", |binding: &TerminalBinding| {
            CheckProbe {
                runner: binding.runner().clone(),
            }
        });
        let terminal = tool.with_registry(Arc::new(
            builder.build(test_outputs(dir.path().join("outputs"))),
        ));
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

        let runner = terminal.binding().runner().clone();
        let unsafe_run = runner
            .run(
                TerminalScript::new("true", &clone_dir, Duration::from_secs(30))
                    .with_mode(SandboxMode::Unsandboxed),
            )
            .await;
        assert!(
            matches!(
                unsafe_run,
                Err(TerminalRunError::Terminal(TerminalError::NoApprover(_)))
            ),
            "an unsafe run is put to an approver, and this terminal has none: {unsafe_run:?}"
        );

        drop(terminal);
        let gone = runner
            .run(TerminalScript::new(
                "true",
                &clone_dir,
                Duration::from_secs(30),
            ))
            .await;
        assert!(
            matches!(gone, Err(TerminalRunError::TerminalGone)),
            "the runner ends with its terminal: {gone:?}"
        );
    }

    // ---- call identity ------------------------------------------------------

    use aither_core::llm::{AgentIdentity, SubagentIdentity, ToolCallId};

    /// Answers with the identity its call carries, and reports it to the test
    /// the moment it arrives. With `hold` set it then waits for the test's go
    /// signal, so two calls can be made to overlap inside their handlers.
    struct IdentityProbe {
        entered: async_channel::Sender<CallIdentity>,
        hold: Option<Receiver<()>>,
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    struct IdentityProbeArgs {}

    impl Tool for IdentityProbe {
        fn name(&self) -> Cow<'static, str> {
            "identity_probe".into()
        }

        type Arguments = IdentityProbeArgs;
        type Res = ToolResult;

        async fn call(
            &self,
            _arguments: Self::Arguments,
            cx: ToolContext,
        ) -> aither_core::Result<Self::Res> {
            let identity = cx.identity().expect("a terminal call identifies itself");
            self.entered.send(identity.clone()).await.unwrap();
            if let Some(hold) = &self.hold {
                hold.recv().await.unwrap();
            }
            Ok(ToolResult::text(format!("seen={}", identity.call())))
        }
    }

    /// Reports the identity of every command call the registry serves.
    struct IdentityObserver(async_channel::Sender<(String, Option<CallIdentity>)>);

    impl crate::CommandObserver for IdentityObserver {
        async fn command_called(&self, call: &crate::CommandCall<'_>) {
            self.0
                .send((call.name.to_string(), call.call.cloned()))
                .await
                .unwrap();
        }
    }

    fn call_identity(id: &str, agent: AgentIdentity) -> CallIdentity {
        CallIdentity::new(ToolCallId::new(id), agent)
    }

    /// Fails the test instead of hanging it when `future` does not finish.
    async fn within_deadline<T>(future: impl std::future::Future<Output = T>) -> T {
        futures_lite::future::or(async { Some(future.await) }, async {
            async_io::Timer::after(Duration::from_secs(60)).await;
            None
        })
        .await
        .expect("the probe should finish within its deadline")
    }

    #[cfg(unix)]
    async fn identity_terminal(
        dir: &Path,
        probe: IdentityProbe,
        observer: async_channel::Sender<(String, Option<CallIdentity>)>,
    ) -> (
        crate::tool_runtime::tests::UnlockingTempDir,
        ConfiguredTestTool,
    ) {
        let (cache, shell_sessions) = prepared_test_registry().await;
        let tool = TerminalTool::new_exact(
            dir,
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let mut builder = ToolRegistryBuilder::new();
        builder.configure_tool(probe);
        builder.observe_commands(crate::AnyCommandObserver::new(IdentityObserver(observer)));
        (
            cache,
            tool.with_registry(Arc::new(builder.build(test_outputs(dir.join("outputs"))))),
        )
    }

    /// A command a `terminal` call's script runs is handed that call's id and
    /// agent, and so is the registry's observer.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn an_ipc_command_receives_the_identity_of_its_terminal_call() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (entered, entered_rx) = async_channel::unbounded();
        let (observed, observed_rx) = async_channel::unbounded();
        let (_cache, tool) = identity_terminal(
            dir.path(),
            IdentityProbe {
                entered,
                hold: None,
            },
            observed,
        )
        .await;
        let subagent = AgentIdentity::Subagent(SubagentIdentity::new(
            "explore",
            Some(call_identity("call_parent", AgentIdentity::Main)),
        ));
        let identity = call_identity("call_a", subagent);

        let result = within_deadline(tool.call(
            terminal_args("identity probe", "identity_probe"),
            ToolContext::new().with_identity(identity.clone()),
        ))
        .await
        .expect("terminal call should succeed");

        assert!(
            render_terminal_result(&result).contains("seen=call_a"),
            "{}",
            render_terminal_result(&result)
        );
        assert_eq!(entered_rx.recv().await.unwrap(), identity);
        assert_eq!(
            observed_rx.recv().await.unwrap(),
            ("identity_probe".to_string(), Some(identity))
        );
    }

    /// Two overlapping `terminal` calls each get their own id in the command
    /// their script runs, even though both commands are in flight at once.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_terminal_calls_deliver_their_own_ids() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (entered, entered_rx) = async_channel::unbounded();
        let (hold_tx, hold) = async_channel::unbounded();
        let (observed, _observed_rx) = async_channel::unbounded();
        let (_cache, tool) = identity_terminal(
            dir.path(),
            IdentityProbe {
                entered,
                hold: Some(hold),
            },
            observed,
        )
        .await;

        let run = |id: &'static str| {
            let tool = &tool;
            async move {
                let result = tool
                    .call(
                        terminal_args("identity probe", "identity_probe"),
                        ToolContext::new().with_identity(call_identity(id, AgentIdentity::Main)),
                    )
                    .await
                    .expect("terminal call should succeed");
                render_terminal_result(&result)
            }
        };
        let release = async {
            // Both handlers are inside their command before either answers.
            let mut inside = [
                entered_rx.recv().await.unwrap(),
                entered_rx.recv().await.unwrap(),
            ];
            inside.sort_by(|a, b| a.call().as_str().cmp(b.call().as_str()));
            assert_eq!(
                inside
                    .iter()
                    .map(|identity| identity.call().as_str())
                    .collect::<Vec<_>>(),
                ["call_a", "call_b"]
            );
            hold_tx.send(()).await.unwrap();
            hold_tx.send(()).await.unwrap();
        };
        let (a, b, ()) = Box::pin(within_deadline(async {
            futures::join!(run("call_a"), run("call_b"), release)
        }))
        .await;

        assert!(a.contains("seen=call_a") && !a.contains("call_b"), "{a}");
        assert!(b.contains("seen=call_b") && !b.contains("call_a"), "{b}");
    }

    /// A command run by a backgrounded job is told the id of the call that
    /// started the job, long after that call returned.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_backgrounded_job_keeps_the_id_of_the_call_that_started_it() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (entered, entered_rx) = async_channel::unbounded();
        let (observed, observed_rx) = async_channel::unbounded();
        let (_cache, tool) = identity_terminal(
            dir.path(),
            IdentityProbe {
                entered,
                hold: None,
            },
            observed,
        )
        .await;
        let identity = call_identity("call_bg", AgentIdentity::Main);

        let mut args = terminal_args("identity probe", "identity_probe");
        args.background = true;
        within_deadline(tool.call(args, ToolContext::new().with_identity(identity.clone())))
            .await
            .expect("terminal call should succeed");

        assert_eq!(within_deadline(entered_rx.recv()).await.unwrap(), identity);
        assert_eq!(
            within_deadline(observed_rx.recv()).await.unwrap().1,
            Some(identity)
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
        let container_tools =
            crate::tool_runtime::tests::container_tools(&crate::tool_runtime::tests::image_tools());
        let shell_sessions = ShellSessionRegistry::new(ShellRuntimeAvailability {
            local: false,
            container: true,
        })
        .with_container_runtime(runtime)
        .with_test_container_tools(container_tools, "aither-test-confinement".to_string());
        let tool = TerminalTool::new_exact(
            dir,
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted");
        let parent = tool.with_registry(Arc::new(
            ToolRegistryBuilder::new().build(test_outputs(dir)),
        ));
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
        assert!(
            script.contains(&format!(
                "export {}='{}'",
                crate::memory::JOB_MARKER_ENV,
                job.task_id
            )),
            "{script}"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_local_terminal_judges_no_container() {
        let dir = tempfile::tempdir().expect("temp dir should be created");
        let (_cache, tool) = configured_test_terminal(
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
        assert!(sub.contains("\"$MAY_IPC_BIN\" run --working-dir"), "{sub}");
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

    /// A configured local terminal in `dir` whose registry `configure` set
    /// up, for the command policy tests.
    #[cfg(unix)]
    async fn policy_tool(
        dir: &Path,
        configure: impl FnOnce(&mut ToolRegistryBuilder),
    ) -> (
        crate::tool_runtime::tests::UnlockingTempDir,
        TerminalTool<executor_core::tokio::TokioGlobal, Configured>,
    ) {
        let (cache, shell_sessions) = prepared_test_registry().await;
        let tool: TerminalTool<executor_core::tokio::TokioGlobal> = TerminalTool::new_exact(
            dir,
            executor_core::tokio::TokioGlobal,
            crate::limits::fixture::limits(),
            crate::output_policy::fixture::policy(),
            TestWording::any(),
        )
        .await
        .expect("terminal tool should initialize")
        .adopt_prepared_tools(shell_sessions)
        .await
        .expect("prepared tools should be adopted")
        .with_shell_runtime_availability(crate::ShellRuntimeAvailability {
            local: true,
            container: false,
        });
        let mut registry = ToolRegistryBuilder::new();
        configure(&mut registry);
        let registry = Arc::new(registry.build(tool.output_store().clone()));
        (cache, tool.with_registry(registry))
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
        let (_cache, tool) =
            policy_tool(dir.path(), |registry| registry.script_policy(policy)).await;
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
        let (_cache, tool) =
            policy_tool(dir.path(), |registry| registry.script_policy(policy)).await;
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
        let (_cache, tool) =
            policy_tool(dir.path(), |registry| registry.script_policy(policy)).await;
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
        let (_cache, tool) =
            policy_tool(dir.path(), |registry| registry.script_policy(policy)).await;
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
    async fn starts_after(
        tool: &TerminalTool<executor_core::tokio::TokioGlobal, Configured>,
    ) -> ScriptDir {
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
        let (_cache, tool) = approval_tool(
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
        let (_cache, tool) = policy_tool(dir.path(), |_| {}).await;
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
        let (_cache, tool) = policy_tool(dir.path(), |registry| {
            registry.script_policy(RefuseRepeat);
        })
        .await;
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
    async fn monitor_alerts(
        receiver: &ShellEvents,
        duration: Duration,
    ) -> Vec<ShellEvent> {
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
        let (_cache, tool) = configured_test_terminal(
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
        let (_cache, tool) = configured_test_terminal(
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
        let (_cache, tool) = configured_test_terminal(
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

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn monitor_reports_a_stall_once() {
        let dir = tempfile::tempdir().unwrap();
        let (_cache, tool) = configured_test_terminal(
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
        let (_cache, tool) = configured_test_terminal(
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
        let (_cache, tool) = authority_terminal(dir.path()).await;
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
        let (_cache, tool) = authority_terminal(dir.path()).await;
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
        let (_cache, tool, ends) = carry_terminal(dir.path()).await;
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
        let (_cache, tool) = authority_terminal(dir.path()).await;
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
        let (_cache, parent) = authority_terminal(dir.path()).await;
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
        let (_cache, tool) = authority_terminal(dir.path()).await;
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
