//! Background job registry for tracking and managing background terminal jobs.
//!
//! Uses an internal command channel to avoid shared locks. Each job is exposed
//! externally by its real inside-environment OS `pid: u32` while internally
//! tracked by a monotonic [`LaunchId`] / [`LaunchLease`] so pre-spawn output
//! appenders and OS PID reuse never confuse two distinct jobs.

use std::collections::HashMap;
use std::io::SeekFrom;
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use async_channel::{Receiver, Sender, WeakSender};
use futures_lite::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use crate::LaunchId;
use crate::interpreter::Interpreter;
use crate::memory::MemoryDomain;
use crate::output_policy::OutputPolicy;
use crate::policy::SandboxMode;
#[cfg(unix)]
use crate::process_tree::KillTreeError;
use crate::process_tree::{JobTree, ProcessScanner};
use crate::terminal::TimeoutKill;
use crate::wording::{AnyShellWording, RunningJob, render_default_running_jobs};

/// Maximum bytes retained per stream buffer in memory once a job redirects its
/// output to disk.
const REDIRECTED_BUFFER_CAP: usize = 256 * 1024;

/// Maximum bytes of complete lines retained in memory once a job redirects its
/// output to disk.
const REDIRECTED_LINE_BUFFER_CAP: usize = 512 * 1024;

/// Strong handle keeping a launch's internal identity alive while its flow or
/// registry entry still refers to it.
#[derive(Clone, Debug)]
pub struct LaunchLease(Arc<LaunchId>);

impl LaunchLease {
    /// Allocates a fresh launch lease.
    #[must_use]
    pub fn fresh() -> Self {
        Self::new(LaunchId::next())
    }

    /// Wraps `id` in a new lease.
    #[must_use]
    pub fn new(id: LaunchId) -> Self {
        Self(Arc::new(id))
    }

    /// Underlying [`LaunchId`].
    #[must_use]
    pub fn id(&self) -> LaunchId {
        *self.0
    }

    /// Whether `other` is a clone of this exact lease allocation.
    #[must_use]
    pub fn same_lease(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Downgrades to a weak reference that does not keep the launch alive.
    #[must_use]
    pub fn downgrade(&self) -> WeakLaunchLease {
        WeakLaunchLease(Arc::downgrade(&self.0))
    }
}

/// Weak reference to a [`LaunchLease`].
#[derive(Clone, Debug)]
pub struct WeakLaunchLease(Weak<LaunchId>);

impl WeakLaunchLease {
    /// Upgrades to a strong [`LaunchLease`] when the launch is still held.
    #[must_use]
    pub fn upgrade(&self) -> Option<LaunchLease> {
        self.0.upgrade().map(LaunchLease)
    }

    /// Whether this weak handle refers to `lease`.
    #[must_use]
    pub fn refers_to(&self, lease: &LaunchLease) -> bool {
        Weak::ptr_eq(&self.0, &Arc::downgrade(&lease.0))
    }
}

/// Information about a running or completed background job.
#[derive(Debug, Clone)]
pub struct JobInfo {
    /// Process ID inside the job's execution environment (model-facing).
    pub pid: u32,
    /// Stateless execution key that owns this job.
    pub execution_key: String,
    /// The script that was executed.
    pub script: String,
    /// The interpreter of `script`.
    pub interpreter: Interpreter,
    /// The permission mode used.
    pub mode: SandboxMode,
    /// When the job was started.
    pub started_at: Instant,
    /// Current status of the job.
    pub status: JobStatus,
    /// Path to output file.
    pub output_path: Option<PathBuf>,
    /// Where the job runs.
    pub place: JobPlace,
    /// The bound the job was started under: it is killed once this long has
    /// passed since it started. `None` when nothing bounds it.
    pub timeout: Option<Duration>,
}

impl JobInfo {
    /// How long until the job's `timeout` bound kills it, zero once passed;
    /// `None` when nothing bounds it.
    #[must_use]
    pub fn remaining(&self) -> Option<Duration> {
        self.timeout
            .map(|timeout| timeout.saturating_sub(self.started_at.elapsed()))
    }
}

/// Status of a background job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    /// Job is currently running.
    Running,
    /// Job completed successfully.
    Completed {
        /// Exit code of the process.
        exit_code: i32,
        /// Whether the agent has taken the job's completion.
        delivery: Delivery,
    },
    /// Job failed with an error.
    Failed {
        /// Error message.
        error: String,
        /// Whether the agent has taken the job's completion.
        delivery: Delivery,
    },
    /// Job was killed.
    Killed {
        /// Why it was killed.
        cause: KillCause,
        /// Whether the agent has taken the job's completion.
        delivery: Delivery,
    },
}

/// Why a job was killed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KillCause {
    /// Host-side administrative kill (UI stop, monitor `kill_on_stop`, or
    /// terminal shutdown).
    User,
    /// Its `timeout` bound passed.
    Timeout(TimeoutKill),
    /// Its home environment's memory ran out of room for the agent's jobs
    /// (see [`MemoryDomain`]), and the agent left it running past the
    /// response window that followed the final warning.
    MemoryRunaway {
        /// The tree's physical footprint when it was killed.
        footprint_bytes: u64,
        /// The memory of the domain it ran in: the machine's physical memory,
        /// or the container's limit.
        capacity_bytes: u64,
        /// The domain it ran in.
        domain: MemoryDomain,
    },
}

/// Where a job runs, which decides what is watched of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JobPlace {
    /// A process of the machine the embedder runs on.
    Host,
    /// A process inside the container the agent works in.
    Container,
    /// A process on a remote host over SSH.
    Remote,
}

impl JobStatus {
    /// How far a finished job's completion got; `None` while it runs.
    #[must_use]
    pub const fn delivery(&self) -> Option<Delivery> {
        match self {
            Self::Running => None,
            Self::Completed { delivery, .. }
            | Self::Failed { delivery, .. }
            | Self::Killed { delivery, .. } => Some(*delivery),
        }
    }

    /// Why a killed job was killed; `None` for a job that was not.
    #[must_use]
    pub const fn kill_cause(&self) -> Option<KillCause> {
        match self {
            Self::Killed { cause, .. } => Some(*cause),
            Self::Running | Self::Completed { .. } | Self::Failed { .. } => None,
        }
    }

    /// Whether the job finished and its agent has yet to take the completion.
    #[must_use]
    pub const fn completion_pending(&self) -> bool {
        matches!(self.delivery(), Some(Delivery::Pending))
    }
}

/// Why bytes were not written to a job's stdin.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JobInputError {
    /// The registry holds no job with that PID.
    #[error("unknown pid: {pid}")]
    UnknownJob {
        /// Process ID named.
        pid: u32,
    },
    /// The job no longer runs.
    #[error("process {pid} is not running; current status: {status:?}")]
    NotRunning {
        /// Process ID of the job.
        pid: u32,
        /// How it ended.
        status: JobStatus,
    },
    /// The job's stdin is the null device, not a pipe: it was started in the
    /// foreground, and a job promoted past the foreground window keeps the
    /// stdin it started with.
    #[error("process {pid} reads no input: it started in the foreground, with no stdin pipe")]
    NoStdin {
        /// Process ID of the job.
        pid: u32,
    },
    /// The job's stdin pipe is closed.
    #[error("terminal input channel closed for process {pid}")]
    Closed {
        /// Process ID of the job.
        pid: u32,
    },
}

/// Whether a finished job's completion has reached its agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delivery {
    /// The completion is on its way to the agent, which has not taken it.
    Pending,
    /// The agent took the completion: its [`PendingCompletion`] was dropped.
    Taken,
}

/// A finished job's completion on its way to the agent.
#[must_use = "dropping it marks the completion taken"]
pub struct PendingCompletion {
    tx: Sender<JobCommand>,
    launch: Option<LaunchLease>,
}

impl std::fmt::Debug for PendingCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingCompletion")
            .field("launch", &self.launch)
            .finish_non_exhaustive()
    }
}

impl PendingCompletion {
    /// Takes the completion as dropping it does, returning once the registry
    /// recorded the take.
    pub async fn take(mut self) {
        let Some(launch) = self.launch.take() else {
            return;
        };
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        if self
            .tx
            .send(
                JobRequest::TakeCompletion {
                    launch,
                    reply: reply_tx,
                }
                .into(),
            )
            .await
            .is_ok()
        {
            let _ = reply_rx.recv().await;
        }
    }
}

impl Drop for PendingCompletion {
    fn drop(&mut self) {
        if let Some(launch) = self.launch.take() {
            let _ = self
                .tx
                .try_send(StateCommand::TakeCompletion { launch }.into());
        }
    }
}

/// Keeps a job in its registry, finished or not, while a [`Job`](crate::Job)
/// handle holds it.
pub struct JobWatch {
    tx: Sender<JobCommand>,
    launch: LaunchLease,
    pid: u32,
}

impl JobWatch {
    pub(crate) fn launch(&self) -> LaunchId {
        self.launch.id()
    }

    pub(crate) const fn pid(&self) -> u32 {
        self.pid
    }
}

impl std::fmt::Debug for JobWatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobWatch")
            .field("pid", &self.pid)
            .field("launch", &self.launch)
            .finish_non_exhaustive()
    }
}

impl Drop for JobWatch {
    fn drop(&mut self) {
        let launch = self.launch.clone();
        let _ = self.tx.try_send(StateCommand::Unwatch { launch }.into());
    }
}

/// What `JobRegistry::owe_completion` settled for a job.
#[derive(Debug)]
pub struct OwedCompletion {
    /// The job as it ended.
    pub info: JobInfo,
    /// Keeps the completion pending until the agent takes it; `None` when a
    /// waiter (`JobRegistry::wait_taking`) already took it.
    pub pending: Option<PendingCompletion>,
}

impl JobRead {
    /// The delta, or `None` for an unchanged read.
    #[must_use]
    pub fn into_delta(self) -> Option<JobDelta> {
        match self {
            Self::Delta(delta) => Some(delta),
            Self::Unchanged { .. } => None,
        }
    }
}

/// Incremental read result for a background terminal buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobDelta {
    /// Process ID exposed to the model.
    pub pid: u32,
    /// Cursor the caller should pass on the next read.
    pub cursor: usize,
    /// Total bytes currently buffered for this job.
    pub total_bytes: usize,
    /// Newly read bytes from the terminal buffer.
    pub bytes: Vec<u8>,
    /// Current job status at the time of the read.
    pub status: JobStatus,
}

/// What a read of a job's output returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobRead {
    /// Output past the cursor, or the same cursor with a changed status.
    Delta(JobDelta),
    /// Nothing changed since this job's last read.
    Unchanged {
        /// Process ID whose output did not change.
        pid: u32,
    },
}

/// Why a tool refused a read of a job's output: nothing changed since the
/// job's last read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnchangedRead {
    /// Process ID whose output did not change.
    pub pid: u32,
}

/// Sender used by the registry to push bytes into a running process stdin.
pub type TerminalInputSender = Sender<Vec<u8>>;

/// One of a job's two output streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalStream {
    Stdout,
    Stderr,
}

#[derive(Debug)]
enum KillSwitch {
    /// One-way kill switch used by test fixtures with no child process.
    Fire(Sender<()>),
    /// Two-phase backend kill that completes once the backend process is
    /// killed and its output chunks have drained.
    Backend(Sender<Sender<bool>>),
}

#[derive(Debug)]
#[allow(clippy::struct_excessive_bools)]
struct JobState {
    info: JobInfo,
    host_pid: u32,
    launch: LaunchLease,
    terminal_buffer: Vec<u8>,
    terminal_evicted: usize,
    line_buffer: Vec<u8>,
    line_evicted: usize,
    stdout_partial: Vec<u8>,
    stderr_partial: Vec<u8>,
    stdout_buffer: Vec<u8>,
    stderr_buffer: Vec<u8>,
    stdout_closed: bool,
    stderr_closed: bool,
    input_tx: Option<TerminalInputSender>,
    kill_switch: Option<KillSwitch>,
    switch_kill_in_flight: bool,
    switch_kill_waiters: Vec<Sender<bool>>,
    process_tree: Option<JobTree>,
    kill_cause: Option<KillCause>,
    kills_in_flight: usize,
    deferred_end: Option<Ending>,
    settled_waiters: Vec<SettledWaiter>,
    ended_waiters: Vec<Sender<Option<JobInfo>>>,
    output_redirect: Option<async_fs::File>,
    redirected_disk: bool,
    finished_at: Option<Instant>,
    completion_owed: bool,
    claimed: bool,
    watchers: usize,
}

#[derive(Debug)]
enum SettledWaiter {
    Read(Sender<Option<JobInfo>>),
    Owe {
        ending: Result<i32, String>,
        reply: Sender<Option<OwedCompletion>>,
    },
}

enum StateCommand {
    Register {
        pid: u32,
        host_pid: u32,
        launch: LaunchLease,
        execution_key: String,
        script: String,
        interpreter: Interpreter,
        mode: SandboxMode,
        output_path: Option<PathBuf>,
        process_tree: Option<JobTree>,
        place: JobPlace,
    },
    Bound {
        launch: LaunchLease,
        timeout: Duration,
    },
    Launching {
        launch: LaunchLease,
    },
    Abandon {
        launch: LaunchLease,
    },
    AttachTerminalInput {
        launch: LaunchLease,
        input_tx: TerminalInputSender,
    },
    AttachKillSwitch {
        launch: LaunchLease,
        kill_switch: KillSwitch,
    },
    AppendTerminalOutput {
        launch: LaunchLease,
        stream: TerminalStream,
        chunk: Vec<u8>,
    },
    CloseTerminalStream {
        launch: LaunchLease,
        stream: TerminalStream,
    },
    Complete {
        launch: LaunchLease,
        exit_code: i32,
        output_path: Option<PathBuf>,
    },
    Fail {
        launch: LaunchLease,
        error: String,
        output_path: Option<PathBuf>,
    },
    TakeCompletion {
        launch: LaunchLease,
    },
    Unwatch {
        launch: LaunchLease,
    },
}

enum JobRequest {
    List {
        reply: Sender<Vec<JobInfo>>,
    },
    ListHostRunning {
        reply: Sender<Vec<(JobInfo, u32)>>,
    },
    GetByPid {
        pid: u32,
        reply: Sender<Option<JobInfo>>,
    },
    GetByLaunch {
        launch: LaunchId,
        reply: Sender<Option<JobInfo>>,
    },
    GetByLaunchSettled {
        launch: LaunchId,
        reply: Sender<Option<JobInfo>>,
    },
    AppendTaskStdout {
        launch: LaunchLease,
        chunk: Vec<u8>,
        reply: Sender<bool>,
    },
    Lease {
        launch: LaunchId,
        reply: Sender<Option<WeakLaunchLease>>,
    },
    OweCompletion {
        launch: LaunchId,
        ending: Result<i32, String>,
        reply: Sender<Option<OwedCompletion>>,
    },
    BeginKillByPid {
        pid: u32,
        cause: KillCause,
        reply: Sender<Option<(LaunchId, KillPlan)>>,
    },
    BeginKillByLaunch {
        launch: LaunchId,
        cause: KillCause,
        reply: Sender<KillPlan>,
    },
    BeginKillByExecutionKey {
        execution_key: String,
        cause: KillCause,
        reply: Sender<Vec<(LaunchId, KillPlan)>>,
    },
    BeginKillAllRunning {
        cause: KillCause,
        reply: Sender<Vec<(LaunchId, KillPlan)>>,
    },
    EndKill {
        launch: LaunchId,
        killed: bool,
        reply: Sender<bool>,
    },
    InputTerminal {
        launch: LaunchId,
        pid: u32,
        bytes: Vec<u8>,
        reply: Sender<Result<(), JobInputError>>,
    },
    InputByPid {
        pid: u32,
        bytes: Vec<u8>,
        reply: Sender<Result<(), JobInputError>>,
    },
    StartOutputRedirect {
        launch: LaunchId,
        output_path: PathBuf,
        reply: Sender<Result<Vec<u8>, String>>,
    },
    FlushOutputRedirect {
        launch: LaunchId,
        reply: Sender<Option<PathBuf>>,
    },
    TerminalOutput {
        launch: LaunchId,
        reply: Sender<Option<(Vec<u8>, Vec<u8>)>>,
    },
    ReadJobDelta {
        launch: LaunchId,
        cursor: usize,
        max_bytes: usize,
        tracked: bool,
        reply: Sender<Result<JobRead, String>>,
    },
    ReadByPid {
        pid: u32,
        cursor: usize,
        max_bytes: usize,
        tracked: bool,
        reply: Sender<Result<JobRead, String>>,
    },
    WaitEnded {
        launch: LaunchId,
        claim: bool,
        reply: Sender<Option<JobInfo>>,
    },
    TerminalStreamsClosed {
        launch: LaunchId,
        reply: Sender<bool>,
    },
    FormatRunning {
        reply: Sender<String>,
    },
    HasRunning {
        reply: Sender<bool>,
    },
    WatchByPid {
        pid: u32,
        reply: Sender<Option<JobWatch>>,
    },
    WatchByLaunch {
        launch: LaunchId,
        reply: Sender<Option<JobWatch>>,
    },
    TakeCompletion {
        launch: LaunchLease,
        reply: Sender<()>,
    },
}

enum KillPlan {
    Done(bool),
    Switch(KillSwitch),
    AwaitSwitch(Receiver<bool>),
    Tree(JobTree),
    Group(u32),
}

#[derive(Debug)]
enum Ending {
    Complete {
        exit_code: i32,
        output_path: Option<PathBuf>,
    },
    Fail {
        error: String,
        output_path: Option<PathBuf>,
    },
}

enum JobCommand {
    State(StateCommand),
    Request(JobRequest),
}

impl From<StateCommand> for JobCommand {
    fn from(command: StateCommand) -> Self {
        Self::State(command)
    }
}

impl From<JobRequest> for JobCommand {
    fn from(request: JobRequest) -> Self {
        Self::Request(request)
    }
}

/// Registry handle for tracking background jobs.
#[derive(Clone, Debug)]
pub struct JobRegistry {
    tx: Sender<JobCommand>,
    #[cfg_attr(target_os = "linux", allow(dead_code))]
    scanner: ProcessScanner,
}

/// Background service that owns the job registry state.
#[derive(Debug)]
pub struct JobRegistryService {
    rx: Receiver<JobCommand>,
    tx: WeakSender<JobCommand>,
    policy: OutputPolicy,
    wording: Option<AnyShellWording>,
}

/// Creates a registry handle and its background service; job listings are
/// cut at the `job_preview_bytes` of `policy`.
#[must_use]
pub fn job_registry_channel(policy: OutputPolicy) -> (JobRegistry, JobRegistryService) {
    let (tx, rx) = async_channel::unbounded();
    let weak = tx.downgrade();
    (
        JobRegistry {
            tx,
            scanner: ProcessScanner::default(),
        },
        JobRegistryService {
            rx,
            tx: weak,
            policy,
            wording: None,
        },
    )
}

impl JobRegistryService {
    /// Formats running job summaries through `wording`.
    #[must_use]
    pub fn with_wording(mut self, wording: AnyShellWording) -> Self {
        self.wording = Some(wording);
        self
    }
}

impl JobRegistry {
    async fn send_command(&self, command: impl Into<JobCommand>) {
        self.tx
            .send(command.into())
            .await
            .expect("job registry service unavailable");
    }

    async fn recv_reply<T>(reply_rx: Receiver<T>) -> T {
        reply_rx
            .recv()
            .await
            .expect("job registry response dropped")
    }

    /// Registers a new running job on the host under `launch` and `pid`.
    pub async fn register(
        &self,
        pid: u32,
        launch: &LaunchLease,
        execution_key: &str,
        script: &str,
        mode: SandboxMode,
        output_path: Option<PathBuf>,
    ) {
        self.register_with_tree(
            pid,
            pid,
            launch,
            execution_key,
            script,
            Interpreter::Shell,
            mode,
            output_path,
            None,
            JobPlace::Host,
        )
        .await;
    }

    /// Registers a new running job at `place`.
    #[allow(clippy::too_many_arguments)]
    pub async fn register_at(
        &self,
        place: JobPlace,
        pid: u32,
        launch: &LaunchLease,
        execution_key: &str,
        script: &str,
        interpreter: Interpreter,
        mode: SandboxMode,
    ) {
        self.register_with_tree(
            pid,
            pid,
            launch,
            execution_key,
            script,
            interpreter,
            mode,
            None,
            None,
            place,
        )
        .await;
    }

    /// Registers a new running job with both its inside-environment `pid` and
    /// host `host_pid` plus an optional [`JobTree`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn register_with_tree(
        &self,
        pid: u32,
        host_pid: u32,
        launch: &LaunchLease,
        execution_key: &str,
        script: &str,
        interpreter: Interpreter,
        mode: SandboxMode,
        output_path: Option<PathBuf>,
        process_tree: Option<JobTree>,
        place: JobPlace,
    ) {
        self.send_command(StateCommand::Register {
            place,
            pid,
            host_pid,
            launch: launch.clone(),
            execution_key: execution_key.to_string(),
            script: script.to_string(),
            interpreter,
            mode,
            output_path,
            process_tree,
        })
        .await;
    }

    /// Records that `launch` runs under a `timeout` bound.
    pub async fn bound_by_timeout(&self, launch: &LaunchLease, timeout: Duration) {
        self.send_command(StateCommand::Bound {
            launch: launch.clone(),
            timeout,
        })
        .await;
    }

    /// Attaches a stdin input channel to a running job.
    pub async fn attach_terminal_input(&self, launch: &LaunchLease, input_tx: TerminalInputSender) {
        self.send_command(StateCommand::AttachTerminalInput {
            launch: launch.clone(),
            input_tx,
        })
        .await;
    }

    /// Attaches a kill switch used by test fixtures or runtimes without host PIDs.
    pub async fn attach_kill_switch(&self, launch: &LaunchLease, kill_tx: Sender<()>) {
        self.send_command(StateCommand::AttachKillSwitch {
            launch: launch.clone(),
            kill_switch: KillSwitch::Fire(kill_tx),
        })
        .await;
    }

    /// Attaches a two-phase backend killer that replies once the backend kill
    /// and its output drain have completed.
    pub(crate) async fn attach_backend_killer(
        &self,
        launch: &LaunchLease,
        kill_tx: Sender<Sender<bool>>,
    ) {
        self.send_command(StateCommand::AttachKillSwitch {
            launch: launch.clone(),
            kill_switch: KillSwitch::Backend(kill_tx),
        })
        .await;
    }

    /// The process-table scanner shared by local non-Linux jobs.
    #[cfg_attr(target_os = "linux", allow(dead_code))]
    pub(crate) fn scanner(&self) -> ProcessScanner {
        self.scanner.clone()
    }

    /// Appends stdout bytes for a running job.
    pub async fn append_stdout(&self, launch: &LaunchLease, chunk: Vec<u8>) {
        self.append_output(launch.clone(), TerminalStream::Stdout, chunk)
            .await;
    }

    /// Appends stderr bytes for a running job.
    pub async fn append_stderr(&self, launch: &LaunchLease, chunk: Vec<u8>) {
        self.append_output(launch.clone(), TerminalStream::Stderr, chunk)
            .await;
    }

    /// Appends bytes to `stream` of a running job.
    pub(crate) async fn append_output(
        &self,
        launch: LaunchLease,
        stream: TerminalStream,
        chunk: Vec<u8>,
    ) {
        self.send_command(StateCommand::AppendTerminalOutput {
            launch,
            stream,
            chunk,
        })
        .await;
    }

    /// Appends stdout bytes to `launch`, parking until registration if the
    /// launch is still starting.
    pub async fn append_task_stdout(&self, launch: &LaunchLease, chunk: Vec<u8>) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::AppendTaskStdout {
            launch: launch.clone(),
            chunk,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Weak lease of `launch` while registered or starting.
    pub(crate) async fn launch_lease(&self, launch: LaunchId) -> Option<WeakLaunchLease> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::Lease {
            launch,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Records that `launch` is about to start so early appends are accepted.
    pub(crate) async fn launching(&self, launch: &LaunchLease) {
        self.send_command(StateCommand::Launching {
            launch: launch.clone(),
        })
        .await;
    }

    /// Records that `launch` ended without registering.
    pub(crate) async fn abandon_launch(&self, launch: &LaunchLease) {
        self.send_command(StateCommand::Abandon {
            launch: launch.clone(),
        })
        .await;
    }

    /// Marks stdout stream closed.
    pub async fn close_stdout(&self, launch: &LaunchLease) {
        self.close_output(launch.clone(), TerminalStream::Stdout)
            .await;
    }

    /// Marks stderr stream closed.
    pub async fn close_stderr(&self, launch: &LaunchLease) {
        self.close_output(launch.clone(), TerminalStream::Stderr)
            .await;
    }

    /// Marks `stream` of `launch` closed.
    pub(crate) async fn close_output(&self, launch: LaunchLease, stream: TerminalStream) {
        self.send_command(StateCommand::CloseTerminalStream { launch, stream })
            .await;
    }

    /// Sends bytes to the stdin of the job currently identified by `pid`.
    ///
    /// # Errors
    /// Returns an error when `pid` is unknown, no longer runs, has no stdin
    /// pipe or its channel closed.
    pub async fn terminal_input(&self, pid: u32, bytes: Vec<u8>) -> Result<(), JobInputError> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::InputByPid {
            pid,
            bytes,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Sends bytes to the stdin of a specific watched `launch`.
    pub(crate) async fn terminal_input_launch(
        &self,
        launch: LaunchId,
        pid: u32,
        bytes: Vec<u8>,
    ) -> Result<(), JobInputError> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::InputTerminal {
            launch,
            pid,
            bytes,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Starts continuous output redirection for `launch` and returns current snapshot bytes.
    ///
    /// # Errors
    /// Returns an error when `launch` does not exist or output cannot be redirected.
    pub async fn start_output_redirect(
        &self,
        launch: LaunchId,
        output_path: PathBuf,
    ) -> Result<Vec<u8>, String> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::StartOutputRedirect {
            launch,
            output_path,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Flushes any active output redirect file for `launch` and returns its disk path.
    pub(crate) async fn flush_output_redirect(&self, launch: LaunchId) -> Option<PathBuf> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::FlushOutputRedirect {
            launch,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Returns true when both stdout and stderr streams are closed for `launch`.
    pub async fn terminal_streams_closed(&self, launch: LaunchId) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::TerminalStreamsClosed {
            launch,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Returns collected stdout/stderr buffers for `launch`.
    pub async fn terminal_output(&self, launch: LaunchId) -> Option<(Vec<u8>, Vec<u8>)> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::TerminalOutput {
            launch,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Reads terminal output added since `cursor` for the job named by `pid`.
    ///
    /// # Errors
    /// Returns an error when `pid` is unknown or `cursor` exceeds the buffer.
    pub async fn terminal_read(
        &self,
        pid: u32,
        cursor: usize,
        max_bytes: usize,
    ) -> Result<JobRead, String> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::ReadByPid {
            pid,
            cursor,
            max_bytes,
            tracked: true,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Reads terminal output added since `cursor` for a specific watched `launch`.
    pub(crate) async fn terminal_read_launch(
        &self,
        launch: LaunchId,
        cursor: usize,
        max_bytes: usize,
    ) -> Result<JobRead, String> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::ReadJobDelta {
            launch,
            cursor,
            max_bytes,
            tracked: true,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Untracked line-buffer read for the `monitor` pump on `launch`.
    ///
    /// # Errors
    /// Returns an error when `launch` is unknown.
    pub async fn monitor_read_launch(
        &self,
        launch: LaunchId,
        cursor: usize,
        max_bytes: usize,
    ) -> Result<JobDelta, String> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::ReadJobDelta {
            launch,
            cursor,
            max_bytes,
            tracked: false,
            reply: reply_tx,
        })
        .await;
        match Self::recv_reply(reply_rx).await? {
            JobRead::Delta(delta) => Ok(delta),
            JobRead::Unchanged { pid } => {
                unreachable!("an untracked read of pid {pid} is never unchanged")
            }
        }
    }

    /// Waits until `launch` ends and returns its final [`JobInfo`].
    pub async fn wait_ended(&self, launch: LaunchId) -> Option<JobInfo> {
        self.wait(launch, false).await
    }

    /// Waits until `launch` ends and claims its completion.
    pub async fn wait_taking(&self, launch: LaunchId) -> Option<JobInfo> {
        self.wait(launch, true).await
    }

    async fn wait(&self, launch: LaunchId, claim: bool) -> Option<JobInfo> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::WaitEnded {
            launch,
            claim,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Marks `launch` as completed.
    pub async fn complete(
        &self,
        launch: &LaunchLease,
        exit_code: i32,
        output_path: Option<PathBuf>,
    ) {
        self.send_command(StateCommand::Complete {
            launch: launch.clone(),
            exit_code,
            output_path,
        })
        .await;
    }

    /// Marks `launch` as failed.
    pub async fn fail(&self, launch: &LaunchLease, error: &str, output_path: Option<PathBuf>) {
        self.send_command(StateCommand::Fail {
            launch: launch.clone(),
            error: error.to_string(),
            output_path,
        })
        .await;
    }

    /// Lists all held jobs.
    pub async fn list(&self) -> Vec<JobInfo> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::List { reply: reply_tx })
            .await;
        Self::recv_reply(reply_rx).await
    }

    /// Gets the latest held job with `pid`.
    pub async fn get_by_pid(&self, pid: u32) -> Option<JobInfo> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::GetByPid {
            pid,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Lists running host jobs alongside their host OS PIDs.
    pub(crate) async fn list_host_running(&self) -> Vec<(JobInfo, u32)> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::ListHostRunning { reply: reply_tx })
            .await;
        Self::recv_reply(reply_rx).await
    }

    /// Gets information about a specific `launch`.
    pub async fn get_by_launch(&self, launch: LaunchId) -> Option<JobInfo> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::GetByLaunch {
            launch,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Gets information about `launch` once no kill of it is in flight.
    pub async fn get_by_launch_settled(&self, launch: LaunchId) -> Option<JobInfo> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::GetByLaunchSettled {
            launch,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Settles how `launch` ended and hands out the completion it owes its agent.
    pub async fn owe_completion(
        &self,
        launch: LaunchId,
        ending: Result<i32, String>,
    ) -> Option<OwedCompletion> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::OweCompletion {
            launch,
            ending,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Kills a running `launch` whose `timeout` bound passed.
    pub async fn kill_timed_out(&self, launch: LaunchId, kill: TimeoutKill) -> bool {
        self.kill_launch_for(launch, KillCause::Timeout(kill)).await
    }

    /// Kills a running job whose process tree exceeded memory capacity.
    pub async fn kill_runaway(
        &self,
        pid: u32,
        footprint_bytes: u64,
        capacity_bytes: u64,
        domain: MemoryDomain,
    ) -> bool {
        self.kill_pid_for(
            pid,
            KillCause::MemoryRunaway {
                footprint_bytes,
                capacity_bytes,
                domain,
            },
        )
        .await
    }

    async fn kill_launch_for(&self, launch: LaunchId, cause: KillCause) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::BeginKillByLaunch {
            launch,
            cause,
            reply: reply_tx,
        })
        .await;
        let plan = Self::recv_reply(reply_rx).await;
        self.finish_kill(launch, plan).await
    }

    async fn kill_pid_for(&self, pid: u32, cause: KillCause) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::BeginKillByPid {
            pid,
            cause,
            reply: reply_tx,
        })
        .await;
        let Some((launch, plan)) = Self::recv_reply(reply_rx).await else {
            return false;
        };
        self.finish_kill(launch, plan).await
    }

    /// Kills a running job by its `pid` (host-side administrative kill).
    pub async fn kill_by_pid(&self, pid: u32) -> bool {
        self.kill_pid_for(pid, KillCause::User).await
    }

    /// Kills a running job by its `launch` (host-side administrative kill).
    pub async fn kill_launch(&self, launch: LaunchId) -> bool {
        self.kill_launch_for(launch, KillCause::User).await
    }

    /// Kills every job that is still running in this registry.
    pub async fn kill_running(&self) -> usize {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::BeginKillAllRunning {
            cause: KillCause::User,
            reply: reply_tx,
        })
        .await;
        let plans = Self::recv_reply(reply_rx).await;
        futures::future::join_all(
            plans
                .into_iter()
                .map(|(launch, plan)| async move { self.finish_kill(launch, plan).await }),
        )
        .await
        .into_iter()
        .filter(|killed| *killed)
        .count()
    }

    /// Kills all running jobs owned by an execution key.
    pub async fn kill_by_execution_key(&self, execution_key: &str) -> usize {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::BeginKillByExecutionKey {
            execution_key: execution_key.to_string(),
            cause: KillCause::User,
            reply: reply_tx,
        })
        .await;
        let plans = Self::recv_reply(reply_rx).await;
        futures::future::join_all(
            plans
                .into_iter()
                .map(|(launch, plan)| async move { self.finish_kill(launch, plan).await }),
        )
        .await
        .into_iter()
        .filter(|killed| *killed)
        .count()
    }

    async fn finish_kill(&self, launch: LaunchId, plan: KillPlan) -> bool {
        let killed = match plan {
            KillPlan::Done(killed) => return killed,
            KillPlan::Switch(KillSwitch::Fire(kill_tx)) => {
                if kill_tx.send(()).await.is_err() {
                    tracing::warn!(%launch, "failed to signal runtime kill switch");
                    false
                } else {
                    true
                }
            }
            KillPlan::Switch(KillSwitch::Backend(kill_tx)) => {
                let (done_tx, done_rx) = async_channel::bounded(1);
                if kill_tx.send(done_tx).await.is_err() {
                    tracing::warn!(%launch, "failed to signal backend kill switch");
                    false
                } else {
                    done_rx.recv().await.unwrap_or(false)
                }
            }
            KillPlan::AwaitSwitch(wait_rx) => wait_rx.recv().await.unwrap_or(false),
            KillPlan::Tree(tree) => match tree.kill().await {
                Ok(()) => true,
                #[cfg(unix)]
                Err(KillTreeError::Ended) => {
                    tracing::debug!(%launch, "the job ended before the kill reached it");
                    false
                }
                Err(error) => {
                    tracing::warn!(%launch, error = %error, "failed to kill job");
                    false
                }
            },
            KillPlan::Group(group) => match crate::process_tree::kill_process_tree(group).await {
                Ok(()) => true,
                Err(error) => {
                    tracing::warn!(%launch, error = %error, "failed to kill job");
                    false
                }
            },
        };
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::EndKill {
            launch,
            killed,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Formats only RUNNING jobs for compression preservation.
    pub async fn format_running_jobs(&self) -> String {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::FormatRunning { reply: reply_tx })
            .await;
        Self::recv_reply(reply_rx).await
    }

    /// Returns true if there are any running jobs.
    pub async fn has_running(&self) -> bool {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::HasRunning { reply: reply_tx })
            .await;
        Self::recv_reply(reply_rx).await
    }

    /// Keeps the latest job with `pid` in the registry until the returned watch drops.
    pub(crate) async fn watch(&self, pid: u32) -> Option<JobWatch> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::WatchByPid {
            pid,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }

    /// Keeps `launch` in the registry until the returned watch drops.
    pub(crate) async fn watch_launch(&self, launch: LaunchId) -> Option<JobWatch> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.send_command(JobRequest::WatchByLaunch {
            launch,
            reply: reply_tx,
        })
        .await;
        Self::recv_reply(reply_rx).await
    }
}

#[derive(Default)]
struct ParkedOutput {
    by_launch: HashMap<LaunchId, Parked>,
    forgotten: Vec<WeakLaunchLease>,
    bounds: HashMap<LaunchId, (WeakLaunchLease, Duration)>,
    launching: HashMap<LaunchId, WeakLaunchLease>,
}

struct Parked {
    lease: WeakLaunchLease,
    appends: Vec<ParkedAppend>,
}

type ParkedAppend = (Vec<u8>, Sender<bool>);

impl ParkedOutput {
    fn launching(&mut self, launch: &LaunchLease) {
        self.prune();
        self.launching.insert(launch.id(), launch.downgrade());
    }

    fn launching_lease(&mut self, launch: LaunchId) -> Option<WeakLaunchLease> {
        self.prune();
        self.launching.get(&launch).cloned()
    }

    fn abandon(&mut self, launch: &LaunchLease) -> Vec<ParkedAppend> {
        self.launching.remove(&launch.id());
        let waiting = self.take(launch);
        self.forgot(launch);
        waiting
    }

    fn park(
        &mut self,
        launch: &LaunchLease,
        chunk: Vec<u8>,
        reply: Sender<bool>,
    ) -> Option<Sender<bool>> {
        self.prune();
        if self
            .forgotten
            .iter()
            .any(|forgotten| forgotten.refers_to(launch))
        {
            return Some(reply);
        }
        let parked = self.by_launch.entry(launch.id()).or_insert_with(|| Parked {
            lease: launch.downgrade(),
            appends: Vec::new(),
        });
        assert!(
            parked.lease.refers_to(launch),
            "two live leases share launch {}",
            launch.id()
        );
        parked.appends.push((chunk, reply));
        None
    }

    fn take(&mut self, launch: &LaunchLease) -> Vec<ParkedAppend> {
        self.prune();
        self.by_launch
            .remove(&launch.id())
            .map_or_default(|parked| {
                assert!(
                    parked.lease.refers_to(launch),
                    "two live leases share launch {}",
                    launch.id()
                );
                parked.appends
            })
    }

    fn bound(&mut self, launch: &LaunchLease, timeout: Duration) {
        self.prune();
        self.bounds
            .insert(launch.id(), (launch.downgrade(), timeout));
    }

    fn take_bound(&mut self, launch: &LaunchLease) -> Option<Duration> {
        self.prune();
        self.bounds.remove(&launch.id()).map(|(lease, timeout)| {
            assert!(
                lease.refers_to(launch),
                "two live leases share launch {}",
                launch.id()
            );
            timeout
        })
    }

    fn forgot(&mut self, launch: &LaunchLease) {
        self.prune();
        self.forgotten.push(launch.downgrade());
    }

    fn prune(&mut self) {
        self.by_launch
            .retain(|_, parked| parked.lease.upgrade().is_some());
        self.forgotten
            .retain(|forgotten| forgotten.upgrade().is_some());
        self.bounds
            .retain(|_, (lease, _)| lease.upgrade().is_some());
        self.launching.retain(|_, lease| lease.upgrade().is_some());
    }
}

#[derive(Default)]
struct ReadTracker {
    delta_reads: HashMap<LaunchId, (usize, usize, JobStatus)>,
}

#[derive(Default)]
struct RegistryStore {
    jobs: HashMap<LaunchId, JobState>,
    by_pid: HashMap<u32, Vec<LaunchId>>,
}

impl RegistryStore {
    fn latest_launch(&self, pid: u32) -> Option<LaunchId> {
        self.by_pid
            .get(&pid)
            .and_then(|launches| launches.last().copied())
    }

    fn insert(&mut self, launch: LaunchId, state: JobState) {
        let pid = state.info.pid;
        let replaced = self.jobs.insert(launch, state);
        assert!(replaced.is_none(), "two live leases share launch {launch}");
        self.by_pid.entry(pid).or_default().push(launch);
    }

    fn remove(&mut self, launch: LaunchId) -> Option<JobState> {
        let state = self.jobs.remove(&launch)?;
        if let Some(entries) = self.by_pid.get_mut(&state.info.pid) {
            entries.retain(|item| *item != launch);
            if entries.is_empty() {
                self.by_pid.remove(&state.info.pid);
            }
        }
        Some(state)
    }
}

impl JobRegistryService {
    /// Runs the registry service loop until all senders are dropped.
    pub async fn serve(self) {
        let mut store = RegistryStore::default();
        let mut reads = ReadTracker::default();
        let mut parked = ParkedOutput::default();

        while let Ok(command) = self.rx.recv().await {
            match command {
                JobCommand::State(command) => {
                    handle_state_command(&mut store, &mut reads, &mut parked, command).await;
                }
                JobCommand::Request(request) => {
                    let preview = self.policy.get().job_preview_bytes;
                    handle_job_request(
                        &mut store,
                        &mut reads,
                        &mut parked,
                        &self.tx,
                        preview,
                        self.wording.as_deref(),
                        request,
                    )
                    .await;
                }
            }
        }
    }
}

async fn handle_state_command(
    store: &mut RegistryStore,
    reads: &mut ReadTracker,
    parked: &mut ParkedOutput,
    command: StateCommand,
) {
    match command {
        StateCommand::Register {
            pid,
            host_pid,
            launch,
            execution_key,
            script,
            interpreter,
            mode,
            output_path,
            process_tree,
            place,
        } => {
            let key = launch.id();
            parked.launching.remove(&key);
            let waiting = parked.take(&launch);
            let timeout = parked.take_bound(&launch);
            register_job(
                store,
                pid,
                host_pid,
                launch,
                execution_key,
                script,
                interpreter,
                mode,
                output_path,
                place,
                timeout,
            );
            if let Some(job) = store.jobs.get_mut(&key) {
                job.process_tree = process_tree;
            }
            for (chunk, reply) in waiting {
                append_terminal_output(&mut store.jobs, key, TerminalStream::Stdout, chunk).await;
                let _ = reply.send(true).await;
            }
        }
        StateCommand::Bound { launch, timeout } => parked.bound(&launch, timeout),
        StateCommand::Launching { launch } => parked.launching(&launch),
        StateCommand::Abandon { launch } => {
            if !store.jobs.contains_key(&launch.id()) {
                for (_, reply) in parked.abandon(&launch) {
                    let _ = reply.send(false).await;
                }
            }
        }
        StateCommand::AttachTerminalInput { launch, input_tx } => {
            if let Some(job) = store
                .jobs
                .get_mut(&launch.id())
                .filter(|job| matches!(job.info.status, JobStatus::Running))
            {
                job.input_tx = Some(input_tx);
            }
        }
        StateCommand::AttachKillSwitch {
            launch,
            kill_switch,
        } => {
            if let Some(job) = store.jobs.get_mut(&launch.id()) {
                job.kill_switch = Some(kill_switch);
            }
        }
        StateCommand::AppendTerminalOutput {
            launch,
            stream,
            chunk,
        } => {
            append_terminal_output(&mut store.jobs, launch.id(), stream, chunk).await;
        }
        StateCommand::CloseTerminalStream { launch, stream } => {
            close_terminal_stream(&mut store.jobs, launch.id(), stream);
        }
        StateCommand::Complete {
            launch,
            exit_code,
            output_path,
        } => complete_job(&mut store.jobs, launch.id(), exit_code, output_path).await,
        StateCommand::Fail {
            launch,
            error,
            output_path,
        } => fail_job(&mut store.jobs, launch.id(), error, output_path).await,
        StateCommand::TakeCompletion { launch } => {
            take_completion(store, reads, parked, launch.id());
        }
        StateCommand::Unwatch { launch } => {
            let key = launch.id();
            if let Some(job) = store.jobs.get_mut(&key) {
                job.watchers -= 1;
                forget_if_done(store, reads, parked, key);
            }
        }
    }
}

fn forget_if_done(
    store: &mut RegistryStore,
    reads: &mut ReadTracker,
    parked: &mut ParkedOutput,
    launch: LaunchId,
) {
    let Some(job) = store.jobs.get(&launch) else {
        return;
    };
    if job.completion_owed
        && job.info.status.delivery() == Some(Delivery::Taken)
        && job.watchers == 0
    {
        let job = store.remove(launch).expect("the job is registered");
        reads.delta_reads.remove(&launch);
        parked.forgot(&job.launch);
        tracing::debug!(%launch, pid = %job.info.pid, "forgot finished job");
    }
}

#[allow(clippy::too_many_lines)]
async fn handle_job_request(
    store: &mut RegistryStore,
    reads: &mut ReadTracker,
    parked: &mut ParkedOutput,
    tx: &WeakSender<JobCommand>,
    script_preview_bytes: usize,
    wording: Option<&dyn crate::wording::ShellWording>,
    request: JobRequest,
) {
    match request {
        JobRequest::AppendTaskStdout {
            launch,
            chunk,
            reply,
        } => {
            let key = launch.id();
            if let Some(job) = store.jobs.get(&key) {
                assert!(
                    job.launch.same_lease(&launch),
                    "two live leases share launch {key}"
                );
                let running = matches!(job.info.status, JobStatus::Running);
                if running {
                    append_terminal_output(&mut store.jobs, key, TerminalStream::Stdout, chunk)
                        .await;
                }
                let _ = reply.send(running).await;
            } else if let Some(reply) = parked.park(&launch, chunk, reply) {
                let _ = reply.send(false).await;
            }
        }
        JobRequest::Lease { launch, reply } => {
            let lease = store
                .jobs
                .get(&launch)
                .map(|job| job.launch.downgrade())
                .or_else(|| parked.launching_lease(launch));
            let _ = reply.send(lease).await;
        }
        JobRequest::List { reply } => {
            let items = store
                .jobs
                .values()
                .map(|state| state.info.clone())
                .collect();
            let _ = reply.send(items).await;
        }
        JobRequest::ListHostRunning { reply } => {
            let items = store
                .jobs
                .values()
                .filter(|state| {
                    state.info.place == JobPlace::Host
                        && matches!(state.info.status, JobStatus::Running)
                })
                .map(|state| (state.info.clone(), state.host_pid))
                .collect();
            let _ = reply.send(items).await;
        }
        JobRequest::GetByPid { pid, reply } => {
            let item = store
                .latest_launch(pid)
                .and_then(|launch| store.jobs.get(&launch))
                .map(|state| state.info.clone());
            let _ = reply.send(item).await;
        }
        JobRequest::GetByLaunch { launch, reply } => {
            let item = store.jobs.get(&launch).map(|state| state.info.clone());
            let _ = reply.send(item).await;
        }
        JobRequest::GetByLaunchSettled { launch, reply } => {
            get_settled_by_launch(&mut store.jobs, launch, reply).await;
        }
        JobRequest::OweCompletion {
            launch,
            ending,
            reply,
        } => owe_settled_completion(store, reads, parked, launch, ending, reply, tx).await,
        JobRequest::BeginKillByPid { pid, cause, reply } => {
            let plan = store
                .latest_launch(pid)
                .map(|launch| (launch, begin_kill(&mut store.jobs, launch, cause)));
            let _ = reply.send(plan).await;
        }
        JobRequest::BeginKillByLaunch {
            launch,
            cause,
            reply,
        } => {
            let plan = begin_kill(&mut store.jobs, launch, cause);
            let _ = reply.send(plan).await;
        }
        JobRequest::BeginKillByExecutionKey {
            execution_key,
            cause,
            reply,
        } => {
            let plans = begin_kill_by_execution_key(&mut store.jobs, &execution_key, cause);
            let _ = reply.send(plans).await;
        }
        JobRequest::BeginKillAllRunning { cause, reply } => {
            let keys: Vec<LaunchId> = store
                .jobs
                .iter()
                .filter(|(_, job)| matches!(job.info.status, JobStatus::Running))
                .map(|(launch, _)| *launch)
                .collect();
            let plans = keys
                .into_iter()
                .map(|launch| (launch, begin_kill(&mut store.jobs, launch, cause)))
                .collect();
            let _ = reply.send(plans).await;
        }
        JobRequest::EndKill {
            launch,
            killed,
            reply,
        } => {
            let result = end_kill(&mut store.jobs, launch, killed).await;
            release_settled_waiters(store, reads, parked, launch, tx).await;
            let _ = reply.send(result).await;
        }
        JobRequest::InputTerminal {
            launch,
            pid,
            bytes,
            reply,
        } => {
            let result = input_terminal(&store.jobs, launch, pid, bytes).await;
            let _ = reply.send(result).await;
        }
        JobRequest::InputByPid { pid, bytes, reply } => {
            let result = match store.latest_launch(pid) {
                Some(launch) => input_terminal(&store.jobs, launch, pid, bytes).await,
                None => Err(JobInputError::UnknownJob { pid }),
            };
            let _ = reply.send(result).await;
        }
        JobRequest::StartOutputRedirect {
            launch,
            output_path,
            reply,
        } => {
            let result = redirect_task_output(&mut store.jobs, launch, output_path).await;
            let _ = reply.send(result).await;
        }
        JobRequest::FlushOutputRedirect { launch, reply } => {
            let path = if let Some(job) = store.jobs.get_mut(&launch) {
                if let Some(file) = job.output_redirect.as_mut() {
                    let _ = file.flush().await;
                }
                job.redirected_disk
                    .then(|| job.info.output_path.clone())
                    .flatten()
            } else {
                None
            };
            let _ = reply.send(path).await;
        }
        JobRequest::TerminalOutput { launch, reply } => {
            send_terminal_output(&store.jobs, launch, reply).await;
        }
        JobRequest::ReadJobDelta {
            launch,
            cursor,
            max_bytes,
            tracked,
            reply,
        } => {
            let result = read_terminal_delta(&mut store.jobs, launch, cursor, max_bytes, !tracked)
                .await
                .map(|delta| tracked_delta(reads, launch, delta, tracked));
            let _ = reply.send(result).await;
        }
        JobRequest::ReadByPid {
            pid,
            cursor,
            max_bytes,
            tracked,
            reply,
        } => {
            let result = match store.latest_launch(pid) {
                Some(launch) => {
                    read_terminal_delta(&mut store.jobs, launch, cursor, max_bytes, !tracked)
                        .await
                        .map(|delta| tracked_delta(reads, launch, delta, tracked))
                }
                None => Err(format!("unknown pid: {pid}")),
            };
            let _ = reply.send(result).await;
        }
        JobRequest::WaitEnded {
            launch,
            claim,
            reply,
        } => {
            wait_for_end(&mut store.jobs, launch, claim, reply).await;
        }
        JobRequest::TerminalStreamsClosed { launch, reply } => {
            let closed = store
                .jobs
                .get(&launch)
                .is_some_and(|job| job.stdout_closed && job.stderr_closed);
            let _ = reply.send(closed).await;
        }
        JobRequest::FormatRunning { reply } => {
            let _ = reply
                .send(format_running_jobs(
                    &store.jobs,
                    script_preview_bytes,
                    wording,
                ))
                .await;
        }
        JobRequest::HasRunning { reply } => {
            let _ = reply.send(has_running_job(&store.jobs)).await;
        }
        JobRequest::TakeCompletion { launch, reply } => {
            take_completion(store, reads, parked, launch.id());
            let _ = reply.send(()).await;
        }
        JobRequest::WatchByPid { pid, reply } => {
            let watch = store
                .latest_launch(pid)
                .and_then(|launch| watch_job(&mut store.jobs, launch, tx));
            let _ = reply.send(watch).await;
        }
        JobRequest::WatchByLaunch { launch, reply } => {
            let watch = watch_job(&mut store.jobs, launch, tx);
            let _ = reply.send(watch).await;
        }
    }
}

fn watch_job(
    jobs: &mut HashMap<LaunchId, JobState>,
    launch: LaunchId,
    tx: &WeakSender<JobCommand>,
) -> Option<JobWatch> {
    jobs.get_mut(&launch).map(|job| {
        job.watchers += 1;
        JobWatch {
            tx: tx
                .upgrade()
                .expect("a request arrives through a live registry handle"),
            launch: job.launch.clone(),
            pid: job.info.pid,
        }
    })
}

async fn send_terminal_output(
    jobs: &HashMap<LaunchId, JobState>,
    launch: LaunchId,
    reply: Sender<Option<(Vec<u8>, Vec<u8>)>>,
) {
    let output = jobs
        .get(&launch)
        .map(|job| (job.stdout_buffer.clone(), job.stderr_buffer.clone()));
    let _ = reply.send(output).await;
}

fn has_running_job(jobs: &HashMap<LaunchId, JobState>) -> bool {
    jobs.values()
        .any(|job| matches!(job.info.status, JobStatus::Running))
}

#[allow(clippy::too_many_arguments)]
fn register_job(
    store: &mut RegistryStore,
    pid: u32,
    host_pid: u32,
    launch: LaunchLease,
    execution_key: String,
    script: String,
    interpreter: Interpreter,
    mode: SandboxMode,
    output_path: Option<PathBuf>,
    place: JobPlace,
    timeout: Option<Duration>,
) {
    let key = launch.id();
    let info = JobInfo {
        timeout,
        place,
        pid,
        execution_key,
        script,
        interpreter,
        mode,
        started_at: Instant::now(),
        status: JobStatus::Running,
        output_path,
    };
    store.insert(key, JobState::new(info, host_pid, launch));
    tracing::debug!(%key, pid = %pid, host_pid = %host_pid, "registered background job");
}

impl JobState {
    const fn new(info: JobInfo, host_pid: u32, launch: LaunchLease) -> Self {
        Self {
            info,
            host_pid,
            launch,
            terminal_buffer: Vec::new(),
            terminal_evicted: 0,
            line_buffer: Vec::new(),
            line_evicted: 0,
            stdout_partial: Vec::new(),
            stderr_partial: Vec::new(),
            stdout_buffer: Vec::new(),
            stderr_buffer: Vec::new(),
            stdout_closed: false,
            stderr_closed: false,
            input_tx: None,
            kill_switch: None,
            switch_kill_in_flight: false,
            switch_kill_waiters: Vec::new(),
            process_tree: None,
            kill_cause: None,
            kills_in_flight: 0,
            deferred_end: None,
            settled_waiters: Vec::new(),
            ended_waiters: Vec::new(),
            output_redirect: None,
            redirected_disk: false,
            finished_at: None,
            completion_owed: false,
            claimed: false,
            watchers: 0,
        }
    }
}

fn flush_stream_partial(job: &mut JobState, stream: TerminalStream) {
    let partial = match stream {
        TerminalStream::Stdout => &mut job.stdout_partial,
        TerminalStream::Stderr => &mut job.stderr_partial,
    };
    if !partial.is_empty() {
        job.line_buffer.append(partial);
        job.line_buffer.push(b'\n');
        trim_redirected_buffers(job);
    }
}

fn trim_redirected_buffers(job: &mut JobState) {
    if !job.redirected_disk {
        return;
    }
    if job.terminal_buffer.len() > REDIRECTED_BUFFER_CAP {
        let drop_len = job.terminal_buffer.len() - REDIRECTED_BUFFER_CAP;
        job.terminal_buffer.drain(..drop_len);
        job.terminal_evicted += drop_len;
        if job.terminal_buffer.capacity() > REDIRECTED_BUFFER_CAP * 4 {
            job.terminal_buffer.shrink_to(REDIRECTED_BUFFER_CAP * 2);
        }
    }
    if job.stdout_buffer.len() > REDIRECTED_BUFFER_CAP {
        let drop_len = job.stdout_buffer.len() - REDIRECTED_BUFFER_CAP;
        job.stdout_buffer.drain(..drop_len);
        if job.stdout_buffer.capacity() > REDIRECTED_BUFFER_CAP * 4 {
            job.stdout_buffer.shrink_to(REDIRECTED_BUFFER_CAP * 2);
        }
    }
    if job.stderr_buffer.len() > REDIRECTED_BUFFER_CAP {
        let drop_len = job.stderr_buffer.len() - REDIRECTED_BUFFER_CAP;
        job.stderr_buffer.drain(..drop_len);
        if job.stderr_buffer.capacity() > REDIRECTED_BUFFER_CAP * 4 {
            job.stderr_buffer.shrink_to(REDIRECTED_BUFFER_CAP * 2);
        }
    }
    if job.line_buffer.len() > REDIRECTED_LINE_BUFFER_CAP {
        let min_drop = job.line_buffer.len() - REDIRECTED_LINE_BUFFER_CAP;
        let drop_len = job.line_buffer[min_drop..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(min_drop, |pos| min_drop + pos + 1);
        job.line_buffer.drain(..drop_len);
        job.line_evicted += drop_len;
        if job.line_buffer.capacity() > REDIRECTED_LINE_BUFFER_CAP * 4 {
            job.line_buffer.shrink_to(REDIRECTED_LINE_BUFFER_CAP * 2);
        }
    }
}

enum JobEnd {
    Completed { exit_code: i32 },
    Failed { error: String },
    Killed(KillCause),
}

fn finish(job: &mut JobState, end: JobEnd) {
    assert!(
        matches!(job.info.status, JobStatus::Running),
        "job {} (pid {}) ended twice: it already reads {:?}",
        job.launch.id(),
        job.info.pid,
        job.info.status
    );
    flush_stream_partial(job, TerminalStream::Stdout);
    flush_stream_partial(job, TerminalStream::Stderr);
    let delivery = if job.claimed {
        Delivery::Taken
    } else {
        Delivery::Pending
    };
    job.info.status = match end {
        JobEnd::Completed { exit_code } => JobStatus::Completed {
            exit_code,
            delivery,
        },
        JobEnd::Failed { error } => JobStatus::Failed { error, delivery },
        JobEnd::Killed(cause) => JobStatus::Killed { cause, delivery },
    };
    job.finished_at = Some(Instant::now());
    for waiter in std::mem::take(&mut job.ended_waiters) {
        let _ = waiter.try_send(Some(job.info.clone()));
    }
    job.kill_switch = None;
    job.input_tx = None;
}

fn take_completion(
    store: &mut RegistryStore,
    reads: &mut ReadTracker,
    parked: &mut ParkedOutput,
    launch: LaunchId,
) {
    if let Some(job) = store.jobs.get_mut(&launch) {
        mark_taken(job);
        forget_if_done(store, reads, parked, launch);
    }
}

fn mark_taken(job: &mut JobState) {
    let pid = job.info.pid;
    let delivery = match &mut job.info.status {
        JobStatus::Running => panic!("process {pid}'s completion was taken while it runs"),
        JobStatus::Completed { delivery, .. }
        | JobStatus::Failed { delivery, .. }
        | JobStatus::Killed { delivery, .. } => delivery,
    };
    assert_eq!(
        *delivery,
        Delivery::Pending,
        "process {pid}'s completion was taken twice"
    );
    *delivery = Delivery::Taken;
}

async fn owe_settled_completion(
    store: &mut RegistryStore,
    reads: &mut ReadTracker,
    parked: &mut ParkedOutput,
    launch: LaunchId,
    ending: Result<i32, String>,
    reply: Sender<Option<OwedCompletion>>,
    tx: &WeakSender<JobCommand>,
) {
    let Some(job) = store.jobs.get_mut(&launch) else {
        let _ = reply.send(None).await;
        return;
    };
    if job.kills_in_flight > 0 {
        job.settled_waiters
            .push(SettledWaiter::Owe { ending, reply });
        return;
    }
    let owed = owe_completion(&mut store.jobs, launch, ending, tx).await;
    forget_if_done(store, reads, parked, launch);
    let _ = reply.send(Some(owed)).await;
}

async fn owe_completion(
    jobs: &mut HashMap<LaunchId, JobState>,
    launch: LaunchId,
    ending: Result<i32, String>,
    tx: &WeakSender<JobCommand>,
) -> OwedCompletion {
    if matches!(jobs[&launch].info.status, JobStatus::Running) {
        match ending {
            Ok(exit_code) => complete_job(jobs, launch, exit_code, None).await,
            Err(error) => fail_job(jobs, launch, error, None).await,
        }
    }
    let job = jobs.get_mut(&launch).expect("the job is registered");
    assert!(
        !job.completion_owed,
        "job {launch}'s completion was handed out twice"
    );
    let delivery = if job.claimed {
        Delivery::Taken
    } else {
        Delivery::Pending
    };
    assert_eq!(
        job.info.status.delivery(),
        Some(delivery),
        "job {launch} owes a completion it never recorded: {:?}",
        job.info.status
    );
    job.completion_owed = true;
    let info = job.info.clone();
    if job.claimed {
        return OwedCompletion {
            info,
            pending: None,
        };
    }
    OwedCompletion {
        info,
        pending: Some(PendingCompletion {
            tx: tx
                .upgrade()
                .expect("a request arrives through a live registry handle"),
            launch: Some(job.launch.clone()),
        }),
    }
}

async fn append_terminal_output(
    jobs: &mut HashMap<LaunchId, JobState>,
    launch: LaunchId,
    stream: TerminalStream,
    chunk: Vec<u8>,
) {
    if chunk.is_empty() {
        return;
    }
    if let Some(job) = jobs.get_mut(&launch) {
        job.terminal_buffer.extend_from_slice(&chunk);
        if let Some(file) = job.output_redirect.as_mut()
            && let Err(error) = file.write_all(&chunk).await
        {
            if matches!(job.info.status, JobStatus::Running) {
                mark_job_failed(
                    job,
                    format!(
                        "failed to append redirected output for pid {}: {error}",
                        job.info.pid
                    ),
                );
            }
            tracing::error!(
                %launch,
                pid = %job.info.pid,
                error = %error,
                "failed to append redirected output"
            );
        }
        let (buffer, partial) = match stream {
            TerminalStream::Stdout => (&mut job.stdout_buffer, &mut job.stdout_partial),
            TerminalStream::Stderr => (&mut job.stderr_buffer, &mut job.stderr_partial),
        };
        buffer.extend_from_slice(&chunk);
        partial.extend_from_slice(&chunk);
        if let Some(last_newline) = partial.iter().rposition(|byte| *byte == b'\n') {
            job.line_buffer.extend(partial.drain(..=last_newline));
        }
        trim_redirected_buffers(job);
    }
}

fn close_terminal_stream(
    jobs: &mut HashMap<LaunchId, JobState>,
    launch: LaunchId,
    stream: TerminalStream,
) {
    if let Some(job) = jobs.get_mut(&launch) {
        flush_stream_partial(job, stream);
        match stream {
            TerminalStream::Stdout => job.stdout_closed = true,
            TerminalStream::Stderr => job.stderr_closed = true,
        }
    }
}

async fn complete_job(
    jobs: &mut HashMap<LaunchId, JobState>,
    launch: LaunchId,
    exit_code: i32,
    output_path: Option<PathBuf>,
) {
    if let Some(job) = jobs.get_mut(&launch) {
        if !matches!(job.info.status, JobStatus::Running) {
            tracing::debug!(
                %launch,
                status = ?job.info.status,
                "ignoring complete for non-running job"
            );
            return;
        }
        if job.kills_in_flight > 0 {
            job.deferred_end = Some(Ending::Complete {
                exit_code,
                output_path,
            });
            return;
        }
        if let Err(error) = finalize_output_redirect(job).await {
            tracing::error!(
                %launch,
                error = %error,
                "failed to flush redirected output on complete"
            );
            mark_job_failed(job, error);
            return;
        }
        finish(job, JobEnd::Completed { exit_code });
        if output_path.is_some() {
            job.info.output_path = output_path;
        }
        tracing::debug!(%launch, pid = %job.info.pid, exit_code = %exit_code, "job completed");
    }
}

async fn fail_job(
    jobs: &mut HashMap<LaunchId, JobState>,
    launch: LaunchId,
    error: String,
    output_path: Option<PathBuf>,
) {
    if let Some(job) = jobs.get_mut(&launch) {
        if !matches!(job.info.status, JobStatus::Running) {
            tracing::debug!(
                %launch,
                status = ?job.info.status,
                "ignoring fail for non-running job"
            );
            return;
        }
        if job.kills_in_flight > 0 {
            job.deferred_end = Some(Ending::Fail { error, output_path });
            return;
        }
        let status_error = if let Err(finalize_error) = finalize_output_redirect(job).await {
            format!("{error}; additionally failed to flush redirected output: {finalize_error}")
        } else {
            error
        };
        mark_job_failed(job, status_error);
        if output_path.is_some() {
            job.info.output_path = output_path;
        }
        tracing::debug!(%launch, pid = %job.info.pid, "job failed");
    }
}

fn begin_kill_by_execution_key(
    jobs: &mut HashMap<LaunchId, JobState>,
    execution_key: &str,
    cause: KillCause,
) -> Vec<(LaunchId, KillPlan)> {
    let launches: Vec<LaunchId> = jobs
        .iter()
        .filter(|(_, state)| {
            state.info.execution_key == execution_key
                && matches!(state.info.status, JobStatus::Running)
        })
        .map(|(launch, _)| *launch)
        .collect();

    let mut plans = Vec::with_capacity(launches.len());
    for launch in launches {
        let plan = begin_kill(jobs, launch, cause);
        plans.push((launch, plan));
    }
    plans
}

async fn input_terminal(
    jobs: &HashMap<LaunchId, JobState>,
    launch: LaunchId,
    pid: u32,
    bytes: Vec<u8>,
) -> Result<(), JobInputError> {
    let Some(job) = jobs.get(&launch) else {
        return Err(JobInputError::UnknownJob { pid });
    };
    let pid = job.info.pid;
    if !matches!(job.info.status, JobStatus::Running) {
        return Err(JobInputError::NotRunning {
            pid,
            status: job.info.status.clone(),
        });
    }
    let Some(tx) = &job.input_tx else {
        return Err(JobInputError::NoStdin { pid });
    };
    tx.send(bytes)
        .await
        .map_err(|_| JobInputError::Closed { pid })
}

async fn redirect_task_output(
    jobs: &mut HashMap<LaunchId, JobState>,
    launch: LaunchId,
    output_path: PathBuf,
) -> Result<Vec<u8>, String> {
    let Some(job) = jobs.get_mut(&launch) else {
        return Err(format!("unknown launch: {launch}"));
    };
    start_output_redirect(job, output_path).await
}

fn tracked_delta(
    reads: &mut ReadTracker,
    launch: LaunchId,
    delta: JobDelta,
    tracked: bool,
) -> JobRead {
    let returned = (delta.cursor, delta.total_bytes, delta.status.clone());
    if !tracked {
        return JobRead::Delta(delta);
    }
    if reads.delta_reads.get(&launch) == Some(&returned) {
        JobRead::Unchanged { pid: delta.pid }
    } else {
        reads.delta_reads.insert(launch, returned);
        JobRead::Delta(delta)
    }
}

async fn wait_for_end(
    jobs: &mut HashMap<LaunchId, JobState>,
    launch: LaunchId,
    claim: bool,
    reply: Sender<Option<JobInfo>>,
) {
    match jobs.get_mut(&launch) {
        Some(job) if matches!(job.info.status, JobStatus::Running) => {
            job.claimed |= claim && !job.completion_owed;
            job.ended_waiters.push(reply);
        }
        Some(job) => {
            if claim && !job.completion_owed && !job.claimed {
                job.claimed = true;
                mark_taken(job);
            }
            let _ = reply.send(Some(job.info.clone())).await;
        }
        None => {
            let _ = reply.send(None).await;
        }
    }
}

async fn read_terminal_delta(
    jobs: &mut HashMap<LaunchId, JobState>,
    launch: LaunchId,
    cursor: usize,
    max_bytes: usize,
    lines: bool,
) -> Result<JobDelta, String> {
    let Some(job) = jobs.get_mut(&launch) else {
        return Err(format!("unknown launch: {launch}"));
    };
    terminal_read(job, cursor, max_bytes, lines).await
}

async fn start_output_redirect(
    job: &mut JobState,
    output_path: PathBuf,
) -> Result<Vec<u8>, String> {
    if let Some(existing) = job.info.output_path.as_ref() {
        if existing != &output_path {
            return Err(format!(
                "process {} already redirects output to {}",
                job.info.pid,
                existing.display()
            ));
        }
        return Ok(job.terminal_buffer.clone());
    }

    let mut file = async_fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&output_path)
        .await
        .map_err(|error| {
            format!(
                "failed to open output redirect file {}: {error}",
                output_path.display()
            )
        })?;

    file.write_all(&job.terminal_buffer)
        .await
        .map_err(|error| {
            format!(
                "failed to write initial output redirect snapshot for pid {}: {error}",
                job.info.pid
            )
        })?;
    file.flush().await.map_err(|error| {
        format!(
            "failed to flush initial output redirect snapshot for pid {}: {error}",
            job.info.pid
        )
    })?;

    let snapshot = job.terminal_buffer.clone();
    job.info.output_path = Some(output_path);
    job.output_redirect = Some(file);
    job.redirected_disk = true;
    trim_redirected_buffers(job);
    Ok(snapshot)
}

fn mark_job_failed(job: &mut JobState, error: String) {
    finish(job, JobEnd::Failed { error });
}

async fn finalize_output_redirect(job: &mut JobState) -> Result<(), String> {
    if let Some(mut file) = job.output_redirect.take()
        && let Err(error) = file.flush().await
    {
        return Err(format!(
            "failed to flush redirected output for pid {}: {error}",
            job.info.pid
        ));
    }
    Ok(())
}

fn begin_kill(
    jobs: &mut HashMap<LaunchId, JobState>,
    launch: LaunchId,
    cause: KillCause,
) -> KillPlan {
    let Some(job) = jobs.get_mut(&launch) else {
        return KillPlan::Done(false);
    };
    if !matches!(job.info.status, JobStatus::Running) {
        tracing::warn!(%launch, pid = %job.info.pid, "cannot kill non-running job");
        return KillPlan::Done(false);
    }

    if job.kills_in_flight == 0 {
        job.kill_cause = Some(cause);
    }
    if let Some(kill_switch) = job.kill_switch.take() {
        job.kills_in_flight += 1;
        job.switch_kill_in_flight = true;
        return KillPlan::Switch(kill_switch);
    }
    if job.switch_kill_in_flight {
        job.kills_in_flight += 1;
        let (tx, rx) = async_channel::bounded(1);
        job.switch_kill_waiters.push(tx);
        return KillPlan::AwaitSwitch(rx);
    }

    job.kills_in_flight += 1;
    let group = job.host_pid;
    job.process_tree
        .as_ref()
        .map_or(KillPlan::Group(group), |tree| KillPlan::Tree(tree.clone()))
}

async fn end_kill(jobs: &mut HashMap<LaunchId, JobState>, launch: LaunchId, killed: bool) -> bool {
    let Some(job) = jobs.get_mut(&launch) else {
        return false;
    };
    if job.switch_kill_in_flight {
        job.switch_kill_in_flight = false;
        for waiter in std::mem::take(&mut job.switch_kill_waiters) {
            let _ = waiter.try_send(killed);
        }
    }
    job.kills_in_flight = job.kills_in_flight.saturating_sub(1);
    if killed {
        job.deferred_end = None;
        if matches!(job.info.status, JobStatus::Running) {
            if let Err(error) = finalize_output_redirect(job).await {
                tracing::warn!(
                    %launch,
                    error = %error,
                    "failed to flush redirected output after kill"
                );
            }
            let cause = job
                .kill_cause
                .expect("a kill in flight was asked for with a cause");
            finish(job, JobEnd::Killed(cause));
            job.stdout_closed = true;
            job.stderr_closed = true;
            tracing::info!(%launch, pid = %job.info.pid, ?cause, "killed job");
        }
        return true;
    }
    match job.deferred_end.take() {
        Some(Ending::Complete {
            exit_code,
            output_path,
        }) => complete_job(jobs, launch, exit_code, output_path).await,
        Some(Ending::Fail { error, output_path }) => {
            fail_job(jobs, launch, error, output_path).await;
        }
        None => {}
    }
    false
}

async fn get_settled_by_launch(
    jobs: &mut HashMap<LaunchId, JobState>,
    launch: LaunchId,
    reply: Sender<Option<JobInfo>>,
) {
    match jobs.get_mut(&launch) {
        Some(job) if job.kills_in_flight > 0 => {
            job.settled_waiters.push(SettledWaiter::Read(reply));
        }
        Some(job) => {
            let _ = reply.send(Some(job.info.clone())).await;
        }
        None => {
            let _ = reply.send(None).await;
        }
    }
}

async fn release_settled_waiters(
    store: &mut RegistryStore,
    reads: &mut ReadTracker,
    parked: &mut ParkedOutput,
    launch: LaunchId,
    tx: &WeakSender<JobCommand>,
) {
    let Some(job) = store.jobs.get_mut(&launch) else {
        return;
    };
    if job.kills_in_flight > 0 {
        return;
    }
    for waiter in std::mem::take(&mut job.settled_waiters) {
        match waiter {
            SettledWaiter::Read(reply) => {
                let _ = reply.send(Some(store.jobs[&launch].info.clone())).await;
            }
            SettledWaiter::Owe { ending, reply } => {
                let owed = owe_completion(&mut store.jobs, launch, ending, tx).await;
                let _ = reply.send(Some(owed)).await;
            }
        }
    }
    forget_if_done(store, reads, parked, launch);
}

fn script_preview(script: &str, limit: usize) -> String {
    if script.len() <= limit {
        return script.to_string();
    }
    let end = script.floor_char_boundary(limit);
    format!("{}...", script.get(..end).unwrap_or_default())
}

fn format_running_jobs(
    jobs: &HashMap<LaunchId, JobState>,
    script_preview_bytes: usize,
    wording: Option<&dyn crate::wording::ShellWording>,
) -> String {
    let mut running: Vec<_> = jobs
        .values()
        .filter(|job| matches!(job.info.status, JobStatus::Running))
        .collect();
    running.sort_by_key(|job| job.info.pid);

    if running.is_empty() {
        return String::new();
    }

    let prepared: Vec<(String, Option<String>)> = running
        .iter()
        .map(|job| {
            let preview = script_preview(&job.info.script, script_preview_bytes);
            let path = job
                .info
                .output_path
                .as_ref()
                .map(|p| p.display().to_string());
            (preview, path)
        })
        .collect();
    let entries: Vec<RunningJob<'_>> = running
        .iter()
        .zip(&prepared)
        .map(|(job, (preview, path))| RunningJob {
            pid: job.info.pid,
            script_preview: preview,
            output_path: path.as_deref(),
        })
        .collect();
    wording.map_or_else(
        || render_default_running_jobs(&entries),
        |wording| wording.running_jobs(&entries),
    )
}

async fn terminal_read(
    job: &mut JobState,
    cursor: usize,
    max_bytes: usize,
    lines: bool,
) -> Result<JobDelta, String> {
    if lines {
        let total_bytes = job.line_evicted + job.line_buffer.len();
        if cursor > total_bytes {
            return Err(format!(
                "cursor {cursor} exceeds terminal buffer size {total_bytes} for pid {}",
                job.info.pid
            ));
        }
        let effective_cursor = cursor.max(job.line_evicted);
        let end = total_bytes.min(effective_cursor.saturating_add(max_bytes));
        let start_idx = effective_cursor - job.line_evicted;
        let end_idx = end - job.line_evicted;
        return Ok(JobDelta {
            pid: job.info.pid,
            cursor: end,
            total_bytes,
            bytes: job.line_buffer[start_idx..end_idx].to_vec(),
            status: job.info.status.clone(),
        });
    }

    let total_bytes = job.terminal_evicted + job.terminal_buffer.len();
    if cursor > total_bytes {
        return Err(format!(
            "cursor {cursor} exceeds terminal buffer size {total_bytes} for pid {}",
            job.info.pid
        ));
    }
    let end = total_bytes.min(cursor.saturating_add(max_bytes));
    let bytes = if cursor >= job.terminal_evicted {
        let start_idx = cursor - job.terminal_evicted;
        let end_idx = end - job.terminal_evicted;
        job.terminal_buffer[start_idx..end_idx].to_vec()
    } else if let Some(path) = job.info.output_path.clone() {
        if let Some(file) = job.output_redirect.as_mut() {
            let _ = file.flush().await;
        }
        let wanted = end - cursor;
        let mut out = Vec::with_capacity(wanted.min(REDIRECTED_BUFFER_CAP));
        let mut reader = async_fs::File::open(&path).await.map_err(|error| {
            format!(
                "failed to open redirected output {} for pid {}: {error}",
                path.display(),
                job.info.pid
            )
        })?;
        reader
            .seek(SeekFrom::Start(cursor as u64))
            .await
            .map_err(|error| {
                format!(
                    "failed to seek redirected output {} for pid {}: {error}",
                    path.display(),
                    job.info.pid
                )
            })?;
        reader
            .take(wanted as u64)
            .read_to_end(&mut out)
            .await
            .map_err(|error| {
                format!(
                    "failed to read redirected output {} for pid {}: {error}",
                    path.display(),
                    job.info.pid
                )
            })?;
        out
    } else {
        Vec::new()
    };

    Ok(JobDelta {
        pid: job.info.pid,
        cursor: end,
        total_bytes,
        bytes,
        status: job.info.status.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use executor_core::Executor;
    use executor_core::Task;
    use executor_core::tokio::TokioGlobal;
    use tempfile::tempdir;

    fn spawn_registry() -> JobRegistry {
        let (registry, service) = job_registry_channel(crate::output_policy::fixture::policy());
        TokioGlobal
            .spawn(async move { service.serve().await })
            .detach();
        registry
    }

    fn lease() -> LaunchLease {
        LaunchLease::fresh()
    }

    #[tokio::test]
    async fn a_job_reports_the_timeout_it_runs_under() {
        let registry = spawn_registry();
        let bounded = lease();
        let free = lease();

        registry
            .bound_by_timeout(&bounded, Duration::from_secs(600))
            .await;
        registry
            .register(101, &bounded, "exec", "run", SandboxMode::Sandboxed, None)
            .await;
        registry
            .register(102, &free, "exec", "run", SandboxMode::Sandboxed, None)
            .await;

        let info = registry.get_by_pid(101).await.expect("job exists");
        assert_eq!(info.timeout, Some(Duration::from_secs(600)));
        let remaining = info.remaining().expect("the job is bounded");
        assert!(remaining > Duration::from_secs(590), "{remaining:?}");
        let info = registry.get_by_pid(102).await.expect("job exists");
        assert_eq!(info.timeout, None);
        assert_eq!(info.remaining(), None);
    }

    #[tokio::test]
    async fn a_job_sharing_a_held_jobs_pid_does_not_replace_it() {
        let registry = spawn_registry();
        let (old, new) = (lease(), lease());
        registry
            .register(4242, &old, "exec", "first", SandboxMode::Sandboxed, None)
            .await;
        registry.append_stdout(&old, b"old output\n".to_vec()).await;
        registry.complete(&old, 3, None).await;

        registry
            .register(4242, &new, "exec", "second", SandboxMode::Sandboxed, None)
            .await;
        registry.append_stdout(&new, b"new output\n".to_vec()).await;

        assert_eq!(registry.list().await.len(), 2);
        assert_eq!(
            registry.get_by_launch(old.id()).await.unwrap().status,
            JobStatus::Completed {
                exit_code: 3,
                delivery: Delivery::Pending
            }
        );
        assert_eq!(
            registry.get_by_pid(4242).await.unwrap().status,
            JobStatus::Running,
            "pid lookup resolves to the active job when an earlier PID is recycled"
        );
        let (stdout, _) = registry.terminal_output(old.id()).await.unwrap();
        assert_eq!(stdout, b"old output\n");
        let (stdout, _) = registry.terminal_output(new.id()).await.unwrap();
        assert_eq!(stdout, b"new output\n");
        let read = registry
            .terminal_read(4242, 0, 1024)
            .await
            .unwrap()
            .into_delta()
            .unwrap();
        assert_eq!(read.bytes, b"new output\n");

        let owed = registry
            .owe_completion(old.id(), Ok(3))
            .await
            .expect("the first job is still registered");
        assert!(owed.pending.is_some());
        drop(owed);
        assert!(registry.get_by_launch(old.id()).await.is_none());
        assert_eq!(
            registry.get_by_pid(4242).await.unwrap().status,
            JobStatus::Running,
            "the second job is untouched"
        );
    }

    #[tokio::test]
    async fn a_claiming_wait_reads_the_job_taken_before_its_flow_settles() {
        let registry = spawn_registry();
        let (ended, ending) = (lease(), lease());
        registry
            .register(1, &ended, "exec", "run", SandboxMode::Sandboxed, None)
            .await;
        registry
            .register(2, &ending, "exec", "run", SandboxMode::Sandboxed, None)
            .await;
        let ended_watch = registry.watch(1).await.expect("ended is held");
        let ending_watch = registry.watch(2).await.expect("ending is held");

        registry.complete(&ended, 0, None).await;
        let taken = JobStatus::Completed {
            exit_code: 0,
            delivery: Delivery::Taken,
        };
        assert_eq!(
            registry.wait_taking(ended.id()).await.unwrap().status,
            taken
        );
        assert_eq!(registry.get_by_pid(1).await.unwrap().status, taken);

        let mut waiting = std::pin::pin!(registry.wait_taking(ending.id()));
        assert!(
            futures_lite::future::poll_once(waiting.as_mut())
                .await
                .is_none()
        );
        registry.complete(&ending, 0, None).await;
        assert_eq!(waiting.await.unwrap().status, taken);
        assert_eq!(registry.get_by_pid(2).await.unwrap().status, taken);
        assert!(
            !registry
                .list()
                .await
                .iter()
                .any(|job| job.status.completion_pending())
        );

        drop(ended_watch);
        assert!(registry.terminal_output(ended.id()).await.is_some());
        for id in [ended.id(), ending.id()] {
            let owed = registry.owe_completion(id, Ok(0)).await.unwrap();
            assert!(owed.pending.is_none());
        }
        drop(ending_watch);
        assert!(registry.get_by_pid(2).await.is_none());
    }

    #[tokio::test]
    async fn a_taken_completion_forgets_the_job_and_clears_read_history_on_pid_reuse() {
        let registry = spawn_registry();
        let first = lease();
        registry
            .register(51, &first, "exec", "run", SandboxMode::Sandboxed, None)
            .await;
        registry.append_stdout(&first, b"done\n".to_vec()).await;
        registry.complete(&first, 0, None).await;
        registry
            .terminal_read(51, 0, 1024)
            .await
            .expect("a finished job reads until delivered");

        let watch = registry.watch(51).await.expect("job is held");
        let owed = registry.owe_completion(first.id(), Ok(0)).await.unwrap();
        drop(owed);
        assert_eq!(
            registry.get_by_pid(51).await.unwrap().status,
            JobStatus::Completed {
                exit_code: 0,
                delivery: Delivery::Taken
            }
        );

        drop(watch);
        assert!(registry.list().await.is_empty());
        assert!(registry.get_by_pid(51).await.is_none());
        assert!(registry.terminal_read(51, 0, 1024).await.is_err());
        assert!(!registry.kill_by_pid(51).await);
        assert!(registry.terminal_input(51, b"x".to_vec()).await.is_err());

        let again = lease();
        registry
            .register(51, &again, "exec", "run", SandboxMode::Sandboxed, None)
            .await;
        registry.append_stdout(&again, b"done\n".to_vec()).await;
        registry.complete(&again, 0, None).await;
        registry
            .terminal_read(51, 0, 1024)
            .await
            .expect("a new job under a reused PID reads afresh");
    }

    #[tokio::test]
    async fn task_output_appended_before_registration_lands_first_and_in_order() {
        let registry = spawn_registry();
        let early = lease();
        let appends = {
            let registry = registry.clone();
            let early = early.clone();
            tokio::spawn(async move {
                registry.append_task_stdout(&early, b"one\n".to_vec()).await;
                registry.append_task_stdout(&early, b"two\n".to_vec()).await;
            })
        };
        tokio::task::yield_now().await;
        assert!(!appends.is_finished());

        registry
            .register(9, &early, "exec-e", "run", SandboxMode::Sandboxed, None)
            .await;
        appends.await.expect("appends are released by registration");
        registry.append_stdout(&early, b"three\n".to_vec()).await;

        let (stdout, _) = registry
            .terminal_output(early.id())
            .await
            .expect("job exists");
        assert_eq!(stdout, b"one\ntwo\nthree\n");
    }

    #[tokio::test]
    async fn unchanged_terminal_read_is_refused_then_succeeds_after_new_output() {
        let registry = spawn_registry();
        let alpha = lease();
        registry
            .register(7, &alpha, "exec-a", "run", SandboxMode::Sandboxed, None)
            .await;
        registry.append_stdout(&alpha, b"one".to_vec()).await;

        let first = registry
            .terminal_read(7, 0, 1024)
            .await
            .expect("first read succeeds")
            .into_delta()
            .expect("the first read is a delta");
        let second = registry.terminal_read(7, first.cursor, 1024).await;
        assert!(matches!(&second, Ok(JobRead::Unchanged { pid: 7 })));

        registry.append_stdout(&alpha, b"two".to_vec()).await;
        let third = registry
            .terminal_read(7, first.cursor, 1024)
            .await
            .expect("read succeeds once output changed")
            .into_delta()
            .expect("changed output reads as a delta");
        assert_eq!(third.bytes, b"two");
    }

    #[tokio::test]
    async fn test_format_running_jobs_cuts_a_multibyte_script_at_a_boundary() {
        let registry = spawn_registry();
        let script = "日".repeat(30);
        let preview_bytes = crate::output_policy::fixture::LIMITS.job_preview_bytes;
        assert!(!script.is_char_boundary(preview_bytes));
        let cjk = lease();
        registry
            .register(42, &cjk, "exec-cjk", &script, SandboxMode::Sandboxed, None)
            .await;

        let running = registry.format_running_jobs().await;
        assert!(running.contains("pid 42"), "{running}");
        assert!(
            running.contains(&format!("`{}...`", "日".repeat(16))),
            "{running}"
        );
    }

    #[tokio::test]
    async fn two_phase_backend_kill_waits_for_drain_and_preserves_unterminated_eof_line() {
        let registry = spawn_registry();
        let id = lease();
        registry
            .register(
                700,
                &id,
                "exec",
                "echo partial",
                SandboxMode::Sandboxed,
                None,
            )
            .await;

        let (kill_tx, kill_rx) = async_channel::bounded::<Sender<bool>>(1);
        let (kill_entered_tx, kill_entered_rx) = async_channel::bounded(1);
        let (release_kill_tx, release_kill_rx) = async_channel::bounded::<()>(1);
        registry.attach_backend_killer(&id, kill_tx).await;

        let reg_for_killer = registry.clone();
        let id_for_killer = id.clone();
        let killer_task = tokio::spawn(async move {
            let done_tx = kill_rx.recv().await.unwrap();
            let _ = kill_entered_tx.send(()).await;
            let _ = release_kill_rx.recv().await;
            reg_for_killer
                .append_stdout(&id_for_killer, b"final-unterminated".to_vec())
                .await;
            reg_for_killer.close_stdout(&id_for_killer).await;
            let _ = done_tx.send(true).await;
        });

        let reg_kill = registry.clone();
        let kill_task = tokio::spawn(async move { reg_kill.kill_by_pid(700).await });

        kill_entered_rx.recv().await.unwrap();
        assert_eq!(
            registry.get_by_pid(700).await.unwrap().status,
            JobStatus::Running
        );

        release_kill_tx.send(()).await.unwrap();
        assert!(kill_task.await.unwrap());
        killer_task.await.unwrap();

        let settled = registry.get_by_pid(700).await.unwrap();
        assert!(matches!(settled.status, JobStatus::Killed { .. }));

        let (stdout, _stderr) = registry.terminal_output(id.id()).await.unwrap();
        assert_eq!(stdout, b"final-unterminated");

        let delta = registry
            .monitor_read_launch(id.id(), 0, 1024)
            .await
            .unwrap();
        assert_eq!(delta.bytes, b"final-unterminated\n");
    }

    #[tokio::test]
    async fn redirected_job_bounds_in_memory_buffers_and_serves_evicted_bytes_from_disk() {
        let dir = tempdir().unwrap();
        let redirect_path = dir.path().join("redirected.log");
        let registry = spawn_registry();
        let id = lease();
        registry
            .register(800, &id, "exec", "yes line", SandboxMode::Sandboxed, None)
            .await;

        registry
            .start_output_redirect(id.id(), redirect_path.clone())
            .await
            .unwrap();

        let chunk = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abc\n".repeat(256);
        let mut total = 0_usize;
        for _ in 0..64 {
            registry.append_stdout(&id, chunk.as_bytes().to_vec()).await;
            total += chunk.len();
        }

        let delta = registry
            .terminal_read(800, 0, 4096)
            .await
            .unwrap()
            .into_delta()
            .unwrap();
        assert_eq!(delta.total_bytes, total);
        assert_eq!(delta.bytes, &chunk.as_bytes()[..4096]);
    }
}
