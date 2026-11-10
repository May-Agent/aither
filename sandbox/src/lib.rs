//! Sandboxed shell execution with heel.
//!
//! This crate provides the general capabilities an embedder builds a
//! terminal-style agent tool on:
//!
//! - [`Sandbox`] spawns a process under heel's sandbox, in a container or on
//!   an SSH server, with live [`Grants`] and an [`EnvSource`] the embedder
//!   fills at each launch.
//! - [`Shell`] executes scripts for a model on a [`Sandbox`]: a carried
//!   directory, jobs, waiting, output handling and [`ScriptPolicy`] checks
//!   before a script runs.
//! - [`ShellTool`] and the job control tools are ordinary
//!   [`Tool`](aither_core::llm::Tool)s the embedder names and describes.
//!
//! The embedder owns the commands its scripts call back into, the programs
//! its sandboxes run with and every sentence the model reads.
//!
//! ```rust,ignore
//! use aither_sandbox::{Backend, Sandbox, Shell};
//!
//! let sandbox = Sandbox::new(Backend::Host { heel }, executor, working_dir);
//! let shell = Shell::builder(sandbox)
//!     .interpreters(interpreters)
//!     .limits(limits)
//!     .output(output)
//!     .wording(wording)
//!     .tools(tools)
//!     .build()
//!     .await?;
//! let tool = shell.tool();
//! ```

#![allow(clippy::module_name_repetitions)]

mod bollard_exec;
mod container;
#[cfg(all(unix, feature = "jobserver"))]
mod container_jobserver;
mod events;
mod foreground_wait;
mod interpreter;
mod job_registry;
mod jobs;
mod limits;
mod memory;
pub mod monitor;
mod outage;
mod output;
mod output_compress;
mod output_policy;
mod policy;
mod process_tree;
mod shell_session;
mod shell_syntax;
mod spawn;
mod ssh;
mod stdin_watch;
mod subagent;
mod terminal;
mod wording;

/// The job control tools and the identities an embedder gives its shell tools.
pub mod builtin;

/// Jobserver token pool brokering across commands.
#[cfg(all(unix, feature = "jobserver"))]
pub mod jobserver;

/// Declarative host-based network policy.
pub mod network_policy;

/// Isolated git for tests, shared with dependent crates through the `test-git` feature.
#[cfg(any(test, feature = "test-git"))]
pub mod test_git;

/// A `heel` for tests, shared with dependent crates through the `test-heel`
/// feature.
#[cfg(any(test, feature = "test-heel"))]
pub mod test_heel;

/// Budgets for tests, shared through the `test-budgets` feature.
#[cfg(any(test, feature = "test-budgets"))]
pub mod test_budgets;

/// The wording tests read, shared through the `test-budgets` feature.
#[cfg(any(test, feature = "test-budgets"))]
pub mod test_wording;

/// Id allocators for tests, shared with dependent crates through the
/// `test-ids` feature.
#[cfg(any(test, feature = "test-ids"))]
pub mod test_ids;

pub use bollard_exec::BollardContainerExec;
pub use builtin::{
    JobInputArgs, JobInputTool, JobKillArgs, JobKillTool, JobReadArgs, JobReadTool,
    ShellToolSettings, ToolIdentity,
};
pub use container::{
    ContainerImageSpec, ContainerLaunchSpec, ContainerPath, ContainerRuntimeKind, MountAccess,
    MountRoot, MountRootError, MountSpec, RelativeContainerPath, RuntimePreference,
};
pub use events::{
    DescendantsKilled, JobEnded, JobStarted, LongRunning, ShellEvent, ShellEvents, StdinBlocked,
    StdinWatchFailed,
};
pub use foreground_wait::ForegroundWait;
/// Re-export of heel's rolling network audit log for terminal-tool configuration.
pub use heel::NetworkAuditLog;
/// Re-export of heel's scheduling priority types for terminal-tool configuration.
pub use heel::{CpuPriority, IoPriority, NiceLevel, ProcessPriority};
pub use interpreter::Interpreter;
pub use job_registry::{
    Delivery, JobDelta, JobInfo, JobPlace, JobRead, JobStatus, KillCause, OwedCompletion,
    PendingCompletion, UnchangedRead,
};
pub use jobs::{Job, JobWaitError, Jobs, RunOutcome, ScriptResult};
#[cfg(all(unix, feature = "jobserver"))]
pub use jobserver::{
    CommandFifo, JobBroker, JobBrokerService, JobPool, JobserverError, job_broker_channel,
};
pub use limits::{LiveDuration, ShellLimits};
pub use memory::{
    AnyContainerMemory, AnyFootprintReader, AnyFootprintRegistry, CgroupMemory, ContainerMemory,
    ContainerSnapshot, FileFootprintRegistry, FootprintError, FootprintReader, FootprintRegistry,
    JOB_MARKER_ENV, JobFootprint, JobMemory, MemoryDomain, MemorySize, MemorySnapshot,
    ProbedContainerMemory, ResponseWindow, SystemFootprints,
};
pub use monitor::{
    MonitorArgs, MonitorEnd, MonitorEnded, MonitorIncrement, MonitorRefusal, MonitorSettings,
    MonitorSource, MonitorStalled, MonitorTool,
};
pub use network_policy::{
    CompiledNetworkPolicy, NetworkPolicyConfig, NetworkPolicyError, NetworkPolicyMode,
    NetworkVerdict,
};
pub use output::{MediaResolution, OutputFormat, OutputStore};
pub use output_policy::{MediaBounds, OutputLimits, OutputPolicy};
pub use policy::{
    AnyScriptPolicy, Carry, SandboxMode, ScriptCheck, ScriptDecision, ScriptDir, ScriptEnd,
    ScriptOrigin, ScriptOutcome, ScriptPolicies, ScriptPolicy, ScriptRefusal,
};
pub use process_tree::{
    JOB_MARKER_VAR, JobProcess, KillTreeError, ProcessTableError, kill_process_tree,
};
pub use shell_session::{
    AnyContainerExec, ContainerExec, ContainerExecOutcome, ContainerProbe, ContainerProbeError,
    ContainerShellRuntime, ExecLifetime, ShellSessionRegistry,
};
pub use shell_syntax::{
    CommandId, LineId, PipelineId, PipelineStage, ShellCommand, ShellParseError, ShellScript,
};
pub use spawn::{
    AnyDomainApprover, ApproveAllDomains, Backend, ChildExit, ContainerLaunch, DenyAllDomains,
    Descendants, DomainApprover, DomainRequest, EnvSource, GitDirAccess, GrantSnapshot, Grants,
    HostRoute, HostRouteError, Pipe, RemoteLaunch, Sandbox, SandboxedChild, SpawnCommand,
    SpawnError, SpawnGrant, SpawnSpec, Stdio, TreeLifetime,
};
pub use ssh::{PreparedSsh, SshPlatform, SshProbeError, SshServer, SshServerError, probe_ssh_os};
pub use stdin_watch::StdinWatchSignal;
pub use subagent::{
    AnySubagentSpaces, CommitLedger, CommitLedgerError, CommitMark, ImportBudget, PreparedSpace,
    RunSpace, SpacePreparationError, SubagentCommit, SubagentGrant, SubagentGrantError,
    SubagentSpaces,
};
pub use terminal::{
    CompletedCommand, DirectoryResetReason, Given, Missing, Script, Shell, ShellArgs, ShellBuilder,
    ShellRunError, ShellTool, TerminalError, TimeoutKill, open_network_audit_log,
};
pub use wording::{
    AnyShellWording, CallRefusal, CommandKilled, FoldedLines, JobInputSent, JobKillAsked,
    JsonAsTsv, LineRange, MonitorStarted, OmittedLines, OutputSaved, Refused, RepeatedLines,
    RunFailure, ShellWording, StoredBinaryPart,
};
