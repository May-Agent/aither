//! Terminal-first command execution with heel sandboxing.
//!
//! This crate provides a terminal-first execution model where:
//! - LLM has a single `terminal` tool with different permission modes
//! - All tools are exposed as CLI commands via IPC
//! - Commands can be piped and composed freely
//! - Outputs are stored with URLs for context management
//!
//! # Setting Up the Terminal Tool
//!
//! ```rust,ignore
//! use aither_sandbox::{Shell, ToolRegistryBuilder};
//! use std::sync::Arc;
//!
//! // Creates a working directory heel names with four random words (e.g.,
//! // amber-forest-thunder-pearl/), not a leased id, with an outputs/
//! // subdirectory and IPC command wrapper scripts; the ids the model reads are
//! // leased from `with_id_allocator`'s allocator
//! let shell = Shell::builder(sandbox)
//!     .limits(limits)
//!     .output(output)
//!     .wording(wording)
//!     .tools(tools)
//!     .build()
//!     .await?;
//! let registry = Arc::new(ToolRegistryBuilder::new().build(shell.output_store().clone()));
//! let tool = shell.with_registry(registry);
//!
//! // Access the working directory path
//! tracing::info!("Working dir: {}", tool.working_dir().display());
//! ```
//!
//! # Registering Tools as IPC Commands
//!
//! Tools can be made available to terminal sessions running in the sandbox:
//!
//! ```rust,ignore
//! use aither_sandbox::{ToolRegistryBuilder, register_tool_command};
//! use aither_sandbox::builtin::builtin_router;
//!
//! // 1. Configure tools
//! let mut registry = ToolRegistryBuilder::new();
//! registry.configure_tool(websearch_tool);
//! registry.configure_tool(webfetch_tool);
//! let registry = std::sync::Arc::new(registry.build("./outputs"));
//!
//! // 2. Create router with built-in commands (reload)
//! let router = builtin_router();
//!
//! // 3. Register tool commands with the router
//! let router = register_tool_command(router, registry.clone(), "websearch");
//! let router = register_tool_command(router, registry.clone(), "webfetch");
//! ```
//!
//! # Built-in Commands
//!
//! The [`builtin`] module provides commands always available in the sandbox:
//!
//! - `reload <url>` - Request to load file content back into agent context

#![allow(clippy::module_name_repetitions)]

mod bollard_exec;
mod command;
mod command_set;
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
mod tool_distribution;
mod tool_environment;
mod tool_python;
mod tool_runtime;
mod tool_transport;
mod wording;

/// Built-in IPC commands (ask, reload).
pub mod builtin;

/// Jobserver token pool brokering across commands.
#[cfg(all(unix, feature = "jobserver"))]
pub mod jobserver;

/// Declarative host-based network policy.
pub mod network_policy;

/// Isolated git for tests, shared with dependent crates through the `test-git` feature.
#[cfg(any(test, feature = "test-git"))]
pub mod test_git;

/// The distribution's pinned heel for tests, shared with dependent crates
/// through the `test-heel` feature.
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
pub use command::{
    AnyCommandGate, AnyCommandObserver, CommandBlocked, CommandCall, CommandEnvelope, CommandGate,
    CommandObserver, CommandPayload, CommandRole, CommandUsage, DynTerminalTool, DynToolHandler,
    GateDecision, IpcToolCommand, KeptExchangeReceiver, POSITIONAL_KEYWORD, PositionalLayoutError,
    ToolCallCommand, ToolCommand, ToolRegistry, ToolRegistryBuilder, cli_to_json,
    output_saved_text, register_ipc_gateway_command, register_tool_command, register_tool_direct,
    schema_to_help, tool_output_to_payload, validate_positional_layout,
};
pub use command_set::{AnyCommandSet, CommandSet};
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
    ContainerShellRuntime, ExecLifetime, ShellBackend, ShellSessionRegistry,
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
    CompletedCommand, Configured, DirectoryResetReason, Given, Missing, Script, Shell, ShellArgs,
    ShellBuilder, ShellRunError, ShellTool, TerminalBinding, TerminalError, TerminalToolFactory,
    TerminalToolFactoryError, TimeoutKill, Unconfigured, open_network_audit_log,
};
pub use tool_distribution::{
    HostProgramArtifact, PinnedArtifact, ToolArtifact, ToolArtifactFormat, ToolArtifactLocation,
    ToolDistribution,
};
pub use tool_runtime::{
    DistributedTool, HostProgram, HostProgramPath, ImageTool, ImageToolDefect, ImageTools,
    ManagedTool, PreparedEnvironment, PreparedTools, PythonPin, PythonRuntime, ToolBinary,
    ToolBundle, ToolCatalog, ToolRuntimeError, ToolTarget,
};
pub use tool_transport::{
    ContainerToolTransport, SshToolTransport, ToolEnvironmentInfo, ToolTransport,
    probe_tool_environment,
};
pub use wording::{
    AnyShellWording, CallRefusal, CommandKilled, FoldedLines, JobInputSent, JobKillAsked,
    JsonAsTsv, LineRange, MonitorStarted, OmittedLines, OutputSaved, Refused, RepeatedLines,
    ReservedForMain, RunFailure, ShellWording, StoredBinaryPart,
};
