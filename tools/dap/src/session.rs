//! High-level DAP [`DebugSession`] coordinating handshake, breakpoints,
//! stopped snapshots, bounded output capture, and async stop notifications.

use std::time::{Duration, Instant};

use async_channel::{Receiver, Sender};
use async_io::Timer;
use executor_core::{Executor, Task};
use futures_lite::future;
use serde_json::Value;
use tracing::debug;

use crate::breakpoints::{
    BreakpointStore, BreakpointSync, FunctionBreakpointSpec, ManagedBreakpoint,
    SourceBreakpointSpec,
};
use crate::client::DapClient;
use crate::error::DapError;
use crate::output::{OutputBuffer, OutputBufferConfig, OutputCategory, OutputSnapshot};
use crate::protocol::{
    BreakpointEventBody, Capabilities, ContinueArguments, ContinuedEventBody, DapEvent,
    DisconnectArguments, EvaluateArguments, EvaluateResponseBody, ExitedEventBody,
    InitializeRequestArguments, OutputEventBody, ScopesArguments, ScopesResponseBody,
    SetBreakpointsResponseBody, StackTraceArguments, StackTraceResponseBody, StoppedEventBody,
    TerminateArguments, ThreadActionArguments, ThreadsResponseBody, VariablesArguments,
    VariablesResponseBody,
};
use crate::snapshot::{
    EvalResult, ScopeVariables, StackSnapshot, StopSummary, StoppedSnapshot, ThreadSummary,
    VariableEntry, VariablesView, read_source_snippet,
};
use crate::tool::{
    DebugArgs, DebugNotification, DebugResult, DebugSessionState, DebugStatus, EvalContext,
    TerminationSummary,
};
use crate::transport::DapTransport;

/// Default bounded wait timeout for execution-resuming commands (`continue`, stepping).
pub const DEFAULT_EXECUTION_TIMEOUT: Duration = Duration::from_secs(5);

/// Default bounded wait timeout for `wait` commands.
pub const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(10);

/// Default maximum stack frames fetched per stop snapshot.
const DEFAULT_STACK_LEVELS: usize = 64;

/// Default maximum variables per scope or expansion when no explicit limit is passed.
const DEFAULT_VARIABLES_LIMIT: usize = 100;

/// Configuration for starting a [`DebugSession`].
#[derive(Debug, Clone)]
pub struct DebugSessionConfig {
    /// Adapter identifier reported in `initialize` and `status`.
    pub adapter_id: String,
    /// Optional human-readable target descriptor (program path or `host:port`).
    pub target: Option<String>,
    /// Optional OS process ID of the debuggee.
    pub target_pid: Option<u32>,
    /// Capacity limits for the bounded target output ring buffer.
    pub output_buffer: OutputBufferConfig,
}

impl DebugSessionConfig {
    /// Creates a session configuration for `adapter_id` with default output bounds.
    #[must_use]
    pub fn new(adapter_id: impl Into<String>) -> Self {
        Self {
            adapter_id: adapter_id.into(),
            target: None,
            target_pid: None,
            output_buffer: OutputBufferConfig::default(),
        }
    }

    /// Sets the target label shown in [`DebugStatus`].
    #[must_use]
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Sets the known target PID shown in [`DebugStatus`].
    #[must_use]
    pub const fn with_target_pid(mut self, pid: u32) -> Self {
        self.target_pid = Some(pid);
        self
    }

    /// Overrides the output ring buffer limits.
    #[must_use]
    pub const fn with_output_buffer(mut self, config: OutputBufferConfig) -> Self {
        self.output_buffer = config;
        self
    }
}

/// Outcome of an execution-control operation (`launch`, `attach`, `continue`, `wait`, stepping).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionOutcome {
    /// Target stopped synchronously before the wait timeout elapsed.
    Stopped(Box<StopSummary>),
    /// Target is running past the wait timeout; an asynchronous [`DebugNotification`]
    /// will fire when it later stops or terminates.
    Running,
    /// Target exited or terminated.
    Terminated(TerminationSummary),
}

impl ExecutionOutcome {
    /// Converts this outcome into a [`DebugResult`].
    #[must_use]
    pub fn into_result(self, running_note: impl Into<String>) -> DebugResult {
        match self {
            Self::Stopped(summary) => DebugResult::Stopped(summary),
            Self::Running => DebugResult::Running {
                note: running_note.into(),
            },
            Self::Terminated(summary) => DebugResult::Terminated(summary),
        }
    }
}

enum StepKind {
    Continue,
    Next,
    StepIn,
    StepOut,
    Pause,
}

impl StepKind {
    const fn dap_command(&self) -> &'static str {
        match self {
            Self::Continue => "continue",
            Self::Next => "next",
            Self::StepIn => "stepIn",
            Self::StepOut => "stepOut",
            Self::Pause => "pause",
        }
    }

    const fn requires_stopped(&self) -> bool {
        !matches!(self, Self::Pause)
    }
}

enum SessionCommand {
    LaunchOrAttach {
        command: &'static str,
        arguments: Value,
        timeout: Duration,
        reply: Sender<Result<ExecutionOutcome, DapError>>,
    },
    Status {
        reply: Sender<DebugStatus>,
    },
    SetSourceBreakpoint {
        spec: SourceBreakpointSpec,
        reply: Sender<Result<ManagedBreakpoint, DapError>>,
    },
    SetFunctionBreakpoint {
        spec: FunctionBreakpointSpec,
        reply: Sender<Result<ManagedBreakpoint, DapError>>,
    },
    ListBreakpoints {
        reply: Sender<Vec<ManagedBreakpoint>>,
    },
    RemoveBreakpoint {
        id: u32,
        reply: Sender<Result<ManagedBreakpoint, DapError>>,
    },
    Step {
        kind: StepKind,
        thread_id: Option<i64>,
        timeout: Duration,
        reply: Sender<Result<ExecutionOutcome, DapError>>,
    },
    Wait {
        timeout: Duration,
        reply: Sender<Result<ExecutionOutcome, DapError>>,
    },
    Threads {
        reply: Sender<Result<Vec<ThreadSummary>, DapError>>,
    },
    Stack {
        thread_id: Option<i64>,
        limit: Option<usize>,
        reply: Sender<Result<StackSnapshot, DapError>>,
    },
    Variables {
        frame: Option<usize>,
        expand: Option<usize>,
        limit: Option<usize>,
        reply: Sender<Result<VariablesView, DapError>>,
    },
    Evaluate {
        expression: String,
        frame: Option<usize>,
        context: EvalContext,
        reply: Sender<Result<EvalResult, DapError>>,
    },
    Output {
        category: OutputCategory,
        tail: Option<usize>,
        clear: bool,
        reply: Sender<OutputSnapshot>,
    },
    Detach {
        reply: Sender<Result<(), DapError>>,
    },
    Terminate {
        reply: Sender<Result<TerminationSummary, DapError>>,
    },
}

/// High-level Debug Adapter Protocol session handle.
///
/// Owns a single background actor that processes adapter events continuously,
/// keeps per-file breakpoint tables synchronized, indexes stopped snapshots,
/// invalidates handles on resume, captures bounded target output, and emits
/// one-shot [`DebugNotification`]s when a running target later stops or exits.
#[derive(Debug, Clone)]
pub struct DebugSession {
    commands: Sender<SessionCommand>,
    capabilities: Capabilities,
}

impl DebugSession {
    /// Connects over `transport`, performs the DAP `initialize` handshake, and
    /// spawns the session actor on `executor`.
    ///
    /// Returns the [`DebugSession`] handle paired with the channel carrying
    /// asynchronous [`DebugNotification`] wakeups.
    ///
    /// # Errors
    ///
    /// Returns [`DapError`] if `initialize` fails or the transport closes.
    pub async fn start<T, E>(
        executor: &E,
        transport: T,
        config: DebugSessionConfig,
    ) -> Result<(Self, Receiver<DebugNotification>), DapError>
    where
        T: DapTransport + 'static,
        E: Executor,
    {
        let (client, event_rx) = DapClient::connect(executor, transport);
        let init_args = InitializeRequestArguments::for_adapter(&config.adapter_id);
        let capabilities: Capabilities = client.request("initialize", &init_args).await?;

        let (cmd_tx, cmd_rx) = async_channel::unbounded();
        let (notif_tx, notif_rx) = async_channel::unbounded();

        let actor = SessionActor::new(client, capabilities.clone(), config, notif_tx);
        executor.spawn(actor.run(cmd_rx, event_rx)).detach();

        Ok((
            Self {
                commands: cmd_tx,
                capabilities,
            },
            notif_rx,
        ))
    }

    /// Returns the adapter capabilities reported during `initialize`.
    #[must_use]
    pub const fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    /// Runs the DAP `launch` + `initialized` + `configurationDone` sequence and
    /// waits up to `timeout` for the initial stop or exit.
    ///
    /// # Errors
    ///
    /// Returns [`DapError`] if the adapter rejects launch or closes before completing setup.
    pub async fn launch(
        &self,
        arguments: Value,
        timeout: Duration,
    ) -> Result<ExecutionOutcome, DapError> {
        self.call(|reply| SessionCommand::LaunchOrAttach {
            command: "launch",
            arguments,
            timeout,
            reply,
        })
        .await?
    }

    /// Runs the DAP `attach` + `initialized` + `configurationDone` sequence and
    /// waits up to `timeout` for the initial stop or exit.
    ///
    /// # Errors
    ///
    /// Returns [`DapError`] if the adapter rejects attach or closes before completing setup.
    pub async fn attach(
        &self,
        arguments: Value,
        timeout: Duration,
    ) -> Result<ExecutionOutcome, DapError> {
        self.call(|reply| SessionCommand::LaunchOrAttach {
            command: "attach",
            arguments,
            timeout,
            reply,
        })
        .await?
    }

    /// Queries the current session status without blocking on adapter RPCs.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::SessionClosed`] if the session actor has shut down.
    pub async fn status(&self) -> Result<DebugStatus, DapError> {
        self.call(|reply| SessionCommand::Status { reply }).await
    }

    /// Sets a source breakpoint incrementally, re-sending the full file breakpoint
    /// list to the adapter and returning the verified breakpoint record.
    ///
    /// # Errors
    ///
    /// Returns [`DapError`] if the adapter rejects `setBreakpoints` or the session closed.
    pub async fn set_source_breakpoint(
        &self,
        spec: SourceBreakpointSpec,
    ) -> Result<ManagedBreakpoint, DapError> {
        self.call(|reply| SessionCommand::SetSourceBreakpoint { spec, reply })
            .await?
    }

    /// Sets a function breakpoint incrementally.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::Unsupported`] if the adapter lacks function breakpoint
    /// support, or any transport/adapter error.
    pub async fn set_function_breakpoint(
        &self,
        spec: FunctionBreakpointSpec,
    ) -> Result<ManagedBreakpoint, DapError> {
        self.call(|reply| SessionCommand::SetFunctionBreakpoint { spec, reply })
            .await?
    }

    /// Lists all active client-managed breakpoints.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::SessionClosed`] if the session actor has shut down.
    pub async fn list_breakpoints(&self) -> Result<Vec<ManagedBreakpoint>, DapError> {
        self.call(|reply| SessionCommand::ListBreakpoints { reply })
            .await
    }

    /// Removes a breakpoint by client ID and reconciles the adapter's breakpoint list.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::BreakpointNotFound`] if `id` does not exist or any adapter error.
    pub async fn remove_breakpoint(&self, id: u32) -> Result<ManagedBreakpoint, DapError> {
        self.call(|reply| SessionCommand::RemoveBreakpoint { id, reply })
            .await?
    }

    /// Resumes target execution and waits up to `timeout` for the next stop or termination.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::NotStopped`] if the target is not stopped, or any adapter error.
    pub async fn continue_execution(
        &self,
        thread_id: Option<i64>,
        timeout: Duration,
    ) -> Result<ExecutionOutcome, DapError> {
        self.call(|reply| SessionCommand::Step {
            kind: StepKind::Continue,
            thread_id,
            timeout,
            reply,
        })
        .await?
    }

    /// Waits up to `timeout` for a running target to stop or terminate.
    ///
    /// # Errors
    ///
    /// Returns [`DapError`] if the session closed.
    pub async fn wait(&self, timeout: Duration) -> Result<ExecutionOutcome, DapError> {
        self.call(|reply| SessionCommand::Wait { timeout, reply })
            .await?
    }

    /// Requests the adapter to pause execution and waits up to `timeout` for the stop.
    ///
    /// # Errors
    ///
    /// Returns [`DapError`] if the adapter rejects `pause` or the session ended.
    pub async fn pause(
        &self,
        thread_id: Option<i64>,
        timeout: Duration,
    ) -> Result<ExecutionOutcome, DapError> {
        self.call(|reply| SessionCommand::Step {
            kind: StepKind::Pause,
            thread_id,
            timeout,
            reply,
        })
        .await?
    }

    /// Steps over the current source line and waits up to `timeout` for the next stop.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::NotStopped`] if the target is not stopped, or any adapter error.
    pub async fn next(
        &self,
        thread_id: Option<i64>,
        timeout: Duration,
    ) -> Result<ExecutionOutcome, DapError> {
        self.call(|reply| SessionCommand::Step {
            kind: StepKind::Next,
            thread_id,
            timeout,
            reply,
        })
        .await?
    }

    /// Steps into a call on the current source line and waits up to `timeout` for the next stop.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::NotStopped`] if the target is not stopped, or any adapter error.
    pub async fn step_in(
        &self,
        thread_id: Option<i64>,
        timeout: Duration,
    ) -> Result<ExecutionOutcome, DapError> {
        self.call(|reply| SessionCommand::Step {
            kind: StepKind::StepIn,
            thread_id,
            timeout,
            reply,
        })
        .await?
    }

    /// Steps out of the current frame and waits up to `timeout` for the next stop.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::NotStopped`] if the target is not stopped, or any adapter error.
    pub async fn step_out(
        &self,
        thread_id: Option<i64>,
        timeout: Duration,
    ) -> Result<ExecutionOutcome, DapError> {
        self.call(|reply| SessionCommand::Step {
            kind: StepKind::StepOut,
            thread_id,
            timeout,
            reply,
        })
        .await?
    }

    /// Lists threads in the debug target.
    ///
    /// # Errors
    ///
    /// Returns [`DapError`] if the session ended or the adapter failed.
    pub async fn threads(&self) -> Result<Vec<ThreadSummary>, DapError> {
        self.call(|reply| SessionCommand::Threads { reply }).await?
    }

    /// Returns the indexed call stack for the stopped thread (or `thread_id`).
    ///
    /// # Errors
    ///
    /// Returns [`DapError::NotStopped`] if execution is not stopped.
    pub async fn stack(
        &self,
        thread_id: Option<i64>,
        limit: Option<usize>,
    ) -> Result<StackSnapshot, DapError> {
        self.call(|reply| SessionCommand::Stack {
            thread_id,
            limit,
            reply,
        })
        .await?
    }

    /// Inspects variables for `frame` (default `0`) or lazily expands one level
    /// of `expand` handle from the current stopped snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::NotStopped`], [`DapError::InvalidFrameIndex`], or
    /// [`DapError::InvalidVariableHandle`] when the snapshot or handle is invalid.
    pub async fn variables(
        &self,
        frame: Option<usize>,
        expand: Option<usize>,
        limit: Option<usize>,
    ) -> Result<VariablesView, DapError> {
        self.call(|reply| SessionCommand::Variables {
            frame,
            expand,
            limit,
            reply,
        })
        .await?
    }

    /// Evaluates `expression` in the context of `frame` (default `0`).
    ///
    /// # Errors
    ///
    /// Returns [`DapError::NotStopped`], [`DapError::InvalidFrameIndex`], or an
    /// adapter evaluation error.
    pub async fn evaluate(
        &self,
        expression: impl Into<String>,
        frame: Option<usize>,
        context: EvalContext,
    ) -> Result<EvalResult, DapError> {
        self.call(|reply| SessionCommand::Evaluate {
            expression: expression.into(),
            frame,
            context,
            reply,
        })
        .await?
    }

    /// Reads recent captured target output from the bounded ring buffer.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::SessionClosed`] if the session actor shut down.
    pub async fn output(
        &self,
        category: OutputCategory,
        tail: Option<usize>,
        clear: bool,
    ) -> Result<OutputSnapshot, DapError> {
        self.call(|reply| SessionCommand::Output {
            category,
            tail,
            clear,
            reply,
        })
        .await
    }

    /// Detaches from the target without terminating it and closes the adapter transport.
    ///
    /// # Errors
    ///
    /// Returns [`DapError`] if the session closed before detaching.
    pub async fn detach(&self) -> Result<(), DapError> {
        self.call(|reply| SessionCommand::Detach { reply }).await?
    }

    /// Terminates the target process and closes the adapter transport.
    ///
    /// # Errors
    ///
    /// Returns [`DapError`] if the session closed before terminating.
    pub async fn terminate(&self) -> Result<TerminationSummary, DapError> {
        self.call(|reply| SessionCommand::Terminate { reply })
            .await?
    }

    /// Dispatches an in-session [`DebugArgs`] command and returns its [`DebugResult`].
    ///
    /// Session-creating commands (`launch`, `attach`, `connect`) return
    /// [`DapError::RequiresHostBroker`] because creating or connecting transports
    /// belongs to the host broker.
    ///
    /// # Errors
    ///
    /// Returns [`DapError`] for invalid state, adapter errors, or host-only commands.
    pub async fn handle_command(&self, args: &DebugArgs) -> Result<DebugResult, DapError> {
        match args {
            DebugArgs::Launch { .. } => Err(DapError::RequiresHostBroker("launch")),
            DebugArgs::Attach { .. } => Err(DapError::RequiresHostBroker("attach")),
            DebugArgs::Connect { .. } => Err(DapError::RequiresHostBroker("connect")),
            DebugArgs::Status => Ok(DebugResult::Status(self.status().await?)),
            DebugArgs::Detach => {
                let target_pid = self.status().await.ok().and_then(|s| s.target_pid);
                self.detach().await?;
                Ok(DebugResult::Detached { target_pid })
            }
            DebugArgs::Terminate => Ok(DebugResult::Terminated(self.terminate().await?)),
            DebugArgs::Break {
                file,
                line,
                column,
                condition,
                hit_condition,
                log_message,
            } => {
                let bp = self
                    .set_source_breakpoint(SourceBreakpointSpec {
                        file: file.clone(),
                        line: *line,
                        column: *column,
                        condition: condition.clone(),
                        hit_condition: hit_condition.clone(),
                        log_message: log_message.clone(),
                    })
                    .await?;
                Ok(DebugResult::BreakpointSet(bp))
            }
            DebugArgs::BreakFunction {
                name,
                condition,
                hit_condition,
            } => {
                let bp = self
                    .set_function_breakpoint(FunctionBreakpointSpec {
                        name: name.clone(),
                        condition: condition.clone(),
                        hit_condition: hit_condition.clone(),
                    })
                    .await?;
                Ok(DebugResult::BreakpointSet(bp))
            }
            DebugArgs::Breaks => Ok(DebugResult::Breakpoints(self.list_breakpoints().await?)),
            DebugArgs::Unbreak { id } => Ok(DebugResult::BreakpointRemoved(
                self.remove_breakpoint(*id).await?,
            )),
            DebugArgs::Continue { .. }
            | DebugArgs::Wait { .. }
            | DebugArgs::Pause { .. }
            | DebugArgs::Next { .. }
            | DebugArgs::StepIn { .. }
            | DebugArgs::StepOut { .. } => self.handle_execution_command(args).await,
            DebugArgs::Threads => Ok(DebugResult::Threads(self.threads().await?)),
            DebugArgs::Stack { thread_id, limit } => {
                Ok(DebugResult::Stack(self.stack(*thread_id, *limit).await?))
            }
            DebugArgs::Vars {
                frame,
                expand,
                limit,
            } => Ok(DebugResult::Variables(
                self.variables(*frame, *expand, *limit).await?,
            )),
            DebugArgs::Eval {
                expression,
                frame,
                context,
            } => Ok(DebugResult::Eval(
                self.evaluate(expression.clone(), *frame, context.unwrap_or_default())
                    .await?,
            )),
            DebugArgs::Output {
                category,
                tail,
                clear,
            } => Ok(DebugResult::Output(
                self.output(category.unwrap_or_default(), *tail, *clear)
                    .await?,
            )),
        }
    }

    async fn handle_execution_command(&self, args: &DebugArgs) -> Result<DebugResult, DapError> {
        match args {
            DebugArgs::Continue {
                thread_id,
                timeout_ms,
            } => {
                let timeout = timeout_ms.map_or(DEFAULT_EXECUTION_TIMEOUT, Duration::from_millis);
                Ok(self
                    .continue_execution(*thread_id, timeout)
                    .await?
                    .into_result("Execution resumed; target is running."))
            }
            DebugArgs::Wait { timeout_ms } => {
                let timeout = timeout_ms.map_or(DEFAULT_WAIT_TIMEOUT, Duration::from_millis);
                Ok(self
                    .wait(timeout)
                    .await?
                    .into_result("Target is still running."))
            }
            DebugArgs::Pause {
                thread_id,
                timeout_ms,
            } => {
                let timeout = timeout_ms.map_or(DEFAULT_EXECUTION_TIMEOUT, Duration::from_millis);
                Ok(self
                    .pause(*thread_id, timeout)
                    .await?
                    .into_result("Pause requested; target has not stopped yet."))
            }
            DebugArgs::Next {
                thread_id,
                timeout_ms,
            } => {
                let timeout = timeout_ms.map_or(DEFAULT_EXECUTION_TIMEOUT, Duration::from_millis);
                Ok(self
                    .next(*thread_id, timeout)
                    .await?
                    .into_result("Step over is still running."))
            }
            DebugArgs::StepIn {
                thread_id,
                timeout_ms,
            } => {
                let timeout = timeout_ms.map_or(DEFAULT_EXECUTION_TIMEOUT, Duration::from_millis);
                Ok(self
                    .step_in(*thread_id, timeout)
                    .await?
                    .into_result("Step in is still running."))
            }
            DebugArgs::StepOut {
                thread_id,
                timeout_ms,
            } => {
                let timeout = timeout_ms.map_or(DEFAULT_EXECUTION_TIMEOUT, Duration::from_millis);
                Ok(self
                    .step_out(*thread_id, timeout)
                    .await?
                    .into_result("Step out is still running."))
            }
            _ => unreachable!("only execution commands dispatch here"),
        }
    }

    async fn call<R>(
        &self,
        build: impl FnOnce(Sender<R>) -> SessionCommand,
    ) -> Result<R, DapError> {
        let (tx, rx) = async_channel::bounded(1);
        self.commands
            .send(build(tx))
            .await
            .map_err(|_| DapError::SessionClosed)?;
        rx.recv().await.map_err(|_| DapError::SessionClosed)
    }
}

enum ActorWake {
    Command(Option<SessionCommand>),
    Event(Option<DapEvent>),
}

enum WaitStep {
    Event(Option<DapEvent>),
    TimedOut,
}

struct SessionActor {
    client: DapClient,
    capabilities: Capabilities,
    config: DebugSessionConfig,
    notifications: Sender<DebugNotification>,
    state: DebugSessionState,
    initialized_received: bool,
    configuration_done_sent: bool,
    breakpoints: BreakpointStore,
    output: OutputBuffer,
    snapshot: Option<StoppedSnapshot>,
    last_thread_id: Option<i64>,
    exit_code: Option<i64>,
    termination_reason: Option<String>,
    notify_on_async_stop: bool,
}

impl SessionActor {
    const fn new(
        client: DapClient,
        capabilities: Capabilities,
        config: DebugSessionConfig,
        notifications: Sender<DebugNotification>,
    ) -> Self {
        let output = OutputBuffer::new(config.output_buffer);
        Self {
            client,
            capabilities,
            config,
            notifications,
            state: DebugSessionState::Starting,
            initialized_received: false,
            configuration_done_sent: false,
            breakpoints: BreakpointStore::new(),
            output,
            snapshot: None,
            last_thread_id: None,
            exit_code: None,
            termination_reason: None,
            notify_on_async_stop: false,
        }
    }

    async fn run(mut self, commands: Receiver<SessionCommand>, events: Receiver<DapEvent>) {
        loop {
            let wake = future::or(
                async { ActorWake::Event(events.recv().await.ok()) },
                async { ActorWake::Command(commands.recv().await.ok()) },
            )
            .await;

            match wake {
                ActorWake::Event(Some(event)) => {
                    self.handle_event(event, true).await;
                }
                ActorWake::Event(None) => {
                    self.mark_terminated(Some("adapter connection closed".to_owned()), true);
                    // Keep answering read-only commands (`status`, `output`) until
                    // all handles drop.
                    while let Ok(cmd) = commands.recv().await {
                        if self.handle_command(cmd, &events).await {
                            break;
                        }
                    }
                    break;
                }
                ActorWake::Command(Some(cmd)) => {
                    if self.handle_command(cmd, &events).await {
                        break;
                    }
                }
                ActorWake::Command(None) => {
                    break;
                }
            }
        }
        self.client.close().await;
    }

    /// Returns `true` when the actor loop should exit.
    async fn handle_command(
        &mut self,
        command: SessionCommand,
        events: &Receiver<DapEvent>,
    ) -> bool {
        match command {
            SessionCommand::LaunchOrAttach {
                command,
                arguments,
                timeout,
                reply,
            } => {
                let result = self
                    .run_launch_or_attach(command, arguments, timeout, events)
                    .await;
                let _ = reply.try_send(result);
            }
            SessionCommand::Status { reply } => {
                let _ = reply.try_send(self.build_status());
            }
            SessionCommand::SetSourceBreakpoint { spec, reply } => {
                let result = self.add_source_breakpoint(spec).await;
                let _ = reply.try_send(result);
            }
            SessionCommand::SetFunctionBreakpoint { spec, reply } => {
                let result = self.add_function_breakpoint(spec).await;
                let _ = reply.try_send(result);
            }
            SessionCommand::ListBreakpoints { reply } => {
                let _ = reply.try_send(self.breakpoints.list());
            }
            SessionCommand::RemoveBreakpoint { id, reply } => {
                let result = self.remove_breakpoint(id).await;
                let _ = reply.try_send(result);
            }
            SessionCommand::Step {
                kind,
                thread_id,
                timeout,
                reply,
            } => {
                let result = self.run_step(&kind, thread_id, timeout, events).await;
                let _ = reply.try_send(result);
            }
            SessionCommand::Wait { timeout, reply } => {
                let result = self.run_wait(timeout, events).await;
                let _ = reply.try_send(result);
            }
            SessionCommand::Threads { reply } => {
                let result = self.fetch_threads().await;
                let _ = reply.try_send(result);
            }
            SessionCommand::Stack {
                thread_id,
                limit,
                reply,
            } => {
                let result = self.fetch_stack(thread_id, limit).await;
                let _ = reply.try_send(result);
            }
            SessionCommand::Variables {
                frame,
                expand,
                limit,
                reply,
            } => {
                let result = self.fetch_variables(frame, expand, limit).await;
                let _ = reply.try_send(result);
            }
            SessionCommand::Evaluate {
                expression,
                frame,
                context,
                reply,
            } => {
                let result = self.eval_expression(expression, frame, context).await;
                let _ = reply.try_send(result);
            }
            SessionCommand::Output {
                category,
                tail,
                clear,
                reply,
            } => {
                let _ = reply.try_send(self.output.snapshot(category, tail, clear));
            }
            SessionCommand::Detach { reply } => {
                self.detach_target().await;
                let _ = reply.try_send(Ok(()));
                return true;
            }
            SessionCommand::Terminate { reply } => {
                let summary = self.terminate_target().await;
                let _ = reply.try_send(Ok(summary));
                return true;
            }
        }
        false
    }

    async fn detach_target(&mut self) {
        self.notify_on_async_stop = false;
        let disconnect_args = DisconnectArguments {
            terminate_debuggee: Some(false),
        };
        let _ = self
            .client
            .request_unit("disconnect", &disconnect_args)
            .await;
        self.invalidate_snapshot();
        self.state = DebugSessionState::Terminated;
        self.termination_reason = Some("detached".to_owned());
        self.client.close().await;
    }

    async fn terminate_target(&mut self) -> TerminationSummary {
        self.notify_on_async_stop = false;
        if self.capabilities.supports_terminate_request {
            let _ = self
                .client
                .request_unit("terminate", &TerminateArguments::default())
                .await;
        }
        let disconnect_args = DisconnectArguments {
            terminate_debuggee: Some(true),
        };
        let _ = self
            .client
            .request_unit("disconnect", &disconnect_args)
            .await;
        self.invalidate_snapshot();
        self.state = DebugSessionState::Terminated;
        if self.termination_reason.is_none() {
            self.termination_reason = Some("terminated".to_owned());
        }
        let summary = TerminationSummary {
            exit_code: self.exit_code,
            reason: self.termination_reason.clone(),
        };
        self.client.close().await;
        summary
    }

    async fn run_launch_or_attach(
        &mut self,
        command: &'static str,
        arguments: Value,
        timeout: Duration,
        events: &Receiver<DapEvent>,
    ) -> Result<ExecutionOutcome, DapError> {
        let deadline = Instant::now() + timeout;
        self.notify_on_async_stop = false;

        // Drive the launch/attach request concurrently with incoming DAP events
        // because adapters frequently emit `initialized` and wait for
        // `configurationDone` before completing the `launch` or `attach` reply.
        let req_client = self.client.clone();
        let req_fut = req_client.request_raw(command, Some(arguments));
        futures_lite::pin!(req_fut);

        let mut req_done = false;

        // If `initialized` arrived right after `initialize`, send `configurationDone`
        // once `launch`/`attach` is in flight or ready.
        while !req_done
            || (self.capabilities.supports_configuration_done_request
                && !self.configuration_done_sent)
        {
            if self.initialized_received
                && self.capabilities.supports_configuration_done_request
                && !self.configuration_done_sent
            {
                self.send_configuration_done().await?;
                if req_done {
                    break;
                }
            }

            let now = Instant::now();
            if now >= deadline {
                if !req_done {
                    return Err(DapError::InitializationFailed(format!(
                        "`{command}` timed out before completing DAP handshake"
                    )));
                }
                break;
            }
            let remaining = deadline.saturating_duration_since(now);

            if req_done {
                enum InitWait {
                    Event(Option<DapEvent>),
                    TimedOut,
                }
                let step = future::or(async { InitWait::Event(events.recv().await.ok()) }, async {
                    Timer::after(remaining).await;
                    InitWait::TimedOut
                })
                .await;
                match step {
                    InitWait::Event(Some(event)) => {
                        self.handle_event(event, false).await;
                    }
                    InitWait::Event(None) => {
                        return Err(DapError::ConnectionClosed);
                    }
                    InitWait::TimedOut => {
                        return Err(DapError::InitializationFailed(
                            "timed out waiting for DAP `initialized` event".to_owned(),
                        ));
                    }
                }
            } else {
                enum HandshakeStep {
                    Reply(Result<crate::protocol::DapResponse, DapError>),
                    Event(Option<DapEvent>),
                    TimedOut,
                }
                let step = future::or(
                    async { HandshakeStep::Reply((&mut req_fut).await) },
                    future::or(
                        async { HandshakeStep::Event(events.recv().await.ok()) },
                        async {
                            Timer::after(remaining).await;
                            HandshakeStep::TimedOut
                        },
                    ),
                )
                .await;
                match step {
                    HandshakeStep::Reply(Ok(_)) => {
                        req_done = true;
                        if !self.capabilities.supports_configuration_done_request {
                            break;
                        }
                    }
                    HandshakeStep::Reply(Err(err)) => return Err(err),
                    HandshakeStep::Event(Some(event)) => {
                        self.handle_event(event, false).await;
                    }
                    HandshakeStep::Event(None) => return Err(DapError::ConnectionClosed),
                    HandshakeStep::TimedOut => {
                        return Err(DapError::InitializationFailed(format!(
                            "`{command}` handshake timed out"
                        )));
                    }
                }
            }
        }

        if self.state == DebugSessionState::Starting {
            self.state = DebugSessionState::Running;
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        self.wait_for_transition(remaining, events).await
    }

    async fn send_configuration_done(&mut self) -> Result<(), DapError> {
        let _ = self.client.request_raw("configurationDone", None).await?;
        self.configuration_done_sent = true;
        Ok(())
    }

    async fn run_step(
        &mut self,
        kind: &StepKind,
        thread_id: Option<i64>,
        timeout: Duration,
        events: &Receiver<DapEvent>,
    ) -> Result<ExecutionOutcome, DapError> {
        if self.state == DebugSessionState::Terminated {
            return Err(DapError::SessionEnded {
                state: self.state.as_str().to_owned(),
            });
        }
        if kind.requires_stopped() && self.state != DebugSessionState::Stopped {
            return Err(DapError::NotStopped {
                state: self.state.as_str().to_owned(),
            });
        }

        let resolved_thread = self.resolve_thread_id(thread_id).await?;
        self.notify_on_async_stop = false;

        if kind.requires_stopped() {
            // Invalidate stopped snapshot tables immediately on resume.
            self.invalidate_snapshot();
            self.state = DebugSessionState::Running;
        }

        let command = kind.dap_command();
        match kind {
            StepKind::Continue => {
                self.client
                    .request_unit(
                        command,
                        &ContinueArguments {
                            thread_id: resolved_thread,
                        },
                    )
                    .await?;
            }
            StepKind::Next | StepKind::StepIn | StepKind::StepOut | StepKind::Pause => {
                self.client
                    .request_unit(
                        command,
                        &ThreadActionArguments {
                            thread_id: resolved_thread,
                        },
                    )
                    .await?;
            }
        }

        self.wait_for_transition(timeout, events).await
    }

    async fn run_wait(
        &mut self,
        timeout: Duration,
        events: &Receiver<DapEvent>,
    ) -> Result<ExecutionOutcome, DapError> {
        self.notify_on_async_stop = false;
        self.wait_for_transition(timeout, events).await
    }

    async fn wait_for_transition(
        &mut self,
        timeout: Duration,
        events: &Receiver<DapEvent>,
    ) -> Result<ExecutionOutcome, DapError> {
        // Drain any events already buffered before checking current state.
        while let Ok(event) = events.try_recv() {
            self.handle_event(event, false).await;
        }

        if let Some(outcome) = self.current_settled_outcome() {
            return Ok(outcome);
        }

        let deadline = Instant::now() + timeout;
        loop {
            let now = Instant::now();
            if now >= deadline {
                // Target is still running past the synchronous wait: arm the one-shot
                // async wakeup so the host gets notified when it stops or terminates.
                self.notify_on_async_stop = true;
                return Ok(ExecutionOutcome::Running);
            }

            let step = future::or(async { WaitStep::Event(events.recv().await.ok()) }, async {
                Timer::after(deadline.saturating_duration_since(now)).await;
                WaitStep::TimedOut
            })
            .await;

            match step {
                WaitStep::Event(Some(event)) => {
                    self.handle_event(event, false).await;
                    if let Some(outcome) = self.current_settled_outcome() {
                        return Ok(outcome);
                    }
                }
                WaitStep::Event(None) => {
                    self.mark_terminated(Some("adapter connection closed".to_owned()), false);
                    return Ok(ExecutionOutcome::Terminated(TerminationSummary {
                        exit_code: self.exit_code,
                        reason: self.termination_reason.clone(),
                    }));
                }
                WaitStep::TimedOut => {
                    self.notify_on_async_stop = true;
                    return Ok(ExecutionOutcome::Running);
                }
            }
        }
    }

    fn current_settled_outcome(&self) -> Option<ExecutionOutcome> {
        match self.state {
            DebugSessionState::Stopped => self
                .snapshot
                .as_ref()
                .map(|snap| ExecutionOutcome::Stopped(Box::new(snap.summary.clone()))),
            DebugSessionState::Terminated => {
                Some(ExecutionOutcome::Terminated(TerminationSummary {
                    exit_code: self.exit_code,
                    reason: self.termination_reason.clone(),
                }))
            }
            DebugSessionState::Inactive
            | DebugSessionState::Starting
            | DebugSessionState::Running => None,
        }
    }

    async fn handle_event(&mut self, event: DapEvent, emit_async_notification: bool) {
        match event.event.as_str() {
            "initialized" => {
                debug!("DAP adapter emitted initialized");
                self.initialized_received = true;
            }
            "stopped" => {
                let body: StoppedEventBody = event
                    .body
                    .and_then(|v| serde_json::from_value(v).ok())
                    .unwrap_or_else(|| StoppedEventBody {
                        reason: "stopped".to_owned(),
                        ..StoppedEventBody::default()
                    });
                if let Some(tid) = body.thread_id {
                    self.last_thread_id = Some(tid);
                }
                let snapshot = self.build_stopped_snapshot(&body).await;
                let summary = snapshot.summary.clone();
                self.snapshot = Some(snapshot);
                self.state = DebugSessionState::Stopped;

                if emit_async_notification && self.notify_on_async_stop {
                    self.notify_on_async_stop = false;
                    let _ = self
                        .notifications
                        .try_send(DebugNotification::Stopped(Box::new(summary)));
                }
            }
            "continued" => {
                if let Some(body) = event
                    .body
                    .and_then(|v| serde_json::from_value::<ContinuedEventBody>(v).ok())
                    && body.thread_id != 0
                {
                    self.last_thread_id = Some(body.thread_id);
                }
                self.invalidate_snapshot();
                self.state = DebugSessionState::Running;
            }
            "output" => {
                if let Some(body) = event
                    .body
                    .and_then(|v| serde_json::from_value::<OutputEventBody>(v).ok())
                    && let Some(category) = OutputCategory::from_dap(body.category.as_deref())
                {
                    self.output.push(category, body.output);
                }
            }
            "breakpoint" => {
                if let Some(body) = event
                    .body
                    .and_then(|v| serde_json::from_value::<BreakpointEventBody>(v).ok())
                {
                    self.breakpoints.apply_event(&body);
                }
            }
            "exited" => {
                if let Some(body) = event
                    .body
                    .and_then(|v| serde_json::from_value::<ExitedEventBody>(v).ok())
                {
                    self.exit_code = Some(body.exit_code);
                }
            }
            "terminated" => {
                self.mark_terminated(None, emit_async_notification);
            }
            other => {
                debug!(event = other, "ignoring unhandled DAP event");
            }
        }
    }

    fn mark_terminated(&mut self, reason: Option<String>, emit_async_notification: bool) {
        let already_terminated = self.state == DebugSessionState::Terminated;
        self.invalidate_snapshot();
        self.state = DebugSessionState::Terminated;
        if reason.is_some() && self.termination_reason.is_none() {
            self.termination_reason = reason;
        }
        if !already_terminated && emit_async_notification && self.notify_on_async_stop {
            self.notify_on_async_stop = false;
            let _ =
                self.notifications
                    .try_send(DebugNotification::Terminated(TerminationSummary {
                        exit_code: self.exit_code,
                        reason: self.termination_reason.clone(),
                    }));
        }
    }

    async fn build_stopped_snapshot(&self, event: &StoppedEventBody) -> StoppedSnapshot {
        let threads = self
            .client
            .request_empty::<ThreadsResponseBody>("threads")
            .await
            .map_or_default(|body| body.threads);

        let stack_thread_id = event
            .thread_id
            .or_else(|| threads.first().map(|thread| thread.id))
            .or(self.last_thread_id);

        let (raw_frames, total_frames) = if let Some(thread_id) = stack_thread_id {
            self.client
                .request::<_, StackTraceResponseBody>(
                    "stackTrace",
                    &StackTraceArguments {
                        thread_id,
                        start_frame: Some(0),
                        levels: Some(DEFAULT_STACK_LEVELS),
                    },
                )
                .await
                .map_or_else(
                    |_| (Vec::new(), None),
                    |body| (body.stack_frames, body.total_frames),
                )
        } else {
            (Vec::new(), None)
        };

        let source_snippet = if let Some(top) = raw_frames.first()
            && let Some(path) = top.source.as_ref().and_then(|s| s.path.as_deref())
        {
            read_source_snippet(path, top.line).await
        } else {
            None
        };

        let hit_breakpoint_ids = self
            .breakpoints
            .resolve_client_ids(&event.hit_breakpoint_ids);

        StoppedSnapshot::build(
            event,
            &threads,
            stack_thread_id,
            &raw_frames,
            total_frames,
            hit_breakpoint_ids,
            source_snippet,
        )
    }

    async fn add_source_breakpoint(
        &mut self,
        spec: SourceBreakpointSpec,
    ) -> Result<ManagedBreakpoint, DapError> {
        let (id, sync) = self.breakpoints.add_source(spec);
        if let Err(err) = self.apply_breakpoint_sync(sync).await {
            self.breakpoints.rollback_add(id);
            return Err(err);
        }
        self.breakpoints
            .get(id)
            .cloned()
            .ok_or(DapError::BreakpointNotFound(id))
    }

    async fn add_function_breakpoint(
        &mut self,
        spec: FunctionBreakpointSpec,
    ) -> Result<ManagedBreakpoint, DapError> {
        if !self.capabilities.supports_function_breakpoints {
            return Err(DapError::Unsupported("function breakpoints"));
        }
        let (id, sync) = self.breakpoints.add_function(spec);
        if let Err(err) = self.apply_breakpoint_sync(sync).await {
            self.breakpoints.rollback_add(id);
            return Err(err);
        }
        self.breakpoints
            .get(id)
            .cloned()
            .ok_or(DapError::BreakpointNotFound(id))
    }

    async fn remove_breakpoint(&mut self, id: u32) -> Result<ManagedBreakpoint, DapError> {
        let (removed, sync) = self.breakpoints.remove(id)?;
        if let Err(err) = self.apply_breakpoint_sync(sync).await {
            self.breakpoints.restore(removed);
            return Err(err);
        }
        Ok(removed)
    }

    async fn apply_breakpoint_sync(&mut self, sync: BreakpointSync) -> Result<(), DapError> {
        match sync {
            BreakpointSync::Source { key, args } => {
                let body: SetBreakpointsResponseBody =
                    self.client.request("setBreakpoints", &args).await?;
                self.breakpoints
                    .apply_source_response(&key, &body.breakpoints);
            }
            BreakpointSync::Function { args } => {
                let body: SetBreakpointsResponseBody =
                    self.client.request("setFunctionBreakpoints", &args).await?;
                self.breakpoints.apply_function_response(&body.breakpoints);
            }
        }
        Ok(())
    }

    async fn fetch_threads(&mut self) -> Result<Vec<ThreadSummary>, DapError> {
        if self.state == DebugSessionState::Terminated {
            return Err(DapError::SessionEnded {
                state: self.state.as_str().to_owned(),
            });
        }
        let body: ThreadsResponseBody = self.client.request_empty("threads").await?;
        let active_id = self
            .snapshot
            .as_ref()
            .and_then(|s| s.summary.thread_id)
            .or(self.last_thread_id);
        let threads: Vec<ThreadSummary> = body
            .threads
            .into_iter()
            .map(|thread| ThreadSummary {
                id: thread.id,
                name: thread.name,
                active: active_id == Some(thread.id),
            })
            .collect();
        if let Some(snapshot) = self.snapshot.as_mut() {
            snapshot.threads.clone_from(&threads);
        }
        Ok(threads)
    }

    async fn fetch_stack(
        &mut self,
        thread_id: Option<i64>,
        limit: Option<usize>,
    ) -> Result<StackSnapshot, DapError> {
        let snapshot = self.require_stopped_mut()?;
        let target_thread = thread_id.or(snapshot.stack_thread_id);
        if let Some(tid) = target_thread
            && snapshot.stack_thread_id != Some(tid)
        {
            let levels = limit.unwrap_or(DEFAULT_STACK_LEVELS);
            let body: StackTraceResponseBody = self
                .client
                .request(
                    "stackTrace",
                    &StackTraceArguments {
                        thread_id: tid,
                        start_frame: Some(0),
                        levels: Some(levels),
                    },
                )
                .await?;
            let snapshot = self.require_stopped_mut()?;
            snapshot.replace_frames(tid, &body.stack_frames, body.total_frames);
        }
        Ok(self.require_stopped()?.stack_snapshot(limit))
    }

    async fn fetch_variables(
        &mut self,
        frame: Option<usize>,
        expand: Option<usize>,
        limit: Option<usize>,
    ) -> Result<VariablesView, DapError> {
        let count = Some(limit.unwrap_or(DEFAULT_VARIABLES_LIMIT));

        if let Some(handle) = expand {
            return self.expand_variable_handle(handle, count).await;
        }

        let frame_idx = frame.unwrap_or(0);
        let (frame_id, frame_name) = {
            let snapshot = self.require_stopped()?;
            let (frame_id, name) = snapshot.resolve_frame(frame_idx)?;
            (frame_id, name.to_owned())
        };

        let scopes_body: ScopesResponseBody = self
            .client
            .request("scopes", &ScopesArguments { frame_id })
            .await?;

        let mut scopes = Vec::with_capacity(scopes_body.scopes.len());
        for scope in scopes_body.scopes {
            if scope.variables_reference <= 0 {
                continue;
            }
            if scope.expensive {
                // Expose expensive scopes (such as globals/registers) lazily as a
                // single expandable entry rather than fetching thousands of entries eagerly.
                let snapshot = self.require_stopped_mut()?;
                let handle =
                    snapshot.register_variable_ref(scope.variables_reference, scope.name.clone());
                scopes.push(ScopeVariables {
                    name: scope.name,
                    variables: vec![VariableEntry {
                        name: "(lazy scope)".to_owned(),
                        value: "expand to inspect".to_owned(),
                        value_type: None,
                        expand: handle,
                    }],
                });
                continue;
            }
            let vars_body: VariablesResponseBody = self
                .client
                .request(
                    "variables",
                    &VariablesArguments {
                        variables_reference: scope.variables_reference,
                        start: Some(0),
                        count,
                    },
                )
                .await?;
            let snapshot = self.require_stopped_mut()?;
            let variables = vars_body
                .variables
                .into_iter()
                .map(|var| {
                    let expand_handle =
                        snapshot.register_variable_ref(var.variables_reference, var.name.clone());
                    VariableEntry {
                        name: var.name,
                        value: var.value,
                        value_type: var.r#type,
                        expand: expand_handle,
                    }
                })
                .collect();
            scopes.push(ScopeVariables {
                name: scope.name,
                variables,
            });
        }

        Ok(VariablesView::Frame {
            frame: frame_idx,
            frame_name,
            scopes,
        })
    }

    async fn expand_variable_handle(
        &mut self,
        handle: usize,
        count: Option<usize>,
    ) -> Result<VariablesView, DapError> {
        let (dap_ref, label) = {
            let snapshot = self.require_stopped()?;
            let (dap_ref, label) = snapshot.resolve_variable_handle(handle)?;
            (dap_ref, label.to_owned())
        };
        let body: VariablesResponseBody = self
            .client
            .request(
                "variables",
                &VariablesArguments {
                    variables_reference: dap_ref,
                    start: Some(0),
                    count,
                },
            )
            .await?;
        let snapshot = self.require_stopped_mut()?;
        let variables = body
            .variables
            .into_iter()
            .map(|var| {
                let expand_handle = snapshot.register_variable_ref(
                    var.variables_reference,
                    format!("{label}.{}", var.name),
                );
                VariableEntry {
                    name: var.name,
                    value: var.value,
                    value_type: var.r#type,
                    expand: expand_handle,
                }
            })
            .collect();
        Ok(VariablesView::Expanded {
            handle,
            label,
            variables,
        })
    }

    async fn eval_expression(
        &mut self,
        expression: String,
        frame: Option<usize>,
        context: EvalContext,
    ) -> Result<EvalResult, DapError> {
        let frame_idx = frame.unwrap_or(0);
        let frame_id = {
            let snapshot = self.require_stopped()?;
            let (id, _) = snapshot.resolve_frame(frame_idx)?;
            id
        };

        let body: EvaluateResponseBody = self
            .client
            .request(
                "evaluate",
                &EvaluateArguments {
                    expression: expression.clone(),
                    frame_id: Some(frame_id),
                    context: Some(context.as_dap_str().to_owned()),
                },
            )
            .await?;

        let snapshot = self.require_stopped_mut()?;
        let expand = snapshot.register_variable_ref(body.variables_reference, expression.clone());

        Ok(EvalResult {
            expression,
            value: body.result,
            value_type: body.r#type,
            expand,
        })
    }

    async fn resolve_thread_id(&mut self, requested: Option<i64>) -> Result<i64, DapError> {
        if let Some(tid) = requested {
            return Ok(tid);
        }
        if let Some(tid) = self
            .snapshot
            .as_ref()
            .and_then(|s| s.summary.thread_id)
            .or(self.last_thread_id)
        {
            return Ok(tid);
        }
        let body: ThreadsResponseBody = self.client.request_empty("threads").await?;
        let first = body.threads.first().map_or(1, |thread| thread.id);
        self.last_thread_id = Some(first);
        Ok(first)
    }

    fn require_stopped(&self) -> Result<&StoppedSnapshot, DapError> {
        if self.state != DebugSessionState::Stopped {
            return Err(DapError::NotStopped {
                state: self.state.as_str().to_owned(),
            });
        }
        self.snapshot.as_ref().ok_or_else(|| DapError::NotStopped {
            state: self.state.as_str().to_owned(),
        })
    }

    fn require_stopped_mut(&mut self) -> Result<&mut StoppedSnapshot, DapError> {
        if self.state != DebugSessionState::Stopped {
            return Err(DapError::NotStopped {
                state: self.state.as_str().to_owned(),
            });
        }
        let state = self.state.as_str().to_owned();
        self.snapshot.as_mut().ok_or(DapError::NotStopped { state })
    }

    fn invalidate_snapshot(&mut self) {
        self.snapshot = None;
    }

    fn build_status(&self) -> DebugStatus {
        DebugStatus {
            state: self.state,
            adapter: Some(self.config.adapter_id.clone()),
            target_pid: self.config.target_pid,
            target: self.config.target.clone(),
            breakpoint_count: self.breakpoints.len(),
            stop_location: self
                .snapshot
                .as_ref()
                .and_then(|snap| snap.summary.top_frame.clone()),
            stop_reason: self
                .snapshot
                .as_ref()
                .map(|snap| snap.summary.reason.clone()),
            exit_code: self.exit_code,
        }
    }
}
