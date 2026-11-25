//! Tool schema, broker types, notifications, and Askama-rendered outputs for `debug`.

use std::borrow::Cow;
use std::fmt;

use aither_core::llm::{IntoToolResult, Tool, ToolContext, ToolResult};
use askama::Template;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::breakpoints::{BreakpointTarget, ManagedBreakpoint};
use crate::output::{OutputCategory, OutputSnapshot};
use crate::snapshot::{
    EvalResult, IndexedFrame, StackSnapshot, StopSummary, ThreadSummary, VariablesView,
};
use crate::tool_request::{
    ToolRequestBroker, ToolRequestError, ToolRequestQueue, channel as tool_request_channel,
};

/// Mode used when connecting to an existing TCP DAP server.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum ConnectMode {
    /// Attach to a target already managed by the remote DAP server.
    #[default]
    Attach,
    /// Send a DAP `launch` request over the TCP connection.
    Launch,
}

/// Evaluation context passed to DAP `evaluate`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum EvalContext {
    /// Interactive REPL evaluation (default).
    #[default]
    Repl,
    /// Watch expression evaluation.
    Watch,
    /// Hover expression evaluation.
    Hover,
    /// Variables view evaluation.
    Variables,
}

impl EvalContext {
    /// Returns the DAP protocol context string.
    #[must_use]
    pub const fn as_dap_str(self) -> &'static str {
        match self {
            Self::Repl => "repl",
            Self::Watch => "watch",
            Self::Hover => "hover",
            Self::Variables => "variables",
        }
    }
}

/// Drive a structured Debug Adapter Protocol (DAP) debugging session.
///
/// Use `launch`, `attach`, or `connect` to start a session, `break` / `unbreak`
/// to manage breakpoints, `continue` / `next` / `step-in` / `step-out` / `pause`
/// to control execution, and `threads` / `stack` / `vars` / `eval` / `output`
/// to inspect state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "command", rename_all = "kebab-case")]
pub enum DebugArgs {
    /// Show current debug session status, target state, and breakpoint count.
    Status,
    /// Launch a program under a debug adapter.
    Launch {
        /// Path to the executable or script to debug.
        program: String,
        /// Optional adapter override (auto-selected when omitted).
        #[serde(default)]
        adapter: Option<String>,
        /// Working directory for the target process.
        #[serde(default)]
        cwd: Option<String>,
        /// Environment overrides as `KEY=VALUE` entries.
        #[serde(default)]
        env: Vec<String>,
        /// Pause immediately at program entry.
        #[serde(default)]
        stop_on_entry: bool,
        /// Bounded wait timeout in milliseconds before returning while running.
        #[serde(default)]
        timeout_ms: Option<u64>,
        /// Program arguments passed to the target.
        #[serde(default)]
        #[schemars(extend("x-positional" = true))]
        args: Vec<String>,
    },
    /// Attach to a running process by PID.
    Attach {
        /// Target process ID.
        pid: u32,
        /// Optional adapter override (auto-selected when omitted).
        #[serde(default)]
        adapter: Option<String>,
        /// Bounded wait timeout in milliseconds.
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    /// Connect to a listening TCP DAP server (`host:port`).
    Connect {
        /// TCP endpoint (`host:port`, for example `127.0.0.1:5678`).
        endpoint: String,
        /// Optional adapter hint.
        #[serde(default)]
        adapter: Option<String>,
        /// Whether to send `attach` (default) or `launch` after connecting.
        #[serde(default)]
        mode: ConnectMode,
        /// Optional program path when `mode` is `launch`.
        #[serde(default)]
        program: Option<String>,
        /// Optional working directory when `mode` is `launch`.
        #[serde(default)]
        cwd: Option<String>,
        /// Pause at entry when `mode` is `launch`.
        #[serde(default)]
        stop_on_entry: bool,
        /// Bounded wait timeout in milliseconds.
        #[serde(default)]
        timeout_ms: Option<u64>,
        /// Optional program arguments when `mode` is `launch`.
        #[serde(default)]
        #[schemars(extend("x-positional" = true))]
        args: Vec<String>,
    },
    /// Detach from the target process without terminating it.
    Detach,
    /// Terminate the target process and close the debug session.
    Terminate,
    /// Set a source breakpoint at `file` and `line`.
    Break {
        /// Source file path.
        file: String,
        /// 1-based line number.
        line: u32,
        /// Optional 1-based column number.
        #[serde(default)]
        column: Option<u32>,
        /// Optional conditional expression.
        #[serde(default)]
        condition: Option<String>,
        /// Optional hit-count condition.
        #[serde(default)]
        hit_condition: Option<String>,
        /// Optional logpoint message (logs instead of stopping).
        #[serde(default)]
        log_message: Option<String>,
    },
    /// Set a function breakpoint by symbol or function name.
    BreakFunction {
        /// Function name.
        name: String,
        /// Optional conditional expression.
        #[serde(default)]
        condition: Option<String>,
        /// Optional hit-count condition.
        #[serde(default)]
        hit_condition: Option<String>,
    },
    /// List active breakpoints with client IDs and verification status.
    Breaks,
    /// Remove a breakpoint by its client-assigned ID.
    Unbreak {
        /// Client-assigned breakpoint ID from `break` or `breaks`.
        id: u32,
    },
    /// Resume execution and wait up to `timeout_ms` for the next stop or exit.
    Continue {
        /// Optional thread ID to resume (defaults to the stopped thread).
        #[serde(default)]
        thread_id: Option<i64>,
        /// Bounded wait timeout in milliseconds before returning while running.
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    /// Wait up to `timeout_ms` for a running target to stop or exit.
    Wait {
        /// Bounded wait timeout in milliseconds.
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    /// Pause a running target.
    Pause {
        /// Optional thread ID to pause.
        #[serde(default)]
        thread_id: Option<i64>,
        /// Bounded wait timeout in milliseconds.
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    /// Step over the current source line.
    Next {
        /// Optional thread ID to step.
        #[serde(default)]
        thread_id: Option<i64>,
        /// Bounded wait timeout in milliseconds.
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    /// Step into a call on the current source line.
    StepIn {
        /// Optional thread ID to step.
        #[serde(default)]
        thread_id: Option<i64>,
        /// Bounded wait timeout in milliseconds.
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    /// Step out of the current stack frame.
    StepOut {
        /// Optional thread ID to step.
        #[serde(default)]
        thread_id: Option<i64>,
        /// Bounded wait timeout in milliseconds.
        #[serde(default)]
        timeout_ms: Option<u64>,
    },
    /// List threads in the debug target.
    Threads,
    /// Show the call stack with small integer frame indices (`0, 1, 2...`).
    Stack {
        /// Optional thread ID (defaults to the stopped thread).
        #[serde(default)]
        thread_id: Option<i64>,
        /// Maximum number of frames to return.
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Inspect variables in a frame or lazily expand one level of a variable handle.
    Vars {
        /// 0-based frame index from `stack` (defaults to `0` when `expand` is omitted).
        #[serde(default)]
        frame: Option<usize>,
        /// 1-based variable handle index to expand one level.
        #[serde(default)]
        expand: Option<usize>,
        /// Optional maximum number of child variables to return.
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Evaluate an expression in a stack frame.
    Eval {
        /// Expression to evaluate.
        expression: String,
        /// 0-based frame index from `stack` (defaults to `0`).
        #[serde(default)]
        frame: Option<usize>,
        /// Evaluation context (`repl`, `watch`, `hover`, `variables`).
        #[serde(default)]
        context: Option<EvalContext>,
    },
    /// Read recent captured target output (`stdout`, `stderr`, `console`).
    Output {
        /// Output stream category (`all`, `stdout`, `stderr`, `console`).
        #[serde(default)]
        category: Option<OutputCategory>,
        /// Maximum number of recent entries to return.
        #[serde(default)]
        tail: Option<usize>,
        /// Clear buffered output after reading.
        #[serde(default)]
        clear: bool,
    },
}

/// Request sent from [`DebugTool`] to the host broker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugRequest {
    /// Logical session ID owning the debug session.
    pub session_id: String,
    /// Debug command arguments.
    pub args: DebugArgs,
}

/// Lifecycle state of a debug session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DebugSessionState {
    /// No active debug session exists.
    Inactive,
    /// Session is initializing and launching/attaching.
    Starting,
    /// Target is currently executing.
    Running,
    /// Target is paused at a stop event.
    Stopped,
    /// Target exited or terminated.
    Terminated,
}

impl DebugSessionState {
    /// Short lowercase label for templates and errors.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Inactive => "inactive",
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Stopped => "stopped",
            Self::Terminated => "terminated",
        }
    }
}

/// High-level summary of a debug session returned by `status` and host inventory queries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugStatus {
    /// Current lifecycle state.
    pub state: DebugSessionState,
    /// Debug adapter identifier (for example `lldb-dap`, `gdb`, `dlv`, `debugpy`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    /// Target OS process ID when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_pid: Option<u32>,
    /// Target program or TCP endpoint description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Number of currently registered breakpoints.
    pub breakpoint_count: usize,
    /// Current top stop frame when stopped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_location: Option<IndexedFrame>,
    /// Last stop reason when stopped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// Exit code if the target exited.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
}

/// Summary of a target termination or exit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminationSummary {
    /// Target process exit code, if reported before termination.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    /// Optional human-readable termination reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Structured asynchronous notification emitted after a synchronous execution
/// wait timed out while the target remained running.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum DebugNotification {
    /// Running target later hit a breakpoint, exception, or pause.
    Stopped(Box<StopSummary>),
    /// Running target later exited or terminated.
    Terminated(TerminationSummary),
}

impl DebugNotification {
    /// Renders this notification into concise natural language via a typed Askama template.
    ///
    /// # Panics
    ///
    /// Panics only if the compiled-in Askama notification template fails to render.
    #[must_use]
    pub fn render(&self) -> String {
        NotificationTemplate { notification: self }
            .render()
            .expect("debug notification template must render")
    }
}

/// Structured result produced by executing a [`DebugArgs`] command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum DebugResult {
    /// Response to `status`.
    Status(DebugStatus),
    /// Target stopped synchronously during `launch`, `attach`, `connect`, `continue`,
    /// `wait`, `pause`, or stepping.
    Stopped(Box<StopSummary>),
    /// Target is running and did not stop before the bounded wait elapsed; an
    /// asynchronous notification will be delivered if it stops or exits.
    Running {
        /// Summary note (for example `"execution resumed"` or `"target is running"`).
        note: String,
    },
    /// Breakpoint added or updated.
    BreakpointSet(ManagedBreakpoint),
    /// Active breakpoint list.
    Breakpoints(Vec<ManagedBreakpoint>),
    /// Breakpoint removed.
    BreakpointRemoved(ManagedBreakpoint),
    /// Active threads list.
    Threads(Vec<ThreadSummary>),
    /// Indexed call stack snapshot.
    Stack(StackSnapshot),
    /// Frame scopes or lazily expanded variable children.
    Variables(VariablesView),
    /// Expression evaluation result.
    Eval(EvalResult),
    /// Captured target output snapshot.
    Output(OutputSnapshot),
    /// Detached cleanly from target.
    Detached {
        /// Optional target PID left running.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target_pid: Option<u32>,
    },
    /// Target terminated or exited.
    Terminated(TerminationSummary),
}

impl DebugResult {
    /// Renders this result into concise natural language via a typed Askama template.
    ///
    /// # Panics
    ///
    /// Panics only if the compiled-in Askama result template fails to render.
    #[must_use]
    pub fn render(&self) -> String {
        ResultTemplate { result: self }
            .render()
            .expect("debug result template must render")
    }
}

/// Error returned by [`DebugTool`] or the host debug broker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
pub enum DebugToolError {
    /// Debug commands are unavailable in this runtime.
    #[error("debug command is unavailable in this runtime")]
    UnavailableInRuntime,
    /// Debug command failed.
    #[error("{message}")]
    RequestFailed {
        /// Human-readable error message.
        message: String,
    },
}

impl DebugToolError {
    /// Creates a request-failed error with `message`.
    #[must_use]
    pub fn request_failed(message: impl Into<String>) -> Self {
        Self::RequestFailed {
            message: message.into(),
        }
    }
}

/// Tool output wrapper converting `Result<DebugResult, DebugToolError>` into a
/// rendered [`ToolResult`].
#[derive(Debug, Clone)]
pub struct DebugToolOutput(Result<DebugResult, DebugToolError>);

impl From<Result<DebugResult, DebugToolError>> for DebugToolOutput {
    fn from(value: Result<DebugResult, DebugToolError>) -> Self {
        Self(value)
    }
}

impl IntoToolResult for DebugToolOutput {
    fn into_tool_result(self) -> ToolResult {
        match self.0 {
            Ok(result) => ToolResult::text(result.render()),
            Err(error) => ToolResult::error(error.to_string()),
        }
    }
}

#[derive(Template)]
#[template(path = "result.txt", escape = "none")]
struct ResultTemplate<'a> {
    result: &'a DebugResult,
}

#[derive(Template)]
#[template(path = "notification.txt", escape = "none")]
struct NotificationTemplate<'a> {
    notification: &'a DebugNotification,
}

impl fmt::Display for ManagedBreakpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{} ", self.id)?;
        match &self.target {
            BreakpointTarget::Source {
                file,
                requested_line,
                resolved_line,
                column,
            } => {
                write!(f, "{file}:{requested_line}")?;
                if let Some(actual) = resolved_line
                    && *actual != *requested_line
                {
                    write!(f, " (resolved line {actual})")?;
                }
                if let Some(col) = column {
                    write!(f, ":{col}")?;
                }
            }
            BreakpointTarget::Function {
                name,
                resolved_file,
                resolved_line,
            } => {
                write!(f, "fn {name}")?;
                if let Some(file) = resolved_file {
                    write!(f, " ({file}")?;
                    if let Some(line) = resolved_line {
                        write!(f, ":{line}")?;
                    }
                    f.write_str(")")?;
                }
            }
        }
        if self.verified {
            f.write_str(" [verified]")?;
        } else {
            f.write_str(" [unverified]")?;
        }
        if let Some(cond) = &self.condition {
            write!(f, " if ({cond})")?;
        }
        if let Some(hit) = &self.hit_condition {
            write!(f, " hit ({hit})")?;
        }
        if let Some(log) = &self.log_message {
            write!(f, " log ({log})")?;
        }
        if let Some(msg) = &self.message {
            write!(f, " — {msg}")?;
        }
        Ok(())
    }
}

impl fmt::Display for IndexedFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{} {}", self.index, self.name)?;
        if let Some(file) = &self.file {
            write!(f, " at {file}")?;
            if self.line > 0 {
                write!(f, ":{}", self.line)?;
                if self.column > 0 {
                    write!(f, ":{}", self.column)?;
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for crate::snapshot::VariableEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(ty) = &self.value_type
            && !ty.is_empty()
        {
            write!(f, ": {ty}")?;
        }
        write!(f, " = {}", self.value)?;
        if let Some(handle) = self.expand {
            write!(f, " [expand={handle}]")?;
        }
        Ok(())
    }
}

/// Broker used by [`DebugTool`] to dispatch requests to the host debug service.
pub type DebugBroker = ToolRequestBroker<DebugRequest, Result<DebugResult, DebugToolError>>;

/// Queue consumed by the host debug service.
pub type DebugQueue = ToolRequestQueue<DebugRequest, Result<DebugResult, DebugToolError>>;

/// Creates a debug tool broker/queue channel pair.
#[must_use]
pub fn channel() -> (DebugBroker, DebugQueue) {
    tool_request_channel()
}

/// Agent [`Tool`] implementation for structured DAP debugging.
#[derive(Debug, Clone)]
pub struct DebugTool {
    session_id: String,
    broker: Option<DebugBroker>,
}

impl DebugTool {
    /// Creates a debug tool bound to `session_id` and an optional host `broker`.
    #[must_use]
    pub const fn new(session_id: String, broker: Option<DebugBroker>) -> Self {
        Self { session_id, broker }
    }
}

impl Tool for DebugTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("debug")
    }

    type Arguments = DebugArgs;
    type Res = DebugToolOutput;
    type Error = ToolRequestError;

    async fn call(
        &self,
        args: Self::Arguments,
        _cx: ToolContext,
    ) -> Result<Self::Res, Self::Error> {
        let Some(broker) = self.broker.as_ref() else {
            return Ok(DebugToolOutput::from(Err(
                DebugToolError::UnavailableInRuntime,
            )));
        };
        let response = broker
            .request(DebugRequest {
                session_id: self.session_id.clone(),
                args,
            })
            .await?;
        Ok(DebugToolOutput::from(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_is_tagged_enum_with_terse_descriptions() {
        let schema = schemars::schema_for!(DebugArgs);
        let json = serde_json::to_string(&schema).unwrap();
        assert!(json.contains("\"command\""));
        assert!(json.contains("\"break-function\""));
        assert!(json.contains("\"step-in\""));
        assert!(json.contains("\"step-out\""));
        assert!(!json.contains("lldb-dap"));
        assert!(!json.contains("debugpy"));
    }

    #[test]
    fn renders_results_and_notifications_without_raw_json() {
        let stop = StopSummary {
            reason: "breakpoint".to_owned(),
            description: None,
            text: None,
            thread_id: Some(1),
            hit_breakpoint_ids: vec![2],
            top_frame: Some(IndexedFrame {
                index: 0,
                name: "compute".to_owned(),
                file: Some("src/main.rs".to_owned()),
                line: 42,
                column: 5,
            }),
            source_snippet: None,
        };
        let rendered = DebugResult::Stopped(Box::new(stop.clone())).render();
        assert!(rendered.contains("Stopped (breakpoint)"));
        assert!(rendered.contains("src/main.rs:42:5"));
        assert!(!rendered.contains("\"reason\""));

        let notif = DebugNotification::Stopped(Box::new(stop)).render();
        assert!(notif.contains("Debug target stopped"));
        assert!(notif.contains("breakpoint"));
    }
}
