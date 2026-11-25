//! Debug Adapter Protocol (DAP) client, stateful session manager, and [`DebugTool`].
//!
//! `aither-dap` provides a runtime-agnostic DAP client over stdio and TCP,
//! incremental client-indexed breakpoint management over DAP's replace-all
//! `setBreakpoints` API, stopped-state snapshot indexing for stack frames and
//! lazy variable expansion, bounded target output buffering, asynchronous stop
//! notifications, and a broker-backed [`DebugTool`] rendering concise natural
//! language through Askama templates.
//!
//! # Example
//!
//! ```rust
//! use aither_dap::{DebugArgs, DebugResult, DebugSessionState, DebugStatus, DebugTool, channel};
//! use aither_core::llm::{IntoToolResult, Tool, ToolContext};
//!
//! # async fn demo() {
//! let (broker, queue) = channel();
//! let tool = DebugTool::new("session-1".to_owned(), Some(broker));
//!
//! let handle = tokio::spawn(async move {
//!     if let Some(req) = queue.next().await {
//!         let _ = req.respond(Ok(DebugResult::Status(DebugStatus {
//!             state: DebugSessionState::Inactive,
//!             adapter: None,
//!             target_pid: None,
//!             target: None,
//!             breakpoint_count: 0,
//!             stop_location: None,
//!             stop_reason: None,
//!             exit_code: None,
//!         })));
//!     }
//! });
//!
//! let output = tool.call(DebugArgs::Status, ToolContext::new()).await.unwrap();
//! let rendered = output.into_tool_result().render_for_model();
//! assert!(rendered.contains("inactive"));
//! handle.await.unwrap();
//! # }
//! ```

pub mod breakpoints;
pub mod client;
pub mod error;
pub mod framing;
pub mod output;
pub mod protocol;
pub mod session;
pub mod snapshot;
pub mod tool;
pub mod tool_request;
pub mod transport;

pub use breakpoints::{
    BreakpointStore, BreakpointSync, BreakpointTarget, FunctionBreakpointSpec, ManagedBreakpoint,
    SourceBreakpointSpec,
};
pub use client::DapClient;
pub use error::DapError;
pub use framing::{FrameReader, MAX_FRAME_BYTES, write_message};
pub use output::{OutputBuffer, OutputBufferConfig, OutputCategory, OutputEntry, OutputSnapshot};
pub use protocol::{
    Breakpoint, BreakpointEventBody, Capabilities, ContinueArguments, ContinuedEventBody, DapEvent,
    DapRequest, DapResponse, DisconnectArguments, EvaluateArguments, EvaluateResponseBody,
    ExitedEventBody, FunctionBreakpoint, InitializeRequestArguments, OutputEventBody,
    ProtocolMessage, Scope, ScopesArguments, ScopesResponseBody, SetBreakpointsArguments,
    SetBreakpointsResponseBody, SetFunctionBreakpointsArguments, Source, SourceBreakpoint,
    StackFrame, StackTraceArguments, StackTraceResponseBody, StoppedEventBody, TerminateArguments,
    TerminatedEventBody, Thread, ThreadActionArguments, ThreadsResponseBody, Variable,
    VariablesArguments, VariablesResponseBody,
};
pub use session::{
    DEFAULT_EXECUTION_TIMEOUT, DEFAULT_WAIT_TIMEOUT, DebugSession, DebugSessionConfig,
    ExecutionOutcome,
};
pub use snapshot::{
    EvalResult, IndexedFrame, ScopeVariables, SourceLine, SourceSnippet, StackSnapshot,
    StopSummary, StoppedSnapshot, ThreadSummary, VariableEntry, VariablesView,
};
pub use tool::{
    ConnectMode, DebugArgs, DebugBroker, DebugNotification, DebugQueue, DebugRequest, DebugResult,
    DebugSessionState, DebugStatus, DebugTool, DebugToolError, DebugToolOutput, EvalContext,
    TerminationSummary, channel,
};
pub use tool_request::{ToolRequest, ToolRequestBroker, ToolRequestError, ToolRequestQueue};
pub use transport::{AnyDapTransport, DapTransport, StdioTransport, StreamTransport, TcpTransport};
