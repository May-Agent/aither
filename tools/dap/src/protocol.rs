//! Debug Adapter Protocol (DAP) wire message and payload types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Top-level DAP protocol message transmitted over a framed channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ProtocolMessage {
    /// Client- or adapter-initiated request expecting a matching response.
    Request(DapRequest),
    /// Response correlated to an earlier request via `request_seq`.
    Response(DapResponse),
    /// Unsolicited event emitted by the debug adapter.
    Event(DapEvent),
}

/// A DAP request frame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DapRequest {
    /// Monotonic sequence number assigned by the sender.
    pub seq: i64,
    /// DAP command name (for example `"initialize"`, `"setBreakpoints"`).
    pub command: String,
    /// Optional command arguments payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

/// A DAP response frame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DapResponse {
    /// Sequence number assigned by the responder.
    pub seq: i64,
    /// Sequence number of the request this response answers.
    pub request_seq: i64,
    /// Whether the request succeeded.
    pub success: bool,
    /// Command name echoed from the request.
    pub command: String,
    /// Optional short error identifier or message when `success` is `false`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Optional structured response body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<Value>,
}

/// A DAP event frame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DapEvent {
    /// Sequence number assigned by the adapter.
    pub seq: i64,
    /// Event name (for example `"stopped"`, `"output"`, `"terminated"`).
    pub event: String,
    /// Optional structured event body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<Value>,
}

/// Arguments for the DAP `initialize` request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(
    clippy::struct_excessive_bools,
    reason = "DAP initialize schema defines independent boolean capability flags"
)]
pub struct InitializeRequestArguments {
    /// Client identifier sent to the adapter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    /// Human-readable client name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_name: Option<String>,
    /// Debug adapter type identifier.
    pub adapter_id: String,
    /// Whether source lines are 1-based (`true`).
    pub lines_start_at1: bool,
    /// Whether source columns are 1-based (`true`).
    pub columns_start_at1: bool,
    /// Path format expected by the client (`"path"`).
    pub path_format: String,
    /// Whether the client handles `runInTerminal` reverse requests.
    pub supports_run_in_terminal_request: bool,
    /// Whether the client supports variable type metadata.
    pub supports_variable_type: bool,
    /// Whether the client supports variable paging.
    pub supports_variable_paging: bool,
}

impl InitializeRequestArguments {
    /// Constructs standard client initialize arguments for `adapter_id`.
    #[must_use]
    pub fn for_adapter(adapter_id: impl Into<String>) -> Self {
        Self {
            client_id: Some("aither-dap".to_owned()),
            client_name: Some("aither-dap".to_owned()),
            adapter_id: adapter_id.into(),
            lines_start_at1: true,
            columns_start_at1: true,
            path_format: "path".to_owned(),
            supports_run_in_terminal_request: false,
            supports_variable_type: true,
            supports_variable_paging: true,
        }
    }
}

/// Adapter capabilities reported in the `initialize` response body.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(
    clippy::struct_excessive_bools,
    reason = "DAP Capabilities schema defines independent boolean feature flags"
)]
pub struct Capabilities {
    /// Adapter supports `configurationDone` request.
    #[serde(default)]
    pub supports_configuration_done_request: bool,
    /// Adapter supports `setFunctionBreakpoints` request.
    #[serde(default)]
    pub supports_function_breakpoints: bool,
    /// Adapter supports conditional breakpoints.
    #[serde(default)]
    pub supports_conditional_breakpoints: bool,
    /// Adapter supports hit-count conditional breakpoints.
    #[serde(default)]
    pub supports_hit_conditional_breakpoints: bool,
    /// Adapter supports logpoints (`logMessage`).
    #[serde(default)]
    pub supports_log_points: bool,
    /// Adapter supports the `terminate` request.
    #[serde(default)]
    pub supports_terminate_request: bool,
    /// Adapter supports `evaluate` in hover context.
    #[serde(default)]
    pub supports_evaluate_for_hovers: bool,
    /// Adapter supports `terminateDebuggee` on `disconnect`.
    #[serde(default)]
    pub support_terminate_debuggee: bool,
}

/// A DAP source file descriptor.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Source {
    /// Short display name for the source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Filesystem path to the source file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Non-zero adapter handle for generated/in-memory sources.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_reference: Option<i64>,
}

/// Source breakpoint specification sent in `setBreakpoints`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceBreakpoint {
    /// Requested 1-based line number.
    pub line: u32,
    /// Optional 1-based column number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column: Option<u32>,
    /// Optional conditional expression.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    /// Optional hit count condition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hit_condition: Option<String>,
    /// Optional logpoint template; when set the adapter logs instead of stopping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_message: Option<String>,
}

/// Arguments for DAP `setBreakpoints`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetBreakpointsArguments {
    /// Source file whose breakpoint list is being replaced.
    pub source: Source,
    /// Complete list of breakpoints for `source`.
    pub breakpoints: Vec<SourceBreakpoint>,
}

/// Response body for DAP `setBreakpoints` and `setFunctionBreakpoints`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetBreakpointsResponseBody {
    /// Adapter-resolved breakpoint statuses in request order.
    #[serde(default)]
    pub breakpoints: Vec<Breakpoint>,
}

/// Function breakpoint specification sent in `setFunctionBreakpoints`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionBreakpoint {
    /// Function name.
    pub name: String,
    /// Optional conditional expression.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    /// Optional hit count condition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hit_condition: Option<String>,
}

/// Arguments for DAP `setFunctionBreakpoints`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetFunctionBreakpointsArguments {
    /// Complete list of function breakpoints.
    pub breakpoints: Vec<FunctionBreakpoint>,
}

/// Adapter-reported breakpoint state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Breakpoint {
    /// Adapter-assigned breakpoint identifier, if provided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<i64>,
    /// Whether the breakpoint was verified and bound by the adapter.
    #[serde(default)]
    pub verified: bool,
    /// Optional diagnostic message (for example why verification failed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Resolved source location.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
    /// Resolved 1-based line number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    /// Resolved 1-based column number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column: Option<u32>,
}

/// Body of a DAP `breakpoint` event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BreakpointEventBody {
    /// Reason for update (`"changed"`, `"new"`, `"removed"`).
    pub reason: String,
    /// Updated breakpoint state.
    pub breakpoint: Breakpoint,
}

/// Body of a DAP `stopped` event.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoppedEventBody {
    /// Stop reason (for example `"breakpoint"`, `"step"`, `"exception"`, `"pause"`, `"entry"`).
    pub reason: String,
    /// Optional full description of the stop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Thread that triggered the stop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<i64>,
    /// Additional stop detail text (for example exception message).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Whether all threads stopped.
    #[serde(default)]
    pub all_threads_stopped: Option<bool>,
    /// Adapter breakpoint IDs hit at this stop.
    #[serde(default)]
    pub hit_breakpoint_ids: Vec<i64>,
}

/// Body of a DAP `continued` event.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContinuedEventBody {
    /// Thread that resumed.
    #[serde(default)]
    pub thread_id: i64,
    /// Whether all threads resumed.
    #[serde(default)]
    pub all_threads_continued: Option<bool>,
}

/// Body of a DAP `exited` event.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExitedEventBody {
    /// Target exit status code.
    pub exit_code: i64,
}

/// Body of a DAP `terminated` event.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminatedEventBody {
    /// Optional restart payload from the adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restart: Option<Value>,
}

/// Body of a DAP `output` event.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OutputEventBody {
    /// Output stream category (`"console"`, `"stdout"`, `"stderr"`, `"telemetry"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Output text payload.
    pub output: String,
}

/// A DAP thread entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Thread {
    /// Adapter thread identifier.
    pub id: i64,
    /// Thread display name.
    pub name: String,
}

/// Response body for DAP `threads`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadsResponseBody {
    /// Active threads reported by the adapter.
    #[serde(default)]
    pub threads: Vec<Thread>,
}

/// Arguments for DAP `stackTrace`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StackTraceArguments {
    /// Thread whose call stack is requested.
    pub thread_id: i64,
    /// Index of the first frame to return.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_frame: Option<usize>,
    /// Maximum number of frames to return.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub levels: Option<usize>,
}

/// A DAP stack frame entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StackFrame {
    /// Transient adapter frame identifier valid until execution resumes.
    pub id: i64,
    /// Frame function or label name.
    pub name: String,
    /// Optional source file descriptor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
    /// 1-based line number within `source` (0 if unknown).
    #[serde(default)]
    pub line: u32,
    /// 1-based column number within `source` (0 if unknown).
    #[serde(default)]
    pub column: u32,
}

/// Response body for DAP `stackTrace`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StackTraceResponseBody {
    /// Stack frames ordered innermost-first.
    #[serde(default)]
    pub stack_frames: Vec<StackFrame>,
    /// Optional total frame count reported by the adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_frames: Option<usize>,
}

/// Arguments for DAP `scopes`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopesArguments {
    /// Adapter stack frame ID.
    pub frame_id: i64,
}

/// A variable scope in a stack frame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Scope {
    /// Scope display name (for example `"Locals"`, `"Arguments"`, `"Globals"`).
    pub name: String,
    /// Transient `variablesReference` for fetching variables in this scope.
    pub variables_reference: i64,
    /// Whether fetching variables in this scope may be expensive.
    #[serde(default)]
    pub expensive: bool,
}

/// Response body for DAP `scopes`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopesResponseBody {
    /// Scopes available in the requested frame.
    #[serde(default)]
    pub scopes: Vec<Scope>,
}

/// Arguments for DAP `variables`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VariablesArguments {
    /// Transient adapter variable reference handle.
    pub variables_reference: i64,
    /// Optional start offset for paged variable requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<usize>,
    /// Optional maximum number of child variables to return.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<usize>,
}

/// A DAP variable entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Variable {
    /// Variable name.
    pub name: String,
    /// Formatted value string.
    pub value: String,
    /// Optional type name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    /// Non-zero when the variable has expandable child entries.
    #[serde(default)]
    pub variables_reference: i64,
    /// Optional count of named children.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub named_variables: Option<usize>,
    /// Optional count of indexed children.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indexed_variables: Option<usize>,
}

/// Response body for DAP `variables`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VariablesResponseBody {
    /// Variables in the requested scope or container.
    #[serde(default)]
    pub variables: Vec<Variable>,
}

/// Arguments for DAP `evaluate`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluateArguments {
    /// Expression to evaluate.
    pub expression: String,
    /// Optional adapter frame ID for evaluation context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_id: Option<i64>,
    /// Evaluation context (`"watch"`, `"repl"`, `"hover"`, `"variables"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

/// Response body for DAP `evaluate`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluateResponseBody {
    /// Formatted evaluation result.
    pub result: String,
    /// Optional type of the evaluated result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    /// Non-zero when the evaluated value has expandable child entries.
    #[serde(default)]
    pub variables_reference: i64,
}

/// Arguments for DAP `continue`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContinueArguments {
    /// Thread to resume.
    pub thread_id: i64,
}

/// Arguments for DAP `next`, `stepIn`, `stepOut`, and `pause`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadActionArguments {
    /// Target thread identifier.
    pub thread_id: i64,
}

/// Arguments for DAP `disconnect`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DisconnectArguments {
    /// Whether the adapter should terminate the debuggee on disconnect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminate_debuggee: Option<bool>,
}

/// Arguments for DAP `terminate`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminateArguments {
    /// Whether the adapter should restart the session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restart: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_protocol_messages() {
        let req = ProtocolMessage::Request(DapRequest {
            seq: 1,
            command: "initialize".to_owned(),
            arguments: Some(
                serde_json::to_value(InitializeRequestArguments::for_adapter("lldb-dap")).unwrap(),
            ),
        });
        let json = serde_json::to_string(&req).unwrap();
        let parsed: ProtocolMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, req);

        let evt = ProtocolMessage::Event(DapEvent {
            seq: 2,
            event: "stopped".to_owned(),
            body: Some(serde_json::json!({
                "reason": "breakpoint",
                "threadId": 1,
                "allThreadsStopped": true,
                "hitBreakpointIds": [7]
            })),
        });
        let json = serde_json::to_string(&evt).unwrap();
        let parsed: ProtocolMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, evt);
    }
}
