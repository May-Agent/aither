//! Integration tests for `aither-dap` against in-process TCP and stdio mock DAP adapters.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_core::llm::{IntoToolResult, Tool, ToolContext};
use aither_dap::{
    AnyDapTransport, BreakpointTarget, DapError, DapEvent, DapRequest, DapResponse, DebugArgs,
    DebugNotification, DebugResult, DebugSession, DebugSessionConfig, DebugSessionState, DebugTool,
    EvalContext, ExecutionOutcome, FrameReader, FunctionBreakpointSpec, OutputBufferConfig,
    OutputCategory, ProtocolMessage, SourceBreakpointSpec, TcpTransport, VariablesView, channel,
    write_message,
};
use async_net::{TcpListener, TcpStream};
use executor_core::tokio::TokioGlobal;
use futures_lite::io::BufReader;
use serde_json::{Value, json};

#[derive(Default)]
struct MockAdapterState {
    set_breakpoints_calls: Vec<Vec<u32>>,
    set_function_breakpoints_calls: Vec<Vec<String>>,
    continue_count: usize,
    stop_delay: Duration,
    terminate_on_second_continue: bool,
    source_path: String,
    reverse_request_rejected: bool,
}

async fn spawn_mock_tcp_adapter(state: Arc<Mutex<MockAdapterState>>) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        serve_mock_dap(stream, state).await;
    });

    addr
}

#[allow(clippy::too_many_lines)]
async fn serve_mock_dap(stream: TcpStream, state: Arc<Mutex<MockAdapterState>>) {
    let mut reader = BufReader::new(stream.clone());
    let mut writer = stream;
    let mut framer = FrameReader::new();
    let mut seq: i64 = 1;

    let mut next_seq = || {
        let current = seq;
        seq += 1;
        current
    };

    while let Ok(Some(message)) = framer.read_message(&mut reader).await {
        match message {
            ProtocolMessage::Response(resp) => {
                if resp.command == "runInTerminal" && !resp.success {
                    state.lock().unwrap().reverse_request_rejected = true;
                }
            }
            ProtocolMessage::Event(_) => {}
            ProtocolMessage::Request(req) => {
                let done = handle_mock_request(&mut writer, &mut next_seq, req, &state).await;
                if done {
                    break;
                }
            }
        }
    }
}

#[allow(clippy::too_many_lines)]
async fn handle_mock_request(
    writer: &mut TcpStream,
    next_seq: &mut impl FnMut() -> i64,
    req: DapRequest,
    state: &Arc<Mutex<MockAdapterState>>,
) -> bool {
    let source_path = state.lock().unwrap().source_path.clone();

    match req.command.as_str() {
        "initialize" => {
            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: true,
                command: req.command,
                message: None,
                body: Some(json!({
                    "supportsConfigurationDoneRequest": true,
                    "supportsFunctionBreakpoints": true,
                    "supportsConditionalBreakpoints": true,
                    "supportsHitConditionalBreakpoints": true,
                    "supportsLogPoints": true,
                    "supportsTerminateRequest": true,
                    "supportTerminateDebuggee": true,
                })),
            });
            write_message(writer, &resp).await.unwrap();
            // Send an unsupported reverse request to verify client rejects it cleanly
            // without deadlocking.
            let rev = ProtocolMessage::Request(DapRequest {
                seq: next_seq(),
                command: "runInTerminal".to_owned(),
                arguments: Some(json!({})),
            });
            write_message(writer, &rev).await.unwrap();
        }
        "launch" | "attach" => {
            // Emit `initialized` BEFORE responding to launch/attach (matching
            // debugpy / lldb-dap / dlv dap behavior).
            let init_evt = ProtocolMessage::Event(DapEvent {
                seq: next_seq(),
                event: "initialized".to_owned(),
                body: None,
            });
            write_message(writer, &init_evt).await.unwrap();

            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: true,
                command: req.command,
                message: None,
                body: None,
            });
            write_message(writer, &resp).await.unwrap();
        }
        "configurationDone" => {
            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: true,
                command: req.command,
                message: None,
                body: None,
            });
            write_message(writer, &resp).await.unwrap();

            // Emit initial stdout/stderr output and entry stop.
            for (cat, text) in [
                ("stdout", "booting target\n"),
                ("stderr", "warning: verbose mode\n"),
                ("telemetry", "ignored telemetry\n"),
            ] {
                let evt = ProtocolMessage::Event(DapEvent {
                    seq: next_seq(),
                    event: "output".to_owned(),
                    body: Some(json!({ "category": cat, "output": text })),
                });
                write_message(writer, &evt).await.unwrap();
            }

            let stopped = ProtocolMessage::Event(DapEvent {
                seq: next_seq(),
                event: "stopped".to_owned(),
                body: Some(json!({
                    "reason": "entry",
                    "threadId": 1,
                    "allThreadsStopped": true
                })),
            });
            write_message(writer, &stopped).await.unwrap();
        }
        "setBreakpoints" => {
            let lines: Vec<u32> = req
                .arguments
                .as_ref()
                .and_then(|a| a.get("breakpoints"))
                .and_then(Value::as_array)
                .map_or_default(|arr| {
                    arr.iter()
                        .filter_map(|bp| bp.get("line").and_then(Value::as_u64))
                        .map(|n| u32::try_from(n).unwrap())
                        .collect()
                });

            state
                .lock()
                .unwrap()
                .set_breakpoints_calls
                .push(lines.clone());

            // Simulate adapter adjusting requested line by +1 for line 10.
            let reported: Vec<Value> = lines
                .iter()
                .enumerate()
                .map(|(i, &line)| {
                    let resolved = if line == 10 { 11 } else { line };
                    json!({
                        "id": 100 + i,
                        "verified": true,
                        "line": resolved,
                    })
                })
                .collect();

            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: true,
                command: req.command,
                message: None,
                body: Some(json!({ "breakpoints": reported })),
            });
            write_message(writer, &resp).await.unwrap();
        }
        "setFunctionBreakpoints" => {
            let names: Vec<String> = req
                .arguments
                .as_ref()
                .and_then(|a| a.get("breakpoints"))
                .and_then(Value::as_array)
                .map_or_default(|arr| {
                    arr.iter()
                        .filter_map(|bp| bp.get("name").and_then(Value::as_str).map(str::to_owned))
                        .collect()
                });
            state
                .lock()
                .unwrap()
                .set_function_breakpoints_calls
                .push(names.clone());

            let reported: Vec<Value> = names
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    json!({
                        "id": 200 + i,
                        "verified": true,
                        "source": { "path": source_path },
                        "line": 3,
                    })
                })
                .collect();

            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: true,
                command: req.command,
                message: None,
                body: Some(json!({ "breakpoints": reported })),
            });
            write_message(writer, &resp).await.unwrap();
        }
        "threads" => {
            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: true,
                command: req.command,
                message: None,
                body: Some(json!({
                    "threads": [
                        { "id": 1, "name": "main" },
                        { "id": 2, "name": "worker-1" }
                    ]
                })),
            });
            write_message(writer, &resp).await.unwrap();
        }
        "stackTrace" => {
            let continue_count = state.lock().unwrap().continue_count;
            let frame_id_base = 1000 + i64::try_from(continue_count).unwrap() * 10;
            let top_line = if continue_count == 0 { 4 } else { 6 };
            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: true,
                command: req.command,
                message: None,
                body: Some(json!({
                    "stackFrames": [
                        {
                            "id": frame_id_base,
                            "name": "compute",
                            "source": { "name": "demo.rs", "path": source_path },
                            "line": top_line,
                            "column": 5
                        },
                        {
                            "id": frame_id_base + 1,
                            "name": "main",
                            "source": { "name": "demo.rs", "path": source_path },
                            "line": 2,
                            "column": 1
                        }
                    ],
                    "totalFrames": 2
                })),
            });
            write_message(writer, &resp).await.unwrap();
        }
        "scopes" => {
            let continue_count = state.lock().unwrap().continue_count;
            let scope_ref = 500 + i64::try_from(continue_count).unwrap() * 100;
            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: true,
                command: req.command,
                message: None,
                body: Some(json!({
                    "scopes": [
                        { "name": "Locals", "variablesReference": scope_ref, "expensive": false },
                        { "name": "Globals", "variablesReference": scope_ref + 50, "expensive": true }
                    ]
                })),
            });
            write_message(writer, &resp).await.unwrap();
        }
        "variables" => {
            let vref = req
                .arguments
                .as_ref()
                .and_then(|a| a.get("variablesReference"))
                .and_then(Value::as_i64)
                .unwrap_or_default();
            let vars = match vref {
                500 | 600 => json!([
                    { "name": "count", "value": "42", "type": "usize", "variablesReference": 0 },
                    { "name": "config", "value": "Config { .. }", "type": "Config", "variablesReference": vref + 1 }
                ]),
                501 | 601 => json!([
                    { "name": "retries", "value": "3", "type": "u8", "variablesReference": 0 },
                    { "name": "nested", "value": "Inner { .. }", "type": "Inner", "variablesReference": vref + 2 }
                ]),
                503 | 603 => json!([
                    { "name": "flag", "value": "true", "type": "bool", "variablesReference": 0 }
                ]),
                550 | 650 => json!([
                    { "name": "VERSION", "value": "\"1.0\"", "type": "&str", "variablesReference": 0 }
                ]),
                _ => json!([]),
            };
            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: true,
                command: req.command,
                message: None,
                body: Some(json!({ "variables": vars })),
            });
            write_message(writer, &resp).await.unwrap();
        }
        "evaluate" => {
            let expr = req
                .arguments
                .as_ref()
                .and_then(|a| a.get("expression"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let body = if expr == "count + 1" {
                json!({ "result": "43", "type": "usize", "variablesReference": 0 })
            } else {
                json!({ "result": "Config { retries: 3 }", "type": "Config", "variablesReference": 501 })
            };
            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: true,
                command: req.command,
                message: None,
                body: Some(body),
            });
            write_message(writer, &resp).await.unwrap();
        }
        "continue" | "next" | "stepIn" | "stepOut" => {
            let (delay, terminate_now) = {
                let mut guard = state.lock().unwrap();
                guard.continue_count += 1;
                let term = guard.terminate_on_second_continue && guard.continue_count >= 2;
                (guard.stop_delay, term)
            };

            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: true,
                command: req.command,
                message: None,
                body: Some(json!({ "allThreadsContinued": true })),
            });
            write_message(writer, &resp).await.unwrap();

            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }

            if terminate_now {
                let exited = ProtocolMessage::Event(DapEvent {
                    seq: next_seq(),
                    event: "exited".to_owned(),
                    body: Some(json!({ "exitCode": 0 })),
                });
                write_message(writer, &exited).await.unwrap();
                let terminated = ProtocolMessage::Event(DapEvent {
                    seq: next_seq(),
                    event: "terminated".to_owned(),
                    body: None,
                });
                write_message(writer, &terminated).await.unwrap();
            } else {
                let stopped = ProtocolMessage::Event(DapEvent {
                    seq: next_seq(),
                    event: "stopped".to_owned(),
                    body: Some(json!({
                        "reason": "breakpoint",
                        "threadId": 1,
                        "allThreadsStopped": true,
                        "hitBreakpointIds": [100]
                    })),
                });
                write_message(writer, &stopped).await.unwrap();
            }
        }
        "terminate" => {
            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: true,
                command: req.command,
                message: None,
                body: None,
            });
            write_message(writer, &resp).await.unwrap();
        }
        "disconnect" => {
            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: true,
                command: req.command,
                message: None,
                body: None,
            });
            write_message(writer, &resp).await.unwrap();
            return true;
        }
        other => {
            let resp = ProtocolMessage::Response(DapResponse {
                seq: next_seq(),
                request_seq: req.seq,
                success: false,
                command: other.to_owned(),
                message: Some(format!("unsupported command `{other}`")),
                body: None,
            });
            write_message(writer, &resp).await.unwrap();
        }
    }
    false
}

#[tokio::test]
#[allow(
    clippy::too_many_lines,
    reason = "end-to-end DAP lifecycle integration test exercising all subcommands on one session"
)]
async fn tcp_lifecycle_breakpoints_snapshots_lazy_vars_and_output() {
    let dir = tempfile::tempdir().unwrap();
    let src_path = dir.path().join("demo.rs");
    std::fs::write(
        &src_path,
        "fn main() {\n    compute();\n}\nfn compute() {\n    let count = 42;\n    println!(\"{count}\");\n}\n",
    )
    .unwrap();
    let src_str = src_path.to_str().unwrap().to_owned();

    let state = Arc::new(Mutex::new(MockAdapterState {
        source_path: src_str.clone(),
        ..MockAdapterState::default()
    }));
    let addr = spawn_mock_tcp_adapter(Arc::clone(&state)).await;

    let transport = AnyDapTransport::new(TcpTransport::connect(addr).await.unwrap());
    let config = DebugSessionConfig::new("mock-tcp")
        .with_target(addr.to_string())
        .with_output_buffer(OutputBufferConfig {
            max_entries: 16,
            max_bytes: 1024,
        });

    let (session, notif_rx) = DebugSession::start(&TokioGlobal, transport, config)
        .await
        .unwrap();

    // Launch stops on entry with source snippet and captured output.
    let outcome = session
        .launch(
            json!({ "program": "demo", "stopOnEntry": true }),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
    let ExecutionOutcome::Stopped(stop) = outcome else {
        panic!("expected entry stop, got {outcome:?}");
    };
    assert_eq!(stop.reason, "entry");
    assert_eq!(stop.top_frame.as_ref().unwrap().index, 0);
    assert_eq!(stop.top_frame.as_ref().unwrap().line, 4);
    let snippet = stop.source_snippet.as_ref().unwrap();
    assert!(snippet.lines.iter().any(|l| l.line == 4 && l.current));

    // Verify unsupported reverse request was cleanly rejected.
    assert!(state.lock().unwrap().reverse_request_rejected);

    // Output buffer captured stdout & stderr and dropped telemetry.
    let all_out = session
        .output(OutputCategory::All, None, false)
        .await
        .unwrap();
    assert_eq!(all_out.entries.len(), 2);
    let stderr_only = session
        .output(OutputCategory::Stderr, None, true)
        .await
        .unwrap();
    assert_eq!(stderr_only.entries.len(), 1);
    assert!(stderr_only.entries[0].text.contains("warning"));
    assert_eq!(
        session
            .output(OutputCategory::All, None, false)
            .await
            .unwrap()
            .entries,
        [] as [aither_dap::OutputEntry; 0]
    );

    // Incremental source breakpoints reconcile over DAP's replace-all setBreakpoints.
    let bp1 = session
        .set_source_breakpoint(SourceBreakpointSpec {
            file: src_str.clone(),
            line: 10,
            column: None,
            condition: None,
            hit_condition: None,
            log_message: None,
        })
        .await
        .unwrap();
    assert_eq!(bp1.id, 1);
    assert!(bp1.verified);
    assert!(matches!(
        bp1.target,
        BreakpointTarget::Source {
            requested_line: 10,
            resolved_line: Some(11),
            ..
        }
    ));

    let bp2 = session
        .set_source_breakpoint(SourceBreakpointSpec {
            file: src_str.clone(),
            line: 20,
            column: None,
            condition: Some("count > 10".to_owned()),
            hit_condition: None,
            log_message: None,
        })
        .await
        .unwrap();
    assert_eq!(bp2.id, 2);

    let bp3 = session
        .set_function_breakpoint(FunctionBreakpointSpec {
            name: "compute".to_owned(),
            condition: None,
            hit_condition: None,
        })
        .await
        .unwrap();
    assert_eq!(bp3.id, 3);
    assert!(bp3.verified);

    let removed = session.remove_breakpoint(bp1.id).await.unwrap();
    assert_eq!(removed.id, 1);
    assert_eq!(session.list_breakpoints().await.unwrap().len(), 2);

    {
        let calls = &state.lock().unwrap().set_breakpoints_calls;
        assert_eq!(calls, &vec![vec![10], vec![10, 20], vec![20]]);
    }

    // Inspect threads, stack, scopes, lazy one-level variable expansion, and eval.
    let threads = session.threads().await.unwrap();
    assert_eq!(threads.len(), 2);
    assert!(threads[0].active);

    let stack = session.stack(None, None).await.unwrap();
    assert_eq!(stack.frames.len(), 2);
    assert_eq!(stack.frames[0].index, 0);
    assert_eq!(stack.frames[1].index, 1);

    let vars = session.variables(Some(0), None, None).await.unwrap();
    let VariablesView::Frame { scopes, .. } = vars else {
        panic!("expected frame variables");
    };
    assert_eq!(scopes.len(), 2);
    assert_eq!(scopes[0].name, "Locals");
    let config_handle = scopes[0].variables[1].expand.unwrap();
    assert_eq!(config_handle, 1);
    // Expensive Globals scope is exposed lazily as an expandable handle.
    let globals_handle = scopes[1].variables[0].expand.unwrap();
    assert_eq!(globals_handle, 2);

    let expanded = session
        .variables(None, Some(config_handle), None)
        .await
        .unwrap();
    let VariablesView::Expanded {
        handle, variables, ..
    } = expanded
    else {
        panic!("expected expanded variables");
    };
    assert_eq!(handle, 1);
    assert_eq!(variables.len(), 2);
    assert_eq!(variables[0].name, "retries");
    assert_eq!(variables[1].expand, Some(3));

    let eval_scalar = session
        .evaluate("count + 1", Some(0), EvalContext::Repl)
        .await
        .unwrap();
    assert_eq!(eval_scalar.value, "43");
    assert_eq!(eval_scalar.expand, None);

    // Continuing synchronously hits breakpoint without emitting a duplicate async notification,
    // and invalidates previous stopped-snapshot variable handles.
    let cont = session
        .continue_execution(None, Duration::from_secs(2))
        .await
        .unwrap();
    assert!(matches!(cont, ExecutionOutcome::Stopped(_)));
    assert!(notif_rx.try_recv().is_err());

    // Old handle #1 from previous stop was cleared on resume.
    assert!(matches!(
        session.variables(None, Some(config_handle), None).await,
        Err(DapError::InvalidVariableHandle(1))
    ));

    // Verify handle_command rejects host-only commands and supports detach cleanly.
    assert!(matches!(
        session
            .handle_command(&DebugArgs::Launch {
                program: "demo".to_owned(),
                adapter: None,
                cwd: None,
                env: Vec::new(),
                stop_on_entry: false,
                timeout_ms: None,
                args: Vec::new(),
            })
            .await,
        Err(DapError::RequiresHostBroker("launch"))
    ));

    let detached = session.handle_command(&DebugArgs::Detach).await.unwrap();
    assert!(matches!(detached, DebugResult::Detached { .. }));
}

#[tokio::test]
async fn async_stop_and_termination_notifications_fire_after_timed_out_continue() {
    let dir = tempfile::tempdir().unwrap();
    let src_path = dir.path().join("async_demo.rs");
    std::fs::write(&src_path, "fn main() {}\nfn compute() {\n    let x = 1;\n    let y = 2;\n    let z = 3;\n    let w = 4;\n}\n").unwrap();

    let state = Arc::new(Mutex::new(MockAdapterState {
        source_path: src_path.to_str().unwrap().to_owned(),
        stop_delay: Duration::from_millis(180),
        terminate_on_second_continue: true,
        ..MockAdapterState::default()
    }));
    let addr = spawn_mock_tcp_adapter(Arc::clone(&state)).await;

    let transport = TcpTransport::connect(addr).await.unwrap();
    let (session, notif_rx) = DebugSession::start(
        &TokioGlobal,
        transport,
        DebugSessionConfig::new("mock-async"),
    )
    .await
    .unwrap();

    let init = session
        .launch(json!({ "program": "async_demo" }), Duration::from_secs(2))
        .await
        .unwrap();
    assert!(matches!(init, ExecutionOutcome::Stopped(_)));

    // 1. Continue with a short 40ms timeout -> returns Running immediately.
    let cont1 = session
        .continue_execution(None, Duration::from_millis(40))
        .await
        .unwrap();
    assert_eq!(cont1, ExecutionOutcome::Running);

    // While running, snapshot inspection fails fast with NotStopped.
    assert!(matches!(
        session.stack(None, None).await,
        Err(DapError::NotStopped { .. })
    ));

    // Asynchronous Stopped notification arrives once the delayed breakpoint hits.
    let notif1 = tokio::time::timeout(Duration::from_secs(2), notif_rx.recv())
        .await
        .expect("timed out waiting for async stop notification")
        .unwrap();
    let DebugNotification::Stopped(stop) = notif1 else {
        panic!("expected async Stopped notification, got {notif1:?}");
    };
    assert_eq!(stop.reason, "breakpoint");
    assert!(
        DebugNotification::Stopped(stop)
            .render()
            .contains("breakpoint")
    );

    // 2. Second continue times out while running, then target exits and terminates ->
    // async Terminated notification arrives with exit code 0.
    let cont2 = session
        .continue_execution(None, Duration::from_millis(40))
        .await
        .unwrap();
    assert_eq!(cont2, ExecutionOutcome::Running);

    let notif2 = tokio::time::timeout(Duration::from_secs(2), notif_rx.recv())
        .await
        .expect("timed out waiting for async termination notification")
        .unwrap();
    let DebugNotification::Terminated(term) = notif2 else {
        panic!("expected async Terminated notification, got {notif2:?}");
    };
    assert_eq!(term.exit_code, Some(0));

    let status = session.status().await.unwrap();
    assert_eq!(status.state, DebugSessionState::Terminated);
    assert_eq!(status.exit_code, Some(0));
}

#[tokio::test]
async fn debug_tool_reports_unavailable_without_broker_and_renders_broker_response() {
    let unavailable = DebugTool::new("s1".to_owned(), None);
    let out = unavailable
        .call(DebugArgs::Status, ToolContext::new())
        .await
        .unwrap()
        .into_tool_result();
    assert!(out.is_error());
    assert!(out.render_for_model().contains("unavailable"));

    let (broker, queue) = channel();
    let tool = DebugTool::new("s1".to_owned(), Some(broker));
    let host = tokio::spawn(async move {
        let req = queue.next().await.unwrap();
        assert_eq!(req.args.session_id, "s1");
        req.respond(Ok(DebugResult::Running {
            note: "Target is running.".to_owned(),
        }))
        .unwrap();
    });

    let out = tool
        .call(
            DebugArgs::Continue {
                thread_id: None,
                timeout_ms: Some(100),
            },
            ToolContext::new(),
        )
        .await
        .unwrap()
        .into_tool_result();
    assert!(!out.is_error());
    assert!(out.render_for_model().contains("asynchronous notification"));
    host.await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn stdio_transport_frames_dap_messages_with_child_process() {
    use aither_dap::StdioTransport;

    // Minimal POSIX shell mock DAP adapter speaking Content-Length framing over stdio.
    let script = r#"
send_frame() {
    body="$1"
    len=$(printf '%s' "$body" | wc -c | tr -d ' ')
    printf 'Content-Length: %s\r\n\r\n%s' "$len" "$body"
}
while IFS= read -r line; do
    line=$(printf '%s' "$line" | tr -d '\r')
    case "$line" in
        Content-Length:*)
            len=$(printf '%s' "${line#Content-Length:}" | tr -d ' ')
            ;;
        "")
            body=$(dd bs=1 count="$len" 2>/dev/null)
            case "$body" in
                *'"command":"initialize"'*)
                    send_frame '{"seq":1,"type":"response","request_seq":1,"success":true,"command":"initialize","body":{"supportsConfigurationDoneRequest":true}}'
                    ;;
                *'"command":"launch"'*)
                    send_frame '{"seq":2,"type":"event","event":"initialized"}'
                    send_frame '{"seq":3,"type":"response","request_seq":2,"success":true,"command":"launch"}'
                    ;;
                *'"command":"configurationDone"'*)
                    send_frame '{"seq":4,"type":"response","request_seq":3,"success":true,"command":"configurationDone"}'
                    send_frame '{"seq":5,"type":"event","event":"stopped","body":{"reason":"entry","threadId":1}}'
                    ;;
                *'"command":"threads"'*)
                    send_frame '{"seq":6,"type":"response","request_seq":4,"success":true,"command":"threads","body":{"threads":[{"id":1,"name":"main"}]}}'
                    ;;
                *'"command":"stackTrace"'*)
                    send_frame '{"seq":7,"type":"response","request_seq":5,"success":true,"command":"stackTrace","body":{"stackFrames":[{"id":1,"name":"main","line":1,"column":1}]}}'
                    ;;
                *'"command":"disconnect"'*)
                    send_frame '{"seq":8,"type":"response","request_seq":6,"success":true,"command":"disconnect"}'
                    exit 0
                    ;;
            esac
            ;;
    esac
done
"#;

    let transport = StdioTransport::spawn("/bin/sh", &["-c", script]).unwrap();
    assert!(transport.pid() > 0);
    let (session, _notif_rx) = DebugSession::start(
        &TokioGlobal,
        transport,
        DebugSessionConfig::new("mock-stdio"),
    )
    .await
    .unwrap();

    let outcome = session
        .launch(json!({ "program": "stdio_demo" }), Duration::from_secs(3))
        .await
        .unwrap();
    let ExecutionOutcome::Stopped(stop) = outcome else {
        panic!("expected entry stop over stdio, got {outcome:?}");
    };
    assert_eq!(stop.reason, "entry");
    assert_eq!(stop.top_frame.as_ref().unwrap().name, "main");
    session.detach().await.unwrap();
}
