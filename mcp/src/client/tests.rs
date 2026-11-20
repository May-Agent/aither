//! The client's receive loop over the child-process and HTTP transports.

use std::time::Duration;

use async_channel::{Receiver, Sender};
use async_process::Command;
use executor_core::tokio::TokioGlobal;
use serde_json::json;

use super::{McpClient, McpConnection, McpHandler, ServerNotification};
use crate::protocol::{ErrorCode, McpError};
use crate::test_server::{serve, serve_routes, sse};
use crate::transport::{ChildProcessTransport, HttpTransport};

/// Forwards every notification to the test.
struct Recorder(Sender<ServerNotification>);

impl McpHandler for Recorder {
    async fn notification(&self, notification: ServerNotification) {
        self.0.send(notification).await.unwrap();
    }
}

fn recorder() -> (Recorder, Receiver<ServerNotification>) {
    let (sender, receiver) = async_channel::unbounded();
    (Recorder(sender), receiver)
}

/// Wait for the next notification, bounded so a regression fails the test
/// instead of hanging it.
async fn next(notifications: &Receiver<ServerNotification>) -> ServerNotification {
    tokio::time::timeout(Duration::from_secs(10), notifications.recv())
        .await
        .expect("no notification within 10s")
        .unwrap()
}

/// The `params` of a `test/echo` notification.
fn echoed(notification: ServerNotification) -> serde_json::Value {
    let ServerNotification::Other(echo) = notification else {
        panic!("expected a test/echo notification, got {notification:?}");
    };
    assert_eq!(echo.method, "test/echo");
    echo.params.unwrap()
}

fn assert_ping_answered(response: &serde_json::Value) {
    assert_eq!(response["id"], "ping-1");
    assert_eq!(response["result"], json!({}));
}

fn assert_method_not_found(response: &serde_json::Value) {
    assert_eq!(response["id"], "sampling-1");
    assert_eq!(response["error"]["code"], ErrorCode::METHOD_NOT_FOUND.0);
}

fn script(name: &str) -> Command {
    let mut command = Command::new("sh");
    command.arg(format!(
        "{}/src/client/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ));
    command
}

#[tokio::test]
async fn child_process_server_messages_interleave_with_a_request() {
    let (handler, notifications) = recorder();
    let transport = ChildProcessTransport::from_command(&mut script("interleaving_server.sh"))
        .expect("spawn sh");
    let client = McpClient::connect(&TokioGlobal, transport, handler)
        .await
        .unwrap();
    assert_eq!(client.server_info().name, "script");

    // The request gets its response although the server pings, notifies and
    // sends an unadvertised request first.
    assert!(client.list_tools().await.unwrap().is_empty());

    assert!(matches!(
        next(&notifications).await,
        ServerNotification::ResourcesListChanged
    ));
    assert_ping_answered(&echoed(next(&notifications).await));
    assert_method_not_found(&echoed(next(&notifications).await));

    client.close().await;
}

#[tokio::test]
async fn child_process_tool_list_changed_refreshes_the_connection_tools() {
    let (handler, notifications) = recorder();
    let transport = ChildProcessTransport::from_command(&mut script("changing_tools_server.sh"))
        .expect("spawn sh");
    let conn = McpConnection::connect(&TokioGlobal, transport, handler)
        .await
        .unwrap();
    let names = |conn: &McpConnection| {
        conn.mcp_definitions()
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(names(&conn), ["alpha", "beta"]);
    let before = conn.mcp_definitions();

    // The server answers this call, then announces a different tool list.
    conn.call("alpha", json!({})).await.unwrap();
    assert!(matches!(
        next(&notifications).await,
        ServerNotification::ToolsListChanged
    ));

    // The re-listing runs beside the handler; wait for it, bounded.
    tokio::time::timeout(Duration::from_secs(10), async {
        while !conn.has_tool("gamma") {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the tool list was not refreshed within 10s");

    assert_eq!(names(&conn), ["beta", "gamma"]);
    assert_eq!(conn.definitions().len(), 2);
    // A list taken before the refresh is unchanged, never a mixture.
    assert_eq!(before.len(), 2);
    assert_eq!(before[0].name, "alpha");

    // The removed tool is no longer offered or called.
    assert!(!conn.has_tool("alpha"));
    assert!(matches!(
        conn.call("alpha", json!({})).await,
        Err(McpError::UnknownTool(name)) if name == "alpha"
    ));

    conn.close().await;
}

#[tokio::test]
async fn the_servers_tools_are_ordinary_tools_that_follow_its_list() {
    use aither_core::llm::{Tool, ToolContext, ToolResult};

    let transport = ChildProcessTransport::from_command(&mut script("changing_tools_server.sh"))
        .expect("spawn sh");
    let conn = std::sync::Arc::new(
        McpConnection::connect(&TokioGlobal, transport, ())
            .await
            .unwrap(),
    );
    let names = |tools: &[crate::McpTool]| {
        let mut names: Vec<_> = tools.iter().map(|tool| tool.name().into_owned()).collect();
        names.sort();
        names
    };
    let tools = conn.tools();
    assert_eq!(names(&tools), ["alpha", "beta"]);
    assert_eq!(
        tools[0].arguments_schema().to_value()["type"],
        "object",
        "the schema is the server's"
    );

    // The server answers this call, then announces a different tool list.
    let alpha = tools.iter().find(|tool| tool.name() == "alpha").unwrap();
    assert_eq!(
        alpha.call(json!({}), ToolContext::new()).await.unwrap(),
        ToolResult::parts(Vec::new())
    );
    tokio::time::timeout(Duration::from_secs(10), async {
        while !conn.has_tool("gamma") {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the tool list was not refreshed within 10s");

    // Taking the tools again follows the server's list, and a tool the
    // server dropped is no longer callable.
    assert_eq!(names(&conn.tools()), ["beta", "gamma"]);
    let gone = alpha
        .call(json!({}), ToolContext::new())
        .await
        .expect_err("the server no longer has alpha");
    assert!(gone.to_string().contains("alpha"), "{gone}");
}

#[tokio::test]
async fn child_process_exit_fails_the_pending_request() {
    let transport =
        ChildProcessTransport::from_command(&mut script("exiting_server.sh")).expect("spawn sh");
    let client = McpClient::connect(&TokioGlobal, transport, ())
        .await
        .unwrap();

    let error = client.list_tools().await.unwrap_err();
    let McpError::Transport(message) = &error else {
        panic!("expected a transport error naming the exit, got {error:?}");
    };
    assert!(message.contains("server process"), "{message}");

    // The loop is gone: later requests fail too, and closing is a no-op.
    assert!(client.list_tools().await.is_err());
    client.close().await;
}

#[tokio::test]
async fn http_slow_reply_does_not_stall_other_requests_or_server_messages() {
    use std::sync::Arc;

    use tokio::sync::Notify;

    let slow_arrived = Arc::new(Notify::new());
    let release_slow = Arc::new(Notify::new());
    let url = {
        let slow_arrived = slow_arrived.clone();
        let release_slow = release_slow.clone();
        serve_routes(move |request| {
            let slow_arrived = slow_arrived.clone();
            let release_slow = release_slow.clone();
            async move {
                let reply = |result: serde_json::Value| {
                    json!({"jsonrpc": "2.0", "id": request["id"], "result": result}).to_string()
                };
                match request["method"].as_str().unwrap() {
                    "initialize" => (
                        "Content-Type: application/json\r\n",
                        reply(json!({
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "serverInfo": {"name": "http-script"}
                        })),
                    ),
                    // Plain-JSON reply, held back until the test lets go.
                    "tools/call" => {
                        slow_arrived.notify_one();
                        release_slow.notified().await;
                        (
                            "Content-Type: application/json\r\n",
                            reply(json!({"content": [], "isError": false})),
                        )
                    }
                    // SSE reply carrying a server message before the result.
                    "tools/list" => (
                        "Content-Type: text/event-stream\r\n",
                        sse(&[
                            json!({"jsonrpc": "2.0", "method": "notifications/resources/list_changed"}),
                            serde_json::from_str(&reply(json!({"tools": []}))).unwrap(),
                        ]),
                    ),
                    _ => ("", String::new()),
                }
            }
        })
        .await
    };

    let (handler, notifications) = recorder();
    let client = McpClient::connect(&TokioGlobal, HttpTransport::new(url), handler)
        .await
        .unwrap();

    let both = async {
        tokio::join!(client.call_tool("slow", json!({})), async {
            // Only once the slow request is at the server, issue another.
            slow_arrived.notified().await;
            let tools = client.list_tools().await.unwrap();
            let first = next(&notifications).await;
            // The slow request is still outstanding; let it finish.
            release_slow.notify_one();
            (tools, first)
        })
    };
    let (slow, (tools, first)) = tokio::time::timeout(Duration::from_secs(10), both)
        .await
        .expect("a slow reply stalled the connection");

    assert!(tools.is_empty());
    assert!(matches!(first, ServerNotification::ResourcesListChanged));
    assert!(!slow.unwrap().is_error);
}

#[tokio::test]
async fn http_server_messages_interleave_with_a_request() {
    let initialize = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "serverInfo": {"name": "http-script"}
        }
    })
    .to_string();
    let list = sse(&[
        json!({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progressToken": 7, "progress": 1.0}}),
        json!({"jsonrpc": "2.0", "id": "ping-1", "method": "ping"}),
        json!({"jsonrpc": "2.0", "id": "sampling-1", "method": "sampling/createMessage", "params": {}}),
        json!({"jsonrpc": "2.0", "id": 2, "result": {"tools": []}}),
    ]);
    let (url, seen) = serve(vec![
        ("Content-Type: application/json\r\n", initialize),
        // The `notifications/initialized` acknowledgement.
        ("", String::new()),
        ("Content-Type: text/event-stream\r\n", list),
        // Acknowledgements of the ping answer and of the rejection.
        ("", String::new()),
        ("", String::new()),
    ])
    .await;

    let (handler, notifications) = recorder();
    let client = McpClient::connect(&TokioGlobal, HttpTransport::new(url), handler)
        .await
        .unwrap();
    assert!(client.list_tools().await.unwrap().is_empty());

    let ServerNotification::Progress(progress) = next(&notifications).await else {
        panic!("expected the progress notification");
    };
    assert!((progress.progress - 1.0).abs() < f64::EPSILON);

    // The answers were POSTed back before the response was routed.
    let seen = seen.lock().unwrap();
    // The body may be chunk-framed; the JSON object is what lies between the
    // outermost braces.
    let body = |request: &str| {
        let (_, body) = request.split_once("\r\n\r\n").unwrap();
        let object = &body[body.find('{').unwrap()..=body.rfind('}').unwrap()];
        serde_json::from_str::<serde_json::Value>(object).unwrap()
    };
    assert_ping_answered(&body(&seen[3]));
    assert_method_not_found(&body(&seen[4]));
}

#[tokio::test]
async fn dropping_call_tool_or_cancelling_mcp_tool_notifies_server_and_cleans_up() {
    use std::borrow::Cow;
    use std::sync::Arc;

    use aither_core::llm::tool::{Canceller, Tool, ToolContext, ToolResult, Tools};
    use schemars::JsonSchema;
    use serde::Deserialize;

    use crate::McpServer;
    use crate::transport::DuplexTransport;

    /// Arguments for the test hang tool.
    #[derive(Deserialize, JsonSchema)]
    struct HangArgs {
        tag: String,
    }

    struct HangTool {
        started_tx: async_channel::Sender<String>,
        cancelled_tx: async_channel::Sender<String>,
    }

    impl Tool for HangTool {
        fn name(&self) -> Cow<'static, str> {
            Cow::Borrowed("hang")
        }

        fn description(&self) -> Cow<'static, str> {
            Cow::Borrowed("Hangs until cancelled")
        }

        type Arguments = HangArgs;
        type Res = ToolResult;

        type Error = core::convert::Infallible;

        async fn call(
            &self,
            args: Self::Arguments,
            cx: ToolContext,
        ) -> Result<Self::Res, Self::Error> {
            let cancelled_tx = self.cancelled_tx.clone();
            let signal = cx.cancellation().clone();
            let tag = args.tag;
            tokio::spawn(async move {
                signal.cancelled().await;
                let _ = cancelled_tx.send(tag).await;
            });
            let _ = self.started_tx.send(String::from("started")).await;
            std::future::pending::<()>().await;
            Ok(ToolResult::text("unreachable"))
        }
    }

    let (started_tx, started_rx) = async_channel::unbounded::<String>();
    let (cancelled_tx, cancelled_rx) = async_channel::unbounded::<String>();

    let mut server_tools = Tools::new();
    server_tools
        .register(HangTool {
            started_tx,
            cancelled_tx,
        })
        .unwrap();

    let (client_end, server_end) = DuplexTransport::pair();
    let mut server = McpServer::new(server_end, server_tools, "cancel-srv", "1.0.0");
    let server_task = tokio::spawn(async move { server.run().await });

    let conn = Arc::new(
        McpConnection::connect(&TokioGlobal, client_end, ())
            .await
            .unwrap(),
    );

    // 1. Dropping McpConnection::call (McpClient::call_tool) sends notifications/cancelled.
    {
        let mut call_fut = Box::pin(conn.call("hang", json!({"tag": "dropped"})));
        tokio::select! {
            _ = &mut call_fut => panic!("hang never resolves"),
            _ = started_rx.recv() => {}
        }
    }
    let first_cancelled = tokio::time::timeout(Duration::from_secs(5), cancelled_rx.recv())
        .await
        .expect("server must receive notifications/cancelled on drop")
        .unwrap();
    assert_eq!(first_cancelled, "dropped");

    // 2. Firing Canceller on McpTool::call cancels cleanly and notifies server.
    let mcp_tool = conn.tools().into_iter().next().expect("hang tool");
    let canceller = Canceller::new();
    let cx = ToolContext::new().with_cancellation(canceller.signal());
    let tool_task =
        tokio::spawn(async move { mcp_tool.call(json!({"tag": "signalled"}), cx).await });

    started_rx.recv().await.unwrap();
    canceller.cancel();

    let err = tokio::time::timeout(Duration::from_secs(5), tool_task)
        .await
        .expect("McpTool::call must abort on cancel")
        .unwrap()
        .expect_err("cancelled tool call returns Err");
    assert!(err.to_string().contains("cancelled"), "{err}");

    let second_cancelled = tokio::time::timeout(Duration::from_secs(5), cancelled_rx.recv())
        .await
        .expect("server must receive notifications/cancelled on signal")
        .unwrap();
    assert_eq!(second_cancelled, "signalled");

    // Connection remains usable after cancelled calls.
    assert_eq!(conn.definitions().len(), 1);
    conn.close().await;
    server_task.await.unwrap().unwrap();
}

#[tokio::test]
async fn http_single_failed_post_does_not_tear_down_mcp_client() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());

    tokio::spawn(async move {
        let mut req_num = 0usize;
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            req_num += 1;
            let mut buf = vec![0u8; 4096];
            let n = stream.read(&mut buf).await.unwrap_or(0);
            let raw = String::from_utf8_lossy(&buf[..n]);
            let id = raw
                .find("\"id\":")
                .and_then(|pos| {
                    raw[pos + 5..]
                        .chars()
                        .take_while(char::is_ascii_digit)
                        .collect::<String>()
                        .parse::<i64>()
                        .ok()
                })
                .unwrap_or(1);

            let (status, headers, body) = match req_num {
                1 => (
                    "200 OK",
                    "Content-Type: application/json\r\n",
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "serverInfo": {"name": "resilient-http"}
                        }
                    })
                    .to_string(),
                ),
                2 => ("202 Accepted", "", String::new()),
                3 => (
                    "500 Internal Server Error",
                    "Content-Type: text/plain\r\n",
                    "transient error".to_string(),
                ),
                _ => (
                    "200 OK",
                    "Content-Type: application/json\r\n",
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {"tools": []}
                    })
                    .to_string(),
                ),
            };
            let response = format!(
                "HTTP/1.1 {status}\r\n{headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.shutdown().await;
        }
    });

    let client = McpClient::connect(&TokioGlobal, HttpTransport::new(url), ())
        .await
        .unwrap();

    // First call hits HTTP 500 and fails.
    assert!(client.call_tool("broken", json!({})).await.is_err());

    // Client connection stays alive for subsequent requests.
    assert!(client.list_tools().await.unwrap().is_empty());
    client.close().await;
}

#[tokio::test]
async fn from_configs_connects_servers_concurrently() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Notify;

    use crate::{McpServerConfig, McpServersConfig};

    let arrived = Arc::new(AtomicUsize::new(0));
    let both_arrived = Arc::new(Notify::new());

    let mut configs = McpServersConfig::new();
    for name in ["srv-a", "srv-b"] {
        let arrived = Arc::clone(&arrived);
        let both_arrived = Arc::clone(&both_arrived);
        let url = serve_routes(move |request| {
            let arrived = Arc::clone(&arrived);
            let both_arrived = Arc::clone(&both_arrived);
            async move {
                let reply = |result: serde_json::Value| {
                    json!({"jsonrpc": "2.0", "id": request["id"], "result": result}).to_string()
                };
                match request["method"].as_str().unwrap_or("") {
                    "initialize" => {
                        let waiter = both_arrived.notified();
                        if arrived.fetch_add(1, Ordering::SeqCst) + 1 >= 2 {
                            both_arrived.notify_waiters();
                        } else {
                            waiter.await;
                        }
                        (
                            "Content-Type: application/json\r\n",
                            reply(json!({
                                "protocolVersion": "2025-06-18",
                                "capabilities": {},
                                "serverInfo": {"name": "parallel"}
                            })),
                        )
                    }
                    "tools/list" => (
                        "Content-Type: application/json\r\n",
                        reply(json!({"tools": []})),
                    ),
                    _ => ("", String::new()),
                }
            }
        })
        .await;

        configs.insert(
            name.to_string(),
            McpServerConfig {
                command: None,
                args: Vec::new(),
                url: Some(url),
                env: std::collections::HashMap::new(),
            },
        );
    }

    // Sequential connection would deadlock waiting for the second `initialize`.
    let connected = tokio::time::timeout(
        Duration::from_secs(5),
        McpConnection::from_configs(&TokioGlobal, &configs),
    )
    .await
    .expect("from_configs must connect concurrently without deadlocking")
    .unwrap();
    assert_eq!(connected.len(), 2);
}
