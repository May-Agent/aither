//! MCP server that exposes aither tools.

use std::collections::HashMap;
use std::sync::Arc;

use aither_core::llm::tool::{
    Canceller, Progress, ProgressSink, ToolContext, ToolResult, ToolResultPart, Tools,
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use futures_lite::future;
use futures_util::future::{AbortHandle, Abortable, Aborted, BoxFuture};
use futures_util::stream::{FuturesUnordered, StreamExt};
use tracing::debug;

use crate::protocol::{
    CallToolParams, CallToolResult, CancelledParams, Content, ImageContent, InitializeParams,
    InitializeResult, JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest,
    JsonRpcResponse, ListToolsResult, McpError, McpToolDefinition, PROTOCOL_VERSION,
    ProgressNotificationParams, ProgressToken, RequestId, ServerCapabilities, ServerInfo,
    TextContent, ToolsCapability,
};
use crate::transport::{BidirectionalTransport, StdioTransport};

/// A `tools/call` execution that resolves to its request ID, execution slot,
/// and response, or to [`Aborted`] once a `notifications/cancelled`
/// notification names its ID.
type InFlightCall = Abortable<BoxFuture<'static, (RequestId, CallSlot, JsonRpcResponse)>>;

/// The `tools/call` executions currently running, resolved in any order.
type InFlightCalls = FuturesUnordered<InFlightCall>;

/// How many progress reports may wait for the server loop to write them
/// before a reporting tool's [`ToolContext::report_progress`] waits too.
const PROGRESS_BACKLOG: usize = 32;

/// Distinguishes one `tools/call` execution from every other one this server
/// runs, including a later call that reuses the same client-chosen request ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CallSlot(u64);

/// A running `tools/call`, as the server loop tracks it.
struct InFlight {
    /// Drops the call's future on `notifications/cancelled`.
    abort: AbortHandle,
    /// Fires the call's [`ToolContext::cancellation`] signal so work outliving
    /// the call's future stops as well.
    canceller: Canceller,
    /// This execution's identity, which progress reports quote.
    slot: CallSlot,
}

/// A progress report on its way from a running tool to the wire.
struct ProgressReport {
    request: RequestId,
    slot: CallSlot,
    params: ProgressNotificationParams,
}

/// The [`ProgressSink`] behind the context of a `tools/call` that carried a
/// progress token. Calls without one get [`ToolContext::new`], so reporting
/// there is a no-op, as the MCP specification requires.
struct CallProgress {
    request: RequestId,
    slot: CallSlot,
    token: ProgressToken,
    reports: async_channel::Sender<ProgressReport>,
}

impl ProgressSink for CallProgress {
    async fn report(&mut self, progress: Progress) {
        let report = ProgressReport {
            request: self.request.clone(),
            slot: self.slot,
            params: ProgressNotificationParams {
                progress_token: self.token.clone(),
                progress: progress.progress(),
                total: progress.total(),
                message: progress.message().map(str::to_owned),
            },
        };
        if self.reports.send(report).await.is_err() {
            debug!("Server stopped; dropping progress for {:?}", self.request);
        }
    }
}

/// One event driving an iteration of the server loop.
enum ServerEvent {
    /// The client sent a message, closed the connection, or the transport failed.
    Message(Result<Option<JsonRpcMessage>, McpError>),
    /// An in-flight `tools/call` produced its response or was aborted.
    Call(Result<(RequestId, CallSlot, JsonRpcResponse), Aborted>),
    /// A tool reported progress.
    Progress(ProgressReport),
}

/// MCP server that exposes aither tools to external clients.
///
/// # Example
///
/// ```ignore
/// use aither_mcp::McpServer;
/// use aither_core::llm::tool::Tools;
///
/// let mut tools = Tools::new();
/// tools.register(my_tool);
///
/// let mut server = McpServer::stdio(tools, "my-server", "1.0.0");
/// server.run().await?;
/// ```
pub struct McpServer<T: BidirectionalTransport> {
    transport: T,
    tools: Arc<Tools>,
    info: ServerInfo,
    instructions: Option<String>,
    initialized: bool,
}

impl<T: BidirectionalTransport> std::fmt::Debug for McpServer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpServer")
            .field("info", &self.info)
            .field("initialized", &self.initialized)
            .finish_non_exhaustive()
    }
}

impl McpServer<StdioTransport> {
    /// Create an MCP server using stdio transport.
    #[must_use]
    pub fn stdio(tools: Tools, name: impl Into<String>, version: impl Into<String>) -> Self {
        Self::new(StdioTransport::new(), tools, name, version)
    }
}

impl<T: BidirectionalTransport + Sync> McpServer<T> {
    /// Create a new MCP server with a custom transport.
    #[must_use]
    pub fn new(
        transport: T,
        tools: Tools,
        name: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self {
            transport,
            tools: Arc::new(tools),
            info: ServerInfo {
                name: name.into(),
                version: Some(version.into()),
            },
            instructions: None,
            initialized: false,
        }
    }

    /// Adds server-wide instructions returned during MCP initialization.
    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Run the server main loop.
    ///
    /// Processes incoming messages until the connection is closed.
    /// Requests other than `initialize` and `ping` are rejected before
    /// `initialize` completes, and duplicate in-flight request IDs are
    /// rejected without disturbing the running execution.
    ///
    /// # Errors
    ///
    /// Returns an error if a fatal transport error occurs.
    ///
    /// # Panics
    ///
    /// Panics only on an internal bug: the in-flight call set is polled only
    /// while it is non-empty, and the progress channel cannot close while this
    /// loop holds a sender.
    pub async fn run(&mut self) -> Result<(), McpError> {
        debug!("MCP server starting: {}", self.info.name);

        let mut calls = InFlightCalls::new();
        let mut in_flight: HashMap<RequestId, InFlight> = HashMap::new();
        let (reports_tx, reports_rx) = async_channel::bounded(PROGRESS_BACKLOG);
        let mut next_slot = 0;

        let outcome = loop {
            let incoming = future::race(
                async { ServerEvent::Message(self.transport.recv().await) },
                async {
                    ServerEvent::Progress(
                        reports_rx
                            .recv()
                            .await
                            .expect("the server loop holds a progress sender"),
                    )
                },
            );
            let event = if calls.is_empty() {
                incoming.await
            } else {
                future::race(incoming, async {
                    ServerEvent::Call(calls.next().await.expect("call set is not empty"))
                })
                .await
            };

            match event {
                ServerEvent::Message(Ok(Some(JsonRpcMessage::Request(req)))) => {
                    if in_flight.contains_key(&req.id) {
                        let response = JsonRpcResponse::error(
                            req.id,
                            JsonRpcError::invalid_request("Request ID is already in flight"),
                        );
                        self.transport.respond(response).await?;
                        continue;
                    }
                    if !self.initialized && req.method != "initialize" && req.method != "ping" {
                        let response = JsonRpcResponse::error(
                            req.id,
                            JsonRpcError::invalid_request("Server not initialized"),
                        );
                        self.transport.respond(response).await?;
                        continue;
                    }
                    if req.method == "tools/call" {
                        let slot = CallSlot(next_slot);
                        next_slot += 1;
                        self.enqueue_call(req, slot, &reports_tx, &calls, &mut in_flight);
                    } else {
                        let response = self.handle_request(req);
                        self.transport.respond(response).await?;
                    }
                }
                ServerEvent::Message(Ok(Some(msg))) => {
                    if let Err(e) = self.handle_message(msg, &mut in_flight).await {
                        debug!("Error handling message: {e}");
                    }
                }
                ServerEvent::Message(Ok(None)) => {
                    debug!("Connection closed");
                    break Ok(());
                }
                ServerEvent::Message(Err(e)) => break Err(e),
                ServerEvent::Call(Ok((id, slot, response))) => {
                    let still_active = in_flight.get(&id).is_some_and(|call| call.slot == slot);
                    if !still_active {
                        continue;
                    }
                    while let Ok(report) = reports_rx.try_recv() {
                        if let Err(err) = self.forward_progress(report, &in_flight).await {
                            cancel_all_in_flight(&mut in_flight);
                            return Err(err);
                        }
                    }
                    in_flight.remove(&id);
                    if let Err(err) = self.transport.respond(response).await {
                        cancel_all_in_flight(&mut in_flight);
                        return Err(err);
                    }
                }
                ServerEvent::Call(Err(Aborted)) => {
                    debug!("In-flight tool call was cancelled");
                }
                ServerEvent::Progress(report) => {
                    if let Err(err) = self.forward_progress(report, &in_flight).await {
                        cancel_all_in_flight(&mut in_flight);
                        return Err(err);
                    }
                }
            }
        };

        cancel_all_in_flight(&mut in_flight);
        outcome
    }

    async fn forward_progress(
        &mut self,
        report: ProgressReport,
        in_flight: &HashMap<RequestId, InFlight>,
    ) -> Result<(), McpError> {
        if in_flight
            .get(&report.request)
            .is_some_and(|call| call.slot == report.slot)
        {
            self.transport
                .notify(JsonRpcNotification::with_params(
                    "notifications/progress",
                    report.params,
                ))
                .await
        } else {
            debug!("Dropping progress for finished call {:?}", report.request);
            Ok(())
        }
    }

    fn enqueue_call(
        &self,
        req: JsonRpcRequest,
        slot: CallSlot,
        reports: &async_channel::Sender<ProgressReport>,
        calls: &InFlightCalls,
        in_flight: &mut HashMap<RequestId, InFlight>,
    ) {
        let (abort, registration) = AbortHandle::new_pair();
        let canceller = Canceller::new();
        let id = req.id;
        let call: BoxFuture<'static, (RequestId, CallSlot, JsonRpcResponse)> =
            match parse_call_params(req.params) {
                Ok(params) => {
                    let cx = params
                        .meta
                        .and_then(|meta| meta.progress_token)
                        .map_or_else(ToolContext::new, |token| {
                            ToolContext::with_progress(CallProgress {
                                request: id.clone(),
                                slot,
                                token,
                                reports: reports.clone(),
                            })
                        })
                        .with_cancellation(canceller.signal());
                    let tools = Arc::clone(&self.tools);
                    let call_id = id.clone();
                    Box::pin(async move {
                        let response = Self::handle_call_tool(
                            &tools,
                            call_id.clone(),
                            &params.name,
                            &params.arguments,
                            cx,
                        )
                        .await;
                        (call_id, slot, response)
                    })
                }
                Err(error) => {
                    let response = JsonRpcResponse::error(id.clone(), error);
                    Box::pin(future::ready((id.clone(), slot, response)))
                }
            };
        calls.push(Abortable::new(call, registration));
        in_flight.insert(
            id,
            InFlight {
                abort,
                canceller,
                slot,
            },
        );
    }

    async fn handle_message(
        &mut self,
        msg: JsonRpcMessage,
        in_flight: &mut HashMap<RequestId, InFlight>,
    ) -> Result<(), McpError> {
        match msg {
            JsonRpcMessage::Request(req) => {
                let response = self.handle_request(req);
                self.transport.respond(response).await?;
            }
            JsonRpcMessage::Notification(notif) => {
                Self::handle_notification(notif, in_flight);
            }
            JsonRpcMessage::Response(_) => {
                debug!("Unexpected response message");
            }
        }
        Ok(())
    }

    fn handle_request(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        debug!("Handling request: {}", req.method);

        match req.method.as_str() {
            "initialize" => self.handle_initialize(req),
            "ping" => JsonRpcResponse::success(req.id, serde_json::json!({})),
            "tools/list" => self.handle_list_tools(req),
            method => JsonRpcResponse::error(req.id, JsonRpcError::method_not_found(method)),
        }
    }

    fn handle_notification(
        notif: JsonRpcNotification,
        in_flight: &mut HashMap<RequestId, InFlight>,
    ) {
        debug!("Received notification: {}", notif.method);

        match notif.method.as_str() {
            "notifications/initialized" => {
                debug!("Client initialized");
            }
            "notifications/cancelled" => {
                match notif
                    .params
                    .map(serde_json::from_value::<CancelledParams>)
                    .transpose()
                {
                    Ok(Some(params)) => {
                        if let Some(call) = in_flight.remove(&params.request_id) {
                            debug!("Cancelling tool call {:?}", params.request_id);
                            call.canceller.cancel();
                            call.abort.abort();
                        } else {
                            debug!("No in-flight call {:?}", params.request_id);
                        }
                    }
                    Ok(None) => debug!("notifications/cancelled without params"),
                    Err(e) => debug!("Invalid notifications/cancelled params: {e}"),
                }
            }
            _ => {}
        }
    }

    fn handle_initialize(&mut self, req: JsonRpcRequest) -> JsonRpcResponse {
        if self.initialized {
            return JsonRpcResponse::error(
                req.id,
                JsonRpcError::invalid_request("Server is already initialized"),
            );
        }

        let _params: InitializeParams = match req.params.map(serde_json::from_value).transpose() {
            Ok(p) => p.unwrap_or_default(),
            Err(e) => {
                return JsonRpcResponse::error(req.id, JsonRpcError::invalid_params(e.to_string()));
            }
        };

        self.initialized = true;

        let result = InitializeResult {
            protocol_version: PROTOCOL_VERSION.to_string(),
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability { list_changed: None }),
                ..Default::default()
            },
            server_info: self.info.clone(),
            instructions: self.instructions.clone(),
        };

        JsonRpcResponse::success(req.id, result)
    }

    fn handle_list_tools(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        let definitions = self.tools.definitions();

        let mcp_tools: Vec<McpToolDefinition> = definitions
            .iter()
            .map(|def| McpToolDefinition {
                name: def.name().to_string(),
                description: Some(def.description().to_string()),
                input_schema: def.arguments_openai_schema(),
            })
            .collect();

        let result = ListToolsResult {
            tools: mcp_tools,
            next_cursor: None,
        };

        JsonRpcResponse::success(req.id, result)
    }

    async fn handle_call_tool(
        tools: &Tools,
        id: RequestId,
        name: &str,
        arguments: &serde_json::Value,
        cx: ToolContext,
    ) -> JsonRpcResponse {
        let args_str = serde_json::to_string(arguments).unwrap_or_default();

        match tools.call(name, &args_str, cx).await {
            Ok(output) => {
                let content = contents_for(&output);
                let result = CallToolResult {
                    content,
                    is_error: output.is_error(),
                };
                JsonRpcResponse::success(id, result)
            }
            Err(e) => {
                let result = CallToolResult {
                    content: vec![Content::Text(TextContent {
                        text: e.to_string(),
                        annotations: None,
                    })],
                    is_error: true,
                };
                JsonRpcResponse::success(id, result)
            }
        }
    }
}

fn cancel_all_in_flight(in_flight: &mut HashMap<RequestId, InFlight>) {
    for (_, call) in in_flight.drain() {
        call.canceller.cancel();
        call.abort.abort();
    }
}

fn parse_call_params(params: Option<serde_json::Value>) -> Result<CallToolParams, JsonRpcError> {
    match params.map(serde_json::from_value).transpose() {
        Ok(Some(params)) => Ok(params),
        Ok(None) => Err(JsonRpcError::invalid_params("Missing params")),
        Err(e) => Err(JsonRpcError::invalid_params(e.to_string())),
    }
}

fn contents_for(output: &ToolResult) -> Vec<Content> {
    match output {
        ToolResult::Parts { parts } => parts.iter().map(content_for_part).collect(),
        output => vec![content_for(output)],
    }
}

fn content_for(output: &ToolResult) -> Content {
    if let ToolResult::Binary { mime, content } = output
        && let Some(image) = image_content(mime, content)
    {
        return image;
    }
    Content::Text(TextContent {
        text: output.render_for_model(),
        annotations: None,
    })
}

fn content_for_part(part: &ToolResultPart) -> Content {
    if let ToolResultPart::Binary { mime, content } = part
        && let Some(image) = image_content(mime, content)
    {
        return image;
    }
    Content::Text(TextContent {
        text: part.render_for_model(),
        annotations: None,
    })
}

fn image_content(mime: &str, content: &[u8]) -> Option<Content> {
    let mime: mime::Mime = mime.parse().ok()?;
    (mime.type_() == mime::IMAGE).then(|| {
        Content::Image(ImageContent {
            data: BASE64.encode(content),
            mime_type: mime.essence_str().to_string(),
            annotations: None,
        })
    })
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::pin::Pin;
    use std::time::Duration;

    use aither_core::llm::tool::{ToolDefinition, ToolDispatchError, ToolResult};
    use serde_json::json;

    use super::*;
    use crate::protocol::ErrorCode;
    use crate::transport::{DuplexTransport, Transport};

    type ToolHandler = dyn Fn(
            &str,
            ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<ToolResult, ToolDispatchError>> + Send>>
        + Send
        + Sync;

    struct DropMarker(async_channel::Sender<()>);

    impl Drop for DropMarker {
        fn drop(&mut self) {
            let _ = self.0.try_send(());
        }
    }

    fn register(tools: &mut Tools, name: &'static str, handler: Box<ToolHandler>) {
        let definition = ToolDefinition::from_parts(
            name.into(),
            "test tool".into(),
            json!({"type": "object", "properties": {}}),
        )
        .expect("valid schema");
        tools.register_dyn(definition, handler).expect("registers");
    }

    fn call(id: i64, name: &str) -> JsonRpcRequest {
        JsonRpcRequest::with_params(
            id,
            "tools/call",
            CallToolParams {
                name: name.to_string(),
                arguments: json!({}),
                meta: None,
            },
        )
    }

    fn call_with_progress(id: i64, name: &str, token: &str) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: RequestId::Number(id),
            method: "tools/call".to_string(),
            params: Some(json!({
                "name": name,
                "arguments": {},
                "_meta": { "progressToken": token },
            })),
        }
    }

    async fn next_message(client: &mut DuplexTransport) -> JsonRpcMessage {
        client.recv().await.expect("recv").expect("connection open")
    }

    async fn next_response(client: &mut DuplexTransport) -> JsonRpcResponse {
        match client.recv().await.expect("recv") {
            Some(JsonRpcMessage::Response(response)) => response,
            other => panic!("expected a response, got {other:?}"),
        }
    }

    async fn initialize_server(client: &mut DuplexTransport) {
        client
            .send_request(JsonRpcRequest::with_params(
                0i64,
                "initialize",
                InitializeParams::default(),
            ))
            .await
            .expect("send initialize");
        let response = next_response(client).await;
        assert!(!response.is_error(), "{response:?}");
    }

    fn progress_params(message: JsonRpcMessage) -> serde_json::Value {
        match message {
            JsonRpcMessage::Notification(notif) if notif.method == "notifications/progress" => {
                notif.params.expect("progress params")
            }
            other => panic!("expected notifications/progress, got {other:?}"),
        }
    }

    fn register_slow(
        tools: &mut Tools,
        gate: async_channel::Receiver<()>,
        reported: async_channel::Sender<()>,
    ) {
        register(
            tools,
            "slow",
            Box::new(move |_args, mut cx| {
                let gate = gate.clone();
                let reported = reported.clone();
                Box::pin(async move {
                    cx.report_progress(Progress::new(1.0).with_total(2.0).with_message("first"))
                        .await
                        .map_err(|e| ToolDispatchError::Execution {
                            tool: "slow".to_string(),
                            source: Box::new(e),
                        })?;
                    reported.send(()).await.expect("reported send");
                    gate.recv().await.expect("gate open");
                    cx.report_progress(Progress::new(2.0).with_total(2.0).with_message("second"))
                        .await
                        .map_err(|e| ToolDispatchError::Execution {
                            tool: "slow".to_string(),
                            source: Box::new(e),
                        })?;
                    Ok(ToolResult::text("done"))
                })
            }),
        );
    }

    #[tokio::test]
    async fn rejects_requests_before_initialize_and_duplicate_initialize() {
        let mut tools = Tools::new();
        register(
            &mut tools,
            "echo",
            Box::new(|_args, _cx| Box::pin(async { Ok(ToolResult::text("ok")) })),
        );

        let (mut client, transport) = DuplexTransport::pair();
        let mut server = McpServer::new(transport, tools, "test-server", "0.0.0");
        let server_task = tokio::spawn(async move { server.run().await });

        // Pre-initialize tools/list and tools/call must fail with INVALID_REQUEST.
        client
            .send_request(JsonRpcRequest::new(10, "tools/list"))
            .await
            .expect("send");
        let pre_list = next_response(&mut client).await;
        assert_eq!(
            pre_list.error.as_ref().map(|e| e.code),
            Some(ErrorCode::INVALID_REQUEST)
        );

        client.send_request(call(11, "echo")).await.expect("send");
        let pre_call = next_response(&mut client).await;
        assert_eq!(
            pre_call.error.as_ref().map(|e| e.code),
            Some(ErrorCode::INVALID_REQUEST)
        );

        // Ping is permitted before initialize.
        client
            .send_request(JsonRpcRequest::new(12, "ping"))
            .await
            .expect("send");
        assert!(!next_response(&mut client).await.is_error());

        initialize_server(&mut client).await;

        // Duplicate initialize is rejected.
        client
            .send_request(JsonRpcRequest::with_params(
                13,
                "initialize",
                InitializeParams::default(),
            ))
            .await
            .expect("send");
        let dup_init = next_response(&mut client).await;
        assert_eq!(
            dup_init.error.as_ref().map(|e| e.code),
            Some(ErrorCode::INVALID_REQUEST)
        );

        client.close().await.expect("close");
        server_task.await.expect("join").expect("run");
    }

    #[tokio::test]
    async fn rejects_duplicate_in_flight_request_id_without_clobbering_original_call() {
        let (gate_tx, gate_rx) = async_channel::unbounded::<()>();
        let (started_tx, started_rx) = async_channel::unbounded::<()>();

        let mut tools = Tools::new();
        register(
            &mut tools,
            "hold",
            Box::new(move |_args, _cx| {
                let gate_rx = gate_rx.clone();
                let started_tx = started_tx.clone();
                Box::pin(async move {
                    started_tx.send(()).await.expect("started");
                    gate_rx.recv().await.expect("gate");
                    Ok(ToolResult::text("original-finished"))
                })
            }),
        );

        let (mut client, transport) = DuplexTransport::pair();
        let mut server = McpServer::new(transport, tools, "test-server", "0.0.0");
        let server_task = tokio::spawn(async move { server.run().await });
        initialize_server(&mut client).await;

        client.send_request(call(1, "hold")).await.expect("send");
        started_rx.recv().await.expect("started");

        // Send a duplicate request with ID 1 while the first is still running.
        client
            .send_request(call(1, "hold"))
            .await
            .expect("send dup");
        let dup_resp = next_response(&mut client).await;
        assert_eq!(dup_resp.id, RequestId::Number(1));
        assert_eq!(
            dup_resp.error.as_ref().map(|e| e.code),
            Some(ErrorCode::INVALID_REQUEST)
        );

        // Release the original call; it still completes normally.
        gate_tx.send(()).await.expect("release");
        let orig_resp = next_response(&mut client).await;
        assert_eq!(orig_resp.id, RequestId::Number(1));
        assert!(!orig_resp.is_error());

        client.close().await.expect("close");
        server_task.await.expect("join").expect("run");
    }

    #[tokio::test]
    async fn progress_notifications_precede_the_response() {
        let (gate_tx, gate_rx) = async_channel::unbounded::<()>();
        let (reported_tx, reported_rx) = async_channel::unbounded::<()>();
        let mut tools = Tools::new();
        register_slow(&mut tools, gate_rx, reported_tx);

        let (mut client, transport) = DuplexTransport::pair();
        let mut server = McpServer::new(transport, tools, "test-server", "0.0.0");
        let server_task = tokio::spawn(async move { server.run().await });
        initialize_server(&mut client).await;

        client
            .send_request(call_with_progress(1, "slow", "tok-1"))
            .await
            .expect("send");
        reported_rx.recv().await.expect("reported");

        assert_eq!(
            progress_params(next_message(&mut client).await),
            json!({
                "progressToken": "tok-1",
                "progress": 1.0,
                "total": 2.0,
                "message": "first",
            })
        );

        gate_tx.send(()).await.expect("gate send");
        assert_eq!(
            progress_params(next_message(&mut client).await),
            json!({
                "progressToken": "tok-1",
                "progress": 2.0,
                "total": 2.0,
                "message": "second",
            })
        );
        match next_message(&mut client).await {
            JsonRpcMessage::Response(response) => assert_eq!(response.id, RequestId::Number(1)),
            other => panic!("expected the call's response, got {other:?}"),
        }

        client.close().await.expect("close");
        server_task.await.expect("join").expect("run");
    }

    #[tokio::test]
    async fn no_progress_notifications_without_a_token() {
        let (gate_tx, gate_rx) = async_channel::unbounded::<()>();
        let (reported_tx, reported_rx) = async_channel::unbounded::<()>();
        let mut tools = Tools::new();
        register_slow(&mut tools, gate_rx, reported_tx);

        let (mut client, transport) = DuplexTransport::pair();
        let mut server = McpServer::new(transport, tools, "test-server", "0.0.0");
        let server_task = tokio::spawn(async move { server.run().await });
        initialize_server(&mut client).await;

        client.send_request(call(1, "slow")).await.expect("send");
        reported_rx.recv().await.expect("reported");
        gate_tx.send(()).await.expect("gate send");

        match next_message(&mut client).await {
            JsonRpcMessage::Response(response) => assert_eq!(response.id, RequestId::Number(1)),
            other => panic!("expected only the response, got {other:?}"),
        }
        assert!(future::poll_once(client.recv()).await.is_none());

        client.close().await.expect("close");
        server_task.await.expect("join").expect("run");
    }

    #[tokio::test]
    async fn progress_after_the_response_is_dropped() {
        let (late_gate_tx, late_gate_rx) = async_channel::unbounded::<()>();
        let (late_done_tx, late_done_rx) = async_channel::unbounded::<()>();
        let mut tools = Tools::new();
        register(
            &mut tools,
            "detached",
            Box::new(move |_args, mut cx| {
                let late_gate_rx = late_gate_rx.clone();
                let late_done_tx = late_done_tx.clone();
                tokio::spawn(async move {
                    late_gate_rx.recv().await.expect("late gate open");
                    cx.report_progress(Progress::new(1.0))
                        .await
                        .expect("valid report");
                    late_done_tx.send(()).await.expect("late done send");
                });
                Box::pin(async move { Ok(ToolResult::text("answered")) })
            }),
        );

        let (mut client, transport) = DuplexTransport::pair();
        let mut server = McpServer::new(transport, tools, "test-server", "0.0.0");
        let server_task = tokio::spawn(async move { server.run().await });
        initialize_server(&mut client).await;

        client
            .send_request(call_with_progress(1, "detached", "tok-late"))
            .await
            .expect("send");
        assert_eq!(next_response(&mut client).await.id, RequestId::Number(1));

        late_gate_tx.send(()).await.expect("late gate send");
        late_done_rx.recv().await.expect("late report handed over");

        client
            .send_request(JsonRpcRequest::new(2, "tools/list"))
            .await
            .expect("send");
        assert_eq!(next_response(&mut client).await.id, RequestId::Number(2));
        assert!(future::poll_once(client.recv()).await.is_none());

        client.close().await.expect("close");
        server_task.await.expect("join").expect("run");
    }

    #[tokio::test]
    async fn tool_calls_run_concurrently() {
        let (gate_tx, gate_rx) = async_channel::unbounded::<()>();

        let mut tools = Tools::new();
        register(
            &mut tools,
            "wait",
            Box::new(move |_args, _cx| {
                let gate_rx = gate_rx.clone();
                Box::pin(async move {
                    gate_rx.recv().await.expect("gate open");
                    Ok(ToolResult::text("waited"))
                })
            }),
        );
        register(
            &mut tools,
            "release",
            Box::new(move |_args, _cx| {
                let gate_tx = gate_tx.clone();
                Box::pin(async move {
                    gate_tx.send(()).await.expect("gate send");
                    Ok(ToolResult::text("released"))
                })
            }),
        );

        let (mut client, transport) = DuplexTransport::pair();
        let mut server = McpServer::new(transport, tools, "test-server", "0.0.0");
        let server_task = tokio::spawn(async move { server.run().await });
        initialize_server(&mut client).await;

        client.send_request(call(1, "wait")).await.expect("send");
        client.send_request(call(2, "release")).await.expect("send");

        assert_eq!(next_response(&mut client).await.id, RequestId::Number(2));
        assert_eq!(next_response(&mut client).await.id, RequestId::Number(1));

        client.close().await.expect("close");
        server_task.await.expect("join").expect("run");
    }

    #[tokio::test]
    async fn cancelled_call_fires_context_cancellation_and_is_dropped() {
        let (gate_tx, gate_rx) = async_channel::unbounded::<()>();
        let (started_tx, started_rx) = async_channel::unbounded::<()>();
        let (dropped_tx, dropped_rx) = async_channel::unbounded::<()>();
        let (signal_fired_tx, signal_fired_rx) = async_channel::unbounded::<()>();

        let mut tools = Tools::new();
        register(
            &mut tools,
            "wait",
            Box::new(move |_args, cx| {
                let gate_rx = gate_rx.clone();
                let started_tx = started_tx.clone();
                let dropped_tx = dropped_tx.clone();
                let signal_fired_tx = signal_fired_tx.clone();
                let signal = cx.cancellation().clone();
                tokio::spawn(async move {
                    signal.cancelled().await;
                    let _ = signal_fired_tx.send(()).await;
                });
                Box::pin(async move {
                    let _marker = DropMarker(dropped_tx);
                    started_tx.send(()).await.expect("started send");
                    gate_rx.recv().await.expect("gate open");
                    Ok(ToolResult::text("waited"))
                })
            }),
        );
        register(
            &mut tools,
            "release",
            Box::new(move |_args, _cx| {
                let gate_tx = gate_tx.clone();
                Box::pin(async move {
                    gate_tx.send(()).await.expect("gate send");
                    Ok(ToolResult::text("released"))
                })
            }),
        );

        let (mut client, transport) = DuplexTransport::pair();
        let mut server = McpServer::new(transport, tools, "test-server", "0.0.0");
        let server_task = tokio::spawn(async move { server.run().await });
        initialize_server(&mut client).await;

        client.send_request(call(1, "wait")).await.expect("send");
        started_rx.recv().await.expect("started");

        client
            .notify(JsonRpcNotification::with_params(
                "notifications/cancelled",
                CancelledParams {
                    request_id: RequestId::Number(1),
                    reason: None,
                },
            ))
            .await
            .expect("notify");
        client.send_request(call(2, "release")).await.expect("send");

        assert_eq!(next_response(&mut client).await.id, RequestId::Number(2));

        tokio::time::timeout(Duration::from_secs(5), dropped_rx.recv())
            .await
            .expect("call dropped")
            .expect("marker channel");
        tokio::time::timeout(Duration::from_secs(5), signal_fired_rx.recv())
            .await
            .expect("ToolContext::cancellation() fired")
            .expect("signal channel");

        assert!(future::poll_once(client.recv()).await.is_none());

        client.close().await.expect("close");
        server_task.await.expect("join").expect("run");
    }

    #[tokio::test]
    async fn image_tool_result_is_image_content() {
        let bytes = vec![0x89, b'P', b'N', b'G', 1, 2, 3];

        let mut tools = Tools::new();
        let image_bytes = bytes.clone();
        register(
            &mut tools,
            "png",
            Box::new(move |_args, _cx| {
                let image_bytes = image_bytes.clone();
                Box::pin(async move { Ok(ToolResult::image(image_bytes, "image/png")) })
            }),
        );

        let (mut client, transport) = DuplexTransport::pair();
        let mut server = McpServer::new(transport, tools, "test-server", "0.0.0");
        let server_task = tokio::spawn(async move { server.run().await });
        initialize_server(&mut client).await;

        client.send_request(call(1, "png")).await.expect("send");
        let response = next_response(&mut client).await;
        assert_eq!(response.id, RequestId::Number(1));

        let result: CallToolResult =
            serde_json::from_value(response.result.expect("result")).expect("call tool result");
        assert!(!result.is_error);
        let [Content::Image(image)] = result.content.as_slice() else {
            panic!(
                "expected exactly one image content item, got {:?}",
                result.content
            );
        };
        assert_eq!(image.mime_type, "image/png");
        let decoded = BASE64.decode(&image.data).expect("base64 decodes");
        assert_eq!(decoded, bytes);

        client.close().await.expect("close");
        server_task.await.expect("join").expect("run");
    }

    #[tokio::test]
    async fn parts_tool_result_becomes_multiple_content_items() {
        let bytes = vec![0x89, b'P', b'N', b'G'];

        let mut tools = Tools::new();
        let image_bytes = bytes.clone();
        register(
            &mut tools,
            "snapshot",
            Box::new(move |_args, _cx| {
                let image_bytes = image_bytes.clone();
                Box::pin(async move {
                    Ok(ToolResult::parts(vec![
                        ToolResultPart::text("screen state"),
                        ToolResultPart::image(image_bytes, "image/png"),
                    ]))
                })
            }),
        );

        let (mut client, transport) = DuplexTransport::pair();
        let mut server = McpServer::new(transport, tools, "test-server", "0.0.0");
        let server_task = tokio::spawn(async move { server.run().await });
        initialize_server(&mut client).await;

        client
            .send_request(call(1, "snapshot"))
            .await
            .expect("send");
        let response = next_response(&mut client).await;

        let result: CallToolResult =
            serde_json::from_value(response.result.expect("result")).expect("call tool result");
        let [Content::Text(text), Content::Image(image)] = result.content.as_slice() else {
            panic!(
                "expected [text, image] content items, got {:?}",
                result.content
            );
        };
        assert_eq!(text.text, "screen state");
        assert_eq!(BASE64.decode(&image.data).expect("base64 decodes"), bytes);

        client.close().await.expect("close");
        server_task.await.expect("join").expect("run");
    }
}
