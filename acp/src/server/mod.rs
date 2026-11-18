//! ACP agent server.
//!
//! [`serve`] and [`serve_stdio`] drive an [`AgentHandler`] over any
//! [`BidirectionalTransport`], mirroring [`AcpClient::connect`](crate::AcpClient::connect)
//! on the agent side. A single connection task owns the transport, polls active
//! handler futures concurrently with inbound reads and outbound writes, and
//! hands a cloneable [`ClientPeer`] to every handler call so turns can stream
//! `session/update` notifications and await correlated client requests (such as
//! `session/request_permission`, `fs/*`, `terminal/*`, and
//! `elicitation/create`) without head-of-line blocking.

#[cfg(feature = "agent")]
mod agent;
mod error;
mod handler;

#[cfg(feature = "agent")]
pub use agent::{AcpServer, SessionSetup};
pub use error::ServerError;
pub use handler::AgentHandler;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::task::{Context, Poll, ready};

use aither_mcp::protocol::{
    JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse, McpError,
    RequestId,
};
use aither_mcp::transport::{BidirectionalTransport, StdioTransport};
use async_channel::{Receiver, Sender};
use futures_lite::future::or;
use futures_lite::stream::Stream;
use futures_util::stream::{FuturesUnordered, StreamExt};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tracing::{debug, warn};

use crate::protocol::{
    ElicitationCompleteParams, ElicitationCreateParams, ElicitationCreateResult, ExtMethod,
    ReadTextFileParams, ReadTextFileResult, RequestPermissionParams, RequestPermissionResult,
    SessionNotification, TerminalCreateParams, TerminalCreateResult, TerminalExitStatus,
    TerminalKillParams, TerminalKillResult, TerminalOutputParams, TerminalOutputResult,
    TerminalReleaseParams, TerminalReleaseResult, TerminalWaitForExitParams, WriteTextFileParams,
    WriteTextFileResult,
};

/// An outbound agent-to-client message plus, for requests, the channel that
/// receives the matching client response.
enum Outbound {
    /// An agent-to-client request; `reply` receives its response.
    Request {
        /// The request to write to the transport.
        request: JsonRpcRequest,
        /// Channel that receives the matching response, or the terminal error.
        reply: Sender<Result<JsonRpcResponse, ServerError>>,
    },
    /// An agent-to-client notification.
    Notification(JsonRpcNotification),
}

/// Serve an [`AgentHandler`] over an existing bidirectional transport.
///
/// Returns a [`ClientPeer`] handle for initiating agent-to-client traffic and
/// the connection future that owns `transport`. The caller must poll the
/// connection future (for example via `tokio::spawn` or `.await`) for messages
/// to flow. Dropping the returned [`ClientPeer`] does **not** close the server;
/// the connection runs until `transport` reaches EOF, encounters a transport
/// error, or [`ClientPeer::close`] is called.
pub fn serve<T, H>(transport: T, handler: H) -> (ClientPeer, impl Future<Output = ()> + Send)
where
    T: BidirectionalTransport,
    H: AgentHandler,
{
    let handler = Arc::new(handler);
    let (peer, outbound_rx) = ClientPeer::channel();
    let driver_peer = peer.clone();
    let connection = async move {
        let mut transport = transport;
        let _ = drive_transport(&mut transport, outbound_rx, driver_peer, handler).await;
    };
    (peer, connection)
}

/// Serve an [`AgentHandler`] over standard input and output.
///
/// Equivalent to `serve(StdioTransport::new(), handler)`.
pub fn serve_stdio<H>(handler: H) -> (ClientPeer, impl Future<Output = ()> + Send)
where
    H: AgentHandler,
{
    serve(StdioTransport::new(), handler)
}

/// Handle to the connected ACP client, passed to [`AgentHandler`] methods and
/// returned by [`serve`].
///
/// Cheaply cloneable and safe to use concurrently from multiple tasks. When the
/// underlying transport closes, every pending request fails immediately with
/// [`ServerError::Closed`] (or [`ServerError::Transport`]) rather than hanging.
#[derive(Debug, Clone)]
pub struct ClientPeer {
    /// Requests and notifications drained by the connection task.
    outbound: Sender<Outbound>,
    /// Request ID counter shared by all clones of this peer handle.
    next_id: Arc<AtomicI64>,
}

impl ClientPeer {
    /// Create a fresh peer handle and the outbound receiver for its driver.
    fn channel() -> (Self, Receiver<Outbound>) {
        let (outbound_tx, outbound_rx) = async_channel::unbounded();
        let peer = Self {
            outbound: outbound_tx,
            next_id: Arc::new(AtomicI64::new(1)),
        };
        (peer, outbound_rx)
    }

    /// Queue a notification without awaiting, used by synchronous callbacks.
    #[cfg(feature = "agent")]
    fn enqueue_notification(&self, notification: JsonRpcNotification) -> Result<(), ServerError> {
        // The outbound channel is unbounded, so `try_send` fails only when the
        // connection has closed.
        self.outbound
            .try_send(Outbound::Notification(notification))
            .map_err(|_| ServerError::Closed)
    }

    /// Send a `session/update` notification to the client.
    ///
    /// # Errors
    ///
    /// Returns [`ServerError::Closed`] if the connection is closed.
    pub async fn session_update(
        &self,
        notification: SessionNotification,
    ) -> Result<(), ServerError> {
        self.notify(JsonRpcNotification::with_params(
            "session/update",
            notification,
        ))
        .await
    }

    /// Send an `elicitation/complete` notification to the client.
    ///
    /// # Errors
    ///
    /// Returns [`ServerError::Closed`] if the connection is closed.
    pub async fn elicitation_complete(
        &self,
        params: ElicitationCompleteParams,
    ) -> Result<(), ServerError> {
        self.notify(JsonRpcNotification::with_params(
            "elicitation/complete",
            params,
        ))
        .await
    }

    /// Send a raw JSON-RPC notification to the client.
    ///
    /// # Errors
    ///
    /// Returns [`ServerError::Closed`] if the connection is closed.
    pub async fn notify(&self, notification: JsonRpcNotification) -> Result<(), ServerError> {
        self.outbound
            .send(Outbound::Notification(notification))
            .await
            .map_err(|_| ServerError::Closed)
    }

    /// Send a `_`-prefixed extension notification to the client.
    ///
    /// # Errors
    ///
    /// Returns [`ServerError::Closed`] if the connection is closed.
    pub async fn ext_notify(
        &self,
        method: impl Borrow<ExtMethod> + Send + Sync,
        params: Option<Value>,
    ) -> Result<(), ServerError> {
        let method_name = method.borrow().as_str();
        let notification = params.map_or_else(
            || JsonRpcNotification::new(method_name),
            |params| JsonRpcNotification::with_params(method_name, params),
        );
        self.notify(notification).await
    }

    /// Send a `session/request_permission` request to the client and await the
    /// user's decision.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection closes or the client replies with an
    /// error or a malformed result.
    pub async fn request_permission(
        &self,
        params: RequestPermissionParams,
    ) -> Result<RequestPermissionResult, ServerError> {
        self.call("session/request_permission", &params).await
    }

    /// Send an `fs/read_text_file` request to the client.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection closes or the client replies with an
    /// error or a malformed result.
    pub async fn read_text_file(
        &self,
        params: ReadTextFileParams,
    ) -> Result<ReadTextFileResult, ServerError> {
        self.call("fs/read_text_file", &params).await
    }

    /// Send an `fs/write_text_file` request to the client.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection closes or the client replies with an
    /// error or a malformed result.
    pub async fn write_text_file(
        &self,
        params: WriteTextFileParams,
    ) -> Result<WriteTextFileResult, ServerError> {
        self.call("fs/write_text_file", &params).await
    }

    /// Send a `terminal/create` request to the client.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection closes or the client replies with an
    /// error or a malformed result.
    pub async fn terminal_create(
        &self,
        params: TerminalCreateParams,
    ) -> Result<TerminalCreateResult, ServerError> {
        self.call("terminal/create", &params).await
    }

    /// Send a `terminal/output` request to the client.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection closes or the client replies with an
    /// error or a malformed result.
    pub async fn terminal_output(
        &self,
        params: TerminalOutputParams,
    ) -> Result<TerminalOutputResult, ServerError> {
        self.call("terminal/output", &params).await
    }

    /// Send a `terminal/wait_for_exit` request to the client.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection closes or the client replies with an
    /// error or a malformed result.
    pub async fn terminal_wait_for_exit(
        &self,
        params: TerminalWaitForExitParams,
    ) -> Result<TerminalExitStatus, ServerError> {
        self.call("terminal/wait_for_exit", &params).await
    }

    /// Send a `terminal/kill` request to the client.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection closes or the client replies with an
    /// error or a malformed result.
    pub async fn terminal_kill(
        &self,
        params: TerminalKillParams,
    ) -> Result<TerminalKillResult, ServerError> {
        self.call("terminal/kill", &params).await
    }

    /// Send a `terminal/release` request to the client.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection closes or the client replies with an
    /// error or a malformed result.
    pub async fn terminal_release(
        &self,
        params: TerminalReleaseParams,
    ) -> Result<TerminalReleaseResult, ServerError> {
        self.call("terminal/release", &params).await
    }

    /// Send an `elicitation/create` request to the client.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection closes or the client replies with an
    /// error or a malformed result.
    pub async fn elicitation_create(
        &self,
        params: ElicitationCreateParams,
    ) -> Result<ElicitationCreateResult, ServerError> {
        self.call("elicitation/create", &params).await
    }

    /// Send a `_`-prefixed extension request to the client and return its raw
    /// JSON result.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection closes or the client replies with an
    /// error or a malformed result.
    pub async fn ext_request(
        &self,
        method: impl Borrow<ExtMethod> + Send + Sync,
        params: Option<Value>,
    ) -> Result<Value, ServerError> {
        let method_name = method.borrow().as_str().to_owned();
        let id = RequestId::Number(self.next_id.fetch_add(1, Ordering::SeqCst));
        let request = match params {
            Some(params) => JsonRpcRequest::with_params(id, &method_name, params),
            None => JsonRpcRequest::new(id, &method_name),
        };
        self.enqueue_request(method_name, request).await?.await
    }

    /// Send a JSON-RPC request for an arbitrary method and deserialize its
    /// result.
    ///
    /// Prefer the typed methods and [`ext_request`](Self::ext_request); this
    /// exists for vendor methods that predate the `_` naming rule.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection closes or the client replies with an
    /// error or a malformed result.
    pub async fn request<P, R>(&self, method: &str, params: &P) -> Result<R, ServerError>
    where
        P: Serialize + Sync,
        R: DeserializeOwned,
    {
        self.call(method, params).await
    }

    /// Send a JSON-RPC request for an arbitrary method and return a future
    /// resolving with its deserialized result.
    ///
    /// # Errors
    ///
    /// Returns [`ServerError::Closed`] if the connection is closed before the
    /// request is queued.
    pub async fn start_request<P, R>(
        &self,
        method: &str,
        params: &P,
    ) -> Result<ServerResponseFuture<R>, ServerError>
    where
        P: Serialize + Sync,
        R: DeserializeOwned,
    {
        let id = RequestId::Number(self.next_id.fetch_add(1, Ordering::SeqCst));
        let request = JsonRpcRequest::with_params(id, method, params);
        self.enqueue_request(method.to_string(), request).await
    }

    /// Close the server connection.
    ///
    /// Closes the outbound channel so the connection task stops accepting new
    /// messages, fails any pending requests, closes the transport, and exits.
    pub fn close(&self) {
        self.outbound.close();
    }

    async fn enqueue_request<R>(
        &self,
        method: String,
        request: JsonRpcRequest,
    ) -> Result<ServerResponseFuture<R>, ServerError> {
        let (reply_tx, reply_rx) = async_channel::bounded(1);
        self.outbound
            .send(Outbound::Request {
                request,
                reply: reply_tx,
            })
            .await
            .map_err(|_| ServerError::Closed)?;

        Ok(ServerResponseFuture {
            method,
            reply: Box::pin(reply_rx),
            marker: PhantomData,
        })
    }

    async fn call<P, R>(&self, method: &str, params: &P) -> Result<R, ServerError>
    where
        P: Serialize + Sync,
        R: DeserializeOwned,
    {
        self.start_request(method, params).await?.await
    }
}

/// The response half of a request sent with [`ClientPeer::start_request`].
///
/// Awaiting it yields the deserialized result, a client-reported JSON-RPC
/// error, or [`ServerError::Closed`] / [`ServerError::Transport`] when the
/// connection drops first.
#[derive(Debug)]
pub struct ServerResponseFuture<R> {
    /// Method name, for error context.
    method: String,
    /// Receives the matching response from the connection task.
    reply: Pin<Box<Receiver<Result<JsonRpcResponse, ServerError>>>>,
    /// Result type without owning a value.
    marker: PhantomData<fn() -> R>,
}

impl<R: DeserializeOwned> Future for ServerResponseFuture<R> {
    type Output = Result<R, ServerError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let response = match ready!(this.reply.as_mut().poll_next(cx)) {
            Some(response) => response?,
            None => return Poll::Ready(Err(ServerError::Closed)),
        };
        Poll::Ready(
            response
                .into_result()
                .map_err(ServerError::from)
                .and_then(|value| {
                    serde_json::from_value(value).map_err(|error| {
                        ServerError::Protocol(format!("malformed {} result: {error}", this.method))
                    })
                }),
        )
    }
}

/// Drives a bidirectional ACP server transport until EOF, [`ClientPeer::close`],
/// or a fatal transport error.
async fn drive_transport<T, H>(
    transport: &mut T,
    outbound: Receiver<Outbound>,
    peer: ClientPeer,
    handler: Arc<H>,
) -> Result<(), ServerError>
where
    T: BidirectionalTransport,
    H: AgentHandler,
{
    /// One step of the server connection loop.
    enum Event {
        /// An outbound request or notification from a [`ClientPeer`].
        Outbound(Result<Outbound, async_channel::RecvError>),
        /// An in-flight handler future completed (producing `Some(response)`
        /// for requests or `None` for notifications).
        Handled(Option<JsonRpcResponse>),
        /// An inbound message from the client, or transport EOF / error.
        Incoming(Result<Option<JsonRpcMessage>, McpError>),
    }

    let mut pending: HashMap<RequestId, Sender<Result<JsonRpcResponse, ServerError>>> =
        HashMap::new();
    let mut in_flight: FuturesUnordered<
        Pin<Box<dyn Future<Output = Option<JsonRpcResponse>> + Send>>,
    > = FuturesUnordered::new();

    let outcome = loop {
        let event = or(
            async { Event::Outbound(outbound.recv().await) },
            or(
                async {
                    match in_flight.next().await {
                        Some(response) => Event::Handled(response),
                        // An empty `FuturesUnordered` resolves to `None`
                        // immediately; park instead of spinning the loop.
                        None => std::future::pending::<Event>().await,
                    }
                },
                async { Event::Incoming(transport.recv().await) },
            ),
        )
        .await;

        match event {
            Event::Outbound(Ok(item)) => {
                if let Err(error) = send_outbound(transport, &mut pending, item).await {
                    break Err(error);
                }
            }
            Event::Outbound(Err(_)) | Event::Incoming(Ok(None)) => {
                break Ok(());
            }
            Event::Handled(Some(response)) => {
                // Flush any notifications or requests queued by the handler
                // before it resolved so causal wire order holds (e.g.
                // `session/update` notifications precede `PromptResult`).
                let mut flush_err = None;
                while let Ok(item) = outbound.try_recv() {
                    if let Err(error) = send_outbound(transport, &mut pending, item).await {
                        flush_err = Some(error);
                        break;
                    }
                }
                if let Some(error) = flush_err {
                    break Err(error);
                }
                if let Err(error) = transport.respond(response).await {
                    break Err(ServerError::Transport(error.to_string()));
                }
            }
            Event::Handled(None) => {}
            Event::Incoming(Ok(Some(JsonRpcMessage::Response(response)))) => {
                if let Some(reply) = pending.remove(&response.id) {
                    let _ = reply.try_send(Ok(response));
                } else {
                    debug!(id = ?response.id, "ignoring response with no pending request");
                }
            }
            Event::Incoming(Ok(Some(JsonRpcMessage::Request(request)))) => {
                let handler = handler.clone();
                let peer = peer.clone();
                in_flight.push(Box::pin(async move {
                    Some(handle_client_request(handler, peer, request).await)
                }));
            }
            Event::Incoming(Ok(Some(JsonRpcMessage::Notification(notification)))) => {
                let handler = handler.clone();
                let peer = peer.clone();
                in_flight.push(Box::pin(async move {
                    handle_client_notification(handler, peer, notification).await;
                    None
                }));
            }
            Event::Incoming(Err(error)) => {
                break Err(ServerError::Transport(error.to_string()));
            }
        }
    };

    outbound.close();
    drop(peer);
    drop(in_flight);
    let _ = transport.close().await;

    let terminal = match &outcome {
        Ok(()) => ServerError::Closed,
        Err(error) => error.clone(),
    };
    debug!(error = %terminal, "ACP server connection closed");

    while let Ok(item) = outbound.try_recv() {
        if let Outbound::Request { reply, .. } = item {
            let _ = reply.try_send(Err(terminal.clone()));
        }
    }
    for (_, reply) in pending {
        let _ = reply.try_send(Err(terminal.clone()));
    }

    outcome
}

/// Write one outbound item to `transport`, recording its reply channel when it
/// is a correlated request.
async fn send_outbound<T: BidirectionalTransport>(
    transport: &mut T,
    pending: &mut HashMap<RequestId, Sender<Result<JsonRpcResponse, ServerError>>>,
    item: Outbound,
) -> Result<(), ServerError> {
    match item {
        Outbound::Request { request, reply } => {
            pending.insert(request.id.clone(), reply);
            transport
                .send_request(request)
                .await
                .map_err(|error| ServerError::Transport(error.to_string()))
        }
        Outbound::Notification(notification) => transport
            .notify(notification)
            .await
            .map_err(|error| ServerError::Transport(error.to_string())),
    }
}

/// Dispatch a client-to-agent request to `handler` and wrap its outcome as a
/// [`JsonRpcResponse`].
async fn handle_client_request<H: AgentHandler>(
    handler: Arc<H>,
    peer: ClientPeer,
    request: JsonRpcRequest,
) -> JsonRpcResponse {
    debug!(method = %request.method, "handling client request");
    match request.method.as_str() {
        "initialize" => {
            dispatch_optional(&request, |params| handler.initialize(&peer, params)).await
        }
        "authenticate" => dispatch(&request, |params| handler.authenticate(&peer, params)).await,
        "logout" => dispatch_default(&request, |params| handler.logout(&peer, params)).await,
        "session/new" => dispatch(&request, |params| handler.new_session(&peer, params)).await,
        "session/load" => dispatch(&request, |params| handler.load_session(&peer, params)).await,
        "session/resume" => {
            dispatch(&request, |params| handler.resume_session(&peer, params)).await
        }
        "session/fork" => dispatch(&request, |params| handler.fork_session(&peer, params)).await,
        "session/list" => {
            dispatch_default(&request, |params| handler.list_sessions(&peer, params)).await
        }
        "session/close" => dispatch(&request, |params| handler.close_session(&peer, params)).await,
        "session/delete" => {
            dispatch(&request, |params| handler.delete_session(&peer, params)).await
        }
        "session/prompt" => dispatch(&request, |params| handler.prompt(&peer, params)).await,
        "session/set_mode" => dispatch(&request, |params| handler.set_mode(&peer, params)).await,
        "session/set_config_option" => {
            dispatch(&request, |params| handler.set_config_option(&peer, params)).await
        }
        _ => handle_custom_request(handler.as_ref(), &peer, request).await,
    }
}

/// Dispatch a `_`-prefixed extension request or an unmodeled vendor request.
async fn handle_custom_request<H: AgentHandler>(
    handler: &H,
    peer: &ClientPeer,
    request: JsonRpcRequest,
) -> JsonRpcResponse {
    if let Ok(method) = ExtMethod::try_new(request.method.clone()) {
        match handler.ext_request(peer, method, request.params).await {
            Ok(result) => JsonRpcResponse::success(request.id, result),
            Err(error) => JsonRpcResponse::error(request.id, error),
        }
    } else {
        let id = request.id.clone();
        match handler.request(peer, request).await {
            Ok(result) => JsonRpcResponse::success(id, result),
            Err(error) => JsonRpcResponse::error(id, error),
        }
    }
}

/// Dispatch a client-to-agent notification to `handler`.
async fn handle_client_notification<H: AgentHandler>(
    handler: Arc<H>,
    peer: ClientPeer,
    notification: JsonRpcNotification,
) {
    match notification.method.as_str() {
        "session/cancel" => match notification.params.map(serde_json::from_value) {
            Some(Ok(params)) => handler.cancel(&peer, params).await,
            Some(Err(error)) => warn!(%error, "malformed session/cancel notification"),
            None => warn!("session/cancel notification without params"),
        },
        "notifications/initialized" => {
            handler.initialized(&peer).await;
        }
        method if method.starts_with('_') => {
            if let Ok(method) = ExtMethod::try_new(notification.method) {
                handler.ext_notify(&peer, method, notification.params).await;
            }
        }
        _ => {
            handler.notification(&peer, notification).await;
        }
    }
}

/// Parse required request params, call `f`, and wrap the result as a JSON-RPC
/// response carrying `request.id`.
async fn dispatch<P, R, F, Fut>(request: &JsonRpcRequest, f: F) -> JsonRpcResponse
where
    P: DeserializeOwned,
    R: Serialize,
    F: FnOnce(P) -> Fut,
    Fut: Future<Output = Result<R, JsonRpcError>>,
{
    let Some(params) = request.params.clone() else {
        return JsonRpcResponse::error(
            request.id.clone(),
            JsonRpcError::invalid_params("missing params"),
        );
    };
    let params: P = match serde_json::from_value(params) {
        Ok(params) => params,
        Err(error) => {
            return JsonRpcResponse::error(
                request.id.clone(),
                JsonRpcError::invalid_params(error.to_string()),
            );
        }
    };
    match f(params).await {
        Ok(result) => JsonRpcResponse::success(request.id.clone(), result),
        Err(error) => JsonRpcResponse::error(request.id.clone(), error),
    }
}

/// Parse optional request params (`None` or `null` -> `None`), call `f`, and
/// wrap the result as a JSON-RPC response.
async fn dispatch_optional<P, R, F, Fut>(request: &JsonRpcRequest, f: F) -> JsonRpcResponse
where
    P: DeserializeOwned,
    R: Serialize,
    F: FnOnce(Option<P>) -> Fut,
    Fut: Future<Output = Result<R, JsonRpcError>>,
{
    let params: Option<P> = match request
        .params
        .clone()
        .filter(|value| !value.is_null())
        .map(serde_json::from_value)
        .transpose()
    {
        Ok(params) => params,
        Err(error) => {
            return JsonRpcResponse::error(
                request.id.clone(),
                JsonRpcError::invalid_params(error.to_string()),
            );
        }
    };
    match f(params).await {
        Ok(result) => JsonRpcResponse::success(request.id.clone(), result),
        Err(error) => JsonRpcResponse::error(request.id.clone(), error),
    }
}

/// Parse defaultable request params (`None` or `null` -> `P::default()`), call
/// `f`, and wrap the result as a JSON-RPC response.
async fn dispatch_default<P, R, F, Fut>(request: &JsonRpcRequest, f: F) -> JsonRpcResponse
where
    P: DeserializeOwned + Default,
    R: Serialize,
    F: FnOnce(P) -> Fut,
    Fut: Future<Output = Result<R, JsonRpcError>>,
{
    let params: P = match request.params.clone().filter(|value| !value.is_null()) {
        None => P::default(),
        Some(value) => match serde_json::from_value(value) {
            Ok(params) => params,
            Err(error) => {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    JsonRpcError::invalid_params(error.to_string()),
                );
            }
        },
    };
    match f(params).await {
        Ok(result) => JsonRpcResponse::success(request.id.clone(), result),
        Err(error) => JsonRpcResponse::error(request.id.clone(), error),
    }
}

#[cfg(test)]
mod tests {
    use super::ClientPeer;

    const fn assert_peer_traits<T: Clone + Send + Sync + std::fmt::Debug>() {}

    #[test]
    fn client_peer_is_clone_send_sync_debug() {
        assert_peer_traits::<ClientPeer>();
    }
}
