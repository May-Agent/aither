//! The receive loop: the sole owner of a client's transport.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use arc_swap::ArcSwapOption;
use async_channel::{Receiver, RecvError, Sender, WeakSender};
use futures_lite::future::or;
use futures_util::stream::{FuturesUnordered, StreamExt};
use tracing::{debug, warn};

use super::handler::{McpHandler, ServerNotification};
use crate::protocol::{
    CancelledParams, JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest,
    JsonRpcResponse, ListToolsResult, McpError, McpToolDefinition, RequestId,
};
use crate::transport::BidirectionalTransport;

/// What the client asks the loop to write to the transport.
pub(super) enum Command {
    /// A request; `reply` receives its response, or the error that ended the
    /// connection first.
    Request {
        request: JsonRpcRequest,
        reply: Sender<Result<JsonRpcResponse, McpError>>,
    },
    /// A notification; `reply` receives the outcome of the write.
    Notification {
        notification: JsonRpcNotification,
        reply: Sender<Result<(), McpError>>,
    },
    /// Withdraw a pending request when its caller drops or cancels it.
    Cancel {
        id: RequestId,
        notify: bool,
        reason: Option<String>,
    },
}

impl Command {
    fn fail(self, ending: &Ending) {
        match self {
            Self::Request { reply, .. } => {
                let _ = reply.try_send(Err(ending.error()));
            }
            Self::Notification { reply, .. } => {
                let _ = reply.try_send(Err(ending.error()));
            }
            Self::Cancel { .. } => {}
        }
    }
}

const fn is_fatal_transport_error(error: &McpError) -> bool {
    matches!(error, McpError::ConnectionClosed | McpError::Io(_))
}

/// Why the loop stopped.
enum Ending {
    /// The peer hung up, or the client closed the connection.
    Closed(Option<String>),
    /// Reading or writing the transport failed fatally.
    Failed(String),
}

impl Ending {
    fn error(&self) -> McpError {
        match self {
            Self::Closed(None) => McpError::ConnectionClosed,
            Self::Closed(Some(status)) => {
                McpError::Transport(format!("connection closed: server process {status}"))
            }
            Self::Failed(message) => McpError::Transport(format!("connection failed: {message}")),
        }
    }
}

/// One step of the loop.
enum Event {
    /// A command from the client, or every client handle closing.
    Command(Result<Command, RecvError>),
    /// A message from the server, or the end of the transport.
    Incoming(Result<Option<JsonRpcMessage>, McpError>),
    /// A notification handler or a tool refresh finished.
    Handled(Done),
}

/// The tool list the loop keeps current, and the request ids it draws on.
pub(super) struct ToolCache {
    pub(super) tools: Arc<ArcSwapOption<Vec<McpToolDefinition>>>,
    pub(super) next_id: Arc<AtomicI64>,
}

/// Work the loop polls alongside the server's traffic.
enum Job<H> {
    /// Deliver a notification to the handler.
    Notify(Arc<H>, ServerNotification),
    /// List the server's tools through the loop's own command queue.
    ListTools(WeakSender<Command>, Arc<AtomicI64>),
}

/// The outcome of a [`Job`].
enum Done {
    Notified,
    Listed(Result<Vec<McpToolDefinition>, McpError>),
}

async fn work<H: McpHandler>(job: Job<H>) -> Done {
    match job {
        Job::Notify(handler, notification) => {
            handler.notification(notification).await;
            Done::Notified
        }
        Job::ListTools(commands, next_id) => Done::Listed(list_tools(commands, &next_id).await),
    }
}

async fn list_tools(
    commands: WeakSender<Command>,
    next_id: &AtomicI64,
) -> Result<Vec<McpToolDefinition>, McpError> {
    let (reply, response) = async_channel::bounded(1);
    let request = JsonRpcRequest::new(next_id.fetch_add(1, Ordering::SeqCst), "tools/list");
    commands
        .upgrade()
        .ok_or(McpError::ConnectionClosed)?
        .send(Command::Request { request, reply })
        .await
        .map_err(|_| McpError::ConnectionClosed)?;
    let response = response
        .recv()
        .await
        .map_err(|_| McpError::ConnectionClosed)??;
    let result: ListToolsResult =
        serde_json::from_value(response.into_result().map_err(McpError::JsonRpc)?)?;
    Ok(result.tools)
}

/// Run the connection until the transport ends or the client closes it.
pub(super) async fn run<T, H>(
    mut transport: T,
    commands: Receiver<Command>,
    requests: WeakSender<Command>,
    handler: Arc<H>,
    cache: ToolCache,
    alive: Sender<()>,
) where
    T: BidirectionalTransport,
    H: McpHandler,
{
    let mut pending: HashMap<RequestId, Sender<Result<JsonRpcResponse, McpError>>> = HashMap::new();
    let mut handling = FuturesUnordered::new();
    let mut listing = false;
    let mut listing_again = false;

    let ending = loop {
        let event = or(
            async {
                match handling.next().await {
                    Some(done) => Event::Handled(done),
                    None => std::future::pending().await,
                }
            },
            or(async { Event::Command(commands.recv().await) }, async {
                Event::Incoming(transport.recv().await)
            }),
        )
        .await;

        match event {
            Event::Command(Ok(command)) => {
                if let Err(message) = handle_command(&mut transport, &mut pending, command).await {
                    break Ending::Failed(message);
                }
            }
            Event::Command(Err(_)) | Event::Incoming(Ok(None)) => break Ending::Closed(None),
            Event::Incoming(Err(error)) => break Ending::Failed(error.to_string()),
            Event::Incoming(Ok(Some(JsonRpcMessage::Response(response)))) => {
                if let Some(reply) = pending.remove(&response.id) {
                    let _ = reply.try_send(Ok(response));
                } else {
                    debug!(id = ?response.id, "ignoring response with no pending request");
                }
            }
            Event::Incoming(Ok(Some(JsonRpcMessage::Request(request)))) => {
                if let Err(error) = transport.respond(answer(request)).await {
                    if is_fatal_transport_error(&error) {
                        break Ending::Failed(error.to_string());
                    }
                    warn!(%error, "failed to send server request response");
                }
            }
            Event::Incoming(Ok(Some(JsonRpcMessage::Notification(notification)))) => {
                let notification = ServerNotification::decode(notification);
                if matches!(notification, ServerNotification::ToolsListChanged) {
                    if listing {
                        listing_again = true;
                    } else {
                        listing = true;
                        handling.push(work(Job::ListTools(
                            requests.clone(),
                            cache.next_id.clone(),
                        )));
                    }
                }
                handling.push(work(Job::Notify(handler.clone(), notification)));
            }
            Event::Handled(Done::Notified) => {}
            Event::Handled(Done::Listed(listed)) => {
                listing = false;
                match listed {
                    Ok(tools) => cache.tools.store(Some(Arc::new(tools))),
                    Err(error) => warn!(%error, "re-listing the server's tools failed"),
                }
                if std::mem::take(&mut listing_again) {
                    listing = true;
                    handling.push(work(Job::ListTools(
                        requests.clone(),
                        cache.next_id.clone(),
                    )));
                }
            }
        }
    };

    drop(handling);
    let exit = transport.exit_status().await;
    let ending = match (ending, exit) {
        (Ending::Closed(_), Some(exit)) => Ending::Closed(Some(exit.to_string())),
        (ending, _) => ending,
    };
    let _ = transport.close().await;
    debug!(error = %ending.error(), "MCP connection closed");

    commands.close();
    while let Ok(command) = commands.try_recv() {
        command.fail(&ending);
    }
    for (_, reply) in pending {
        let _ = reply.try_send(Err(ending.error()));
    }
    drop(alive);
}

async fn handle_command<T: BidirectionalTransport>(
    transport: &mut T,
    pending: &mut HashMap<RequestId, Sender<Result<JsonRpcResponse, McpError>>>,
    command: Command,
) -> std::result::Result<(), String> {
    match command {
        Command::Request { request, reply } => {
            let id = request.id.clone();
            pending.insert(id.clone(), reply);
            if let Err(error) = transport.send_request(request).await {
                if is_fatal_transport_error(&error) {
                    return Err(error.to_string());
                }
                if let Some(reply) = pending.remove(&id) {
                    let _ = reply.try_send(Err(error));
                }
            }
        }
        Command::Notification {
            notification,
            reply,
        } => {
            let sent = transport.notify(notification).await;
            let fatal = sent
                .as_ref()
                .err()
                .filter(|err| is_fatal_transport_error(err))
                .map(ToString::to_string);
            let _ = reply.try_send(sent);
            if let Some(message) = fatal {
                return Err(message);
            }
        }
        Command::Cancel { id, notify, reason } => {
            if pending.remove(&id).is_some() && notify {
                let notification = JsonRpcNotification::with_params(
                    "notifications/cancelled",
                    CancelledParams {
                        request_id: id,
                        reason,
                    },
                );
                if let Err(error) = transport.notify(notification).await {
                    if is_fatal_transport_error(&error) {
                        return Err(error.to_string());
                    }
                    warn!(%error, "failed to send notifications/cancelled");
                }
            }
        }
    }
    Ok(())
}

fn answer(request: JsonRpcRequest) -> JsonRpcResponse {
    if request.method == "ping" {
        JsonRpcResponse::success(request.id, serde_json::json!({}))
    } else {
        warn!(method = %request.method, "rejecting a server request for an unadvertised capability");
        JsonRpcResponse::error(request.id, JsonRpcError::method_not_found(&request.method))
    }
}
