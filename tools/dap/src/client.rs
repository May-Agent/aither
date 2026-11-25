//! Asynchronous DAP JSON-RPC client with request/response correlation and event demuxing.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use async_channel::{Receiver, Sender};
use executor_core::{Executor, Task};
use futures_lite::future;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tracing::{debug, warn};

use crate::error::DapError;
use crate::protocol::{DapEvent, DapRequest, DapResponse, ProtocolMessage};
use crate::transport::DapTransport;

/// Internal command sent to the [`DapClient`] I/O loop.
enum ClientCommand {
    Request {
        request: DapRequest,
        reply: Sender<Result<DapResponse, DapError>>,
    },
    Close {
        done: Sender<()>,
    },
}

/// Low-level DAP JSON-RPC client owning a single background I/O loop.
///
/// Every outgoing request receives a monotonic sequence number and is correlated
/// against adapter responses by `request_seq`. Adapter events are forwarded in
/// arrival order to the returned [`Receiver<DapEvent>`].
#[derive(Debug, Clone)]
pub struct DapClient {
    commands: Sender<ClientCommand>,
    next_seq: Arc<AtomicI64>,
}

impl DapClient {
    /// Connects a new [`DapClient`] over `transport`, spawning its I/O loop on `executor`.
    ///
    /// Returns the client handle paired with the ordered stream of inbound [`DapEvent`]s.
    #[must_use]
    pub fn connect<T, E>(executor: &E, transport: T) -> (Self, Receiver<DapEvent>)
    where
        T: DapTransport + 'static,
        E: Executor,
    {
        let (commands, command_rx) = async_channel::unbounded();
        let (event_tx, event_rx) = async_channel::unbounded();
        let next_seq = Arc::new(AtomicI64::new(1));
        let loop_seq = Arc::clone(&next_seq);

        executor
            .spawn(run_io_loop(transport, command_rx, event_tx, loop_seq))
            .detach();

        (Self { commands, next_seq }, event_rx)
    }

    /// Sends a raw DAP request and awaits the correlated [`DapResponse`].
    ///
    /// Returns [`DapError::RequestFailed`] when the adapter replies with `success: false`.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::ConnectionClosed`] if the transport ended, or
    /// [`DapError::RequestFailed`] if the adapter rejected the request.
    pub async fn request_raw(
        &self,
        command: impl Into<String>,
        arguments: Option<Value>,
    ) -> Result<DapResponse, DapError> {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let command = command.into();
        let (reply_tx, reply_rx) = async_channel::bounded(1);

        self.commands
            .send(ClientCommand::Request {
                request: DapRequest {
                    seq,
                    command: command.clone(),
                    arguments,
                },
                reply: reply_tx,
            })
            .await
            .map_err(|_| DapError::ConnectionClosed)?;

        let response = reply_rx
            .recv()
            .await
            .map_err(|_| DapError::ConnectionClosed)??;

        if response.success {
            Ok(response)
        } else {
            Err(DapError::RequestFailed {
                command,
                message: response
                    .message
                    .unwrap_or_else(|| "adapter reported failure without details".to_owned()),
            })
        }
    }

    /// Sends a typed DAP request and deserializes the response body.
    ///
    /// # Errors
    ///
    /// Returns any serialization, transport, or adapter failure.
    pub async fn request<Args, Body>(
        &self,
        command: &str,
        arguments: &Args,
    ) -> Result<Body, DapError>
    where
        Args: Serialize + Sync,
        Body: DeserializeOwned,
    {
        let args_value = serde_json::to_value(arguments)?;
        let response = self.request_raw(command, Some(args_value)).await?;
        let body_value = response
            .body
            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
        Ok(serde_json::from_value(body_value)?)
    }

    /// Sends a DAP request without arguments and deserializes the response body.
    ///
    /// # Errors
    ///
    /// Returns any transport, adapter, or deserialization failure.
    pub async fn request_empty<Body>(&self, command: &str) -> Result<Body, DapError>
    where
        Body: DeserializeOwned,
    {
        let response = self.request_raw(command, None).await?;
        let body_value = response
            .body
            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
        Ok(serde_json::from_value(body_value)?)
    }

    /// Sends a typed DAP request ignoring the response body.
    ///
    /// # Errors
    ///
    /// Returns any serialization, transport, or adapter failure.
    pub async fn request_unit<Args>(&self, command: &str, arguments: &Args) -> Result<(), DapError>
    where
        Args: Serialize + Sync,
    {
        let args_value = serde_json::to_value(arguments)?;
        let _ = self.request_raw(command, Some(args_value)).await?;
        Ok(())
    }

    /// Closes the underlying transport and stops the I/O loop.
    pub async fn close(&self) {
        let (done_tx, done_rx) = async_channel::bounded(1);
        if self
            .commands
            .send(ClientCommand::Close { done: done_tx })
            .await
            .is_ok()
        {
            let _ = done_rx.recv().await;
        }
    }
}

enum LoopEvent {
    Command(Option<ClientCommand>),
    Incoming(Result<Option<ProtocolMessage>, DapError>),
}

async fn run_io_loop<T: DapTransport>(
    mut transport: T,
    commands: Receiver<ClientCommand>,
    events: Sender<DapEvent>,
    next_seq: Arc<AtomicI64>,
) {
    let mut pending: HashMap<i64, Sender<Result<DapResponse, DapError>>> = HashMap::new();

    loop {
        let next = future::or(
            async { LoopEvent::Command(commands.recv().await.ok()) },
            async { LoopEvent::Incoming(transport.recv().await) },
        )
        .await;

        match next {
            LoopEvent::Command(Some(ClientCommand::Request { request, reply })) => {
                let seq = request.seq;
                debug!(seq, command = %request.command, "DAP TX request");
                if let Err(err) = transport.send(&ProtocolMessage::Request(request)).await {
                    let _ = reply.try_send(Err(err));
                    fail_all_pending(&mut pending);
                    break;
                }
                pending.insert(seq, reply);
            }
            LoopEvent::Command(Some(ClientCommand::Close { done })) => {
                let _ = transport.close().await;
                fail_all_pending(&mut pending);
                let _ = done.try_send(());
                break;
            }
            LoopEvent::Command(None) => {
                let _ = transport.close().await;
                fail_all_pending(&mut pending);
                break;
            }
            LoopEvent::Incoming(Ok(Some(message))) => match message {
                ProtocolMessage::Response(response) => {
                    debug!(
                        request_seq = response.request_seq,
                        command = %response.command,
                        success = response.success,
                        "DAP RX response"
                    );
                    if let Some(reply) = pending.remove(&response.request_seq) {
                        let _ = reply.try_send(Ok(response));
                    } else {
                        warn!(
                            request_seq = response.request_seq,
                            command = %response.command,
                            "received DAP response for unknown request_seq"
                        );
                    }
                }
                ProtocolMessage::Event(event) => {
                    debug!(seq = event.seq, event = %event.event, "DAP RX event");
                    let _ = events.try_send(event);
                }
                ProtocolMessage::Request(reverse_req) => {
                    // Reject unsupported adapter-initiated reverse requests cleanly so
                    // the adapter never blocks waiting for a response.
                    warn!(
                        seq = reverse_req.seq,
                        command = %reverse_req.command,
                        "rejecting unsupported adapter reverse request"
                    );
                    let reply_seq = next_seq.fetch_add(1, Ordering::Relaxed);
                    let response = DapResponse {
                        seq: reply_seq,
                        request_seq: reverse_req.seq,
                        success: false,
                        command: reverse_req.command,
                        message: Some("reverse request is not supported by client".to_owned()),
                        body: None,
                    };
                    if let Err(err) = transport.send(&ProtocolMessage::Response(response)).await {
                        warn!(%err, "failed to send reverse-request rejection");
                        fail_all_pending(&mut pending);
                        break;
                    }
                }
            },
            LoopEvent::Incoming(Ok(None)) => {
                debug!("DAP transport closed by peer (EOF)");
                fail_all_pending(&mut pending);
                break;
            }
            LoopEvent::Incoming(Err(err)) => {
                warn!(%err, "DAP transport read error");
                fail_all_pending(&mut pending);
                break;
            }
        }
    }

    let _ = transport.exit_status().await;
}

fn fail_all_pending(pending: &mut HashMap<i64, Sender<Result<DapResponse, DapError>>>) {
    for (_, reply) in pending.drain() {
        let _ = reply.try_send(Err(DapError::ConnectionClosed));
    }
}
