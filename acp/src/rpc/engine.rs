//! The connection engine: one task owns the transport and routes every
//! message in both directions.
//!
//! [`connection`] pairs a [`Peer`] handle with a [`Connection`] future.
//! The connection task runs three sources concurrently — the outbound
//! queue every `Peer` clone writes to, inbound reads, and a
//! `FuturesUnordered` of in-flight handler futures — so a long-running
//! request handler never blocks other traffic.
//!
//! Cancellation travels both ways through `$/cancel_request`: a peer's
//! cancel resolves our pending request with `request cancelled` and drops
//! the in-flight handler future answering `-32800`, while dropping a
//! [`ResponseFuture`] tells the peer to abort our own request.

use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::process::ExitStatus;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, ready};

use async_channel::{Receiver, Sender};
use futures_lite::future::or;
use futures_util::stream::{FuturesUnordered, Stream, StreamExt};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror::Error;
use tracing::{debug, warn};

use super::error::RpcError;
use super::message::{
    CANCEL_REQUEST_METHOD, Inbound, Notification, Outbound, Request, RequestId, Response,
};
use super::transport::Transport;

/// The application side of a connection: answers inbound requests and
/// notifications.
///
/// Handler futures run concurrently inside the connection task — a
/// long-running request (a permission prompt waiting on a human) never
/// blocks other traffic.
pub trait Handler: Send + Sync + 'static {
    /// Handle an inbound request.
    ///
    /// The returned [`Response`] must carry `cx.id`; the engine wires it
    /// to the peer after first draining everything the handler queued on
    /// [`cx.peer`](RequestContext::peer), so a handler's own outbound
    /// messages precede its response on the wire.
    fn request(&self, req: Request, cx: RequestContext) -> impl Future<Output = Response> + Send;

    /// Handle an inbound notification.
    fn notification(&self, n: Notification) -> impl Future<Output = ()> + Send;
}

/// Context handed to [`Handler::request`].
#[derive(Debug)]
pub struct RequestContext {
    /// Id the returned [`Response`] must carry — also the key a peer's
    /// `$/cancel_request` targets.
    pub id: RequestId,
    /// Peer handle for outbound requests, notifications, and extra
    /// responses issued mid-handling.
    pub peer: Peer,
}

/// A handle to the remote end of a connection.
///
/// Clones share one outbound queue drained by the [`Connection`] task, so
/// every method is safe to call concurrently — including from inside
/// [`Handler`] futures via [`RequestContext::peer`].
#[derive(Debug, Clone)]
pub struct Peer {
    /// Queue drained by the connection task.
    outbound: Sender<Envelope>,
    /// Request id counter shared by all clones.
    next_id: Arc<AtomicI64>,
}

/// Why a [`Peer`] request or notification failed.
#[derive(Debug, Clone, Error)]
pub enum PeerError {
    /// The transport itself failed.
    #[error("transport error: {0}")]
    Transport(RpcError),
    /// The connection ended before a response arrived. `status` carries
    /// the peer's exit status when it is a process we spawned.
    #[error("connection closed (exit status: {status:?})")]
    Closed {
        /// Exit status of a process-backed peer, if any.
        status: Option<ExitStatus>,
    },
    /// The peer answered the request with a JSON-RPC error.
    #[error("{0}")]
    Rejected(RpcError),
    /// The response payload could not be deserialized as expected.
    #[error("malformed response: {0}")]
    Malformed(String),
}

/// The response half of a request sent through [`Peer::request`].
///
/// Awaiting it yields the deserialized result, a [`PeerError::Rejected`]
/// carrying the peer's error, [`PeerError::Closed`] when the connection
/// drops first, or [`PeerError::Malformed`] when the result does not fit
/// `R`. Dropping it before it resolves sends `$/cancel_request` to the
/// peer and abandons the request.
pub struct ResponseFuture<R> {
    /// Method name, for error context.
    method: String,
    /// Receives the matching response from the connection task. Pinned
    /// behind a `Box` because `Receiver` is `!Unpin`.
    reply: Pin<Box<Receiver<Result<Response, PeerError>>>>,
    /// While armed, dropping this future cancels the pending request.
    cancel: Option<CancelGuard>,
    /// Result type without owning a value.
    marker: PhantomData<fn() -> R>,
}

impl<R> std::fmt::Debug for ResponseFuture<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponseFuture")
            .field("method", &self.method)
            .finish_non_exhaustive()
    }
}

/// Sends `$/cancel_request` for its request id when dropped armed.
#[derive(Debug)]
struct CancelGuard {
    /// The shared outbound queue.
    outbound: Sender<Envelope>,
    /// Id of the request to cancel.
    id: RequestId,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        let _ = self.outbound.try_send(Envelope::Cancel(self.id.clone()));
    }
}

impl<R: DeserializeOwned> Future for ResponseFuture<R> {
    type Output = Result<R, PeerError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let resolved = ready!(this.reply.as_mut().poll_next(cx));
        // Resolved one way or another — never cancel a finished request.
        this.cancel = None;
        let response = match resolved {
            Some(Ok(response)) => response,
            Some(Err(error)) => return Poll::Ready(Err(error)),
            // The engine dropped the sender without answering.
            None => return Poll::Ready(Err(PeerError::Closed { status: None })),
        };
        Poll::Ready(
            response
                .into_result()
                .map_err(PeerError::Rejected)
                .and_then(|value| {
                    serde_json::from_value(value).map_err(|error| {
                        PeerError::Malformed(format!("malformed {} result: {error}", this.method))
                    })
                }),
        )
    }
}

impl Peer {
    /// Send a request with typed `params` and await its response as `R`.
    ///
    /// A `params` value that serializes to `null` is omitted on the wire.
    /// Dropping the returned future before it resolves sends
    /// `$/cancel_request` to the peer.
    #[must_use]
    pub fn request<P: Serialize, R: DeserializeOwned>(
        &self,
        method: &str,
        params: &P,
    ) -> ResponseFuture<R> {
        match serde_json::to_value(params) {
            Ok(params) => self.start_request(method, (!params.is_null()).then_some(params)),
            Err(error) => ResponseFuture::failed(
                method,
                PeerError::Malformed(format!("failed to serialize {method} params: {error}")),
            ),
        }
    }

    /// Send a request with raw `params` and await its untyped result.
    ///
    /// Dropping the returned future before it resolves sends
    /// `$/cancel_request` to the peer.
    #[must_use]
    pub fn request_untyped(&self, method: &str, params: Option<Value>) -> ResponseFuture<Value> {
        self.start_request(method, params)
    }

    /// Send a notification with typed `params`.
    ///
    /// # Errors
    ///
    /// Returns an error if `params` cannot be serialized or the connection
    /// is already gone.
    pub fn notify<P: Serialize>(
        &self,
        method: &str,
        params: &P,
    ) -> impl Future<Output = Result<(), RpcError>> + Send {
        let result = match serde_json::to_value(params) {
            Ok(params) => self.send_outbound(Outbound::Notification(Notification {
                method: method.to_owned(),
                params: (!params.is_null()).then_some(params),
            })),
            Err(error) => Err(RpcError::internal_error(format!(
                "failed to serialize {method} params: {error}"
            ))),
        };
        std::future::ready(result)
    }

    /// Queue an extra [`Response`] outside the [`Handler::request`] return
    /// path — for answering a request from somewhere other than its
    /// handler future.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection is already gone.
    pub fn respond(&self, resp: Response) -> impl Future<Output = Result<(), RpcError>> + Send {
        std::future::ready(self.send_outbound(Outbound::Response(resp)))
    }

    /// Close the outbound queue. Queued messages still flush; the
    /// connection then ends and every pending request fails with
    /// [`PeerError::Closed`].
    pub fn close(&self) {
        self.outbound.close();
    }

    /// Queue `message` on the outbound channel.
    fn send_outbound(&self, message: Outbound) -> Result<(), RpcError> {
        self.outbound
            .try_send(Envelope::Message(message))
            .map_err(|_| RpcError::internal_error("connection is closed"))
    }

    /// Allocate an id, queue the request, and return its response future.
    fn start_request<R>(&self, method: &str, params: Option<Value>) -> ResponseFuture<R> {
        let id = RequestId::Number(self.next_id.fetch_add(1, Ordering::SeqCst));
        let (reply, rx) = async_channel::bounded(1);
        let request = Request {
            id: id.clone(),
            method: method.to_owned(),
            params,
        };
        let mut cancel = None;
        if let Err(error) = self.outbound.try_send(Envelope::Request { request, reply }) {
            // The connection is gone: recover the reply sender and fail
            // the waiter immediately instead of leaving it queued.
            if let Envelope::Request { reply, .. } = error.into_inner() {
                let _ = reply.try_send(Err(PeerError::Closed { status: None }));
            }
        } else {
            cancel = Some(CancelGuard {
                outbound: self.outbound.clone(),
                id,
            });
        }
        ResponseFuture {
            method: method.to_owned(),
            reply: Box::pin(rx),
            cancel,
            marker: PhantomData,
        }
    }
}

impl<R> ResponseFuture<R> {
    /// A `ResponseFuture` that resolves to `error` immediately — nothing
    /// reached the wire, so no cancellation is armed.
    fn failed(method: &str, error: PeerError) -> Self {
        let (reply, rx) = async_channel::bounded(1);
        let _ = reply.try_send(Err(error));
        Self {
            method: method.to_owned(),
            reply: Box::pin(rx),
            cancel: None,
            marker: PhantomData,
        }
    }
}

/// The running connection: sole owner of the transport and router of
/// every message in both directions.
///
/// Resolves when the peer sends EOF, the transport errors, or every
/// [`Peer`] handle is dropped or [`Peer::close`]d. The peer makes progress
/// only while this future is being polled — spawn it on the caller's
/// executor. On termination it drops in-flight handler futures, collects
/// the peer's exit status, closes the transport, and fails every pending
/// request with [`PeerError::Closed`].
pub struct Connection {
    /// The engine loop, boxed once at construction.
    inner: Pin<Box<dyn Future<Output = ()> + Send>>,
}

impl std::fmt::Debug for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Connection").finish_non_exhaustive()
    }
}

impl Future for Connection {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
    }
}

/// Create a `(Peer, Connection)` pair over `transport`, dispatching
/// inbound traffic to `handler`.
///
/// The returned [`Connection`] must be driven — for example
/// `executor.spawn(connection)` — for the [`Peer`] to make progress.
pub fn connection<T, H>(transport: T, handler: H) -> (Peer, Connection)
where
    T: Transport + 'static,
    H: Handler,
{
    let (outbound_tx, outbound_rx) = async_channel::unbounded();
    let peer = Peer {
        outbound: outbound_tx,
        next_id: Arc::new(AtomicI64::new(1)),
    };
    let inner = Box::pin(drive(
        transport,
        outbound_rx,
        Arc::new(handler),
        peer.clone(),
    ));
    (peer, Connection { inner })
}

/// One item on the outbound queue drained by the connection task.
#[derive(Debug)]
enum Envelope {
    /// Our request; `reply` gets the matching response or the terminal error.
    Request {
        /// The request to write to the transport.
        request: Request,
        /// Channel that receives the matching response.
        reply: Sender<Result<Response, PeerError>>,
    },
    /// A notification or response to write verbatim.
    Message(Outbound),
    /// The local side abandoned a pending request — tell the peer.
    Cancel(RequestId),
}

/// What an in-flight handler future produced.
#[derive(Debug)]
enum Handled {
    /// A request handler finished; `id` also frees its cancellation slot.
    Request {
        /// The answered request's id.
        id: RequestId,
        /// The response to write.
        response: Response,
    },
    /// A notification handler finished.
    Notification,
    /// An inbound response whose turn to resolve has come: every
    /// notification the peer sent before it has been delivered.
    Response(Response),
}

/// One step of the connection loop.
enum Event {
    /// Something a [`Peer`] sent, or all handles closing.
    Outbound(Result<Envelope, async_channel::RecvError>),
    /// Something the peer sent, or the end of the transport.
    Incoming(Result<Option<Inbound>, RpcError>),
    /// A handler future finished.
    Handled(Handled),
}

/// Connection state owned by the single engine task.
struct Engine<T, H> {
    /// The transport, only ever touched from this task.
    transport: T,
    /// Inbound request/notification dispatch target.
    handler: Arc<H>,
    /// Handle cloned into every [`RequestContext`].
    peer: Peer,
    /// The shared outbound queue.
    outbound: Receiver<Envelope>,
    /// Our requests awaiting a response: id -> reply channel.
    pending: HashMap<RequestId, Sender<Result<Response, PeerError>>>,
    /// In-flight inbound handler futures: request id -> cancel handle.
    /// Dropping the sender resolves the handler's race with a `-32800`
    /// response and drops the handler future itself.
    cancellers: HashMap<RequestId, Sender<()>>,
    /// Handler futures run concurrently — a long-running request never
    /// blocks other traffic.
    in_flight: FuturesUnordered<Pin<Box<dyn Future<Output = Handled> + Send>>>,
    /// Notification ordering: lets an inbound response wait until every
    /// notification the peer sent before it has been delivered.
    notifications: Arc<NotificationOrder>,
}

/// Bookkeeping that keeps inbound notifications ahead of the response
/// that arrived after them.
///
/// Notification handlers run concurrently inside `in_flight`, so they
/// finish out of order — but a response must not overtake the
/// notifications that preceded it on the wire (a peer that emits
/// `session/update`s right before its `session/prompt` response expects
/// the client to see the updates first; otherwise turn consumers can
/// close before the trailing chunks land and drop them).
///
/// `spawned` counts notification futures dispatched so far; `done`
/// counts the ones that ran to completion. An inbound response snapshots
/// `spawned` at arrival and its resolution is deferred through
/// `in_flight` until `done` catches up with that snapshot — notifications
/// dispatched *after* the response do not delay it.
#[derive(Debug, Default)]
struct NotificationOrder {
    /// Notification handler futures dispatched so far.
    spawned: AtomicUsize,
    /// Notification handler futures that completed.
    done: AtomicUsize,
    /// One-shot wakers for response futures parked on `done`.
    waiters: Mutex<Vec<Sender<()>>>,
}

impl NotificationOrder {
    /// Record a dispatched notification handler.
    fn track_spawn(&self) {
        self.spawned.fetch_add(1, Ordering::AcqRel);
    }

    /// Record one notification handler's completion and wake every
    /// parked response so each can re-check `done` against its snapshot.
    fn track_done(&self) {
        self.done.fetch_add(1, Ordering::AcqRel);
        for waiter in self.waiters.lock().expect("waiters").drain(..) {
            let _ = waiter.try_send(());
        }
    }

    /// How many notification handlers have been dispatched — the
    /// snapshot an inbound response waits on.
    fn spawned_now(&self) -> usize {
        self.spawned.load(Ordering::Acquire)
    }

    /// Sleep until every notification handler dispatched before the
    /// snapshot was taken has completed. Parks on the `waiters` list —
    /// never spins.
    async fn wait_for(&self, snapshot: usize) {
        loop {
            if self.done.load(Ordering::Acquire) >= snapshot {
                return;
            }
            let (tx, rx) = async_channel::bounded(1);
            self.waiters.lock().expect("waiters").push(tx);
            // Re-check after registering: a completion that landed
            // between the first check and the `push` would otherwise
            // leave this waiter parked forever.
            if self.done.load(Ordering::Acquire) >= snapshot {
                return;
            }
            let _ = rx.recv().await;
        }
    }
}

/// The connection task: drives `transport` until EOF, all [`Peer`] handles
/// closing, or a fatal transport error — then fails everything still in
/// flight so no caller hangs.
async fn drive<T, H>(transport: T, outbound: Receiver<Envelope>, handler: Arc<H>, peer: Peer)
where
    T: Transport,
    H: Handler,
{
    let mut engine = Engine {
        transport,
        handler,
        peer,
        outbound,
        pending: HashMap::new(),
        cancellers: HashMap::new(),
        in_flight: FuturesUnordered::new(),
        notifications: Arc::new(NotificationOrder::default()),
    };
    let outcome = engine.run().await;

    // Shutdown: handler futures were already dropped with `in_flight`.
    // Reap the peer's exit status (killing a still-running child), close
    // the transport, and fail every pending request so no caller hangs.
    engine.outbound.close();
    let status = engine.transport.exit_status().await;
    let _ = engine.transport.close().await;
    let terminal = match outcome {
        Ok(()) => PeerError::Closed { status },
        Err(error) => error,
    };
    debug!(error = %terminal, "connection closed");
    engine.fail_outbound(&terminal);
}

impl<T: Transport, H: Handler> Engine<T, H> {
    /// The select loop over the three event sources.
    async fn run(&mut self) -> Result<(), PeerError> {
        let outcome = loop {
            let event = or(
                or(
                    async { Event::Outbound(self.outbound.recv().await) },
                    async { Event::Incoming(self.transport.recv().await) },
                ),
                async {
                    match self.in_flight.next().await {
                        Some(handled) => Event::Handled(handled),
                        // An empty `FuturesUnordered` resolves to `None`
                        // immediately; park instead of spinning the loop.
                        None => std::future::pending::<Event>().await,
                    }
                },
            )
            .await;

            let result = match event {
                Event::Outbound(Ok(item)) => self.send_envelope(item).await,
                Event::Outbound(Err(_)) | Event::Incoming(Ok(None)) => break Ok(()),
                Event::Incoming(Ok(Some(inbound))) => self.dispatch(inbound).await,
                Event::Incoming(Err(error)) => break Err(PeerError::Transport(error)),
                Event::Handled(handled) => self.handled(handled).await,
            };
            if let Err(error) = result {
                break Err(error);
            }
        };

        // Dropping the in-flight handler futures cancels them.
        drop(std::mem::take(&mut self.in_flight));
        outcome
    }

    /// Write one queued outbound item, recording its reply channel when it
    /// is a request and expanding a local cancellation into a
    /// `$/cancel_request` notification.
    async fn send_envelope(&mut self, item: Envelope) -> Result<(), PeerError> {
        match item {
            Envelope::Request { request, reply } => {
                self.pending.insert(request.id.clone(), reply);
                self.send(&Outbound::Request(request)).await
            }
            Envelope::Message(message) => self.send(&message).await,
            Envelope::Cancel(id) => {
                if self.pending.remove(&id).is_some() {
                    self.send(&Outbound::Notification(Notification::cancel_request(id)))
                        .await
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Route one inbound message: dispatch handler work into `in_flight`,
    /// answer `Invalid` with `-32600`, and resolve matching pending
    /// requests on responses.
    async fn dispatch(&mut self, inbound: Inbound) -> Result<(), PeerError> {
        match inbound {
            Inbound::Request(request) => {
                let id = request.id.clone();
                let (cancel_tx, cancel_rx) = async_channel::bounded(1);
                self.cancellers.insert(id.clone(), cancel_tx);
                let cx = RequestContext {
                    id: id.clone(),
                    peer: self.peer.clone(),
                };
                // The handler future borrows `handler`, so the `Arc` is
                // moved in and `request` is called inside the boxed future
                // to keep it `'static`.
                let handler = self.handler.clone();
                let cancel_id = id.clone();
                self.in_flight.push(Box::pin(async move {
                    let response = or(handler.request(request, cx), async move {
                        // Resolves only when the engine drops the sender:
                        // the request was cancelled in flight.
                        let _ = cancel_rx.recv().await;
                        Response::error(Some(cancel_id), RpcError::request_cancelled())
                    })
                    .await;
                    Handled::Request { id, response }
                }));
                Ok(())
            }
            Inbound::Notification(notification) => {
                if notification.method == CANCEL_REQUEST_METHOD {
                    self.cancel(&notification);
                } else {
                    let handler = self.handler.clone();
                    let order = self.notifications.clone();
                    order.track_spawn();
                    self.in_flight.push(Box::pin(async move {
                        handler.notification(notification).await;
                        order.track_done();
                        Handled::Notification
                    }));
                }
                Ok(())
            }
            Inbound::Response(response) => {
                self.defer_response(response);
                Ok(())
            }
            Inbound::Invalid => {
                self.send(&Outbound::Response(Response::error(
                    None,
                    RpcError::invalid_request(),
                )))
                .await
            }
        }
    }

    /// Apply a peer's `$/cancel_request` to our pending request or to an
    /// in-flight handler future.
    fn cancel(&mut self, notification: &Notification) {
        let Some(id) = notification.params.as_ref().and_then(cancel_target) else {
            warn!(params = ?notification.params, "ignoring $/cancel_request without an id");
            return;
        };
        if let Some(reply) = self.pending.remove(&id) {
            // The peer aborted a request we sent: fail its waiter.
            let _ = reply.try_send(Err(PeerError::Rejected(RpcError::request_cancelled())));
        } else if let Some(canceller) = self.cancellers.remove(&id) {
            // The peer aborted a request it sent: dropping the sender ends
            // the handler race with a `-32800` response.
            drop(canceller);
        } else {
            debug!(id = %id, "ignoring $/cancel_request for an unknown id");
        }
    }

    /// Defer a response behind every notification dispatched before it.
    ///
    /// The waiter runs inside `in_flight` like any handler future so the
    /// loop stays single-owner: it sleeps on [`NotificationOrder`] until
    /// the notification handlers the snapshot covers complete, then
    /// reports [`Handled::Response`], which routes the response to its
    /// pending request.
    fn defer_response(&self, response: Response) {
        let order = self.notifications.clone();
        let snapshot = order.spawned_now();
        self.in_flight.push(Box::pin(async move {
            order.wait_for(snapshot).await;
            Handled::Response(response)
        }));
    }

    /// Route an inbound response to its pending request.
    fn route_response(&mut self, response: Response) {
        let Some(id) = response.id.clone() else {
            warn!("ignoring response with null id");
            return;
        };
        if let Some(reply) = self.pending.remove(&id) {
            let _ = reply.try_send(Ok(response));
        } else {
            debug!(id = %id, "ignoring response with no pending request");
        }
    }

    /// A handler future finished: flush anything it queued so a handler's
    /// own notifications and requests precede its response on the wire,
    /// then write the response. A deferred inbound response routes to
    /// its pending request once its notification snapshot completed.
    async fn handled(&mut self, handled: Handled) -> Result<(), PeerError> {
        match handled {
            Handled::Request { id, response } => {
                self.cancellers.remove(&id);
                while let Ok(item) = self.outbound.try_recv() {
                    self.send_envelope(item).await?;
                }
                self.send(&Outbound::Response(response)).await
            }
            Handled::Notification => Ok(()),
            Handled::Response(response) => {
                self.route_response(response);
                Ok(())
            }
        }
    }

    /// Write one message to the transport, mapping its failure.
    async fn send(&mut self, message: &Outbound) -> Result<(), PeerError> {
        self.transport
            .send(message)
            .await
            .map_err(PeerError::Transport)
    }

    /// Fail every queued and pending request with the terminal error.
    fn fail_outbound(&mut self, terminal: &PeerError) {
        while let Ok(item) = self.outbound.try_recv() {
            if let Envelope::Request { reply, .. } = item {
                let _ = reply.try_send(Err(terminal.clone()));
            }
        }
        for (_, reply) in self.pending.drain() {
            let _ = reply.try_send(Err(terminal.clone()));
        }
    }
}

/// Extract the request id a `$/cancel_request` notification targets:
/// `params.id` by name, or `params` itself when it is a bare id.
fn cancel_target(params: &Value) -> Option<RequestId> {
    let raw = params.get("id").unwrap_or(params);
    serde_json::from_value(raw.clone()).ok()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures_lite::future::zip;
    use serde_json::json;

    use super::super::error::REQUEST_CANCELLED;
    use super::super::transport::Memory;
    use super::*;

    /// A transport scripted with `Inbound` items and recording every
    /// `Outbound` — `Memory` carries only valid `Outbound` lines, so
    /// injecting `Inbound::Invalid` needs typed channels instead.
    #[derive(Debug)]
    struct Scripted {
        /// Scripted inbound stream; EOF when the sender drops.
        rx: Receiver<Inbound>,
        /// Everything the engine writes.
        sent: Sender<Outbound>,
    }

    /// A `Scripted` transport plus the ends to feed and inspect it.
    fn scripted() -> (Scripted, Sender<Inbound>, Receiver<Outbound>) {
        let (in_tx, in_rx) = async_channel::unbounded();
        let (out_tx, out_rx) = async_channel::unbounded();
        (
            Scripted {
                rx: in_rx,
                sent: out_tx,
            },
            in_tx,
            out_rx,
        )
    }

    impl Transport for Scripted {
        async fn recv(&mut self) -> Result<Option<Inbound>, RpcError> {
            Ok(self.rx.recv().await.ok())
        }

        async fn send(&mut self, msg: &Outbound) -> Result<(), RpcError> {
            self.sent
                .send(msg.clone())
                .await
                .map_err(|_| RpcError::internal_error("scripted channel closed"))
        }

        fn close(&mut self) -> impl Future<Output = Result<(), RpcError>> + Send {
            self.sent.close();
            std::future::ready(Ok(()))
        }
    }

    /// Drive `conn` beside `body` until both end; `body` must arrange the
    /// connection's end itself (drop the peer's transport or the peer).
    async fn run(conn: Connection, body: impl Future<Output = ()>) {
        let ((), timed) = zip(conn, tokio::time::timeout(Duration::from_secs(10), body)).await;
        timed.expect("test body timed out");
    }

    /// A handler echoing request params back and reporting notifications.
    struct Echo {
        /// Each notification method seen, in arrival order.
        seen: Sender<String>,
    }

    impl Handler for Echo {
        fn request(
            &self,
            req: Request,
            _: RequestContext,
        ) -> impl Future<Output = Response> + Send {
            std::future::ready(Response::success(req.id, req.params.unwrap_or_default()))
        }

        async fn notification(&self, n: Notification) {
            let _ = self.seen.send(n.method).await;
        }
    }

    /// A handler that parks every request forever.
    struct Park;
    impl Handler for Park {
        async fn request(&self, req: Request, cx: RequestContext) -> Response {
            // Prove traffic flows while this request is in flight.
            cx.peer.notify("during", &json!({})).await.unwrap();
            let _ = req;
            std::future::pending::<Response>().await
        }
        async fn notification(&self, _: Notification) {}
    }

    /// A request round trip: `Peer::request` reaches the peer and its
    /// typed result routes back.
    #[tokio::test]
    async fn request_response() {
        let (a, mut b) = Memory::pair();
        let (peer, conn) = connection(a, Park);
        run(conn, async move {
            let request = peer.request::<_, Value>("echo", &json!({ "x": 1 }));
            let Some(Inbound::Request(req)) = b.recv().await.unwrap() else {
                panic!("expected the outbound request");
            };
            b.send(&Outbound::Response(Response::success(
                req.id.clone(),
                json!({ "echo": true }),
            )))
            .await
            .unwrap();
            assert_eq!(request.await.unwrap(), json!({ "echo": true }));
            drop(b);
        })
        .await;
    }

    /// An in-flight request handler neither blocks its own outbound
    /// traffic nor its `$/cancel_request`: `Park` notifies mid-flight, the
    /// notification arrives, and the cancel drops the future with a
    /// `-32800` answer.
    #[tokio::test]
    async fn in_flight_cancel_drops_the_handler_and_answers() {
        let (a, mut b) = Memory::pair();
        let (_peer, conn) = connection(a, Park);
        run(conn, async move {
            b.send(&Outbound::Request(Request::new("s-1", "slow")))
                .await
                .unwrap();
            // The parked handler's own notification flows while in flight.
            let Some(Inbound::Notification(n)) = b.recv().await.unwrap() else {
                panic!("expected the handler's notification");
            };
            assert_eq!(n.method, "during");

            b.send(&Outbound::Notification(Notification::cancel_request("s-1")))
                .await
                .unwrap();
            let Some(Inbound::Response(response)) = b.recv().await.unwrap() else {
                panic!("expected a response");
            };
            assert_eq!(response.id, Some(RequestId::String("s-1".to_owned())));
            let error = response.into_result().unwrap_err();
            assert_eq!(error.code(), REQUEST_CANCELLED);
            drop(b);
        })
        .await;
    }

    /// `$/cancel_request` from the peer resolves our pending request with
    /// `request cancelled` (-32800).
    #[tokio::test]
    async fn inbound_cancel_fails_our_pending_request() {
        let (a, mut b) = Memory::pair();
        let (peer, conn) = connection(a, Park);
        run(conn, async move {
            let request = peer.request::<_, Value>("echo", &json!({}));
            let Some(Inbound::Request(req)) = b.recv().await.unwrap() else {
                panic!("expected the outbound request");
            };
            b.send(&Outbound::Notification(Notification::cancel_request(
                req.id,
            )))
            .await
            .unwrap();
            match request.await {
                Err(PeerError::Rejected(error)) => {
                    assert_eq!(error.code(), REQUEST_CANCELLED);
                }
                other => panic!("expected Rejected(cancelled), got {other:?}"),
            }
            drop(b);
        })
        .await;
    }

    /// Dropping a `ResponseFuture` sends `$/cancel_request` to the peer.
    #[tokio::test]
    async fn dropping_a_request_cancels_it() {
        let (a, mut b) = Memory::pair();
        let (peer, conn) = connection(a, Park);
        run(conn, async move {
            let request = peer.request::<_, Value>("echo", &json!({}));
            let Some(Inbound::Request(req)) = b.recv().await.unwrap() else {
                panic!("expected the outbound request");
            };
            drop(request);
            let Some(Inbound::Notification(n)) = b.recv().await.unwrap() else {
                panic!("expected a cancel notification");
            };
            assert_eq!(n.method, CANCEL_REQUEST_METHOD);
            assert_eq!(n.params, Some(json!({ "id": req.id })));
            drop(b);
        })
        .await;
    }

    /// A handler's own outbound messages precede its response on the wire.
    #[tokio::test]
    async fn handler_output_precedes_its_response() {
        struct Chatty;
        impl Handler for Chatty {
            async fn request(&self, req: Request, cx: RequestContext) -> Response {
                cx.peer.notify("progress", &json!({})).await.unwrap();
                Response::success(req.id, json!({ "done": true }))
            }
            async fn notification(&self, _: Notification) {}
        }

        let (a, mut b) = Memory::pair();
        let (_peer, conn) = connection(a, Chatty);
        run(conn, async move {
            b.send(&Outbound::Request(Request::new(7_i64, "work")))
                .await
                .unwrap();
            let Some(Inbound::Notification(n)) = b.recv().await.unwrap() else {
                panic!("expected the progress notification first");
            };
            assert_eq!(n.method, "progress");
            let Some(Inbound::Response(response)) = b.recv().await.unwrap() else {
                panic!("expected the response second");
            };
            assert_eq!(response.into_result().unwrap(), json!({ "done": true }));
            drop(b);
        })
        .await;
    }

    /// `Invalid` inbound is answered `-32600` with `id: null`.
    #[tokio::test]
    async fn invalid_input_gets_invalid_request() {
        let (scripted_transport, inbound, sent) = scripted();
        let (_peer, conn) = connection(scripted_transport, Park);
        run(conn, async move {
            inbound.send(Inbound::Invalid).await.unwrap();
            let Outbound::Response(response) = sent.recv().await.unwrap() else {
                panic!("expected a response");
            };
            assert_eq!(response.id, None);
            let error = response.into_result().unwrap_err();
            assert_eq!(error.code(), super::super::error::INVALID_REQUEST);
            drop(inbound);
        })
        .await;
    }

    /// EOF fails every pending request with `Closed` and ends the future.
    #[tokio::test]
    async fn eof_fails_pending_requests() {
        let (a, mut b) = Memory::pair();
        let (peer, conn) = connection(a, Park);
        run(conn, async move {
            let request = peer.request::<_, Value>("echo", &json!({}));
            // Wait for the request to go out so the failure below is the
            // EOF, not a failed write.
            assert!(matches!(b.recv().await.unwrap(), Some(Inbound::Request(_))));
            drop(b); // EOF
            match request.await {
                Err(PeerError::Closed { status: None }) => {}
                other => panic!("expected Closed, got {other:?}"),
            }
        })
        .await;
    }

    /// A response with no matching pending request is ignored, not fatal.
    #[tokio::test]
    async fn stray_responses_are_ignored() {
        let (a, mut b) = Memory::pair();
        let (peer, conn) = connection(a, Park);
        run(conn, async move {
            b.send(&Outbound::Response(Response::success(99_i64, json!(null))))
                .await
                .unwrap();
            // The connection still works: a fresh request round-trips.
            let request = peer.request::<_, Value>("echo", &json!({ "ok": 1 }));
            let Some(Inbound::Request(req)) = b.recv().await.unwrap() else {
                panic!("expected the outbound request");
            };
            b.send(&Outbound::Response(Response::success(
                req.id,
                json!({ "ok": 1 }),
            )))
            .await
            .unwrap();
            assert_eq!(request.await.unwrap(), json!({ "ok": 1 }));
            drop(b);
        })
        .await;
    }

    /// Notifications reach the handler, concurrently with requests.
    #[tokio::test]
    async fn notifications_are_dispatched() {
        let (seen_tx, seen_rx) = async_channel::unbounded();
        let (a, mut b) = Memory::pair();
        let (_peer, conn) = connection(a, Echo { seen: seen_tx });
        run(conn, async move {
            b.send(&Outbound::Notification(Notification::new("note-1")))
                .await
                .unwrap();
            assert_eq!(seen_rx.recv().await.unwrap(), "note-1");
            drop(b);
        })
        .await;
    }

    /// Two engines over a `Memory` pair talk to each other end to end:
    /// `peer_a`'s request is dispatched by engine B to `Park`, which never
    /// answers — `peer_b.respond` resolves it outside the handler path.
    #[tokio::test]
    async fn engine_to_engine_round_trip() {
        let (a, b) = Memory::pair();
        let (peer_a, conn_a) = connection(a, Park);
        let (peer_b, conn_b) = connection(b, Park);
        // Race rather than join: B's parked handler keeps `cx.peer` — and
        // therefore B's outbound channel — alive, so neither connection
        // ends on its own once the assertions have run.
        or(
            conn_a,
            or(conn_b, async move {
                let request = peer_a.request::<_, Value>("echo", &json!({ "v": 2 }));
                peer_b
                    .respond(Response::success(1_i64, json!({ "v": 2 })))
                    .await
                    .unwrap();
                assert_eq!(request.await.unwrap(), json!({ "v": 2 }));
            }),
        )
        .await;
    }

    /// A response must not overtake the notifications dispatched before
    /// it. The scripted peer sends update A, update B, then the prompt
    /// response; with both handlers gated, the request's waiter stays
    /// unresolved — releasing the handlers delivers A and B and only
    /// then does the response land.
    #[tokio::test]
    async fn response_waits_for_earlier_notifications() {
        /// Records each notification after a release token.
        struct Gated {
            /// Delivery order of notifications.
            delivered: Sender<String>,
            /// One token releases one notification handler.
            release: Receiver<()>,
        }
        impl Handler for Gated {
            fn request(
                &self,
                req: Request,
                _: RequestContext,
            ) -> impl Future<Output = Response> + Send {
                std::future::ready(Response::success(req.id, json!(null)))
            }
            async fn notification(&self, n: Notification) {
                let _ = self.release.recv().await;
                let _ = self.delivered.send(n.method).await;
            }
        }

        let (delivered, delivered_rx) = async_channel::unbounded::<String>();
        let (release_tx, release_rx) = async_channel::unbounded::<()>();
        let (transport, inbound, sent) = scripted();
        let (peer, conn) = connection(
            transport,
            Gated {
                delivered,
                release: release_rx,
            },
        );
        run(conn, async move {
            let mut request = peer.request::<_, Value>("session/prompt", &json!({}));
            // The outbound request is on the wire, so a pending entry
            // exists for the response that follows.
            let Some(Outbound::Request(req)) = sent.recv().await.ok() else {
                panic!("expected the outbound request");
            };
            // The peer's wire order: A, B, then the prompt response.
            inbound
                .send(Inbound::Notification(Notification::new("update-a")))
                .await
                .unwrap();
            inbound
                .send(Inbound::Notification(Notification::new("update-b")))
                .await
                .unwrap();
            inbound
                .send(Inbound::Response(Response::success(
                    req.id.clone(),
                    json!({ "messageId": "m-1" }),
                )))
                .await
                .unwrap();
            // Both handlers are still gated: the response must not
            // overtake them.
            tokio::time::sleep(Duration::from_millis(100)).await;
            assert!(
                futures_lite::future::poll_once(&mut request)
                    .await
                    .is_none(),
                "response resolved while earlier notifications were in flight"
            );
            release_tx.send(()).await.unwrap();
            release_tx.send(()).await.unwrap();
            let mut got = [
                delivered_rx.recv().await.unwrap(),
                delivered_rx.recv().await.unwrap(),
            ];
            got.sort_unstable();
            assert_eq!(got, ["update-a", "update-b"]);
            // …and only now does the prompt response resolve.
            assert_eq!(request.await.unwrap(), json!({ "messageId": "m-1" }));
            drop(inbound);
        })
        .await;
    }
}
