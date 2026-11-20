//! [`PromptTurn`]: the per-turn event stream and outcome.

use std::collections::BTreeMap;
use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures_lite::{FutureExt, Stream, StreamExt};

use crate::domain::{Open, StopReason, Usage};
use crate::rpc::RpcError;

use super::connection::PromptReply;
use super::error::ClientError;
use super::session::ClientSession;
use crate::domain::SessionUpdate;
use crate::state::{MessageKind, SessionEvent};

type PromptFuture =
    Pin<Box<dyn std::future::Future<Output = Result<PromptReply, ClientError>> + Send>>;

/// How a prompt turn resolved.
#[derive(Debug)]
pub enum TurnOutcome {
    /// The turn ended.
    Finished {
        /// Why the agent stopped.
        stop_reason: Option<Open<StopReason>>,
        /// Agent message text accumulated during the turn.
        text: String,
        /// Token usage (v1 result only; on v2 context usage lives at
        /// `state().usage`). Boxed to keep the enum small.
        usage: Option<Box<Usage>>,
        /// The error payload when `stop_reason` is `error`.
        error: Option<RpcError>,
    },
    /// The prompt was rejected before insertion (an error response to
    /// `session/prompt`, other than cancellation).
    Rejected(RpcError),
    /// The turn was cancelled — by `cancel`, by a dropped pending
    /// request, or by the agent reporting `stop_reason: cancelled`.
    Cancelled,
}

/// A running prompt turn: a stream of this session's events from the
/// accept point until the turn ends.
///
/// `poll_next` also drives the pending `session/prompt` response, so the
/// stream works whether or not the user ever calls [`finish`](Self::finish).
pub struct PromptTurn {
    session: ClientSession,
    rx: Pin<Box<async_channel::Receiver<SessionEvent>>>,
    /// The outstanding `session/prompt` call.
    pending: Option<PromptFuture>,
    /// Agent text accumulated this turn.
    text: String,
    /// The id the agent assigned the prompt's user message (v2
    /// `session/prompt` acceptance).
    message_id: Option<crate::domain::MessageId>,
    /// v1 result usage.
    usage: Option<Usage>,
    /// Final outcome once the turn ended.
    outcome: Option<TurnOutcome>,
    /// Set when no more events will be produced.
    done: bool,
}

impl fmt::Debug for PromptTurn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PromptTurn")
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl PromptTurn {
    pub(crate) fn new(
        session: ClientSession,
        rx: Pin<Box<async_channel::Receiver<SessionEvent>>>,
        pending: PromptFuture,
    ) -> Self {
        Self {
            session,
            rx,
            pending: Some(pending),
            text: String::new(),
            message_id: None,
            usage: None,
            outcome: None,
            done: false,
        }
    }

    /// A turn that carries no pending request (v1 `_session/steering`
    /// outcome): streams session events, `finish()` reports `Finished`
    /// with no stop reason once the receiver closes. Rarely what you
    /// want — `steer` uses it for API symmetry.
    pub(crate) fn detached(
        session: ClientSession,
        rx: Pin<Box<async_channel::Receiver<SessionEvent>>>,
    ) -> Self {
        Self {
            session,
            rx,
            pending: None,
            text: String::new(),
            message_id: None,
            usage: None,
            outcome: None,
            done: false,
        }
    }

    /// `session/cancel` for this turn.
    ///
    /// # Errors
    ///
    /// Propagates transport errors.
    pub fn cancel(&self) -> impl std::future::Future<Output = Result<(), ClientError>> + Send + '_ {
        // Borrow `session` rather than `self`: `pending` makes `PromptTurn`
        // `!Sync`, so `&PromptTurn` cannot cross the await.
        self.session.cancel()
    }

    /// Drain the remaining events and return the final outcome.
    pub async fn finish(&mut self) -> TurnOutcome {
        while self.next().await.is_some() {}
        self.outcome.take().unwrap_or(TurnOutcome::Cancelled)
    }

    /// Map a `TurnEnded` event into the outcome.
    fn end_outcome(&self, stop: Option<Open<StopReason>>, error: Option<RpcError>) -> TurnOutcome {
        if matches!(stop, Some(Open::Known(StopReason::Cancelled))) {
            return TurnOutcome::Cancelled;
        }
        TurnOutcome::Finished {
            stop_reason: stop,
            text: self.text.clone(),
            usage: self.usage.clone().map(Box::new),
            error,
        }
    }

    /// Map a failed `session/prompt` response into the outcome.
    fn reply_error(error: ClientError) -> TurnOutcome {
        match error {
            ClientError::Rejected(e) if e.code() == crate::rpc::REQUEST_CANCELLED => {
                TurnOutcome::Cancelled
            }
            ClientError::Rejected(e) | ClientError::Transport(e) => TurnOutcome::Rejected(e),
            ClientError::PeerClosed { status } => TurnOutcome::Rejected(RpcError::internal_error(
                format!("connection closed ({status:?})"),
            )),
            ClientError::Protocol(m) => {
                TurnOutcome::Rejected(RpcError::internal_error(format!("protocol: {m}")))
            }
            ClientError::VersionUnsupported { .. } | ClientError::Unsupported(_) => {
                TurnOutcome::Rejected(RpcError::internal_error(
                    "operation unsupported on this connection",
                ))
            }
        }
    }
}

impl Stream for PromptTurn {
    type Item = SessionEvent;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<SessionEvent>> {
        let this = self.get_mut();
        if this.done {
            return Poll::Ready(None);
        }
        // Drive the pending prompt response.
        if let Some(pending) = &mut this.pending {
            match pending.poll(cx) {
                Poll::Ready(Ok(reply)) => {
                    this.pending = None;
                    match reply {
                        PromptReply::Accepted { message_id } => {
                            // v2: the turn ends at the idle state_update.
                            this.message_id = message_id;
                        }
                        PromptReply::Finished(finished) => {
                            // v1: the agent never reports `running`, so
                            // synthesize it first — the reducer only
                            // produces TurnEnded when the foreground
                            // leaves a known non-idle state for idle.
                            this.usage.clone_from(&finished.usage);
                            this.session.push_update(&SessionUpdate::StateUpdate(
                                crate::domain::StateUpdate {
                                    state: Open::Known(crate::domain::ForegroundState::Running),
                                    stop_reason: None,
                                    error: None,
                                    background: Vec::new(),
                                    meta: None,
                                    extra: BTreeMap::new(),
                                },
                            ));
                            this.session.push_update(&SessionUpdate::StateUpdate(
                                super::connection::v1_turn_end_update(&finished),
                            ));
                        }
                    }
                }
                Poll::Ready(Err(e)) => {
                    this.outcome = Some(Self::reply_error(e));
                    this.done = true;
                    return Poll::Ready(None);
                }
                Poll::Pending => {}
            }
        }
        // Then drain session events.
        match this.rx.as_mut().poll_next(cx) {
            Poll::Ready(Some(event)) => {
                if let SessionEvent::MessageDelta {
                    kind: MessageKind::Agent,
                    text: Some(text),
                    ..
                } = &event
                {
                    this.text.push_str(text);
                }
                if let SessionEvent::TurnEnded { stop_reason, error } = &event {
                    let error = error
                        .as_ref()
                        .and_then(|e| serde_json::from_value(e.clone()).ok());
                    this.outcome = Some(this.end_outcome(stop_reason.clone(), error));
                    this.done = true;
                }
                Poll::Ready(Some(event))
            }
            Poll::Ready(None) => {
                // Every sender dropped: the connection is gone.
                this.outcome.get_or_insert(TurnOutcome::Cancelled);
                this.done = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
