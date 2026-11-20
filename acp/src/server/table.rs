//! The session table and per-session scheduling state.
//!
//! Everything here is `pub(crate)` plumbing shared by [`ServerCore`]
//! (dispatch), [`SessionContext`], [`PendingPrompt`], and [`Turn`]:
//! the FIFO prompt queue, the running [`TurnState`], the idempotency
//! map, and the waiters that `session/cancel` and the v2
//! `session/tool_call_cancel`/`session/terminal_cancel` resolve.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::domain::{
    ElicitComplete, ElicitationId, MessageId, PromptFinished, PromptParams, SessionId, TerminalId,
    ToolCallId,
};
use crate::rpc::RpcError;

use super::context::SessionContext;
use super::session::AnySession;

/// How a queued prompt resolves: `Ok` carries the turn-end signal its
/// `session/prompt` request waits on, `Err` is the RPC error reported
/// for prompts that never ran (rejected, cancelled, abandoned).
pub type Finish = Result<PromptFinished, RpcError>;

/// A `session/prompt` accepted into a session's queue but not yet bound
/// to a [`TurnState`].
#[derive(Debug)]
pub struct QueuedItem {
    /// The decoded `session/prompt` params.
    pub params: PromptParams,
    /// The id the agent assigned this prompt's user message at insertion.
    pub message_id: MessageId,
    /// Waiters for this item's resolution: one per `session/prompt`
    /// request (idempotent retries add another).
    finishers: Mutex<Vec<async_channel::Sender<Finish>>>,
    /// `true` once [`complete`](Self::complete) has run.
    done: AtomicBool,
    /// `true` when `PendingPrompt::reject` reported the prompt (the
    /// scheduler skips its own terminal emission).
    rejected: AtomicBool,
    /// The outcome, retained so late idempotent retries resolve at once.
    outcome: Mutex<Option<Finish>>,
}

impl QueuedItem {
    /// An item for `params` with its assigned `message_id`.
    pub(crate) const fn new(params: PromptParams, message_id: MessageId) -> Self {
        Self {
            params,
            message_id,
            finishers: Mutex::new(Vec::new()),
            done: AtomicBool::new(false),
            rejected: AtomicBool::new(false),
            outcome: Mutex::new(None),
        }
    }

    /// Register `tx` for the outcome; resolves it immediately when the
    /// item already completed.
    pub(crate) fn add_finisher(&self, tx: async_channel::Sender<Finish>) {
        let stored = self.outcome.lock().expect("outcome mutex").clone();
        if let Some(finish) = stored {
            let _ = tx.try_send(finish);
        } else {
            self.finishers.lock().expect("finishers mutex").push(tx);
        }
    }

    /// Mark the item rejected (already reported to the client).
    pub(crate) fn mark_rejected(&self) {
        self.rejected.store(true, Ordering::SeqCst);
    }

    /// Whether the item was explicitly rejected.
    pub(crate) fn was_rejected(&self) -> bool {
        self.rejected.load(Ordering::SeqCst)
    }

    /// Resolve every registered finisher with `finish`; idempotent.
    pub(crate) fn complete(&self, finish: &Finish) {
        if self.done.swap(true, Ordering::SeqCst) {
            return;
        }
        *self.outcome.lock().expect("outcome mutex") = Some(finish.clone());
        for tx in self.finishers.lock().expect("finishers mutex").drain(..) {
            let _ = tx.try_send(finish.clone());
        }
    }
}

/// A turn in flight: the scheduler installed this on
/// [`SessionShared::current`] before calling `Session::prompt`, and
/// [`PendingPrompt::accept`](crate::server::PendingPrompt::accept)
/// binds the driving item to it.
#[derive(Debug)]
pub struct TurnState {
    /// Prompts bound to this turn's outcome (the driving item first,
    /// merged steering prompts after).
    bound: Mutex<Vec<std::sync::Arc<QueuedItem>>>,
    /// `accept()` ran.
    accepted: AtomicBool,
    /// The turn is live; requires-action transitions may be emitted.
    open: AtomicBool,
    /// `session/cancel` (or `session/close`) was observed.
    cancelled: AtomicBool,
    /// Signal for [`Turn::cancelled`](crate::server::Turn::cancelled).
    cancel_tx: async_channel::Sender<()>,
    /// Receives the cancel signal.
    cancel_rx: async_channel::Receiver<()>,
    /// Signal for `session/close` waiting on the final updates.
    end_tx: async_channel::Sender<()>,
    /// Receives turn end.
    end_rx: async_channel::Receiver<()>,
    /// Outstanding agent→client requests driving `requires_action`.
    pub(crate) outstanding: AtomicUsize,
}

impl TurnState {
    /// A fresh live turn.
    pub(crate) fn new() -> Self {
        let (cancel_tx, cancel_rx) = async_channel::bounded(1);
        let (end_tx, end_rx) = async_channel::bounded(1);
        Self {
            bound: Mutex::new(Vec::new()),
            accepted: AtomicBool::new(false),
            open: AtomicBool::new(true),
            cancelled: AtomicBool::new(false),
            cancel_tx,
            cancel_rx,
            end_tx,
            end_rx,
            outstanding: AtomicUsize::new(0),
        }
    }

    /// Bind `item` so it resolves with this turn's outcome.
    pub(crate) fn bind(&self, item: std::sync::Arc<QueuedItem>) {
        self.bound.lock().expect("bound mutex").push(item);
    }

    /// Mark the turn accepted (its user message was emitted).
    pub(crate) fn mark_accepted(&self) {
        self.accepted.store(true, Ordering::SeqCst);
    }

    /// Whether `accept()` ran on this turn.
    pub(crate) fn is_accepted(&self) -> bool {
        self.accepted.load(Ordering::SeqCst)
    }

    /// Whether the turn is live.
    pub(crate) fn is_open(&self) -> bool {
        self.open.load(Ordering::SeqCst)
    }

    /// Whether cancellation was observed.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Cancel the turn: flags it and wakes [`Turn::cancelled`] waiters.
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        let _ = self.cancel_tx.try_send(());
    }

    /// Wait for cancellation.
    pub(crate) async fn cancelled(&self) {
        let _ = self.cancel_rx.recv().await;
    }

    /// End the turn: close the open flag, resolve every bound item with
    /// `finish`, and wake `session/close` waiters.
    pub(crate) fn finish(&self, finish: &Finish) {
        self.open.store(false, Ordering::SeqCst);
        for item in self.bound.lock().expect("bound mutex").drain(..) {
            item.complete(finish);
        }
        let _ = self.end_tx.try_send(());
    }

    /// Wait for [`finish`](Self::finish).
    pub(crate) async fn ended(&self) {
        let _ = self.end_rx.recv().await;
    }

    /// Track one agent→client request send; returns whether to emit the
    /// `requires_action` transition.
    pub(crate) fn action_begin(&self) -> bool {
        self.is_open() && self.outstanding.fetch_add(1, Ordering::SeqCst) == 0
    }

    /// Track one agent→client request resolution; returns whether to
    /// emit the `running` transition.
    pub(crate) fn action_end(&self) -> bool {
        let hit_zero = self
            .outstanding
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                (n > 0).then(|| n - 1)
            })
            .is_ok_and(|new| new == 0);
        self.is_open() && hit_zero
    }
}

/// Mutable per-session scheduling state shared by the session entry,
/// the [`SessionContext`], and prompt/turn handles.
#[derive(Debug)]
pub struct SessionShared {
    /// FIFO of inserted-but-unclaimed prompts.
    pub(crate) queue: Mutex<VecDeque<std::sync::Arc<QueuedItem>>>,
    /// The running turn, if any.
    pub(crate) current: Mutex<Option<std::sync::Arc<TurnState>>>,
    /// Idempotency key → the prompt originally inserted under it.
    pub(crate) idempotent: Mutex<HashMap<String, std::sync::Arc<QueuedItem>>>,
    /// Assigned message ids → their items (idempotent retry lookup).
    pub(crate) by_message: Mutex<HashMap<MessageId, std::sync::Arc<QueuedItem>>>,
    /// `session/tool_call_cancel` waiters.
    pub(crate) tool_cancels: Mutex<HashMap<ToolCallId, async_channel::Sender<()>>>,
    /// `session/terminal_cancel` waiters.
    pub(crate) terminal_cancels: Mutex<HashMap<TerminalId, async_channel::Sender<()>>>,
    /// In-flight url-mode elicitations awaiting `elicitation/complete`.
    pub(crate) elicits: Mutex<HashMap<ElicitationId, async_channel::Sender<ElicitComplete>>>,
    /// The session is closed/closing; no new prompts are accepted.
    pub(crate) closed: AtomicBool,
}

impl SessionShared {
    /// Fresh scheduling state for a session.
    pub(crate) fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            current: Mutex::new(None),
            idempotent: Mutex::new(HashMap::new()),
            by_message: Mutex::new(HashMap::new()),
            tool_cancels: Mutex::new(HashMap::new()),
            terminal_cancels: Mutex::new(HashMap::new()),
            elicits: Mutex::new(HashMap::new()),
            closed: AtomicBool::new(false),
        }
    }

    /// Mark closed, cancel the running turn, and fail every queued
    /// prompt with `error`. Returns the running turn, if any.
    pub(crate) fn cancel_all(&self, error: &RpcError) -> Option<std::sync::Arc<TurnState>> {
        self.closed.store(true, Ordering::SeqCst);
        for item in self.queue.lock().expect("queue mutex").drain(..) {
            item.complete(&Err(error.clone()));
        }
        let turn = self.current.lock().expect("current mutex").clone();
        if let Some(turn) = &turn {
            turn.cancel();
        }
        turn
    }

    /// Cancel the running turn and fail queued prompts, without closing.
    /// Returns the running turn, if any.
    pub(crate) fn cancel_turn(&self) -> Option<std::sync::Arc<TurnState>> {
        for item in self.queue.lock().expect("queue mutex").drain(..) {
            item.complete(&Err(RpcError::request_cancelled()));
        }
        let turn = self.current.lock().expect("current mutex").clone();
        if let Some(turn) = &turn {
            turn.cancel();
        }
        turn
    }
}

/// One live session in the server's table.
#[derive(Debug)]
pub struct SessionEntry {
    /// The session id the client addresses.
    pub(crate) id: SessionId,
    /// The user's session object.
    pub(crate) session: AnySession,
    /// Shared scheduling state (also held by `cx` and handles).
    pub(crate) shared: std::sync::Arc<SessionShared>,
    /// The context handed to the host.
    pub(crate) cx: SessionContext,
}
