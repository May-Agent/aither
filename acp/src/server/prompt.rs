//! [`PendingPrompt`] — the decision object handed to
//! [`Session::prompt`](crate::server::Session::prompt) — and
//! [`PromptTicket`], the server's handle on an inserted prompt.

use std::sync::Arc;

use crate::domain::{ContentBlock, ForegroundState, Message, MessageId, SessionId, SessionUpdate};
use crate::rpc::RpcError;

use super::context::SessionContext;
use super::error::PromptRejected;
use super::table::{Finish, QueuedItem, SessionShared, TurnState};
use super::turn::Turn;

/// A `session/prompt` that reached the front of its session's queue.
///
/// The host either [`accept`](Self::accept)s it — yielding a
/// [`Turn`] — or [`reject`](Self::reject)s it. Dropping a
/// `PendingPrompt` without doing either reports `-32800`
/// (`request cancelled`) to the client.
///
/// A prompt pulled by [`Turn::try_recv_steered`] is a *steered* prompt:
/// it has no turn of its own yet — `accept` merges it into the running
/// turn (equivalent to [`Turn::merge_prompt`]) or `reject` dismisses it.
#[derive(Debug)]
pub struct PendingPrompt {
    item: Arc<QueuedItem>,
    shared: Arc<SessionShared>,
    cx: SessionContext,
    /// Pre-installed turn state (driving item); `None` for steered ones.
    turn: Option<Arc<TurnState>>,
    consumed: bool,
}

impl PendingPrompt {
    /// The driving prompt for `item` on the freshly installed `turn`.
    pub(crate) const fn driving(
        item: Arc<QueuedItem>,
        shared: Arc<SessionShared>,
        cx: SessionContext,
        turn: Arc<TurnState>,
    ) -> Self {
        Self {
            item,
            shared,
            cx,
            turn: Some(turn),
            consumed: false,
        }
    }

    /// A steered prompt dequeued by [`Turn::try_recv_steered`].
    pub(crate) const fn steered(
        item: Arc<QueuedItem>,
        shared: Arc<SessionShared>,
        cx: SessionContext,
    ) -> Self {
        Self {
            item,
            shared,
            cx,
            turn: None,
            consumed: false,
        }
    }

    /// The session this prompt belongs to.
    #[must_use]
    pub fn session_id(&self) -> &SessionId {
        &self.item.params.session_id
    }

    /// The id assigned to this prompt's user message at insertion.
    ///
    /// This is the `messageId` already reported to v2 clients.
    #[must_use]
    pub fn message_id(&self) -> &MessageId {
        &self.item.message_id
    }

    /// The prompt content.
    #[must_use]
    pub fn content(&self) -> &[ContentBlock] {
        &self.item.params.prompt
    }

    /// The client-supplied idempotency key, if any.
    #[must_use]
    pub fn idempotency_key(&self) -> Option<&str> {
        self.item.params.idempotency_key.as_deref()
    }

    /// The context for the owning session.
    #[must_use]
    pub const fn context(&self) -> &SessionContext {
        &self.cx
    }

    /// Accept the prompt: emits `user_message` and `state_update`
    /// `running`, binds the prompt to the turn, and returns the
    /// [`Turn`] to drive.
    ///
    /// On a steered prompt this merges into the currently running turn
    /// (see [`Turn::merge_prompt`]).
    ///
    /// # Errors
    ///
    /// [`PromptRejected`] when the prompt or turn was already cancelled —
    /// the client has already been answered.
    ///
    /// # Panics
    ///
    /// Panics if the session's turn-state mutex is poisoned.
    pub async fn accept(mut self) -> Result<Turn, PromptRejected> {
        self.consumed = true;
        let turn = match self.turn.take() {
            Some(turn) => turn,
            None => self
                .shared
                .current
                .lock()
                .expect("current")
                .clone()
                .ok_or_else(|| {
                    self.item.complete(&Err(RpcError::request_cancelled()));
                    PromptRejected::new(RpcError::request_cancelled())
                })?,
        };
        if turn.is_cancelled() || !turn.is_open() {
            self.item.complete(&Err(RpcError::request_cancelled()));
            return Err(PromptRejected::new(RpcError::request_cancelled()));
        }
        turn.mark_accepted();
        turn.bind(self.item.clone());
        // user_message on accept, then running on turn start.
        self.cx
            .update(SessionUpdate::UserMessage(Message::new(
                self.item.message_id.clone(),
                self.item.params.prompt.clone(),
            )))
            .await
            .map_err(PromptRejected::new)?;
        self.cx
            .emit_state(ForegroundState::Running, None, None)
            .await;
        Ok(Turn::new(
            self.item.message_id.clone(),
            turn,
            self.shared.clone(),
            self.cx.clone(),
        ))
    }

    /// Reject the prompt outright.
    ///
    /// On v1 the `session/prompt` response is `error`; on v2 the
    /// response already went out — a `state_update` `idle` with
    /// `stopReason: "error"` is emitted so the client unblocks.
    pub async fn reject(mut self, error: RpcError) {
        self.consumed = true;
        self.item.mark_rejected();
        self.item.complete(&Err(error.clone()));
        // v2: the insertion response was already sent — unblock the
        // client's turn tracking with an explicit error idle. The v1
        // codec drops state updates, so emitting unconditionally is safe.
        self.cx
            .emit_state(
                ForegroundState::Idle,
                Some(crate::domain::Open::Known(crate::domain::StopReason::Error)),
                serde_json::to_value(&error).ok(),
            )
            .await;
    }
}

impl Drop for PendingPrompt {
    fn drop(&mut self) {
        if !self.consumed {
            self.item.complete(&Err(RpcError::request_cancelled()));
        }
    }
}

/// The server's handle on an inserted `session/prompt`.
///
/// [`message_id`](Self::message_id) is the v2 response; the v1 adapter
/// additionally awaits [`finish`](Self::finish) for the turn's
/// `stopReason`.
#[derive(Debug)]
pub struct PromptTicket {
    message_id: MessageId,
    finish: async_channel::Receiver<Finish>,
}

impl PromptTicket {
    /// A ticket that resolves through `finish`.
    pub(crate) const fn new(
        message_id: MessageId,
        finish: async_channel::Receiver<Finish>,
    ) -> Self {
        Self { message_id, finish }
    }

    /// The id reported as `{messageId}` at insertion.
    #[must_use]
    pub const fn message_id(&self) -> &MessageId {
        &self.message_id
    }

    /// Wait for the prompt's resolution.
    ///
    /// `Ok` carries the turn's `stopReason` (an errored turn still
    /// resolves `Ok` — post-acceptance failures are data, not RPC
    /// errors); `Err` is the RPC error for prompts that never ran.
    /// Resolves `internal error` if the turn machinery went away.
    ///
    /// # Errors
    ///
    /// `Err` when the prompt was rejected, cancelled, or its turn
    /// tracking was lost.
    pub async fn finish(&self) -> Finish {
        self.finish
            .recv()
            .await
            .unwrap_or_else(|_| Err(RpcError::internal_error("prompt tracking lost")))
    }
}
