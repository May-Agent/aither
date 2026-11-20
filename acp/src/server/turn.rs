//! [`Turn`] — the handle a [`Session::prompt`] implementation drives
//! after [`PendingPrompt::accept`].
//!
//! A `Turn` emits message/thought chunks, tool calls, plans, and
//! arbitrary updates, waits on [`cancelled`](Turn::cancelled), and can
//! absorb queued follow-up prompts ([`try_recv_steered`] /
//! [`merge_prompt`](Turn::merge_prompt)). The turn ends when `prompt()`
//! returns; there is no `Drop`-based completion.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;

use crate::domain::{
    ContentBlock, ContentChunk, Location, MessageId, Notice, NoticeSeverity, Open, Patch,
    PermissionOutcome, PermissionRequest, PlanId, PlanItems, PlanUpdate, SessionUpdate,
    ToolCallCaps, ToolCallContent, ToolCallContentChunk, ToolCallId, ToolCallStatus,
    ToolCallUpdate, ToolKind, patch::EmptyObject,
};
use crate::rpc::RpcError;

use super::context::SessionContext;
use super::prompt::PendingPrompt;
use super::table::{SessionShared, TurnState};

/// A prompt accepted into a running turn.
///
/// Obtained from [`PendingPrompt::accept`]. Methods emit
/// `session/update`s through the session's [`SessionContext`]; the
/// `requires_action` bookkeeping is applied automatically to
/// interactive calls.
#[derive(Debug)]
pub struct Turn {
    id: MessageId,
    state: Arc<TurnState>,
    shared: Arc<SessionShared>,
    cx: SessionContext,
}

impl Turn {
    /// A turn for `id` bound to `state`.
    pub(crate) const fn new(
        id: MessageId,
        state: Arc<TurnState>,
        shared: Arc<SessionShared>,
        cx: SessionContext,
    ) -> Self {
        Self {
            id,
            state,
            shared,
            cx,
        }
    }

    /// The turn id — the accepted prompt's `messageId`.
    #[must_use]
    pub const fn id(&self) -> &MessageId {
        &self.id
    }

    /// The session context this turn runs in.
    #[must_use]
    pub const fn context(&self) -> &SessionContext {
        &self.cx
    }

    /// Whether cancellation was observed (cheap poll; the
    /// [`cancelled`](Self::cancelled) future is the awaitable form).
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.state.is_cancelled()
    }

    /// Resolves when `session/cancel` (or `session/close`) hits this
    /// turn. The turn is expected to end with
    /// [`Err(PromptError::Cancelled)`](crate::server::PromptError) or
    /// `Ok(StopReason::Cancelled)` soon after.
    pub async fn cancelled(&self) {
        self.state.cancelled().await;
    }

    /// An `agent_message` stream with a fresh message id.
    ///
    /// Chunks append to the same message; drop the stream when done.
    #[must_use]
    pub fn message(&self) -> MessageStream {
        MessageStream::new(self.cx.clone(), MessageKind::Agent)
    }

    /// An `agent_thought` stream with a fresh message id.
    #[must_use]
    pub fn thought(&self) -> MessageStream {
        MessageStream::new(self.cx.clone(), MessageKind::Thought)
    }

    /// Begin a tool call; [`ToolCallBuilder::start`] emits it.
    #[must_use]
    pub fn tool_call(&self, name: impl Into<String>, title: impl Into<String>) -> ToolCallBuilder {
        ToolCallBuilder {
            cx: self.cx.clone(),
            update: {
                let mut update = ToolCallUpdate::new(ToolCallId::new());
                update.name = Patch::Set(name.into());
                update.title = Patch::Set(title.into());
                update
            },
        }
    }

    /// Emit a plan (`plan_update`).
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn plan(
        &self,
        plan_id: impl Into<PlanId>,
        entries: Vec<crate::domain::PlanEntry>,
    ) -> Result<(), RpcError> {
        self.cx
            .update(SessionUpdate::PlanUpdate(PlanUpdate {
                plan: Open::Known(crate::domain::Plan::Items(PlanItems {
                    plan_id: plan_id.into(),
                    entries,
                    meta: None,
                    extra: BTreeMap::new(),
                })),
                meta: None,
                extra: BTreeMap::new(),
            }))
            .await
    }

    /// `session/request_permission`, tracked for `requires_action`.
    ///
    /// # Errors
    ///
    /// Propagates the client's error response or a lane failure.
    pub async fn confirm(&self, request: PermissionRequest) -> Result<PermissionOutcome, RpcError> {
        self.cx.request_permission(request).await
    }

    /// Pull the next queued prompt for steering, if any.
    ///
    /// The pulled prompt is removed from the queue — drive it with
    /// [`merge_prompt`](Self::merge_prompt) or `accept`, or `reject`
    /// it. `None` when the queue is empty.
    ///
    /// # Panics
    ///
    /// Panics if the session's prompt-queue mutex is poisoned.
    #[must_use]
    pub fn try_recv_steered(&self) -> Option<PendingPrompt> {
        let item = self.shared.queue.lock().expect("queue").pop_front()?;
        Some(PendingPrompt::steered(
            item,
            self.shared.clone(),
            self.cx.clone(),
        ))
    }

    /// Merge a queued `prompt` into this turn: emits its `user_message`
    /// and binds its outcome to this turn's end.
    ///
    /// Returns the merged prompt's `messageId`.
    ///
    /// # Errors
    ///
    /// [`RpcError`] when the turn already ended or was cancelled — the
    /// prompt is answered with `-32800` before returning.
    pub async fn merge_prompt(&self, prompt: PendingPrompt) -> Result<MessageId, RpcError> {
        prompt
            .accept()
            .await
            .map(|turn| turn.id)
            .map_err(super::error::PromptRejected::into_inner)
    }

    /// A transient `notice` scoped to this session.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn notice(
        &self,
        severity: impl Into<Open<NoticeSeverity>>,
        title: impl Into<String>,
        description: Option<String>,
    ) -> Result<(), RpcError> {
        self.cx
            .update(SessionUpdate::Notice(Notice {
                severity: severity.into(),
                title: title.into(),
                description,
                meta: None,
                extra: BTreeMap::new(),
            }))
            .await
    }

    /// Emit an arbitrary `session/update` (the escape hatch for updates
    /// without a typed helper).
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn update(&self, update: impl Into<SessionUpdate>) -> Result<(), RpcError> {
        self.cx.update(update).await
    }
}

/// Which message stream a [`MessageStream`] appends to.
#[derive(Debug, Clone, Copy)]
enum MessageKind {
    /// `agent_message_chunk`.
    Agent,
    /// `agent_thought_chunk`.
    Thought,
}

/// A streaming message: each call appends a chunk to the same
/// `messageId`. Drop the handle when the message is complete — there is
/// no explicit end marker on the wire.
#[derive(Debug)]
pub struct MessageStream {
    cx: SessionContext,
    message_id: MessageId,
    kind: MessageKind,
}

impl MessageStream {
    fn new(cx: SessionContext, kind: MessageKind) -> Self {
        Self {
            cx,
            message_id: MessageId::new(),
            kind,
        }
    }

    /// The message id all chunks append to.
    #[must_use]
    pub const fn id(&self) -> &MessageId {
        &self.message_id
    }

    /// Append a text chunk.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn text(&self, text: impl AsRef<str>) -> Result<(), RpcError> {
        self.chunk(ContentBlock::text(text.as_ref())).await
    }

    /// Append an arbitrary content block.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn chunk(&self, content: impl Into<ContentBlock>) -> Result<(), RpcError> {
        let chunk = ContentChunk::new(self.message_id.clone(), content.into());
        let update = match self.kind {
            MessageKind::Agent => SessionUpdate::AgentMessageChunk(chunk),
            MessageKind::Thought => SessionUpdate::AgentThoughtChunk(chunk),
        };
        self.cx.update(update).await
    }
}

/// Builder for a tool call, emitted by [`start`](Self::start).
#[derive(Debug)]
pub struct ToolCallBuilder {
    cx: SessionContext,
    update: ToolCallUpdate,
}

impl ToolCallBuilder {
    /// The tool's semantic kind.
    #[must_use]
    pub fn kind(mut self, kind: impl Into<Open<ToolKind>>) -> Self {
        self.update.kind = Patch::Set(kind.into());
        self
    }

    /// A file location the call affects.
    #[must_use]
    pub fn location(mut self, path: impl Into<PathBuf>, line: Option<u32>) -> Self {
        let mut locations = self
            .update
            .locations
            .clone()
            .into_option()
            .unwrap_or_default();
        locations.push(Location {
            path: path.into(),
            line,
            meta: None,
        });
        self.update.locations = Patch::Set(locations);
        self
    }

    /// The call's raw input payload (shown to the client verbatim).
    #[must_use]
    pub fn input(mut self, input: impl Into<Value>) -> Self {
        self.update.raw_input = Patch::Set(input.into());
        self
    }

    /// Declare the call cancellable by the client.
    #[must_use]
    pub fn cancellable(mut self) -> Self {
        self.update.capabilities = Patch::Set(ToolCallCaps {
            cancel: Some(EmptyObject),
            meta: None,
            extra: BTreeMap::new(),
        });
        self
    }

    /// Emit the `tool_call_update` and get the driving handle.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn start(self) -> Result<ToolCallHandle, RpcError> {
        let id = self.update.tool_call_id.clone();
        self.cx
            .update(SessionUpdate::ToolCallUpdate(self.update))
            .await?;
        Ok(ToolCallHandle { cx: self.cx, id })
    }
}

/// Handle on a live tool call, for status transitions and streamed
/// output.
#[derive(Debug)]
pub struct ToolCallHandle {
    cx: SessionContext,
    id: ToolCallId,
}

impl ToolCallHandle {
    /// The call's id.
    #[must_use]
    pub const fn id(&self) -> &ToolCallId {
        &self.id
    }

    /// Set `status`.
    async fn status(&self, status: ToolCallStatus) -> Result<(), RpcError> {
        let mut update = ToolCallUpdate::new(self.id.clone());
        update.status = Patch::Set(Open::Known(status));
        self.cx.update(SessionUpdate::ToolCallUpdate(update)).await
    }

    /// Mark the call in progress.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn in_progress(&self) -> Result<(), RpcError> {
        self.status(ToolCallStatus::InProgress).await
    }

    /// Mark the call paused (a v1-compatible status is not defined; the
    /// update still goes out for clients that understand it).
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn paused(&self) -> Result<(), RpcError> {
        let mut update = ToolCallUpdate::new(self.id.clone());
        update.status = Patch::Set(Open::Unknown(
            "paused".to_string(),
            Value::String("paused".to_string()),
        ));
        self.cx.update(SessionUpdate::ToolCallUpdate(update)).await
    }

    /// Mark the call completed, optionally with `raw_output`.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn complete(&self, raw_output: Option<Value>) -> Result<(), RpcError> {
        let mut update = ToolCallUpdate::new(self.id.clone());
        update.status = Patch::Set(Open::Known(ToolCallStatus::Completed));
        if let Some(output) = raw_output {
            update.raw_output = Patch::Set(output);
        }
        self.cx.update(SessionUpdate::ToolCallUpdate(update)).await
    }

    /// Mark the call failed, optionally with `raw_output`.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn fail(&self, raw_output: Option<Value>) -> Result<(), RpcError> {
        let mut update = ToolCallUpdate::new(self.id.clone());
        update.status = Patch::Set(Open::Known(ToolCallStatus::Failed));
        if let Some(output) = raw_output {
            update.raw_output = Patch::Set(output);
        }
        self.cx.update(SessionUpdate::ToolCallUpdate(update)).await
    }

    /// Append a text output chunk.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn stream_text(&self, text: impl AsRef<str>) -> Result<(), RpcError> {
        self.stream_content(ToolCallContent::Content(crate::domain::tool::Content {
            content: ContentBlock::text(text.as_ref()),
            meta: None,
            extra: BTreeMap::new(),
        }))
        .await
    }

    /// Append an arbitrary `tool_call_content_chunk`.
    ///
    /// # Errors
    ///
    /// Propagates the update lane's error.
    pub async fn stream_content(&self, content: ToolCallContent) -> Result<(), RpcError> {
        self.cx
            .update(SessionUpdate::ToolCallContentChunk(ToolCallContentChunk {
                tool_call_id: self.id.clone(),
                content,
                meta: None,
                extra: BTreeMap::new(),
            }))
            .await
    }

    /// Resolves when the client cancels this call
    /// (`session/tool_call_cancel`). Only meaningful for calls
    /// [declared cancellable](ToolCallBuilder::cancellable).
    ///
    /// # Panics
    ///
    /// Panics if the session's cancel-waiter mutex is poisoned.
    pub async fn cancelled(&self) {
        let (tx, rx) = async_channel::bounded(1);
        self.cx
            .shared()
            .tool_cancels
            .lock()
            .expect("tool_cancels")
            .insert(self.id.clone(), tx);
        let _ = rx.recv().await;
    }
}
