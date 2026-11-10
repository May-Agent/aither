//! Core agent implementation.
//!
//! The `Agent` struct is the main entry point for the agent framework.
//! It manages the context window, forks compaction in the background, and
//! handles tool execution in an agent-controlled loop.

use std::collections::VecDeque;
use std::ops::ControlFlow;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::Poll;
use std::time::{Duration, Instant};

#[cfg(feature = "skills")]
use crate::context::SkillCatalogEntry;
use crate::{
    checkpoint::AgentCheckpoint,
    compaction_state::CompactionState,
    config::AgentConfig,
    context::{
        AnyContextSource, AnyFileReferences, AnyLiveSource, AttachmentRef, Block, CacheLifetime,
        CompactionFork, CompletedCompaction, Context, ContextBlock, ContextBudget, HandoffDocument,
        Moment, PendingCompaction, Placement, PrefixChain, PrefixRewrites, PromptCache,
        PromptPrefix, SourceId,
        recovery::{SpillLimits, clean_for_recovery},
        restore_context,
    },
    context_window::{ContextWindowMetrics, ContextWindowPhase, ContextWindowSnapshot},
    error::AgentError,
    event::{
        AgentEvent, AgentPath, BlockId, CheckpointReason, CompactionOutcome, CompactionTrigger,
        InjectionKind, OutputKind, PauseId, RequestId, RequestOutcome, RunId, RunOutcome, RunPause,
        TimedEvent, ToolOrigin, ToolVerdict,
    },
    hook::{
        AfterTool, Hooks, RequestCx, StopCx, StopDecision, ToolDecision, ToolInvocation,
        ToolOutcome, TurnBoundaryAction, TurnBoundaryContext,
    },
    request_parameters::WithParameters,
    tools::AgentTools,
};
use aither_core::{
    LanguageModel,
    llm::{
        AnyApprover, Approval, ApprovalRequest, Approver, ApproverFailure, Attachment,
        CancelSignal, Canceller, Event, LLMRequest, Message, ReasoningState, RetryPolicy, Role,
        TokenCounting, ToolCall, ToolCallId,
        model::{Ability, Parameters, Profile as ModelProfile},
        tool::{Progress, ProgressSink},
    },
};
use askama::Template;
use futures_core::Stream;
use futures_lite::StreamExt;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::PathBuf;

/// Which of the two futures raced during a turn finished first.
enum TurnStep {
    /// The model turn completed.
    Turn(Result<ModelTurn, AgentError>),
    /// The forked compaction landed while the turn was still running.
    Compacted(Result<CompletedCompaction, crate::context::CompactionError>),
}

/// The number of characters an assembled request renders to, tool calls
/// and tool definitions included — the provider counts all of them.
fn request_chars(
    messages: &[Message],
    definitions: &[aither_core::llm::tool::ToolDefinition],
) -> usize {
    let messages: usize = messages
        .iter()
        .map(|message| {
            let calls: usize = message
                .tool_calls()
                .iter()
                .map(|call| call.name.len() + call.arguments.to_string().len())
                .sum();
            message.content().len() + calls
        })
        .sum();
    let definitions: usize = definitions
        .iter()
        .map(|definition| {
            definition.name().len()
                + definition.description().len()
                + definition.arguments_openai_schema().to_string().len()
        })
        .sum();
    messages + definitions
}

struct ModelTurn {
    response_text: String,
    tool_calls: Vec<ToolCall>,
    /// Opaque provider reasoning produced this turn, in the order it arrived.
    ///
    /// Replayed on the next request: providers that verify their own reasoning
    /// reject a turn whose blocks were reordered or partially dropped.
    reasoning: Vec<ReasoningState>,
    /// What the provider reported the request used, if it did.
    usage: Option<aither_core::llm::Usage>,
    malformed_function_call: bool,
}

/// Streams agent events to the consumer as they happen.
///
/// Wraps an unbounded channel: emitting never blocks, and events emitted
/// after the consumer dropped the stream are discarded.
#[derive(Clone)]
struct EventSink {
    sender: async_channel::Sender<TimedEvent>,
    /// The agent the sink emits for, stamped on every event.
    agent: AgentPath,
}

/// What the loop waits on while tools run.
enum ToolStep {
    Finished(Option<ToolExecutionResult>),
    Child(Option<TimedEvent>),
}

impl EventSink {
    /// Puts an event another agent stamped on this stream as it is.
    fn forward(&self, event: TimedEvent) {
        let _ = self.sender.try_send(event);
    }

    fn emit(&self, event: AgentEvent) {
        let _ = self.sender.try_send(TimedEvent {
            at: jiff::Timestamp::now(),
            agent: self.agent.clone(),
            event,
        });
    }
}

/// Cancels a [`Run`] from outside, such as from another task while the run is
/// being driven. Cloneable and cheap.
#[derive(Debug, Clone, Default)]
pub struct CancelHandle(Arc<Canceller>);

impl CancelHandle {
    /// Cancels the run: it stops at its next await point and ends with
    /// [`RunOutcome::Cancelled`]. Calling it again, or after the run ended,
    /// does nothing.
    pub fn cancel(&self) {
        self.0.cancel();
    }

    /// Whether the run was cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }

    pub(crate) fn signal(&self) -> CancelSignal {
        self.0.signal()
    }
}

/// One run of an [`Agent`], from [`Agent::run`] or [`Agent::resume`].
///
/// A run does nothing until [`drive`](Self::drive) consumes it. Driving hands
/// each event to the caller's handler while the run itself keeps going, so a
/// handler that waits on something the run holds lets the run release it.
///
/// A run ends with exactly one [`AgentEvent::RunEnded`]: cancelling it with
/// [`cancel`](Self::cancel) or a [`CancelHandle`] ends it with
/// [`RunOutcome::Cancelled`], and so does dropping it unfinished (though a
/// dropped run has no consumer left to read the event).
#[must_use = "a run does nothing until it is driven"]
pub struct Run<F> {
    /// The run's state machine spans every turn and is tens of kilobytes;
    /// boxed so a `Run` moves into the caller's future without copying it.
    body: Pin<Box<F>>,
    events: async_channel::Receiver<TimedEvent>,
    cancel: CancelHandle,
}

impl<F> std::fmt::Debug for Run<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Run")
            .field("queued_events", &self.events.len())
            .field("cancelled", &self.cancel.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl<F> Run<F>
where
    F: Future<Output = Result<(), AgentError>>,
{
    fn new(agent: AgentPath, start: impl FnOnce(EventSink, CancelSignal) -> F) -> Self {
        let (sender, events) = async_channel::unbounded();
        let cancel = CancelHandle::default();
        // The run owns the sink and drops it as it finishes.
        Self {
            body: Box::pin(start(EventSink { sender, agent }, cancel.signal())),
            events,
            cancel,
        }
    }

    /// Cancels the run. Driving a cancelled run ends it at once with
    /// [`RunOutcome::Cancelled`].
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// A handle that cancels this run after [`drive`](Self::drive) took it.
    #[must_use]
    pub fn cancel_handle(&self) -> CancelHandle {
        self.cancel.clone()
    }

    /// Runs to the end, handing each event to `handler` in the order the run
    /// emitted it, one call at a time.
    ///
    /// The run is polled together with every handler call, so it keeps
    /// making progress while the handler awaits; events it emits meanwhile
    /// queue for the next call. The run stays on the caller's task.
    ///
    /// `state` is what the handler works on, typically `&mut` borrows of the
    /// caller's locals: each call takes it and returns it in
    /// [`ControlFlow::Continue`] for the next. Threading it through, rather
    /// than capturing it in an async closure, keeps the driven future `Send`
    /// whenever the state is. A handler returns [`ControlFlow::Break`] to
    /// stop: the run is cancelled and dropped, its queued events are
    /// discarded, and `drive` returns that `Break`. Otherwise `drive` returns
    /// the state in `Continue` once the run has finished and its every event
    /// was handled.
    ///
    /// `stop` cancels the run from outside, such as on a user's request: it
    /// is polled, ahead of the run and its next event, only while no handler
    /// call is in progress, so a call already running always finishes. Once
    /// it completes, the run is cancelled and driven to its end, so the
    /// handler still sees the run's [`RunEnded`](AgentEvent::RunEnded) with
    /// [`RunOutcome::Cancelled`], and `drive` returns the value in `Break`. A
    /// caller with nothing to stop on passes [`std::future::pending`].
    ///
    /// # Errors
    ///
    /// The run's error, after the handler has seen every event emitted
    /// before it.
    pub async fn drive<S, B, H>(
        self,
        mut state: S,
        stop: impl Future<Output = B>,
        mut handler: impl FnMut(S, TimedEvent) -> H,
    ) -> Result<ControlFlow<B, S>, AgentError>
    where
        H: Future<Output = ControlFlow<B, S>>,
    {
        let Self {
            body: mut run,
            events,
            cancel,
        } = self;
        futures_lite::pin!(events);
        futures_lite::pin!(stop);
        let mut outcome = None;
        let mut stopped = None;
        loop {
            // Read before the closure: borrowing `stopped` across the await
            // would make the future need `B: Sync`.
            let stopping = stopped.is_none();
            let next = std::future::poll_fn(|cx| {
                if stopping && let Poll::Ready(value) = stop.as_mut().poll(cx) {
                    return Poll::Ready(RunStep::Stopped(value));
                }
                poll_run(run.as_mut(), &mut outcome, cx);
                if let Some(result) = outcome.take() {
                    // Finished: what it emitted is all queued.
                    return Poll::Ready(match events.try_recv() {
                        Ok(event) => {
                            outcome = Some(result);
                            RunStep::Event(event)
                        }
                        Err(_) => RunStep::Finished(result),
                    });
                }
                match Stream::poll_next(events.as_mut(), cx) {
                    Poll::Ready(Some(event)) => Poll::Ready(RunStep::Event(event)),
                    // Every sink is gone but the run has not returned; its
                    // own wake-up reports the end.
                    Poll::Ready(None) | Poll::Pending => Poll::Pending,
                }
            })
            .await;
            let event = match next {
                RunStep::Event(event) => event,
                RunStep::Finished(result) => {
                    result?;
                    return Ok(
                        stopped.map_or_else(|| ControlFlow::Continue(state), ControlFlow::Break)
                    );
                }
                RunStep::Stopped(value) => {
                    stopped = Some(value);
                    cancel.cancel();
                    continue;
                }
            };
            let call = handler(state, event);
            futures_lite::pin!(call);
            let flow = std::future::poll_fn(|cx| {
                poll_run(run.as_mut(), &mut outcome, cx);
                call.as_mut().poll(cx)
            })
            .await;
            state = match flow {
                ControlFlow::Continue(state) => state,
                ControlFlow::Break(value) => {
                    cancel.cancel();
                    return Ok(ControlFlow::Break(value));
                }
            };
        }
    }
}

/// What [`Run::drive`] waits for between handler calls.
enum RunStep<B> {
    Event(TimedEvent),
    Finished(Result<(), AgentError>),
    Stopped(B),
}

/// Polls `run` unless it already finished, recording its result.
fn poll_run<F>(
    run: Pin<&mut F>,
    outcome: &mut Option<Result<(), AgentError>>,
    cx: &mut std::task::Context<'_>,
) where
    F: Future<Output = Result<(), AgentError>>,
{
    if outcome.is_none()
        && let Poll::Ready(result) = run.poll(cx)
    {
        *outcome = Some(result);
    }
}

/// How the loop of a run ended, short of an error.
enum LoopEnd {
    /// The model answered and nothing held the run, on turn `turn`.
    Answered { final_text: String, turn: usize },
    /// The run stopped for a reason that needs the embedder.
    Paused(RunPause),
}

/// What the loop of one run works with.
#[derive(Clone, Copy)]
struct RunCx<'a> {
    events: &'a EventSink,
    cancel: &'a CancelSignal,
    run: &'a RunId,
}

enum TurnControl {
    Continue,
    Break(LoopEnd),
}

/// A response stream that ended with an error instead of finishing.
struct BrokenStream {
    error: AgentError,
    /// Whether an event arrived before the error: the stream had opened.
    opened: bool,
    /// What the provider reported the broken request used, if it did.
    usage: Option<aither_core::llm::Usage>,
}

/// A finished tool call, waiting to be paired with its call in the context.
#[derive(Debug)]
pub struct FinishedToolCall {
    id: String,
    outcome: ToolOutcome,
    /// Context an `after_tool` hook appended, to follow the turn's results.
    appended: Option<String>,
}

pub type ToolExecutionResult = Result<FinishedToolCall, AgentError>;

/// One tool call on its way through the hooks and the tool.
struct ToolRun<'a, H> {
    hooks: &'a H,
    tools: &'a AgentTools,
    events: &'a EventSink,
    cancel: &'a CancelSignal,
    approver: Option<&'a AnyApprover>,
    identity: aither_core::llm::CallIdentity,
    turn: usize,
    message_count: usize,
}

impl<H: Hooks> ToolRun<'_, H> {
    fn decided(&self, decision: ToolVerdict) {
        self.events.emit(AgentEvent::ToolCallDecided {
            call: self.identity.call().clone(),
            decision,
        });
    }

    /// What the tool is given besides its arguments.
    fn context(&self) -> aither_core::llm::ToolContext {
        let mut cx = aither_core::llm::ToolContext::with_progress(ProgressEvents {
            events: self.events.clone(),
            call: self.identity.call().clone(),
        })
        .with_identity(self.identity.clone())
        .with_cancellation(self.cancel.clone());
        if let Some(approver) = self.approver {
            cx = cx.with_approver(approver.clone());
        }
        cx
    }

    /// Puts `request` to the approver.
    async fn ask(&self, request: ApprovalRequest) -> Result<Approval, AgentError> {
        let kind = request.kind().clone();
        let Some(approver) = self.approver else {
            return Err(AgentError::Approval(Arc::new(
                aither_core::llm::ApprovalError::NoApprover { kind },
            )));
        };
        let request = if request.call().is_some() {
            request
        } else {
            request.with_call(self.identity.clone())
        };
        approver.approve(request).await.map_err(|failure| {
            AgentError::Approval(Arc::new(aither_core::llm::ApprovalError::Failed {
                kind,
                failure,
            }))
        })
    }

    async fn execute(self, call: &ToolCall) -> ToolExecutionResult {
        let invocation = ToolInvocation {
            call: &self.identity,
            name: &call.name,
            arguments: &call.arguments,
            turn: self.turn,
            message_count: self.message_count,
        };
        let mut arguments = call.arguments.clone();
        let mut denied = None;
        match self.hooks.before_tool(&invocation).await {
            ToolDecision::Allow => self.decided(ToolVerdict::Allowed),
            ToolDecision::Deny(reason) => {
                self.decided(ToolVerdict::Denied {
                    reason: reason.clone(),
                });
                denied = Some(reason);
            }
            ToolDecision::Rewrite(rewritten) => {
                self.decided(ToolVerdict::Rewritten);
                arguments = rewritten;
            }
            ToolDecision::Ask(request) => {
                self.decided(ToolVerdict::AwaitingApproval);
                match self.ask(request).await? {
                    Approval::Approved => self.decided(ToolVerdict::Allowed),
                    Approval::Denied { reason } => {
                        self.decided(ToolVerdict::Denied {
                            reason: reason.clone(),
                        });
                        denied = Some(reason);
                    }
                }
            }
            ToolDecision::Abort(reason) => {
                return Err(AgentError::HookRejected {
                    hook: "before_tool".into(),
                    reason,
                });
            }
        }

        let (result, duration) = if let Some(reason) = denied {
            (
                Ok(aither_core::llm::ToolResult::error(reason)),
                Duration::ZERO,
            )
        } else {
            let start = Instant::now();
            let result = self
                .tools
                .call(&call.name, &arguments.to_string(), self.context())
                .await;
            (result, start.elapsed())
        };
        let mut outcome = ToolOutcome {
            call: self.identity.clone(),
            name: call.name.clone(),
            arguments,
            result,
            duration,
        };
        let mut appended = None;
        match self.hooks.after_tool(&outcome).await {
            AfterTool::Keep => {}
            AfterTool::Replace(replacement) => outcome.result = Ok(replacement),
            AfterTool::Append(context) => appended = Some(context),
            AfterTool::Abort(reason) => {
                return Err(AgentError::HookRejected {
                    hook: "after_tool".into(),
                    reason,
                });
            }
        }
        self.events.emit(AgentEvent::ToolCallEnded {
            call: self.identity.call().clone(),
            result: outcome.for_model(),
        });
        Ok(FinishedToolCall {
            id: call.id.clone(),
            outcome,
            appended,
        })
    }
}

/// Reports a running tool's progress as events.
struct ProgressEvents {
    events: EventSink,
    call: ToolCallId,
}

impl ProgressSink for ProgressEvents {
    async fn report(&mut self, progress: Progress) {
        self.events.emit(AgentEvent::ToolProgress {
            call: self.call.clone(),
            progress,
        });
    }
}

/// Wraps the embedder's approver so a wait for an answer is visible: a
/// [`Paused`](AgentEvent::Paused) goes out only when the answer is not there
/// on the first poll, and a [`Resumed`](AgentEvent::Resumed) when it comes.
struct PausingApprover {
    inner: AnyApprover,
    events: EventSink,
    pauses: Arc<AtomicU64>,
}

impl Approver for PausingApprover {
    async fn approve(&self, request: ApprovalRequest) -> Result<Approval, ApproverFailure> {
        let call = request.call().map(|identity| identity.call().clone());
        let kind = request.kind().clone();
        let subject = request.subject().to_string();
        let answer = self.inner.approve(request);
        futures_lite::pin!(answer);
        if let Some(settled) = futures_lite::future::poll_once(answer.as_mut()).await {
            return settled;
        }
        let pause = PauseId::new(self.pauses.fetch_add(1, Ordering::Relaxed));
        self.events.emit(AgentEvent::Paused {
            pause,
            call,
            kind,
            subject,
        });
        let settled = answer.await;
        self.events.emit(AgentEvent::Resumed { pause });
        settled
    }
}

/// Ends the run's event stream with [`RunOutcome::Cancelled`] when the run
/// is dropped before it ended itself.
struct RunEndGuard {
    events: EventSink,
    run: RunId,
    ended: bool,
}

impl RunEndGuard {
    fn end(&mut self, outcome: RunOutcome) {
        self.ended = true;
        self.events.emit(AgentEvent::RunEnded {
            run: self.run.clone(),
            outcome,
        });
    }
}

impl Drop for RunEndGuard {
    fn drop(&mut self) {
        if !self.ended {
            self.end(RunOutcome::Cancelled);
        }
    }
}

/// The blocks of one response, as they stream in.
struct Response<'a> {
    events: &'a EventSink,
    request: RequestId,
    text_chunks: Vec<String>,
    tool_calls: Vec<ToolCall>,
    reasoning: Vec<ReasoningState>,
    /// The block open now, and the id the next one gets.
    open: Option<(BlockId, OutputKind)>,
    next_block: u32,
    usage: Option<aither_core::llm::Usage>,
}

impl<'a> Response<'a> {
    const fn new(events: &'a EventSink, request: RequestId) -> Self {
        Self {
            events,
            request,
            text_chunks: Vec::new(),
            tool_calls: Vec::new(),
            reasoning: Vec::new(),
            open: None,
            next_block: 0,
            usage: None,
        }
    }

    /// Streams `text` into a block of `kind`, closing the open block when it
    /// is of another kind.
    fn delta(&mut self, kind: OutputKind, text: String) {
        let block = match self.open {
            Some((block, open)) if open == kind => block,
            _ => {
                self.end_block();
                let block = BlockId::new(self.next_block);
                self.next_block += 1;
                self.open = Some((block, kind));
                block
            }
        };
        self.events.emit(AgentEvent::BlockDelta {
            request: self.request,
            block,
            kind,
            text,
        });
    }

    fn end_block(&mut self) {
        if let Some((block, kind)) = self.open.take() {
            self.events.emit(AgentEvent::BlockEnded {
                request: self.request,
                block,
                kind,
            });
        }
    }
}

/// What a model event asks the stream loop to do.
enum ModelEventAction {
    Continue,
    RetryMalformed,
}

fn handle_model_event(
    cache_stats: &mut crate::CacheStats,
    budget: &mut ContextBudget,
    event: Result<Event, AgentError>,
    response: &mut Response<'_>,
) -> Result<ModelEventAction, AgentError> {
    match event {
        Ok(Event::Text(text)) => {
            response.text_chunks.push(text.clone());
            response.delta(OutputKind::Text, text);
            Ok(ModelEventAction::Continue)
        }
        Ok(Event::Reasoning(reasoning)) => {
            response.delta(OutputKind::Reasoning, reasoning);
            Ok(ModelEventAction::Continue)
        }
        Ok(Event::ToolCallDelta {
            id,
            name,
            arguments_fragment,
        }) => {
            response.end_block();
            response.events.emit(AgentEvent::ToolCallDelta {
                call: ToolCallId::new(id),
                name,
                arguments_fragment,
            });
            Ok(ModelEventAction::Continue)
        }
        Ok(Event::ToolCall(call)) => {
            response.end_block();
            response.tool_calls.push(call);
            Ok(ModelEventAction::Continue)
        }
        // State, unlike Reasoning, is not for the reader: it is collected so
        // the next request can hand it back to the provider.
        Ok(Event::ReasoningState(state)) => {
            response.reasoning.push(state);
            Ok(ModelEventAction::Continue)
        }
        Ok(Event::BuiltInToolCall { id, tool, detail }) => {
            response.end_block();
            if !detail.is_empty() {
                response
                    .text_chunks
                    .push(format_builtin_tool_result(&tool, &detail));
            }
            let call = ToolCallId::new(id);
            response.events.emit(AgentEvent::ToolCallStarted {
                call,
                name: tool,
                arguments: serde_json::Value::String(detail),
                origin: ToolOrigin::Hosted,
            });
            Ok(ModelEventAction::Continue)
        }
        Ok(Event::BuiltInToolResult { id, tool, result }) => {
            response.end_block();
            response
                .text_chunks
                .push(format_builtin_tool_result(&tool, &result));
            response.events.emit(AgentEvent::ToolCallEnded {
                call: ToolCallId::new(id),
                result: aither_core::llm::ToolResult::text(result),
            });
            Ok(ModelEventAction::Continue)
        }
        Ok(Event::Usage(usage)) => {
            cache_stats.record(&usage);
            // The provider just counted the prompt it received; that is the
            // only honest measure of how much window the window costs.
            budget.calibrate(&usage);
            response.usage = Some(usage);
            Ok(ModelEventAction::Continue)
        }
        Err(AgentError::Llm { message, .. }) if message.contains("malformed function call") => {
            tracing::warn!("Model generated malformed function call, retrying...");
            Ok(ModelEventAction::RetryMalformed)
        }
        Err(error) => Err(error),
    }
}

fn ensure_non_empty_tool_call_ids(
    tool_calls: &[ToolCall],
    response_text: &str,
) -> Result<(), AgentError> {
    if tool_calls.iter().all(|call| !call.id.trim().is_empty()) {
        return Ok(());
    }
    tracing::error!(
        response_text,
        ?tool_calls,
        "provider emitted tool call with empty id"
    );
    let tool_calls_text = tool_calls
        .iter()
        .map(|call| format!("{} ({})", call.name, call.id))
        .collect::<Vec<_>>()
        .join(", ");
    Err(AgentError::llm_local(format!(
        "provider emitted tool call with empty id; response_text={response_text:?}; tool_calls=[{tool_calls_text}]"
    )))
}

/// The blocks collected into a request's tail, each with the context source
/// that returned it, so a commit reaches only that source.
#[derive(Debug, Default)]
struct RenderedTail {
    blocks: Vec<Block>,
    /// The index into the agent's context sources of the source that returned
    /// the block at the same position; `None` for a hook's block.
    sources: Vec<Option<usize>>,
}

impl std::ops::Deref for RenderedTail {
    type Target = [Block];

    fn deref(&self) -> &[Block] {
        &self.blocks
    }
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompactResult {
    /// Number of messages that were compacted.
    pub messages_compacted: usize,
    /// Number of messages remaining (preserved).
    pub messages_remaining: usize,
    /// The generated summary.
    pub summary: String,
}

/// A stop hook's reminder, as the model reads it.
#[derive(Template)]
#[template(path = "system_reminder.txt", escape = "none")]
struct SystemReminderTemplate {
    content: String,
}

/// The result of a tool call the run left unfinished.
#[derive(Template)]
#[template(path = "tool_call_cancelled.txt", escape = "none")]
struct CancelledToolCallTemplate<'a> {
    tool: &'a str,
}

/// An autonomous agent that runs a conversation with a language model,
/// executing the tool calls it makes.
///
/// The agent manages conversation context, handles tool execution in a loop,
/// and applies context compression when needed. Unlike the raw `LanguageModel`
/// trait which only emits events, the agent handles the full tool execution
/// cycle.
///
/// # Type Parameters
///
/// - `M`: The conversation model
/// - `H`: Composed hooks for customizing behavior (defaults to `()`)
/// - `C`: The optional compaction model, see
///   [`AgentBuilder::compaction_model`](crate::builder::AgentBuilder::compaction_model)
///   (defaults to `M`; unused while none is set)
/// - `Schema`: the [`HandoffSchema`](crate::HandoffSchema) every compaction
///   asks the model for; [`HandoffDocument`] unless the builder's
///   [`handoff`](crate::AgentBuilder::handoff) sets another.
///
/// Compaction runs on the conversation model, so it reads the conversation's
/// prompt cache, unless a compaction model is set.
///
/// # Example
///
/// ```rust,ignore
/// let agent = Agent::new(claude);
/// let response = agent.query("What is 2+2?").await?;
/// ```
#[derive(Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct Agent<M, H = (), C = M, Schema = HandoffDocument> {
    /// The conversation model, behind an Arc so a forked compaction can own
    /// it while the run continues.
    pub(crate) model: Arc<M>,

    /// The model compaction runs on, when it is not the conversation's own.
    ///
    /// A different model cannot read the conversation's prompt cache, so with
    /// one set compaction is never run early to catch the cache warm (see
    /// [`Agent::compact_when_idle`]), and an idle gap past the cache lifetime
    /// reassembles the window instead. Its requests carry its own parameters.
    pub(crate) compaction_model: Option<WithParameters<Arc<C>>>,

    /// The parameters of the requests sent on the model.
    pub(crate) parameters: Box<Parameters>,

    /// The request-level reasoning effort the agent's own requests carry.
    pub(crate) effort: crate::effort::EffortAnchor,

    /// Whether the idle compaction already ran since the last request, so one
    /// idle gap compacts once.
    pub(crate) idle_compacted: bool,

    /// Registered tools.
    pub(crate) tools: AgentTools,

    /// The embedder's decisions.
    pub(crate) hooks: H,

    /// Answers the approvals tool calls and hooks ask for; without one an
    /// approval fails.
    pub(crate) approver: Option<AnyApprover>,

    /// The children in flight, whose events join this agent's stream.
    pub(crate) subagents: Option<crate::SubagentRuns>,

    /// The pauses emitted, numbering the next.
    pub(crate) pauses: Arc<AtomicU64>,

    /// Agent configuration.
    pub(crate) config: AgentConfig,

    /// Unified context manager.
    pub(crate) context: Context,

    /// Window accounting, calibrated against the provider's token counts.
    pub(crate) budget: ContextBudget,

    /// Prompt cache controls derived from the current window layout.
    pub(crate) prompt_cache: PromptCache,

    /// Sources rendered into the volatile tail of every request.
    pub(crate) context_sources: Vec<AnyContextSource>,

    /// Live content: sources rendered into the cacheable prefix, after the
    /// system prompt, in this order.
    pub(crate) live_sources: Vec<AnyLiveSource>,

    /// The tail of the request assembled last, which a preview or a
    /// measurement of the window shows.
    pub(crate) last_tail: Vec<Block>,

    /// Whether the window was rewritten since the sources last heard a
    /// request begin, which wipes every durable alert they pushed before.
    pub(crate) reassembled: bool,

    /// What rewrote the prompt prefix since the previous request, reported
    /// with the next one's [`AgentEvent::RequestStarted`].
    pub(crate) prefix_rewrites: PrefixRewrites,

    /// How many requests this agent has assembled.
    pub(crate) requests_sent: u64,

    /// The next [`RequestId`]: every attempt at a response is one.
    pub(crate) request_ids: u64,

    /// The one compaction in the making: running alongside the turn, or
    /// finished and waiting for a request boundary.
    pub(crate) compaction: CompactionState,

    /// Why the compaction forked alongside the run failed, until one lands:
    /// the next compaction the window forces recovers first.
    pub(crate) compaction_failure: Option<crate::context::CompactionError>,

    /// Whether an [`AgentEvent::CompactionStarted`] was emitted whose
    /// [`AgentEvent::CompactionEnded`] is still owed.
    pub(crate) compaction_announced: bool,

    /// Results of the tool calls of the turn in flight, in completion order,
    /// until they are moved into the context.
    pub(crate) tool_results: VecDeque<ToolExecutionResult>,

    /// Cached model profile.
    pub(crate) profile: Option<ModelProfile>,

    /// The compaction model's profile, when one is set. It tells the cache
    /// controls of a compaction request what that model's client will do with
    /// the messages.
    pub(crate) compaction_profile: Option<ModelProfile>,

    /// Whether the profile was fetched and the window sized.
    pub(crate) initialized: bool,

    /// Ends a provider stream in flight when the machine wakes or goes
    /// offline.
    pub(crate) connectivity: crate::connectivity::Watch,

    /// Directory files are restored from after a forked compaction.
    pub(crate) restore_dir: Option<PathBuf>,

    /// How the handoff's file index finds the files tool results name.
    pub(crate) file_references: Option<AnyFileReferences>,

    /// Whether the provider's prompt-cache entry is warm; stamped on every
    /// request and keep-alive, shared with hosts through
    /// [`Agent::cache_warmth`].
    pub(crate) warmth: crate::CacheWarmth,

    /// The most recent LLM request issued by this agent, retained so a
    /// prompt-cache keep-alive can re-send it verbatim.
    pub(crate) last_request: Option<LLMRequest>,

    /// The items of the request in flight, as the sequence number after the
    /// last of them, until the provider begins answering it: only then has
    /// the model been delivered what it carried.
    pub(crate) delivering: Option<u64>,

    /// Where every request this agent sends is published for a fork
    /// subagent to start from; `None` when no fork reads this agent.
    pub(crate) fork_source: Option<crate::fork::ForkSource>,

    /// The agent every tool call of this loop is made by.
    pub(crate) identity: aither_core::llm::AgentIdentity,

    /// Rolling KV-cache statistics accumulated from emitted `Usage` events.
    ///
    /// Hosts can read [`Agent::cache_stats`] at any time to observe the
    /// cumulative prompt-caching hit rate for this agent session.
    pub(crate) cache_stats: crate::CacheStats,

    /// Runtime skill registry loaded by host integrations.
    #[cfg(feature = "skills")]
    pub(crate) skill_registry: Option<Arc<aither_skills::SkillRegistry>>,

    /// The schema compactions ask for; no value is held.
    pub(crate) handoff_schema: std::marker::PhantomData<fn() -> Schema>,
}

impl<M: LanguageModel> Agent<M> {
    /// Creates a new agent with default configuration.
    #[must_use]
    pub fn new(llm: M) -> Self {
        Self::with_config(llm, AgentConfig::default())
    }

    /// Creates a new agent with the specified configuration.
    #[must_use]
    pub fn with_config(llm: M, config: AgentConfig) -> Self {
        let prompt_cache = PromptCache::new(uuid::Uuid::new_v4().to_string());
        Self {
            warmth: crate::CacheWarmth::new(prompt_cache.lifetime(), crate::AnyClock::default()),
            model: Arc::new(llm),
            compaction_model: None,
            parameters: Box::default(),
            effort: crate::effort::EffortAnchor::default(),
            idle_compacted: false,
            tools: AgentTools::new(),
            hooks: (),
            approver: None,
            pauses: Arc::default(),
            config,
            context: Context::default(),
            budget: ContextBudget::default(),
            prompt_cache,
            context_sources: Vec::new(),
            live_sources: Vec::new(),
            last_tail: Vec::new(),
            reassembled: false,
            prefix_rewrites: PrefixRewrites::default(),
            requests_sent: 0,
            request_ids: 0,
            compaction: CompactionState::default(),
            compaction_failure: None,
            compaction_announced: false,
            tool_results: VecDeque::new(),
            profile: None,
            compaction_profile: None,
            initialized: false,
            connectivity: crate::connectivity::Watch::default(),
            restore_dir: None,
            file_references: None,
            last_request: None,
            delivering: None,
            fork_source: None,
            identity: aither_core::llm::AgentIdentity::Main,
            subagents: None,
            cache_stats: crate::CacheStats::new(),
            #[cfg(feature = "skills")]
            skill_registry: None,
            handoff_schema: std::marker::PhantomData,
        }
    }

    /// Returns a builder for more complex agent construction.
    pub fn builder(llm: M) -> crate::builder::AgentBuilder<M> {
        crate::builder::AgentBuilder::new(llm)
    }
}

impl<M, H, C, Schema> Agent<M, H, C, Schema>
where
    Schema: crate::HandoffSchema,
    M: LanguageModel + 'static,
    // Owned by the compaction future, which outlives the turn it races
    // against.
    C: LanguageModel + 'static,
    H: Hooks,
{
    /// Response stream of the model, boxed so the provider's concrete stream
    /// type does not inflate this async chain's poll frames in debug builds
    /// (aither#143).
    fn respond_stream<'a>(
        model: &'a Arc<M>,
        connectivity: &crate::connectivity::Watch,
        request: LLMRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<Event, AgentError>> + Send + 'a>> {
        let stream: crate::connectivity::EventStream<'a> = Box::pin(
            model
                .respond(request)
                .map(|event| event.map_err(|error| AgentError::from_model_error(&error))),
        );
        connectivity.guard(stream)
    }

    /// The retry policy of the conversation model.
    fn retry_policy(&self) -> RetryPolicy {
        self.model.retry_policy()
    }

    /// Streams the model's response to `request`, sending the request again
    /// when the stream breaks after it opened.
    ///
    /// A failure before the first event is the provider's to retry, since it
    /// retries establishing the stream itself; a break after it is retried
    /// here, under the model's [`RetryPolicy`], unless the machine is
    /// offline. The broken attempt's text, calls and reasoning are dropped
    /// with it, and the request sent again is the one recorded as the last
    /// request, so the provider reads the same cached prefix.
    async fn collect_model_turn(
        &mut self,
        mut request: LLMRequest,
        prefix: &PromptPrefix,
        events: &EventSink,
    ) -> Result<ModelTurn, AgentError> {
        let policy = self.retry_policy();
        let mut retries = 0;
        loop {
            let id = RequestId::new(self.request_ids);
            self.request_ids += 1;
            events.emit(AgentEvent::RequestStarted {
                request: id,
                prefix: prefix.clone(),
            });
            let broken = match self.stream_model_turn(request, id, events).await {
                Ok(turn) => {
                    events.emit(AgentEvent::RequestEnded {
                        request: id,
                        usage: turn.usage.clone(),
                        outcome: RequestOutcome::Completed,
                    });
                    return Ok(turn);
                }
                Err(broken) => broken,
            };
            let failure = match broken.error.failure() {
                Some(failure)
                    if broken.opened
                        && policy.allows(retries, failure)
                        && !self.connectivity.is_offline() =>
                {
                    *failure
                }
                _ => {
                    events.emit(AgentEvent::RequestEnded {
                        request: id,
                        usage: broken.usage,
                        outcome: RequestOutcome::Failed {
                            error: broken.error.clone(),
                        },
                    });
                    return Err(broken.error);
                }
            };
            let delay = policy.delay(retries, &failure);
            retries += 1;
            tracing::warn!(
                retry = retries,
                max_retries = policy.max_retries,
                delay_ms = delay.as_millis(),
                error = %broken.error,
                "model response stream broke mid-response; sending the request again"
            );
            events.emit(AgentEvent::RequestEnded {
                request: id,
                usage: broken.usage,
                outcome: RequestOutcome::Retrying {
                    error: broken.error,
                    retry: retries,
                    max_retries: policy.max_retries,
                    delay,
                },
            });
            async_io::Timer::after(delay).await;
            request = self
                .last_request
                .clone()
                .expect("a request that reached the model is recorded as the last request");
        }
    }

    /// One attempt at streaming the model's response to `request`.
    async fn stream_model_turn(
        &mut self,
        request: LLMRequest,
        id: RequestId,
        events: &EventSink,
    ) -> Result<ModelTurn, Box<BrokenStream>> {
        let mut response = Response::new(events, id);
        let mut malformed_function_call = false;
        let mut opened = false;

        let stream = Self::respond_stream(&self.model, &self.connectivity, request);
        futures_lite::pin!(stream);
        while let Some(event) = stream.next().await {
            let received = event.is_ok();
            let action = handle_model_event(
                &mut self.cache_stats,
                &mut self.budget,
                event,
                &mut response,
            )
            .map_err(|error| {
                response.end_block();
                Box::new(BrokenStream {
                    error,
                    opened,
                    usage: response.usage.take(),
                })
            })?;
            if received && let Some(carried) = self.delivering.take() {
                self.context.mark_delivered_below(carried);
            }
            opened |= received;
            if matches!(action, ModelEventAction::RetryMalformed) {
                malformed_function_call = true;
                break;
            }
        }
        response.end_block();

        Ok(ModelTurn {
            response_text: response.text_chunks.join(""),
            tool_calls: response.tool_calls,
            reasoning: response.reasoning,
            usage: response.usage,
            malformed_function_call,
        })
    }

    /// Runs on the user's `prompt`.
    async fn run_streaming(
        &mut self,
        prompt: String,
        attachments: Vec<Attachment>,
        events: &EventSink,
        cancel: &CancelSignal,
    ) -> Result<(), AgentError> {
        self.run_to_end(
            events,
            cancel,
            Some(Message::user(prompt).with_attachments(attachments)),
        )
        .await
    }

    /// A run without a new user message: the model acts on what the host
    /// pushed with [`push_message`](Self::push_message).
    async fn resume_streaming(
        &mut self,
        events: &EventSink,
        cancel: &CancelSignal,
    ) -> Result<(), AgentError> {
        self.run_to_end(events, cancel, None).await
    }

    /// Runs the loop on `message` (when there is one) and ends the run's
    /// event stream with exactly one [`AgentEvent::RunEnded`], whatever way
    /// the run ends.
    async fn run_to_end(
        &mut self,
        events: &EventSink,
        cancel: &CancelSignal,
        message: Option<Message>,
    ) -> Result<(), AgentError> {
        let run = RunId::new(uuid::Uuid::new_v4().to_string());
        events.emit(AgentEvent::RunStarted {
            run: run.clone(),
            agent: events.agent.clone(),
        });
        let mut guard = RunEndGuard {
            events: events.clone(),
            run: run.clone(),
            ended: false,
        };
        let cx = RunCx {
            events,
            cancel,
            run: &run,
        };
        // Cancellation is polled first: a run cancelled before it started
        // must not get a turn in.
        let finished = futures_lite::future::or(
            async {
                cancel.cancelled().await;
                None
            },
            async { Some(self.run_body(&cx, message).await) },
        )
        .await;
        match finished {
            Some(Ok(LoopEnd::Answered { final_text, .. })) => {
                guard.end(RunOutcome::Completed { final_text });
                Ok(())
            }
            Some(Ok(LoopEnd::Paused(pause))) => {
                guard.end(RunOutcome::Paused { pause });
                Ok(())
            }
            Some(Err(error @ AgentError::MaxIterations { .. })) => {
                guard.end(RunOutcome::IterationLimit);
                Err(error)
            }
            Some(Err(error)) => {
                guard.end(RunOutcome::Failed {
                    error: error.clone(),
                });
                Err(error)
            }
            None => {
                guard.end(RunOutcome::Cancelled);
                Ok(())
            }
        }
    }

    async fn run_body(
        &mut self,
        cx: &RunCx<'_>,
        message: Option<Message>,
    ) -> Result<LoopEnd, AgentError> {
        self.finish_lost_compaction(cx.events);
        self.ensure_initialized().await?;
        self.settle_interrupted_tool_calls();
        // An effort change lands before the user message it applies to.
        self.settle_effort()?;
        if let Some(message) = message {
            self.context.push(message);
        }
        let end = self.run_agent_loop(cx).await?;
        if let LoopEnd::Answered { turn, .. } = &end {
            self.emit_checkpoint(cx, CheckpointReason::Stop, *turn);
        }
        Ok(end)
    }

    async fn run_agent_loop(&mut self, cx: &RunCx<'_>) -> Result<LoopEnd, AgentError> {
        let mut turn = 0;
        loop {
            turn += 1;
            if turn > self.config.max_iterations {
                return Err(AgentError::MaxIterations {
                    limit: self.config.max_iterations,
                });
            }
            cx.events.emit(AgentEvent::TurnStarted {
                run: cx.run.clone(),
                turn,
            });
            match self.run_agent_turn(cx, turn).await? {
                TurnControl::Continue => {}
                TurnControl::Break(end) => return Ok(end),
            }
        }
    }

    async fn run_agent_turn(
        &mut self,
        cx: &RunCx<'_>,
        turn: usize,
    ) -> Result<TurnControl, AgentError> {
        let events = cx.events;
        let Some((request, prefix)) = self.prepare_request(cx, turn).await? else {
            Self::end_turn(cx, turn, 0);
            return Ok(TurnControl::Break(LoopEnd::Paused(
                RunPause::ContextExhausted,
            )));
        };
        let model_turn = self
            .run_turn_racing_compaction(request, &prefix, events)
            .await?;

        if model_turn.malformed_function_call {
            Self::end_turn(cx, turn, 0);
            return Ok(TurnControl::Continue);
        }
        let response_text = model_turn.response_text;
        let tool_calls = model_turn.tool_calls;
        ensure_non_empty_tool_call_ids(&tool_calls, &response_text)?;

        self.commit_assistant_turn(&response_text, &tool_calls, model_turn.reasoning);
        if tool_calls.is_empty() {
            return self.finish_no_tool_turn(cx, response_text, turn).await;
        }

        Self::push_tool_start_events(&tool_calls, events);
        self.execute_tool_calls(cx, &tool_calls, turn).await;
        let (outcomes, appended) = self.apply_tool_results()?;
        self.push_tool_turn_context(cx, appended).await;
        self.push_turn_boundary_events(cx, turn, &response_text, &outcomes)
            .await
    }

    fn end_turn(cx: &RunCx<'_>, turn: usize, tool_calls: usize) {
        cx.events.emit(AgentEvent::TurnEnded {
            run: cx.run.clone(),
            turn,
            tool_calls,
        });
    }

    /// Runs one model turn while a forked compaction keeps running.
    ///
    /// The two are polled together instead of sequenced: compaction is a full
    /// request, and waiting for it would stall the run at exactly the moment
    /// the window is tightest. Whichever finishes first is handled — a landed
    /// handoff is parked for the next request boundary, where replacing the
    /// conversation is safe.
    async fn run_turn_racing_compaction(
        &mut self,
        request: LLMRequest,
        prefix: &PromptPrefix,
        events: &EventSink,
    ) -> Result<ModelTurn, AgentError> {
        let mut pending = self.compaction.take_running();
        let mut landed = None;

        let turn = {
            let turn = self.collect_model_turn(request, prefix, events);
            futures_lite::pin!(turn);
            loop {
                let step = match pending.as_mut() {
                    Some(compaction) => {
                        futures_lite::future::or(
                            async { TurnStep::Turn((&mut turn).await) },
                            async { TurnStep::Compacted(compaction.wait().await) },
                        )
                        .await
                    }
                    None => TurnStep::Turn((&mut turn).await),
                };
                match step {
                    TurnStep::Turn(turn) => break turn,
                    TurnStep::Compacted(result) => {
                        landed = Some(result);
                        pending = None;
                    }
                }
            }
        };

        match landed {
            Some(Ok(completed)) => self.compaction.land(completed),
            Some(Err(error)) => {
                tracing::warn!(%error, "forked compaction failed");
                self.finish_compaction(events, CompactionOutcome::Failed(error.clone().into()));
                self.compaction_failure = Some(error);
            }
            None => {}
        }
        self.compaction.resume(pending);
        turn
    }

    /// Stores what the model produced this turn: its text, calls and reasoning.
    ///
    /// The reasoning travels with the turn that produced it, or later requests
    /// carry the calls and the answer without the thinking behind them. A turn
    /// that produced nothing at all stores nothing.
    fn commit_assistant_turn(
        &mut self,
        response_text: &str,
        tool_calls: &[ToolCall],
        reasoning: Vec<ReasoningState>,
    ) {
        if response_text.is_empty() && tool_calls.is_empty() && reasoning.is_empty() {
            return;
        }
        self.context.push(Message::assistant_with_reasoning(
            response_text,
            tool_calls.to_vec(),
            reasoning,
        ));
    }

    async fn finish_no_tool_turn(
        &mut self,
        cx: &RunCx<'_>,
        response_text: String,
        turn: usize,
    ) -> Result<TurnControl, AgentError> {
        Self::end_turn(cx, turn, 0);
        let objected = self.push_final_turn_context(cx).await;
        let held = self.decide_stop(cx, &response_text, turn).await;
        if objected || held {
            // The alerts stay in the conversation either way; a host that has
            // a message waiting ends the run here so the message travels with
            // them in the next request instead of one request later.
            let boundary_ctx = TurnBoundaryContext {
                assistant_text: &response_text,
                turn,
                message_count: self.context.len_recent(),
                outcomes: &[],
            };
            if self.hooks.on_turn_boundary(&boundary_ctx).await == TurnBoundaryAction::EndTurn {
                return Ok(TurnControl::Break(LoopEnd::Answered {
                    final_text: response_text,
                    turn,
                }));
            }
            return Ok(TurnControl::Continue);
        }
        Ok(TurnControl::Break(LoopEnd::Answered {
            final_text: response_text,
            turn,
        }))
    }

    /// Gives every tool call in the context a result.
    ///
    /// A run stream dropped while tools execute leaves the assistant message
    /// carrying the calls with no results behind it, and providers reject a
    /// request holding such calls. Runs at the start of the next run (and of a
    /// manual compaction), so the repair needs nothing from the host and only
    /// appends: the prefix, thinking blocks included, stays byte-identical.
    /// Calls whose results had already landed keep them; the rest get a
    /// cancelled result.
    fn settle_interrupted_tool_calls(&mut self) {
        let mut parked = std::mem::take(&mut self.tool_results);
        let mut unresolved: Vec<ToolCall> = Vec::new();
        for message in self.context.recent() {
            unresolved.extend(message.tool_calls().iter().cloned());
            if let Some(id) = message.tool_call_id() {
                unresolved.retain(|call| call.id != id);
            }
        }
        for call in unresolved {
            let finished = parked
                .iter()
                .position(|result| matches!(result, Ok(finished) if finished.id == call.id))
                .and_then(|index| parked.remove(index));
            let content = match finished {
                Some(Ok(finished)) => finished.outcome.for_model().render_for_model(),
                _ => CancelledToolCallTemplate { tool: &call.name }
                    .render()
                    .expect("cancelled tool call template must render"),
            };
            self.context.push(Message::tool(&call.id, content));
        }
    }

    fn push_tool_start_events(tool_calls: &[ToolCall], events: &EventSink) {
        for call in tool_calls {
            events.emit(AgentEvent::ToolCallStarted {
                call: ToolCallId::new(call.id.clone()),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                origin: ToolOrigin::Local,
            });
        }
    }

    /// Runs `tool_calls` concurrently, parking each result in
    /// `self.tool_results` the moment it lands.
    ///
    /// The results live on the agent rather than in this future so a run
    /// stream dropped mid-execution still leaves the finished ones for
    /// [`Self::settle_interrupted_tool_calls`] to pair.
    async fn execute_tool_calls(&mut self, cx: &RunCx<'_>, tool_calls: &[ToolCall], turn: usize) {
        let message_count = self.context.len_recent();
        let approver = self.approver.clone().map(|inner| {
            AnyApprover::new(PausingApprover {
                inner,
                events: cx.events.clone(),
                pauses: Arc::clone(&self.pauses),
            })
        });
        let incoming = self.subagents.as_ref().map(|runs| runs.incoming().clone());
        let (hooks, tools, agent) = (&self.hooks, &self.tools, &self.identity);
        let results = &mut self.tool_results;
        let tool_futures = tool_calls.iter().map(|call| {
            let identity = aither_core::llm::CallIdentity::new(
                ToolCallId::new(call.id.clone()),
                agent.clone(),
            );
            let approver = approver.clone();
            async move {
                let run = ToolRun {
                    hooks,
                    tools,
                    events: cx.events,
                    cancel: cx.cancel,
                    approver: approver.as_ref(),
                    identity,
                    turn,
                    message_count,
                };
                run.execute(call).await
            }
        });

        let mut pending = tool_futures.collect::<futures::stream::FuturesUnordered<_>>();
        loop {
            // A child a tool runs reports through the registry; its events
            // join this stream as they arrive, not when the tool returns.
            let step = futures_lite::future::or(
                async { ToolStep::Finished(pending.next().await) },
                async {
                    match &incoming {
                        Some(incoming) => ToolStep::Child(incoming.recv().await.ok()),
                        None => std::future::pending().await,
                    }
                },
            )
            .await;
            match step {
                ToolStep::Finished(Some(result)) => results.push_back(result),
                ToolStep::Finished(None) => break,
                ToolStep::Child(Some(event)) => cx.events.forward(event),
                ToolStep::Child(None) => {}
            }
        }
        drop(pending);
        self.forward_child_events(cx.events);
    }

    /// Puts the events children reported onto this agent's stream.
    fn forward_child_events(&self, events: &EventSink) {
        if let Some(runs) = &self.subagents {
            while let Ok(event) = runs.incoming().try_recv() {
                events.forward(event);
            }
        }
    }

    /// Moves the parked tool results into the context, and returns the
    /// turn's outcomes and the context its `after_tool` hooks appended.
    ///
    /// Results not reached because one failed stay parked for
    /// [`Self::settle_interrupted_tool_calls`].
    fn apply_tool_results(&mut self) -> Result<(Vec<ToolOutcome>, Vec<String>), AgentError> {
        let mut outcomes = Vec::new();
        let mut appended = Vec::new();
        while let Some(result) = self.tool_results.pop_front() {
            let finished = result?;
            self.context.push(Message::tool(
                &finished.id,
                finished.outcome.for_model().render_for_model(),
            ));
            outcomes.push(finished.outcome);
            appended.extend(finished.appended);
        }
        Ok((outcomes, appended))
    }

    async fn push_turn_boundary_events(
        &self,
        cx: &RunCx<'_>,
        turn: usize,
        response_text: &str,
        outcomes: &[ToolOutcome],
    ) -> Result<TurnControl, AgentError> {
        let boundary_ctx = TurnBoundaryContext {
            assistant_text: response_text,
            turn,
            message_count: self.context.len_recent(),
            outcomes,
        };
        self.emit_checkpoint(cx, CheckpointReason::TurnBoundary, turn);
        Self::end_turn(cx, turn, outcomes.len());
        if self.hooks.on_turn_boundary(&boundary_ctx).await == TurnBoundaryAction::EndTurn {
            Ok(TurnControl::Break(LoopEnd::Answered {
                final_text: response_text.to_string(),
                turn,
            }))
        } else {
            Ok(TurnControl::Continue)
        }
    }

    /// Re-sends the last request to refresh the cache entry of an idle agent.
    ///
    /// Re-sends the last request with no output (`max_tokens: 0`,
    /// non-streaming, the same thinking and effort settings), so the
    /// provider restarts the entry's lifetime. The history is untouched, the
    /// request's usage is recorded in [`cache_stats`](Self::cache_stats) and
    /// the entry's [`cache_warmth`](Self::cache_warmth) restarts. The caller
    /// decides when to call it, shortly before the entry goes cold.
    ///
    /// Returns the request's usage so the caller can account for its cost
    /// (all fields empty when the provider reported none), or `None` when
    /// there is nothing to refresh: no request was sent yet, or the entry is
    /// permanent or already cold.
    ///
    /// # Errors
    ///
    /// Returns the provider error when the keep-alive request fails.
    pub async fn keep_cache_warm(&mut self) -> Result<Option<aither_core::llm::Usage>, AgentError> {
        if self.warmth.cold_in().is_none() {
            return Ok(None);
        }
        let Some(last_request) = self.last_request.clone() else {
            return Ok(None);
        };
        let (messages, parameters, tool_definitions) = last_request.into_parts();
        let request = LLMRequest::new(messages)
            .with_tool_definitions(tool_definitions)
            .with_parameters(parameters.max_tokens(0).stream(false));
        let stream = Self::respond_stream(&self.model, &self.connectivity, request);
        futures_lite::pin!(stream);
        let mut refreshed = aither_core::llm::Usage::default();
        while let Some(event) = stream.next().await {
            match event {
                Ok(Event::Usage(usage)) => {
                    self.cache_stats.record(&usage);
                    refreshed = usage;
                }
                Ok(_) => {}
                Err(error) => return Err(error),
            }
        }
        self.warmth.start_entry_now();
        Ok(Some(refreshed))
    }

    /// A shared handle on whether the provider's prompt-cache entry for this
    /// agent's context is warm and how long until it goes cold. It keeps
    /// tracking this agent's later requests and keep-alives.
    #[must_use]
    pub fn cache_warmth(&self) -> crate::CacheWarmth {
        self.warmth.clone()
    }

    /// How long the provider's prompt-cache entry for this agent's context
    /// stays warm without a hit.
    #[must_use]
    pub const fn cache_lifetime(&self) -> CacheLifetime {
        self.prompt_cache.lifetime()
    }

    /// Performs a one-shot query and returns the final response.
    ///
    /// This is the simplest way to use the agent. The agent handles tool
    /// execution internally and returns the final text response.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The LLM returns an error
    /// - A hook aborts the operation
    /// - Tool execution fails
    /// - The run paused ([`AgentError::Paused`]) or was cancelled
    ///   ([`AgentError::Cancelled`]) before it had a reply
    pub async fn query(&mut self, prompt: &str) -> Result<String, AgentError> {
        let limit = self.config.max_iterations;
        let agent = AgentPath::of(&self.identity);
        let flow = self
            .run(prompt, std::iter::empty())
            .drive((), std::future::pending(), |(), timed| {
                let own_end = timed.agent == agent;
                std::future::ready(match timed.event {
                    AgentEvent::RunEnded { outcome, .. } if own_end => ControlFlow::Break(outcome),
                    _ => ControlFlow::Continue(()),
                })
            })
            .await?;
        match flow {
            ControlFlow::Break(RunOutcome::Completed { final_text }) => Ok(final_text),
            ControlFlow::Break(RunOutcome::Paused { pause }) => Err(AgentError::Paused(pause)),
            ControlFlow::Break(RunOutcome::Failed { error }) => Err(error),
            ControlFlow::Break(RunOutcome::IterationLimit) => {
                Err(AgentError::MaxIterations { limit })
            }
            ControlFlow::Break(_) | ControlFlow::Continue(()) => Err(AgentError::Cancelled),
        }
    }

    /// Starts a run on `prompt`; [`Run::drive`] runs it and hands each event
    /// to the caller as it happens, text chunks as they arrive from the LLM.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::ops::ControlFlow;
    ///
    /// let mut shown = String::new();
    /// agent
    ///     .run("Implement the feature", std::iter::empty())
    ///     .drive(&mut shown, std::future::pending(), |shown, timed| async move {
    ///         if let AgentEvent::BlockDelta { kind: OutputKind::Text, text, .. } = timed.event {
    ///             shown.push_str(&text);
    ///         }
    ///         ControlFlow::<(), _>::Continue(shown)
    ///     })
    ///     .await?;
    /// ```
    pub fn run(
        &mut self,
        prompt: &str,
        attachments: impl IntoIterator<Item = Attachment>,
    ) -> Run<impl Future<Output = Result<(), AgentError>> + '_> {
        let prompt = prompt.to_string();
        let attachments: Vec<Attachment> = attachments.into_iter().collect();
        Run::new(
            AgentPath::of(&self.identity),
            move |sink, cancel| async move {
                self.run_streaming(prompt, attachments, &sink, &cancel)
                    .await
            },
        )
    }

    /// Continues the conversation without a new user message, so the model
    /// acts on the messages the host added with
    /// [`push_message`](Self::push_message). Driven like [`run`](Self::run).
    pub fn resume(&mut self) -> Run<impl Future<Output = Result<(), AgentError>> + '_> {
        Run::new(
            AgentPath::of(&self.identity),
            move |sink, cancel| async move { self.resume_streaming(&sink, &cancel).await },
        )
    }

    /// Registers a tool for the agent to use.
    ///
    /// # Errors
    ///
    /// Returns [`RegisterError`](aither_core::llm::tool::RegisterError) if the
    /// name collides with a registered tool, or the tool has no description.
    pub fn register_tool<T: aither_core::llm::Tool + 'static>(
        &mut self,
        tool: T,
    ) -> Result<(), aither_core::llm::tool::RegisterError> {
        self.tools.register(tool)
    }

    /// Returns a reference to the unified context manager.
    #[must_use]
    pub const fn context(&self) -> &Context {
        &self.context
    }

    /// Returns a mutable reference to the unified context manager.
    ///
    /// Use this to insert/update system blocks, push messages, etc.
    pub const fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    /// Adds a message to the conversation history.
    pub fn push_message(&mut self, message: Message) {
        self.context.push(message);
    }

    /// Returns the rolling KV-cache statistics for this agent session.
    ///
    /// The stats are updated every time the underlying provider emits a
    /// `Usage` event (which, for providers like Claude with prompt
    /// caching, includes `cache_read_tokens` and `cache_write_tokens`).
    /// Hosts can read the hit rate, reset the accumulator, or snapshot
    /// it for per-turn diffs.
    #[must_use]
    pub const fn cache_stats(&self) -> &crate::CacheStats {
        &self.cache_stats
    }

    /// Returns a mutable reference to the KV-cache statistics accumulator.
    ///
    /// Hosts typically only use this to [`CacheStats::reset`] between
    /// distinct sessions, since the agent itself records new observations.
    ///
    /// [`CacheStats::reset`]: crate::CacheStats::reset
    pub const fn cache_stats_mut(&mut self) -> &mut crate::CacheStats {
        &mut self.cache_stats
    }

    /// Clears the conversation history.
    pub fn clear_history(&mut self) {
        self.compaction.retire();
        self.context.clear_history();
    }

    /// Compacts the conversation by generating a structured handoff and
    /// starting fresh, focused on `focus` when one is given. Driven like
    /// [`run`](Self::run).
    ///
    /// The run emits [`AgentEvent::CompactionStarted`] and then
    /// [`AgentEvent::CompactionEnded`], whose
    /// [`CompactionOutcome`] carries the [`CompactResult`] when the handoff
    /// applied. A compaction forked earlier alongside a run is stale once this
    /// one starts: it finishes as [`CompactionOutcome::Discarded`] first.
    ///
    /// # Errors
    ///
    /// [`drive`](Run::drive) returns the failure when handoff generation
    /// fails, after the finish event reported it.
    pub fn compact(
        &mut self,
        focus: Option<&str>,
    ) -> Run<impl Future<Output = Result<(), AgentError>> + '_> {
        let focus = focus.map(str::to_string);
        Run::new(
            AgentPath::of(&self.identity),
            move |sink, _cancel| async move {
                self.finish_lost_compaction(&sink);
                Box::pin(self.compact_streaming(
                    focus.as_deref(),
                    CompactionTrigger::Requested,
                    &sink,
                ))
                .await
                .map(drop)
            },
        )
    }

    /// Runs one compaction to its end, announcing it on `events`.
    async fn compact_streaming(
        &mut self,
        focus: Option<&str>,
        trigger: CompactionTrigger,
        events: &EventSink,
    ) -> Result<Option<CompactResult>, AgentError> {
        self.ensure_initialized().await?;
        self.settle_interrupted_tool_calls();
        // This handoff describes the whole conversation as it stands, so a
        // compaction forked earlier, running or landed, is stale from here.
        self.compaction.retire();
        self.finish_compaction(events, CompactionOutcome::Discarded);
        self.announce_compaction(events, trigger);
        match self.fork_and_apply(focus, events).await {
            Ok(Some(result)) => Ok(Some(result)),
            Ok(None) => {
                self.finish_compaction(events, CompactionOutcome::NothingToCompact);
                Ok(None)
            }
            Err(error) => {
                self.finish_compaction(events, CompactionOutcome::Failed(error.clone()));
                Err(error)
            }
        }
    }

    /// Generates the handoff for the whole conversation and applies it;
    /// `None` when there is no conversation to compact.
    async fn fork_and_apply(
        &mut self,
        focus: Option<&str>,
        events: &EventSink,
    ) -> Result<Option<CompactResult>, AgentError> {
        if self.context.is_conversation_empty() {
            return Ok(None);
        }
        let definitions = self.tools.active_definitions();
        let messages = self.request_messages(&self.last_tail);
        let request = self.compaction_request(messages, definitions);
        #[cfg(feature = "skills")]
        let skills = self.skill_catalog();
        #[cfg(not(feature = "skills"))]
        let skills = Vec::new();
        let attachments = self.context.attachment_catalog(self.context.fork_point());
        let fork = match CompactionFork::<Schema>::new(
            &request,
            self.context.fork_point(),
            self.context.handoff_file_index(),
            focus,
            &skills,
            &attachments,
        ) {
            Ok(fork) => fork.with_file_references(self.file_references.clone()),
            // Only system messages: nothing a handoff could summarize.
            Err(crate::context::CompactionError::Empty) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let completed = self
            .compaction_future(fork)
            .await
            .map_err(AgentError::from)?;

        self.compaction.land(completed);
        Ok(Some(
            self.apply_completed_compaction(events)
                .await
                .expect("the compaction just landed"),
        ))
    }

    /// The parameters of the requests the agent sends on its model.
    #[must_use]
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    /// Changes the parameters of the requests sent on the model, from the
    /// next request on.
    ///
    /// A reasoning effort change keeps the prompt cache warm when the model
    /// takes per-message effort updates
    /// ([`Ability::PerMessageEffort`](aither_core::llm::model::Ability::PerMessageEffort)):
    /// the next request carries the update inside the conversation, before
    /// the next user message when one starts the turn, and the request-level
    /// effort stays as the first request sent it. Clearing the effort there
    /// is an update to the model's documented default
    /// ([`Profile::default_reasoning_effort`](aither_core::llm::model::Profile::default_reasoning_effort));
    /// with none recorded, the next turn fails with [`AgentError::Config`]
    /// rather than leave the earlier level in force. Any other model, and any
    /// other parameter, takes the new value at the request level, which
    /// restarts the cache.
    pub fn set_parameters(&mut self, parameters: Parameters) {
        *self.parameters = parameters;
    }

    /// Replaces the conversation model.
    ///
    /// The next request goes to `model`; history, context and tools stay. A
    /// new model reads none of the old one's prompt cache, so the change
    /// applies at once. A compaction already running finishes on the model it
    /// started on.
    pub fn set_model(&mut self, model: M) {
        self.model = Arc::new(model);
        // Forget what belonged to the replaced model: its profile, fetched
        // again before the next request, and the last request, which a
        // keep-alive would otherwise send it.
        self.profile = None;
        self.last_request = None;
    }

    /// Whether the agent's own model takes reasoning effort changes inside
    /// the conversation.
    fn takes_per_message_effort(&self) -> bool {
        self.profile
            .as_ref()
            .is_some_and(|profile| profile.abilities.contains(&Ability::PerMessageEffort))
    }

    /// Whether the agent's own model takes a system message inside the
    /// conversation, which decides whether a provider that cannot keeps the
    /// live tail out of its system prompt.
    fn takes_mid_conversation_system(&self) -> bool {
        Self::profile_takes_mid_conversation_system(self.profile.as_ref())
    }

    /// Whether the model a compaction request goes to takes a system message
    /// inside the conversation: the compaction model when one is set, else the
    /// agent's own.
    ///
    /// A model's client decides this from the same ability its profile
    /// reports, so the cache controls and the client cannot disagree.
    ///
    /// # Panics
    ///
    /// Panics when a compaction model is set and its profile is not loaded,
    /// which initialization does before any request.
    fn compaction_takes_mid_conversation_system(&self) -> bool {
        if self.compaction_model.is_none() {
            return self.takes_mid_conversation_system();
        }
        let profile = self
            .compaction_profile
            .as_ref()
            .expect("an initialized agent has its compaction model's profile");
        Self::profile_takes_mid_conversation_system(Some(profile))
    }

    fn profile_takes_mid_conversation_system(profile: Option<&ModelProfile>) -> bool {
        profile.is_some_and(|profile| profile.abilities.contains(&Ability::MidConversationSystem))
    }

    /// Brings the effort in force to the one the agent wants: appends a
    /// per-message update when its model takes one, clearing the effort to
    /// the model's default level, and otherwise settles the request-level
    /// effort.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Config`] when the effort is cleared on a model
    /// that takes per-message updates but whose default effort is not
    /// recorded, while another level is in force: no update can undo it.
    ///
    /// # Panics
    ///
    /// Panics before the agent is initialized, since only the model's profile
    /// says how an effort change is sent.
    fn settle_effort(&mut self) -> Result<(), AgentError> {
        let profile = self
            .profile
            .as_ref()
            .expect("the effort is settled only by an initialized agent, which has its profile");
        let changes = if profile.abilities.contains(&Ability::PerMessageEffort) {
            crate::effort::EffortChanges::PerMessage {
                default: profile.default_reasoning_effort,
            }
        } else {
            crate::effort::EffortChanges::RequestLevel
        };
        let update = self
            .effort
            .plan(
                self.parameters.reasoning_effort,
                changes,
                self.context.configured_effort(),
            )
            .map_err(|crate::effort::UnknownDefaultEffort| {
                AgentError::Config(format!(
                    "the reasoning effort was cleared, but model '{}' takes effort changes \
                     inside the conversation and its default effort is not recorded, so no \
                     update can undo the level in force; set an explicit effort",
                    profile.name
                ))
            })?;
        if let Some(effort) = update {
            self.context.push(Message::configuration(effort));
        }
        Ok(())
    }

    /// The parameters of the agent's own requests as they are sent: the
    /// configured ones, with the request-level effort
    /// [`settle_effort`](Self::settle_effort) settled on.
    fn request_parameters(&self) -> Parameters {
        let mut parameters = (*self.parameters).clone();
        parameters.reasoning_effort = self.effort.request_effort(parameters.reasoning_effort);
        parameters
    }

    /// The window as a request to the agent's own model carries it, with the
    /// live `tail`: its effort updates are left out when the model cannot
    /// read them.
    fn request_messages(&self, tail: &[Block]) -> Vec<Message> {
        let mut messages = self.context.build_request(tail);
        if !self.takes_per_message_effort() {
            messages.retain(|message| message.role() != Role::Configuration);
        }
        messages
    }

    /// A compaction request over `messages` and `definitions`: the
    /// compaction model's parameters when one is set, else the session
    /// model's, under the cache controls the requests that wrote the prefix
    /// carry, so a compaction on the session's own model is served from that
    /// cache.
    ///
    /// `messages` is the window as the session model's requests carry it. A
    /// compaction model runs at its own effort, so the session's effort
    /// updates are left out of its request.
    fn compaction_request(
        &self,
        mut messages: Vec<Message>,
        definitions: Vec<aither_core::llm::tool::ToolDefinition>,
    ) -> LLMRequest {
        if self.compaction_model.is_some() {
            messages.retain(|message| message.role() != Role::Configuration);
        }
        let parameters = self.compaction_model.as_ref().map_or_else(
            || self.request_parameters(),
            |compactor| (*compactor.parameters).clone(),
        );
        let parameters = self.prompt_cache.apply(
            parameters,
            &messages,
            !definitions.is_empty(),
            self.compaction_takes_mid_conversation_system(),
        );
        LLMRequest::new(messages)
            .with_parameters(parameters)
            .with_tool_definitions(definitions)
    }

    /// Runs `fork` to completion on the compaction model: the configured one,
    /// or else the conversation model.
    async fn compaction_future(
        &self,
        fork: CompactionFork<Schema>,
    ) -> Result<CompletedCompaction, crate::context::CompactionError> {
        if let Some(compactor) = &self.compaction_model {
            return fork.run(Arc::clone(&compactor.model)).await;
        }
        fork.run(Arc::clone(&self.model)).await
    }

    /// Starts `fork` in the background on the compaction model: the
    /// configured one, or else the conversation model.
    fn start_pending_compaction(&self, fork: CompactionFork<Schema>) -> PendingCompaction {
        if let Some(compactor) = &self.compaction_model {
            return PendingCompaction::start(fork, Arc::clone(&compactor.model));
        }
        PendingCompaction::start(fork, Arc::clone(&self.model))
    }

    /// How long from now the idle compaction is due: `Some(Duration::ZERO)`
    /// when it is due now, `None` when none is pending.
    ///
    /// The window is compacted once, shortly before the conversation's cache
    /// lifetime ends ([`ContextPolicy::idle_compact_before_expiry`] ahead of
    /// it), counted from the entry's last use, so the compaction reads the
    /// still-warm cache. None is pending when:
    ///
    /// - a compaction model is set, which cannot read that cache;
    /// - the lifetime is [`CacheLifetime::Permanent`];
    /// - nothing has been sent yet, or the conversation is empty;
    /// - the context holds fewer tokens than the policy's
    ///   [`compaction_floor`](crate::ContextPolicy::compaction_floor);
    /// - it already ran since the last request;
    /// - the cache has expired already, when compacting would only pay to
    ///   write the whole context again.
    ///
    /// The agent has no timer of its own: the host sleeps this long while the
    /// session is idle, then calls [`compact_when_idle`](Self::compact_when_idle).
    ///
    /// [`ContextPolicy::idle_compact_before_expiry`]: crate::ContextPolicy::idle_compact_before_expiry
    #[must_use]
    pub fn idle_compaction_due_in(&self) -> Option<Duration> {
        if self.compaction_model.is_some()
            || self.idle_compacted
            || self.compaction.is_active()
            || self.context.is_conversation_empty()
            || self
                .config
                .context_policy
                .below_floor(self.budget.estimated_tokens())
        {
            return None;
        }
        let cold_in = self.warmth.cold_in()?;
        Some(cold_in.saturating_sub(self.config.context_policy.idle_compact_before_expiry))
    }

    /// Compacts the idle session once its cache is about to expire.
    ///
    /// Runs one compaction through the normal fork and apply path when
    /// [`idle_compaction_due_in`](Self::idle_compaction_due_in) says it is due
    /// now, and returns `None` otherwise, so a host may call it whenever its
    /// idle timer fires. One idle gap compacts once: the next request re-arms
    /// it. Nothing is kept across the handoff, so no thinking produced under
    /// the old prefix is replayed after it.
    ///
    /// Driven like [`compact`](Self::compact), and emits its compaction
    /// events only when the compaction is due; otherwise the run emits
    /// nothing.
    ///
    /// # Errors
    ///
    /// [`drive`](Run::drive) returns the compaction failure; the window is
    /// left as it was.
    pub fn compact_when_idle(&mut self) -> Run<impl Future<Output = Result<(), AgentError>> + '_> {
        Run::new(
            AgentPath::of(&self.identity),
            move |sink, _cancel| async move {
                self.finish_lost_compaction(&sink);
                if self.idle_compaction_due_in() != Some(Duration::ZERO) {
                    return Ok(());
                }
                let result =
                    Box::pin(self.compact_streaming(None, CompactionTrigger::Idle, &sink)).await?;
                self.idle_compacted = true;
                if let Some(result) = &result {
                    tracing::info!(
                        compacted = result.messages_compacted,
                        "compacted the idle context shortly before its cache expires"
                    );
                }
                Ok(())
            },
        )
    }

    /// Forks a compaction off `request` to run alongside the turns that follow.
    ///
    /// The compaction replays the session's current model unless a compaction
    /// model is set: the handoff has to carry the same understanding of the
    /// conversation the next requests are made of, and only that model reads
    /// the cache the run just paid for.
    ///
    /// A failure to even start — an empty conversation — is not an error for
    /// the run: there is simply nothing to compact yet, and nothing is
    /// announced.
    fn start_compaction(&mut self, request: &LLMRequest, focus: Option<&str>, events: &EventSink) {
        #[cfg(feature = "skills")]
        let skills = self.skill_catalog();
        #[cfg(not(feature = "skills"))]
        let skills = Vec::new();
        let attachments = self.context.attachment_catalog(self.context.fork_point());
        match CompactionFork::<Schema>::new(
            request,
            self.context.fork_point(),
            self.context.handoff_file_index(),
            focus,
            &skills,
            &attachments,
        )
        .map(|fork| fork.with_file_references(self.file_references.clone()))
        {
            Ok(fork) => {
                tracing::debug!(
                    fork_point = fork.fork_point(),
                    usage = self.budget.usage(),
                    "forking compaction alongside the run"
                );
                self.announce_compaction(events, CompactionTrigger::WindowFilling);
                self.compaction.start(self.start_pending_compaction(fork));
            }
            Err(error) => {
                tracing::debug!(%error, "compaction not forked");
            }
        }
    }

    /// The session's skill catalog, offered to compaction prompts and used to
    /// resolve skill names during restoration.
    #[cfg(feature = "skills")]
    fn skill_catalog(&self) -> Vec<SkillCatalogEntry> {
        if let Some(registry) = &self.skill_registry {
            return registry
                .all()
                .into_iter()
                .map(|skill| SkillCatalogEntry {
                    name: skill.name.clone(),
                    description: skill.description.clone(),
                    instructions: skill.instructions.clone(),
                })
                .collect();
        }
        Vec::new()
    }

    /// Announces on `events` that a compaction starts.
    ///
    /// # Panics
    ///
    /// Panics when the compaction announced before it has not finished:
    /// compactions never overlap.
    fn announce_compaction(&mut self, events: &EventSink, trigger: CompactionTrigger) {
        assert!(
            !self.compaction_announced,
            "a compaction started before the previous one finished"
        );
        self.compaction_announced = true;
        events.emit(AgentEvent::CompactionStarted { trigger });
    }

    /// Reports on `events` how the announced compaction ended, if one is
    /// announced and not yet finished.
    fn finish_compaction(&mut self, events: &EventSink, outcome: CompactionOutcome) {
        if std::mem::take(&mut self.compaction_announced) {
            events.emit(AgentEvent::CompactionEnded { outcome });
        }
    }

    /// Finishes as discarded an announced compaction that is gone: retired,
    /// or dropped with its run, where no stream was open to report it.
    fn finish_lost_compaction(&mut self, events: &EventSink) {
        if !self.compaction.is_active() {
            self.finish_compaction(events, CompactionOutcome::Discarded);
        }
    }

    /// Folds a landed handoff into the window and restores what it named,
    /// reporting it on `events`; `None` when no handoff had landed.
    ///
    /// Called only at request boundaries: the conversation is whole there, so
    /// replacing everything below the fork point cannot strand a tool call
    /// from its result.
    async fn apply_completed_compaction(&mut self, events: &EventSink) -> Option<CompactResult> {
        // The files are read while the completed compaction is still stored:
        // a caller that drops this future during the read leaves it pending
        // for the next boundary, with nothing half applied.
        let completed = self.compaction.landed()?;
        let handoff = completed.handoff.clone();
        let restoration = completed.restoration.clone();
        let attachment_catalog = completed.attachment_catalog.clone();
        let restored = match self.restore_dir.clone() {
            Some(restore_dir) => {
                let tokens = self
                    .budget
                    .fraction_tokens(self.config.context_policy.restore_budget);
                let budget_chars = self.budget.chars_for_tokens(tokens);
                #[cfg(feature = "skills")]
                let skills = self.skill_catalog();
                #[cfg(not(feature = "skills"))]
                let skills = Vec::new();
                restore_context(
                    &restoration,
                    handoff.file_index(),
                    &[restore_dir],
                    &skills,
                    &attachment_catalog,
                    budget_chars,
                )
                .await
            }
            None => None,
        };

        // From here there is no await: the handoff, the cleanup and the
        // restored block commit together.
        let CompletedCompaction { fork_point, .. } = self.compaction.take_landed()?;
        let messages_compacted = self.context.len_below(fork_point);
        let summary = handoff.text().to_string();
        // Compaction rewrites the cacheable prefix anyway, so this is where
        // staged system-block updates and superseded tool results fold in.
        self.context.reassemble_below(fork_point);
        let user_message_chars = self
            .budget
            .chars_for_tokens(self.config.context_policy.user_message_budget);
        let kept_attachments: HashSet<AttachmentRef> = restoration
            .attachments
            .iter()
            .filter_map(|id| {
                attachment_catalog
                    .iter()
                    .find(|entry| &entry.id == id)
                    .map(|entry| entry.reference)
            })
            .collect();
        self.context
            .apply_handoff(handoff, fork_point, user_message_chars, &kept_attachments);
        if let Some(block) = restored {
            self.context.push_block_alert(&block);
        }
        self.reassembled = true;
        self.prefix_rewrites.compacted = true;
        self.compaction_failure = None;
        tracing::info!(
            fork_point,
            kept = self.context.len_recent(),
            "applied handoff from forked compaction"
        );
        let result = CompactResult {
            messages_compacted,
            messages_remaining: self.context.len_recent(),
            summary,
        };
        self.finish_compaction(events, CompactionOutcome::Applied(result.clone()));
        Some(result)
    }

    /// Makes room at the pause point: waits for the compaction in flight and
    /// applies it, or runs one now when none is.
    ///
    /// A compaction that fails — the one in flight, one that failed earlier
    /// alongside the run, or the one run here — sends the context through
    /// recovery: alerts and reminders are removed, long tool outputs move to
    /// files, [`AgentEvent::CompactionRecovery`] tells the host, and the
    /// compaction reruns synchronously.
    ///
    /// Returns `false` when there was nothing to compact, so the window is
    /// still over the pause point and the run cannot issue another request.
    ///
    /// # Errors
    ///
    /// Returns the error of the rerun when it fails too, or of the file
    /// write recovery needed.
    async fn make_room(&mut self, events: &EventSink) -> Result<bool, AgentError> {
        let mut failure = self.compaction_failure.take().map(AgentError::from);
        if let Some(mut pending) = self.compaction.take_running() {
            match pending.wait().await {
                Ok(completed) => {
                    self.compaction.land(completed);
                    self.apply_completed_compaction(events).await;
                    return Ok(true);
                }
                Err(error) => {
                    tracing::warn!(%error, "compaction failed with the window exhausted");
                    let error = AgentError::from(error);
                    self.finish_compaction(events, CompactionOutcome::Failed(error.clone()));
                    failure = Some(error);
                }
            }
        }
        let failure = match failure {
            Some(failure) => failure,
            None => {
                match Box::pin(self.compact_streaming(None, CompactionTrigger::PausePoint, events))
                    .await
                {
                    Ok(result) => return Ok(result.is_some()),
                    Err(error) => {
                        tracing::warn!(%error, "compaction failed with the window exhausted");
                        error
                    }
                }
            }
        };

        let policy = self.config.context_policy;
        let limits = SpillLimits {
            threshold: self.budget.chars_for_tokens(policy.spill_threshold),
            head: self.budget.chars_for_tokens(policy.spill_head),
        };
        let fork_point = self.context.fork_point();
        let report = clean_for_recovery(
            &mut self.context,
            fork_point,
            self.restore_dir.as_deref(),
            limits,
        )
        .await
        .map_err(|error| {
            AgentError::llm_local(format!(
                "compaction recovery could not move a long tool output to a file: {error}"
            ))
        })?;
        tracing::warn!(
            alerts_removed = report.alerts_removed,
            outputs_spilled = report.outputs_spilled,
            "rerunning the failed compaction on the cleaned context"
        );
        events.emit(AgentEvent::CompactionRecovery {
            error: failure,
            alerts_removed: report.alerts_removed,
            outputs_spilled: report.outputs_spilled,
        });
        self.reassembled = true;
        self.prefix_rewrites.reassembled = true;
        let result =
            Box::pin(self.compact_streaming(None, CompactionTrigger::PausePoint, events)).await?;
        Ok(result.is_some())
    }

    /// Tells each source which of its tail blocks have been sent.
    ///
    /// Separate from assembling the tail so that only a request that
    /// actually goes out advances a source's baseline.
    async fn commit_tail(&mut self, tail: &RenderedTail) {
        for (index, source) in self.context_sources.iter_mut().enumerate() {
            let sent: Vec<Block> = tail
                .blocks
                .iter()
                .zip(&tail.sources)
                .filter(|(_, owner)| **owner == Some(index))
                .map(|(block, _)| block.clone())
                .collect();
            if !sent.is_empty() {
                source.committed(&sent).await;
            }
        }
    }

    /// Brings every live source's block to what it renders now.
    ///
    /// The blocks are prefix content: a change the model has not heard of is
    /// announced once, by the alert the source writes for it, and the next
    /// reassembly folds the new block into the prefix. Rendering is
    /// concurrent because the sources are independent.
    async fn sync_live_sources(&mut self, events: &EventSink) {
        let rendered =
            futures::future::join_all(self.live_sources.iter().map(AnyLiveSource::render)).await;
        for (source, desired) in self.live_sources.iter().zip(rendered) {
            // An empty block costs a message and says nothing.
            let desired = desired.filter(|content| !content.is_empty());
            if let Some(alert) =
                self.context
                    .update_live_source(&source.slot(), desired, |change| source.announce(change))
            {
                self.budget.append(alert);
                events.emit(AgentEvent::ContextInjected {
                    source: SourceId::new(source.tag()),
                    kind: InjectionKind::LiveChange,
                });
            }
        }
    }

    /// Appends a source's durable block to the conversation.
    fn push_stable_block(&mut self, events: &EventSink, source: &SourceId, block: &Block) {
        self.budget.append(block.len());
        self.context.push_block_alert(block);
        events.emit(AgentEvent::ContextInjected {
            source: source.clone(),
            kind: InjectionKind::Stable,
        });
    }

    /// Takes in the blocks one source or hook returned at `moment`: a stable
    /// block becomes a durable alert at once, a tail block joins `tail`.
    ///
    /// Returns whether any block was stable.
    ///
    /// # Panics
    ///
    /// Panics on a tail block at any moment but a request: only a request
    /// has a tail, so the source breaks its contract.
    fn take_blocks(
        &mut self,
        events: &EventSink,
        source: &SourceId,
        owner: Option<usize>,
        moment: Moment,
        blocks: Vec<ContextBlock>,
        tail: &mut RenderedTail,
    ) -> bool {
        let mut stable = false;
        for block in blocks {
            match block.placement() {
                Placement::Stable => {
                    self.push_stable_block(events, source, block.block());
                    stable = true;
                }
                Placement::Tail => {
                    assert!(
                        matches!(moment, Moment::Request { .. }),
                        "context source `{}` returned a tail block at {moment:?}; only a request has a tail",
                        source.as_str(),
                    );
                    events.emit(AgentEvent::ContextInjected {
                        source: source.clone(),
                        kind: InjectionKind::Tail,
                    });
                    tail.blocks.push(block.block().clone());
                    tail.sources.push(owner);
                }
            }
        }
        stable
    }

    /// Collects what the sources and the `before_request` hook put in front
    /// of a request: the one moment live content and sources are read, so
    /// nothing is written while the agent is idle.
    ///
    /// Stable blocks are appended to the conversation; tail blocks are
    /// returned, and kept as the tail previews and snapshots show. Sources
    /// are told whether the window was reassembled since the last time,
    /// which is when their earlier stable blocks stopped being in it.
    async fn collect_request_context(&mut self, cx: &RunCx<'_>, turn: usize) -> RenderedTail {
        self.sync_live_sources(cx.events).await;
        let reassembled = std::mem::take(&mut self.reassembled);
        let moment = Moment::Request { reassembled };
        let mut tail = RenderedTail::default();
        for index in 0..self.context_sources.len() {
            let blocks = self.context_sources[index].blocks(moment).await;
            let id = self.context_sources[index].id();
            self.take_blocks(cx.events, &id, Some(index), moment, blocks, &mut tail);
        }
        let request = RequestCx {
            run: cx.run,
            turn,
            message_count: self.context.len_recent(),
        };
        let blocks = self.hooks.before_request(&request).await;
        self.take_blocks(
            cx.events,
            &SourceId::new("hooks"),
            None,
            moment,
            blocks,
            &mut tail,
        );
        tail
    }

    /// Collects what sources return after a tool turn, and the context the
    /// turn's `after_tool` hooks appended.
    async fn push_tool_turn_context(&mut self, cx: &RunCx<'_>, appended: Vec<String>) {
        let mut tail = RenderedTail::default();
        for index in 0..self.context_sources.len() {
            let blocks = self.context_sources[index]
                .blocks(Moment::ToolTurnEnd)
                .await;
            let id = self.context_sources[index].id();
            self.take_blocks(
                cx.events,
                &id,
                Some(index),
                Moment::ToolTurnEnd,
                blocks,
                &mut tail,
            );
        }
        let hooks = SourceId::new("hooks");
        for content in appended {
            self.push_stable_block(cx.events, &hooks, &Block::new("tool-context", content));
        }
    }

    /// Collects what sources return when the model wants to stop.
    ///
    /// Returns `true` when at least one source objected, which keeps the loop
    /// running for another turn so the model can act on the objection.
    async fn push_final_turn_context(&mut self, cx: &RunCx<'_>) -> bool {
        let mut tail = RenderedTail::default();
        let mut objected = false;
        for index in 0..self.context_sources.len() {
            let blocks = self.context_sources[index].blocks(Moment::FinalTurn).await;
            let id = self.context_sources[index].id();
            objected |= self.take_blocks(
                cx.events,
                &id,
                Some(index),
                Moment::FinalTurn,
                blocks,
                &mut tail,
            );
        }
        objected
    }

    /// Asks the `on_stop` hook whether the run ends as the model ends its
    /// turn, and pushes the hook's reason as a reminder when it holds the
    /// run.
    ///
    /// Returns whether the hook holds the run.
    async fn decide_stop(&mut self, cx: &RunCx<'_>, response_text: &str, turn: usize) -> bool {
        let stop = StopCx {
            final_text: response_text,
            turn,
            message_count: self.context.len_recent(),
        };
        match self.hooks.on_stop(&stop).await {
            StopDecision::End => false,
            StopDecision::Continue { reason } => {
                let block = Block::new(
                    "system_reminder",
                    SystemReminderTemplate { content: reason }
                        .render()
                        .expect("system reminder template must render"),
                );
                self.push_stable_block(cx.events, &SourceId::new("hooks"), &block);
                true
            }
        }
    }

    /// Returns the current conversation history.
    #[must_use]
    pub fn history(&self) -> Vec<Message> {
        self.context.conversation_messages()
    }

    fn tool_surface_hash(&self) -> String {
        let mut names = self
            .tools
            .active_definitions()
            .into_iter()
            .map(|definition| definition.name().to_string())
            .collect::<Vec<_>>();
        names.sort();
        let mut hasher = Sha256::new();
        for name in names {
            hasher.update(name.as_bytes());
            hasher.update(b"\n");
        }
        hex_encode(hasher.finalize().as_slice())
    }

    /// Export a checkpoint bundle for restart-safe persistence.
    ///
    /// # Errors
    ///
    /// Returns an error when checkpoint construction or checkpoint hooks fail.
    pub async fn export_checkpoint(&mut self) -> Result<AgentCheckpoint, AgentError> {
        self.ensure_initialized().await?;
        let context_window = self.snapshot_context_window().await?;
        Ok(AgentCheckpoint {
            context: self.context.checkpoint(),
            tool_surface_hash: self.tool_surface_hash(),
            context_window,
        })
    }

    /// Restore a previously exported checkpoint bundle.
    ///
    /// # Errors
    ///
    /// Returns an error when the checkpoint cannot be restored into this agent.
    pub fn restore_checkpoint(&mut self, checkpoint: AgentCheckpoint) -> Result<(), AgentError> {
        self.context.restore(checkpoint.context);
        // Everything below belongs to the window just replaced: its fork
        // point, its failure, its parked tool results and its last request
        // mean nothing in the restored one.
        self.compaction.retire();
        self.compaction_failure = None;
        self.idle_compacted = false;
        self.tool_results.clear();
        self.last_request = None;
        Ok(())
    }

    /// Returns a structured snapshot of the current context window.
    ///
    /// This is a pure measurement: no source baseline is advanced and no
    /// alert is consumed, so a host may poll it as often as it likes.
    ///
    /// # Errors
    ///
    /// Returns an error when the model profiles cannot be fetched during
    /// initialization.
    pub async fn snapshot_context_window(&mut self) -> Result<ContextWindowSnapshot, AgentError> {
        self.ensure_initialized().await?;
        Ok(self.context_window_snapshot(&self.last_tail))
    }

    /// The identity this agent was built with.
    #[must_use]
    pub const fn identity(&self) -> &aither_core::llm::AgentIdentity {
        &self.identity
    }

    /// Returns the model profile if available.
    #[must_use]
    pub const fn profile(&self) -> Option<&ModelProfile> {
        self.profile.as_ref()
    }

    /// Ensures the agent is initialized: the profiles fetched and the window
    /// sized to the model.
    ///
    /// The profile is fetched again after the agent's own model was
    /// replaced.
    async fn ensure_initialized(&mut self) -> Result<(), AgentError> {
        if self.profile.is_none() {
            self.load_profile().await?;
        }
        if self.compaction_profile.is_none()
            && let Some(compactor) = &self.compaction_model
        {
            self.compaction_profile = Some(
                compactor
                    .model
                    .profile()
                    .await
                    .map_err(|e| AgentError::from_model_error(&e))?,
            );
        }
        self.initialized = true;
        Ok(())
    }

    /// Fetches the profile of the agent's own model and sizes the window to
    /// it.
    async fn load_profile(&mut self) -> Result<(), AgentError> {
        // The window of the conversation model is the only one the budget
        // may spend: compaction replays the main request's prefix through the
        // same model, so no second profile is involved.
        let profile = self
            .model
            .profile()
            .await
            .map_err(|e| AgentError::from_model_error(&e))?;
        let policy = self.config.context_policy;
        if !policy.admits(profile.context_length) {
            return Err(AgentError::Config(format!(
                "the {}-token context window is no larger than the {}-token handoff reserve, \
                 so nothing fits below the pause point; lower `ContextPolicy::handoff_reserve`",
                profile.context_length, policy.handoff_reserve,
            )));
        }
        self.budget.set_window(profile.context_length);
        self.profile = Some(profile);
        Ok(())
    }

    /// How the conversation model counts prompt tokens, if it can.
    fn active_token_counting(&self) -> TokenCounting {
        self.model.token_counting()
    }

    /// Asks the conversation model for the request's exact prompt tokens.
    ///
    /// Local counters run on every assembly; an endpoint counter costs a
    /// round trip, so it counts only once the estimate is within
    /// [`ContextPolicy::count_distance`] of the warn point. A counting
    /// failure never fails the request — the estimate stands.
    async fn count_active_request(&self, request: &LLMRequest) -> Option<u32> {
        let should_count = match self.active_token_counting() {
            TokenCounting::None => return None,
            TokenCounting::Local => true,
            TokenCounting::Endpoint => self
                .config
                .context_policy
                .should_count_endpoint(self.budget.estimated_tokens(), self.budget.window()),
        };
        if !should_count {
            return None;
        }
        match self
            .model
            .count_tokens(request)
            .await
            .map_err(|error| error.to_string())
        {
            Ok(tokens) => tokens,
            Err(error) => {
                tracing::warn!(%error, "token counting failed; using the estimate");
                None
            }
        }
    }

    /// Records the assembled prompt's size — exactly, when the model can
    /// count it.
    async fn measure_assembled(
        &mut self,
        messages: &[Message],
        definitions: &[aither_core::llm::tool::ToolDefinition],
    ) {
        self.budget
            .observe_assembled(request_chars(messages, definitions));
        if matches!(self.active_token_counting(), TokenCounting::None) {
            return;
        }
        let request =
            LLMRequest::new(messages.to_vec()).with_tool_definitions(definitions.to_vec());
        if let Some(tokens) = self.count_active_request(&request).await {
            self.budget.observe_counted(tokens);
        }
    }

    /// Assembles the next request, handling the window's lifecycle first.
    ///
    /// This is the only request boundary there is, and everything that must
    /// not happen mid-turn happens here, in order: a landed handoff is
    /// folded in, the window is measured against the provider's own token
    /// counts, a compaction is forked if usage crossed
    /// [`ContextPolicy::start_point`], and the compaction is awaited (or run,
    /// with recovery when it fails) once the room left before the
    /// pause point is reached — the window is down to the handoff reserve.
    ///
    /// With a compaction model set, a cache that went cold since the last
    /// request is rebuilt first (`rebuild_if_cache_is_cold`).
    ///
    /// Returns `None` when the window is exhausted and no compaction can
    /// relieve it: the caller pauses the run rather than sending a request
    /// the provider would reject for length.
    ///
    /// # Errors
    ///
    /// Returns the compaction failure when the cold-cache compaction fails;
    /// nothing is sent on an uncompacted context.
    ///
    /// [`ContextPolicy::start_point`]: crate::ContextPolicy::start_point
    async fn prepare_request(
        &mut self,
        cx: &RunCx<'_>,
        turn: usize,
    ) -> Result<Option<(LLMRequest, PromptPrefix)>, AgentError> {
        let events = cx.events;
        self.forward_child_events(events);
        self.apply_completed_compaction(events).await;
        self.rebuild_if_cache_is_cold(events).await?;
        self.settle_effort()?;

        let definitions = self.tools.active_definitions();
        let mut tail = self.collect_request_context(cx, turn).await;
        let mut messages = self.request_messages(&tail);
        self.measure_assembled(&messages, &definitions).await;

        if self
            .config
            .context_policy
            .should_compact(self.budget.estimated_tokens(), self.budget.window())
            && !self.compaction.is_active()
        {
            let fork_request = self.compaction_request(messages.clone(), definitions.clone());
            self.start_compaction(&fork_request, None, events);
        }

        if self
            .config
            .context_policy
            .should_pause(self.budget.estimated_tokens(), self.budget.window())
        {
            tracing::warn!(
                tokens = self.budget.estimated_tokens(),
                "context window at the safe buffer; waiting for compaction"
            );
            if !self.make_room(events).await? {
                return Ok(None);
            }
            tail = self.collect_request_context(cx, turn).await;
            messages = self.request_messages(&tail);
            self.measure_assembled(&messages, &definitions).await;
            // A landed compaction is still no guarantee the rebuilt request
            // fits; an unshrinkable request is paused rather than sent.
            if self
                .config
                .context_policy
                .should_pause(self.budget.estimated_tokens(), self.budget.window())
            {
                return Ok(None);
            }
        }

        self.commit_tail(&tail).await;
        self.context.mark_prefix_carried();
        self.delivering = Some(self.context.fork_point());
        self.warmth.start_entry_now();
        self.idle_compacted = false;
        self.publish_fork_snapshot(&tail, &definitions);
        self.last_tail.clone_from(&tail.blocks);

        let has_tools = !definitions.is_empty();
        let prefix = PromptPrefix {
            chain: PrefixChain::of(&definitions, &messages),
            rewrites: std::mem::take(&mut self.prefix_rewrites),
            first_request: self.requests_sent == 0,
        };
        self.requests_sent += 1;
        self.prompt_cache
            .observe_prefix(self.context.stable_prefix_fingerprint());
        let parameters = self.prompt_cache.apply(
            self.request_parameters(),
            &messages,
            has_tools,
            self.takes_mid_conversation_system(),
        );
        let request = LLMRequest::new(messages)
            .with_parameters(parameters)
            .with_tool_definitions(definitions);
        self.last_request = Some(request.clone());
        Ok(Some((request, prefix)))
    }

    /// Publishes the request about to be sent for a fork to start from.
    fn publish_fork_snapshot(
        &self,
        tail: &[Block],
        definitions: &[aither_core::llm::tool::ToolDefinition],
    ) {
        let Some(source) = &self.fork_source else {
            return;
        };
        let mut context = self.context.clone();
        context.push_tail(tail);
        source.publish(crate::fork::ForkSnapshot {
            context,
            prompt_cache: self.prompt_cache.clone(),
            budget: self.budget,
            profile: self
                .profile
                .clone()
                .expect("a request is only sent by an initialized agent, which has its profile"),
            request_parameters: (*self.parameters).clone(),
            effort: self.effort,
            context_policy: self.config.context_policy,
            definitions: definitions.to_vec(),
            usage: self.budget.usage(),
        });
    }

    /// Rebuilds the prefix once, before a send, when the provider's cache has
    /// gone cold.
    ///
    /// Only with a compaction model set (otherwise the session is compacted
    /// shortly before the cache expires, see [`Agent::compact_when_idle`], and
    /// a send never rebuilds). The cache is cold when the conversation's
    /// [`CacheLifetime`] has passed since the entry was last used; a permanent
    /// lifetime never is, and nothing runs while the session is idle.
    ///
    /// The compaction model reads the context uncached whatever the timing, so
    /// compacting now costs no more than later, and the send then pays its
    /// cache write on the compacted context instead of the whole one. The
    /// compaction's apply already reassembles the window, so the prefix is
    /// rewritten once: reassembly alone runs only when there is no
    /// conversation to compact. Nothing is kept across the handoff, so no
    /// thinking produced under the old prefix is replayed after it.
    ///
    /// A forked compaction already in flight lands and rebuilds the prefix
    /// itself, so it is left alone.
    ///
    /// [`CacheLifetime`]: crate::context::CacheLifetime
    async fn rebuild_if_cache_is_cold(&mut self, events: &EventSink) -> Result<(), AgentError> {
        if self.compaction_model.is_none() || self.compaction.is_running() {
            return Ok(());
        }
        if !self.warmth.has_expired() {
            return Ok(());
        }
        if self.context.is_conversation_empty() {
            self.reassemble_if_cache_is_cold();
            return Ok(());
        }
        // Boxed: the compaction future is large and sits on every run's path
        // (see `run_futures_stay_small_with_a_real_provider_type`).
        if let Some(result) =
            Box::pin(self.compact_streaming(None, CompactionTrigger::ColdCache, events)).await?
        {
            tracing::info!(
                compacted = result.messages_compacted,
                "compacted the context before sending on a cold cache"
            );
        }
        Ok(())
    }

    /// Rebuilds the window once the provider's cache has gone cold anyway.
    ///
    /// Reached only for a session with a compaction model and nothing to
    /// compact (`rebuild_if_cache_is_cold`).
    ///
    /// Reassembly rewrites the cacheable prefix, so it is only free once the
    /// cached copy has expired — an idle gap at least as long as the
    /// conversation's [`CacheLifetime`]. A staged system-block rewrite and
    /// superseded tool results wait for exactly that moment; a permanent
    /// cache lifetime means no idle gap ever justifies the rewrite.
    ///
    /// [`CacheLifetime`]: crate::context::CacheLifetime
    fn reassemble_if_cache_is_cold(&mut self) -> crate::context::ReassemblyReport {
        let unchanged = crate::context::ReassemblyReport::default();
        if !self.warmth.has_expired() {
            return unchanged;
        }
        let report = self.context.reassemble();
        self.reassembled |= report.system_messages_pruned > 0;
        self.prefix_rewrites.reassembled |= report.changed();
        if report.changed() {
            tracing::debug!(
                system_messages = report.system_messages_pruned,
                deduped = report.tool_results_deduped,
                bytes_freed = report.bytes_freed,
                "rebuilt the context window on a cold cache"
            );
        }
        report
    }

    /// Measures the window exactly as it stands, against a rendered `tail`.
    ///
    /// Pure, so checkpoints and status displays can call it at any point in
    /// a turn without disturbing what the next request will carry.
    fn context_window_snapshot(&self, tail: &[Block]) -> ContextWindowSnapshot {
        let messages = self.context.build_request(tail);
        let metrics = self.context_window_metrics(&messages);
        ContextWindowSnapshot {
            phase: self.classify_context_window_phase(&metrics),
            metrics,
            messages,
        }
    }

    fn context_window_metrics(&self, messages: &[Message]) -> ContextWindowMetrics {
        let mut budget = self.budget;
        budget.observe_assembled(request_chars(messages, &self.tools.active_definitions()));
        ContextWindowMetrics {
            usage_fraction: budget.usage(),
            context_window: budget.window(),
            estimated_tokens: budget.estimated_tokens(),
            chars_per_token: budget.chars_per_token(),
            system_block_count: self.context.system_block_count(),
            recent_message_count: self.context.len_recent(),
            recent_system_message_count: self.context.count_recent_system_messages(),
            has_handoff: self.context.handoff().is_some(),
            has_staged_system_updates: self.context.has_staged_system_updates(),
            prompt_cache_epoch: self.prompt_cache.epoch(),
        }
    }

    fn classify_context_window_phase(&self, metrics: &ContextWindowMetrics) -> ContextWindowPhase {
        let policy = &self.config.context_policy;
        if policy.should_pause(metrics.estimated_tokens, metrics.context_window) {
            return ContextWindowPhase::Exhausted;
        }
        if self.compaction.is_active() {
            return ContextWindowPhase::Compacting;
        }
        if metrics.has_staged_system_updates
            || (self.compaction_model.is_some() && self.warmth.has_expired())
        {
            return ContextWindowPhase::ReassemblyDue;
        }
        ContextWindowPhase::Stable
    }

    /// Returns the exact request message sequence that would be sent for the next turn.
    ///
    /// Intended for observability and debug UIs: the prompt is added to a
    /// forked context and the sources are rendered without committing, so a
    /// preview cannot consume a change the real request should surface.
    ///
    /// # Errors
    ///
    /// Returns an error when the model profiles cannot be fetched during
    /// initialization.
    pub async fn preview_request_messages(
        &mut self,
        prompt: &str,
        attachments: impl IntoIterator<Item = Attachment>,
    ) -> Result<Vec<Message>, AgentError> {
        self.preview_request_messages_with(prompt, std::iter::empty(), attachments)
            .await
    }

    /// Returns the exact request message sequence that would be sent for the
    /// next turn, with `extra_messages` appended at the conversation tail
    /// before the prompt.
    ///
    /// This is the non-mutating counterpart of pushing those messages onto the
    /// live context and then running the turn — the shape a host gets for
    /// per-turn hidden context injected ahead of the user's message. The
    /// extras are not persisted: the context and the sources' committed state
    /// are left exactly as a normal first assembly would, because sources are
    /// only told about a request once it is actually sent.
    ///
    /// # Errors
    ///
    /// Returns an error when the model profiles cannot be fetched during
    /// initialization.
    pub async fn preview_request_messages_with(
        &mut self,
        prompt: &str,
        extra_messages: impl IntoIterator<Item = Message>,
        attachments: impl IntoIterator<Item = Attachment>,
    ) -> Result<Vec<Message>, AgentError> {
        self.ensure_initialized().await?;
        let checkpoint = self.context.checkpoint();
        self.context.extend(extra_messages);
        self.context
            .push(Message::user(prompt).with_attachments(attachments));
        let messages = self.context.build_request(&self.last_tail);
        self.context.restore(checkpoint);
        Ok(messages)
    }

    /// Tells the embedder the agent's state changed, so it can persist it
    /// with [`export_checkpoint`](Self::export_checkpoint).
    fn emit_checkpoint(&self, cx: &RunCx<'_>, reason: CheckpointReason, turn: usize) {
        let window = self.context_window_snapshot(&self.last_tail);
        cx.events.emit(AgentEvent::Checkpoint {
            run: cx.run.clone(),
            reason,
            turn,
            phase: window.phase,
            message_count: self.context.len_recent(),
        });
    }
}

fn format_builtin_tool_result(tool: &str, result: &str) -> String {
    format!("[{tool}] {result}\n")
}

pub fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests;
