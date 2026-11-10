//! What a run emits: the events that observe it.
//!
//! Events observe; [`Hooks`](crate::Hooks) decide. Every enum, every variant
//! and every struct here is `#[non_exhaustive]`, and none has a public
//! constructor, because only aither emits them: a consumer matches with a
//! wildcard arm and keeps working when a release adds an event.
//!
//! Every event reaches the consumer as a [`TimedEvent`], which names the agent
//! that emitted it by its [`AgentPath`]: a subagent's events are forwarded
//! on its parent's stream, tagged with the child's path.

use core::time::Duration;
use std::fmt;

use aither_core::llm::tool::{ApprovalKind, Progress};
use aither_core::llm::{AgentIdentity, ToolCallId, ToolResult, Usage};
use jiff::Timestamp;

use crate::context_window::ContextWindowPhase;
use crate::error::AgentError;

/// Where an agent stands in the tree of agents a run started: empty for the
/// main agent, the subagent's name beneath its parent's path otherwise.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct AgentPath(Vec<String>);

impl AgentPath {
    /// The main agent.
    #[must_use]
    pub const fn main() -> Self {
        Self(Vec::new())
    }

    /// The path of the agent `identity` names, beneath the agent that
    /// started it.
    #[must_use]
    pub fn of(identity: &AgentIdentity) -> Self {
        match identity {
            AgentIdentity::Main => Self::main(),
            AgentIdentity::Subagent(subagent) => {
                let mut path = subagent
                    .started_by()
                    .map_or_else(Self::main, |call| Self::of(call.agent()));
                path.0.push(subagent.name().to_string());
                path
            }
        }
    }

    /// The path of a child named `name` beneath this agent.
    #[must_use]
    pub fn child(&self, name: impl Into<String>) -> Self {
        let mut path = self.clone();
        path.0.push(name.into());
        path
    }

    /// The names from the main agent down to this one.
    #[must_use]
    pub fn segments(&self) -> &[String] {
        &self.0
    }

    /// Whether this is the main agent.
    #[must_use]
    pub const fn is_main(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for AgentPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("main");
        }
        f.write_str(&self.0.join("/"))
    }
}

macro_rules! text_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(String);

        impl $name {
            pub(crate) fn new(id: impl Into<String>) -> Self {
                Self(id.into())
            }

            /// The identifier as text.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

macro_rules! counter_id {
    ($(#[$meta:meta])* $name:ident($inner:ty)) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name($inner);

        impl $name {
            pub(crate) const fn new(id: $inner) -> Self {
                Self(id)
            }

            /// The identifier's number.
            #[must_use]
            pub const fn get(self) -> $inner {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

text_id! {
    /// Identifies one run of an agent: from [`Agent::run`](crate::Agent::run)
    /// or [`Agent::resume`](crate::Agent::resume) to its
    /// [`RunEnded`](AgentEvent::RunEnded).
    RunId
}

/// Identifies a [context source](crate::ContextSource) in
/// [`ContextInjected`](AgentEvent::ContextInjected).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(String);

impl SourceId {
    /// Names a source.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The name as text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

counter_id! {
    /// Identifies one request sent to the model, unique within an agent. A
    /// request sent again after its response stream broke has a new id.
    RequestId(u64)
}

counter_id! {
    /// Identifies one block of a response within its request.
    BlockId(u32)
}

counter_id! {
    /// Identifies one pause, unique within an agent.
    PauseId(u64)
}

/// What a streamed block of the model's response is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OutputKind {
    /// Text for the reader.
    Text,
    /// The model's reasoning.
    Reasoning,
}

/// Where a tool call runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ToolOrigin {
    /// A tool registered on the agent, run by the agent.
    Local,
    /// A tool the provider runs itself; nothing runs locally.
    Hosted,
}

/// What the agent settled on for a tool call before running it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ToolVerdict {
    /// The call runs as the model made it.
    Allowed,
    /// The call does not run; the model is told `reason`.
    Denied {
        /// Why the call was refused.
        reason: String,
    },
    /// The call runs with arguments a hook replaced.
    Rewritten,
    /// The call waits on the approver; a [`Paused`](AgentEvent::Paused)
    /// follows when the answer is not immediate, and the verdict is
    /// [`Allowed`](Self::Allowed) or [`Denied`](Self::Denied) once it comes.
    AwaitingApproval,
}

/// Why a run stopped short of an answer without failing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RunPause {
    /// The context window is full and the run cannot continue.
    ///
    /// Emitted when usage reaches the
    /// [pause point](crate::ContextPolicy::pause_point) and no compaction can
    /// relieve it: there was nothing to compact, or the compaction landed and
    /// the window is still full. Sending the request anyway would have the
    /// provider reject it for length. A compaction that fails twice is a
    /// failed run, not this pause.
    ContextExhausted,
}

/// How a run ended.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RunOutcome {
    /// The model answered and nothing held the run.
    Completed {
        /// The model's final reply.
        final_text: String,
    },
    /// The run stopped for a reason that needs the embedder.
    Paused {
        /// Why.
        pause: RunPause,
    },
    /// [`Run::cancel`](crate::Run::cancel) was called, or the run was dropped.
    Cancelled,
    /// The run failed; [`Run::drive`](crate::Run::drive) returns the error.
    Failed {
        /// What went wrong.
        error: AgentError,
    },
    /// The model kept calling tools past
    /// [`max_iterations`](crate::AgentConfig::max_iterations).
    IterationLimit,
}

/// How a request ended.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RequestOutcome {
    /// The response stream ended.
    Completed,
    /// The response stream broke after it opened, and the same request is
    /// sent again after `delay` as a new request.
    ///
    /// Everything streamed since the request started — text, reasoning,
    /// tool-call deltas and hosted tool calls — belonged to the broken
    /// attempt: none of it enters the conversation and none of its tool calls
    /// run. A client drops what it rendered of that attempt.
    Retrying {
        /// Why the stream broke.
        error: AgentError,
        /// Which retry this is, counting from 1.
        retry: u32,
        /// The most retries the model's
        /// [`RetryPolicy`](aither_core::llm::RetryPolicy) allows.
        max_retries: u32,
        /// How long the agent waits before sending the request again.
        delay: Duration,
    },
    /// The request failed and the run fails with the error.
    Failed {
        /// What went wrong.
        error: AgentError,
    },
}

/// Why a compaction started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompactionTrigger {
    /// [`Agent::compact`](crate::Agent::compact) was called.
    Requested,
    /// The window crossed the policy's
    /// [start point](crate::ContextPolicy::start_point); the compaction is
    /// forked alongside the run.
    WindowFilling,
    /// The window reached the pause point and the run made room.
    PausePoint,
    /// [`Agent::compact_when_idle`](crate::Agent::compact_when_idle) ran
    /// shortly before the cache expired.
    Idle,
    /// A send found the cache cold and a compaction model set.
    ColdCache,
}

/// How a compaction announced by [`AgentEvent::CompactionStarted`] ended.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CompactionOutcome {
    /// The handoff replaced the conversation below its fork point.
    Applied(crate::CompactResult),
    /// There was no conversation to compact; the window is as it was.
    NothingToCompact,
    /// The compaction failed; the window is as it was.
    Failed(AgentError),
    /// The handoff was dropped unapplied: the window it was forked from was
    /// replaced first, by an explicit compaction,
    /// [`clear_history`](crate::Agent::clear_history),
    /// [`restore_checkpoint`](crate::Agent::restore_checkpoint), or by
    /// dropping the run that carried it.
    Discarded,
}

/// What an injection into the conversation was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InjectionKind {
    /// A durable block appended to the conversation, which stays in the
    /// cached prefix from the next request on.
    Stable,
    /// A block rendered into one request's volatile tail.
    Tail,
    /// The diff alert of a live source whose content changed.
    LiveChange,
}

/// Why a checkpoint was emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CheckpointReason {
    /// A turn that called tools ended.
    TurnBoundary,
    /// The run is stopping.
    Stop,
}

/// Events emitted during agent execution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AgentEvent {
    /// A run began.
    #[non_exhaustive]
    RunStarted {
        /// The run.
        run: RunId,
        /// The agent that runs it.
        agent: AgentPath,
    },

    /// A run ended; the last event of the run.
    #[non_exhaustive]
    RunEnded {
        /// The run.
        run: RunId,
        /// How it ended.
        outcome: RunOutcome,
    },

    /// A turn began: one request to the model and the tool calls it makes.
    #[non_exhaustive]
    TurnStarted {
        /// The run.
        run: RunId,
        /// The turn, counting from 1.
        turn: usize,
    },

    /// A turn ended.
    #[non_exhaustive]
    TurnEnded {
        /// The run.
        run: RunId,
        /// The turn.
        turn: usize,
        /// How many tool calls the turn made.
        tool_calls: usize,
    },

    /// A request to the model was sent.
    #[non_exhaustive]
    RequestStarted {
        /// The request.
        request: RequestId,
        /// The hash chain of its prompt prefix and what rewrote that prefix
        /// since the previous request, so a host can tell where a cache miss
        /// began.
        prefix: crate::context::PromptPrefix,
    },

    /// A request ended.
    #[non_exhaustive]
    RequestEnded {
        /// The request.
        request: RequestId,
        /// The usage the provider reported for it, if it did.
        usage: Option<Usage>,
        /// How it ended.
        outcome: RequestOutcome,
    },

    /// A chunk of a block of the response.
    #[non_exhaustive]
    BlockDelta {
        /// The request.
        request: RequestId,
        /// The block within the request.
        block: BlockId,
        /// What the block is.
        kind: OutputKind,
        /// The chunk.
        text: String,
    },

    /// A block of the response is complete.
    #[non_exhaustive]
    BlockEnded {
        /// The request.
        request: RequestId,
        /// The block.
        block: BlockId,
        /// What the block is.
        kind: OutputKind,
    },

    /// The model is streaming a tool call's name and arguments, before the
    /// call is assembled.
    #[non_exhaustive]
    ToolCallDelta {
        /// The call.
        call: ToolCallId,
        /// The tool; empty in the earliest deltas of some providers.
        name: String,
        /// The next fragment of the JSON arguments.
        arguments_fragment: String,
    },

    /// A tool call was made.
    #[non_exhaustive]
    ToolCallStarted {
        /// The call.
        call: ToolCallId,
        /// The tool.
        name: String,
        /// The arguments as the model made them; the detail of a hosted
        /// call as a string.
        arguments: serde_json::Value,
        /// Where the call runs.
        origin: ToolOrigin,
    },

    /// The agent settled whether the call runs, and how.
    #[non_exhaustive]
    ToolCallDecided {
        /// The call.
        call: ToolCallId,
        /// What was decided.
        decision: ToolVerdict,
    },

    /// A running tool reported progress.
    #[non_exhaustive]
    ToolProgress {
        /// The call.
        call: ToolCallId,
        /// The tool's typed report.
        progress: Progress,
    },

    /// A tool call ended.
    #[non_exhaustive]
    ToolCallEnded {
        /// The call.
        call: ToolCallId,
        /// What the model reads of it.
        result: ToolResult,
    },

    /// Work is waiting on something outside the agent: an approval, or a
    /// question a tool put through the approver under a kind of its own.
    #[non_exhaustive]
    Paused {
        /// The pause.
        pause: PauseId,
        /// The tool call that waits, if a tool call does.
        call: Option<ToolCallId>,
        /// What is waited for, as the embedder named it.
        kind: ApprovalKind,
        /// What the approval is about.
        subject: String,
    },

    /// The pause ended.
    #[non_exhaustive]
    Resumed {
        /// The pause.
        pause: PauseId,
    },

    /// A compaction started: one forked alongside the run, or one run by
    /// [`Agent::compact`](crate::Agent::compact),
    /// [`Agent::compact_when_idle`](crate::Agent::compact_when_idle) or the
    /// run itself at the pause point or on a cold cache.
    ///
    /// Exactly one [`CompactionEnded`](Self::CompactionEnded) follows, and no
    /// other compaction starts before it. It is emitted on the stream of
    /// whatever ends the compaction, which for one forked alongside a run may
    /// be a later run or explicit compaction; one ended where no stream is
    /// open finishes as [`CompactionOutcome::Discarded`] first thing on the
    /// next stream the agent opens.
    #[non_exhaustive]
    CompactionStarted {
        /// Why it started.
        trigger: CompactionTrigger,
    },

    /// The compaction last [started](Self::CompactionStarted) ended.
    #[non_exhaustive]
    CompactionEnded {
        /// How it ended.
        outcome: CompactionOutcome,
    },

    /// A compaction failed with the window at the
    /// [pause point](crate::ContextPolicy::pause_point), so the context was
    /// cleaned and the compaction reruns on it now.
    ///
    /// The cleanup removes system alerts and reminders and moves long tool
    /// outputs to files. When the rerun fails too the run fails; otherwise
    /// the run continues on the compacted window.
    #[non_exhaustive]
    CompactionRecovery {
        /// Why the compaction failed.
        error: AgentError,
        /// System alerts and reminders the cleanup removed.
        alerts_removed: usize,
        /// Long tool outputs the cleanup moved to files.
        outputs_spilled: usize,
    },

    /// A context source or a hook put a block into the conversation.
    #[non_exhaustive]
    ContextInjected {
        /// Who injected it.
        source: SourceId,
        /// What kind of injection it was.
        kind: InjectionKind,
    },

    /// A subagent began. Its own events follow, tagged with its path.
    #[non_exhaustive]
    SubagentStarted {
        /// The child's path.
        child: AgentPath,
        /// The child's name.
        name: String,
        /// What it was asked to do.
        prompt: String,
    },

    /// A subagent ended.
    #[non_exhaustive]
    SubagentEnded {
        /// The child's path.
        child: AgentPath,
        /// How its run ended.
        outcome: RunOutcome,
    },

    /// A snapshot of the agent's state was taken after it changed.
    #[non_exhaustive]
    Checkpoint {
        /// The run.
        run: RunId,
        /// Why the checkpoint was emitted.
        reason: CheckpointReason,
        /// The turn.
        turn: usize,
        /// The lifecycle phase of the assembled context window.
        phase: ContextWindowPhase,
        /// Number of recent conversation messages currently in memory.
        message_count: usize,
    },
}

/// An [`AgentEvent`] with the UTC time it was emitted and the agent that
/// emitted it.
///
/// The time is taken at the emission point inside the run, not when the
/// consumer reads the event.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TimedEvent {
    /// When the agent emitted the event.
    pub at: Timestamp,
    /// Which agent emitted it: the main agent, or a subagent whose events
    /// are forwarded on its parent's stream.
    pub agent: AgentPath,
    /// The event itself.
    pub event: AgentEvent,
}

impl AgentEvent {
    /// The run this event belongs to, for the events that carry one.
    #[must_use]
    pub const fn run(&self) -> Option<&RunId> {
        match self {
            Self::RunStarted { run, .. }
            | Self::RunEnded { run, .. }
            | Self::TurnStarted { run, .. }
            | Self::TurnEnded { run, .. }
            | Self::Checkpoint { run, .. } => Some(run),
            _ => None,
        }
    }
}
