//! Subagents as a mechanism: a child agent run on behalf of a parent's tool.
//!
//! aither defines no `subagent` command, no subagent types and no definition
//! files; the embedder writes the tool and decides what a child is. What
//! aither provides is the part every embedder needs and cannot write against
//! the loop's internals:
//!
//! - [`SubagentRuns`] records the children in flight, lists them and cancels
//!   one or all, without aborting the parent's turn;
//! - [`SubagentRun`] is one registered child: the identity its agent is
//!   built with, the signal that cancels it, and the sink its events go to;
//! - [`ChildAgent`] is anything that can be run as a child: an in-process
//!   [`Agent`](crate::Agent), or an external agent behind a protocol.
//!
//! The parent reports a child on its own stream: [`SubagentStarted`] and
//! [`SubagentEnded`] bracket the run, and every event of the child follows
//! with the child's [`AgentPath`], so one consumer sees the whole tree.
//!
//! [`SubagentStarted`]: crate::AgentEvent::SubagentStarted
//! [`SubagentEnded`]: crate::AgentEvent::SubagentEnded
//!
//! # Example
//!
//! ```rust,ignore
//! let runs = SubagentRuns::default();
//! let agent = Agent::builder(model).subagents(runs.clone()).tool(SubagentTool::new(runs)).build();
//!
//! // Inside the tool's `call`:
//! let started_by = cx.identity().expect("the agent loop identifies its calls");
//! let run = runs.start(started_by, "researcher", &prompt);
//! let mut child = Agent::builder(model).identity(run.identity()).build();
//! let outcome = run.run(&mut child, &prompt).await;
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use aither_core::LanguageModel;
use aither_core::llm::tool::CancelSignal;
use aither_core::llm::{AgentIdentity, CallIdentity, SubagentIdentity};

use crate::agent::{Agent, CancelHandle};
use crate::context::HandoffSchema;
use crate::event::{AgentEvent, AgentPath, RunOutcome, TimedEvent};
use crate::hook::Hooks;

/// An agent that can run as a child of another.
///
/// Implemented for [`Agent`]; an embedder implements it for an agent that
/// lives elsewhere, such as an external session behind a protocol. The child
/// reports its events to `events`, each stamped with its own path, and stops
/// when `cancel` fires.
pub trait ChildAgent: Send {
    /// Runs the child on `prompt` until it ends, and says how it ended.
    fn run(
        &mut self,
        prompt: &str,
        events: &ChildEvents,
        cancel: CancelSignal,
    ) -> impl Future<Output = RunOutcome> + Send;
}

impl<M, H, C, Schema> ChildAgent for Agent<M, H, C, Schema>
where
    Schema: HandoffSchema,
    M: LanguageModel + 'static,
    C: LanguageModel + 'static,
    H: Hooks,
{
    async fn run(
        &mut self,
        prompt: &str,
        events: &ChildEvents,
        cancel: CancelSignal,
    ) -> RunOutcome {
        let own = AgentPath::of(self.identity());
        let driven = Self::run(self, prompt, std::iter::empty())
            .drive(None, cancel.cancelled(), |ended, timed| {
                let outcome = match (&timed.event, timed.agent == own) {
                    (AgentEvent::RunEnded { outcome, .. }, true) => Some(outcome.clone()),
                    _ => ended,
                };
                events.forward(timed);
                std::future::ready(ControlFlow::<(), _>::Continue(outcome))
            })
            .await;
        match driven {
            Ok(ControlFlow::Continue(Some(outcome))) => outcome,
            Ok(ControlFlow::Continue(None) | ControlFlow::Break(())) => RunOutcome::Cancelled,
            Err(error) => RunOutcome::Failed { error },
        }
    }
}

/// Where a child's events go: the parent's stream.
#[derive(Debug, Clone)]
pub struct ChildEvents {
    sender: async_channel::Sender<TimedEvent>,
}

impl ChildEvents {
    /// Hands an event of the child to the parent's stream as it is: the
    /// event already names the child that emitted it.
    pub fn forward(&self, event: TimedEvent) {
        let _ = self.sender.try_send(event);
    }
}

/// Snapshot of one child in flight.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SubagentRunInfo {
    /// Unique run id within the owning [`SubagentRuns`].
    pub id: u64,
    /// The name the child was started under.
    pub name: String,
    /// The task prompt the child was given.
    pub prompt: String,
    /// When the child started.
    pub started: Instant,
}

#[derive(Debug)]
struct Slot {
    info: SubagentRunInfo,
    cancel: CancelHandle,
}

#[derive(Debug)]
struct Shared {
    runs: Mutex<HashMap<u64, Slot>>,
    next_id: AtomicU64,
    events: async_channel::Sender<TimedEvent>,
    incoming: async_channel::Receiver<TimedEvent>,
}

/// The registry of children in flight.
///
/// Create one, give a clone to the agent with
/// [`AgentBuilder::subagents`](crate::AgentBuilder::subagents) and one to the
/// tool that starts children. The agent forwards the events the registry
/// receives onto its own stream; a host lists and cancels children through
/// any clone.
#[derive(Debug, Clone)]
pub struct SubagentRuns {
    shared: Arc<Shared>,
}

impl Default for SubagentRuns {
    fn default() -> Self {
        let (events, incoming) = async_channel::unbounded();
        Self {
            shared: Arc::new(Shared {
                runs: Mutex::default(),
                next_id: AtomicU64::new(0),
                events,
                incoming,
            }),
        }
    }
}

impl SubagentRuns {
    /// Registers a child named `name` that the call `started_by` starts
    /// with `prompt`, and announces it on the parent's stream.
    ///
    /// # Panics
    ///
    /// Panics if the registry lock is poisoned.
    pub fn start(&self, started_by: &CallIdentity, name: &str, prompt: &str) -> SubagentRun {
        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let identity =
            AgentIdentity::Subagent(SubagentIdentity::new(name, Some(started_by.clone())));
        let parent = AgentPath::of(started_by.agent());
        let path = AgentPath::of(&identity);
        let cancel = CancelHandle::default();
        self.shared
            .runs
            .lock()
            .expect("subagent runs lock poisoned")
            .insert(
                id,
                Slot {
                    info: SubagentRunInfo {
                        id,
                        name: name.to_string(),
                        prompt: prompt.to_string(),
                        started: Instant::now(),
                    },
                    cancel: cancel.clone(),
                },
            );
        let run = SubagentRun {
            id,
            runs: self.clone(),
            parent,
            path,
            identity,
            cancel,
            ended: false,
        };
        run.announce(AgentEvent::SubagentStarted {
            child: run.path.clone(),
            name: name.to_string(),
            prompt: prompt.to_string(),
        });
        run
    }

    /// Every child in flight, ordered by id.
    ///
    /// # Panics
    ///
    /// Panics if the registry lock is poisoned.
    #[must_use]
    pub fn list(&self) -> Vec<SubagentRunInfo> {
        let mut infos: Vec<_> = self
            .shared
            .runs
            .lock()
            .expect("subagent runs lock poisoned")
            .values()
            .map(|slot| slot.info.clone())
            .collect();
        infos.sort_by_key(|info| info.id);
        infos
    }

    /// Cancels one child; the parent's turn goes on.
    ///
    /// Returns `false` only when `id` is unknown. A cancelled child keeps
    /// appearing in [`list`](Self::list) until it unwinds.
    ///
    /// # Panics
    ///
    /// Panics if the registry lock is poisoned.
    #[must_use]
    pub fn cancel(&self, id: u64) -> bool {
        self.shared
            .runs
            .lock()
            .expect("subagent runs lock poisoned")
            .get(&id)
            .map(|slot| slot.cancel.cancel())
            .is_some()
    }

    /// Cancels every child in flight; returns how many were signalled.
    ///
    /// # Panics
    ///
    /// Panics if the registry lock is poisoned.
    #[must_use]
    pub fn cancel_all(&self) -> usize {
        let runs = self
            .shared
            .runs
            .lock()
            .expect("subagent runs lock poisoned");
        for slot in runs.values() {
            slot.cancel.cancel();
        }
        runs.len()
    }

    /// The events children have reported and the parent has not taken yet.
    pub(crate) fn incoming(&self) -> &async_channel::Receiver<TimedEvent> {
        &self.shared.incoming
    }
}

/// One registered child.
///
/// Ends with [`run`](Self::run) or [`finish`](Self::finish); dropped before
/// that, it ends as cancelled, so the parent's stream always closes what it
/// opened.
#[derive(Debug)]
#[must_use = "a registered child ends when it is run, finished or dropped"]
pub struct SubagentRun {
    id: u64,
    runs: SubagentRuns,
    /// The path of the agent whose call started the child.
    parent: AgentPath,
    path: AgentPath,
    identity: AgentIdentity,
    cancel: CancelHandle,
    ended: bool,
}

impl SubagentRun {
    /// The id [`SubagentRuns::cancel`] takes.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// The identity to build the child agent with, so its events and tool
    /// calls name it and the call that started it.
    #[must_use]
    pub fn identity(&self) -> AgentIdentity {
        self.identity.clone()
    }

    /// The child's path.
    #[must_use]
    pub const fn path(&self) -> &AgentPath {
        &self.path
    }

    /// The signal that fires when the child is cancelled.
    #[must_use]
    pub fn cancel_signal(&self) -> CancelSignal {
        self.cancel.signal()
    }

    /// Where the child reports its events.
    #[must_use]
    pub fn events(&self) -> ChildEvents {
        ChildEvents {
            sender: self.runs.shared.events.clone(),
        }
    }

    /// Runs `child` on `prompt` to its end and closes the registration with
    /// the outcome.
    pub async fn run<C: ChildAgent>(self, child: &mut C, prompt: &str) -> RunOutcome {
        let outcome = child
            .run(prompt, &self.events(), self.cancel_signal())
            .await;
        self.finish(outcome.clone());
        outcome
    }

    /// Closes the registration with `outcome`, for a child that ran outside
    /// [`run`](Self::run).
    pub fn finish(mut self, outcome: RunOutcome) {
        self.close(outcome);
    }

    fn close(&mut self, outcome: RunOutcome) {
        if std::mem::replace(&mut self.ended, true) {
            return;
        }
        self.runs
            .shared
            .runs
            .lock()
            .expect("subagent runs lock poisoned")
            .remove(&self.id);
        self.announce(AgentEvent::SubagentEnded {
            child: self.path.clone(),
            outcome,
        });
    }

    /// Reports `event` on the parent's stream.
    fn announce(&self, event: AgentEvent) {
        let _ = self.runs.shared.events.try_send(TimedEvent {
            at: jiff::Timestamp::now(),
            agent: self.parent.clone(),
            event,
        });
    }
}

impl Drop for SubagentRun {
    fn drop(&mut self) {
        self.close(RunOutcome::Cancelled);
    }
}
