//! The facts a shell raises about its jobs.
//!
//! A [`ShellEvent`] carries no text: the embedder turns each fact into the
//! alert, reminder or wake its agent reads. Every event reaches exactly one
//! reader of the [`ShellEvents`] it was raised on: clones share their queues,
//! so an embedder that aggregates the jobs of several shells hands each the
//! same [`ShellEvents`] and reads them all from one place.

use std::time::Duration;

use aither_core::llm::{CallIdentity, StoredFile};
use async_channel::{Receiver, Sender};
use futures_lite::future::or;

use crate::job_registry::PendingCompletion;
use crate::monitor::{MonitorEnded, MonitorIncrement, MonitorStalled};
use crate::process_tree::JobProcess;
use crate::terminal::{CompletedCommand, TerminalError, TimeoutKill};

/// A job the shell has just detached: started with `background`, or promoted
/// past [`ShellLimits::foreground_wait`](crate::ShellLimits).
///
/// A job raises exactly one.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct JobStarted {
    pub(crate) task_id: String,
    pub(crate) output_preview: String,
    pub(crate) output_file: StoredFile,
}

impl JobStarted {
    /// Task ID of the job, a word leased from the shell's id allocator.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// First lines of the redirected output; empty when nothing was written
    /// yet.
    #[must_use]
    pub fn output_preview(&self) -> &str {
        &self.output_preview
    }

    /// File receiving the full redirected output.
    #[must_use]
    pub const fn output_file(&self) -> &StoredFile {
        &self.output_file
    }
}

/// A job that ran to its end: one promoted past the foreground window or
/// started in the background. A script whose call returned its result, and
/// every [`Shell::run`](crate::Shell::run), raises none.
#[derive(Debug)]
#[non_exhaustive]
pub struct JobEnded {
    pub(crate) info: crate::JobInfo,
    pub(crate) result: Result<CompletedCommand, TerminalError>,
    pub(crate) timeout_kill: Option<TimeoutKill>,
    pub(crate) pending: Option<PendingCompletion>,
}

impl JobEnded {
    /// The job as it ended: its task id (a word leased from the shell's id
    /// allocator), script, status with the cause of a kill, and the path of
    /// its stored output.
    #[must_use]
    pub const fn info(&self) -> &crate::JobInfo {
        &self.info
    }

    /// The result of the execution: the output as the model reads it and
    /// the outages the job ran through.
    #[must_use]
    pub const fn result(&self) -> &Result<CompletedCommand, TerminalError> {
        &self.result
    }

    /// Set when the kill was the job's `timeout` bound passing.
    #[must_use]
    pub const fn timeout_kill(&self) -> Option<&TimeoutKill> {
        self.timeout_kill.as_ref()
    }

    /// The job, split into what it ended as, its result, the bound that
    /// killed it and the hold that keeps it in its [`Jobs`](crate::Jobs).
    ///
    /// The hold keeps the job listed, and its task id leased, until the
    /// consumer takes the completion by dropping it; it is `None` when a
    /// waiter ([`Job::wait`](crate::Job::wait)) took the completion first, so
    /// nothing is owed to the agent.
    pub fn into_parts(
        self,
    ) -> (
        crate::JobInfo,
        Result<CompletedCommand, TerminalError>,
        Option<TimeoutKill>,
        Option<PendingCompletion>,
    ) {
        (self.info, self.result, self.timeout_kill, self.pending)
    }
}

/// A job that crossed the long-running threshold; raised once per job.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LongRunning {
    pub(crate) task_id: String,
    pub(crate) script: String,
    pub(crate) after: Duration,
}

impl LongRunning {
    /// Task ID of the job.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// The script the job runs.
    #[must_use]
    pub fn script(&self) -> &str {
        &self.script
    }

    /// How long the job had been running.
    #[must_use]
    pub const fn after(&self) -> Duration {
        self.after
    }
}

/// A job whose stdin has a reader waiting for input; raised once per
/// blocking episode, so again only after the job was fed and blocks anew.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StdinBlocked {
    pub(crate) task_id: String,
    pub(crate) script: String,
}

impl StdinBlocked {
    /// Task ID of the job.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// The script the job runs.
    #[must_use]
    pub fn script(&self) -> &str {
        &self.script
    }
}

/// A job whose stdin watch failed; raised once, after which the job can no
/// longer report being blocked on stdin.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StdinWatchFailed {
    pub(crate) task_id: String,
    pub(crate) script: String,
    pub(crate) error: String,
}

impl StdinWatchFailed {
    /// Task ID of the job.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// The script the job runs.
    #[must_use]
    pub fn script(&self) -> &str {
        &self.script
    }

    /// Why the watch failed.
    #[must_use]
    pub fn error(&self) -> &str {
        &self.error
    }
}

/// Processes a foreground command left running when its shell exited, killed
/// with it.
///
/// Raised once per command, at the moment they were killed.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DescendantsKilled {
    pub(crate) task_id: String,
    pub(crate) call: Option<CallIdentity>,
    pub(crate) processes: Vec<JobProcess>,
}

impl DescendantsKilled {
    /// Task ID of the command.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// The tool call that ran the command, when it came from one.
    #[must_use]
    pub const fn call(&self) -> Option<&CallIdentity> {
        self.call.as_ref()
    }

    /// Every process killed.
    #[must_use]
    pub fn processes(&self) -> &[JobProcess] {
        &self.processes
    }
}

/// A fact about a job of a shell.
#[derive(Debug)]
#[non_exhaustive]
pub enum ShellEvent {
    /// A job was detached as a background task.
    JobStarted(JobStarted),
    /// A job ended; boxed to keep the enum small.
    JobEnded(Box<JobEnded>),
    /// A job crossed the long-running threshold.
    LongRunning(LongRunning),
    /// A job is waiting for input on its stdin.
    StdinBlocked(StdinBlocked),
    /// Watching a job's stdin failed.
    StdinWatchFailed(StdinWatchFailed),
    /// A foreground command's shell exited while descendants were still
    /// running, and they were killed.
    DescendantsKilled(DescendantsKilled),
    /// A monitor collected new output lines from its watched job.
    MonitorIncrement(MonitorIncrement),
    /// A monitor saw no new output for its stall window.
    MonitorStalled(MonitorStalled),
    /// A monitor ended, reporting why.
    MonitorEnded(MonitorEnded),
}

/// The raising half of a shell's events, cloned into every task that reports
/// a fact.
///
/// Events of different kinds queue apart, so a reader of one kind never waits
/// behind another; [`ShellEvents`] reads them back in the order a job lives
/// them: started, then the facts about it, then its end.
#[derive(Clone, Debug)]
pub struct EventSender {
    started: Sender<ShellEvent>,
    facts: Sender<ShellEvent>,
    ended: Sender<ShellEvent>,
}

impl EventSender {
    /// Raises `event`.
    ///
    /// # Errors
    ///
    /// Returns the event when every [`ShellEvents`] of its queue is gone.
    pub(crate) async fn send(
        &self,
        event: ShellEvent,
    ) -> Result<(), async_channel::SendError<ShellEvent>> {
        match event {
            ShellEvent::JobStarted(_) => self.started.send(event).await,
            ShellEvent::JobEnded(_) => self.ended.send(event).await,
            ShellEvent::LongRunning(_)
            | ShellEvent::StdinBlocked(_)
            | ShellEvent::StdinWatchFailed(_)
            | ShellEvent::DescendantsKilled(_)
            | ShellEvent::MonitorIncrement(_)
            | ShellEvent::MonitorStalled(_)
            | ShellEvent::MonitorEnded(_) => self.facts.send(event).await,
        }
    }
}

/// One queue of events: both halves, so it never closes while a
/// [`ShellEvents`] holds it.
#[derive(Clone, Debug)]
struct Queue {
    tx: Sender<ShellEvent>,
    rx: Receiver<ShellEvent>,
}

impl Queue {
    fn new() -> Self {
        let (tx, rx) = async_channel::unbounded();
        Self { tx, rx }
    }
}

/// The events of one or several shells, and where they are read.
///
/// Cloning shares the queues: an event goes to one of the clones that read. A
/// handle owns both halves of its queues, so they never close: [`recv`]
/// waits for as long as the embedder keeps the handle.
///
/// A shell built [with a handle](crate::ShellBuilder::events) raises its
/// events on it, and so does every subagent shell it launches.
///
/// [`recv`]: Self::recv
#[derive(Clone, Debug)]
pub struct ShellEvents {
    started: Queue,
    facts: Queue,
    ended: Queue,
}

impl Default for ShellEvents {
    fn default() -> Self {
        Self::new()
    }
}

impl ShellEvents {
    /// A handle with no events raised on it yet.
    #[must_use]
    pub fn new() -> Self {
        Self {
            started: Queue::new(),
            facts: Queue::new(),
            ended: Queue::new(),
        }
    }

    /// A raising half of these queues.
    pub(crate) fn sender(&self) -> EventSender {
        EventSender {
            started: self.started.tx.clone(),
            facts: self.facts.tx.clone(),
            ended: self.ended.tx.clone(),
        }
    }

    /// The next event already raised, without waiting.
    #[must_use]
    pub fn try_recv(&self) -> Option<ShellEvent> {
        self.started
            .rx
            .try_recv()
            .or_else(|_| self.facts.rx.try_recv())
            .or_else(|_| self.ended.rx.try_recv())
            .ok()
    }

    /// Waits for the next event.
    ///
    /// Cancel-safe: an event is taken only when this future completes, so a
    /// caller racing it against other work loses nothing.
    ///
    /// # Panics
    /// Never in practice: the handle holds the senders of its own queues, so
    /// they cannot close.
    pub async fn recv(&self) -> ShellEvent {
        or(
            self.started.rx.recv(),
            or(self.facts.rx.recv(), self.ended.rx.recv()),
        )
        .await
        .expect("a handle holds the senders of its own queues")
    }

    /// Waits for the next event, for at most `duration`.
    pub async fn recv_timeout(&self, duration: Duration) -> Option<ShellEvent> {
        or(async { Some(self.recv().await) }, async {
            async_io::Timer::after(duration).await;
            None
        })
        .await
    }

    /// Whether no event is waiting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.started.rx.is_empty() && self.facts.rx.is_empty() && self.ended.rx.is_empty()
    }

    /// The events of one kind, for tests that assert on a single stream of
    /// facts.
    #[cfg(test)]
    pub(crate) fn take_started(&self) -> Vec<JobStarted> {
        std::iter::from_fn(|| self.started.rx.try_recv().ok())
            .filter_map(|event| match event {
                ShellEvent::JobStarted(started) => Some(started),
                _ => None,
            })
            .collect()
    }

    /// Waits for the next job to end, leaving every other event queued.
    #[cfg(test)]
    pub(crate) async fn recv_completed(&self) -> JobEnded {
        match self
            .ended
            .rx
            .recv()
            .await
            .expect("a handle holds the senders of its own queues")
        {
            ShellEvent::JobEnded(ended) => *ended,
            _ => unreachable!("the ended queue holds ended jobs"),
        }
    }

    /// The jobs that ended and were not yet read.
    #[cfg(test)]
    pub(crate) fn take_completed(&self) -> Vec<JobEnded> {
        std::iter::from_fn(|| self.ended.rx.try_recv().ok())
            .filter_map(|event| match event {
                ShellEvent::JobEnded(ended) => Some(*ended),
                _ => None,
            })
            .collect()
    }

    /// The facts raised and not yet read.
    #[cfg(test)]
    pub(crate) fn take_job_alerts(&self) -> Vec<ShellEvent> {
        std::iter::from_fn(|| self.facts.rx.try_recv().ok()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::MonitorStalled;

    fn stalled(task_id: &str) -> ShellEvent {
        ShellEvent::MonitorStalled(MonitorStalled {
            task_id: task_id.to_string(),
            stall_secs: 1,
        })
    }

    #[tokio::test]
    async fn a_handle_waits_for_events_it_never_loses_the_queues_of() {
        let events = ShellEvents::new();
        let reader = events.clone();
        // Nothing is raised: the read waits, it neither ends nor spins.
        assert!(
            reader
                .recv_timeout(Duration::from_millis(50))
                .await
                .is_none()
        );
        events.sender().send(stalled("late")).await.unwrap();
        let ShellEvent::MonitorStalled(event) = reader.recv().await else {
            panic!("the raised event is the one read");
        };
        assert_eq!(event.task_id(), "late");
    }

    #[tokio::test]
    async fn senders_of_one_handle_reach_every_reader_of_it_once() {
        let events = ShellEvents::new();
        let (first, second) = (events.sender(), events.sender());
        first.send(stalled("one")).await.unwrap();
        second.send(stalled("two")).await.unwrap();
        let reader = events.clone();
        assert!(
            matches!(events.try_recv(), Some(ShellEvent::MonitorStalled(e)) if e.task_id() == "one")
        );
        assert!(
            matches!(reader.try_recv(), Some(ShellEvent::MonitorStalled(e)) if e.task_id() == "two")
        );
        assert!(events.is_empty());
    }
}
