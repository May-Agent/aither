//! The facts a shell raises about its jobs.
//!
//! A [`ShellEvent`] carries no text: the embedder turns each fact into the
//! alert, reminder or wake its agent reads. Every event reaches exactly one
//! consumer of the [`ShellEvents`] it was raised on: handles share their
//! queues, so an embedder that aggregates the jobs of several agent rebuilds
//! reads them all from one handle.

use std::time::Duration;

use aither_core::llm::StoredFile;
use async_channel::{Receiver, Sender};
use futures_lite::future::or;

use crate::job_registry::PendingCompletion;
use crate::monitor::{MonitorEnded, MonitorIncrement, MonitorStalled};
use crate::terminal::{CompletedCommand, DescendantsKilled, TerminalError, TimeoutKill};

/// A job the shell has just detached: started with `background`, or promoted
/// past [`ShellLimits::foreground_wait`](crate::ShellLimits).
///
/// A job raises exactly one.
#[derive(Debug, Clone)]
pub struct JobStarted {
    /// Task ID of the job, a word leased from the shell's id allocator.
    pub task_id: String,
    /// First lines of the redirected output; empty when nothing was written
    /// yet.
    pub output_preview: String,
    /// File receiving the full redirected output.
    pub output_file: StoredFile,
}

/// A job that ran to its end: one promoted past the foreground window or
/// started in the background. A script whose call returned its result, and
/// every [`Shell::run`](crate::Shell::run), raises none.
#[derive(Debug)]
pub struct JobEnded {
    /// The job as it ended: its task id (a word leased from the shell's id
    /// allocator), script, status with the cause of a kill, and the path of
    /// its stored output.
    pub info: crate::JobInfo,
    /// The result of the execution: the output as the model reads it and
    /// the outages the job ran through.
    pub result: Result<CompletedCommand, TerminalError>,
    /// Set when the kill was the job's `timeout` bound passing.
    pub timeout_kill: Option<TimeoutKill>,
    /// Keeps the job in its [`Jobs`](crate::Jobs), and its task id leased,
    /// until the consumer takes the completion by dropping it; `None` when a
    /// waiter ([`Job::wait`](crate::Job::wait)) took it first, so nothing is
    /// owed to the agent.
    pub pending: Option<PendingCompletion>,
}

/// A job that crossed the long-running threshold; raised once per job.
#[derive(Debug, Clone)]
pub struct LongRunning {
    /// Task ID of the job.
    pub task_id: String,
    /// The script the job runs.
    pub script: String,
    /// How long the job had been running.
    pub after: Duration,
}

/// A job whose stdin has a reader waiting for input; raised once per
/// blocking episode, so again only after the job was fed and blocks anew.
#[derive(Debug, Clone)]
pub struct StdinBlocked {
    /// Task ID of the job.
    pub task_id: String,
    /// The script the job runs.
    pub script: String,
}

/// A job whose stdin watch failed; raised once, after which the job can no
/// longer report being blocked on stdin.
#[derive(Debug, Clone)]
pub struct StdinWatchFailed {
    /// Task ID of the job.
    pub task_id: String,
    /// The script the job runs.
    pub script: String,
    /// Why the watch failed.
    pub error: String,
}

/// A fact about a job of a shell.
#[derive(Debug)]
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
/// Events of different kinds queue apart, so a consumer reading one kind
/// never waits behind another; [`ShellEvents`] reads them back in the order
/// a job lives them: started, then the facts about it, then its end.
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

/// The reading half of a shell's events.
///
/// Cloning shares the queues: an event goes to one of the clones that read.
#[derive(Clone, Debug)]
pub struct ShellEvents {
    started: Receiver<ShellEvent>,
    facts: Receiver<ShellEvent>,
    ended: Receiver<ShellEvent>,
}

/// A raising half and the reading half of one shell's events.
pub fn event_channel() -> (EventSender, ShellEvents) {
    let (started_tx, started) = async_channel::unbounded();
    let (facts_tx, facts) = async_channel::unbounded();
    let (ended_tx, ended) = async_channel::unbounded();
    (
        EventSender {
            started: started_tx,
            facts: facts_tx,
            ended: ended_tx,
        },
        ShellEvents {
            started,
            facts,
            ended,
        },
    )
}

impl ShellEvents {
    /// The next event already raised, without waiting.
    #[must_use]
    pub fn try_recv(&self) -> Option<ShellEvent> {
        self.started
            .try_recv()
            .or_else(|_| self.facts.try_recv())
            .or_else(|_| self.ended.try_recv())
            .ok()
    }

    /// Waits for the next event; `None` once the shell is gone and every
    /// event was read.
    ///
    /// Cancel-safe: an event is taken only when this future completes, so a
    /// caller racing it against other work loses nothing.
    pub async fn recv(&self) -> Option<ShellEvent> {
        loop {
            if let Some(event) = self.try_recv() {
                return Some(event);
            }
            let any = or(
                self.started.recv(),
                or(self.facts.recv(), self.ended.recv()),
            )
            .await;
            match any {
                Ok(event) => return Some(event),
                Err(_) if self.is_closed() => return None,
                // One queue closed: the others may still be open or hold
                // events.
                Err(_) => futures_lite::future::yield_now().await,
            }
        }
    }

    /// Waits for the next event, for at most `duration`.
    pub async fn recv_timeout(&self, duration: Duration) -> Option<ShellEvent> {
        or(self.recv(), async {
            async_io::Timer::after(duration).await;
            None
        })
        .await
    }

    /// Whether no event is waiting.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.started.is_empty() && self.facts.is_empty() && self.ended.is_empty()
    }

    fn is_closed(&self) -> bool {
        self.started.is_closed()
            && self.started.is_empty()
            && self.facts.is_closed()
            && self.facts.is_empty()
            && self.ended.is_closed()
            && self.ended.is_empty()
    }

    /// The events of one kind, for tests that assert on a single stream of
    /// facts.
    #[cfg(test)]
    pub(crate) fn take_started(&self) -> Vec<JobStarted> {
        std::iter::from_fn(|| self.started.try_recv().ok())
            .filter_map(|event| match event {
                ShellEvent::JobStarted(started) => Some(started),
                _ => None,
            })
            .collect()
    }

    /// Waits for the next job to end, leaving every other event queued.
    #[cfg(test)]
    pub(crate) async fn recv_completed(&self) -> Option<JobEnded> {
        match self.ended.recv().await.ok()? {
            ShellEvent::JobEnded(ended) => Some(*ended),
            _ => unreachable!("the ended queue holds ended jobs"),
        }
    }

    /// The jobs that ended and were not yet read.
    #[cfg(test)]
    pub(crate) fn take_completed(&self) -> Vec<JobEnded> {
        std::iter::from_fn(|| self.ended.try_recv().ok())
            .filter_map(|event| match event {
                ShellEvent::JobEnded(ended) => Some(*ended),
                _ => None,
            })
            .collect()
    }

    /// The facts raised and not yet read.
    #[cfg(test)]
    pub(crate) fn take_job_alerts(&self) -> Vec<ShellEvent> {
        std::iter::from_fn(|| self.facts.try_recv().ok()).collect()
    }
}
