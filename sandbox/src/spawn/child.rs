//! The handle on a process a [`Sandbox`](crate::Sandbox) spawned.

use std::process::ExitStatus;
use std::sync::{Arc, Mutex};

use async_channel::{Receiver, Sender};

use super::ssh::RemoteJob;
use crate::process_tree::{JobProcess, JobTree, KillTreeError};
use crate::stdin_watch::StdinWatchSignal;

/// How a spawned process ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChildExit {
    /// The process exited, or a signal ended it, with this status.
    Exited(ExitStatus),
    /// The backend ended the process on [`SandboxedChild::kill`] and has no
    /// status for it (a container exec).
    Killed,
    /// Waiting for the process failed; the process may still be running.
    Failed(String),
}

impl ChildExit {
    /// Whether the process exited successfully.
    #[must_use]
    pub fn success(&self) -> bool {
        matches!(self, Self::Exited(status) if status.success())
    }
}

/// The processes a child left running that the backend ended with it, which
/// only a [`Foreground`](crate::TreeLifetime::Foreground) child has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Descendants {
    /// These were running when the process exited, and were killed; empty
    /// when it left none, or when its lifetime lets them run on.
    Killed(Vec<JobProcess>),
    /// The backend cannot see the processes of its tree: an SSH child's run
    /// on another machine, behind a login shell.
    Unobserved,
}

/// How a child's whole process tree is ended.
pub enum KillHandle {
    /// A local job's tree: its supervisor or its watched process table.
    Tree(JobTree),
    /// A job on a server: its process group there and the local `ssh`
    /// process that stands for it.
    Remote(RemoteJob),
    /// A container exec's kill switch.
    Switch(Sender<()>),
}

/// The exit of one child, set once by the task that waits for it.
pub struct ExitWatch {
    result: Mutex<Option<(ChildExit, Descendants)>>,
    done: Receiver<()>,
}

/// Records a child's exit and wakes every waiter.
pub struct ExitSetter {
    watch: Arc<ExitWatch>,
    // Dropped when the exit is recorded: that closes the channel the waiters
    // park on.
    _done: Sender<()>,
}

pub fn exit_watch() -> (ExitSetter, Arc<ExitWatch>) {
    let (done_tx, done_rx) = async_channel::bounded(1);
    let watch = Arc::new(ExitWatch {
        result: Mutex::new(None),
        done: done_rx,
    });
    (
        ExitSetter {
            watch: Arc::clone(&watch),
            _done: done_tx,
        },
        watch,
    )
}

impl ExitSetter {
    pub(crate) fn finish(self, exit: ChildExit, descendants: Descendants) {
        *self
            .watch
            .result
            .lock()
            .expect("the exit result lock is never held across a panic") =
            Some((exit, descendants));
    }
}

/// A process spawned through [`Sandbox::spawn`](crate::Sandbox::spawn).
///
/// The streams are channels of byte chunks, which every backend can supply:
/// a closed `stdout` or `stderr` is end of stream, and dropping the `stdin`
/// sender closes the process's standard input. Dropping the handle does not
/// end the process; [`kill`](Self::kill) does.
pub struct SandboxedChild {
    id: u32,
    stdin: Option<Sender<Vec<u8>>>,
    stdout: Option<Receiver<Vec<u8>>>,
    stderr: Option<Receiver<Vec<u8>>>,
    stdin_blocked: Option<Receiver<StdinWatchSignal>>,
    kill: KillHandle,
    exit: Arc<ExitWatch>,
}

impl std::fmt::Debug for SandboxedChild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxedChild")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl SandboxedChild {
    pub(crate) fn new(id: u32, streams: Streams, kill: KillHandle, exit: Arc<ExitWatch>) -> Self {
        let Streams {
            stdin,
            stdout,
            stderr,
            stdin_blocked,
        } = streams;
        Self {
            id,
            stdin,
            stdout,
            stderr,
            stdin_blocked,
            kill,
            exit,
        }
    }

    /// The process id: the local process for the host and SSH backends, a
    /// number identifying the exec for a container, which has no host
    /// process.
    #[must_use]
    pub const fn id(&self) -> u32 {
        self.id
    }

    /// The sender feeding the process's standard input, once; `None` when
    /// the spawn did not pipe it or it was taken.
    pub const fn take_stdin(&mut self) -> Option<Sender<Vec<u8>>> {
        self.stdin.take()
    }

    /// The process's standard output as chunks, once; `None` when the spawn
    /// did not pipe it or it was taken.
    pub const fn take_stdout(&mut self) -> Option<Receiver<Vec<u8>>> {
        self.stdout.take()
    }

    /// The process's standard error as chunks, once; `None` when the spawn
    /// did not pipe it or it was taken.
    pub const fn take_stderr(&mut self) -> Option<Receiver<Vec<u8>>> {
        self.stderr.take()
    }

    /// The signals that a process of the child's tree is waiting to read its
    /// piped standard input with nothing buffered, once; `None` when the
    /// spawn did not pipe stdin, it was taken, or the backend cannot watch
    /// it.
    ///
    /// The host and container backends watch the stdin pipe: one
    /// [`StdinWatchSignal::Blocked`] per blocking episode, or a single
    /// [`StdinWatchSignal::Failed`] when the watch cannot run at all. An SSH
    /// child has no watch: its stdin is a pipe to a local `ssh` client,
    /// which cannot tell whether a process on the server reads it.
    pub const fn take_stdin_blocked(&mut self) -> Option<Receiver<StdinWatchSignal>> {
        self.stdin_blocked.take()
    }

    /// Kills the whole process tree: the process and every descendant,
    /// wherever it has gone. Killing a process whose tree has already ended
    /// does nothing.
    ///
    /// # Errors
    /// Returns an error when a process of the tree cannot be signalled or
    /// survives repeated kills.
    pub async fn kill(&self) -> Result<(), KillTreeError> {
        let killed = match &self.kill {
            KillHandle::Tree(tree) => tree.kill().await,
            KillHandle::Remote(job) => job.kill().await,
            // The exec has ended when nobody receives the switch any more.
            KillHandle::Switch(switch) => {
                let _ = switch.send(()).await;
                Ok(())
            }
        };
        match killed {
            #[cfg(unix)]
            Err(KillTreeError::Ended) => Ok(()),
            other => other,
        }
    }

    /// Waits until the process and every process it started have exited and
    /// answers how the process ended. Any number of calls may wait, before
    /// or after the process ended; each gets the same answer.
    ///
    /// # Panics
    /// Panics when another thread panicked holding the exit result's lock,
    /// which nothing does across a panic.
    pub async fn wait(&self) -> ChildExit {
        self.outcome().await.0
    }

    /// Waits like [`wait`](Self::wait) and answers the processes the child
    /// left running that the backend ended: those of a
    /// [`Foreground`](crate::TreeLifetime::Foreground) spawn, found and
    /// killed when its top-level process exited. A
    /// [`Background`](crate::TreeLifetime::Background) child's descendants
    /// are waited for instead, so nothing is killed.
    ///
    /// An SSH child answers [`Descendants::Unobserved`]; a failed wait
    /// answers it too.
    pub async fn descendants_killed(&self) -> Descendants {
        self.outcome().await.1
    }

    async fn outcome(&self) -> (ChildExit, Descendants) {
        // Nothing is ever sent: the channel closes when the exit is recorded.
        let _ = self.exit.done.recv().await;
        self.exit
            .result
            .lock()
            .expect("the exit result lock is never held across a panic")
            .clone()
            .unwrap_or_else(|| {
                (
                    ChildExit::Failed("the task waiting for the process ended".into()),
                    Descendants::Unobserved,
                )
            })
    }
}

/// The streams a backend pipes for a child.
#[derive(Default)]
pub struct Streams {
    pub(crate) stdin: Option<Sender<Vec<u8>>>,
    pub(crate) stdout: Option<Receiver<Vec<u8>>>,
    pub(crate) stderr: Option<Receiver<Vec<u8>>>,
    pub(crate) stdin_blocked: Option<Receiver<StdinWatchSignal>>,
}
