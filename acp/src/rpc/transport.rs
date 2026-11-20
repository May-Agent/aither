//! The [`Transport`] trait and the bundled transport implementations.
//!
//! A transport is a framed, full-duplex channel owned by a single task:
//! [`recv`](Transport::recv) yields the peer's messages one at a time and
//! [`send`](Transport::send) writes ours. Bundled implementations cover
//! byte streams ([`Lines`](crate::rpc::Lines)), process stdio ([`Stdio`],
//! [`ChildProcess`]), and in-process links ([`Memory`]).

use std::future::Future;
use std::process::ExitStatus;

use super::error::RpcError;
use super::message::{Inbound, Outbound};

/// A framed, full-duplex JSON-RPC channel.
///
/// Implementations take `&mut self` so the transport is owned by a single
/// task — the connection engine — which is its only reader and writer.
pub trait Transport: Send {
    /// Receive the next inbound message.
    ///
    /// Returns `None` at EOF. A malformed line is reported as
    /// [`Inbound::Invalid`], never as an error: the engine answers it and
    /// the stream stays usable.
    ///
    /// Dropping the returned future before it completes loses nothing: a
    /// partially read message is resumed by the next call, so `recv` is
    /// safe to race with other futures in a `select`-style loop.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying channel fails (I/O, peer
    /// crash) — the connection cannot continue.
    fn recv(&mut self) -> impl Future<Output = Result<Option<Inbound>, RpcError>> + Send;

    /// Serialize and write one outbound message.
    ///
    /// # Errors
    ///
    /// Returns an error when the message cannot be serialized or written.
    fn send(&mut self, msg: &Outbound) -> impl Future<Output = Result<(), RpcError>> + Send;

    /// Flush and close the write half.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying channel cannot be closed
    /// cleanly. Callers may ignore it — the transport is dropped anyway.
    fn close(&mut self) -> impl Future<Output = Result<(), RpcError>> + Send;

    /// Exit status of a process-backed peer once the connection has ended.
    ///
    /// Waits for the peer to terminate, killing it when it is still
    /// running after the connection broke. Transports not backed by a
    /// process report `None`.
    fn exit_status(&mut self) -> impl Future<Output = Option<ExitStatus>> + Send {
        std::future::ready(None)
    }
}

/// An in-memory transport for tests and in-process links.
///
/// [`pair`](Memory::pair) connects two ends. `send` serializes the message
/// to its JSON line and `recv` runs it through the same
/// [`Inbound`](crate::rpc::Inbound) classification a byte transport uses,
/// so the real serde codec is exercised end to end; only byte-level
/// framing is skipped.
#[derive(Debug)]
pub struct Memory {
    /// Serialized lines towards the peer.
    tx: async_channel::Sender<String>,
    /// Serialized lines from the peer.
    rx: async_channel::Receiver<String>,
}

impl Memory {
    /// Two connected ends of an in-memory transport.
    #[must_use]
    pub fn pair() -> (Self, Self) {
        let (a_tx, a_rx) = async_channel::unbounded();
        let (b_tx, b_rx) = async_channel::unbounded();
        (Self { tx: a_tx, rx: b_rx }, Self { tx: b_tx, rx: a_rx })
    }
}

impl Transport for Memory {
    async fn recv(&mut self) -> Result<Option<Inbound>, RpcError> {
        Ok(self
            .rx
            .recv()
            .await
            .ok()
            .map(|line| Inbound::parse(line.as_bytes())))
    }

    async fn send(&mut self, msg: &Outbound) -> Result<(), RpcError> {
        let line = serde_json::to_string(msg)?;
        self.tx
            .send(line)
            .await
            .map_err(|_| RpcError::internal_error("peer is gone"))
    }

    fn close(&mut self) -> impl Future<Output = Result<(), RpcError>> + Send {
        self.tx.close();
        std::future::ready(Ok(()))
    }
}

/// A transport over this process's stdin and stdout.
///
/// Messages are newline-delimited JSON. Both halves run on the `blocking`
/// pool through `Unblock` rather than a readiness reactor: on Windows the
/// reactor reaches its readiness model through AFD, which works only on
/// sockets, so a console handle or a pipe cannot be registered at all.
/// Completion-based I/O does not help either — this is why tokio also
/// backs its own stdin with a blocking pool.
#[cfg(feature = "stdio")]
pub struct Stdio {
    /// The framed stdin/stdout pair.
    inner:
        super::lines::Lines<blocking::Unblock<std::io::Stdin>, blocking::Unblock<std::io::Stdout>>,
}

#[cfg(feature = "stdio")]
impl std::fmt::Debug for Stdio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stdio").finish_non_exhaustive()
    }
}

#[cfg(feature = "stdio")]
impl Stdio {
    /// A transport on this process's stdin/stdout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: super::lines::Lines::new(
                blocking::Unblock::new(std::io::stdin()),
                blocking::Unblock::new(std::io::stdout()),
            ),
        }
    }
}

#[cfg(feature = "stdio")]
impl Default for Stdio {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "stdio")]
impl Transport for Stdio {
    async fn recv(&mut self) -> Result<Option<Inbound>, RpcError> {
        self.inner.recv().await
    }

    async fn send(&mut self, msg: &Outbound) -> Result<(), RpcError> {
        self.inner.send(msg).await
    }

    async fn close(&mut self) -> Result<(), RpcError> {
        self.inner.close().await
    }
}

/// A transport that spawns a child process and speaks JSON-RPC over its
/// stdio pipes.
///
/// Stdin and stdout are piped for the JSON-RPC stream; stderr is inherited
/// so the child's diagnostics reach ours.
#[cfg(feature = "process")]
pub struct ChildProcess {
    /// The child, kept for reaping and kill-on-drop.
    child: async_process::Child,
    /// The framed stdout/stdin pipe pair.
    inner: super::lines::Lines<async_process::ChildStdout, async_process::ChildStdin>,
    /// Exit status once reaped, cached for repeated `exit_status` calls.
    status: Option<ExitStatus>,
}

#[cfg(feature = "process")]
impl std::fmt::Debug for ChildProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChildProcess")
            .field("id", &self.child.id())
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "process")]
impl ChildProcess {
    /// Spawn `bin` with `args`, `env`, and working directory `cwd`.
    ///
    /// # Errors
    ///
    /// Returns an error if the process cannot be spawned or its stdio
    /// pipes cannot be captured.
    pub fn spawn(
        bin: &str,
        args: &[String],
        env: &[(String, String)],
        cwd: &std::path::Path,
    ) -> Result<Self, RpcError> {
        let mut command = async_process::Command::new(bin);
        command
            .args(args)
            .envs(env.iter().cloned())
            .current_dir(cwd);
        Self::from_command(&mut command)
    }

    /// Create a transport from a prepared [`Command`](async_process::Command).
    ///
    /// Stdin and stdout are piped for the JSON-RPC stream and stderr is
    /// inherited. Use this instead of [`spawn`](Self::spawn) when the child
    /// needs a custom environment or working directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the process cannot be spawned or its stdio
    /// pipes cannot be captured.
    pub fn from_command(command: &mut async_process::Command) -> Result<Self, RpcError> {
        let child = command
            .stdin(async_process::Stdio::piped())
            .stdout(async_process::Stdio::piped())
            .stderr(async_process::Stdio::inherit())
            .spawn()?;
        Self::from_child(child)
    }

    /// Create a transport from a spawned `child` with piped stdio.
    ///
    /// # Errors
    ///
    /// Returns an error if the child's stdin or stdout was not captured.
    pub fn from_child(mut child: async_process::Child) -> Result<Self, RpcError> {
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| RpcError::internal_error("child process stdin not captured"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| RpcError::internal_error("child process stdout not captured"))?;
        Ok(Self {
            child,
            inner: super::lines::Lines::new(stdout, stdin),
            status: None,
        })
    }
}

#[cfg(feature = "process")]
impl Transport for ChildProcess {
    async fn recv(&mut self) -> Result<Option<Inbound>, RpcError> {
        self.inner.recv().await
    }

    async fn send(&mut self, msg: &Outbound) -> Result<(), RpcError> {
        self.inner.send(msg).await
    }

    async fn close(&mut self) -> Result<(), RpcError> {
        let _ = self.inner.close().await;
        if self.status.is_none() {
            let _ = self.child.kill();
            self.status = self.child.status().await.ok();
        }
        Ok(())
    }

    async fn exit_status(&mut self) -> Option<ExitStatus> {
        if let Some(status) = self.status {
            return Some(status);
        }
        let status = if let Ok(Some(status)) = self.child.try_status() {
            Some(status)
        } else {
            // The pipe is gone but the child is still running; it can no
            // longer be reached, so collect a real status by reaping it.
            let _ = self.child.kill();
            self.child.status().await.ok()
        };
        self.status = status;
        status
    }
}

#[cfg(feature = "process")]
impl Drop for ChildProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::rpc::Notification;

    #[tokio::test]
    async fn memory_pair_roundtrips_through_the_real_codec() {
        let (mut a, mut b) = Memory::pair();

        a.send(&Outbound::Notification(Notification::with_params(
            "ping",
            json!({ "n": 1 }),
        )))
        .await
        .unwrap();
        let Inbound::Notification(notification) = b.recv().await.unwrap().unwrap() else {
            panic!("expected a notification");
        };
        assert_eq!(notification.method, "ping");
        assert_eq!(notification.params, Some(json!({ "n": 1 })));

        b.send(&Outbound::Notification(Notification::new("pong")))
            .await
            .unwrap();
        assert!(matches!(
            a.recv().await.unwrap(),
            Some(Inbound::Notification(_))
        ));
    }

    #[tokio::test]
    async fn memory_reports_eof_when_the_peer_closes() {
        let (a, mut b) = Memory::pair();
        drop(a);
        assert_eq!(b.recv().await.unwrap(), None);
    }

    #[tokio::test]
    async fn memory_close_ends_the_stream() {
        let (mut a, mut b) = Memory::pair();
        a.close().await.unwrap();
        assert!(
            a.send(&Outbound::Notification(Notification::new("x")))
                .await
                .is_err()
        );
        assert_eq!(b.recv().await.unwrap(), None);
    }

    #[tokio::test]
    async fn memory_preserves_order() {
        let (mut a, mut b) = Memory::pair();
        for i in 0..3_i64 {
            a.send(&Outbound::Notification(Notification::with_params(
                "seq",
                json!({ "i": i }),
            )))
            .await
            .unwrap();
        }
        for i in 0..3_i64 {
            let Inbound::Notification(notification) = b.recv().await.unwrap().unwrap() else {
                panic!("expected a notification");
            };
            assert_eq!(notification.params, Some(json!({ "i": i })));
        }
    }
}
