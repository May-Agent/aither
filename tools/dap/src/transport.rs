//! Runtime-agnostic DAP transports over stdio, TCP, and generic async streams.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::process::ExitStatus;

use async_net::{AsyncToSocketAddrs, TcpStream};
use async_process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use futures_lite::io::{AsyncRead, AsyncWrite, BufReader};

use crate::error::DapError;
use crate::framing::{FrameReader, write_message};
use crate::protocol::ProtocolMessage;

/// Full-duplex transport carrying Content-Length framed DAP messages.
///
/// Implementations are owned exclusively by [`crate::DapClient`]'s connection
/// loop so no internal locking is required.
pub trait DapTransport: Send {
    /// Sends a single framed protocol message to the debug adapter.
    fn send(
        &mut self,
        message: &ProtocolMessage,
    ) -> impl Future<Output = Result<(), DapError>> + Send;

    /// Receives the next framed protocol message, or `Ok(None)` on clean EOF.
    ///
    /// Implementations must preserve buffered partial reads if the returned
    /// future is dropped before completion.
    fn recv(&mut self) -> impl Future<Output = Result<Option<ProtocolMessage>, DapError>> + Send;

    /// Closes the transport and releases any underlying process or socket.
    fn close(&mut self) -> impl Future<Output = Result<(), DapError>> + Send;

    /// Reports the underlying child process exit status when backed by a subprocess.
    fn exit_status(&mut self) -> impl Future<Output = Option<ExitStatus>> + Send {
        std::future::ready(None)
    }
}

/// Private object-safe twin of [`DapTransport`] used by [`AnyDapTransport`].
trait DapTransportImpl: Send {
    fn send<'a>(
        &'a mut self,
        message: &'a ProtocolMessage,
    ) -> Pin<Box<dyn Future<Output = Result<(), DapError>> + Send + 'a>>;

    fn recv(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<Option<ProtocolMessage>, DapError>> + Send + '_>>;

    fn close(&mut self) -> Pin<Box<dyn Future<Output = Result<(), DapError>> + Send + '_>>;

    fn exit_status(&mut self) -> Pin<Box<dyn Future<Output = Option<ExitStatus>> + Send + '_>>;
}

impl<T: DapTransport> DapTransportImpl for T {
    fn send<'a>(
        &'a mut self,
        message: &'a ProtocolMessage,
    ) -> Pin<Box<dyn Future<Output = Result<(), DapError>> + Send + 'a>> {
        Box::pin(DapTransport::send(self, message))
    }

    fn recv(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<Option<ProtocolMessage>, DapError>> + Send + '_>> {
        Box::pin(DapTransport::recv(self))
    }

    fn close(&mut self) -> Pin<Box<dyn Future<Output = Result<(), DapError>> + Send + '_>> {
        Box::pin(DapTransport::close(self))
    }

    fn exit_status(&mut self) -> Pin<Box<dyn Future<Output = Option<ExitStatus>> + Send + '_>> {
        Box::pin(DapTransport::exit_status(self))
    }
}

/// Type-erased [`DapTransport`] for callers storing heterogeneous transports
/// (such as stdio child adapters alongside TCP adapter connections).
pub struct AnyDapTransport(Box<dyn DapTransportImpl>);

impl AnyDapTransport {
    /// Wraps a concrete [`DapTransport`] in a type-erased handle.
    #[must_use]
    pub fn new(transport: impl DapTransport + 'static) -> Self {
        Self(Box::new(transport))
    }
}

impl fmt::Debug for AnyDapTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyDapTransport(..)")
    }
}

impl DapTransport for AnyDapTransport {
    async fn send(&mut self, message: &ProtocolMessage) -> Result<(), DapError> {
        self.0.send(message).await
    }

    async fn recv(&mut self) -> Result<Option<ProtocolMessage>, DapError> {
        self.0.recv().await
    }

    async fn close(&mut self) -> Result<(), DapError> {
        self.0.close().await
    }

    async fn exit_status(&mut self) -> Option<ExitStatus> {
        self.0.exit_status().await
    }
}

/// DAP transport over any pair of `futures_lite::io` async reader and writer streams.
pub struct StreamTransport<R, W> {
    reader: BufReader<R>,
    writer: W,
    framer: FrameReader,
    closed: bool,
}

impl<R, W> fmt::Debug for StreamTransport<R, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamTransport")
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl<R, W> StreamTransport<R, W>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    /// Constructs a framed DAP transport from an async reader and writer.
    #[must_use]
    pub fn new(reader: R, writer: W) -> Self {
        Self {
            reader: BufReader::new(reader),
            writer,
            framer: FrameReader::new(),
            closed: false,
        }
    }
}

impl<R, W> DapTransport for StreamTransport<R, W>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    async fn send(&mut self, message: &ProtocolMessage) -> Result<(), DapError> {
        if self.closed {
            return Err(DapError::ConnectionClosed);
        }
        write_message(&mut self.writer, message).await
    }

    async fn recv(&mut self) -> Result<Option<ProtocolMessage>, DapError> {
        if self.closed {
            return Ok(None);
        }
        self.framer.read_message(&mut self.reader).await
    }

    fn close(&mut self) -> impl Future<Output = Result<(), DapError>> + Send {
        self.closed = true;
        std::future::ready(Ok(()))
    }
}

/// DAP transport communicating with a debug adapter child process over stdio pipes.
pub struct StdioTransport {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    framer: FrameReader,
    closed: bool,
}

impl fmt::Debug for StdioTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StdioTransport")
            .field("pid", &self.child.id())
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl StdioTransport {
    /// Spawns a debug adapter binary and connects to its stdin/stdout.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::Io`] if the process fails to spawn or its stdio pipes
    /// are unavailable.
    pub fn spawn(program: &str, args: &[&str]) -> Result<Self, DapError> {
        let mut command = Command::new(program);
        command.args(args);
        Self::from_command(&mut command)
    }

    /// Spawns a debug adapter from a prepared [`async_process::Command`].
    ///
    /// Stdin and stdout are piped for DAP frames; stderr is nullified so adapter
    /// diagnostics do not corrupt host terminal output unless the caller overrides
    /// stderr beforehand.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::Io`] if spawning fails or stdio pipes cannot be captured.
    pub fn from_command(command: &mut Command) -> Result<Self, DapError> {
        let child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        Self::from_child(child)
    }

    /// Wraps an already-spawned [`Child`] whose stdin and stdout are piped.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::Protocol`] if either stdin or stdout was not piped.
    pub fn from_child(mut child: Child) -> Result<Self, DapError> {
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| DapError::Protocol("adapter child stdin not piped".to_owned()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| DapError::Protocol("adapter child stdout not piped".to_owned()))?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            framer: FrameReader::new(),
            closed: false,
        })
    }

    /// Returns the OS process identifier of the spawned debug adapter.
    #[must_use]
    pub fn pid(&self) -> u32 {
        self.child.id()
    }
}

impl DapTransport for StdioTransport {
    async fn send(&mut self, message: &ProtocolMessage) -> Result<(), DapError> {
        if self.closed {
            return Err(DapError::ConnectionClosed);
        }
        write_message(&mut self.stdin, message).await
    }

    async fn recv(&mut self) -> Result<Option<ProtocolMessage>, DapError> {
        if self.closed {
            return Ok(None);
        }
        self.framer.read_message(&mut self.stdout).await
    }

    async fn close(&mut self) -> Result<(), DapError> {
        self.closed = true;
        let _ = self.child.kill();
        let _ = self.child.status().await;
        Ok(())
    }

    async fn exit_status(&mut self) -> Option<ExitStatus> {
        if let Ok(Some(status)) = self.child.try_status() {
            Some(status)
        } else {
            let _ = self.child.kill();
            self.child.status().await.ok()
        }
    }
}

/// DAP transport communicating with a debug adapter over a TCP socket.
pub struct TcpTransport {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    framer: FrameReader,
    closed: bool,
}

impl fmt::Debug for TcpTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TcpTransport")
            .field("peer_addr", &self.writer.peer_addr().ok())
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl TcpTransport {
    /// Connects to a TCP DAP server at `addr`.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::Io`] if connecting fails.
    pub async fn connect<A: AsyncToSocketAddrs>(addr: A) -> Result<Self, DapError> {
        let stream = TcpStream::connect(addr).await?;
        Ok(Self::from_stream(stream))
    }

    /// Wraps an already-connected [`TcpStream`].
    #[must_use]
    pub fn from_stream(stream: TcpStream) -> Self {
        let writer = stream.clone();
        Self {
            reader: BufReader::new(stream),
            writer,
            framer: FrameReader::new(),
            closed: false,
        }
    }
}

impl DapTransport for TcpTransport {
    async fn send(&mut self, message: &ProtocolMessage) -> Result<(), DapError> {
        if self.closed {
            return Err(DapError::ConnectionClosed);
        }
        write_message(&mut self.writer, message).await
    }

    async fn recv(&mut self) -> Result<Option<ProtocolMessage>, DapError> {
        if self.closed {
            return Ok(None);
        }
        self.framer.read_message(&mut self.reader).await
    }

    fn close(&mut self) -> impl Future<Output = Result<(), DapError>> + Send {
        self.closed = true;
        let _ = self.writer.shutdown(std::net::Shutdown::Both);
        std::future::ready(Ok(()))
    }
}
