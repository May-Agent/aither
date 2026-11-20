//! LF-delimited NDJSON framing shared by all byte transports.
//!
//! One JSON message per line, no embedded newlines. Reading is
//! cancellation-safe: bytes of a partial line are staged in `read_buf`
//! between `recv` calls, so a `recv` future dropped mid-line resumes the
//! same message on the next call.

use futures_lite::AsyncBufReadExt;
use futures_lite::io::{AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tracing::debug;

use super::error::RpcError;
use super::message::{Inbound, Outbound};
use super::transport::Transport;

/// A transport framing JSON-RPC messages as newline-delimited JSON over an
/// [`AsyncRead`]/[`AsyncWrite`] pair.
///
/// Malformed inbound lines are classified as [`Inbound::Invalid`] rather
/// than failing the transport — the engine answers them `-32600` and the
/// stream stays usable.
pub struct Lines<R, W> {
    /// Buffered read half.
    reader: BufReader<R>,
    /// Bytes consumed but not yet forming a complete line.
    read_buf: Vec<u8>,
    /// Write half.
    writer: W,
    /// Whether the transport has been closed.
    closed: bool,
}

impl<R, W> std::fmt::Debug for Lines<R, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lines")
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl<R, W> Lines<R, W>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    /// Frame `reader`/`writer` as an NDJSON transport.
    #[must_use]
    pub fn new(reader: R, writer: W) -> Self {
        Self {
            reader: BufReader::new(reader),
            read_buf: Vec::new(),
            writer,
            closed: false,
        }
    }

    /// Read one complete line, resuming a partial line staged in
    /// `read_buf`.
    ///
    /// Returns `None` on EOF, skips blank lines, and treats an
    /// unterminated final line as a complete message. Assembled with
    /// `read_until`, which polls the reader once per chunk — `fill_buf`
    /// polls `poll_fill_buf` twice and panics on a second `Pending`,
    /// which is exactly what a `BufReader<Unblock<Stdin>>` produces after
    /// EOF, because every fresh poll hands another read to the blocking
    /// pool.
    async fn read_line(&mut self) -> Result<Option<Vec<u8>>, RpcError> {
        loop {
            let read = self.reader.read_until(b'\n', &mut self.read_buf).await?;
            let complete = self.read_buf.last() == Some(&b'\n');
            if read == 0 && !complete {
                // EOF: whatever is staged is the final, unterminated line.
                if self.read_buf.iter().all(u8::is_ascii_whitespace) {
                    self.read_buf.clear();
                    return Ok(None);
                }
            } else if !complete {
                // The previous future was dropped mid-line; keep reading.
                continue;
            }
            let mut line = std::mem::take(&mut self.read_buf);
            while matches!(line.last(), Some(byte) if byte.is_ascii_whitespace()) {
                line.pop();
            }
            if line.is_empty() {
                continue;
            }
            return Ok(Some(line));
        }
    }
}

impl<R, W> Transport for Lines<R, W>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    async fn recv(&mut self) -> Result<Option<Inbound>, RpcError> {
        if self.closed {
            return Ok(None);
        }
        let Some(line) = self.read_line().await? else {
            return Ok(None);
        };
        debug!("RPC RX: {}", String::from_utf8_lossy(&line));
        Ok(Some(Inbound::parse(&line)))
    }

    async fn send(&mut self, msg: &Outbound) -> Result<(), RpcError> {
        if self.closed {
            return Err(RpcError::internal_error("transport is closed"));
        }
        let mut bytes = serde_json::to_vec(msg)?;
        debug!("RPC TX: {}", String::from_utf8_lossy(&bytes));
        bytes.push(b'\n');
        self.writer.write_all(&bytes).await?;
        self.writer.flush().await?;
        Ok(())
    }

    async fn close(&mut self) -> Result<(), RpcError> {
        self.closed = true;
        self.writer.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use futures_lite::io::Cursor;

    use super::*;
    use crate::rpc::{Notification, Request, RequestId};

    fn lines(input: &'static [u8]) -> Lines<Cursor<&'static [u8]>, Cursor<Vec<u8>>> {
        Lines::new(Cursor::new(input), Cursor::new(Vec::new()))
    }

    #[tokio::test]
    async fn recv_classifies_each_line() {
        let mut transport =
            lines(b"{\"id\":1,\"method\":\"m\"}\n{\"method\":\"n\"}\n{\"id\":2,\"result\":{}}\n");
        assert!(matches!(
            transport.recv().await.unwrap(),
            Some(Inbound::Request(Request { .. }))
        ));
        assert!(matches!(
            transport.recv().await.unwrap(),
            Some(Inbound::Notification(Notification { .. }))
        ));
        assert!(matches!(
            transport.recv().await.unwrap(),
            Some(Inbound::Response(_))
        ));
        assert_eq!(transport.recv().await.unwrap(), None);
    }

    #[tokio::test]
    async fn malformed_lines_become_invalid_not_errors() {
        let mut transport = lines(b"not json\n{\"method\":\"ok\"}\n");
        assert_eq!(transport.recv().await.unwrap(), Some(Inbound::Invalid));
        assert!(matches!(
            transport.recv().await.unwrap(),
            Some(Inbound::Notification(_))
        ));
    }

    #[tokio::test]
    async fn blank_lines_and_unterminated_tail() {
        let mut transport = lines(b"\n  \r\n{\"method\":\"tail\"}");
        assert!(matches!(
            transport.recv().await.unwrap(),
            Some(Inbound::Notification(_))
        ));
        assert_eq!(transport.recv().await.unwrap(), None);
    }

    #[tokio::test]
    async fn send_writes_one_line() {
        let mut transport = lines(b"");
        transport
            .send(&Outbound::Notification(Notification::new("ping")))
            .await
            .unwrap();
        let written = transport.writer.into_inner();
        assert_eq!(
            written,
            b"{\"jsonrpc\":\"2.0\",\"method\":\"ping\"}\n".to_vec()
        );
    }

    #[tokio::test]
    async fn a_partial_line_survives_a_dropped_recv() {
        use std::pin::pin;
        use std::task::{Context, Poll, Waker};

        /// A reader that dribbles one byte per poll and returns `Pending`
        /// exactly once, after `stall` bytes.
        struct Dribble {
            data: &'static [u8],
            stall: usize,
            stalled: bool,
        }
        impl AsyncRead for Dribble {
            fn poll_read(
                mut self: std::pin::Pin<&mut Self>,
                _: &mut Context<'_>,
                buf: &mut [u8],
            ) -> Poll<std::io::Result<usize>> {
                if self.data.is_empty() {
                    return Poll::Ready(Ok(0));
                }
                if self.stall > 0 {
                    self.stall -= 1;
                } else if !self.stalled {
                    self.stalled = true;
                    return Poll::Pending;
                }
                buf[0] = self.data[0];
                self.data = &self.data[1..];
                Poll::Ready(Ok(1))
            }
        }

        let mut cx = Context::from_waker(Waker::noop());

        let mut transport = Lines::new(
            Dribble {
                data: b"{\"id\":9,\"method\":\"m\"}\n{\"method\":\"n\"}\n",
                stall: 5,
                stalled: false,
            },
            Cursor::new(Vec::new()),
        );

        // Poll once mid-line and drop the future at the end of the block
        // (the `pin!` temporary ends its borrow of `transport`); the next
        // recv must complete the same message, not lose the staged bytes.
        {
            let mut recv = pin!(transport.recv());
            assert!(matches!(recv.as_mut().poll(&mut cx), Poll::Pending));
        }

        assert!(matches!(
            transport.recv().await.unwrap(),
            Some(Inbound::Request(Request {
                id: RequestId::Number(9),
                ..
            }))
        ));
    }
}
