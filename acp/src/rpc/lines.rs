//! LF-delimited NDJSON framing shared by all byte transports.
//!
//! One JSON message per line, no embedded newlines. Reading is
//! cancellation-safe: bytes of a partial line are staged in `read_buf`
//! between `recv` calls, so a `recv` future dropped mid-line resumes the
//! same message on the next call.

use std::pin::Pin;
use std::task::{Poll, ready};

use futures_lite::AsyncBufRead;
use futures_lite::future;
use futures_lite::io::{AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tracing::debug;

use super::error::RpcError;
use super::message::{Inbound, Outbound};
use super::transport::Transport;

/// Maximum byte length of a single newline-delimited JSON-RPC frame (16 MiB).
///
/// Accommodates multi-megabyte base64 image or resource payloads while
/// bounding memory against a peer that streams without a newline.
const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// The error for a frame exceeding `max_bytes`.
fn oversized_error(max_bytes: usize) -> RpcError {
    RpcError::internal_error(format!(
        "JSON-RPC message exceeds maximum frame size of {max_bytes} bytes"
    ))
}

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
    /// Whether an oversized frame is being discarded until its terminating
    /// newline. Persists between calls so a future dropped mid-drain resumes
    /// discarding instead of re-reading the frame's tail as a new message.
    discarding_oversized: bool,
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
            discarding_oversized: false,
            writer,
            closed: false,
        }
    }

    /// Read one complete line, resuming a partial line staged in
    /// `read_buf`, bounded by [`MAX_MESSAGE_BYTES`].
    ///
    /// Returns `None` on EOF, skips blank lines, and treats an
    /// unterminated final line as a complete message. A frame over the
    /// cap is consumed, the staging buffer released, and the error
    /// reported — the next read resumes on the following line.
    ///
    /// `poll_fill_buf` is polled directly once per chunk rather than
    /// through `read_until`, which loops internally until the delimiter
    /// and so cannot be bounded mid-line, or `fill_buf`, which polls
    /// `poll_fill_buf` twice and panics when the second poll returns
    /// `Pending` — exactly what a `BufReader<Unblock<Stdin>>` produces
    /// after EOF, because every fresh poll hands another read to the
    /// blocking pool.
    async fn read_line(&mut self) -> Result<Option<Vec<u8>>, RpcError> {
        self.read_line_bounded(MAX_MESSAGE_BYTES).await
    }

    /// `read_line` with the frame cap set to `max_bytes` (tests use a
    /// small bound).
    async fn read_line_bounded(&mut self, max_bytes: usize) -> Result<Option<Vec<u8>>, RpcError> {
        loop {
            let reached_eof = future::poll_fn(|cx| {
                loop {
                    let available = ready!(Pin::new(&mut self.reader).poll_fill_buf(cx))
                        .map_err(RpcError::from)?;
                    if available.is_empty() {
                        if self.discarding_oversized {
                            // EOF reached while draining a rejected frame —
                            // the whole frame is consumed, so report the error.
                            self.discarding_oversized = false;
                            return Poll::Ready(Err(oversized_error(max_bytes)));
                        }
                        return Poll::Ready(Ok::<bool, RpcError>(true));
                    }
                    if self.discarding_oversized {
                        // Draining a rejected frame: discard chunks without
                        // buffering until its terminating newline arrives.
                        if let Some(newline_pos) = available.iter().position(|&b| b == b'\n') {
                            Pin::new(&mut self.reader).consume(newline_pos + 1);
                            self.discarding_oversized = false;
                            return Poll::Ready(Err(oversized_error(max_bytes)));
                        }
                        let chunk_len = available.len();
                        Pin::new(&mut self.reader).consume(chunk_len);
                        continue;
                    }
                    if let Some(newline_pos) = available.iter().position(|&b| b == b'\n') {
                        if self.read_buf.len().saturating_add(newline_pos) > max_bytes {
                            Pin::new(&mut self.reader).consume(newline_pos + 1);
                            self.read_buf.clear();
                            self.read_buf.shrink_to(0);
                            return Poll::Ready(Err(oversized_error(max_bytes)));
                        }
                        self.read_buf.extend_from_slice(&available[..=newline_pos]);
                        Pin::new(&mut self.reader).consume(newline_pos + 1);
                        return Poll::Ready(Ok(false));
                    }
                    let chunk_len = available.len();
                    if self.read_buf.len().saturating_add(chunk_len) > max_bytes {
                        // The line is oversized before any newline: free the
                        // staged bytes immediately and drain the rest of the
                        // frame unbuffered so the next read resumes on the
                        // following line.
                        Pin::new(&mut self.reader).consume(chunk_len);
                        self.read_buf.clear();
                        self.read_buf.shrink_to(0);
                        self.discarding_oversized = true;
                        continue;
                    }
                    self.read_buf.extend_from_slice(available);
                    Pin::new(&mut self.reader).consume(chunk_len);
                }
            })
            .await?;

            if reached_eof && self.read_buf.iter().all(u8::is_ascii_whitespace) {
                self.read_buf.clear();
                return Ok(None);
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

    /// A frame over the cap is rejected without buffering it whole, and
    /// the next well-formed line still reads.
    #[tokio::test]
    async fn oversized_frame_is_rejected_and_the_next_line_reads() {
        let mut input = vec![b'x'; 256];
        input.push(b'\n');
        input.extend_from_slice(b"{\"method\":\"ok\"}\n");
        let mut transport = lines(&[]);
        transport.reader = BufReader::new(Cursor::new(input.leak() as &'static [u8]));

        let error = transport.read_line_bounded(64).await.unwrap_err();
        assert!(error.message.contains("maximum frame size"), "{error}");
        assert_eq!(transport.read_buf, [] as [u8; 0]);
        assert_eq!(transport.read_buf.capacity(), 0);

        let line = transport.read_line_bounded(64).await.unwrap().unwrap();
        assert_eq!(line, b"{\"method\":\"ok\"}");
    }

    /// A frame over the cap spanning several fill chunks fails once and the
    /// read resumes on the following line, not on the rejected frame's tail.
    #[tokio::test]
    async fn oversized_frame_spanning_chunks_drains_to_the_next_line() {
        let mut input = vec![b'x'; 60];
        input.push(b'\n');
        input.extend_from_slice(b"{\"method\":\"ok\"}\n");
        let mut transport = lines(&[]);
        // An 8-byte fill buffer splits the 60-byte frame across chunks.
        transport.reader = BufReader::with_capacity(8, Cursor::new(input.leak() as &'static [u8]));

        let error = transport.read_line_bounded(24).await.unwrap_err();
        assert!(error.message.contains("maximum frame size"), "{error}");
        assert!(!transport.discarding_oversized);
        assert_eq!(transport.read_buf, [] as [u8; 0]);

        let line = transport.read_line_bounded(24).await.unwrap().unwrap();
        assert_eq!(line, b"{\"method\":\"ok\"}");
    }

    /// An unterminated stream over the cap fails instead of growing
    /// `read_buf` until EOF.
    #[tokio::test]
    async fn unterminated_stream_over_the_cap_fails() {
        let mut transport = lines(&[]);
        transport.reader = BufReader::new(Cursor::new(vec![b'y'; 512].leak() as &'static [u8]));
        let error = transport.read_line_bounded(64).await.unwrap_err();
        assert!(error.message.contains("maximum frame size"), "{error}");
    }
}
