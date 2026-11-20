//! Newline-delimited JSON-RPC message reading shared by the transports.

use std::pin::Pin;
use std::task::Poll;

use futures_lite::future;
use futures_lite::io::AsyncBufRead;
use tracing::debug;

use super::traits::Result;
use crate::protocol::{JsonRpcMessage, McpError};

/// Maximum byte length of a single newline-delimited JSON-RPC frame (16 MiB).
///
/// Accommodates multi-megabyte base64 image or resource payloads while bounding
/// memory against an untrusted or malfunctioning peer that streams without a
/// newline.
pub(super) const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// Read one newline-delimited JSON-RPC message from `reader`, bounded by
/// [`MAX_MESSAGE_BYTES`].
///
/// Bytes are staged in `read_buf` between calls so a future dropped mid-line
/// loses nothing: the next call resumes the partial message, which makes the
/// transports' `recv` safe to race in a `select`-style loop. Returns `None`
/// on EOF, skips blank lines, and surfaces deserialization errors without
/// consuming the following message.
///
/// `poll_fill_buf` is polled directly once per chunk rather than through
/// `AsyncBufReadExt::fill_buf`, which polls twice and panics when the second
/// poll returns `Pending` after EOF on `BufReader<Unblock<Stdin>>`.
pub(super) async fn read_message(
    reader: &mut (impl AsyncBufRead + Unpin),
    read_buf: &mut Vec<u8>,
) -> Result<Option<JsonRpcMessage>> {
    read_message_bounded(reader, read_buf, MAX_MESSAGE_BYTES).await
}

pub(super) async fn read_message_bounded(
    reader: &mut (impl AsyncBufRead + Unpin),
    read_buf: &mut Vec<u8>,
    max_bytes: usize,
) -> Result<Option<JsonRpcMessage>> {
    loop {
        let reached_eof = future::poll_fn(|cx| {
            loop {
                let available = futures_lite::ready!(Pin::new(&mut *reader).poll_fill_buf(cx))?;
                if available.is_empty() {
                    return Poll::Ready(Ok::<bool, McpError>(true));
                }

                if let Some(newline_pos) = available.iter().position(|&b| b == b'\n') {
                    if read_buf.len().saturating_add(newline_pos) > max_bytes {
                        Pin::new(&mut *reader).consume(newline_pos + 1);
                        read_buf.clear();
                        read_buf.shrink_to(0);
                        return Poll::Ready(Err(oversized_error(max_bytes)));
                    }
                    if let Some(prefix) = available.get(..=newline_pos) {
                        read_buf.extend_from_slice(prefix);
                    }
                    Pin::new(&mut *reader).consume(newline_pos + 1);
                    return Poll::Ready(Ok(false));
                }

                let chunk_len = available.len();
                if read_buf.len().saturating_add(chunk_len) > max_bytes {
                    Pin::new(&mut *reader).consume(chunk_len);
                    read_buf.clear();
                    read_buf.shrink_to(0);
                    return Poll::Ready(Err(oversized_error(max_bytes)));
                }
                read_buf.extend_from_slice(available);
                Pin::new(&mut *reader).consume(chunk_len);
            }
        })
        .await?;

        if reached_eof && read_buf.iter().all(u8::is_ascii_whitespace) {
            read_buf.clear();
            return Ok(None);
        }

        let mut line = std::mem::take(read_buf);
        while matches!(line.last(), Some(b) if b.is_ascii_whitespace()) {
            line.pop();
        }
        if line.is_empty() {
            continue;
        }
        debug!("MCP RX: {}", String::from_utf8_lossy(&line));
        return serde_json::from_slice(&line).map(Some).map_err(Into::into);
    }
}

fn oversized_error(max_bytes: usize) -> McpError {
    McpError::Transport(format!(
        "JSON-RPC message exceeds maximum frame size of {max_bytes} bytes"
    ))
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use futures_lite::io::{AsyncRead, BufReader, Cursor};

    use super::*;

    /// Yields one byte at a time with a `Pending` pause after `pause_at` so a
    /// mid-line future drop can be exercised deterministically.
    struct PauseReader<'a> {
        data: &'a [u8],
        pos: usize,
        pause_at: Option<usize>,
    }

    impl AsyncRead for PauseReader<'_> {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            if self.pause_at == Some(self.pos) {
                self.pause_at = None;
                return Poll::Pending;
            }
            if self.pos >= self.data.len() || buf.is_empty() {
                return Poll::Ready(Ok(0));
            }
            buf[0] = self.data[self.pos];
            self.pos += 1;
            Poll::Ready(Ok(1))
        }
    }

    #[tokio::test]
    async fn rejects_oversized_frame_without_newline_and_clears_buffer() {
        let oversized = vec![b'x'; 256];
        let mut reader = BufReader::with_capacity(32, Cursor::new(oversized));
        let mut read_buf = Vec::new();

        let err = read_message_bounded(&mut reader, &mut read_buf, 64)
            .await
            .unwrap_err();
        assert!(matches!(err, McpError::Transport(_)), "{err:?}");
        assert_eq!(read_buf, [] as [u8; 0]);
        assert_eq!(read_buf.capacity(), 0);
    }

    #[tokio::test]
    async fn rejects_oversized_newline_terminated_frame_and_recovers_for_next_frame() {
        let mut payload = vec![b'a'; 80];
        payload.push(b'\n');
        payload.extend_from_slice(b"{\"jsonrpc\":\"2.0\",\"method\":\"ping\"}\n");

        let mut reader = BufReader::with_capacity(128, Cursor::new(payload));
        let mut read_buf = Vec::new();

        let err = read_message_bounded(&mut reader, &mut read_buf, 64)
            .await
            .unwrap_err();
        assert!(matches!(err, McpError::Transport(_)), "{err:?}");

        let next = read_message_bounded(&mut reader, &mut read_buf, 64)
            .await
            .unwrap()
            .expect("following valid message");
        assert!(next.is_notification());
    }

    #[tokio::test]
    async fn dropping_mid_line_preserves_staged_bytes_for_next_call() {
        let raw = b"{\"jsonrpc\":\"2.0\",\"method\":\"ping\"}\n";
        let mut reader = BufReader::with_capacity(
            4,
            PauseReader {
                data: raw,
                pos: 0,
                pause_at: Some(10),
            },
        );
        let mut read_buf = Vec::new();

        {
            let fut = read_message_bounded(&mut reader, &mut read_buf, 128);
            futures_lite::pin!(fut);
            assert!(future::poll_once(&mut fut).await.is_none());
        }
        assert_ne!(read_buf, [] as [u8; 0]);

        let msg = read_message_bounded(&mut reader, &mut read_buf, 128)
            .await
            .unwrap()
            .expect("resumed message");
        assert_eq!(msg.as_notification().unwrap().method, "ping");
    }
}
