//! Content-Length framed DAP JSON-RPC reader and writer.

use futures_lite::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::error::DapError;
use crate::protocol::ProtocolMessage;

/// Maximum accepted DAP payload size (16 MiB) to guard against malformed
/// `Content-Length` headers allocating unbounded memory.
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Maximum header block size (16 KiB) before `\r\n\r\n` must terminate headers.
const MAX_HEADER_BYTES: usize = 16 * 1024;

/// Cancellation-safe incremental parser for Content-Length framed DAP messages.
#[derive(Debug, Default)]
pub struct FrameReader {
    header_buf: Vec<u8>,
    body_len: Option<usize>,
    body_buf: Vec<u8>,
}

impl FrameReader {
    /// Creates an empty frame reader.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            header_buf: Vec::new(),
            body_len: None,
            body_buf: Vec::new(),
        }
    }

    /// Reads the next framed [`ProtocolMessage`] from `reader`, returning `Ok(None)`
    /// on a clean EOF between frames.
    ///
    /// Dropping the returned future before completion preserves already-read
    /// bytes in `self`, so callers can safely race this method in `select`-style loops.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::Protocol`] for malformed headers or truncated EOF mid-frame,
    /// [`DapError::FrameTooLarge`] if `Content-Length` exceeds [`MAX_FRAME_BYTES`],
    /// [`DapError::Json`] for invalid JSON payloads, or [`DapError::Io`] on stream failure.
    pub async fn read_message<R: AsyncBufRead + Unpin>(
        &mut self,
        reader: &mut R,
    ) -> Result<Option<ProtocolMessage>, DapError> {
        let Some(content_length) = self.read_headers(reader).await? else {
            return Ok(None);
        };

        while self.body_buf.len() < content_length {
            let remaining = content_length - self.body_buf.len();
            let start = self.body_buf.len();
            self.body_buf.resize(start + remaining, 0);
            let n = reader.read(&mut self.body_buf[start..]).await?;
            self.body_buf.truncate(start + n);
            if n == 0 {
                return Err(DapError::Protocol(format!(
                    "unexpected EOF after reading {} of {content_length} payload bytes",
                    self.body_buf.len()
                )));
            }
        }

        let message: ProtocolMessage = serde_json::from_slice(&self.body_buf)?;
        self.body_buf.clear();
        self.body_len = None;
        Ok(Some(message))
    }

    async fn read_headers<R: AsyncBufRead + Unpin>(
        &mut self,
        reader: &mut R,
    ) -> Result<Option<usize>, DapError> {
        if let Some(len) = self.body_len {
            return Ok(Some(len));
        }

        loop {
            let mut line = Vec::new();
            let n = reader.read_until(b'\n', &mut line).await?;
            if n == 0 {
                if self.header_buf.is_empty() {
                    return Ok(None);
                }
                return Err(DapError::Protocol(
                    "unexpected EOF while reading DAP headers".to_owned(),
                ));
            }
            if self.header_buf.len().saturating_add(line.len()) > MAX_HEADER_BYTES {
                return Err(DapError::Protocol(format!(
                    "DAP headers exceeded {MAX_HEADER_BYTES} bytes"
                )));
            }
            self.header_buf.extend_from_slice(&line);

            if line == b"\r\n" || line == b"\n" {
                let len = parse_content_length(&self.header_buf)?;
                self.header_buf.clear();
                self.body_buf.clear();
                self.body_buf.reserve(len);
                self.body_len = Some(len);
                return Ok(Some(len));
            }
        }
    }
}

/// Writes a single Content-Length framed [`ProtocolMessage`] and flushes `writer`.
///
/// # Errors
///
/// Returns [`DapError::Json`] if serialization fails or [`DapError::Io`] on write failure.
pub async fn write_message<W: AsyncWrite + Unpin>(
    writer: &mut W,
    message: &ProtocolMessage,
) -> Result<(), DapError> {
    let payload = serde_json::to_vec(message)?;
    let header = format!("Content-Length: {}\r\n\r\n", payload.len());
    writer.write_all(header.as_bytes()).await?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}

fn parse_content_length(raw_headers: &[u8]) -> Result<usize, DapError> {
    let text = std::str::from_utf8(raw_headers)
        .map_err(|err| DapError::Protocol(format!("non-UTF-8 DAP header: {err}")))?;

    let mut content_length: Option<usize> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some((name, value)) = trimmed.split_once(':') else {
            return Err(DapError::Protocol(format!(
                "malformed header line: {trimmed}"
            )));
        };
        if name.trim().eq_ignore_ascii_case("content-length") {
            let parsed = value.trim().parse::<usize>().map_err(|err| {
                DapError::Protocol(format!("invalid Content-Length `{value}`: {err}"))
            })?;
            if parsed > MAX_FRAME_BYTES {
                return Err(DapError::FrameTooLarge {
                    length: parsed,
                    limit: MAX_FRAME_BYTES,
                });
            }
            content_length = Some(parsed);
        }
    }

    content_length.ok_or_else(|| DapError::Protocol("missing Content-Length header".to_owned()))
}

#[cfg(test)]
mod tests {
    use futures_lite::io::{BufReader, Cursor};

    use super::*;
    use crate::protocol::DapRequest;

    #[tokio::test]
    async fn roundtrip_framed_messages_with_extra_headers() {
        let req = ProtocolMessage::Request(DapRequest {
            seq: 1,
            command: "threads".to_owned(),
            arguments: None,
        });
        let mut wire = Vec::new();
        write_message(&mut wire, &req).await.unwrap();

        let body = serde_json::to_vec(&req).unwrap();
        let extra = format!(
            "Content-Type: application/vscode-jsonrpc; charset=utf-8\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        wire.extend_from_slice(extra.as_bytes());
        wire.extend_from_slice(&body);

        let mut reader = BufReader::new(Cursor::new(wire));
        let mut framer = FrameReader::new();

        assert_eq!(
            framer.read_message(&mut reader).await.unwrap(),
            Some(req.clone())
        );
        assert_eq!(framer.read_message(&mut reader).await.unwrap(), Some(req));
        assert_eq!(framer.read_message(&mut reader).await.unwrap(), None);
    }

    #[tokio::test]
    async fn rejects_oversized_and_missing_content_length() {
        let mut reader = BufReader::new(Cursor::new(b"Content-Length: 999999999\r\n\r\n".to_vec()));
        let mut framer = FrameReader::new();
        let err = framer.read_message(&mut reader).await.unwrap_err();
        assert!(matches!(err, DapError::FrameTooLarge { .. }));

        let mut reader = BufReader::new(Cursor::new(
            b"Content-Type: application/json\r\n\r\n{}".to_vec(),
        ));
        let mut framer = FrameReader::new();
        let err = framer.read_message(&mut reader).await.unwrap_err();
        assert!(matches!(err, DapError::Protocol(_)));
    }
}
