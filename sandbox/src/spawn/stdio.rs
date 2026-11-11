//! Threads that move a blocking process pipe to and from channels.

use std::io::{Read, Write};

use async_channel::Sender;
use tracing::warn;

use crate::stdin_watch::FeedCounter;

/// Reads `reader` to its end on a thread, handing each chunk to `emit` as
/// `Some` and, once the stream ended or failed, `None` exactly once.
pub fn read_chunks<R>(mut reader: R, mut emit: impl FnMut(Option<Vec<u8>>) + Send + 'static)
where
    R: Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut buffer = vec![0_u8; 8192];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => emit(Some(buffer[..count].to_vec())),
                Err(error) => {
                    warn!(%error, "a process pipe failed to read");
                    break;
                }
            }
        }
        emit(None);
    });
}

/// Writes what arrives on the returned sender to `writer` on a thread, until
/// the sender is dropped or a write fails. Every non-empty chunk is recorded
/// on `feed`.
pub fn write_chunks<W>(mut writer: W, feed: FeedCounter) -> Sender<Vec<u8>>
where
    W: Write + Send + 'static,
{
    let (tx, rx) = async_channel::unbounded::<Vec<u8>>();
    std::thread::spawn(move || {
        while let Ok(bytes) = rx.recv_blocking() {
            if bytes.is_empty() {
                continue;
            }
            feed.record();
            if let Err(error) = writer.write_all(&bytes).and_then(|()| writer.flush()) {
                warn!(%error, "a process pipe failed to write");
                break;
            }
        }
    });
    tx
}
