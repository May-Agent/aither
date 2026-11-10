//! The `heel` binary tests run under.
//!
//! The embedder prepares heel for a sandbox; tests take the binary a
//! developer or CI provides: `HEEL_BIN`, else `heel` on `PATH`, verified
//! against the linked library's version.

#![allow(clippy::missing_panics_doc, clippy::must_use_candidate)]

use heel::HeelBinary;

/// The `heel` binary [`HeelBinary::locate`] finds, verified against the
/// linked library.
///
/// # Errors
/// Fails when no `heel` is provided or it is not the linked library's
/// version.
pub async fn binary() -> heel::Result<HeelBinary> {
    blocking::unblock(HeelBinary::locate).await
}
