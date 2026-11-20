//! Vendor-specific ACP surface.
//!
//! ACP's extension mechanism (`_` methods, `_meta` keys) lets providers
//! ship agent features before — or instead of — standardizing them. The
//! modules here model what a specific provider actually speaks on the
//! wire. Anything usable across providers belongs in the common
//! [`domain`](crate::domain) layer or the shared [`ext`](crate::ext)
//! conventions instead; nothing in the common layer references these
//! modules.
//!
//! Helpers that need a live connection
//! ([`Client`](crate::client::Client)) are marked `TODO(wave-3)` until it
//! lands.

pub mod codex;
pub mod devin;
pub mod jetbrains;
