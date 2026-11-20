//! Shared ACP extension conventions.
//!
//! Extensions are ACP's escape hatches: `_`-prefixed JSON-RPC methods,
//! `_`-prefixed `sessionUpdate` tags, and vendorable `_meta` keys. This
//! module models the provider-neutral conventions — usable with any
//! agent that advertises them — the way `std` carries more than `core`.
//! Genuinely private provider surface lives under [`crate::vendor`]
//! instead.
//!
//! Each submodule exposes the extension's
//! [`ExtMethod`](crate::domain::ExtMethod) name (where the extension is a
//! method), its capability probe, and its param/result serde types.
//! Helpers that need a live connection
//! ([`Client`](crate::client::Client) /
//! [`SessionContext`](crate::server::SessionContext)) are marked
//! `TODO(wave-3)` until those types land.

pub mod async_tasks;
pub mod auth_status;
pub mod goal;
pub mod steering;
pub mod subagents;

use serde_json::Value;

use crate::domain::SessionUpdate;

/// Extracts `(tag, payload)` from a [`SessionUpdate::Custom`] or
/// [`SessionUpdate::Unknown`] update, stripping the `sessionUpdate`
/// discriminator from the payload.
///
/// `None` for every standard variant — those are already decoded.
pub(crate) fn tagged_payload(update: &SessionUpdate) -> Option<(String, Value)> {
    let (SessionUpdate::Custom(_, raw) | SessionUpdate::Unknown(_, raw)) = update else {
        return None;
    };
    let mut object = raw.as_object()?.clone();
    let tag = object.remove("sessionUpdate")?.as_str()?.to_owned();
    Some((tag, Value::Object(object)))
}
