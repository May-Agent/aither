//! Wire codecs for protocol version negotiation.
//!
//! - [`v1`]: the legacy ACP v1 wire model.
//! - [`v2`]: v2 request/result shapes that differ from
//!   [`crate::domain`].
//! - [`codec`]: the stateful v1↔v2 translator.

pub mod codec;
pub mod v1;
pub mod v2;

pub use codec::V1Codec;
