//! Version-agnostic JSON-RPC 2.0 full-duplex engine.
//!
//! `rpc` owns the wire layer every ACP connection shares, independent of
//! protocol version:
//!
//! - [`Request`], [`Notification`], [`Response`], [`RequestId`] — the
//!   message types, with `"jsonrpc": "2.0"` emitted as a constant.
//! - [`Inbound`]/[`Outbound`] — direction-tagged envelopes; inbound
//!   classification rejects batches and `null` request ids.
//! - [`RpcError`] — the wire error object plus the codes ACP assigns.
//! - [`Transport`] — the framed channel trait, with [`Lines`] (NDJSON over
//!   byte streams), [`Memory`] (in-process), and the feature-gated
//!   [`Stdio`] and [`ChildProcess`] implementations.
//! - [`connection`] — the engine pairing a [`Peer`] handle with the
//!   [`Connection`] task that owns the transport, dispatches to a
//!   [`Handler`], and honors `$/cancel_request` in both directions.

mod engine;
mod error;
mod lines;
mod message;
mod transport;

pub use engine::{
    Connection, Handler, Peer, PeerError, RequestContext, ResponseFuture, connection,
};
pub use error::{
    AUTH_REQUIRED, INTERNAL_ERROR, INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND, PARSE_ERROR,
    PROTOCOL_VERSION_UNSUPPORTED, REQUEST_CANCELLED, RESOURCE_NOT_FOUND, RpcError,
};
pub use lines::Lines;
pub use message::{
    CANCEL_REQUEST_METHOD, Inbound, Notification, Outbound, Request, RequestId, Response,
};
#[cfg(feature = "process")]
pub use transport::ChildProcess;
#[cfg(feature = "stdio")]
pub use transport::Stdio;
pub use transport::{Memory, Transport};
