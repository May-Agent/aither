//! The ergonomic dual-stack ACP client.
//!
//! [`Client`] drives an agent over one of two lanes that share the same
//! domain-typed API (see [`Connection`]):
//!
//! - **wire** — [`ClientBuilder::connect`]/[`ClientBuilder::spawn`]/
//!   [`ClientBuilder::connect_stdio`] over a
//!   [`Transport`](crate::rpc::Transport);
//!   `initialize` negotiates ACP v1 or v2 and v1 traffic is translated
//!   through per-session [`V1Codec`](crate::wire::codec::V1Codec)s.
//! - **direct** — [`ClientBuilder::in_process`] /
//!   [`ClientBuilder::in_process_server`]: zero-serde, always v2
//!   semantics, straight into
//!   [`ServerCore`](crate::server::ServerCore).
//!
//! Sessions demultiplex `session/update` into [`SessionReducer`]-folded
//! snapshots plus per-session event broadcasts; [`PromptTurn`] is the
//! per-turn filtered stream with [`TurnOutcome`].

mod builder;
mod connection;
mod direct;
mod error;
mod handle;
mod handler;
mod session;
mod turn;

#[cfg(test)]
mod tests;

pub use builder::ClientBuilder;
pub use connection::{AnyAgentLane, Connection};
pub use error::ClientError;
pub use handle::{Client, ResumeBuilder, SessionBuilder};
pub use handler::{
    AnyAgentHooks, ElicitationHandler, ExtRequestHandler, FsHandler, NotificationHandler,
    PermissionHandler, V1Terminal,
};
pub use session::{ClientSession, IntoPrompt, SessionEvents};
pub use turn::{PromptTurn, TurnOutcome};

pub use crate::state::{MessageKind, SessionEvent, SessionInfoState, SessionReducer, SessionState};
