//! Built-in sandbox tools.

mod ask;
mod terminal;

pub use ask::{AskCommand, AskModelTier};
pub use terminal::{
    JobInputArgs, JobInputTool, JobKillArgs, JobKillTool, JobReadArgs, JobReadTool,
    ShellToolSettings, ToolIdentity,
};

use heel::IpcRouter;

/// Creates an IPC router with built-in commands (excluding ask, which needs LLM).
#[must_use]
pub fn builtin_router() -> IpcRouter {
    IpcRouter::new()
}
