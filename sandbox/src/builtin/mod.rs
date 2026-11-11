//! The control tools of a shell's jobs.

mod terminal;

pub use terminal::{
    JobInputArgs, JobInputTool, JobKillArgs, JobKillTool, JobReadArgs, JobReadTool,
    ShellToolSettings, ToolIdentity,
};
