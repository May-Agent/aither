//! The control tools of a shell's jobs.

mod terminal;

pub use terminal::{
    JobInputArgs, JobInputTool, JobReadArgs, JobReadTool, ShellToolSettings, ToolIdentity,
};
