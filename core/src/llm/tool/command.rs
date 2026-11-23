//! Typed record of a command run, carried by `ToolResult::Command`.
//!
//! The model reads the prose [`CommandResult::render_for_model`] renders from
//! these fields; hosts read the fields and never parse that prose.

use alloc::string::String;
use alloc::vec::Vec;

use askama::Template;

/// A file a command's output was written to.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StoredFile {
    /// Location relative to the session sandbox (for example `outputs/x.txt`).
    pub url: String,
    /// Absolute filesystem path of the same file.
    pub path: String,
}

/// One output stream of a command.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
pub enum CommandStream {
    /// Output small enough to keep in the result.
    Inline {
        /// The output text.
        text: String,
    },
    /// Output written to a file.
    Stored {
        /// Where the full output is stored.
        file: StoredFile,
        /// What the model is shown (head/tail or folded view); empty when the
        /// output has no text view.
        preview: String,
        /// Whether `preview` is partial.
        truncated: bool,
    },
    /// Image output, always written to a file.
    Image {
        /// MIME type of the image.
        media_type: String,
        /// Where the image is stored.
        file: StoredFile,
    },
}

/// How a command run ended.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "status", rename_all = "snake_case"))]
pub enum CommandOutcome {
    /// The command ran to completion.
    Exited {
        /// Process exit code.
        code: i32,
    },
    /// A background process; its output is being written to `output`.
    Running {
        /// Process identifier of the background job.
        pid: u32,
        /// File the background output is written to.
        output: StoredFile,
    },
    /// The foreground run was killed at its time bound.
    Killed {
        /// Process identifier of the killed run.
        pid: u32,
        /// Model-facing reason the run was killed.
        notice: String,
    },
}

/// What was unavailable during an [`Interruption`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum InterruptionCause {
    /// The machine was asleep.
    Asleep,
    /// The machine had no route to the network.
    Offline,
}

/// A span of a command's run during which the machine was asleep or offline.
///
/// The times are wall-clock times of day in the machine's local time zone,
/// exactly as the model reads them.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Interruption {
    /// What was unavailable.
    pub cause: InterruptionCause,
    /// When the span began, or when the command started if it began earlier.
    pub from: String,
    /// When the span ended, or when the command ended if it ended earlier.
    pub to: String,
}

/// Renders `interruptions` as one line each; the empty string when there are
/// none.
///
/// # Panics
///
/// Panics if the template fails to render, which formatting in-memory
/// strings cannot do.
#[must_use]
pub fn render_interruptions(interruptions: &[Interruption]) -> String {
    InterruptionsTemplate { interruptions }
        .render()
        .expect("interruptions template must render")
}

/// Complete typed record of one command run.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CommandResult {
    /// Standard output.
    pub stdout: CommandStream,
    /// Standard error, when the command wrote any.
    pub stderr: Option<CommandStream>,
    /// How the run ended.
    pub outcome: CommandOutcome,
    /// Process identifier whose upstream pipeline stage outputs remain
    /// inspectable after a foreground run completed, when at least one
    /// non-final stage met the retention threshold.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub retained_pid: Option<u32>,
    /// Spans of the run during which the machine was asleep or offline.
    #[cfg_attr(feature = "serde", serde(default))]
    pub interruptions: Vec<Interruption>,
}

#[derive(Template)]
#[template(path = "command_interruptions.txt", escape = "none")]
struct InterruptionsTemplate<'a> {
    interruptions: &'a [Interruption],
}

#[derive(Template)]
#[template(path = "command_stream.txt", escape = "none")]
struct CommandStreamTemplate<'a> {
    stream: &'a CommandStream,
}

#[derive(Template)]
#[template(path = "retained_pid.txt", escape = "none")]
struct RetainedPidTemplate {
    pid: u32,
}

#[derive(Template)]
#[template(path = "command_result.txt", escape = "none")]
struct CommandResultTemplate<'a> {
    stdout: &'a str,
    stderr: Option<&'a str>,
    outcome: &'a CommandOutcome,
    retained_pid: &'a str,
    interruptions: &'a str,
}

impl CommandStream {
    /// Returns `true` when the stream holds no output.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Inline { text } if text.is_empty())
    }

    /// Renders the stream as the newline-terminated block the model reads;
    /// an empty stream renders as the empty string.
    ///
    /// # Panics
    ///
    /// Panics if the template fails to render, which formatting in-memory
    /// strings cannot do.
    #[must_use]
    pub fn render_for_model(&self) -> String {
        if self.is_empty() {
            return String::new();
        }
        let mut block = CommandStreamTemplate { stream: self }
            .render()
            .expect("command stream template must render");
        if !block.ends_with('\n') {
            block.push('\n');
        }
        block
    }
}

impl CommandResult {
    /// Renders the prose the model reads.
    ///
    /// # Panics
    ///
    /// Panics if the template fails to render, which formatting in-memory
    /// strings cannot do.
    #[must_use]
    pub fn render_for_model(&self) -> String {
        let stdout = self.stdout.render_for_model();
        let stderr = self
            .stderr
            .as_ref()
            .map(CommandStream::render_for_model)
            .filter(|block| !block.is_empty());
        let retained_pid = match (&self.outcome, self.retained_pid) {
            (CommandOutcome::Exited { .. }, Some(pid)) => RetainedPidTemplate { pid }
                .render()
                .expect("retained pid template must render"),
            _ => String::new(),
        };
        CommandResultTemplate {
            stdout: &stdout,
            stderr: stderr.as_deref(),
            outcome: &self.outcome,
            retained_pid: &retained_pid,
            interruptions: &render_interruptions(&self.interruptions),
        }
        .render()
        .expect("command result template must render")
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    fn file() -> StoredFile {
        StoredFile {
            url: "outputs/x.txt".to_string(),
            path: "/sandbox/outputs/x.txt".to_string(),
        }
    }

    fn result(
        stdout: CommandStream,
        stderr: Option<CommandStream>,
        outcome: CommandOutcome,
    ) -> String {
        CommandResult {
            stdout,
            stderr,
            outcome,
            retained_pid: None,
            interruptions: Vec::new(),
        }
        .render_for_model()
    }

    #[test]
    fn placeholder_only_when_both_streams_are_empty() {
        assert_eq!(
            result(
                CommandStream::Inline {
                    text: String::new()
                },
                None,
                CommandOutcome::Exited { code: 0 }
            ),
            "(no output)\n"
        );
        assert_eq!(
            result(
                CommandStream::Inline {
                    text: String::new()
                },
                Some(CommandStream::Inline {
                    text: "oops".to_string()
                }),
                CommandOutcome::Exited { code: 2 },
            ),
            "stderr:\noops\nExit code: 2.\n"
        );
    }

    #[test]
    fn stored_streams_state_url_and_absolute_path() {
        let stored = CommandStream::Stored {
            file: file(),
            preview: "head".to_string(),
            truncated: true,
        };
        assert_eq!(
            result(stored, None, CommandOutcome::Exited { code: 0 }),
            "head\n[Partial view: the full output is saved to outputs/x.txt (absolute path: /sandbox/outputs/x.txt). Use head/tail/rg on that file to read the rest.]\n"
        );
        let image = CommandStream::Image {
            media_type: "image/png".to_string(),
            file: file(),
        };
        assert_eq!(
            result(image, None, CommandOutcome::Exited { code: 0 }),
            "[Image: image/png, saved to outputs/x.txt (absolute path: /sandbox/outputs/x.txt).]\n"
        );
    }

    #[test]
    fn running_and_killed_outcomes_render_their_facts() {
        assert_eq!(
            result(
                CommandStream::Inline {
                    text: String::new()
                },
                None,
                CommandOutcome::Running {
                    pid: 4242,
                    output: file(),
                },
            ),
            "Started in background with PID 4242; output is being written to outputs/x.txt (absolute path: /sandbox/outputs/x.txt).\n"
        );
        assert_eq!(
            result(
                CommandStream::Inline {
                    text: "partial\n".to_string()
                },
                None,
                CommandOutcome::Killed {
                    pid: 4243,
                    notice: "Killed at the time bound.".to_string(),
                },
            ),
            "partial\nKilled at the time bound.\nPID: 4243.\n"
        );
    }

    #[test]
    fn interruptions_render_one_line_each_after_the_outcome() {
        let command = CommandResult {
            stdout: CommandStream::Inline {
                text: "out\n".to_string(),
            },
            stderr: None,
            outcome: CommandOutcome::Exited { code: 1 },
            retained_pid: None,
            interruptions: alloc::vec![
                Interruption {
                    cause: InterruptionCause::Asleep,
                    from: "14:02:11".to_string(),
                    to: "14:31:40".to_string(),
                },
                Interruption {
                    cause: InterruptionCause::Offline,
                    from: "14:31:40".to_string(),
                    to: "14:32:05".to_string(),
                },
            ],
        };
        assert_eq!(
            command.render_for_model(),
            "out\nExit code: 1.\nThe machine was asleep from 14:02:11 to 14:31:40 while this ran.\nThe network was offline from 14:31:40 to 14:32:05 while this ran.\n"
        );
    }

    #[test]
    fn retained_pid_renders_only_for_completed_foreground_runs() {
        let exited = CommandResult {
            stdout: CommandStream::Inline {
                text: String::new(),
            },
            stderr: None,
            outcome: CommandOutcome::Exited { code: 1 },
            retained_pid: Some(48210),
            interruptions: Vec::new(),
        };
        assert_eq!(
            exited.render_for_model(),
            "(no output)\nExit code: 1.\nPID: 48210.\n"
        );

        let running = CommandResult {
            stdout: CommandStream::Inline {
                text: String::new(),
            },
            stderr: None,
            outcome: CommandOutcome::Running {
                pid: 48210,
                output: file(),
            },
            retained_pid: Some(48210),
            interruptions: Vec::new(),
        };
        assert_eq!(
            running.render_for_model(),
            "Started in background with PID 48210; output is being written to outputs/x.txt (absolute path: /sandbox/outputs/x.txt).\n"
        );
    }
}
