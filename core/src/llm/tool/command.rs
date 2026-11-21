//! Typed record of a command run, carried by `ToolResult::Command`.
//!
//! The model reads the prose [`CommandResult::render_for_model`] (or
//! [`CommandResult::render_for_model_with`] for embedders supplying custom
//! [`CommandWording`]) renders from these fields; hosts read the fields and
//! never parse that prose.

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
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub stderr: Option<CommandStream>,
    /// How the run ended.
    pub outcome: CommandOutcome,
    /// Spans of the run during which the machine was asleep or offline.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Vec::is_empty")
    )]
    pub interruptions: Vec<Interruption>,
}

/// Customizes the prose [`CommandResult::render_for_model_with`],
/// [`CommandStream::render_for_model_with`], and
/// [`ToolResult::render_for_model_with`](super::ToolResult::render_for_model_with)
/// render for the model.
///
/// Every method carries a neutral default free of shell or product tool names;
/// an embedder overrides the parts whose wording belongs to its own runtime.
pub trait CommandWording: Send + Sync {
    /// Renders a stored text output stream (full or partial preview).
    fn stored_stream(&self, file: &StoredFile, preview: &str, truncated: bool) -> String {
        let stream = CommandStream::Stored {
            file: file.clone(),
            preview: String::from(preview),
            truncated,
        };
        CommandStreamTemplate { stream: &stream }
            .render()
            .expect("command stream template must render")
    }

    /// Renders a stored image output stream.
    fn image_stream(&self, media_type: &str, file: &StoredFile) -> String {
        let stream = CommandStream::Image {
            media_type: String::from(media_type),
            file: file.clone(),
        };
        CommandStreamTemplate { stream: &stream }
            .render()
            .expect("command stream template must render")
    }

    /// Renders the placeholder shown when a completed command produced neither
    /// stdout nor stderr.
    fn empty_output(&self) -> String {
        String::from("(no output)\n")
    }

    /// Renders the command's outcome trailer (`Exit code: ...`, background
    /// start, or kill notice). Returns the empty string for exit code `0`.
    fn outcome(&self, outcome: &CommandOutcome) -> String {
        CommandOutcomeTemplate { outcome }
            .render()
            .expect("command outcome template must render")
    }

    /// Renders `interruptions` appended after the outcome.
    fn interruptions(&self, interruptions: &[Interruption]) -> String {
        render_interruptions(interruptions)
    }

    /// Renders a binary [`ToolResult`](super::ToolResult) or
    /// [`ToolResultPart`](super::ToolResultPart) placeholder.
    fn binary_placeholder(&self, mime: &str, byte_count: usize) -> String {
        BinaryPlaceholderTemplate {
            mime,
            bytes: byte_count,
        }
        .render()
        .expect("binary placeholder template must render")
    }
}

/// The neutral [`CommandWording`] used by [`CommandResult::render_for_model`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DefaultCommandWording;

impl CommandWording for DefaultCommandWording {}

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
#[template(path = "command_outcome.txt", escape = "none")]
struct CommandOutcomeTemplate<'a> {
    outcome: &'a CommandOutcome,
}

#[derive(Template)]
#[template(path = "command_result.txt", escape = "none")]
struct CommandResultTemplate<'a> {
    empty: &'a str,
    stdout: &'a str,
    stderr: Option<&'a str>,
    outcome: &'a str,
    interruptions: &'a str,
}

#[derive(Template)]
#[template(path = "binary_placeholder.txt", escape = "none")]
struct BinaryPlaceholderTemplate<'a> {
    mime: &'a str,
    bytes: usize,
}

impl CommandStream {
    /// Returns `true` when the stream holds no output.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Inline { text } if text.is_empty())
    }

    /// Renders the stream as the newline-terminated block the model reads using
    /// [`DefaultCommandWording`]; an empty stream renders as the empty string.
    ///
    /// # Panics
    ///
    /// Panics if the template fails to render, which formatting in-memory
    /// strings cannot do.
    #[must_use]
    pub fn render_for_model(&self) -> String {
        self.render_for_model_with(&DefaultCommandWording)
    }

    /// Renders the stream as the newline-terminated block the model reads using
    /// `wording`; an empty stream renders as the empty string.
    #[must_use]
    pub fn render_for_model_with(&self, wording: &dyn CommandWording) -> String {
        if self.is_empty() {
            return String::new();
        }
        let mut block = match self {
            Self::Inline { text } => text.clone(),
            Self::Stored {
                file,
                preview,
                truncated,
            } => wording.stored_stream(file, preview, *truncated),
            Self::Image { media_type, file } => wording.image_stream(media_type, file),
        };
        if !block.ends_with('\n') {
            block.push('\n');
        }
        block
    }
}

impl CommandResult {
    /// Renders the prose the model reads using [`DefaultCommandWording`].
    ///
    /// # Panics
    ///
    /// Panics if the template fails to render, which formatting in-memory
    /// strings cannot do.
    #[must_use]
    pub fn render_for_model(&self) -> String {
        self.render_for_model_with(&DefaultCommandWording)
    }

    /// Renders the prose the model reads using `wording`.
    ///
    /// # Panics
    ///
    /// Panics if the template fails to render, which formatting in-memory
    /// strings cannot do.
    #[must_use]
    pub fn render_for_model_with(&self, wording: &dyn CommandWording) -> String {
        let stdout = self.stdout.render_for_model_with(wording);
        let stderr = self
            .stderr
            .as_ref()
            .map(|stream| stream.render_for_model_with(wording))
            .filter(|block| !block.is_empty());
        let empty = if stdout.is_empty()
            && stderr.is_none()
            && !matches!(self.outcome, CommandOutcome::Running { .. })
        {
            wording.empty_output()
        } else {
            String::new()
        };
        let outcome = wording.outcome(&self.outcome);
        let interruptions = wording.interruptions(&self.interruptions);
        CommandResultTemplate {
            empty: &empty,
            stdout: &stdout,
            stderr: stderr.as_deref(),
            outcome: &outcome,
            interruptions: &interruptions,
        }
        .render()
        .expect("command result template must render")
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
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
    fn stored_streams_state_url_and_absolute_path_without_shell_tool_names() {
        let stored = CommandStream::Stored {
            file: file(),
            preview: "head".to_string(),
            truncated: true,
        };
        let rendered = result(stored, None, CommandOutcome::Exited { code: 0 });
        assert_eq!(
            rendered,
            "head\n[Partial view: the full output is saved to outputs/x.txt (absolute path: /sandbox/outputs/x.txt).]\n"
        );
        assert!(
            !rendered.contains("head/tail/rg"),
            "default wording must not hardcode shell tool names: {rendered}"
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
    fn custom_wording_overrides_stored_stream_and_outcome_rendering() {
        struct CustomWording;

        impl CommandWording for CustomWording {
            fn stored_stream(&self, file: &StoredFile, preview: &str, truncated: bool) -> String {
                if truncated {
                    format!("{preview}\n<saved:{}>", file.path)
                } else {
                    format!("<full:{}>", file.path)
                }
            }

            fn outcome(&self, outcome: &CommandOutcome) -> String {
                match outcome {
                    CommandOutcome::Exited { code: 0 } => String::new(),
                    CommandOutcome::Exited { code } => format!("status={code}\n"),
                    _ => DefaultCommandWording.outcome(outcome),
                }
            }
        }

        let command = CommandResult {
            stdout: CommandStream::Stored {
                file: file(),
                preview: "line 1".to_string(),
                truncated: true,
            },
            stderr: None,
            outcome: CommandOutcome::Exited { code: 7 },
            interruptions: Vec::new(),
        };
        assert_eq!(
            command.render_for_model_with(&CustomWording),
            "line 1\n<saved:/sandbox/outputs/x.txt>\nstatus=7\n"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn command_result_skips_empty_stderr_and_interruptions_on_wire() {
        let command = CommandResult {
            stdout: CommandStream::Inline {
                text: "ok\n".to_string(),
            },
            stderr: None,
            outcome: CommandOutcome::Exited { code: 0 },
            interruptions: Vec::new(),
        };
        let value = serde_json::to_value(&command).expect("serializes");
        let obj = value.as_object().expect("object");
        assert!(
            !obj.contains_key("stderr"),
            "None stderr must be omitted: {value}"
        );
        assert!(
            !obj.contains_key("interruptions"),
            "empty interruptions must be omitted: {value}"
        );
        let decoded: CommandResult = serde_json::from_value(value).expect("deserializes");
        assert_eq!(decoded, command);

        let with_fields = CommandResult {
            stdout: CommandStream::Inline {
                text: String::new(),
            },
            stderr: Some(CommandStream::Inline {
                text: "warn\n".to_string(),
            }),
            outcome: CommandOutcome::Exited { code: 1 },
            interruptions: alloc::vec![Interruption {
                cause: InterruptionCause::Offline,
                from: "10:00:00".to_string(),
                to: "10:01:00".to_string(),
            }],
        };
        let value = serde_json::to_value(&with_fields).expect("serializes");
        let obj = value.as_object().expect("object");
        assert!(obj.contains_key("stderr"));
        assert!(obj.contains_key("interruptions"));
        let decoded: CommandResult = serde_json::from_value(value).expect("deserializes");
        assert_eq!(decoded, with_fields);
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
}
