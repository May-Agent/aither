//! The text the model reads: every refusal, notice, marker and reply the
//! shell and its tools render.
//!
//! The shell raises each as typed data: a [`CommandBlocked`], a
//! [`MonitorStarted`], an [`OutputSaved`] and so on. The embedder's
//! [`ShellWording`] turns each into the natural language its model reads;
//! aither ships no implementation, so an embedder words every family itself.
//! A [`ScriptPolicy`](crate::ScriptPolicy) refusal carries its own text and
//! has no family here.

use std::fmt;
use std::sync::Arc;

use crate::command::CommandBlocked;
use crate::job_registry::UnchangedTaskRead;
use crate::monitor::MonitorRefusal;
use crate::terminal::TimeoutKill;

/// A `monitor` call that started watching a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonitorStarted<'a> {
    /// The task the monitor watches.
    pub task_id: &'a str,
    /// How often its new output is delivered, in seconds.
    pub interval_secs: u64,
}

/// A command invoked by a subagent that only the main agent may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReservedForMain<'a> {
    /// The command's name.
    pub command: &'a str,
}

/// Why a foreground command the shell killed was killed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandKilled<'a> {
    /// Its `timeout` bound passed.
    TimedOut(&'a TimeoutKill),
    /// It was killed before its bound passed, by a kill or a watchdog.
    BeforeTimeout,
}

/// An output too large for the window, replaced by a preview and the file
/// that holds all of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputSaved<'a> {
    /// The preview kept in the result, without its trailing newline.
    pub preview: &'a str,
    /// Absolute path of the file holding the full output.
    pub path: &'a str,
    /// Lines of the full output.
    pub line_count: usize,
    /// Bytes of the full output.
    pub byte_count: usize,
}

/// A binary part of a result, stored in a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredBinaryPart<'a> {
    /// The part's media type.
    pub mime: &'a str,
    /// Its size.
    pub byte_count: usize,
    /// The URL the output store names it by.
    pub url: &'a str,
    /// Its file.
    pub path: &'a str,
}

/// A JSON payload shown as TSV, its original stored in a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonAsTsv<'a> {
    /// The TSV, without its trailing newline.
    pub tsv: &'a str,
    /// The URL the original is stored under.
    pub url: &'a str,
    /// The file holding the original.
    pub path: &'a str,
}

/// A run of lines a view of an output leaves out or stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineRange {
    /// The first stored line of the range, 1-based.
    pub first: usize,
    /// The last stored line of the range.
    pub last: usize,
}

/// A block of source code folded into one marker line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldedLines<'a> {
    /// The indentation matching the line numbers' width.
    pub indent: &'a str,
    /// The lines folded.
    pub count: usize,
    /// The unfolded lines the marker stands for.
    pub range: LineRange,
}

/// The middle of a stored output a preview leaves out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OmittedLines {
    /// The lines left out.
    pub range: LineRange,
    /// The lines of the whole output.
    pub total: usize,
}

/// A run of identical lines a preview shows once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepeatedLines {
    /// The lines the run spans.
    pub range: LineRange,
    /// The lines in the run.
    pub count: usize,
}

/// The text the model reads for each refusal, notice and marker the shell
/// raises, one method per family.
///
/// There is no default for any method and aither ships no implementation.
pub trait ShellWording: Send + Sync + 'static {
    /// A background read refused because the task's output has not changed
    /// since its last read.
    fn unchanged_task_read(&self, read: &UnchangedTaskRead) -> String;

    /// A command the shell's command gate blocked, wrapping the reason the
    /// gate gave.
    fn command_blocked(&self, blocked: &CommandBlocked) -> String;

    /// A `monitor` call refused before anything started.
    fn monitor_refusal(&self, refusal: &MonitorRefusal) -> String;

    /// The result of a `monitor` call that started watching.
    fn monitor_started(&self, started: &MonitorStarted<'_>) -> String;

    /// A subagent's call of a command reserved for the main agent.
    fn reserved_for_main(&self, reserved: &ReservedForMain<'_>) -> String;

    /// Why a command was killed, appended to its result.
    fn command_killed(&self, killed: &CommandKilled<'_>) -> String;

    /// A text output replaced by its preview and the file holding all of it.
    fn output_saved(&self, saved: &OutputSaved<'_>) -> String;

    /// A binary part of a result, stored in a file.
    fn stored_binary_part(&self, part: &StoredBinaryPart<'_>) -> String;

    /// A JSON payload shown as TSV.
    fn json_as_tsv(&self, json: &JsonAsTsv<'_>) -> String;

    /// A folded view of source code, without its trailing newline.
    fn folded_code(&self, folded: &str) -> String;

    /// A run of `count` blank lines collapsed into one marker line.
    fn blank_lines(&self, count: usize) -> String;

    /// A block of folded lines.
    fn folded_lines(&self, folded: &FoldedLines<'_>) -> String;

    /// The lines a preview omits.
    fn omitted_lines(&self, omitted: &OmittedLines) -> String;

    /// A run of identical lines a preview collapses.
    fn repeated_lines(&self, repeated: &RepeatedLines) -> String;
}

/// A shared [`ShellWording`], cloned into every shell, store and tool that
/// renders a text.
#[derive(Clone)]
pub struct AnyShellWording(Arc<dyn ShellWording>);

impl AnyShellWording {
    /// Shares `wording`.
    #[must_use]
    pub fn new(wording: impl ShellWording) -> Self {
        Self(Arc::new(wording))
    }
}

/// Shares a value another wrapper already holds, so both render through the
/// same object.
impl From<Arc<dyn ShellWording>> for AnyShellWording {
    fn from(wording: Arc<dyn ShellWording>) -> Self {
        Self(wording)
    }
}

impl std::ops::Deref for AnyShellWording {
    type Target = dyn ShellWording;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl fmt::Debug for AnyShellWording {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyShellWording").finish_non_exhaustive()
    }
}

/// A call refused before it ran: the typed reason, and the text its
/// wording rendered for the model.
///
/// Tools return it as their error, so an agent loop tells one refusal from
/// another, and from any other failure, by downcasting to
/// `Refused<CommandBlocked>` and so on.
#[derive(Debug)]
pub struct Refused<R> {
    reason: R,
    text: String,
}

impl<R> Refused<R> {
    /// The refusal of `reason`, read by the model as what `render` returns.
    pub fn rendered(reason: R, render: impl FnOnce(&R) -> String) -> Self {
        let text = render(&reason);
        Self { reason, text }
    }

    /// Why the call was refused.
    pub const fn reason(&self) -> &R {
        &self.reason
    }

    /// The text the model reads.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Takes the reason, dropping the text.
    pub fn into_reason(self) -> R {
        self.reason
    }
}

impl<R> fmt::Display for Refused<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

impl<R: fmt::Debug> std::error::Error for Refused<R> {}

