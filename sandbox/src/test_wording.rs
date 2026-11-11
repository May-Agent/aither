//! A wording for tests.
//!
//! aither ships no [`ShellWording`]; the tests of the shell and of crates
//! built on it read the text of this one. Its texts stand for nothing but the
//! assertions made against them.

use askama::Template;

use crate::job_registry::{JobRead, JobStatus};
use crate::monitor::MonitorRefusal;
use crate::terminal::{TerminalError, TimeoutKill};
use crate::wording::{
    AnyShellWording, CallRefusal, CommandKilled, FoldedLines, JobInputSent, JobKillAsked,
    JsonAsTsv, MonitorStarted, OmittedLines, OutputSaved, RepeatedLines, RunFailure, ShellWording,
    StoredBinaryPart,
};
use aither_core::llm::CommandStream;

#[derive(Template)]
#[template(path = "test_wording/unchanged_task_read.txt")]
struct UnchangedTaskReadText<'a> {
    task_id: &'a str,
}

#[derive(Template)]
#[template(path = "test_wording/job_read.txt", escape = "none")]
struct JobReadText<'a> {
    delta: &'a str,
    has_more: bool,
    cursor: usize,
    task_id: &'a str,
    status_text: &'a str,
    bytes_read: usize,
    total_bytes: usize,
}

#[derive(Template)]
#[template(path = "test_wording/job_input.txt", escape = "none")]
struct JobInputText<'a> {
    task_id: &'a str,
}

#[derive(Template)]
#[template(path = "test_wording/job_kill.txt", escape = "none")]
struct JobKillText<'a> {
    task_id: &'a str,
    killed: bool,
}

fn job_status_text(status: &JobStatus) -> String {
    match status {
        JobStatus::Running => "is running".to_string(),
        JobStatus::Completed { exit_code, .. } => format!("completed with exit code {exit_code}"),
        JobStatus::Failed { error, .. } => format!("failed: {error}"),
        JobStatus::Killed { .. } => "was killed".to_string(),
    }
}

#[derive(Template)]
#[template(path = "test_wording/call_refused.txt", escape = "none")]
struct CallRefused<'a> {
    refusal: &'a CallRefusal,
}

#[derive(Template)]
#[template(path = "test_wording/run_failure.txt", escape = "none")]
struct RunFailureText<'a> {
    key: &'a str,
    detail: &'a str,
}

/// The key and detail a test wording gives a terminal failure.
fn terminal_failure(error: &TerminalError) -> (&'static str, String) {
    match error {
        TerminalError::TaskLost => ("task lost", String::new()),
        TerminalError::InvalidTarget(detail) => ("invalid target", detail.clone()),
        TerminalError::UnknownLocation(name) => ("unknown location", name.clone()),
        TerminalError::NoRemoteSandbox(name) => ("no remote sandbox", name.clone()),
        TerminalError::SshPrepare(error) => ("ssh prepare", error.to_string()),
        TerminalError::Denied(reason) => ("denied", reason.clone()),
        TerminalError::NoApprover(kind) => ("no approver", kind.to_string()),
        TerminalError::SandboxSetup(detail) => ("sandbox setup failed", detail.clone()),
        TerminalError::Execution(detail) => ("execution failed", detail.clone()),
        TerminalError::Approval(error) => ("approval", error.to_string()),
        TerminalError::Io(error) => ("io", error.to_string()),
        #[cfg(all(unix, feature = "jobserver"))]
        TerminalError::Jobserver(error) => ("jobserver", error.to_string()),
    }
}

#[derive(Template)]
#[template(path = "test_wording/monitor_refused.txt", escape = "none")]
struct MonitorRefused<'a> {
    refusal: &'a MonitorRefusal,
    max_timeout_secs: u64,
}

#[derive(Template)]
#[template(path = "test_wording/monitor_started.txt", escape = "none")]
struct MonitorStartedText<'a> {
    task_id: &'a str,
    interval_secs: u64,
}

#[derive(Template)]
#[template(path = "test_wording/command_timed_out.txt", escape = "none")]
struct Killed<'a> {
    killed: CommandKilled<'a>,
}

#[derive(Template)]
#[template(path = "test_wording/command_output_saved.txt", escape = "none")]
struct OutputSavedText<'a> {
    preview: &'a str,
    path: &'a str,
    line_count: usize,
    byte_count: usize,
}

#[derive(Template)]
#[template(path = "test_wording/stored_binary_part.txt", escape = "none")]
struct StoredBinaryPartText<'a> {
    mime: &'a str,
    byte_count: usize,
    url: &'a str,
    path: &'a str,
}

#[derive(Template)]
#[template(path = "test_wording/json_tsv_output.txt", escape = "none")]
struct JsonTsvText<'a> {
    tsv: &'a str,
    url: &'a str,
    path: &'a str,
}

#[derive(Template)]
#[template(path = "test_wording/folded_code_output.txt", escape = "none")]
struct FoldedCodeText<'a> {
    folded: &'a str,
}

#[derive(Template)]
#[template(path = "test_wording/blank_lines_marker.txt", escape = "none")]
struct BlankLinesText {
    count: usize,
}

#[derive(Template)]
#[template(path = "test_wording/folded_lines_marker.txt", escape = "none")]
struct FoldedLinesText<'a> {
    indent: &'a str,
    count: usize,
    first: usize,
    last: usize,
}

#[derive(Template)]
#[template(path = "test_wording/omitted_lines_marker.txt", escape = "none")]
struct OmittedLinesText {
    first: usize,
    last: usize,
    total: usize,
}

#[derive(Template)]
#[template(path = "test_wording/repeated_lines_marker.txt", escape = "none")]
struct RepeatedLinesText {
    first: usize,
    last: usize,
    count: usize,
}

fn rendered(template: &impl Template) -> String {
    template
        .render()
        .expect("a test wording template renders typed input")
}

/// The argument descriptions the sandbox's tests give every tool: one entry
/// for each argument any of the shell's tools takes.
#[must_use]
pub fn argument_descriptions()
-> std::collections::BTreeMap<&'static str, std::borrow::Cow<'static, str>> {
    [
        ("description", "What the command does."),
        ("script", "The source to run."),
        ("interpreter", "The interpreter of the source."),
        ("sandbox", "Whether the command runs inside the sandbox."),
        ("location", "Where the command runs."),
        ("resolution", "The resolution of returned media."),
        ("max_lines", "The most output lines returned inline."),
        ("raw", "Whether the output is returned verbatim."),
        ("timeout", "The maximum runtime in seconds."),
        ("background", "Whether the command is detached at once."),
        ("task_id", "The job."),
        ("cursor", "The offset after which output is read."),
        ("max_bytes", "The most bytes returned."),
        ("input", "The text written to stdin."),
        ("append_newline", "Whether a newline is appended."),
        ("source", "What to watch."),
        ("command", "The source to run."),
        ("file", "The file to follow."),
        ("interval_secs", "Seconds between collections."),
        ("filter", "Only matching lines are delivered."),
        ("stop_on", "A matching line ends the monitor."),
        ("kill_on_stop", "Whether the process is killed on a stop."),
        ("stall_secs", "Seconds without output before a stall."),
        ("timeout_secs", "The monitor's lifetime in seconds."),
    ]
    .into_iter()
    .map(|(name, description)| (name, description.into()))
    .collect()
}

/// The names and descriptions the sandbox's tests give the job control
/// tools.
#[must_use]
pub fn job_tools() -> crate::ShellToolSettings {
    let identity = |name: &'static str, description: &'static str| crate::ToolIdentity {
        name: name.into(),
        description: description.into(),
        arguments: argument_descriptions(),
    };
    crate::ShellToolSettings {
        shell: identity("terminal", "Run a script."),
        read: identity("terminal_read", "Read a job's output."),
        input: identity("terminal_input", "Write to a job's stdin."),
        kill: identity("terminal_kill", "Kill a job."),
    }
}

/// The wording the sandbox's tests read.
#[derive(Debug, Clone, Copy, Default)]
pub struct TestWording;

impl TestWording {
    /// This wording, shared.
    #[must_use]
    pub fn any() -> AnyShellWording {
        AnyShellWording::new(Self)
    }
}

impl ShellWording for TestWording {
    fn job_read(&self, read: &JobRead) -> String {
        match read {
            JobRead::Unchanged { task_id } => rendered(&UnchangedTaskReadText { task_id }),
            JobRead::Delta(delta) => {
                let output = CommandStream::Inline {
                    text: String::from_utf8_lossy(&delta.bytes).into_owned(),
                }
                .render_for_model();
                rendered(&JobReadText {
                    delta: &output,
                    has_more: delta.cursor < delta.total_bytes,
                    cursor: delta.cursor,
                    task_id: &delta.task_id,
                    status_text: &job_status_text(&delta.status),
                    bytes_read: delta.bytes.len(),
                    total_bytes: delta.total_bytes,
                })
            }
        }
    }

    fn job_input(&self, sent: &JobInputSent<'_>) -> String {
        rendered(&JobInputText {
            task_id: sent.task_id,
        })
    }

    fn job_kill(&self, kill: &JobKillAsked<'_>) -> String {
        rendered(&JobKillText {
            task_id: kill.task_id,
            killed: kill.killed,
        })
    }

    fn call_refused(&self, refusal: &CallRefusal) -> String {
        rendered(&CallRefused { refusal })
    }

    fn run_failure(&self, failure: &RunFailure<'_>) -> String {
        let (key, detail) = match failure {
            RunFailure::NoWorkspaceMount => ("no workspace mount", String::new()),
            RunFailure::OutsideWorkspace(error) => ("outside workspace", error.to_string()),
            RunFailure::IdsExhausted => ("ids exhausted", String::new()),
            RunFailure::Terminal(error) => terminal_failure(error),
            RunFailure::Failed { detail } => ("execution failed", (*detail).to_string()),
        };
        rendered(&RunFailureText {
            key,
            detail: &detail,
        })
    }

    fn monitor_refusal(&self, refusal: &MonitorRefusal) -> String {
        let max_timeout_secs = match refusal {
            MonitorRefusal::TimeoutOutOfRange { max } => max.as_secs(),
            _ => 0,
        };
        rendered(&MonitorRefused {
            refusal,
            max_timeout_secs,
        })
    }

    fn monitor_started(&self, started: &MonitorStarted<'_>) -> String {
        rendered(&MonitorStartedText {
            task_id: started.task_id,
            interval_secs: started.interval_secs,
        })
    }

    fn command_killed(&self, killed: &CommandKilled<'_>) -> String {
        rendered(&Killed { killed: *killed })
    }

    fn output_saved(&self, saved: &OutputSaved<'_>) -> String {
        rendered(&OutputSavedText {
            preview: saved.preview,
            path: saved.path,
            line_count: saved.line_count,
            byte_count: saved.byte_count,
        })
    }

    fn stored_binary_part(&self, part: &StoredBinaryPart<'_>) -> String {
        rendered(&StoredBinaryPartText {
            mime: part.mime,
            byte_count: part.byte_count,
            url: part.url,
            path: part.path,
        })
    }

    fn json_as_tsv(&self, json: &JsonAsTsv<'_>) -> String {
        rendered(&JsonTsvText {
            tsv: json.tsv,
            url: json.url,
            path: json.path,
        })
    }

    fn folded_code(&self, folded: &str) -> String {
        rendered(&FoldedCodeText { folded })
    }

    fn blank_lines(&self, count: usize) -> String {
        rendered(&BlankLinesText { count })
    }

    fn folded_lines(&self, folded: &FoldedLines<'_>) -> String {
        rendered(&FoldedLinesText {
            indent: folded.indent,
            count: folded.count,
            first: folded.range.first,
            last: folded.range.last,
        })
    }

    fn omitted_lines(&self, omitted: &OmittedLines) -> String {
        rendered(&OmittedLinesText {
            first: omitted.range.first,
            last: omitted.range.last,
            total: omitted.total,
        })
    }

    fn repeated_lines(&self, repeated: &RepeatedLines) -> String {
        rendered(&RepeatedLinesText {
            first: repeated.range.first,
            last: repeated.range.last,
            count: repeated.count,
        })
    }
}

/// The kill bound a test names.
#[must_use]
pub fn timed_out(kill: &TimeoutKill) -> String {
    TestWording.command_killed(&CommandKilled::TimedOut(kill))
}
