//! A wording for tests.
//!
//! aither ships no [`ShellWording`]; the tests of the shell and of crates
//! built on it read the text of this one. Its texts stand for nothing but the
//! assertions made against them.

use askama::Template;

use crate::command::CommandBlocked;
use crate::job_registry::UnchangedTaskRead;
use crate::monitor::MonitorRefusal;
use crate::terminal::TimeoutKill;
use crate::wording::{
    AnyShellWording, CommandKilled, FoldedLines, JsonAsTsv, MonitorStarted, OmittedLines,
    OutputSaved, RepeatedLines, ReservedForMain, ShellWording, StoredBinaryPart,
};

#[derive(Template)]
#[template(path = "test_wording/unchanged_task_read.txt")]
struct UnchangedTaskReadText<'a> {
    task_id: &'a str,
}

#[derive(Template)]
#[template(path = "test_wording/command_blocked.txt")]
struct Blocked<'a> {
    reason: &'a str,
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
#[template(path = "test_wording/reserved_for_main.txt", escape = "none")]
struct Reserved<'a> {
    name: &'a str,
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
    fn unchanged_task_read(&self, read: &UnchangedTaskRead) -> String {
        rendered(&UnchangedTaskReadText {
            task_id: &read.task_id,
        })
    }

    fn command_blocked(&self, blocked: &CommandBlocked) -> String {
        rendered(&Blocked {
            reason: &blocked.reason,
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

    fn reserved_for_main(&self, reserved: &ReservedForMain<'_>) -> String {
        rendered(&Reserved {
            name: reserved.command,
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
