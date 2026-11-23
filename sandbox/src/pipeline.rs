//! Per-stage pipeline tapping, dual-stream ANSI SGR handling, and stage read
//! types.
//!
//! When an embedder configures a [`PlanPipelines`] policy on [`crate::ShellBuilder`],
//! multi-stage shell pipelines (`cmd_0 | cmd_1 | ...`) that the policy selects
//! are tapped stage by stage without changing pipeline stage count, exit codes,
//! `$PIPESTATUS`, `pipefail`, or reverse `SIGPIPE` delivery.

use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use askama::Template;
use brush_parser::ast::{
    AndOr, AndOrList, Command, CommandPrefixOrSuffixItem, CompoundList, IoRedirect, Pipeline,
    SeparatorOperator, SimpleCommand,
};
use brush_parser::word::{WordPiece, WordPieceWithSource};
use brush_parser::{Parser, ParserOptions, SourceInfo};
use serde::Serialize;

use crate::job_registry::JobStatus;
use crate::terminal::{TerminalError, shell_escape};
use crate::{LaunchId, ScriptCheck};

/// Embedded Python sidecar driving per-stage FIFOs and multiplexed frames.
const PIPELINE_TAP_PY: &str = include_str!("../templates/pipeline_tap.py");

/// Frame header length: 8-byte magic + 1-byte kind + 1-byte step + 2-byte big-endian length.
const FRAME_HEADER_LEN: usize = 12;

/// Parsed view of one stage inside a candidate multi-stage shell pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageSyntax {
    /// Zero-based index of this stage within its pipeline.
    pub step: usize,
    /// Shell source of this stage as parsed.
    pub source: String,
    /// Literal command name when this stage is a simple command with a static
    /// program word (`"cargo"`, `"rg"`, `"tail"`, ...).
    pub program: Option<String>,
    /// Arguments after the program word: `Some(literal)` when static once
    /// unquoted, `None` when computed or glob-expanded at runtime.
    pub literal_args: Vec<Option<String>>,
    /// Variable names assigned in the command prefix (`FOO=1 cmd`).
    pub assignments: Vec<String>,
    /// Literal text fed via a here-document or here-string on stdin, if any.
    pub stdin_literal: Option<String>,
    /// Whether stdin comes from a here-document.
    pub has_heredoc: bool,
}

/// Tool-scoped ANSI color settings injected onto a pipeline stage when the
/// caller's script specified neither color nor plain output.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ColorInjection {
    /// Stage-local environment variables exported only inside this stage's
    /// subshell (`CARGO_TERM_COLOR=always`, ...).
    pub env: Vec<(String, String)>,
    /// Extra flags inserted immediately after the stage's program word
    /// (`--color=always`, `-c color.ui=always`, ...).
    pub flags: Vec<String>,
}

impl ColorInjection {
    /// Whether this injection adds any environment variable or CLI flag.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.env.is_empty() && self.flags.is_empty()
    }
}

/// Embedder directive for one stage of a tapped pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageDirective {
    /// Human-readable command label for this stage.
    pub command: String,
    /// Optional harness-injected ANSI color settings for this stage.
    pub injected_color: Option<ColorInjection>,
    /// Whether this stage slices lines without inspecting line text (`tail`,
    /// line-mode `head`), allowing upstream harness-injected ANSI color to
    /// pass through to its stdout for UI rendering.
    pub line_slicer: bool,
}

/// Embedder policy choosing which multi-stage shell pipelines are tapped and
/// how each stage handles non-TTY color.
pub trait PlanPipelines: Send + Sync + 'static {
    /// Returns one [`StageDirective`] per stage in `stages` when the pipeline
    /// should be tapped, or `None` to run the pipeline unwrapped.
    fn plan_pipeline(
        &self,
        script: &ScriptCheck<'_>,
        stages: &[StageSyntax],
    ) -> Option<Vec<StageDirective>>;
}

/// Shared type-erased [`PlanPipelines`] carried by a [`crate::Shell`] and its
/// derived subagent shells.
#[derive(Clone)]
pub struct AnyPlanPipelines(Arc<dyn PlanPipelines>);

impl AnyPlanPipelines {
    /// Wraps `planner`.
    #[must_use]
    pub fn new(planner: impl PlanPipelines) -> Self {
        Self(Arc::new(planner))
    }

    /// Evaluates `planner` on `stages`.
    #[must_use]
    pub fn plan_pipeline(
        &self,
        script: &ScriptCheck<'_>,
        stages: &[StageSyntax],
    ) -> Option<Vec<StageDirective>> {
        self.0.plan_pipeline(script, stages)
    }
}

impl fmt::Debug for AnyPlanPipelines {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyPlanPipelines")
    }
}

/// Which output stream(s) of a recorded pipeline stage to read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StageStreamSelector {
    /// Arrival-ordered interleaved `stdout` and `stderr` of the stage.
    #[default]
    Combined,
    /// Only bytes the stage wrote into its pipeline `stdout`.
    Stdout,
    /// Only bytes the stage wrote to its unredirected `stderr`.
    Stderr,
}

/// Execution state of one pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageStatus {
    /// Stage has not reported start yet.
    Pending,
    /// Stage is currently running.
    Running,
    /// Stage exited with `code`.
    Exited {
        /// Process exit status of this stage.
        code: i32,
    },
}

/// Metadata for one `0`-indexed stage of a pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStageInfo {
    /// Zero-based stage index.
    pub step: usize,
    /// Stage command text.
    pub command: String,
    /// Current lifecycle status of the stage.
    pub status: StageStatus,
    /// Measured own active duration once the stage has exited.
    pub own_duration: Option<Duration>,
    /// Whether this stage's output is currently readable (live, or retained
    /// because its own active duration met `pipeline_record_after`).
    pub retained: bool,
    /// Native stdout bytes produced by this stage.
    pub stdout_bytes: usize,
    /// Native stderr bytes produced by this stage.
    pub stderr_bytes: usize,
    /// Combined UI byte count currently buffered for this stage.
    pub ui_bytes: usize,
    /// Newline count across this stage's combined output.
    pub total_lines: usize,
}

/// Snapshot of a running or retained pipeline in [`crate::Jobs`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineSnapshot {
    /// Inside-environment OS process ID of the command.
    pub pid: u32,
    /// Internal monotonic launch identifier disambiguating OS PID reuse.
    pub launch: LaunchId,
    /// Overall job status.
    pub job_status: JobStatus,
    /// When the command started.
    pub started_at: Instant,
    /// When the command finished, if no longer running.
    pub finished_at: Option<Instant>,
    /// When retained intermediate stage buffers expire, if finished.
    pub expires_at: Option<Instant>,
    /// Earliest stage still pending or running while the pipeline is active;
    /// `None` once every stage has exited.
    pub active_step: Option<usize>,
    /// Ordered stages of the pipeline.
    pub stages: Vec<PipelineStageInfo>,
}

/// Incremental UI-colored read slice from one pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageUiDelta {
    /// Process ID of the pipeline command.
    pub pid: u32,
    /// Monotonic launch ID of this pipeline instance.
    pub launch: LaunchId,
    /// Stage index read.
    pub step: usize,
    /// Next byte cursor to pass on a subsequent incremental read.
    pub cursor: usize,
    /// Total UI bytes buffered for this stage so far.
    pub total_bytes: usize,
    /// Newly read UI bytes (preserving ANSI SGR sequences).
    pub bytes: Vec<u8>,
    /// Current status of the stage.
    pub stage_status: StageStatus,
    /// Current status of the overall job.
    pub job_status: JobStatus,
}

/// Errors returned when inspecting or reading a pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PipelineReadError {
    /// No running or retained job matches `pid`.
    #[error("unknown pid: {pid}")]
    UnknownJob {
        /// Process ID named.
        pid: u32,
    },
    /// The job exists but was not a multi-stage tapped pipeline.
    #[error("process {pid} has no recorded pipeline stages")]
    NoPipeline {
        /// Process ID named.
        pid: u32,
    },
    /// The pipeline's retention window elapsed and intermediate stage outputs
    /// were deleted.
    #[error("pipeline stage outputs for process {pid} expired and were cleared")]
    Expired {
        /// Process ID named.
        pid: u32,
    },
    /// Requested `step` is out of range or exited below the retention threshold.
    #[error("pipeline step {step} is not retained for process {pid}")]
    InvalidStep {
        /// Process ID named.
        pid: u32,
        /// Requested `0`-based stage index.
        step: usize,
        /// Retained `0`-based stage indices available on `pid`.
        kept_steps: Vec<usize>,
    },
    /// Reading spilled stage output from disk failed.
    #[error("failed to read stored stage output: {0}")]
    Io(String),
}

#[derive(Debug, Clone, Serialize)]
struct SidecarPipelineSpec {
    stages: Vec<SidecarStageSpec>,
}

#[derive(Debug, Clone, Serialize)]
struct SidecarStageSpec {
    injected: bool,
    line_slicer: bool,
}

/// Prepared instrumentation for a shell script containing one or more tapped
/// pipelines.
#[derive(Debug, Clone)]
pub struct InstrumentedScript {
    magic: [u8; 8],
    rewritten_script: String,
    spec_json: String,
    stage_commands: Vec<String>,
}

#[derive(Template)]
#[template(path = "pipeline_runner.sh", escape = "none")]
struct PipelineRunnerTemplate<'a> {
    tap_py: &'a str,
    magic: &'a str,
    spec: &'a str,
    rewritten_script: &'a str,
    report: Option<&'a str>,
}

impl InstrumentedScript {
    /// Eight-byte frame magic identifying multiplexed frames on stdout.
    #[must_use]
    pub(crate) const fn magic(&self) -> [u8; 8] {
        self.magic
    }

    /// Ordered stage command labels registered in `JobRegistry`.
    #[must_use]
    pub(crate) fn stage_commands(&self) -> &[String] {
        &self.stage_commands
    }

    /// Renders the bash wrapper executing the instrumented script and optional
    /// working-directory report file.
    pub(crate) fn render(&self, report: Option<&Path>) -> Result<String, TerminalError> {
        let tap_py = shell_escape(PIPELINE_TAP_PY);
        let magic_hex = hex::encode(self.magic);
        let magic = shell_escape(&magic_hex);
        let spec = shell_escape(&self.spec_json);
        let rewritten_script = shell_escape(&self.rewritten_script);
        let report = report.map(|p| shell_escape(&p.to_string_lossy()));
        PipelineRunnerTemplate {
            tap_py: &tap_py,
            magic: &magic,
            spec: &spec,
            rewritten_script: &rewritten_script,
            report: report.as_deref(),
        }
        .render()
        .map_err(|error| {
            TerminalError::Execution(format!("failed to render pipeline runner: {error}"))
        })
    }
}

/// Builds an [`InstrumentedScript`] when `planner` selects at least one
/// top-level pipeline in `check`.
pub fn instrument_script(
    planner: &AnyPlanPipelines,
    check: &ScriptCheck<'_>,
) -> Result<Option<InstrumentedScript>, TerminalError> {
    let Ok(program) = Parser::new(
        check.script.as_bytes(),
        &ParserOptions::default(),
        &SourceInfo::default(),
    )
    .parse_program() else {
        return Ok(None);
    };

    let mut sidecar_pipelines = Vec::new();
    let mut stage_commands = Vec::new();
    let mut rendered_lists = Vec::new();
    let mut any_tapped = false;

    for list in &program.complete_commands {
        let rendered = rewrite_compound_list(
            list,
            planner,
            check,
            &mut sidecar_pipelines,
            &mut stage_commands,
            &mut any_tapped,
        );
        if !rendered.is_empty() {
            rendered_lists.push(rendered);
        }
    }

    if !any_tapped {
        return Ok(None);
    }

    let mut magic = [0_u8; 8];
    getrandom::fill(&mut magic)
        .map_err(|error| TerminalError::Execution(format!("cannot generate tap token: {error}")))?;
    let spec_json = serde_json::to_string(&sidecar_pipelines).map_err(|error| {
        TerminalError::Execution(format!("cannot serialize pipeline spec: {error}"))
    })?;

    Ok(Some(InstrumentedScript {
        magic,
        rewritten_script: rendered_lists.join("\n"),
        spec_json,
        stage_commands,
    }))
}

fn rewrite_compound_list(
    list: &CompoundList,
    planner: &AnyPlanPipelines,
    check: &ScriptCheck<'_>,
    sidecar_pipelines: &mut Vec<SidecarPipelineSpec>,
    stage_commands: &mut Vec<String>,
    any_tapped: &mut bool,
) -> String {
    let mut out = String::new();
    for (idx, item) in list.0.iter().enumerate() {
        if idx > 0 {
            out.push(' ');
        }
        if matches!(item.1, SeparatorOperator::Async) {
            out.push_str(&item.0.to_string());
            out.push_str(" &");
        } else {
            out.push_str(&rewrite_and_or(
                &item.0,
                planner,
                check,
                sidecar_pipelines,
                stage_commands,
                any_tapped,
            ));
            out.push(';');
        }
    }
    out
}

fn rewrite_and_or(
    list: &AndOrList,
    planner: &AnyPlanPipelines,
    check: &ScriptCheck<'_>,
    sidecar_pipelines: &mut Vec<SidecarPipelineSpec>,
    stage_commands: &mut Vec<String>,
    any_tapped: &mut bool,
) -> String {
    let mut out = rewrite_pipeline(
        &list.first,
        planner,
        check,
        sidecar_pipelines,
        stage_commands,
        any_tapped,
    );
    for next in &list.additional {
        match next {
            AndOr::And(pipeline) => {
                out.push_str(" && ");
                out.push_str(&rewrite_pipeline(
                    pipeline,
                    planner,
                    check,
                    sidecar_pipelines,
                    stage_commands,
                    any_tapped,
                ));
            }
            AndOr::Or(pipeline) => {
                out.push_str(" || ");
                out.push_str(&rewrite_pipeline(
                    pipeline,
                    planner,
                    check,
                    sidecar_pipelines,
                    stage_commands,
                    any_tapped,
                ));
            }
        }
    }
    out
}

fn rewrite_pipeline(
    pipeline: &Pipeline,
    planner: &AnyPlanPipelines,
    check: &ScriptCheck<'_>,
    sidecar_pipelines: &mut Vec<SidecarPipelineSpec>,
    stage_commands: &mut Vec<String>,
    any_tapped: &mut bool,
) -> String {
    if pipeline.bang
        || pipeline.timed.is_some()
        || pipeline.seq.len() < 2
        || pipeline.seq.iter().any(command_has_heredoc)
    {
        return pipeline.to_string();
    }

    let syntax: Vec<StageSyntax> = pipeline
        .seq
        .iter()
        .enumerate()
        .map(|(step, cmd)| stage_syntax(step, cmd))
        .collect();

    let Some(directives) = planner.plan_pipeline(check, &syntax) else {
        return pipeline.to_string();
    };
    if directives.len() != pipeline.seq.len() || sidecar_pipelines.len() >= 32 {
        return pipeline.to_string();
    }

    let p_idx = sidecar_pipelines.len();
    let mut stage_specs = Vec::with_capacity(directives.len());
    let mut wrapped_stages = Vec::with_capacity(directives.len());

    for (k_idx, (cmd, directive)) in pipeline.seq.iter().zip(directives).enumerate() {
        if stage_commands.len() >= usize::from(u8::MAX) {
            return pipeline.to_string();
        }
        let injection = directive.injected_color.filter(|inj| !inj.is_empty());
        let env_src = injection
            .as_ref()
            .map_or_else(String::new, |inj| render_stage_env(&inj.env));
        let cmd_src = match (&injection, cmd) {
            (Some(inj), Command::Simple(simple)) if !inj.flags.is_empty() => {
                render_simple_with_flags(simple, &inj.flags)
            }
            _ => cmd.to_string(),
        };
        stage_specs.push(SidecarStageSpec {
            injected: injection.is_some(),
            line_slicer: directive.line_slicer,
        });
        stage_commands.push(directive.command);
        wrapped_stages.push(format!(
            "__aither_stage {p_idx} {k_idx} {} {}",
            shell_escape(&env_src),
            shell_escape(&cmd_src),
        ));
    }

    sidecar_pipelines.push(SidecarPipelineSpec {
        stages: stage_specs,
    });
    *any_tapped = true;
    wrapped_stages.join(" | ")
}

fn render_stage_env(env: &[(String, String)]) -> String {
    let mut out = String::new();
    for (key, value) in env {
        if key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
            && key.bytes().next().is_some_and(|b| !b.is_ascii_digit())
        {
            out.push_str("export ");
            out.push_str(key);
            out.push('=');
            out.push_str(&shell_escape(value));
            out.push_str("; ");
        }
    }
    out
}

fn render_simple_with_flags(simple: &SimpleCommand, flags: &[String]) -> String {
    let mut parts = Vec::new();
    if let Some(prefix) = &simple.prefix {
        for item in &prefix.0 {
            parts.push(item.to_string());
        }
    }
    if let Some(program) = &simple.word_or_name {
        parts.push(program.to_string());
    }
    for flag in flags {
        parts.push(shell_escape(flag));
    }
    if let Some(suffix) = &simple.suffix {
        for item in &suffix.0 {
            parts.push(item.to_string());
        }
    }
    parts.join(" ")
}

fn command_has_heredoc(command: &Command) -> bool {
    match command {
        Command::Simple(simple) => simple
            .prefix
            .iter()
            .flat_map(|p| &p.0)
            .chain(simple.suffix.iter().flat_map(|s| &s.0))
            .any(|item| {
                matches!(
                    item,
                    CommandPrefixOrSuffixItem::IoRedirect(IoRedirect::HereDocument(..))
                )
            }),
        Command::Compound(_, redirects) => redirects
            .iter()
            .flat_map(|r| &r.0)
            .any(|red| matches!(red, IoRedirect::HereDocument(..))),
        Command::Function(func) => func
            .body
            .1
            .iter()
            .flat_map(|r| &r.0)
            .any(|red| matches!(red, IoRedirect::HereDocument(..))),
        Command::ExtendedTest(_) => false,
    }
}

fn stage_syntax(step: usize, command: &Command) -> StageSyntax {
    let source = command.to_string();
    let Command::Simple(simple) = command else {
        return StageSyntax {
            step,
            source,
            program: None,
            literal_args: Vec::new(),
            assignments: Vec::new(),
            stdin_literal: None,
            has_heredoc: command_has_heredoc(command),
        };
    };

    let program = simple
        .word_or_name
        .as_ref()
        .and_then(|word| unquote_literal(&word.value));
    let mut literal_args = Vec::new();
    let mut assignments = Vec::new();
    let mut stdin_literal = None;
    let mut has_heredoc = false;

    for item in simple
        .prefix
        .iter()
        .flat_map(|p| &p.0)
        .chain(simple.suffix.iter().flat_map(|s| &s.0))
    {
        match item {
            CommandPrefixOrSuffixItem::Word(word) => {
                literal_args.push(unquote_literal(&word.value));
            }
            CommandPrefixOrSuffixItem::AssignmentWord(assignment, _) => {
                assignments.push(assignment.name.to_string());
            }
            CommandPrefixOrSuffixItem::IoRedirect(IoRedirect::HereDocument(_, doc)) => {
                has_heredoc = true;
                if !doc.requires_expansion || !doc.doc.value.contains(['$', '`', '\\']) {
                    stdin_literal = Some(doc.doc.value.clone());
                }
            }
            CommandPrefixOrSuffixItem::IoRedirect(IoRedirect::HereString(_, word)) => {
                stdin_literal = unquote_literal(&word.value);
            }
            CommandPrefixOrSuffixItem::IoRedirect(_)
            | CommandPrefixOrSuffixItem::ProcessSubstitution(..) => {}
        }
    }

    StageSyntax {
        step,
        source,
        program,
        literal_args,
        assignments,
        stdin_literal,
        has_heredoc,
    }
}

fn unquote_literal(word: &str) -> Option<String> {
    let pieces = brush_parser::word::parse(word, &ParserOptions::default()).ok()?;
    let mut text = String::new();
    unquote_pieces(&pieces, &mut text, false)?;
    Some(text)
}

fn unquote_pieces(pieces: &[WordPieceWithSource], out: &mut String, quoted: bool) -> Option<()> {
    for WordPieceWithSource { piece, .. } in pieces {
        match piece {
            WordPiece::Text(part) => {
                if !quoted && part.contains(['*', '?', '[', '{']) {
                    return None;
                }
                out.push_str(part);
            }
            WordPiece::SingleQuotedText(part) => out.push_str(part),
            WordPiece::EscapeSequence(part) => out.push_str(part.strip_prefix('\\')?),
            WordPiece::DoubleQuotedSequence(inner) => unquote_pieces(inner, out, true)?,
            WordPiece::GettextDoubleQuotedSequence(_)
            | WordPiece::AnsiCQuotedText(_)
            | WordPiece::TildePrefix(_)
            | WordPiece::ParameterExpansion(_)
            | WordPiece::CommandSubstitution(_)
            | WordPiece::BackquotedCommandSubstitution(_)
            | WordPiece::ArithmeticExpression(_) => return None,
        }
    }
    Some(())
}

/// Strips CSI SGR (`\x1b[...m`) color sequences from `bytes`, leaving all
/// other bytes and non-SGR escape sequences intact.
#[must_use]
pub fn strip_sgr_sequences(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut idx = 0;
    while idx < bytes.len() {
        if bytes[idx] == 0x1b && bytes.get(idx + 1) == Some(&b'[') {
            let mut end = idx + 2;
            while end < bytes.len() && matches!(bytes[end], b'0'..=b'9' | b';' | b':') {
                end += 1;
            }
            if bytes.get(end) == Some(&b'm') {
                idx = end + 1;
                continue;
            }
        }
        out.push(bytes[idx]);
        idx += 1;
    }
    out
}

/// Decoded event from the per-launch stdout multiplexer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MuxEvent {
    OuterStdout(Vec<u8>),
    OuterStderr(Vec<u8>),
    StageStarted {
        step: usize,
    },
    StageChunk {
        step: usize,
        stream: crate::job_registry::TerminalStream,
        injected_color: bool,
        bytes: Vec<u8>,
    },
    StageExited {
        step: usize,
        exit_code: i32,
        own_duration: Duration,
    },
}

/// Incremental parser for multiplexed frames emitted by `pipeline_tap.py`.
pub struct MuxDecoder {
    magic: [u8; 8],
    buf: Vec<u8>,
}

impl MuxDecoder {
    #[must_use]
    pub(crate) const fn new(magic: [u8; 8]) -> Self {
        Self {
            magic,
            buf: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, chunk: &[u8]) -> Vec<MuxEvent> {
        self.buf.extend_from_slice(chunk);
        let mut events = Vec::new();
        loop {
            if self.buf.len() < self.magic.len() {
                break;
            }
            let Some(pos) = self
                .buf
                .windows(self.magic.len())
                .position(|window| window == self.magic)
            else {
                let keep = self.magic.len() - 1;
                let emit_len = self.buf.len() - keep;
                let raw: Vec<u8> = self.buf.drain(..emit_len).collect();
                if !raw.is_empty() {
                    events.push(MuxEvent::OuterStdout(raw));
                }
                break;
            };
            if pos > 0 {
                let raw: Vec<u8> = self.buf.drain(..pos).collect();
                events.push(MuxEvent::OuterStdout(raw));
            }
            if self.buf.len() < FRAME_HEADER_LEN {
                break;
            }
            let kind = self.buf[8];
            let step = usize::from(self.buf[9]);
            let payload_len = usize::from(u16::from_be_bytes([self.buf[10], self.buf[11]]));
            let total_len = FRAME_HEADER_LEN + payload_len;
            if self.buf.len() < total_len {
                break;
            }
            let frame: Vec<u8> = self.buf.drain(..total_len).collect();
            let payload = &frame[FRAME_HEADER_LEN..];
            match kind {
                1 => events.push(MuxEvent::OuterStdout(payload.to_vec())),
                2 => events.push(MuxEvent::OuterStderr(payload.to_vec())),
                4 => events.push(MuxEvent::StageStarted { step }),
                5 | 6 => events.push(MuxEvent::StageChunk {
                    step,
                    stream: crate::job_registry::TerminalStream::Stdout,
                    injected_color: kind == 6,
                    bytes: payload.to_vec(),
                }),
                7 | 8 => events.push(MuxEvent::StageChunk {
                    step,
                    stream: crate::job_registry::TerminalStream::Stderr,
                    injected_color: kind == 8,
                    bytes: payload.to_vec(),
                }),
                9 if payload.len() == 12 => {
                    let exit_code =
                        i32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
                    let nanos = u64::from_be_bytes([
                        payload[4],
                        payload[5],
                        payload[6],
                        payload[7],
                        payload[8],
                        payload[9],
                        payload[10],
                        payload[11],
                    ]);
                    events.push(MuxEvent::StageExited {
                        step,
                        exit_code,
                        own_duration: Duration::from_nanos(nanos),
                    });
                }
                _ => {}
            }
        }
        events
    }

    pub(crate) fn finish(&mut self) -> Vec<MuxEvent> {
        if self.buf.is_empty() {
            return Vec::new();
        }
        vec![MuxEvent::OuterStdout(std::mem::take(&mut self.buf))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_sgr_removes_only_sgr_sequences() {
        let raw = b"\x1b[1m\x1b[38;5;9merror[E0308]\x1b[0m: mismatched \x1b]0;title\x07types\n";
        assert_eq!(
            strip_sgr_sequences(raw),
            b"error[E0308]: mismatched \x1b]0;title\x07types\n"
        );
    }

    #[test]
    fn mux_decoder_parses_split_frames_and_preserves_prefix() {
        let magic = *b"AITHPIPE";
        let mut decoder = MuxDecoder::new(magic);
        let mut stream = Vec::new();
        stream.extend_from_slice(b"pre");
        stream.extend_from_slice(&magic);
        stream.extend_from_slice(&[4, 0, 0, 0]);
        stream.extend_from_slice(&magic);
        stream.extend_from_slice(&[6, 0, 0, 5]);
        stream.extend_from_slice(b"hello");
        stream.extend_from_slice(&magic);
        stream.extend_from_slice(&[9, 0, 0, 12]);
        stream.extend_from_slice(&101_i32.to_be_bytes());
        stream.extend_from_slice(&3_500_000_000_u64.to_be_bytes());

        let mut out = Vec::new();
        for chunk in stream.chunks(7) {
            out.extend(decoder.push(chunk));
        }
        out.extend(decoder.finish());
        assert_eq!(
            out,
            vec![
                MuxEvent::OuterStdout(b"pre".to_vec()),
                MuxEvent::StageStarted { step: 0 },
                MuxEvent::StageChunk {
                    step: 0,
                    stream: crate::job_registry::TerminalStream::Stdout,
                    injected_color: true,
                    bytes: b"hello".to_vec(),
                },
                MuxEvent::StageExited {
                    step: 0,
                    exit_code: 101,
                    own_duration: Duration::from_millis(3500),
                },
            ]
        );
    }

    #[cfg(unix)]
    mod shell_tests {
        use std::path::Path;
        use std::sync::Arc;
        use std::time::Duration;

        use aither_core::llm::{CommandOutcome, CommandStream, Tool, ToolContext};
        use executor_core::AnyExecutor;

        use super::*;
        use crate::{
            Backend, CallRefusal, HostLaunch, Interpreter, JobReadArgs, JobReadTool, Refused,
            Sandbox, SandboxMode, Shell, ShellArgs, ShellLimits, UnchangedRead, test_budgets,
            test_wording,
        };

        struct TestPipelinePlanner;

        impl PlanPipelines for TestPipelinePlanner {
            fn plan_pipeline(
                &self,
                _script: &ScriptCheck<'_>,
                stages: &[StageSyntax],
            ) -> Option<Vec<StageDirective>> {
                if stages.first().and_then(|s| s.program.as_deref()) == Some("skip_tap") {
                    return None;
                }
                Some(
                    stages
                        .iter()
                        .map(|stage| {
                            let prog = stage.program.as_deref().unwrap_or("");
                            let has_explicit_color = stage
                                .literal_args
                                .iter()
                                .flatten()
                                .any(|a| a.starts_with("--color="))
                                || stage.assignments.iter().any(|k| k == "FORCE_STAGE_COLOR");
                            let injected_color = if prog == "python3"
                                && !has_explicit_color
                                && stage.source.contains("AUTO_COLOR_TOOL")
                            {
                                Some(ColorInjection {
                                    env: vec![("AUTO_STAGE_COLOR".into(), "1".into())],
                                    flags: Vec::new(),
                                })
                            } else {
                                None
                            };
                            let line_slicer = matches!(prog, "tail" | "head");
                            StageDirective {
                                command: stage.source.clone(),
                                injected_color,
                                line_slicer,
                            }
                        })
                        .collect(),
                )
            }
        }

        async fn build_shell(dir: &Path, limits: ShellLimits) -> Shell<Arc<AnyExecutor>> {
            let heel = crate::test_heel::binary().await.expect("test heel");
            let sandbox = Sandbox::new(
                Backend::Host { heel },
                executor_core::tokio::TokioGlobal,
                dir,
            );
            Shell::builder_on_host(sandbox, HostLaunch::new("/bin/bash"))
                .interpreters([Interpreter::Shell])
                .limits(limits)
                .output(test_budgets::output_policy())
                .wording(test_wording::TestWording)
                .tools(test_wording::job_tools())
                .plan_pipelines(TestPipelinePlanner)
                .build()
                .await
                .expect("shell builds")
        }

        fn shell_args(script: &str, raw: bool, background: bool) -> ShellArgs {
            serde_json::from_value(serde_json::json!({
                "description": "test pipeline",
                "script": script,
                "raw": raw,
                "background": background,
                "timeout": 15
            }))
            .expect("valid ShellArgs")
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn fast_foreground_pipeline_discards_stages_and_omits_pid() {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().canonicalize().unwrap();
            let mut limits = test_budgets::limits();
            limits.pipeline_record_after = crate::LiveDuration::new(Duration::from_secs(2));
            let shell = build_shell(&dir, limits).await;

            let res = shell
                .call(
                    shell_args("printf 'a\\nb\\nc\\n' | tail -n 1", false, false),
                    ToolContext::new(),
                )
                .await
                .expect("pipeline succeeds");
            let cmd = res.command().expect("command result");
            assert_eq!(cmd.retained_pid, None);
            assert_eq!(res.render_for_model(), "c\n");
            assert_eq!(shell.jobs().pipelines().await, Vec::new());
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn per_stage_own_duration_retains_only_expensive_stages_and_excludes_upstream_wait() {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().canonicalize().unwrap();
            let mut limits = test_budgets::limits();
            limits.pipeline_record_after = crate::LiveDuration::new(Duration::from_millis(220));
            limits.pipeline_retention = crate::LiveDuration::new(Duration::from_secs(30));
            let shell = build_shell(&dir, limits).await;

            // Stage 0 sleeps 320ms (kept), Stage 1 & 2 wait on upstream then filter in <20ms (dropped),
            // Stage 3 sleeps 300ms after upstream EOF (kept).
            let script = concat!(
                "python3 -c 'import time; time.sleep(0.32); print(\"alpha\\nbeta\\ngamma\")' ",
                "| grep -E 'alpha|beta' ",
                "| tr 'a-z' 'A-Z' ",
                "| python3 -c 'import sys, time; data = sys.stdin.read(); time.sleep(0.30); sys.stdout.write(data.splitlines()[-1] + \"\\n\")'"
            );
            let res = shell
                .call(shell_args(script, false, false), ToolContext::new())
                .await
                .expect("pipeline runs");
            let cmd = res.command().expect("command result");
            let pid = cmd
                .retained_pid
                .expect("expensive upstream stage retains pid");
            assert!(
                res.render_for_model().contains(&format!("PID: {pid}.")),
                "{}",
                res.render_for_model()
            );

            let snap = shell.jobs().pipeline(pid).await.expect("retained snapshot");
            assert_eq!(snap.stages.len(), 4);
            assert!(snap.stages[0].retained, "stage 0 is retained: {snap:?}");
            assert!(
                !snap.stages[1].retained,
                "fast stage 1 waiting on upstream must be dropped: {snap:?}"
            );
            assert!(
                !snap.stages[2].retained,
                "fast stage 2 waiting on upstream must be dropped: {snap:?}"
            );
            assert!(
                snap.stages[3].retained,
                "slow final stage 3 is retained: {snap:?}"
            );

            let s0 = shell
                .jobs()
                .read_pipeline_stage(pid, 0, StageStreamSelector::Combined, false)
                .await
                .expect("read stage 0");
            assert_eq!(String::from_utf8_lossy(&s0), "alpha\nbeta\ngamma\n");

            let err = shell
                .jobs()
                .read_pipeline_stage(pid, 1, StageStreamSelector::Combined, false)
                .await
                .expect_err("dropped stage 1 is rejected");
            assert_eq!(
                err,
                PipelineReadError::InvalidStep {
                    pid,
                    step: 1,
                    kept_steps: vec![0, 3],
                }
            );

            let s3 = shell
                .jobs()
                .read_pipeline_stage(pid, 3, StageStreamSelector::Combined, false)
                .await
                .expect("read aliased final stage 3");
            assert_eq!(String::from_utf8_lossy(&s3), "BETA\n");
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn pipestatus_pipefail_and_reverse_sigpipe_stay_intact() {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().canonicalize().unwrap();
            let shell = build_shell(&dir, test_budgets::limits()).await;

            // Reverse SIGPIPE: `yes` terminates immediately once `head -n 3` closes stdin.
            let sigpipe_res = shell
                .call(
                    shell_args("yes hi | head -n 3", false, false),
                    ToolContext::new(),
                )
                .await
                .expect("yes | head finishes promptly");
            assert_eq!(sigpipe_res.render_for_model(), "hi\nhi\nhi\n");

            // PIPESTATUS and pipefail preserve exact per-stage exit codes.
            let status_script = "set -o pipefail; (echo ok; exit 7) | cat; ps=(\"${PIPESTATUS[@]}\"); rc=$(( ps[0] != 0 ? ps[0] : ps[1] )); echo \"ps=${ps[0]},${ps[1]}\"; (echo ok2; exit 7) | cat";
            let res = shell
                .call(shell_args(status_script, false, false), ToolContext::new())
                .await
                .expect("script runs");
            let cmd = res.command().expect("command result");
            assert_eq!(cmd.outcome, CommandOutcome::Exited { code: 7 });
            let CommandStream::Inline { text } = &cmd.stdout else {
                panic!("inline stdout");
            };
            assert!(text.contains("ps=7,0"), "{text}");
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn dual_stream_strips_injected_sgr_for_filters_and_raw_but_preserves_explicit_ansi() {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().canonicalize().unwrap();
            let mut limits = test_budgets::limits();
            limits.pipeline_record_after = crate::LiveDuration::new(Duration::from_millis(150));
            let shell = build_shell(&dir, limits).await;

            // 1) Injected color + non-slicer `grep '^error'`: UI has ANSI, grep matches, raw model output has NO ANSI.
            let injected_grep = concat!(
                "python3 -c 'import os, time; AUTO_COLOR_TOOL=1; time.sleep(0.22); ",
                "c = \"\\x1b[31m\" if os.environ.get(\"AUTO_STAGE_COLOR\") == \"1\" else \"\"; ",
                "r = \"\\x1b[0m\" if c else \"\"; ",
                "print(f\"{c}error[E01]{r}: boom\\nok line\")' ",
                "| grep '^error'"
            );
            let res = shell
                .call(shell_args(injected_grep, true, false), ToolContext::new())
                .await
                .expect("injected grep runs");
            let cmd = res.command().unwrap();
            let pid = cmd.retained_pid.expect("stage 0 retained");
            let CommandStream::Inline { text } = &cmd.stdout else {
                panic!("inline");
            };
            assert_eq!(
                text, "error[E01]: boom\n",
                "raw: true must not leak harness-injected ANSI"
            );
            let ui_delta = shell
                .jobs()
                .read_pipeline_stage_ui(pid, None, 0, 0, 4096)
                .await
                .expect("ui read");
            assert!(
                ui_delta.bytes.windows(5).any(|w| w == b"\x1b[31m"),
                "UI stream keeps injected color: {:?}",
                String::from_utf8_lossy(&ui_delta.bytes)
            );
            let native_raw = shell
                .jobs()
                .read_pipeline_stage(pid, 0, StageStreamSelector::Stdout, true)
                .await
                .expect("native raw read");
            assert_eq!(
                String::from_utf8_lossy(&native_raw),
                "error[E01]: boom\nok line\n"
            );

            // 2) Injected color + line slicer `tail -n 1`: UI of stage 1 ALSO keeps ANSI, model raw stays plain!
            let injected_tail = concat!(
                "python3 -c 'import os, time; AUTO_COLOR_TOOL=1; time.sleep(0.22); ",
                "c = \"\\x1b[33m\" if os.environ.get(\"AUTO_STAGE_COLOR\") == \"1\" else \"\"; ",
                "r = \"\\x1b[0m\" if c else \"\"; ",
                "print(f\"first\\n{c}warning: tail{r}\")' ",
                "| python3 -c 'import sys, time; d = sys.stdin.buffer.read(); time.sleep(0.22); sys.stdout.buffer.write(d)'"
            );
            let _ = injected_tail;
            let slicer_script = concat!(
                "python3 -c 'import os, time; AUTO_COLOR_TOOL=1; time.sleep(0.22); ",
                "c = \"\\x1b[33m\" if os.environ.get(\"AUTO_STAGE_COLOR\") == \"1\" else \"\"; ",
                "r = \"\\x1b[0m\" if c else \"\"; ",
                "print(f\"first\\n{c}warning: tail{r}\")' ",
                "| tail -n 1"
            );
            let res_tail = shell
                .call(shell_args(slicer_script, true, false), ToolContext::new())
                .await
                .expect("slicer runs");
            let CommandStream::Inline { text: tail_text } = &res_tail.command().unwrap().stdout
            else {
                panic!("inline");
            };
            assert_eq!(tail_text, "warning: tail\n");

            // 3) Explicit script-authored ANSI (`printf '\x1b[32m...' | rg '\x1b\['` / `cat`) with `raw: true`: 100% preserved!
            let explicit_ansi = "python3 -c 'import time; time.sleep(0.22); print(\"\\x1b[32mexplicit\\x1b[0m\")' | grep $'\\x1b\\['";
            let res_exp = shell
                .call(shell_args(explicit_ansi, true, false), ToolContext::new())
                .await
                .expect("explicit ansi runs");
            let exp_cmd = res_exp.command().unwrap();
            let exp_pid = exp_cmd.retained_pid.expect("retained");
            let exp_bytes = match &exp_cmd.stdout {
                CommandStream::Inline { text } => text.clone().into_bytes(),
                CommandStream::Stored { file, .. } => {
                    std::fs::read(&file.path).expect("stored raw ansi file")
                }
                other @ CommandStream::Image { .. } => panic!("unexpected stream: {other:?}"),
            };
            assert_eq!(exp_bytes, b"\x1b[32mexplicit\x1b[0m\n");
            let exp_stage_raw = shell
                .jobs()
                .read_pipeline_stage(exp_pid, 0, StageStreamSelector::Stdout, true)
                .await
                .expect("stage 0 raw ansi");
            assert_eq!(exp_stage_raw, b"\x1b[32mexplicit\x1b[0m\n");
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn stderr_stage_read_and_job_read_tool_poll() {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().canonicalize().unwrap();
            let mut limits = test_budgets::limits();
            limits.pipeline_record_after = crate::LiveDuration::new(Duration::from_millis(150));
            limits.pipeline_retention = crate::LiveDuration::new(Duration::from_secs(30));
            let shell = build_shell(&dir, limits).await;
            let read_tool = JobReadTool::new(
                test_wording::job_tools().read,
                shell.jobs(),
                test_wording::TestWording::any(),
                test_budgets::output_policy(),
            );

            // Stage 0 writes diagnostics to unredirected stderr (`cargo clippy | grep xxx` pattern).
            let script = "python3 -c 'import sys, time; sys.stderr.write(\"diag-1\\ndiag-2\\n\"); time.sleep(0.22)' | grep nomatch || true";
            let res = shell
                .call(shell_args(script, false, false), ToolContext::new())
                .await
                .expect("runs");
            let pid = res.command().unwrap().retained_pid.expect("retained");

            // `terminal_read` with `step: Some(0)` reads stage 0's stderr diagnostics.
            let first = read_tool
                .call(
                    JobReadArgs {
                        pid,
                        step: Some(0),
                        cursor: 0,
                        max_bytes: None,
                    },
                    ToolContext::new(),
                )
                .await
                .expect("stage 0 read succeeds");
            assert!(
                first.render_for_model().contains("diag-1\ndiag-2"),
                "{}",
                first.render_for_model()
            );

            // Repeat poll at end cursor returns typed UnchangedRead refusal.
            let unchanged = read_tool
                .call(
                    JobReadArgs {
                        pid,
                        step: Some(0),
                        cursor: 14,
                        max_bytes: None,
                    },
                    ToolContext::new(),
                )
                .await
                .expect_err("unchanged poll refused");
            assert!(
                unchanged.downcast_ref::<Refused<UnchangedRead>>().is_some(),
                "{unchanged:?}"
            );
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn pipeline_retention_expiry_and_pid_reuse_isolation() {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().canonicalize().unwrap();
            let retention = crate::LiveDuration::new(Duration::from_secs(30));
            let mut limits = test_budgets::limits();
            limits.pipeline_record_after = crate::LiveDuration::new(Duration::from_millis(150));
            limits.pipeline_retention = retention.clone();
            let shell = build_shell(&dir, limits).await;
            let read_tool = JobReadTool::new(
                test_wording::job_tools().read,
                shell.jobs(),
                test_wording::TestWording::any(),
                test_budgets::output_policy(),
            );

            let script = "python3 -c 'import sys, time; sys.stderr.write(\"diag-1\\ndiag-2\\n\"); time.sleep(0.22)' | grep nomatch || true";
            let res = shell
                .call(shell_args(script, false, false), ToolContext::new())
                .await
                .expect("runs");
            let pid = res.command().unwrap().retained_pid.expect("retained");
            let launch = shell.jobs().pipeline(pid).await.expect("snapshot").launch;

            // Simulate OS PID reuse on the same registry: PID-based lookup switches to the new job,
            // while launch-pinned UI read still reaches the retained pipeline.
            let reused_lease = crate::job_registry::LaunchLease::fresh();
            shell
                .job_registry()
                .register(
                    pid,
                    &reused_lease,
                    "reused",
                    "echo new",
                    SandboxMode::Sandboxed,
                    None,
                )
                .await;
            assert_eq!(
                shell.jobs().pipeline(pid).await.unwrap_err(),
                PipelineReadError::NoPipeline { pid }
            );
            let pinned_ui = shell
                .jobs()
                .read_pipeline_stage_ui(pid, Some(launch), 0, 0, 1024)
                .await
                .expect("launch-pinned read unaffected by PID reuse");
            assert_eq!(
                String::from_utf8_lossy(&pinned_ui.bytes),
                "diag-1\ndiag-2\n"
            );
            shell.job_registry().complete(&reused_lease, 0, None).await;
            let owed = shell
                .job_registry()
                .owe_completion(reused_lease.id(), Ok(0))
                .await
                .unwrap();
            drop(owed);
            drop(reused_lease);

            // Re-run another retained pipeline and shorten live retention to expire it immediately.
            let res2 = shell
                .call(shell_args(script, false, false), ToolContext::new())
                .await
                .expect("runs");
            let pid2 = res2.command().unwrap().retained_pid.expect("retained");
            retention.set(Duration::from_millis(50));
            async_io::Timer::after(Duration::from_millis(120)).await;

            assert_eq!(
                shell.jobs().pipeline(pid2).await.unwrap_err(),
                PipelineReadError::Expired { pid: pid2 }
            );
            let expired_err = read_tool
                .call(
                    JobReadArgs {
                        pid: pid2,
                        step: Some(0),
                        cursor: 0,
                        max_bytes: None,
                    },
                    ToolContext::new(),
                )
                .await
                .expect_err("expired pid refused");
            let expired_refusal = expired_err
                .downcast_ref::<Refused<CallRefusal>>()
                .expect("typed CallRefusal");
            assert_eq!(
                expired_refusal.reason(),
                &CallRefusal::ExpiredJob { pid: pid2 }
            );
        }
    }
}
