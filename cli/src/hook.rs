//! The CLI's decisions at the agent's control points, and its terminal
//! rendering of a run.

use std::io::{self, Write};
use std::time::Duration;

use aither_agent::{
    AgentEvent, Hooks, OutputKind, RunOutcome, TimedEvent, ToolDecision, ToolInvocation,
};
use aither_core::llm::tool::{Approval, ApprovalKind, ApprovalRequest, Approver, ApproverFailure};
use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

/// The tool whose every call the user approves first.
pub const COMMAND_TOOL: &str = "command";

/// Asks the user before the agent runs a command.
#[derive(Debug, Clone, Copy, Default)]
pub struct CliHooks;

impl Hooks for CliHooks {
    fn before_tool(
        &self,
        call: &ToolInvocation<'_>,
    ) -> impl std::future::Future<Output = ToolDecision> + Send {
        std::future::ready(if call.name == COMMAND_TOOL {
            ToolDecision::Ask(
                ApprovalRequest::new(ApprovalKind::new("command"), call.arguments.to_string())
                    .with_reason("the agent wants to run a command"),
            )
        } else {
            ToolDecision::Allow
        })
    }
}

/// Answers approval requests by asking on the terminal.
#[derive(Debug, Clone, Copy, Default)]
pub struct TerminalApprover;

impl Approver for TerminalApprover {
    async fn approve(&self, request: ApprovalRequest) -> Result<Approval, ApproverFailure> {
        eprintln!(
            "\n\x1b[33m⚠ Approval needed ({}):\x1b[0m {}",
            request.kind().as_str(),
            request.subject()
        );
        if let Some(reason) = request.reason() {
            eprintln!("\x1b[2m{reason}\x1b[22m");
        }
        eprint!("\x1b[33mAllow? [y/N]: \x1b[0m");
        io::stderr().flush().ok();

        let approved = tokio::task::spawn_blocking(read_yes_no)
            .await
            .map_err(ApproverFailure::new)?
            .map_err(|error| ApproverFailure::new(std::io::Error::other(error.to_string())))?;
        eprintln!();
        Ok(if approved {
            Approval::Approved
        } else {
            Approval::Denied {
                reason: "the user declined".to_string(),
            }
        })
    }
}

/// Reads a y/n answer from the terminal.
fn read_yes_no() -> Result<bool> {
    enable_raw_mode()?;
    let _guard = RawModeGuard;

    loop {
        if event::poll(Duration::from_secs(30))? {
            if let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                match key.code {
                    KeyCode::Char('y' | 'Y') => {
                        eprint!("y");
                        return Ok(true);
                    }
                    KeyCode::Char('n' | 'N') | KeyCode::Enter | KeyCode::Esc => {
                        eprint!("n");
                        return Ok(false);
                    }
                    _ => {}
                }
            }
        } else {
            eprint!("n (timeout)");
            return Ok(false);
        }
    }
}

/// Restores the terminal's cooked mode when dropped.
#[derive(Debug)]
pub struct RawModeGuard;

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

/// Prints one event of the main agent as a human-friendly line.
pub fn print_event(timed: &TimedEvent) {
    if !timed.agent.is_main() {
        return;
    }
    match &timed.event {
        AgentEvent::BlockDelta {
            kind: OutputKind::Text,
            text,
            ..
        } => {
            print!("{text}");
            io::stdout().flush().ok();
        }
        AgentEvent::ToolCallStarted {
            name, arguments, ..
        } => {
            println!("\n\x1b[36m[tool]\x1b[0m {name} \x1b[90m{arguments}\x1b[0m");
        }
        AgentEvent::ToolCallEnded { result, .. } => {
            if let Some(error) = result.error_message() {
                println!("\x1b[31m✗ {error}\x1b[0m");
            } else {
                let text = result.render_for_model();
                let shown = truncate_output(&text, 500);
                for line in shown.lines().take(10) {
                    println!("  \x1b[90m{line}\x1b[0m");
                }
            }
        }
        AgentEvent::RunEnded { outcome, .. } => match outcome {
            RunOutcome::Completed { .. } => println!(),
            other => println!("\n\x1b[90m[run ended: {other:?}]\x1b[0m"),
        },
        _ => {}
    }
}

/// Truncates output to a maximum character count (UTF-8 safe).
fn truncate_output(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => format!("{}[...]", &s[..byte_idx]),
        None => s.to_string(),
    }
}
