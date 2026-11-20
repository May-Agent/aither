//! The CLI's decisions at the agent's control points, and its terminal
//! rendering of a run.

use std::io::{self, Write};
use std::time::Duration;

use aither_agent::{
    AgentEvent, Hooks, OutputKind, RunOutcome, TimedEvent, ToolDecision, ToolInvocation,
};
use aither_core::llm::tool::{Approval, ApprovalKind, ApprovalRequest, Approver, ApproverFailure};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

/// The command execution tool name.
pub const COMMAND_TOOL: &str = "command";

/// The filesystem tool name.
pub const FILESYSTEM_TOOL: &str = "filesystem";

/// Asks the user before the agent runs a command, mutates the filesystem, or
/// invokes an MCP tool unless `--approve` was passed.
#[derive(Debug, Clone, Copy, Default)]
pub struct CliHooks {
    auto_approve: bool,
}

impl CliHooks {
    /// Creates CLI hooks; when `auto_approve` is `true`, tool calls proceed
    /// without prompting.
    #[must_use]
    pub const fn new(auto_approve: bool) -> Self {
        Self { auto_approve }
    }

    fn decide(self, name: &str, arguments: &serde_json::Value) -> ToolDecision {
        if self.auto_approve {
            return ToolDecision::Allow;
        }
        match name {
            COMMAND_TOOL => ToolDecision::Ask(
                ApprovalRequest::new(ApprovalKind::new("command"), arguments.to_string())
                    .with_reason("the agent wants to run a command"),
            ),
            FILESYSTEM_TOOL => {
                if is_read_only_fs_operation(arguments) {
                    ToolDecision::Allow
                } else {
                    ToolDecision::Ask(
                        ApprovalRequest::new(
                            ApprovalKind::new("filesystem"),
                            arguments.to_string(),
                        )
                        .with_reason("the agent wants to modify the filesystem"),
                    )
                }
            }
            "websearch" | "webfetch" | "compact" => ToolDecision::Allow,
            mcp_tool => ToolDecision::Ask(
                ApprovalRequest::new(ApprovalKind::new("mcp"), format!("{mcp_tool} {arguments}"))
                    .with_reason("the agent wants to invoke an MCP tool"),
            ),
        }
    }
}

fn is_read_only_fs_operation(arguments: &serde_json::Value) -> bool {
    matches!(
        arguments
            .get("operation")
            .and_then(serde_json::Value::as_str),
        Some("read" | "list" | "glob")
    )
}

impl Hooks for CliHooks {
    fn before_tool(
        &self,
        call: &ToolInvocation<'_>,
    ) -> impl std::future::Future<Output = ToolDecision> + Send {
        std::future::ready(self.decide(call.name, call.arguments))
    }
}

/// Answers approval requests by asking on the terminal.
#[derive(Debug, Clone, Copy, Default)]
pub struct TerminalApprover;

impl Approver for TerminalApprover {
    async fn approve(
        &self,
        request: ApprovalRequest,
    ) -> std::result::Result<Approval, ApproverFailure> {
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

        let approved = blocking::unblock(read_yes_no)
            .await
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
                    KeyCode::Char('c' | 'C' | 'd' | 'D')
                        if key.modifiers.contains(KeyModifiers::CONTROL) =>
                    {
                        eprint!("^C");
                        return Ok(false);
                    }
                    KeyCode::Char('y' | 'Y') if !key.modifiers.contains(KeyModifiers::CONTROL) => {
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
        Some((byte_idx, _)) => format!("{}[...]", s.get(..byte_idx).unwrap_or_default()),
        None => s.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_hooks_gate_command_mutating_fs_and_mcp_while_allowing_reads() {
        let hooks = CliHooks::new(false);
        assert!(matches!(
            hooks.decide(COMMAND_TOOL, &serde_json::json!({"program": "ls"})),
            ToolDecision::Ask(req) if req.kind().as_str() == "command"
        ));
        for op in ["write", "append", "delete"] {
            assert!(
                matches!(
                    hooks.decide(FILESYSTEM_TOOL, &serde_json::json!({"operation": op})),
                    ToolDecision::Ask(req) if req.kind().as_str() == "filesystem"
                ),
                "filesystem {op} must ask"
            );
        }
        for op in ["read", "list", "glob"] {
            assert!(
                matches!(
                    hooks.decide(FILESYSTEM_TOOL, &serde_json::json!({"operation": op})),
                    ToolDecision::Allow
                ),
                "filesystem {op} should be allowed"
            );
        }
        assert!(matches!(
            hooks.decide("websearch", &serde_json::json!({})),
            ToolDecision::Allow
        ));
        assert!(matches!(
            hooks.decide("webfetch", &serde_json::json!({})),
            ToolDecision::Allow
        ));
        assert!(matches!(
            hooks.decide("context7_resolve", &serde_json::json!({"query": "rust"})),
            ToolDecision::Ask(req) if req.kind().as_str() == "mcp"
        ));
    }

    #[test]
    fn approve_flag_allows_all_tools_without_prompting() {
        let hooks = CliHooks::new(true);
        for (name, args) in [
            (COMMAND_TOOL, serde_json::json!({"program": "rm"})),
            (FILESYSTEM_TOOL, serde_json::json!({"operation": "write"})),
            ("context7_resolve", serde_json::json!({})),
        ] {
            assert!(matches!(hooks.decide(name, &args), ToolDecision::Allow));
        }
    }
}
