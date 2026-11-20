//! Interactive CLI for testing aither agents.
//!
//! A REPL-style interface for testing agents with `OpenAI`, Claude, or Gemini.
//!
//! The agent has `websearch`, `webfetch`, `filesystem` and `command` tools,
//! plus the tools of any MCP server given. Every `command` call asks for the
//! user's approval on the terminal first.
//!
//! # Usage
//!
//! ```bash
//! # Auto-detect provider from available API keys
//! cargo run -p aither-cli
//!
//! # Explicit provider
//! OPENAI_API_KEY=xxx cargo run -p aither-cli -- --provider openai
//! ANTHROPIC_API_KEY=xxx cargo run -p aither-cli -- --provider claude
//! GEMINI_API_KEY=xxx cargo run -p aither-cli -- --provider gemini
//!
//! # Headless mode (single query, useful for testing/scripting)
//! cargo run -p aither-cli -- --prompt "What is 2+2?"
//! cargo run -p aither-cli -- --prompt "List files" --quiet
//!
//! # With additional MCP servers
//! cargo run -p aither-cli -- --mcp servers.json
//!
//! # Disable Context7 (documentation lookup)
//! cargo run -p aither-cli -- --no-context7
//! ```

mod hook;

use std::io::{self, Write};
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::time::Duration;

use aither_agent::{Agent, AgentEvent, CompactionOutcome, Hooks, RunOutcome};
use aither_core::LanguageModel;
use aither_mcp::{McpConnection, McpServersConfig};
use anyhow::{Context, Result};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::enable_raw_mode;
use executor_core::tokio::TokioGlobal;
use tracing_subscriber::EnvFilter;

mod provider;

use crate::hook::{CliHooks, RawModeGuard, TerminalApprover, print_event};
use crate::provider::Provider;
use aither_cloud::CloudProvider;

/// Interactive CLI for testing aither agents.
#[allow(clippy::struct_excessive_bools)]
#[derive(Parser, Debug)]
#[command(name = "aither", version, about)]
struct Args {
    /// Model to use. Auto-detects provider from model name prefix.
    #[arg(short, long)]
    model: Option<String>,

    /// Provider to use (openai, claude, gemini). Auto-detected if not specified.
    #[arg(short, long)]
    provider: Option<Provider>,

    /// Custom API base URL (for OpenAI-compatible endpoints, proxies, etc.)
    #[arg(short, long)]
    base_url: Option<String>,

    /// Path to MCP servers configuration file (JSON).
    #[arg(long)]
    mcp: Option<PathBuf>,

    /// System prompt for the agent.
    #[arg(short, long)]
    system: Option<String>,

    /// Disable Context7 MCP server (documentation lookup).
    #[arg(long)]
    no_context7: bool,

    /// Single prompt to run (headless mode). Runs query and exits.
    #[arg(long)]
    prompt: Option<String>,

    /// Quiet mode. Only output the response (useful with --prompt for scripting).
    #[arg(short, long)]
    quiet: bool,

    /// Automatically approve commands, mutating filesystem operations, and MCP tools.
    #[arg(long)]
    approve: bool,

    /// Run as ACP server (for integration with Zed and other editors).
    #[arg(long)]
    acp: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    // Create cloud provider
    let base_url = args.base_url.as_deref();
    let (cloud, model, provider_name) = if let Some(provider) = args.provider {
        let model = args
            .model
            .as_deref()
            .unwrap_or_else(|| provider.default_model());
        (
            provider.create(model, base_url)?,
            model.to_string(),
            provider.to_string(),
        )
    } else {
        let (cloud, model) = provider::auto_detect(args.model.as_deref(), base_url)?;
        let provider_name = match &cloud {
            CloudProvider::OpenAI(_) => "OpenAI",
            CloudProvider::Claude(_) => "Claude",
            CloudProvider::Gemini(_) => "Gemini",
            CloudProvider::Copilot(_) => "Copilot",
            CloudProvider::Codex(_) => "ChatGPT",
        };
        (cloud, model, provider_name.to_string())
    };

    if !args.quiet {
        println!("Aither Agent CLI");
        println!("Provider: {provider_name}");
        println!("Model: {model}");
        if args.prompt.is_none() {
            println!("Commands: /quit, /clear, /history, /compact");
        }
        println!();
    }

    // ACP server mode: serve editors over stdio rather than running a REPL.
    if args.acp {
        return run_acp_server(cloud).await;
    }

    // Headless mode: run single prompt and exit
    if let Some(ref prompt) = args.prompt {
        return Box::pin(run_headless(cloud, &args, prompt)).await;
    }

    Box::pin(run_repl(cloud, &args)).await
}

/// Run as ACP server for editor integration.
///
/// Each session the editor opens gets its own agent, rooted at the working
/// directory the editor supplied for that session.
async fn run_acp_server(cloud: CloudProvider) -> Result<()> {
    use aither_acp::aither::{AitherHost, SessionSetup};
    use aither_acp::domain::Implementation;
    use aither_acp::server::Server;

    let server = Server::new(
        AitherHost::new(TokioGlobal, move |setup: SessionSetup| {
            let cloud = cloud.clone();
            async move { Ok(acp_session_agent(cloud, setup)) }
        }),
        Implementation::new("aither", env!("CARGO_PKG_VERSION")),
    );
    let (_peer, serve) = server.serve_stdio();
    serve.await;
    Ok(())
}

/// Build the agent backing a single ACP session.
///
/// It reads files and the web but runs no commands: an ACP session's agent
/// has no hooks, so nothing would ask the editor's user before a command ran.
fn acp_session_agent(
    cloud: CloudProvider,
    setup: aither_acp::aither::SessionSetup,
) -> Agent<CloudProvider> {
    let builder = Agent::builder(cloud)
        .tool(aither_agent::filesystem::FileSystemTool::read_only(
            &setup.cwd,
        ))
        .tool(aither_agent::websearch::WebSearchTool::default())
        .tool(aither_agent::webfetch::WebFetchTool::new());
    setup
        .mcp
        .into_iter()
        .fold(builder, aither_agent::AgentBuilder::mcp)
        .build()
}

async fn build_agent(cloud: CloudProvider, args: &Args) -> Result<Agent<CloudProvider, CliHooks>> {
    let cwd = std::env::current_dir().context("failed to read the working directory")?;
    let mut builder = Agent::builder(cloud)
        .hooks(CliHooks::new(args.approve))
        .approver(TerminalApprover)
        .tool(aither_agent::filesystem::FileSystemTool::new(&cwd))
        .tool(aither_agent::command::CommandTool::new(&cwd))
        .tool(aither_agent::websearch::WebSearchTool::default())
        .tool(aither_agent::webfetch::WebFetchTool::new());

    // Add Context7 MCP server by default (documentation lookup)
    if !args.no_context7 {
        match McpConnection::http(&TokioGlobal, "https://mcp.context7.com/mcp").await {
            Ok(conn) => {
                if !args.quiet {
                    println!("Connected to Context7 MCP server");
                }
                builder = builder.mcp(conn);
            }
            Err(e) => {
                if !args.quiet {
                    eprintln!("Warning: Failed to connect to Context7: {e}");
                }
            }
        }
    }

    // Add MCP connections from config file
    if let Some(ref mcp_path) = args.mcp {
        let config_str = async_fs::read_to_string(mcp_path)
            .await
            .with_context(|| format!("failed to read MCP config from {}", mcp_path.display()))?;
        let config: McpServersConfig =
            serde_json::from_str(&config_str).with_context(|| "failed to parse MCP config")?;

        let connections = McpConnection::from_configs(&TokioGlobal, &config).await?;
        for (name, conn) in connections {
            if !args.quiet {
                println!("Connected to MCP server: {name}");
            }
            builder = builder.mcp(conn);
        }
    }

    if let Some(ref system) = args.system {
        builder = builder.instruction(system);
    }

    Ok(builder.build())
}

/// Runs `prompt` to its end, printing the run as it happens unless `quiet`,
/// and returns the reply.
async fn run_turn<M, H>(agent: &mut Agent<M, H>, prompt: &str, quiet: bool) -> Result<String>
where
    M: LanguageModel + 'static,
    H: Hooks,
{
    if quiet {
        return Ok(Box::pin(agent.query(prompt)).await?);
    }
    let flow = Box::pin(agent.run(prompt, []).drive(
        None,
        std::future::pending::<()>(),
        |reply, timed| async move {
            print_event(&timed);
            let reply = match timed.event {
                AgentEvent::RunEnded {
                    outcome: RunOutcome::Completed { final_text },
                    ..
                } if timed.agent.is_main() => Some(final_text),
                _ => reply,
            };
            ControlFlow::Continue(reply)
        },
    ))
    .await?;
    match flow {
        ControlFlow::Continue(Some(reply)) => Ok(reply),
        ControlFlow::Continue(None) | ControlFlow::Break(()) => {
            anyhow::bail!("the run ended without a reply")
        }
    }
}

/// Run a single prompt and exit (headless mode).
async fn run_headless(cloud: CloudProvider, args: &Args, prompt: &str) -> Result<()> {
    let mut agent = Box::pin(build_agent(cloud, args)).await?;

    match run_turn(&mut agent, prompt, args.quiet).await {
        Ok(response) => {
            if args.quiet {
                println!("{response}");
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

async fn run_repl(cloud: CloudProvider, args: &Args) -> Result<()> {
    let mut agent = Box::pin(build_agent(cloud, args)).await?;

    println!("Agent ready. Type your message or a command.\n");

    // REPL loop
    loop {
        let Some(input) = blocking::unblock(|| read_line("You> ")).await? else {
            break;
        };

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with('/') {
            match input {
                "/quit" | "/exit" | "/q" => break,
                "/clear" => {
                    agent.clear_history();
                    println!("History cleared.");
                    continue;
                }
                "/history" => {
                    print_history(&agent);
                    continue;
                }
                "/compact" => {
                    let driven = Box::pin(agent.compact(None).drive(
                        (),
                        std::future::pending(),
                        |(), timed| async move {
                            if let AgentEvent::CompactionEnded { outcome, .. } = timed.event {
                                match outcome {
                                    CompactionOutcome::Applied(result) => println!(
                                        "\x1b[2mCompacted {} messages into summary, starting fresh session\x1b[22m",
                                        result.messages_compacted
                                    ),
                                    CompactionOutcome::NothingToCompact => {
                                        println!("Nothing to compact (no conversation history).");
                                    }
                                    _ => {}
                                }
                            }
                            ControlFlow::<std::convert::Infallible, _>::Continue(())
                        },
                    ))
                    .await;
                    if let Err(e) = driven {
                        println!("\x1b[31mCompaction failed: {e}\x1b[0m");
                    }
                    continue;
                }
                cmd => {
                    println!("Unknown command: {cmd}");
                    println!("Available: /quit, /clear, /history, /compact");
                    continue;
                }
            }
        }

        // Run the agent
        println!("\nAgent>");
        if let Err(e) = run_turn(&mut agent, input, false).await {
            println!("\x1b[31mError: {e}\x1b[0m");
        }
        println!();
    }

    println!("Goodbye!");
    Ok(())
}

fn print_history<M, H>(agent: &Agent<M, H>)
where
    M: LanguageModel + 'static,
    H: Hooks,
{
    let history = agent.history();
    if history.is_empty() {
        println!("No conversation history.");
    } else {
        println!("Conversation history ({} messages):", history.len());
        for msg in &history {
            let role = msg.role();
            let content = truncate(msg.content(), 100);
            println!("  [{role}] {content}");
        }
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        Some((byte_index, _)) => format!("{}...", s.get(..byte_index).unwrap_or_default()),
        None => s.to_string(),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum LineKeyAction {
    Echo(char),
    Backspace,
    Submit,
    Cancel { interrupted: bool },
    Ignore,
}

fn classify_line_key(key: KeyEvent, buffer_empty: bool) -> LineKeyAction {
    if key.kind != KeyEventKind::Press {
        return LineKeyAction::Ignore;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return match key.code {
            KeyCode::Char('c' | 'C') => LineKeyAction::Cancel { interrupted: true },
            KeyCode::Char('d' | 'D') if buffer_empty => {
                LineKeyAction::Cancel { interrupted: false }
            }
            _ => LineKeyAction::Ignore,
        };
    }
    if key.modifiers.contains(KeyModifiers::ALT) {
        return LineKeyAction::Ignore;
    }
    match key.code {
        KeyCode::Esc => LineKeyAction::Cancel { interrupted: false },
        KeyCode::Enter => LineKeyAction::Submit,
        KeyCode::Backspace => LineKeyAction::Backspace,
        KeyCode::Char(c) => LineKeyAction::Echo(c),
        _ => LineKeyAction::Ignore,
    }
}

fn read_line(prompt: &str) -> Result<Option<String>> {
    print!("{prompt}");
    io::stdout().flush()?;
    enable_raw_mode()?;
    let _guard = RawModeGuard;

    let mut buffer = String::new();
    loop {
        if event::poll(Duration::from_millis(250))?
            && let Event::Key(key) = event::read()?
        {
            match classify_line_key(key, buffer.is_empty()) {
                LineKeyAction::Cancel { interrupted } => {
                    if interrupted {
                        print!("^C\r\n");
                    } else {
                        print!("\r\n");
                    }
                    io::stdout().flush().ok();
                    return Ok(None);
                }
                LineKeyAction::Submit => {
                    print!("\r\n");
                    io::stdout().flush().ok();
                    return Ok(Some(buffer));
                }
                LineKeyAction::Backspace => {
                    if buffer.pop().is_some() {
                        print!("\u{8} \u{8}");
                        io::stdout().flush().ok();
                    }
                }
                LineKeyAction::Echo(c) => {
                    buffer.push(c);
                    print!("{c}");
                    io::stdout().flush().ok();
                }
                LineKeyAction::Ignore => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctrl_c_and_ctrl_d_cancel_without_echoing_characters() {
        assert_eq!(
            classify_line_key(
                KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
                true
            ),
            LineKeyAction::Cancel { interrupted: true }
        );
        assert_eq!(
            classify_line_key(
                KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
                false
            ),
            LineKeyAction::Cancel { interrupted: true }
        );
        assert_eq!(
            classify_line_key(
                KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL),
                true
            ),
            LineKeyAction::Cancel { interrupted: false }
        );
        assert_eq!(
            classify_line_key(
                KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL),
                false
            ),
            LineKeyAction::Ignore
        );
        assert_eq!(
            classify_line_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE), true),
            LineKeyAction::Echo('c')
        );
    }

    #[test]
    fn args_parse_approve_flag() {
        let args = Args::parse_from(["aither", "--approve", "--prompt", "hello"]);
        assert!(args.approve);
        let default_args = Args::parse_from(["aither", "--prompt", "hello"]);
        assert!(!default_args.approve);
    }
}
