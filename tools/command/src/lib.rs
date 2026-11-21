//! Command execution tool for LLM agents.

use std::{
    borrow::Cow,
    fmt,
    path::PathBuf,
    process::{ExitStatus, Stdio},
};

use aither_core::llm::ToolContext;
use aither_core::llm::{Tool, ToolResult};
use askama::Template;
use async_process::Command;
use futures_lite::io::AsyncReadExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Why a [`CommandTool`] call failed.
#[derive(Debug, thiserror::Error)]
pub enum CommandToolError {
    /// The program is not in the tool's allow list.
    #[error("Program '{program}' is not allowed. Allowed commands: {allowed}")]
    NotAllowed {
        /// Program the model asked to run.
        program: String,
        /// The allow list, joined for display.
        allowed: String,
    },
    /// Spawning or awaiting the process failed.
    #[error("failed to run '{program}' in {cwd}: {source}")]
    Run {
        /// Program that was executed.
        program: String,
        /// Working directory it ran in.
        cwd: String,
        /// The underlying IO failure.
        #[source]
        source: std::io::Error,
    },
    /// The process ended without an exit code or signal.
    #[error("process ended with neither an exit code nor a signal: {0}")]
    UnknownTermination(ExitStatus),
    /// Rendering the tool's output failed.
    #[error("{0}")]
    Render(#[from] askama::Error),
}

/// Execute a shell command with arguments.
///
/// Runs a program with the specified arguments and returns stdout, stderr,
/// and exit code. Use this for executing system commands when you need
/// fine-grained control over arguments.
///
/// For complex scripts with pipelines, use `terminal` instead.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CommandArgs {
    /// Program name to execute (e.g., "ls", "grep", "git", "cargo").
    pub program: String,
    /// Command-line arguments as separate strings (e.g., `["-la"]` for ls, `["-r", "TODO", "src/"]` for grep).
    #[serde(default)]
    pub args: Vec<String>,
    /// Working directory for command execution. Omit to use default.
    pub cwd: Option<PathBuf>,
}

/// What a finished command wrote, rendered for the model by `templates/output.txt`.
#[derive(Debug, Clone, Template)]
#[template(path = "output.txt", escape = "none")]
pub struct CommandOutput {
    /// Program name that was executed.
    pub program: String,
    /// How the command terminated.
    pub termination: Termination,
    /// Standard output text.
    pub stdout: String,
    /// Standard error text.
    pub stderr: String,
}

/// How a command ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Termination {
    /// The process exited with this status code.
    Exited(i32),
    /// The process was killed by this signal number.
    Signaled(i32),
}

impl Termination {
    /// Classify a finished process.
    ///
    /// # Errors
    /// Fails when the process neither exited with a code nor was killed by a
    /// signal, which the platform does not report.
    pub fn of(status: ExitStatus) -> Result<Self, CommandToolError> {
        if let Some(code) = status.code() {
            return Ok(Self::Exited(code));
        }
        #[cfg(unix)]
        if let Some(signal) = std::os::unix::process::ExitStatusExt::signal(&status) {
            return Ok(Self::Signaled(signal));
        }
        Err(CommandToolError::UnknownTermination(status))
    }
}

impl fmt::Display for Termination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exited(code) => write!(f, "exited with status {code}"),
            Self::Signaled(number) => {
                #[cfg(unix)]
                if let Ok(signal) = nix::sys::signal::Signal::try_from(*number) {
                    return write!(f, "was killed by signal {signal} ({number})");
                }
                write!(f, "was killed by signal {number}")
            }
        }
    }
}

/// A tool that executes external commands.
#[derive(Debug, Clone)]
pub struct CommandTool {
    allowed: Option<Vec<String>>,
    default_cwd: PathBuf,
    max_output: usize,
    name: String,
}

impl CommandTool {
    /// Create a new command tool with a default working directory.
    pub fn new(default_cwd: impl Into<PathBuf>) -> Self {
        let default_cwd = default_cwd.into();
        Self {
            allowed: None,
            default_cwd,
            max_output: 16 * 1024,
            name: "command".into(),
        }
    }

    /// Restrict execution to the specified list of programs.
    #[must_use]
    pub fn restrict_to(mut self, commands: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.allowed = Some(commands.into_iter().map(Into::into).collect());
        self
    }

    /// Set the maximum output size in bytes.
    #[must_use]
    pub const fn max_output(mut self, bytes: usize) -> Self {
        self.max_output = bytes;
        self
    }

    /// Set a custom name for the tool.
    #[must_use]
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    fn ensure_allowed(&self, program: &str) -> Result<(), CommandToolError> {
        if let Some(allowed) = &self.allowed
            && !allowed.iter().any(|entry| entry == program)
        {
            return Err(CommandToolError::NotAllowed {
                program: program.to_string(),
                allowed: allowed.join(", "),
            });
        }
        Ok(())
    }

    fn truncate(&self, text: String) -> String {
        if text.len() <= self.max_output {
            return text;
        }

        // Cut at the last character boundary within the limit.
        let mut end = self.max_output;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let mut truncated = text;
        truncated.truncate(end);
        truncated.push_str("\n...output truncated...");
        truncated
    }
}

impl Tool for CommandTool {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(self.name.clone())
    }

    type Arguments = CommandArgs;
    type Res = ToolResult;
    type Error = CommandToolError;

    async fn call(
        &self,
        arguments: Self::Arguments,
        cx: ToolContext,
    ) -> Result<Self::Res, Self::Error> {
        self.ensure_allowed(&arguments.program)?;

        let working_dir = arguments.cwd.unwrap_or_else(|| self.default_cwd.clone());
        let mut child = Command::new(&arguments.program)
            .args(&arguments.args)
            .current_dir(&working_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|source| CommandToolError::Run {
                program: arguments.program.clone(),
                cwd: working_dir.display().to_string(),
                source,
            })?;

        let stdout_pipe = child.stdout.take();
        let stderr_pipe = child.stderr.take();
        let byte_cap = self.max_output.saturating_add(8);

        let collect = async {
            let (stdout_res, stderr_res) = futures_lite::future::zip(
                read_bounded_stream(stdout_pipe, byte_cap),
                read_bounded_stream(stderr_pipe, byte_cap),
            )
            .await;
            let stdout = stdout_res?;
            let stderr = stderr_res?;
            let status = child.status().await?;
            Ok::<_, std::io::Error>((status, stdout, stderr))
        };

        let cancellation = cx.cancellation().clone();
        let (status, stdout, stderr) = futures_lite::future::or(collect, async {
            cancellation.cancelled().await;
            Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "command cancelled",
            ))
        })
        .await
        .map_err(|source| CommandToolError::Run {
            program: arguments.program.clone(),
            cwd: working_dir.display().to_string(),
            source,
        })?;

        let response = CommandOutput {
            program: arguments.program,
            termination: Termination::of(status)?,
            stdout: self.truncate(String::from_utf8_lossy(&stdout).into_owned()),
            stderr: self.truncate(String::from_utf8_lossy(&stderr).into_owned()),
        };

        Ok(ToolResult::text(response.render()?))
    }
}

async fn read_bounded_stream<R>(reader: Option<R>, cap: usize) -> std::io::Result<Vec<u8>>
where
    R: futures_lite::io::AsyncRead + Unpin,
{
    let Some(mut reader) = reader else {
        return Ok(Vec::new());
    };
    let mut retained = Vec::new();
    let mut scratch = vec![0_u8; 8192];
    loop {
        match reader.read(&mut scratch).await {
            Ok(0) => return Ok(retained),
            Ok(n) => {
                if retained.len() < cap {
                    let take = (cap - retained.len()).min(n);
                    retained.extend_from_slice(&scratch[..take]);
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
            Err(err) => return Err(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run(tool: &CommandTool, program: &str, args: &[&str]) -> String {
        let result = tool
            .call(
                CommandArgs {
                    program: program.into(),
                    args: args.iter().map(ToString::to_string).collect(),
                    cwd: None,
                },
                ToolContext::new(),
            )
            .await
            .unwrap();
        match result {
            ToolResult::Text { text } => text,
            other => panic!("unexpected result {other:?}"),
        }
    }

    #[test]
    fn truncation_stays_on_character_boundaries() {
        // 'é' is two bytes: every odd limit lands inside a character.
        let text = "é".repeat(8);
        for limit in 0..=text.len() {
            let tool = CommandTool::new(".").max_output(limit);
            let cut = tool.truncate(text.clone());
            if limit >= text.len() {
                assert_eq!(cut, text);
            } else {
                let kept = cut.strip_suffix("\n...output truncated...").unwrap();
                assert_eq!(kept.len(), limit - limit % 2);
            }
        }
        let tool = CommandTool::new(".").max_output(1);
        assert_eq!(tool.truncate("日本語".into()), "\n...output truncated...");
    }

    #[tokio::test]
    async fn long_non_ascii_output_is_truncated_without_panic() {
        let tool = CommandTool::new(".").max_output(5);
        let text = run(&tool, "printf", &["日本語日本語"]).await;
        assert!(
            text.contains("stdout:\n日\n...output truncated..."),
            "{text}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn signal_termination_is_reported_with_the_signal() {
        let tool = CommandTool::new(".");
        let text = run(&tool, "sh", &["-c", "kill -KILL $$"]).await;
        assert!(
            text.starts_with("sh was killed by signal SIGKILL (9)."),
            "{text}"
        );
        assert!(!text.contains("status 0"), "{text}");
    }

    #[tokio::test]
    async fn exit_status_is_still_reported() {
        let tool = CommandTool::new(".");
        let text = run(&tool, "sh", &["-c", "exit 3"]).await;
        assert!(text.starts_with("sh exited with status 3."), "{text}");
    }

    #[test]
    fn output_is_prose_with_labelled_streams() {
        let output = CommandOutput {
            program: "ls".into(),
            termination: Termination::Exited(1),
            stdout: "a\nb".into(),
            stderr: "denied".into(),
        };
        assert_eq!(
            output.render().unwrap(),
            "ls exited with status 1.\nstdout:\na\nb\nstderr:\ndenied\n"
        );
    }

    #[test]
    fn silent_command_says_so() {
        let output = CommandOutput {
            program: "true".into(),
            termination: Termination::Exited(0),
            stdout: String::new(),
            stderr: String::new(),
        };
        assert_eq!(
            output.render().unwrap(),
            "true exited with status 0.\n(no output)\n"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn large_stream_is_bounded_and_truncated() {
        let tool = CommandTool::new(".").max_output(32);
        let text = run(&tool, "sh", &["-c", "yes x | head -c 262144"]).await;
        assert!(text.contains("\n...output truncated..."), "{text}");
        assert!(
            text.len() < 256,
            "retained output must stay bounded: {}",
            text.len()
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn cancellation_aborts_and_kills_spawned_child() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("child.pid");
        let tool = CommandTool::new(dir.path());
        let token = aither_core::llm::tool::Canceller::new();
        let cx = ToolContext::new().with_cancellation(token.signal());

        let pid_path = pid_file.clone();
        let stopper = tokio::spawn(async move {
            for _ in 0..100 {
                if let Ok(raw) = std::fs::read_to_string(&pid_path)
                    && raw.trim().parse::<i32>().is_ok()
                {
                    token.cancel();
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            token.cancel();
        });

        let result = tool
            .call(
                CommandArgs {
                    program: "sh".into(),
                    args: vec![
                        "-c".into(),
                        format!("echo $$ > '{}'; exec sleep 60", pid_file.display()),
                    ],
                    cwd: None,
                },
                cx,
            )
            .await;
        stopper.await.unwrap();
        assert!(result.is_err(), "cancelled command must return an error");

        let pid: i32 = std::fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let mut exited = false;
        for _ in 0..50 {
            if nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_err() {
                exited = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(exited, "kill_on_drop must terminate child pid {pid}");
    }
}
