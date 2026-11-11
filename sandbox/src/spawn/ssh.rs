//! The SSH backend: a process on a server, under the server's own `heel`.

use std::borrow::Cow;
use std::process::Stdio as ProcessStdio;
use std::sync::Arc;

use askama::Template;
use executor_core::{AnyExecutor, Executor};
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};

use super::child::{ChildExit, Descendants, KillHandle, SandboxedChild, Streams, exit_watch};
use super::{
    EnvSource, Pipe, RemoteLaunch, Sandbox, SpawnCommand, SpawnError, SpawnSpec, script_reading,
};
use crate::SshServer;
use crate::process_tree::{KillTreeError, kill_process_tree};
use crate::ssh::SshClient;
use crate::terminal::{inject_env_exports, shell_launch};

#[derive(Template)]
#[template(path = "ssh_tool_execution.sh", escape = "none")]
struct SshToolExecutionTemplate<'a> {
    binary: Cow<'a, str>,
    tools_dir: Cow<'a, str>,
    /// Read, write and execute access to the shared `uv` directory, quoted.
    uv_grant: Option<String>,
    program: Cow<'a, str>,
    arguments: String,
}

/// The command line that runs `script` on a server: the managed shell under
/// the server's `heel` when `heel` is given, directly otherwise.
///
/// # Errors
/// Fails when a template cannot render.
pub fn remote_command(
    launch: &RemoteLaunch,
    heel: Option<&str>,
    script: &str,
) -> Result<String, askama::Error> {
    let script = launch.wrap_script(script)?;
    let (program, args) = shell_launch(launch.shell(), &script);
    let Some(binary) = heel else {
        return Ok(shell_words::join(
            std::iter::once(program.as_str()).chain(args.iter().map(String::as_str)),
        ));
    };
    let uv_grant = launch
        .writable_dir()
        .map(|dir| shell_words::quote(&format!("{dir}=rwx")).into_owned());
    SshToolExecutionTemplate {
        binary: shell_words::quote(binary),
        tools_dir: shell_words::quote(launch.bin_dir()),
        uv_grant,
        program: shell_words::quote(&program),
        arguments: shell_words::join(&args),
    }
    .render()
}

#[derive(Template)]
#[template(path = "ssh_job_start.sh", escape = "none")]
struct SshJobStartTemplate<'a> {
    /// The file the job's process group id is written to, quoted.
    pidfile: &'a str,
    command: &'a str,
}

#[derive(Template)]
#[template(path = "ssh_job_kill.sh", escape = "none")]
struct SshJobKillTemplate<'a> {
    pidfile: &'a str,
}

/// A job running on a server. Ending the local `ssh` client does not end the
/// remote command, so the job records its process group on the server and
/// [`kill`](Self::kill) signals the group there.
#[derive(Debug)]
pub struct RemoteJob {
    client: SshClient,
    server: SshServer,
    /// The pid file on the server, quoted.
    pidfile: String,
    /// The local `ssh` process.
    local: u32,
}

impl RemoteJob {
    /// Ends the remote process group, then the local client.
    pub(crate) async fn kill(&self) -> Result<(), KillTreeError> {
        let script = SshJobKillTemplate {
            pidfile: &self.pidfile,
        }
        .render()
        .map_err(|error| KillTreeError::Signal(std::io::Error::other(error)))?;
        let command = self
            .client
            .command(&self.server, &script)
            .await
            .map_err(|error| KillTreeError::Signal(std::io::Error::other(error)))?;
        let output = async_process::Command::from(command)
            .stdin(ProcessStdio::null())
            .output()
            .await
            .map_err(KillTreeError::Signal)?;
        if !output.status.success() {
            return Err(KillTreeError::Signal(std::io::Error::other(format!(
                "ending the job on {} failed ({}): {}",
                self.server.name,
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ))));
        }
        let local = kill_process_tree(self.local).await;
        // The client ends once its remote command did.
        #[cfg(unix)]
        if matches!(&local, Err(KillTreeError::Signal(error)) if error.raw_os_error() == Some(libc::ESRCH))
        {
            return Ok(());
        }
        local
    }
}

const fn process_stdio(pipe: Pipe) -> fn() -> ProcessStdio {
    match pipe {
        Pipe::Null => ProcessStdio::null,
        Pipe::Piped => ProcessStdio::piped,
    }
}

impl Sandbox {
    pub(super) async fn spawn_ssh(
        &self,
        server: &SshServer,
        server_heel: Option<&str>,
        launch: &RemoteLaunch,
        spec: SpawnSpec,
    ) -> Result<SandboxedChild, SpawnError> {
        let failed = |error: &dyn std::fmt::Display| SpawnError::Setup(error.to_string());
        let SpawnSpec {
            command,
            env,
            cwd,
            stdio,
            grants,
            ipc,
            lifetime: _,
            info,
        } = spec;
        // A grant is a path on one machine and a router is served by this
        // one: nothing of either reaches a server.
        if !grants.is_empty() {
            return Err(SpawnError::Unsupported {
                backend: "ssh",
                what: "a grant",
            });
        }
        if ipc.is_some() {
            return Err(SpawnError::Unsupported {
                backend: "ssh",
                what: "a way back to the host's commands",
            });
        }
        if cwd.is_some() {
            return Err(SpawnError::Setup(
                "an SSH process starts in the server's login directory: a host path names nothing there"
                    .into(),
            ));
        }
        // A confined process runs under the server's `heel`; one that is not
        // confined runs directly.
        let heel = if self.inner.unconfined {
            None
        } else {
            Some(server_heel.ok_or_else(|| {
                SpawnError::Setup(format!(
                    "the server {} has no heel, so a process on it cannot be sandboxed",
                    server.name
                ))
            })?)
        };

        // Only what the spec names travels: the sandbox's own environment and
        // its source describe this machine.
        let script = match command {
            SpawnCommand::Script(script) => script,
            SpawnCommand::Program { program, args } => {
                shell_words::join(std::iter::once(&program).chain(&args))
            }
        };
        // On the server the command's stdin is its session's channel, a pipe
        // even while the local client reads the null device.
        let script = script_reading(stdio.stdin, script)?;
        let spec_env: Arc<dyn EnvSource> = Arc::new(env);
        let script = inject_env_exports(script, Some(&spec_env), &info).map_err(|e| failed(&e))?;
        let remote = remote_command(launch, heel, &script).map_err(|e| failed(&e))?;
        let pidfile = shell_words::quote(&format!(
            "/tmp/aither-spawn-{}.pgid",
            uuid::Uuid::new_v4().simple()
        ))
        .into_owned();
        let remote = SshJobStartTemplate {
            pidfile: &pidfile,
            command: &remote,
        }
        .render()
        .map_err(|e| failed(&e))?;

        let command = self
            .inner
            .ssh_client
            .command(server, &remote)
            .await
            .map_err(|error| SpawnError::Start(error.to_string()))?;
        #[cfg(unix)]
        let command = {
            let mut command = command;
            std::os::unix::process::CommandExt::process_group(&mut command, 0);
            command
        };
        // The pipes are set on the async command: converting a std command
        // resets them, and spawning would then inherit this process's stdio.
        let mut child = async_process::Command::from(command)
            .stdin(process_stdio(stdio.stdin)())
            .stdout(process_stdio(stdio.stdout)())
            .stderr(process_stdio(stdio.stderr)())
            .spawn()
            .map_err(|error| SpawnError::Start(error.to_string()))?;
        let id = child.id();

        let executor = &self.inner.executor;
        let streams = pipe_streams(executor, &mut child);

        let watch = watch_exit(executor, child);
        Ok(SandboxedChild::new(
            id,
            streams,
            KillHandle::Remote(RemoteJob {
                client: self.inner.ssh_client.clone(),
                server: server.clone(),
                pidfile,
                local: id,
            }),
            watch,
        ))
    }
}

/// The watch of the local `ssh` process's exit.
fn watch_exit(
    executor: &AnyExecutor,
    mut child: async_process::Child,
) -> Arc<super::child::ExitWatch> {
    let (setter, watch) = exit_watch();
    executor
        .spawn(async move {
            // The processes on the server are out of sight, so nothing is
            // known of the descendants the exit left.
            setter.finish(
                match child.status().await {
                    Ok(status) => ChildExit::Exited(status),
                    Err(error) => ChildExit::Failed(error.to_string()),
                },
                Descendants::Unobserved,
            );
        })
        .detach();
    watch
}

/// The channels of the local `ssh` process's standard streams that it has
/// pipes for.
fn pipe_streams(executor: &AnyExecutor, child: &mut async_process::Child) -> Streams {
    let mut streams = Streams::default();
    if let Some(mut stdin) = child.stdin.take() {
        let (tx, rx) = async_channel::unbounded::<Vec<u8>>();
        streams.stdin = Some(tx);
        executor
            .spawn(async move {
                while let Ok(bytes) = rx.recv().await {
                    if stdin.write_all(&bytes).await.is_err() || stdin.flush().await.is_err() {
                        break;
                    }
                }
            })
            .detach();
    }
    if let Some(stdout) = child.stdout.take() {
        let (tx, rx) = async_channel::unbounded();
        streams.stdout = Some(rx);
        executor.spawn(pump(stdout, tx)).detach();
    }
    if let Some(stderr) = child.stderr.take() {
        let (tx, rx) = async_channel::unbounded();
        streams.stderr = Some(rx);
        executor.spawn(pump(stderr, tx)).detach();
    }
    streams
}

/// Moves a pipe to a channel of chunks that closes at its end.
async fn pump(
    mut reader: impl futures_lite::AsyncRead + Unpin,
    tx: async_channel::Sender<Vec<u8>>,
) {
    let mut buffer = vec![0_u8; 8192];
    while let Ok(count) = reader.read(&mut buffer).await {
        if count == 0 || tx.send(buffer[..count].to_vec()).await.is_err() {
            break;
        }
    }
}
