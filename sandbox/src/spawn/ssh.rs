//! The SSH backend: a process on a server, under the server's own `heel`.

use std::borrow::Cow;
use std::process::Stdio as ProcessStdio;
use std::sync::Arc;

use askama::Template;
use executor_core::{AnyExecutor, Executor};
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};

use super::child::{ChildExit, Descendants, KillHandle, SandboxedChild, Streams, exit_watch};
use super::{
    EnvSource, Pipe, RemoteLaunch, Sandbox, SpawnCommand, SpawnError, SpawnSpec, Stdio,
    script_reading,
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

fn remote_command_with_exports(
    launch: &RemoteLaunch,
    heel: Option<&str>,
    extra_exports: &str,
    script: &str,
) -> Result<String, askama::Error> {
    let script = launch.wrap_script_with_exports(extra_exports, script)?;
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

fn validate_ssh_spec(has_grants: bool, has_ipc: bool, has_cwd: bool) -> Result<(), SpawnError> {
    // A grant is a path on one machine and a router is served by this
    // one: nothing of either reaches a server.
    if has_grants {
        return Err(SpawnError::Unsupported {
            backend: "ssh",
            what: "a grant",
        });
    }
    if has_ipc {
        return Err(SpawnError::Unsupported {
            backend: "ssh",
            what: "a way back to the host's commands",
        });
    }
    if has_cwd {
        return Err(SpawnError::Setup(
            "an SSH process starts in the server's login directory: a host path names nothing there"
                .into(),
        ));
    }
    Ok(())
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
        validate_ssh_spec(!grants.is_empty(), ipc.is_some(), cwd.is_some())?;
        let (stdin_pipe, stdout_pipe, stderr_pipe) = match stdio {
            Stdio::Pipes {
                stdin,
                stdout,
                stderr,
            } => (stdin, stdout, stderr),
            Stdio::Pty { .. } => {
                return Err(SpawnError::Unsupported {
                    backend: "ssh",
                    what: "a pseudo-terminal",
                });
            }
        };
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
        let script = script_reading(stdin_pipe, script)?;
        let spec_env: Arc<dyn EnvSource> = Arc::new(env);
        let extra_exports =
            inject_env_exports(String::new(), Some(&spec_env), &info).map_err(|e| failed(&e))?;
        let remote = remote_command_with_exports(launch, heel, &extra_exports, &script)
            .map_err(|e| failed(&e))?;
        let pidfile_raw = format!("/tmp/aither-spawn-{}.pgid", uuid::Uuid::new_v4().simple());
        let pidfile = shell_words::quote(&pidfile_raw).into_owned();
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
        // stderr is always piped so we can read and strip the remote PID marker
        // line (`<pidfile> pid <remote_pid>`) before forwarding user stderr.
        let mut child = async_process::Command::from(command)
            .stdin(process_stdio(stdin_pipe)())
            .stdout(process_stdio(stdout_pipe)())
            .stderr(ProcessStdio::piped())
            .spawn()
            .map_err(|error| SpawnError::Start(error.to_string()))?;
        let local_id = child.id();

        let executor = &self.inner.executor;
        let (streams, pid_rx) = pipe_streams(
            executor,
            &mut child,
            stderr_pipe == Pipe::Piped,
            pidfile_raw,
        );
        let remote_pid = pid_rx.recv().await.unwrap_or(local_id);

        let watch = watch_exit(executor, child);
        Ok(SandboxedChild::with_host_id(
            remote_pid,
            local_id,
            streams,
            KillHandle::Remote(RemoteJob {
                client: self.inner.ssh_client.clone(),
                server: server.clone(),
                pidfile,
                local: local_id,
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

/// The channels of the local `ssh` process's standard streams and the one-shot
/// receiver reporting the remote server PID emitted by `ssh_job_start.sh`.
fn pipe_streams(
    executor: &AnyExecutor,
    child: &mut async_process::Child,
    keep_stderr: bool,
    pidfile_raw: String,
) -> (Streams, async_channel::Receiver<u32>) {
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
    let (pid_tx, pid_rx) = async_channel::bounded::<u32>(1);
    if let Some(stderr) = child.stderr.take() {
        let stderr_tx = if keep_stderr {
            let (tx, rx) = async_channel::unbounded();
            streams.stderr = Some(rx);
            Some(tx)
        } else {
            None
        };
        executor
            .spawn(pump_ssh_stderr(stderr, stderr_tx, pid_tx, pidfile_raw))
            .detach();
    }
    (streams, pid_rx)
}

async fn pump_ssh_stderr(
    mut reader: impl futures_lite::AsyncRead + Unpin,
    tx: Option<async_channel::Sender<Vec<u8>>>,
    pid_tx: async_channel::Sender<u32>,
    pidfile_raw: String,
) {
    let prefix = format!("{pidfile_raw} pid ");
    let mut header = Vec::new();
    let mut buffer = [0_u8; 8192];
    let mut header_done = false;
    let mut pid_tx = Some(pid_tx);

    loop {
        match reader.read(&mut buffer).await {
            Ok(0) => {
                if !header_done
                    && !header.is_empty()
                    && let Some(tx) = &tx
                {
                    let _ = tx.send(std::mem::take(&mut header)).await;
                }
                break;
            }
            Ok(count) => {
                let chunk = &buffer[..count];
                if !header_done {
                    header.extend_from_slice(chunk);
                    if let Some(nl) = header.iter().position(|b| *b == b'\n') {
                        header_done = true;
                        let first_line = String::from_utf8_lossy(&header[..nl]);
                        let trimmed = first_line.trim_end_matches('\r');
                        let rest = header[nl + 1..].to_vec();
                        if let Some(raw_pid) = trimmed.strip_prefix(&prefix)
                            && let Ok(pid) = raw_pid.trim().parse::<u32>()
                        {
                            if let Some(ptx) = pid_tx.take() {
                                let _ = ptx.try_send(pid);
                            }
                            header.clear();
                            if !rest.is_empty()
                                && let Some(tx) = &tx
                                && tx.send(rest).await.is_err()
                            {
                                break;
                            }
                        } else {
                            // First line was not the PID banner (e.g. ssh banner or auth error):
                            // release pid waiter so spawn does not hang and forward everything.
                            pid_tx.take();
                            let whole = std::mem::take(&mut header);
                            if let Some(tx) = &tx
                                && tx.send(whole).await.is_err()
                            {
                                break;
                            }
                        }
                    } else if header.len() > 1024 {
                        header_done = true;
                        pid_tx.take();
                        let whole = std::mem::take(&mut header);
                        if let Some(tx) = &tx
                            && tx.send(whole).await.is_err()
                        {
                            break;
                        }
                    }
                } else if let Some(tx) = &tx
                    && tx.send(chunk.to_vec()).await.is_err()
                {
                    break;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
}

/// Moves a pipe to a channel of chunks that closes at its end.
async fn pump(
    mut reader: impl futures_lite::AsyncRead + Unpin,
    tx: async_channel::Sender<Vec<u8>>,
) {
    let mut buffer = vec![0_u8; 8192];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) => break,
            Ok(count) => {
                if tx.send(buffer[..count].to_vec()).await.is_err() {
                    break;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
}
