//! Git inside a subagent's sandbox: the same filesystem grants and network
//! denial as the subagent's commands, as the subagent, with stdout streamed.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use async_channel::Sender;
use blocking::Unblock;
use executor_core::{Executor, Task};
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use heel::{
    Access, DenyAll, Grant, ProcessPriority, ResolvedPath, Sandbox, SandboxConfig, SecurityConfig,
    StdioConfig,
};

use super::{ContainerConfinement, prioritized_container_script};
use crate::MountRoot;
use crate::shell_session::ContainerExecOutcome;
use crate::subagent::{CommittingWorktree, ExportError, SandboxedExit, SandboxedGit, SandboxedRun};

/// How much of what a command printed on stderr is kept for a message.
const STDERR_KEPT: usize = 8 * 1024;

/// How much of a command's stdout travels in one chunk.
const CHUNK: usize = 64 * 1024;

fn stderr_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(&bytes[..bytes.len().min(STDERR_KEPT)])
        .trim()
        .to_string()
}

/// Kills the process, with everything it started, when dropped before it was
/// waited for.
struct KillOnDrop {
    child: heel::Child,
    finished: bool,
}

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.child.kill();
        }
    }
}

/// Git in the local sandbox of a subagent.
pub(super) struct LocalGit<E> {
    pub executor: E,
    /// The subagent's space.
    pub working_dir: PathBuf,
    pub writable: Vec<PathBuf>,
    pub readable: Vec<PathBuf>,
    /// The worktrees the subagent may commit in, to find the worktree a
    /// repository answers to.
    pub committing: Vec<CommittingWorktree>,
    pub priority: ProcessPriority,
    /// The parent terminal's `heel` binary.
    pub heel: heel::HeelBinary,
    /// Cached host `git` binary `(path, parent_dir)` shared across subagents.
    pub host_git: Arc<OnceLock<Option<(PathBuf, PathBuf)>>>,
}

/// The host's own `git` as an absolute path, so a sandboxed git does not
/// depend on the sandbox's `PATH` — which holds only system directories on
/// Windows, where `git` never lives. The directory that holds it is granted
/// beside it: the binary's sibling libraries live there.
fn locate_host_git_binary() -> Option<(PathBuf, PathBuf)> {
    // `/usr/bin/git` on macOS is the xcode-select stub: it execs the real
    // toolchain git, which a grant on `/usr/bin` does not cover. `xcrun`
    // names the resolved binary.
    #[cfg(target_os = "macos")]
    if let Ok(resolved) = std::process::Command::new("xcrun")
        .args(["-f", "git"])
        .output()
        && resolved.status.success()
    {
        let path = PathBuf::from(String::from_utf8_lossy(&resolved.stdout).trim());
        if let (true, Some(dir)) = (path.is_file(), path.parent().map(Path::to_path_buf)) {
            return Some((path, dir));
        }
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for name in ["git", "git.exe"] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some((candidate, dir));
            }
        }
    }
    None
}

/// A path spelled the way the sandbox resolves grants: canonicalized, minus
/// the `\\?\` prefix canonicalization puts on Windows paths. A grant and a
/// spelled path must name the object the same way — an 8.3 alias and its
/// long name do not match.
fn canonical(path: &std::path::Path) -> PathBuf {
    let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    resolved
        .to_str()
        .and_then(|spelled| spelled.strip_prefix(r"\\?\"))
        .map_or_else(|| resolved.clone(), PathBuf::from)
}

/// The object a spelled path names, in the spelling a resolved grant
/// checks.
fn resolved(path: &std::path::Path) -> heel::Result<ResolvedPath> {
    ResolvedPath::new(canonical(path))
}

/// The object stores the repository borrows through its
/// `objects/info/alternates`, as absolute paths. Each line there is relative
/// to the repository's object directory.
fn borrowed_object_stores(repository: &Path) -> Vec<PathBuf> {
    let objects = repository.join("objects");
    std::fs::read_to_string(objects.join("info/alternates"))
        .map_or_default(|alternates| alternates.lines().map(|line| objects.join(line)).collect())
}

impl<E: Executor + Clone + 'static> LocalGit<E> {
    async fn sandbox(&self, repository: &Path) -> Result<(Sandbox, String, PathBuf), ExportError> {
        let writable = self.writable.clone();
        let space_dir = self.working_dir.clone();
        let readable = self.readable.clone();
        let committing = self.committing.clone();
        let repository = repository.to_path_buf();
        let heel = self.heel.clone();
        let priority = self.priority;
        let host_git = Arc::clone(&self.host_git);
        let (config, program, git_dir) = blocking::unblock(move || -> Result<_, ExportError> {
            let enter = committing
                .iter()
                .find(|granted| {
                    canonical(&repository).parent() == Some(&*canonical(granted.git_dir()))
                })
                .map_or(repository.as_path(), CommittingWorktree::worktree);
            let git = host_git.get_or_init(locate_host_git_binary).clone();
            // On Windows the working directory is the git binary's own: git's
            // resolution of the directory it runs in asks more of ancestors than
            // the traverse the sandbox grants them — reading a directory's entry
            // needs its parent's list — where the binary's system ancestors are
            // legible to every package.
            #[cfg(windows)]
            let working_dir = git
                .as_ref()
                .map_or_else(|| enter.to_path_buf(), |(_, dir)| dir.clone());
            #[cfg(not(windows))]
            let working_dir = enter.to_path_buf();
            let borrowed = borrowed_object_stores(&repository);
            let grants: Vec<Grant> = writable
                .iter()
                .map(|path| resolved(path).map(|p| Grant::new(p, Access::WRITE)))
                .chain([resolved(&space_dir).map(|p| Grant::new(p, Access::READ))])
                .chain(
                    readable
                        .iter()
                        .map(|path| resolved(path).map(|p| Grant::new(p, Access::READ))),
                )
                .chain(git.as_ref().map(|(_, dir)| {
                    resolved(dir).map(|p| Grant::new(p, Access::READ | Access::EXEC))
                }))
                .chain([resolved(&repository).map(|p| Grant::new(p, Access::READ))])
                .chain(committing.iter().map(|granted| {
                    resolved(granted.worktree()).map(|p| Grant::new(p, Access::READ))
                }))
                .chain(
                    borrowed
                        .iter()
                        .map(|store| resolved(store).map(|p| Grant::new(p, Access::READ))),
                )
                .chain(
                    borrowed
                        .iter()
                        .filter_map(|store| store.parent().map(Path::to_path_buf))
                        .map(|common| common.join("config"))
                        .filter(|config| config.is_file())
                        .map(|config| resolved(&config).map(|p| Grant::new(p, Access::READ))),
                )
                .collect::<heel::Result<_>>()
                .map_err(|error| ExportError::Launch(error.to_string()))?;
            let config = SandboxConfig::builder()
                .heel_binary(heel)
                .network(DenyAll)
                .working_dir(canonical(&working_dir))
                .grants(grants)
                .security(SecurityConfig::interactive())
                .priority(priority)
                .build();
            let program = git.map_or_else(
                || "git".to_string(),
                |(path, _)| canonical(&path).to_string_lossy().into_owned(),
            );
            let git_dir = canonical(&repository);
            Ok((config, program, git_dir))
        })
        .await?;
        let sandbox = Sandbox::with_config_and_executor(config, self.executor.clone())
            .await
            .map_err(|error| ExportError::Launch(error.to_string()))?;
        Ok((sandbox, program, git_dir))
    }

    async fn run_in_sandbox(&self, run: SandboxedRun<'_>) -> Result<SandboxedExit, ExportError> {
        let (sandbox, program, git_dir) = self.sandbox(run.repository).await?;
        let child = sandbox
            .command(program)
            .args(run.args)
            .env("HOME", self.working_dir.to_string_lossy())
            // An absolute `GIT_DIR` is used as it stands — `git` discovers a
            // relative one by resolving it, a read on every ancestor that a
            // traverse grant does not carry.
            .env("GIT_DIR", git_dir.to_string_lossy())
            .stdin(StdioConfig::Piped)
            .stdout(StdioConfig::Piped)
            .stderr(StdioConfig::Piped)
            .spawn()
            .await
            .map_err(|error| ExportError::Launch(error.to_string()))?;
        let mut child = KillOnDrop {
            child,
            finished: false,
        };

        let stdin = child.child.take_stdin();
        let stdout = child.child.take_stdout();
        let stderr = child.child.take_stderr();
        let feed = async {
            let Some(stdin) = stdin else {
                return Ok(());
            };
            let mut stdin = Unblock::new(stdin);
            // Git may stop reading early; its exit status says why.
            match stdin.write_all(&run.stdin).await {
                Ok(()) => stdin.flush().await.or_else(ignore_broken_pipe)?,
                Err(error) => ignore_broken_pipe(error)?,
            }
            // Closing the pipe is what tells git its input is complete.
            drop(stdin.into_inner().await);
            Ok::<_, ExportError>(())
        };
        let forward = async {
            let Some(stdout) = stdout else {
                return Ok(());
            };
            let mut stdout = Unblock::new(stdout);
            let mut buffer = vec![0_u8; CHUNK];
            loop {
                let read = stdout.read(&mut buffer).await?;
                if read == 0 {
                    return Ok::<_, ExportError>(());
                }
                run.stdout
                    .send(buffer[..read].to_vec())
                    .await
                    .map_err(|_| ExportError::OutputClosed)?;
            }
        };
        let errors = async {
            let Some(stderr) = stderr else {
                return Ok(Vec::new());
            };
            let mut stderr = Unblock::new(stderr);
            let mut kept = Vec::new();
            let mut buffer = vec![0_u8; 4096];
            loop {
                let read = stderr.read(&mut buffer).await?;
                if read == 0 {
                    return Ok::<_, ExportError>(kept);
                }
                // The rest is read and dropped so git is never held back.
                let room = STDERR_KEPT.saturating_sub(kept.len());
                kept.extend_from_slice(&buffer[..read.min(room)]);
            }
        };
        let ((), (), stderr) = futures::future::try_join3(feed, forward, errors).await?;
        let status = child
            .child
            .wait()
            .await
            .map_err(|error| ExportError::Launch(error.to_string()))?;
        child.finished = true;
        Ok(SandboxedExit {
            code: status.code(),
            stderr: stderr_text(&stderr),
        })
    }
}

fn ignore_broken_pipe(error: std::io::Error) -> Result<(), ExportError> {
    if error.kind() == std::io::ErrorKind::BrokenPipe {
        Ok(())
    } else {
        Err(error.into())
    }
}

impl<E: Executor + Clone + 'static> SandboxedGit for LocalGit<E> {
    async fn run<'a>(&'a self, run: SandboxedRun<'a>) -> Result<SandboxedExit, ExportError> {
        self.run_in_sandbox(run).await
    }
}

/// Git in the per-command sandbox of a subagent inside a container.
pub(super) struct ContainerGit<E> {
    pub executor: E,
    pub launch: crate::ContainerLaunch,
    pub runtime: crate::ContainerShellRuntime,
    pub confinement: ContainerConfinement,
    pub mount: MountRoot,
    pub priority: ProcessPriority,
}

impl<E: Executor + Clone + 'static> ContainerGit<E> {
    async fn run_in_container(&self, run: SandboxedRun<'_>) -> Result<SandboxedExit, ExportError> {
        let runtime = self.runtime.clone();
        let repository = self
            .mount
            .container_path(run.repository)
            .map_err(|error| ExportError::Launch(error.to_string()))?;
        let script = shell_words::join(
            ["git", "--git-dir", repository.as_str()]
                .into_iter()
                .chain(run.args.iter().copied()),
        );
        let wrapped = prioritized_container_script(
            &script,
            self.priority,
            self.launch.bin_dir(),
            self.confinement.working_dir.as_str(),
            Some(&self.confinement),
        )
        .map_err(|error| ExportError::Launch(error.to_string()))?;

        // Dropping the kill switch, as this future's being dropped does, is
        // what stops the command inside the container.
        let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
        let (stdin_tx, stdin_rx) = async_channel::bounded::<Vec<u8>>(1);
        if !run.stdin.is_empty() {
            stdin_tx
                .try_send(run.stdin)
                .expect("an empty channel with room for one chunk takes it");
        }
        drop(stdin_tx);
        let (done_tx, done_rx) = async_channel::bounded(1);
        let exec = runtime.exec();
        let container_id = runtime.container_id().to_string();
        let working_dir = self.confinement.working_dir.as_str().to_string();
        let stdout: Sender<Vec<u8>> = run.stdout;
        self.executor
            .spawn(async move {
                let outcome = exec
                    .exec_streamed_boxed(
                        &container_id,
                        &wrapped,
                        &working_dir,
                        kill_rx,
                        stdin_rx,
                        stdout,
                        crate::ExecLifetime::foreground(),
                    )
                    .await;
                let _ = done_tx.send(outcome).await;
            })
            .detach();
        let outcome = done_rx
            .recv()
            .await
            .map_err(|_| ExportError::Launch("the container exec ended without a result".into()))?;
        drop(kill_tx);
        match outcome {
            Ok(ContainerExecOutcome::Completed { output, .. }) => Ok(SandboxedExit {
                code: output.status.code(),
                stderr: stderr_text(&output.stderr),
            }),
            // Only this run's own kill switch, or a gone reader, kills it.
            Ok(ContainerExecOutcome::Killed) => Err(ExportError::OutputClosed),
            Err(error) => Err(ExportError::Launch(error)),
        }
    }
}

impl<E: Executor + Clone + 'static> SandboxedGit for ContainerGit<E> {
    async fn run<'a>(&'a self, run: SandboxedRun<'a>) -> Result<SandboxedExit, ExportError> {
        self.run_in_container(run).await
    }
}
