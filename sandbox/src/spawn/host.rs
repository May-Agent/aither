//! The host backend: a process under heel's sandbox on this machine.

use std::any::Any;
use std::path::{Path, PathBuf};

use executor_core::Executor;
use heel::{
    Access, Audited, Grant, IpcRouter, NetworkPolicy, ResolvedPath, SandboxConfig, SecurityConfig,
    StdioConfig,
};

use super::child::{ChildExit, Descendants, KillHandle, SandboxedChild, Streams, exit_watch};
use super::domain::DomainGate;
use super::stdio::{read_chunks, write_chunks};
use super::{
    Backend, LaunchInfo, Pipe, Sandbox, SpawnCommand, SpawnError, SpawnGrant, SpawnSpec, Stdio,
};
#[cfg(target_os = "linux")]
use crate::process_tree::JobProcess;
use crate::process_tree::{JobMarker, JobTree};
use crate::stdin_watch::{FeedCounter, LocalStdinProbe, StdinWatchSignal};
use crate::terminal::{LocalChild, LocalStdinWatch, wait_for_local_process_exit};

/// How long a job's descendants may outlive its top-level process, which
/// decides how a Linux job is supervised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeLifetime {
    /// None: the caller ends the survivors.
    Foreground,
    /// All of them: the job lasts until its whole tree has exited.
    Background,
}

/// A process spawned on the host: the child, the handle that ends its tree
/// and heel's sandbox, whose drop ends the processes under it.
pub struct LocalSpawn {
    pub(crate) child: LocalChild,
    pub(crate) tree: JobTree,
    /// Kept alive for as long as the process runs.
    pub(crate) sandbox: Box<dyn Any + Send>,
}

/// The object a spelled path names, in the spelling a resolved grant checks:
/// canonicalized, without the verbatim prefix Windows adds; an 8.3 alias or a
/// symlinked component does not name the same object.
pub fn resolved(path: &Path) -> heel::Result<ResolvedPath> {
    ResolvedPath::new(std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()))
}

/// The grants that let git work in a granted linked worktree: its common git
/// directory, with the access `git_dir` gives a writable worktree's (the
/// session's own for the main agent, read for a child). The directory is the
/// repository's, outside the worktree, and a strict sandbox reads only what
/// is granted. Only a pointer git's back-links confirm
/// ([`crate::subagent::confirmed_common_dir`]) grants anything, because the
/// `.git` file sits in a directory the session can write.
pub async fn linked_worktree_grants(
    writable: &[PathBuf],
    readable: &[PathBuf],
    git_dir: crate::GitDirAccess,
) -> heel::Result<Vec<Grant>> {
    let mut grants = Vec::new();
    for (path, access) in writable
        .iter()
        .map(|path| (path, git_dir.writable_worktree_access()))
        .chain(readable.iter().map(|path| (path, Access::READ)))
    {
        if let Some(common) = crate::subagent::confirmed_common_dir(path).await {
            grants.push(Grant::new(resolved(&common)?, access));
        }
    }
    Ok(grants)
}

/// The paths a host sandbox must not read: heel's credential table plus the
/// embedder's `denied` absolute paths. Everything else stays readable.
#[cfg(not(windows))]
fn read_denylist(denied: &[PathBuf]) -> heel::ReadDenylist {
    denied
        .iter()
        .fold(heel::ReadDenylist::credentials(), |denylist, path| {
            denylist.deny(path)
        })
}

const fn stdio_config(pipe: Pipe) -> StdioConfig {
    match pipe {
        Pipe::Null => StdioConfig::Null,
        Pipe::Piped => StdioConfig::Piped,
    }
}

/// What a launch needs besides heel's configuration.
struct Launch {
    program: String,
    args: Vec<String>,
    env: Vec<(String, String)>,
    cwd: Option<PathBuf>,
    stdio: Stdio,
    lifetime: TreeLifetime,
    marker: JobMarker,
}

impl Sandbox {
    fn heel_binary(&self) -> Result<&heel::HeelBinary, SpawnError> {
        match &self.inner.backend {
            Backend::Host { heel } => Ok(heel),
            Backend::Container { .. } | Backend::Ssh { .. } => Err(SpawnError::Setup(
                "a local spawn needs the host backend".into(),
            )),
        }
    }

    /// The canonical spelling of the working directory: a symlink in it fails
    /// heel's own resolution, which does not follow a link a sandboxed
    /// process could have planted.
    fn canonical_working_dir(&self) -> PathBuf {
        std::fs::canonicalize(&self.inner.working_dir)
            .unwrap_or_else(|_| self.inner.working_dir.clone())
    }

    /// The environment of a host spawn: the sandbox's own, its source's, its
    /// space's, then the spec's, each overriding the one before.
    fn host_environment(
        &self,
        spec_env: Vec<(String, String)>,
        info: &LaunchInfo,
    ) -> Result<Vec<(String, String)>, SpawnError> {
        let mut env = self.inner.environment.clone();
        if let Some(source) = &self.inner.env_source {
            env.extend(source.env(info));
        }
        for (name, path) in &self.inner.space_environment {
            let value = path.to_str().ok_or_else(|| {
                SpawnError::Setup(format!(
                    "{} is not UTF-8, so the variable {name} cannot name it",
                    path.display()
                ))
            })?;
            env.push((name.clone(), value.to_owned()));
        }
        env.extend(spec_env);
        Ok(env)
    }

    /// The grants a confined spawn carries: the embedder's fixed ones, every
    /// live writable path with write and execute access (a sandboxed process
    /// can already run anything it writes through an interpreter, and builds
    /// must execute their own outputs), every readable path with read access,
    /// the common git directory of a granted linked worktree, and the spec's
    /// own.
    async fn confined_grants(
        &self,
        snapshot: &super::GrantSnapshot,
        extra: Vec<Grant>,
    ) -> Result<Vec<Grant>, SpawnError> {
        let setup = |error: heel::Error| SpawnError::Setup(error.to_string());
        let mut grants = self.inner.fixed_grants.clone();
        for (path, access) in snapshot
            .writable
            .iter()
            .map(|path| (path, Access::WRITE | Access::EXEC))
            .chain(snapshot.readable.iter().map(|path| (path, Access::READ)))
        {
            // A grant deleted since it was made grants nothing, and the
            // sandbox cannot resolve it.
            if matches!(path.try_exists(), Ok(false)) {
                continue;
            }
            grants.push(Grant::new(resolved(path).map_err(setup)?, access));
        }
        grants.extend(
            linked_worktree_grants(&snapshot.writable, &snapshot.readable, snapshot.git_dir)
                .await
                .map_err(setup)?,
        );
        grants.extend(extra);
        Ok(grants)
    }

    fn confined_config<N: NetworkPolicy>(
        &self,
        policy: N,
        grants: Vec<Grant>,
        snapshot: &super::GrantSnapshot,
        heel: heel::HeelBinary,
        ipc: Option<IpcRouter>,
    ) -> SandboxConfig<N> {
        let builder = SandboxConfig::builder().network(policy);
        // The Windows backend refuses a write denylist rather than run
        // weaker, and the same holds for the read denylist: reads are open
        // everywhere except the credential table and the embedder's
        // additions.
        #[cfg(not(windows))]
        let builder = {
            let builder = match &self.inner.write_denylist {
                Some(denylist) => builder.write_denylist(denylist.clone()),
                None => builder,
            };
            builder
                .filesystem_strict(false)
                .read_denylist(read_denylist(&snapshot.read_denied))
        };
        #[cfg(windows)]
        let _ = snapshot;
        let builder = builder
            .working_dir(self.canonical_working_dir())
            .grants(grants)
            .heel_binary(heel)
            .security(SecurityConfig::interactive())
            // The session's own processes run their build outputs: anything
            // written here could already run through an interpreter.
            .executable_working_dir(true)
            .priority(self.inner.priority);
        #[cfg(target_os = "linux")]
        let builder = builder.supervised_spawn(true);
        // `ipc` changes the builder's type, so both arms build where they are.
        match ipc {
            Some(router) => builder.ipc(router).build(),
            None => builder.build(),
        }
    }

    fn unconfined_config(
        &self,
        heel: heel::HeelBinary,
        ipc: Option<IpcRouter>,
    ) -> SandboxConfig<heel::HostAccess> {
        // Host access is approved execution: the user runs the script
        // themselves in every respect except IPC plumbing, so no credential
        // or filesystem protection may apply.
        let builder = SandboxConfig::builder()
            .working_dir(self.canonical_working_dir())
            .host_access()
            .heel_binary(heel)
            .executable_working_dir(true)
            .priority(self.inner.priority);
        #[cfg(target_os = "linux")]
        let builder = builder.supervised_spawn(true);
        match ipc {
            Some(router) => builder.ipc(router).build(),
            None => builder.build(),
        }
    }

    /// Starts `spec` on the host under heel and hands back the raw child, for
    /// an embedder that supervises the job itself.
    pub(crate) async fn spawn_local(&self, spec: SpawnSpec) -> Result<LocalSpawn, SpawnError> {
        let heel = self.heel_binary()?.clone();
        let SpawnSpec {
            command,
            env,
            cwd,
            stdio,
            grants: extra_grants,
            ipc,
            lifetime,
            info,
        } = spec;
        // A process on the host is a program: only a container or a server
        // has a managed shell to run a script in.
        let SpawnCommand::Program { program, args } = command else {
            return Err(SpawnError::Unsupported {
                backend: "host",
                what: "a script",
            });
        };
        let extra_grants = extra_grants
            .into_iter()
            .map(|grant| match grant {
                SpawnGrant::Host(grant) => Ok(grant),
                SpawnGrant::Container { .. } => Err(SpawnError::Unsupported {
                    backend: "host",
                    what: "a grant on a container path",
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let marker = JobMarker::new();
        let mut env = self.host_environment(env, &info)?;
        env.push(marker.entry());
        let launch = Launch {
            program,
            args,
            env,
            cwd,
            stdio,
            lifetime,
            marker,
        };
        if self.inner.unconfined {
            let config = self.unconfined_config(heel, ipc);
            return self.launch(config, launch).await;
        }
        let snapshot = self.inner.grants.snapshot();
        let grants = self.confined_grants(&snapshot, extra_grants).await?;
        let gate = DomainGate {
            lists: self.inner.network.clone(),
            approver: self.inner.approver.clone(),
            agent: self.inner.agent.clone(),
        };
        if let Some(log) = self.inner.audit.clone() {
            let config =
                self.confined_config(Audited::new(gate, log), grants, &snapshot, heel, ipc);
            self.launch(config, launch).await
        } else {
            let config = self.confined_config(gate, grants, &snapshot, heel, ipc);
            self.launch(config, launch).await
        }
    }

    async fn launch<N: NetworkPolicy + 'static>(
        &self,
        config: SandboxConfig<N>,
        launch: Launch,
    ) -> Result<LocalSpawn, SpawnError> {
        let sandbox = heel::Sandbox::with_config_and_executor(config, self.inner.executor.clone())
            .await
            .map_err(|error| SpawnError::Setup(error.to_string()))?;
        let Launch {
            program,
            args,
            env,
            cwd,
            stdio,
            lifetime,
            marker,
        } = launch;
        let mut command = sandbox.command(program);
        if let Some(cwd) = cwd {
            command = command.current_dir(cwd);
        }
        let command = command
            .args(args)
            .envs(env)
            .stdin(stdio_config(stdio.stdin))
            .stdout(stdio_config(stdio.stdout))
            .stderr(stdio_config(stdio.stderr));
        let start = |error: heel::Error| SpawnError::Start(error.to_string());

        // On Linux the sandbox was built with `supervised_spawn(true)`, so the
        // command runs under `heel supervise`: a child subreaper inside the
        // sandbox that owns the whole tree however its members change
        // session, group or environment.
        #[cfg(target_os = "linux")]
        {
            let _ = marker;
            let reap_timeout = crate::terminal::SUPERVISED_REAP_TIMEOUT;
            let child = match lifetime {
                TreeLifetime::Foreground => LocalChild::Foreground(
                    command
                        .spawn_supervised(heel::supervisor::Foreground { reap_timeout })
                        .await
                        .map_err(start)?,
                ),
                TreeLifetime::Background => LocalChild::Background(
                    command
                        .spawn_supervised(heel::supervisor::Background { reap_timeout })
                        .await
                        .map_err(start)?,
                ),
            };
            let tree = JobTree::supervised(child.tree_control());
            Ok(LocalSpawn {
                child,
                tree,
                sandbox: Box::new(sandbox),
            })
        }

        // Everywhere else the root's process group is watched through the
        // process table, keyed by the job's marker.
        #[cfg(not(target_os = "linux"))]
        {
            let _ = lifetime;
            let child = command.spawn().await.map_err(start)?;
            let tree = JobTree::spawn(
                &self.inner.executor,
                self.inner.scanner.clone(),
                child.id(),
                &marker,
            );
            Ok(LocalSpawn {
                child: LocalChild::Plain(child),
                tree,
                sandbox: Box::new(sandbox),
            })
        }
    }

    pub(super) async fn spawn_host(&self, spec: SpawnSpec) -> Result<SandboxedChild, SpawnError> {
        let stdio = spec.stdio;
        let lifetime = spec.lifetime;
        let LocalSpawn {
            mut child,
            tree,
            sandbox,
        } = self.spawn_local(spec).await?;
        let id = child.id();

        let mut streams = Streams::default();
        let mut stdin_watch = None;
        if stdio.stdin == Pipe::Piped
            && let Some(stdin) = child.take_stdin()
        {
            let feed = FeedCounter::default();
            let (signal_tx, signal_rx) = async_channel::unbounded();
            match LocalStdinProbe::new(id, &stdin) {
                Ok(probe) => {
                    stdin_watch = Some(LocalStdinWatch::new(probe, feed.clone(), signal_tx));
                }
                // The watch cannot run: say so once, as a running watch would.
                Err(error) => {
                    let _ = signal_tx.try_send(StdinWatchSignal::Failed(error.to_string()));
                }
            }
            streams.stdin_blocked = Some(signal_rx);
            streams.stdin = Some(write_chunks(stdin, feed));
        }
        if stdio.stdout == Pipe::Piped
            && let Some(stdout) = child.take_stdout()
        {
            streams.stdout = Some(pump(stdout));
        }
        if stdio.stderr == Pipe::Piped
            && let Some(stderr) = child.take_stderr()
        {
            streams.stderr = Some(pump(stderr));
        }

        let (setter, watch) = exit_watch();
        let watched = tree.clone();
        self.inner
            .executor
            .spawn(async move {
                // Dropping heel's sandbox ends the processes under it.
                let _sandbox = sandbox;
                let (exit, descendants) =
                    wait_for_exit(&mut child, &watched, lifetime, stdin_watch).await;
                setter.finish(exit, descendants);
            })
            .detach();
        Ok(SandboxedChild::new(
            id,
            streams,
            KillHandle::Tree(tree),
            watch,
        ))
    }
}

/// Moves a pipe to a channel of chunks that closes at its end.
fn pump(file: std::fs::File) -> async_channel::Receiver<Vec<u8>> {
    let (tx, rx) = async_channel::unbounded();
    read_chunks(file, move |chunk| {
        if let Some(chunk) = chunk {
            let _ = tx.send_blocking(chunk);
        }
    });
    rx
}

/// Waits until the root process has exited and then ends its descendants as
/// `lifetime` says: a foreground child's are killed, a background child's
/// waited for. The root stays unreaped until its tree is gone, so its pid and
/// process group stay reserved while the job's descendants are tracked by
/// them.
async fn wait_for_exit(
    child: &mut LocalChild,
    tree: &JobTree,
    lifetime: TreeLifetime,
    stdin_watch: Option<LocalStdinWatch>,
) -> (ChildExit, Descendants) {
    let exited = match wait_for_local_process_exit(child, stdin_watch).await {
        Ok(exited) => exited,
        Err(error) => {
            tree.finish().await;
            return (
                ChildExit::Failed(error.to_string()),
                Descendants::Unobserved,
            );
        }
    };
    let mut killed = Vec::new();
    let ended = match lifetime {
        TreeLifetime::Foreground => tree
            .kill_descendants()
            .await
            .map(|processes| killed = processes),
        TreeLifetime::Background => tree.wait_descendants_exit().await,
    };
    // The task ends first: afterwards the root's pid may belong to anyone.
    tree.finish().await;
    let status = match exited {
        Some(status) => Ok(status),
        None => child
            .wait_reported()
            .await
            .map(|(status, report)| {
                // A supervised foreground job's teardown killed what it found.
                #[cfg(target_os = "linux")]
                if let Some(report) = report {
                    killed.extend(report.killed.into_iter().map(|process| JobProcess {
                        pid: process.pid,
                        name: process.name,
                    }));
                }
                #[cfg(not(target_os = "linux"))]
                let _ = report;
                status
            })
            .map_err(|error| error.to_string()),
    };
    // A kill that failed ended an unknown part of the tree, so what it did
    // name is not the whole of it; Windows sees no descendants at all.
    let descendants = if cfg!(windows) || ended.is_err() {
        Descendants::Unobserved
    } else {
        Descendants::Killed(killed)
    };
    let exit = match (ended, status) {
        (Ok(()), Ok(status)) => ChildExit::Exited(status),
        (Err(error), _) => ChildExit::Failed(error.to_string()),
        (Ok(()), Err(error)) => ChildExit::Failed(error),
    };
    (exit, descendants)
}
