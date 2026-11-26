//! The host backend: processes inside a persistent `heel` sandbox per agent.

use std::any::Any;
use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_lock::OnceCell;
use executor_core::Executor;
use heel::{
    Access, Audited, Grant, NetworkPolicy, ResolvedPath, SandboxConfig, SecurityConfig, StdioConfig,
};

use super::child::{
    ChildExit, Descendants, KillHandle, PtyResizeError, PtyResizeHandle, SandboxedChild, Streams,
    exit_watch,
};
use super::domain::DomainGate;
use super::host_ipc::{HostIpcLease, HostIpcMux, prepend_shim_path};
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

/// A process spawned on the host: the child, the handle that ends its tree,
/// an optional PTY master when spawned with [`Stdio::Pty`], and the runtime/IPC
/// guard kept alive while the process runs.
pub struct LocalSpawn {
    pub(crate) child: LocalChild,
    pub(crate) tree: JobTree,
    pub(crate) pty_master: Option<heel::PtyMaster>,
    /// Kept alive for as long as the process runs.
    pub(crate) sandbox: Box<dyn Any + Send>,
}

/// Lazily initialized persistent `heel::Sandbox` owned by one `Sandbox`.
pub struct HostRuntimeSlot {
    cell: OnceCell<Arc<HostRuntime>>,
}

impl HostRuntimeSlot {
    pub(crate) const fn new() -> Self {
        Self {
            cell: OnceCell::new(),
        }
    }
}

struct Launch {
    program: String,
    args: Vec<String>,
    env: Vec<(String, String)>,
    cwd: PathBuf,
    stdio: Stdio,
    lifetime: TreeLifetime,
    marker: JobMarker,
}

type BoxedFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, SpawnError>> + Send + 'a>>;

trait HeelSandboxOps: Send + Sync {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn handle(&self) -> heel::SandboxHandle;

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn pid_in_sandbox(&self, host_pid: u32) -> Option<u32>;

    fn set_grants_boxed(
        &self,
        grants: Vec<Grant>,
        read_denied: Vec<ResolvedPath>,
    ) -> BoxedFuture<'_, ()>;

    fn spawn_boxed<'a>(
        &'a self,
        launch: Launch,
        executor: &'a executor_core::AnyExecutor,
        scanner: crate::process_tree::ProcessScanner,
    ) -> BoxedFuture<'a, (LocalChild, JobTree, Option<heel::PtyMaster>)>;
}

impl<N: NetworkPolicy + 'static> HeelSandboxOps for heel::Sandbox<N> {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn handle(&self) -> heel::SandboxHandle {
        Self::handle(self)
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn pid_in_sandbox(&self, host_pid: u32) -> Option<u32> {
        Self::pid_in_sandbox(self, host_pid)
    }

    fn set_grants_boxed(
        &self,
        grants: Vec<Grant>,
        read_denied: Vec<ResolvedPath>,
    ) -> BoxedFuture<'_, ()> {
        Box::pin(async move {
            #[cfg(any(target_os = "macos", target_os = "linux"))]
            {
                Self::set_grants(self, grants, read_denied)
                    .await
                    .map_err(|error| SpawnError::Setup(error.to_string()))?;
            }
            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
            let _ = (grants, read_denied);
            Ok(())
        })
    }

    #[allow(clippy::too_many_lines)]
    fn spawn_boxed<'a>(
        &'a self,
        launch: Launch,
        executor: &'a executor_core::AnyExecutor,
        scanner: crate::process_tree::ProcessScanner,
    ) -> BoxedFuture<'a, (LocalChild, JobTree, Option<heel::PtyMaster>)> {
        Box::pin(async move {
            let Launch {
                program,
                args,
                env,
                cwd,
                stdio,
                lifetime,
                marker,
            } = launch;
            let start = |error: heel::Error| match error {
                heel::Error::Unsupported { operation, .. } => SpawnError::Unsupported {
                    backend: "host",
                    what: operation,
                },
                other => SpawnError::Start(other.to_string()),
            };
            match stdio {
                Stdio::Pipes {
                    stdin: stdin_pipe,
                    stdout: stdout_pipe,
                    stderr: stderr_pipe,
                } => {
                    let command = self
                        .command(program)
                        .current_dir(cwd)
                        .args(args)
                        .envs(env)
                        .stdin(stdio_config(stdin_pipe))
                        .stdout(stdio_config(stdout_pipe))
                        .stderr(stdio_config(stderr_pipe));

                    #[cfg(target_os = "linux")]
                    {
                        let _ = (executor, scanner, marker);
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
                        Ok((child, tree, None))
                    }

                    #[cfg(not(target_os = "linux"))]
                    {
                        let _ = lifetime;
                        let child = command.spawn().await.map_err(start)?;
                        let tree = JobTree::spawn(executor, scanner, child.id(), &marker);
                        Ok((LocalChild::Plain(child), tree, None))
                    }
                }
                Stdio::Pty { cols, rows } => {
                    let command = self.command(program).current_dir(cwd).args(args).envs(env);

                    #[cfg(target_os = "linux")]
                    {
                        let _ = (executor, scanner, marker);
                        let reap_timeout = crate::terminal::SUPERVISED_REAP_TIMEOUT;
                        let (child, master) = match lifetime {
                            TreeLifetime::Foreground => {
                                let (child, master) = command
                                    .spawn_supervised_pty(
                                        heel::supervisor::Foreground { reap_timeout },
                                        cols,
                                        rows,
                                    )
                                    .await
                                    .map_err(start)?;
                                (LocalChild::Foreground(child), master)
                            }
                            TreeLifetime::Background => {
                                let (child, master) = command
                                    .spawn_supervised_pty(
                                        heel::supervisor::Background { reap_timeout },
                                        cols,
                                        rows,
                                    )
                                    .await
                                    .map_err(start)?;
                                (LocalChild::Background(child), master)
                            }
                        };
                        let tree = JobTree::supervised(child.tree_control());
                        Ok((child, tree, Some(master)))
                    }

                    #[cfg(not(target_os = "linux"))]
                    {
                        let _ = lifetime;
                        let (child, master) = command.spawn_pty(cols, rows).await.map_err(start)?;
                        let tree = JobTree::spawn(executor, scanner, child.id(), &marker);
                        Ok((LocalChild::Plain(child), tree, Some(master)))
                    }
                }
            }
        })
    }
}

pub struct HostRuntime {
    sandbox: Box<dyn HeelSandboxOps>,
    mux: HostIpcMux,
    active_extras: Mutex<HashMap<(PathBuf, Access), (Grant, usize)>>,
}

struct ActiveExtrasLease {
    runtime: Arc<HostRuntime>,
    keys: Vec<(PathBuf, Access)>,
}

impl Drop for ActiveExtrasLease {
    fn drop(&mut self) {
        if self.keys.is_empty() {
            return;
        }
        let mut map = self
            .runtime
            .active_extras
            .lock()
            .expect("host runtime active_extras lock is never poisoned");
        for key in &self.keys {
            if let Some((_, count)) = map.get_mut(key) {
                *count -= 1;
                if *count == 0 {
                    map.remove(key);
                }
            }
        }
    }
}

fn grant_key(grant: &Grant) -> (PathBuf, Access) {
    let path: &Path = grant.path();
    (path.to_path_buf(), grant.access())
}

impl HostRuntime {
    fn new<N: NetworkPolicy + 'static>(sandbox: heel::Sandbox<N>, mux: HostIpcMux) -> Self {
        Self {
            sandbox: Box::new(sandbox),
            mux,
            active_extras: Mutex::new(HashMap::new()),
        }
    }

    fn acquire_extras(self: &Arc<Self>, extras: Vec<Grant>) -> ActiveExtrasLease {
        let keys: Vec<_> = extras.iter().map(grant_key).collect();
        if !extras.is_empty() {
            let mut map = self
                .active_extras
                .lock()
                .expect("host runtime active_extras lock is never poisoned");
            for (grant, key) in extras.into_iter().zip(keys.iter().cloned()) {
                map.entry(key)
                    .and_modify(|(_, count)| *count += 1)
                    .or_insert((grant, 1));
            }
            drop(map);
        }
        ActiveExtrasLease {
            runtime: Arc::clone(self),
            keys,
        }
    }

    async fn sync_grants(
        &self,
        mut grants: Vec<Grant>,
        read_denied: &[PathBuf],
    ) -> Result<(), SpawnError> {
        {
            let map = self
                .active_extras
                .lock()
                .expect("host runtime active_extras lock is never poisoned");
            grants.extend(map.values().map(|(grant, _)| grant.clone()));
        }
        let mut resolved_denied = Vec::with_capacity(read_denied.len());
        for path in read_denied {
            if matches!(path.try_exists(), Ok(false)) {
                continue;
            }
            resolved_denied
                .push(resolved(path).map_err(|error| SpawnError::Setup(error.to_string()))?);
        }
        self.sandbox.set_grants_boxed(grants, resolved_denied).await
    }
}

/// The object a spelled path names, in the spelling a resolved grant checks.
pub fn resolved(path: &Path) -> heel::Result<ResolvedPath> {
    ResolvedPath::new(std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()))
}

/// The grants that let git work in a granted linked worktree.
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

impl Sandbox {
    fn heel_binary(&self) -> Result<&heel::HeelBinary, SpawnError> {
        match &self.inner.backend {
            Backend::Host { heel } => Ok(heel),
            Backend::Container { .. } | Backend::Ssh { .. } => Err(SpawnError::Setup(
                "a local spawn needs the host backend".into(),
            )),
        }
    }

    fn canonical_working_dir(&self) -> PathBuf {
        std::fs::canonicalize(&self.inner.working_dir)
            .unwrap_or_else(|_| self.inner.working_dir.clone())
    }

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

    fn build_confined_config<N: NetworkPolicy>(
        &self,
        policy: N,
        grants: Vec<Grant>,
        snapshot: &super::GrantSnapshot,
        heel: heel::HeelBinary,
        endpoint: heel::ipc::TcpIpcEndpoint,
        #[cfg(any(target_os = "macos", target_os = "linux"))] parent: Option<heel::SandboxHandle>,
    ) -> SandboxConfig<N> {
        let builder = SandboxConfig::builder().network(policy);
        #[cfg(not(windows))]
        let builder = {
            let builder = match &self.inner.write_denylist {
                Some(denylist) => builder.write_denylist(denylist.clone()),
                None => builder,
            };
            #[cfg(target_os = "macos")]
            let initial_denied: &[PathBuf] = {
                let _ = snapshot;
                &[]
            };
            #[cfg(not(target_os = "macos"))]
            let initial_denied: &[PathBuf] = &snapshot.read_denied;
            builder
                .filesystem_strict(false)
                .read_denylist(read_denylist(initial_denied))
        };
        #[cfg(windows)]
        let _ = snapshot;
        let builder = builder
            .working_dir(self.canonical_working_dir())
            .grants(grants)
            .heel_binary(heel)
            .security(SecurityConfig::interactive())
            .executable_working_dir(true)
            .priority(self.inner.priority);
        #[cfg(target_os = "linux")]
        let builder = builder.supervised_spawn(true);
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        let builder = match parent {
            Some(parent) => builder.parent(parent),
            None => builder,
        };
        builder.external_ipc(endpoint).build()
    }

    fn build_unconfined_config(
        &self,
        heel: heel::HeelBinary,
        endpoint: heel::ipc::TcpIpcEndpoint,
        #[cfg(any(target_os = "macos", target_os = "linux"))] parent: Option<heel::SandboxHandle>,
    ) -> SandboxConfig<heel::HostAccess> {
        let builder = SandboxConfig::builder()
            .working_dir(self.canonical_working_dir())
            .host_access()
            .heel_binary(heel)
            .executable_working_dir(true)
            .priority(self.inner.priority);
        #[cfg(target_os = "linux")]
        let builder = builder.supervised_spawn(true);
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        let builder = match parent {
            Some(parent) => builder.parent(parent),
            None => builder,
        };
        builder.external_ipc(endpoint).build()
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn ensure_parent_handle(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<heel::SandboxHandle, SpawnError>> + Send + '_>> {
        Box::pin(async move {
            let runtime = self.ensure_host_runtime().await?;
            Ok(runtime.sandbox.handle())
        })
    }

    async fn ensure_host_runtime(&self) -> Result<Arc<HostRuntime>, SpawnError> {
        let heel = self.heel_binary()?.clone();
        if self.inner.unconfined {
            return self
                .inner
                .unconfined_host
                .cell
                .get_or_try_init(|| async {
                    #[cfg(any(target_os = "macos", target_os = "linux"))]
                    let parent = match &self.inner.parent {
                        Some(parent) => Some(parent.ensure_parent_handle().await?),
                        None => None,
                    };
                    let mux = HostIpcMux::start(&self.inner.executor)?;
                    let config = self.build_unconfined_config(
                        heel,
                        mux.endpoint(),
                        #[cfg(any(target_os = "macos", target_os = "linux"))]
                        parent,
                    );
                    let sandbox = heel::Sandbox::with_config_and_executor(
                        config,
                        self.inner.executor.clone(),
                    )
                    .await
                    .map_err(|error| SpawnError::Setup(error.to_string()))?;
                    Ok(Arc::new(HostRuntime::new(sandbox, mux)))
                })
                .await
                .cloned();
        }

        self.inner
            .confined_host
            .cell
            .get_or_try_init(|| async {
                #[cfg(any(target_os = "macos", target_os = "linux"))]
                let parent = match &self.inner.parent {
                    Some(parent) => Some(parent.ensure_parent_handle().await?),
                    None => None,
                };
                let snapshot = self.inner.grants.snapshot();
                #[cfg(target_os = "macos")]
                let grants = self.inner.fixed_grants.clone();
                #[cfg(not(target_os = "macos"))]
                let grants = self.confined_grants(&snapshot, Vec::new()).await?;
                let mux = HostIpcMux::start(&self.inner.executor)?;
                let gate = DomainGate {
                    lists: self.inner.network.clone(),
                    approver: self.inner.approver.clone(),
                    agent: self.inner.agent.clone(),
                };
                if let Some(log) = self.inner.audit.clone() {
                    let config = self.build_confined_config(
                        Audited::new(gate, log),
                        grants,
                        &snapshot,
                        heel,
                        mux.endpoint(),
                        #[cfg(any(target_os = "macos", target_os = "linux"))]
                        parent,
                    );
                    let sandbox = heel::Sandbox::with_config_and_executor(
                        config,
                        self.inner.executor.clone(),
                    )
                    .await
                    .map_err(|error| SpawnError::Setup(error.to_string()))?;
                    Ok(Arc::new(HostRuntime::new(sandbox, mux)))
                } else {
                    let config = self.build_confined_config(
                        gate,
                        grants,
                        &snapshot,
                        heel,
                        mux.endpoint(),
                        #[cfg(any(target_os = "macos", target_os = "linux"))]
                        parent,
                    );
                    let sandbox = heel::Sandbox::with_config_and_executor(
                        config,
                        self.inner.executor.clone(),
                    )
                    .await
                    .map_err(|error| SpawnError::Setup(error.to_string()))?;
                    Ok(Arc::new(HostRuntime::new(sandbox, mux)))
                }
            })
            .await
            .cloned()
    }

    /// Maps `host_pid` into the PID visible inside this host sandbox's
    /// execution environment, or `None` when `host_pid` is outside it.
    ///
    /// # Errors
    /// Returns an error if the host sandbox is not configured or fails to start.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    pub async fn pid_in_sandbox(&self, host_pid: u32) -> Result<Option<u32>, SpawnError> {
        let runtime = self.ensure_host_runtime().await?;
        Ok(runtime.sandbox.pid_in_sandbox(host_pid))
    }

    /// Resolves a live `sandbox_pid` inside this host sandbox back to its host OS PID.
    #[cfg(all(test, unix))]
    pub(crate) async fn resolve_host_pid(&self, sandbox_pid: u32) -> Option<u32> {
        if self.inner.unconfined {
            return Some(sandbox_pid);
        }
        let runtime = self.ensure_host_runtime().await.ok()?;
        #[cfg(target_os = "macos")]
        {
            runtime.sandbox.pid_in_sandbox(sandbox_pid)
        }
        #[cfg(target_os = "linux")]
        {
            for entry in std::fs::read_dir("/proc").ok()?.flatten() {
                if let Some(host_pid) = entry
                    .file_name()
                    .to_str()
                    .and_then(|name| name.parse::<u32>().ok())
                    && runtime.sandbox.pid_in_sandbox(host_pid) == Some(sandbox_pid)
                {
                    return Some(host_pid);
                }
            }
            None
        }
    }

    /// Starts `spec` on the host under this sandbox's persistent `heel::Sandbox`.
    pub(crate) async fn spawn_local(&self, spec: SpawnSpec) -> Result<LocalSpawn, SpawnError> {
        let _ = self.heel_binary()?;
        let SpawnSpec {
            command,
            env,
            cwd,
            stdio,
            grants: extra_grants,
            ipc: router,
            lifetime,
            info,
        } = spec;
        let SpawnCommand::Program { program, args } = command else {
            return Err(SpawnError::Unsupported {
                backend: "host",
                what: "a script",
            });
        };
        let mut extra_grants = extra_grants
            .into_iter()
            .map(|grant| match grant {
                SpawnGrant::Host(grant) => Ok(grant),
                SpawnGrant::Container { .. } => Err(SpawnError::Unsupported {
                    backend: "host",
                    what: "a grant on a container path",
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let runtime = self.ensure_host_runtime().await?;
        let mut env = self.host_environment(env, &info)?;
        let ipc_lease: Option<HostIpcLease> = match router {
            Some(router) => {
                let lease = runtime.mux.register(router).await?;
                let resolved_shim = resolved(lease.shim_dir())
                    .map_err(|error| SpawnError::Setup(error.to_string()))?;
                prepend_shim_path(&mut env, resolved_shim.as_ref());
                extra_grants.push(Grant::new(resolved_shim, Access::READ | Access::EXEC));
                Some(lease)
            }
            None => None,
        };

        let effective_cwd = cwd.map_or_else(
            || self.canonical_working_dir(),
            |cwd| std::fs::canonicalize(&cwd).unwrap_or(cwd),
        );
        let extras_lease = if self.inner.unconfined {
            None
        } else {
            if matches!(effective_cwd.try_exists(), Ok(true)) {
                let resolved_cwd = resolved(&effective_cwd)
                    .map_err(|error| SpawnError::Setup(error.to_string()))?;
                extra_grants.push(Grant::new(resolved_cwd, Access::READ | Access::EXEC));
            }
            let lease = runtime.acquire_extras(extra_grants);
            let snapshot = self.inner.grants.snapshot();
            let desired = self.confined_grants(&snapshot, Vec::new()).await?;
            runtime.sync_grants(desired, &snapshot.read_denied).await?;
            Some(lease)
        };

        let marker = JobMarker::new();
        env.push(marker.entry());
        let launch = Launch {
            program,
            args,
            env,
            cwd: effective_cwd,
            stdio,
            lifetime,
            marker,
        };
        let (child, tree, pty_master) = runtime
            .sandbox
            .spawn_boxed(launch, &self.inner.executor, self.inner.scanner.clone())
            .await?;

        Ok(LocalSpawn {
            child,
            tree,
            pty_master,
            sandbox: Box::new((runtime, ipc_lease, extras_lease)),
        })
    }

    pub(super) async fn spawn_host(&self, spec: SpawnSpec) -> Result<SandboxedChild, SpawnError> {
        let stdio = spec.stdio;
        let lifetime = spec.lifetime;
        let LocalSpawn {
            mut child,
            tree,
            pty_master,
            sandbox,
        } = self.spawn_local(spec).await?;
        let sandbox_id = child.sandbox_id();
        let host_id = child.id();

        let mut streams = Streams::default();
        let mut stdin_watch = None;
        match stdio {
            Stdio::Pipes {
                stdin: stdin_pipe,
                stdout: stdout_pipe,
                stderr: stderr_pipe,
            } => {
                if stdin_pipe == Pipe::Piped
                    && let Some(stdin) = child.take_stdin()
                {
                    let feed = FeedCounter::default();
                    let (signal_tx, signal_rx) = async_channel::unbounded();
                    match LocalStdinProbe::new(host_id, &stdin) {
                        Ok(probe) => {
                            stdin_watch =
                                Some(LocalStdinWatch::new(probe, feed.clone(), signal_tx));
                        }
                        Err(error) => {
                            let _ = signal_tx.try_send(StdinWatchSignal::Failed(error.to_string()));
                        }
                    }
                    streams.stdin_blocked = Some(signal_rx);
                    streams.stdin = Some(write_chunks(stdin, feed));
                }
                if stdout_pipe == Pipe::Piped
                    && let Some(stdout) = child.take_stdout()
                {
                    streams.stdout = Some(pump(stdout));
                }
                if stderr_pipe == Pipe::Piped
                    && let Some(stderr) = child.take_stderr()
                {
                    streams.stderr = Some(pump(stderr));
                }
            }
            Stdio::Pty { .. } => {
                let master = pty_master.ok_or_else(|| {
                    SpawnError::Start("host PTY spawn returned no master handle".to_string())
                })?;
                let read_file = master
                    .try_clone_file()
                    .map_err(|error| SpawnError::Start(error.to_string()))?;
                let write_file = master
                    .try_clone_file()
                    .map_err(|error| SpawnError::Start(error.to_string()))?;
                let master = Arc::new(master);
                streams.stdout = Some(pump(read_file));
                streams.stdin = Some(write_chunks(write_file, FeedCounter::default()));
                streams.pty_resize = Some(PtyResizeHandle::sync(move |cols, rows| {
                    master
                        .resize(cols, rows)
                        .map_err(|error| PtyResizeError::Backend(error.to_string()))
                }));
            }
        }

        let (setter, watch) = exit_watch();
        let watched = tree.clone();
        self.inner
            .executor
            .spawn(async move {
                let _sandbox = sandbox;
                let (exit, descendants) =
                    wait_for_exit(&mut child, &watched, lifetime, stdin_watch).await;
                setter.finish(exit, descendants);
            })
            .detach();
        Ok(SandboxedChild::with_host_id(
            sandbox_id,
            host_id,
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
/// `lifetime` says.
async fn wait_for_exit(
    child: &mut LocalChild,
    tree: &JobTree,
    lifetime: TreeLifetime,
    mut stdin_watch: Option<LocalStdinWatch>,
) -> (ChildExit, Descendants) {
    let (status, report) = match wait_for_local_process_exit(child, stdin_watch.as_mut()).await {
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
        TreeLifetime::Background => match stdin_watch.as_mut() {
            Some(watch) => {
                futures_lite::future::or(tree.wait_descendants_exit(), async {
                    match watch.run_borrowed().await {}
                })
                .await
            }
            None => tree.wait_descendants_exit().await,
        },
    };
    tree.finish().await;
    #[cfg(target_os = "linux")]
    if let Some(report) = report {
        killed.extend(report.killed.into_iter().map(|process| JobProcess {
            pid: process.pid,
            name: process.name,
        }));
    }
    #[cfg(not(target_os = "linux"))]
    let _ = report;
    let descendants = if cfg!(windows) || ended.is_err() {
        Descendants::Unobserved
    } else {
        Descendants::Killed(killed)
    };
    let exit = match ended {
        Ok(()) => ChildExit::Exited(status),
        Err(error) => ChildExit::Failed(error.to_string()),
    };
    (exit, descendants)
}
