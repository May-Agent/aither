//! The spawn layer: start a process under a sandbox.
//!
//! A [`Sandbox`] is one per home environment. It holds the live [`Grants`],
//! the environment, the network lists and their [`DomainApprover`], and a
//! [`Backend`] saying where processes run. [`Sandbox::spawn`] starts a
//! [`SpawnSpec`] on that backend and answers a [`SandboxedChild`] whose
//! [`kill`](SandboxedChild::kill) ends the whole process tree.
//!
//! What the sandbox enforces per backend:
//!
//! - [`Backend::Host`]: heel's sandbox, with the grants, the environment, the
//!   write and read denylists, the process priority and the network lists
//!   with their approver, whose answer is enforced per connection.
//! - [`Backend::Container`]: the process runs in the container, which is its
//!   boundary. A [derived](Sandbox::derived) sandbox, a child's, is
//!   additionally confined inside the container to its own writable grants.
//!   The environment reaches it as exports of the script.
//! - [`Backend::Ssh`]: the process runs on the server under the server's own
//!   `heel`, with the tools the embedder prepared there. Grants and the
//!   host-side environment never reach another machine.

mod backend;
mod child;
mod container;
mod domain;
mod env;
mod grants;
pub mod host;
mod launch;
mod route;
mod ssh;
pub mod stdio;
#[cfg(all(test, unix))]
mod tests;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use executor_core::{AnyExecutor, Executor};

pub use backend::Backend;
pub use child::{ChildExit, SandboxedChild};
pub use domain::{
    AnyDomainApprover, ApproveAllDomains, DenyAllDomains, DomainApprover, DomainRequest,
};
pub use env::EnvSource;
pub use grants::{GitDirAccess, GrantSnapshot, Grants};
pub use host::{LocalSpawn, TreeLifetime, resolved};
pub use launch::{ContainerLaunch, RemoteLaunch, shell_arguments};
pub use route::{HostRoute, HostRouteError};
pub use ssh::remote_command;

use crate::network_policy::CompiledNetworkPolicy;
use crate::process_tree::ProcessScanner;

/// Why a process could not be spawned.
#[derive(Debug, thiserror::Error)]
pub enum SpawnError {
    /// The sandbox, or the backend it runs on, could not be set up: a grant
    /// cannot be resolved, the heel binary is unusable, the server has no
    /// `heel`, a path has no counterpart in the container.
    #[error("sandbox setup failed: {0}")]
    Setup(String),
    /// The process could not be started.
    #[error("failed to start the process: {0}")]
    Start(String),
}

/// Whether a standard stream of a spawned process is piped to its handle.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Pipe {
    /// Connected to the null device: no input, or output dropped.
    #[default]
    Null,
    /// Connected to the [`SandboxedChild`]'s channel for the stream.
    Piped,
}

/// The standard streams of a spawned process. The default pipes `stdout` and
/// `stderr` and gives `stdin` nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stdio {
    /// The process's standard input.
    pub stdin: Pipe,
    /// The process's standard output.
    pub stdout: Pipe,
    /// The process's standard error.
    pub stderr: Pipe,
}

impl Default for Stdio {
    fn default() -> Self {
        Self {
            stdin: Pipe::Null,
            stdout: Pipe::Piped,
            stderr: Pipe::Piped,
        }
    }
}

/// A process to start under a [`Sandbox`].
pub struct SpawnSpec {
    pub(crate) program: String,
    pub(crate) args: Vec<String>,
    pub(crate) env: Vec<(String, String)>,
    pub(crate) cwd: Option<PathBuf>,
    pub(crate) stdio: Stdio,
    /// Grants this one process needs beyond the sandbox's (a command's
    /// jobserver fifo).
    pub(crate) extra_grants: Vec<heel::Grant>,
    /// The commands the process may call back into the host with.
    pub(crate) ipc: Option<heel::IpcRouter>,
}

impl std::fmt::Debug for SpawnSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpawnSpec")
            .field("program", &self.program)
            .field("args", &self.args)
            .field("cwd", &self.cwd)
            .field("stdio", &self.stdio)
            .finish_non_exhaustive()
    }
}

impl SpawnSpec {
    /// A process running `program` with no arguments, in the sandbox's
    /// working directory, with the default [`Stdio`].
    #[must_use]
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env: Vec::new(),
            cwd: None,
            stdio: Stdio::default(),
            extra_grants: Vec::new(),
            ipc: None,
        }
    }

    /// Appends arguments.
    #[must_use]
    pub fn args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Adds environment for this process alone; it overrides the sandbox's
    /// environment and its [`EnvSource`].
    #[must_use]
    pub fn env(mut self, env: impl IntoIterator<Item = (String, String)>) -> Self {
        self.env.extend(env);
        self
    }

    /// Starts the process in `cwd`, a path on the host the sandbox runs on
    /// (for a container, one under a [mount](crate::ContainerShellRuntime::with_workspace_mount)).
    #[must_use]
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Sets the standard streams.
    #[must_use]
    pub const fn stdio(mut self, stdio: Stdio) -> Self {
        self.stdio = stdio;
        self
    }
}

#[derive(Clone)]
struct SandboxInner {
    backend: Backend,
    executor: Arc<AnyExecutor>,
    working_dir: PathBuf,
    grants: Grants,
    fixed_grants: Vec<heel::Grant>,
    environment: Vec<(String, String)>,
    env_source: Option<Arc<dyn EnvSource>>,
    write_denylist: Option<heel::WriteDenylist>,
    network: Option<CompiledNetworkPolicy>,
    approver: Option<AnyDomainApprover>,
    audit: Option<heel::NetworkAuditLog>,
    priority: heel::ProcessPriority,
    agent: Option<Arc<str>>,
    scanner: ProcessScanner,
    unconfined: bool,
    ssh_client: crate::ssh::SshClient,
}

/// One home environment's sandbox: grants, environment, network policy and
/// the [`Backend`] processes run on. Cheap to clone; clones share the live
/// grants.
#[derive(Clone)]
pub struct Sandbox {
    inner: Arc<SandboxInner>,
}

impl std::fmt::Debug for Sandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sandbox")
            .field("working_dir", &self.inner.working_dir)
            .field("agent", &self.inner.agent)
            .finish_non_exhaustive()
    }
}

impl Sandbox {
    /// A sandbox on `backend` whose processes start in `working_dir`, with no
    /// grants, no environment of its own and no network lists: every domain
    /// is open to the approver, which denies it until
    /// [`with_domain_approver`](Self::with_domain_approver) gives one.
    ///
    /// `executor` runs the tasks that watch spawned processes and heel's
    /// proxy.
    #[must_use]
    pub fn new(
        backend: Backend,
        executor: impl Executor + 'static,
        working_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            inner: Arc::new(SandboxInner {
                backend,
                executor: Arc::new(AnyExecutor::new(executor)),
                working_dir: working_dir.into(),
                grants: Grants::new(),
                fixed_grants: Vec::new(),
                environment: Vec::new(),
                env_source: None,
                write_denylist: None,
                network: None,
                approver: None,
                audit: None,
                priority: heel::ProcessPriority::default(),
                agent: None,
                scanner: ProcessScanner::default(),
                unconfined: false,
                ssh_client: crate::ssh::SshClient::default(),
            }),
        }
    }

    fn edit(mut self, edit: impl FnOnce(&mut SandboxInner)) -> Self {
        edit(Arc::make_mut(&mut self.inner));
        self
    }

    /// Uses `grants` as the sandbox's live grants; the sandbox sees every
    /// change made through any clone of the handle.
    #[must_use]
    pub fn with_grants(self, grants: Grants) -> Self {
        self.edit(|inner| inner.grants = grants)
    }

    /// Heel grants every spawn on the host carries ahead of the live
    /// [`Grants`]: the directories the embedder's tools and toolchains live
    /// in, and what its `PATH` lists.
    #[must_use]
    pub fn with_fixed_grants(self, grants: Vec<heel::Grant>) -> Self {
        self.edit(|inner| inner.fixed_grants = grants)
    }

    /// The environment every spawn on the host starts with: the session's
    /// `HOME`, temporary and cache directories, `PATH` and toolchain
    /// variables. It never reaches a container or an SSH server, whose
    /// filesystems differ.
    #[must_use]
    pub fn with_environment(self, environment: Vec<(String, String)>) -> Self {
        self.edit(|inner| inner.environment = environment)
    }

    /// Environment resolved at each spawn, on the host and in a container;
    /// never on an SSH server, so host-side values such as secrets never
    /// reach another machine.
    #[must_use]
    pub fn with_env_source(self, source: impl EnvSource + 'static) -> Self {
        self.edit(|inner| inner.env_source = Some(Arc::new(source)))
    }

    /// The paths a spawn on the host must not write, however they are
    /// granted. Not supported on Windows, whose backend refuses a denylist
    /// rather than run weaker, so it is not applied there.
    #[must_use]
    pub fn with_write_denylist(self, denylist: heel::WriteDenylist) -> Self {
        self.edit(|inner| inner.write_denylist = Some(denylist))
    }

    /// The network lists: a host they allow or deny is decided without the
    /// approver.
    #[must_use]
    pub fn with_network_policy(self, policy: CompiledNetworkPolicy) -> Self {
        self.edit(|inner| inner.network = Some(policy))
    }

    /// Decides the connections the network lists leave open.
    #[must_use]
    pub fn with_domain_approver(self, approver: impl DomainApprover + 'static) -> Self {
        self.edit(|inner| inner.approver = Some(AnyDomainApprover::new(approver)))
    }

    /// Records every network decision of a spawn on the host in `log`.
    #[must_use]
    pub fn with_audit_log(self, log: heel::NetworkAuditLog) -> Self {
        self.edit(|inner| inner.audit = Some(log))
    }

    /// Lowers the scheduling priority of what the sandbox spawns on the host,
    /// inherited by everything it starts. A container's image needs its own
    /// `heel` for a non-normal priority; an SSH server's priority is its own.
    #[must_use]
    pub fn with_priority(self, priority: heel::ProcessPriority) -> Self {
        self.edit(|inner| inner.priority = priority)
    }

    /// Labels the agent this sandbox serves; the label is the `agent` of
    /// every [`DomainRequest`].
    #[must_use]
    pub fn with_agent(self, agent: impl Into<String>) -> Self {
        let agent: Arc<str> = Arc::from(agent.into());
        self.edit(|inner| inner.agent = Some(agent))
    }

    /// A child's sandbox: the same backend, environment and network policy,
    /// grants [derived](Grants::derived) from this sandbox's with `own`, and
    /// no agent label until [`with_agent`](Self::with_agent) gives one. It
    /// cannot widen this sandbox: its grants never reach it, and a container
    /// confines its processes to its own writable grants.
    #[must_use]
    pub fn derived(&self, own: GrantSnapshot) -> Self {
        let grants = self.inner.grants.derived(own);
        self.clone().edit(|inner| {
            inner.grants = grants;
            inner.agent = None;
        })
    }

    /// The sandbox's live grants.
    #[must_use]
    pub fn grants(&self) -> &Grants {
        &self.inner.grants
    }

    /// The directory processes start in unless their spec says otherwise.
    #[must_use]
    pub fn working_dir(&self) -> &Path {
        &self.inner.working_dir
    }

    /// The executor the sandbox's tasks run on.
    pub(crate) fn executor(&self) -> Arc<AnyExecutor> {
        Arc::clone(&self.inner.executor)
    }

    /// Where the sandbox's processes run.
    pub(crate) fn backend(&self) -> &Backend {
        &self.inner.backend
    }

    /// The embedder's environment source.
    pub(crate) fn env_source(&self) -> Option<Arc<dyn EnvSource>> {
        self.inner.env_source.clone()
    }

    /// The network lists, their approver and the audit log.
    pub(crate) fn network(
        &self,
    ) -> (
        Option<CompiledNetworkPolicy>,
        Option<AnyDomainApprover>,
        Option<heel::NetworkAuditLog>,
    ) {
        (
            self.inner.network.clone(),
            self.inner.approver.clone(),
            self.inner.audit.clone(),
        )
    }

    /// The scheduling priority of the sandbox's processes.
    pub(crate) fn priority(&self) -> heel::ProcessPriority {
        self.inner.priority
    }

    /// The same sandbox without confinement: a host spawn takes host access,
    /// ignoring the grants and denylists and the network lists.
    pub(crate) fn unconfined(&self) -> Self {
        self.clone().edit(|inner| inner.unconfined = true)
    }

    /// Reads process tables through `scanner`, shared with the embedder's
    /// job registry.
    pub(crate) fn with_scanner(self, scanner: ProcessScanner) -> Self {
        self.edit(|inner| inner.scanner = scanner)
    }

    /// Reaches SSH servers through `client` instead of the default one.
    #[cfg(test)]
    pub(crate) fn with_ssh_client(self, client: crate::ssh::SshClient) -> Self {
        self.edit(|inner| inner.ssh_client = client)
    }

    /// Starts `spec` on the sandbox's backend, under its grants and network
    /// policy as they are now: a change made to the live grants applies to
    /// the next spawn, never to a process already running.
    ///
    /// # Errors
    /// Returns an error when the sandbox cannot be set up for the spawn or
    /// the process cannot be started.
    pub async fn spawn(&self, spec: SpawnSpec) -> Result<SandboxedChild, SpawnError> {
        match &self.inner.backend {
            Backend::Host { .. } => self.spawn_host(spec).await,
            Backend::Container { runtime, launch } => self.spawn_container(runtime, launch, spec),
            Backend::Ssh {
                server,
                heel,
                launch,
            } => self.spawn_ssh(server, heel.as_deref(), launch, spec),
        }
    }
}
