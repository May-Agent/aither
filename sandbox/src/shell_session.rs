use std::{future::Future, pin::Pin, sync::Arc};

use async_channel::{Receiver, Sender};

use askama::Template as _;

use crate::ssh::{SshEnvironments, SshServer, SshServerError};
use crate::stdin_watch::StdinWatchSignal;
use crate::{ContainerPath, HostRoute};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// How long the processes a container command leaves running when its
/// top-level shell exits may live.
#[derive(Debug, Clone)]
pub enum ExecLifetime {
    /// None: they are killed with the shell and reported in
    /// [`ContainerExecOutcome::Completed`]'s `killed`. Its stdin is not
    /// watched.
    Foreground,
    /// All of them: the execution ends with its last process. Its stdin is
    /// watched: implementations send one [`StdinWatchSignal::Blocked`] each
    /// time a process of the command's tree starts waiting to read the stdin
    /// pipe with nothing buffered — and send again only after the command was
    /// fed and blocks anew — or a single [`StdinWatchSignal::Failed`] when the
    /// watch cannot run at all.
    Background {
        /// Where the stdin watch reports.
        stdin_blocked: Sender<StdinWatchSignal>,
    },
}

/// Outcome of a container execution request.
#[derive(Debug)]
pub enum ContainerExecOutcome {
    /// The command ran to completion.
    Completed {
        /// Its exit status and output.
        output: std::process::Output,
        /// The processes it left running that an [`ExecLifetime::Foreground`]
        /// execution killed; empty for any other.
        killed: Vec<crate::JobProcess>,
    },
    /// The command was terminated before it finished.
    Killed,
}

/// Trait for executing commands inside a container.
///
/// Implementations provide the bridge between aither-sandbox and a specific
/// container runtime (e.g., Docker via bollard). The framework defines this trait;
/// the application (may) provides the implementation.
pub trait ContainerExec: Send + Sync {
    /// Execute a terminal command payload inside the container, returning stdout/stderr and exit code.
    ///
    /// `stdin_rx` is the command's stdin; a closed channel is EOF. When
    /// `stdin_blocked` is given, implementations send one
    /// [`StdinWatchSignal::Blocked`] each time a process of the command's
    /// tree starts waiting to read the stdin pipe with nothing buffered —
    /// and send again only after the command was fed and blocks anew — or a
    /// single [`StdinWatchSignal::Failed`] when the watch cannot run at all.
    fn exec(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        lifetime: ExecLifetime,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send;

    /// Executes a command like [`Self::exec`], handing its stdout on chunk by
    /// chunk through `stdout` as it arrives instead of collecting it.
    ///
    /// The returned outcome carries the exit status and stderr; its stdout is
    /// empty. `stdout` is bounded, so a consumer that stops reading holds the
    /// command back. The command is killed, as when `kill_rx` fires, once
    /// `kill_rx` fires or closes, or once `stdout` has no receiver left.
    /// `lifetime` is [`Self::exec`]'s.
    #[allow(clippy::too_many_arguments)]
    fn exec_streamed(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdout: Sender<Vec<u8>>,
        lifetime: ExecLifetime,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send;
}

/// Object-safe twin of [`ContainerExec`], used only through
/// [`AnyContainerExec`].
pub trait ContainerExecImpl: Send + Sync {
    /// Execute a command inside a container through a boxed future.
    fn exec_boxed<'a>(
        &'a self,
        container_id: &'a str,
        script: &'a str,
        working_dir: &'a str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        lifetime: ExecLifetime,
    ) -> Pin<Box<dyn Future<Output = Result<ContainerExecOutcome, String>> + Send + 'a>>;

    /// Execute a command inside a container, streaming its stdout, through a
    /// boxed future.
    #[allow(clippy::too_many_arguments)]
    fn exec_streamed_boxed<'a>(
        &'a self,
        container_id: &'a str,
        script: &'a str,
        working_dir: &'a str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdout: Sender<Vec<u8>>,
        lifetime: ExecLifetime,
    ) -> Pin<Box<dyn Future<Output = Result<ContainerExecOutcome, String>> + Send + 'a>>;
}

/// Backend-agnostic container executor handle.
///
/// This wraps any [`ContainerExec`] implementation into a cloneable concrete
/// type that can be passed around without exposing runtime-specific concrete
/// executor types.
#[derive(Clone)]
pub struct AnyContainerExec {
    inner: Arc<dyn ContainerExecImpl>,
}

impl std::fmt::Debug for AnyContainerExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyContainerExec").finish_non_exhaustive()
    }
}

impl AnyContainerExec {
    /// Creates a new handle from a concrete container executor implementation.
    #[must_use]
    pub fn new<T>(inner: Arc<T>) -> Self
    where
        T: ContainerExec + 'static,
    {
        Self { inner }
    }
}

impl ContainerExec for AnyContainerExec {
    fn exec(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        lifetime: ExecLifetime,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
        let inner = Arc::clone(&self.inner);
        let container_id = container_id.to_string();
        let script = script.to_string();
        let working_dir = working_dir.to_string();

        async move {
            inner
                .exec_boxed(
                    &container_id,
                    &script,
                    &working_dir,
                    kill_rx,
                    stdin_rx,
                    lifetime,
                )
                .await
        }
    }

    fn exec_streamed(
        &self,
        container_id: &str,
        script: &str,
        working_dir: &str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdout: Sender<Vec<u8>>,
        lifetime: ExecLifetime,
    ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
        let inner = Arc::clone(&self.inner);
        let container_id = container_id.to_string();
        let script = script.to_string();
        let working_dir = working_dir.to_string();

        async move {
            inner
                .exec_streamed_boxed(
                    &container_id,
                    &script,
                    &working_dir,
                    kill_rx,
                    stdin_rx,
                    stdout,
                    lifetime,
                )
                .await
        }
    }
}

/// Container runtime session metadata.
#[derive(Clone)]
pub struct ContainerShellRuntime {
    exec: Arc<dyn ContainerExecImpl>,
    container_id: String,
    host_route: HostRoute,
    /// The host directory the container mounts as its workspace, declared by
    /// whoever mounts it.
    workspace_mount: Option<crate::MountRoot>,
}

impl std::fmt::Debug for ContainerShellRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContainerShellRuntime")
            .field("container_id", &self.container_id)
            .field("host_route", &self.host_route)
            .field("workspace_mount", &self.workspace_mount)
            .finish_non_exhaustive()
    }
}

impl ContainerShellRuntime {
    /// Creates a container shell runtime.
    ///
    /// # Panics
    /// Panics if `container_id` is empty after trimming.
    #[must_use]
    pub fn new(
        container_id: impl Into<String>,
        host_route: HostRoute,
        exec: AnyContainerExec,
    ) -> Self {
        let container_id = container_id.into();
        assert!(
            !container_id.trim().is_empty(),
            "container runtime requires a non-empty container_id"
        );
        Self {
            exec: exec.inner,
            container_id,
            host_route,
            workspace_mount: None,
        }
    }

    /// Declares the host directory the container mounts as its workspace and
    /// where it appears inside. A subagent's commands are confined in the
    /// container through it, so a container session without it cannot launch
    /// subagents.
    #[must_use]
    pub fn with_workspace_mount(mut self, mount: crate::MountRoot) -> Self {
        self.workspace_mount = Some(mount);
        self
    }

    /// The workspace mount, when declared.
    #[must_use]
    pub const fn workspace_mount(&self) -> Option<&crate::MountRoot> {
        self.workspace_mount.as_ref()
    }

    #[must_use]
    /// Returns the container id.
    pub fn container_id(&self) -> &str {
        &self.container_id
    }

    /// Returns where the host is reachable from inside the container.
    #[must_use]
    pub const fn host_route(&self) -> &HostRoute {
        &self.host_route
    }

    pub(crate) fn exec(&self) -> Arc<dyn ContainerExecImpl> {
        Arc::clone(&self.exec)
    }

    /// A reader of the container's own state: what a harness that watches
    /// the container's memory and disk asks it.
    #[must_use]
    pub fn probe(&self) -> ContainerProbe {
        ContainerProbe::from_parts(self.exec(), self.container_id.clone())
    }
}

/// Why a probe of a container failed.
#[derive(Debug, thiserror::Error)]
pub enum ContainerProbeError {
    /// The container could not run the script.
    #[error("the container could not run the probe: {0}")]
    Exec(String),
    /// The script ran and failed.
    #[error("the probe exited with {status}: {stderr}")]
    Failed {
        /// The exit status it returned.
        status: std::process::ExitStatus,
        /// What it wrote to its standard error.
        stderr: String,
    },
    /// The container killed the probe before it finished.
    #[error("the container killed the probe")]
    Killed,
    /// The script answered bytes that are not text.
    #[error("the probe answered bytes that are not UTF-8")]
    NotText,
    /// The script answered text that is not what it was asked for.
    #[error("the probe answered `{0}`, which is not understood")]
    Unreadable(String),
}

/// The free bytes `df -Pk` reports: the `Available` column, in KiB, of its
/// last line.
fn free_bytes_from_df(output: &str) -> Option<u64> {
    let kib: u64 = output
        .lines()
        .last()?
        .split_whitespace()
        .nth(3)?
        .parse()
        .ok()?;
    kib.checked_mul(1024)
}

/// Runs short read-only scripts inside a container and returns what they
/// print, through the same exec path commands use.
#[derive(Clone)]
pub struct ContainerProbe {
    exec: Arc<dyn ContainerExecImpl>,
    container_id: String,
    /// The directory the agent works in inside the container.
    home: ContainerPath,
}

/// `df` over the filesystem holding a path, in POSIX format.
#[derive(askama::Template)]
#[template(path = "container_disk_free.sh", escape = "none")]
struct DiskFreeScript<'a> {
    /// The path, quoted for the shell.
    path: std::borrow::Cow<'a, str>,
}

impl std::fmt::Debug for ContainerProbe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContainerProbe")
            .field("container_id", &self.container_id)
            .finish_non_exhaustive()
    }
}

impl ContainerProbe {
    /// A probe of container `container_id` through `exec`.
    ///
    /// The agent's home directory in the container is the one commands run
    /// in until [`Self::with_home`] says otherwise.
    #[must_use]
    pub fn new(exec: &AnyContainerExec, container_id: impl Into<String>) -> Self {
        Self::from_parts(Arc::clone(&exec.inner), container_id.into())
    }

    fn from_parts(exec: Arc<dyn ContainerExecImpl>, container_id: String) -> Self {
        Self {
            exec,
            container_id,
            home: ContainerPath::new(crate::terminal::CONTAINER_EXEC_WORKING_DIR)
                .expect("the container working directory is absolute"),
        }
    }

    /// Names the directory the agent works in inside the container.
    #[must_use]
    pub fn with_home(mut self, home: ContainerPath) -> Self {
        self.home = home;
        self
    }

    /// The directory the agent works in inside the container.
    #[must_use]
    pub const fn home(&self) -> &ContainerPath {
        &self.home
    }

    /// The bytes free to the container's processes on the filesystem that
    /// holds the agent's home directory, as that filesystem reports them:
    /// bounded by its own size limit when the container has one.
    ///
    /// # Errors
    ///
    /// Fails when the container cannot run `df` or answers something else.
    ///
    /// # Panics
    ///
    /// Panics when the built-in probe template does not render.
    pub async fn free_bytes(&self) -> Result<u64, ContainerProbeError> {
        let script = DiskFreeScript {
            path: shell_words::quote(self.home.as_str()),
        }
        .render()
        .expect("the disk probe template renders");
        let output = self.read(&script).await?;
        free_bytes_from_df(&output).ok_or(ContainerProbeError::Unreadable(output))
    }

    /// Runs `script` in the container's root directory and returns its
    /// standard output.
    ///
    /// # Errors
    ///
    /// Fails when the container cannot run the script, the script exits
    /// non-zero or its output is not text.
    pub async fn read(&self, script: &str) -> Result<String, ContainerProbeError> {
        // Dropping the kill sender would kill the exec, so it lives to the
        // end of the call; the dropped stdin sender is the EOF the script
        // reads.
        let (_kill, kill_rx) = async_channel::bounded::<()>(1);
        let (stdin, stdin_rx) = async_channel::unbounded::<Vec<u8>>();
        drop(stdin);
        let outcome = self
            .exec
            .exec_boxed(
                &self.container_id,
                script,
                "/",
                kill_rx,
                stdin_rx,
                ExecLifetime::Foreground,
            )
            .await
            .map_err(ContainerProbeError::Exec)?;
        match outcome {
            ContainerExecOutcome::Completed { output, .. } if output.status.success() => {
                String::from_utf8(output.stdout).map_err(|_| ContainerProbeError::NotText)
            }
            ContainerExecOutcome::Completed { output, .. } => Err(ContainerProbeError::Failed {
                status: output.status,
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            }),
            ContainerExecOutcome::Killed => Err(ContainerProbeError::Killed),
        }
    }
}

impl<T: ContainerExec> ContainerExecImpl for T {
    fn exec_boxed<'a>(
        &'a self,
        container_id: &'a str,
        script: &'a str,
        working_dir: &'a str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        lifetime: ExecLifetime,
    ) -> Pin<Box<dyn Future<Output = Result<ContainerExecOutcome, String>> + Send + 'a>> {
        Box::pin(self.exec(
            container_id,
            script,
            working_dir,
            kill_rx,
            stdin_rx,
            lifetime,
        ))
    }

    fn exec_streamed_boxed<'a>(
        &'a self,
        container_id: &'a str,
        script: &'a str,
        working_dir: &'a str,
        kill_rx: Receiver<()>,
        stdin_rx: Receiver<Vec<u8>>,
        stdout: Sender<Vec<u8>>,
        lifetime: ExecLifetime,
    ) -> Pin<Box<dyn Future<Output = Result<ContainerExecOutcome, String>> + Send + 'a>> {
        Box::pin(self.exec_streamed(
            container_id,
            script,
            working_dir,
            kill_rx,
            stdin_rx,
            stdout,
            lifetime,
        ))
    }
}

/// The backend of the home environment: where a command with no `location`
/// runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ShellBackend {
    /// On this machine, inside the local sandbox.
    Local,
    /// Inside a container.
    Container,
}

/// Which home backends are usable in the current deployment.
///
/// SSH servers are never a home environment; they are the registry's server
/// set, which changes while it lives.
#[derive(Debug, Clone, Serialize)]
pub struct ShellRuntimeAvailability {
    /// Whether local execution is available. Always true in practice.
    pub local: bool,
    /// Whether a container runtime has been wired up.
    pub container: bool,
}

impl Default for ShellRuntimeAvailability {
    fn default() -> Self {
        Self {
            local: true,
            container: false,
        }
    }
}

/// Tracks the home backend, the SSH servers commands may run on and how to
/// reach them.
///
/// Cloning shares the SSH server set: a server added or removed through one
/// holder is seen by every other, the terminals of subagents included.
#[derive(Clone)]
pub struct ShellSessionRegistry {
    availability: ShellRuntimeAvailability,
    ssh: SshEnvironments,
    container_runtime: Option<ContainerShellRuntime>,
    pub(crate) prepared_tools: Option<crate::tool_environment::PreparedToolEnvironments>,
}

impl std::fmt::Debug for ShellSessionRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShellSessionRegistry")
            .finish_non_exhaustive()
    }
}

impl ShellSessionRegistry {
    /// Creates a registry advertising the given backends, with no SSH server.
    #[must_use]
    pub fn new(availability: ShellRuntimeAvailability) -> Self {
        Self {
            availability,
            ssh: SshEnvironments::default(),
            container_runtime: None,
            prepared_tools: None,
        }
    }

    /// Replaces runtime availability.
    #[must_use]
    pub const fn with_availability(mut self, availability: ShellRuntimeAvailability) -> Self {
        self.availability = availability;
        self
    }

    /// Sets the container runtime.
    #[must_use]
    pub fn with_container_runtime(mut self, runtime: ContainerShellRuntime) -> Self {
        self.container_runtime = Some(runtime);
        self
    }

    #[must_use]
    /// Returns runtime availability.
    pub fn availability(&self) -> ShellRuntimeAvailability {
        self.availability.clone()
    }

    /// Adds `server` to the set commands may run on, for every holder of this
    /// registry. Its name and target are trimmed. It is prepared on its first
    /// command, which asks the terminal's permission handler first.
    ///
    /// # Errors
    /// Rejects a blank name or target and a name another server has.
    pub fn add_ssh_server(&self, server: SshServer) -> Result<(), SshServerError> {
        self.ssh.add(server)
    }

    /// Removes the server named `name`, returning it; `None` when no server
    /// has that name. Commands already running there keep running; the next
    /// command naming it fails, and a server added again under the name is
    /// approved and prepared anew.
    #[must_use = "the removed server, or `None` when no server has the name"]
    pub fn remove_ssh_server(&self, name: &str) -> Option<SshServer> {
        self.ssh.remove(name)
    }

    /// The SSH servers commands may run on now, in the order they were added.
    #[must_use]
    pub fn ssh_servers(&self) -> Vec<SshServer> {
        self.ssh.list()
    }

    pub(crate) const fn ssh_environments(&self) -> &SshEnvironments {
        &self.ssh
    }

    /// Reaches every server through `client` instead of the default one.
    #[cfg(test)]
    pub(crate) fn with_ssh_client(mut self, client: crate::ssh::SshClient) -> Self {
        self.ssh = self.ssh.with_client(client);
        self
    }

    pub(crate) const fn container_runtime(&self) -> Option<&ContainerShellRuntime> {
        self.container_runtime.as_ref()
    }

    /// Picks the backend to use for a command that did not name one.
    ///
    /// Prefers a container when one is configured, since it isolates better
    /// than running on the host.
    ///
    /// # Errors
    ///
    /// Returns an error if neither a container nor local execution is available.
    pub fn resolve_local_backend(&self) -> Result<ShellBackend, String> {
        let availability = self.availability();
        if availability.container {
            return Ok(ShellBackend::Container);
        }
        if availability.local {
            return Ok(ShellBackend::Local);
        }
        Err("no local backend available".to_string())
    }
}

#[cfg(all(test, unix))]
mod disk_probe_tests {
    use std::os::unix::process::ExitStatusExt as _;
    use std::sync::Mutex;

    use async_channel::{Receiver, Sender};

    use super::*;

    /// Answers `df` the way a container would, and records what it ran.
    struct Df {
        answer: &'static str,
        exit: i32,
        scripts: Mutex<Vec<String>>,
    }

    impl ContainerExec for Df {
        fn exec(
            &self,
            _container_id: &str,
            script: &str,
            working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _lifetime: ExecLifetime,
        ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
            self.scripts
                .lock()
                .unwrap()
                .push(format!("{working_dir}: {script}"));
            std::future::ready(Ok(ContainerExecOutcome::Completed {
                output: std::process::Output {
                    status: std::process::ExitStatus::from_raw(self.exit << 8),
                    stdout: self.answer.as_bytes().to_vec(),
                    stderr: b"df: no such directory".to_vec(),
                },
                killed: Vec::new(),
            }))
        }

        fn exec_streamed(
            &self,
            _container_id: &str,
            _script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _stdout: Sender<Vec<u8>>,
            _lifetime: ExecLifetime,
        ) -> impl Future<Output = Result<ContainerExecOutcome, String>> + Send {
            std::future::ready(Err("a probe is never streamed".to_string()))
        }
    }

    fn probe(answer: &'static str, exit: i32) -> (Arc<Df>, ContainerProbe) {
        let df = Arc::new(Df {
            answer,
            exit,
            scripts: Mutex::new(Vec::new()),
        });
        let exec = AnyContainerExec::new(Arc::clone(&df));
        (df, ContainerProbe::new(&exec, "box"))
    }

    const DF: &str = "Filesystem 1024-blocks Used Available Capacity Mounted on\noverlay 10485760 4194304 6291456 41% /workspace\n";

    #[tokio::test]
    async fn the_container_reports_the_free_space_of_its_home_filesystem() {
        let (df, probe) = probe(DF, 0);
        assert_eq!(probe.free_bytes().await.unwrap(), 6_291_456 * 1024);
        let scripts = df.scripts.lock().unwrap();
        assert_eq!(*scripts, ["/: df -Pk -- /workspace"]);
    }

    #[tokio::test]
    async fn a_home_with_spaces_is_quoted_for_the_shell() {
        let (df, probe) = probe(DF, 0);
        let probe = probe.with_home(ContainerPath::new("/my work").unwrap());
        probe.free_bytes().await.unwrap();
        assert_eq!(*df.scripts.lock().unwrap(), ["/: df -Pk -- '/my work'"]);
    }

    #[tokio::test]
    async fn a_df_that_fails_or_answers_something_else_is_an_error() {
        let (_, failing) = probe("", 1);
        assert!(matches!(
            failing.free_bytes().await,
            Err(ContainerProbeError::Failed { .. })
        ));
        let (_, odd) = probe("all clear\n", 0);
        assert!(matches!(
            odd.free_bytes().await,
            Err(ContainerProbeError::Unreadable(_))
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::{ShellRuntimeAvailability, ShellSessionRegistry, SshServer};

    #[test]
    fn every_clone_of_a_registry_shares_its_live_server_set() {
        let registry = ShellSessionRegistry::new(ShellRuntimeAvailability::default());
        let subagent = registry.clone();
        let prod = SshServer {
            name: "prod".to_string(),
            target: "root@example.com".to_string(),
        };
        registry.add_ssh_server(prod.clone()).unwrap();
        assert_eq!(subagent.ssh_servers(), std::slice::from_ref(&prod));
        assert_eq!(subagent.remove_ssh_server("prod"), Some(prod));
        assert_eq!(registry.ssh_servers(), []);
    }
}
