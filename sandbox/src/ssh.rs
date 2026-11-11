//! SSH servers as execution locations: the server set a registry holds while
//! it lives, the client every connection goes through, the lazily prepared
//! state of each server and the OS probe an embedder runs when a server is
//! added.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::{Arc, OnceLock};

use arc_swap::ArcSwap;
use async_lock::OnceCell;
use futures_lite::io::AsyncWriteExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::RemoteLaunch;

/// A named SSH destination commands may run on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SshServer {
    /// The name a command gives as its `location` to run on this server.
    pub name: String,
    /// SSH target, as passed to `ssh` (for example `user@host` or a host
    /// alias of the user's SSH configuration).
    pub target: String,
}

impl SshServer {
    /// The name used to address this server.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.name
    }
}

/// Why a server cannot join a registry's server set.
#[derive(Debug, thiserror::Error)]
pub enum SshServerError {
    /// The name or the target is blank.
    #[error("an SSH server needs a non-empty name and target")]
    Blank,
    /// Another server already has this name.
    #[error("an SSH server named `{0}` is already configured")]
    Duplicate(String),
}

/// Why a script could not be written to a server, so that nothing ran.
#[derive(Debug, thiserror::Error)]
pub enum SshUploadError {
    /// `ssh` could not be started or waited for.
    #[error("could not start ssh: {0}")]
    Spawn(std::io::Error),
    /// The script could not be sent to `ssh`.
    #[error("could not send the script to ssh: {0}")]
    Write(std::io::Error),
    /// The connection failed or the server could not store the script.
    #[error("writing the script to `{target}` failed ({status}): {stderr}")]
    Failed {
        /// The SSH target written to.
        target: String,
        /// The exit status `ssh` returned.
        status: std::process::ExitStatus,
        /// What it wrote to its standard error.
        stderr: String,
    },
}

/// The operating system and architecture a server reports, as `uname -s`
/// and `uname -m` print them (for example `Linux` and `x86_64`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SshPlatform {
    /// The kernel name, `uname -s`.
    pub os: String,
    /// The machine hardware name, `uname -m`.
    pub arch: String,
}

/// Why [`probe_ssh_os`] could not tell a server's platform.
#[derive(Debug, thiserror::Error)]
pub enum SshProbeError {
    /// `ssh` could not be started.
    #[error("could not start ssh: {0}")]
    Spawn(#[from] std::io::Error),
    /// The probe script could not be written to the server.
    #[error(transparent)]
    Upload(#[from] SshUploadError),
    /// The connection or the remote command failed.
    #[error("the probe of `{target}` exited with {status}: {stderr}")]
    Failed {
        /// The SSH target probed.
        target: String,
        /// The exit status `ssh` returned.
        status: std::process::ExitStatus,
        /// What it wrote to its standard error.
        stderr: String,
    },
    /// The server answered something other than `uname -s -m` output.
    #[error("the probe answered `{0}`, which is not `uname -s -m` output")]
    Unreadable(String),
}

/// The platform `server` runs: one `uname -s -m` over one SSH connection,
/// installing nothing and asking no one. An embedder calls it when the user
/// adds a server, to record what the server is.
///
/// # Errors
///
/// Fails when `ssh` cannot start, the connection or `uname` fails, or the
/// answer is not two words.
pub async fn probe_ssh_os(server: &SshServer) -> Result<SshPlatform, SshProbeError> {
    SshClient::default().probe_os(server).await
}

/// The private directory holding the control sockets of the master
/// connections a client owns, one per server. It exists from the client's
/// first connection until its last clone drops, which ends every master.
#[derive(Debug, Default)]
struct Masters {
    dir: OnceLock<tempfile::TempDir>,
    /// Each target's once-only opening of its master; the map is locked only
    /// to find the cell.
    opened: std::sync::Mutex<HashMap<String, Arc<OnceCell<()>>>>,
}

impl Masters {
    /// The control socket of the master to `target`. The name is a short
    /// digest because a unix socket path is limited to about a hundred bytes
    /// and `ssh` appends a suffix of its own while binding.
    fn socket(&self, target: &str) -> PathBuf {
        let dir = self.dir.get_or_init(|| {
            tempfile::Builder::new()
                .prefix("ssh-")
                .tempdir()
                .expect("the private directory of the SSH control sockets can be created")
        });
        let digest = Sha256::digest(target.as_bytes());
        dir.path().join(hex::encode(&digest[..6]))
    }
}

impl Drop for Masters {
    fn drop(&mut self) {
        let Some(dir) = self.dir.get() else { return };
        let Ok(sockets) = std::fs::read_dir(dir.path()) else {
            return;
        };
        for socket in sockets.flatten() {
            let _ = std::process::Command::new("ssh")
                .args(["-O", "exit", "-o"])
                .arg(format!("ControlPath={}", socket.path().display()))
                .args(["--", "master"])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }
}

/// How every SSH connection is made: `ssh` in batch mode with a bounded
/// connect time, the remote command run by `sh` whatever the login shell.
/// Every command to a server shares one master connection the client opens on
/// the first and ends when its last clone drops.
#[derive(Debug, Clone, Default)]
pub struct SshClient {
    /// An `ssh_config(5)` file read instead of the user's; only tests set
    /// one, to reach their own server.
    config: Option<PathBuf>,
    masters: Arc<Masters>,
}

impl SshClient {
    #[cfg(test)]
    pub(crate) fn with_config(config: PathBuf) -> Self {
        Self {
            config: Some(config),
            masters: Arc::default(),
        }
    }

    /// `ssh` to `server` with `line` as the remote command line.
    ///
    /// The remote login shell parses `line` before anything else runs, and
    /// shells disagree on quoting, so every line this client sends is made
    /// only of letters, digits and `-`, `.`, `/`, `;`, `>` and spaces.
    pub(crate) fn ssh(&self, server: &SshServer, line: &str) -> std::process::Command {
        let mut command = self.options(server);
        command.arg("--").arg(&server.target).arg(line);
        command
    }

    /// `ssh` with every option of this client, before the target.
    fn options(&self, server: &SshServer) -> std::process::Command {
        let mut command = std::process::Command::new("ssh");
        command.args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]);
        // `ControlPersist` keeps the master the first command starts alive
        // for the next ones; the timeout only reaps a master whose client
        // died without ending it.
        command
            .args(["-o", "ControlMaster=auto", "-o", "ControlPersist=10m"])
            .arg("-o")
            .arg(format!(
                "ControlPath={}",
                self.masters.socket(&server.target).display()
            ));
        if let Some(config) = &self.config {
            command.arg("-F").arg(config);
        }
        command
    }

    /// Makes sure the master connection to `server` is up, opening it when
    /// it is not. A server's master opens once, under that server's own cell:
    /// `ControlMaster=auto` makes a second `ssh` that finds the socket still
    /// being bound give up multiplexing and open a connection of its own, so
    /// concurrent first commands to one server wait for the one opening and
    /// commands to other servers wait for nothing. A failed opening is not
    /// remembered. Once opened, a master that later dies is re-created by
    /// the next command's `ControlMaster=auto`.
    async fn ensure_master(&self, server: &SshServer) -> Result<(), SshUploadError> {
        let opened = Arc::clone(
            self.masters
                .opened
                .lock()
                .expect("the map of opened masters is never held across a panic")
                .entry(server.target.clone())
                .or_default(),
        );
        opened
            .get_or_try_init(|| async {
                let output = async_process::Command::from(self.ssh(server, "true"))
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::piped())
                    .output()
                    .await
                    .map_err(SshUploadError::Spawn)?;
                if output.status.success() {
                    return Ok(());
                }
                Err(SshUploadError::Failed {
                    target: server.target.clone(),
                    status: output.status,
                    stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
                })
            })
            .await?;
        Ok(())
    }

    /// `ssh` running `script` under `sh` on `server`, with the command's
    /// standard input free for the script's own input.
    ///
    /// The script never appears on the remote command line: one `ssh` writes
    /// it to a private file on the server through its standard input, and the
    /// command returned runs that file with `sh`, which reads it byte for
    /// byte. The script deletes its own file as its first action.
    ///
    /// # Errors
    ///
    /// Fails, running nothing, when the script cannot be written to the
    /// server.
    pub(crate) async fn command(
        &self,
        server: &SshServer,
        script: &str,
    ) -> Result<std::process::Command, SshUploadError> {
        self.ensure_master(server).await?;
        let path = format!("/tmp/aither-{}.sh", uuid::Uuid::new_v4().simple());
        let upload = |error: std::io::Error| SshUploadError::Spawn(error);
        let mut child =
            async_process::Command::from(self.ssh(server, &format!("umask 077; cat > {path}")))
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .kill_on_drop(true)
                .spawn()
                .map_err(upload)?;
        let mut stdin = child.stdin.take().expect("the upload's stdin is piped");
        let body = format!("rm -f \"$0\"\n{script}");
        let written = stdin.write_all(body.as_bytes()).await;
        drop(stdin);
        let output = child.output().await.map_err(upload)?;
        if !output.status.success() {
            return Err(SshUploadError::Failed {
                target: server.target.clone(),
                status: output.status,
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            });
        }
        written.map_err(SshUploadError::Write)?;
        Ok(self.ssh(server, &format!("sh {path}")))
    }

    pub(crate) async fn probe_os(&self, server: &SshServer) -> Result<SshPlatform, SshProbeError> {
        let output = async_process::Command::from(self.command(server, "uname -s -m").await?)
            .stdin(Stdio::null())
            .output()
            .await?;
        if !output.status.success() {
            return Err(SshProbeError::Failed {
                target: server.target.clone(),
                status: output.status,
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            });
        }
        let answer = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let mut words = answer.split_whitespace();
        match (words.next(), words.next(), words.next()) {
            (Some(os), Some(arch), None) => Ok(SshPlatform {
                os: os.to_string(),
                arch: arch.to_string(),
            }),
            _ => Err(SshProbeError::Unreadable(answer)),
        }
    }
}

/// What the embedder prepared on a server: its `heel` and the programs its
/// scripts run with.
#[derive(Debug, Clone)]
pub struct PreparedSsh {
    /// The `heel` binary on the server, when it has one; a default-mode
    /// command needs it, an `unsafe` one does not.
    pub(crate) heel: Option<String>,
    pub(crate) launch: RemoteLaunch,
}

impl PreparedSsh {
    /// The preparation of a server whose `heel` is at the path `heel` on the
    /// server, if it has one, and whose programs `launch` describes.
    #[must_use]
    pub const fn new(heel: Option<String>, launch: RemoteLaunch) -> Self {
        Self { heel, launch }
    }
}

/// Why a server could not be prepared. The server stays in the set, and the
/// next command naming it prepares it again.
#[derive(Debug, thiserror::Error)]
#[error("preparing the SSH server `{server}` failed: {source}")]
pub struct SshPrepareError {
    /// The name of the server.
    pub server: String,
    /// What the embedder's [`PrepareSsh`] reported.
    #[source]
    pub source: Box<dyn std::error::Error + Send + Sync>,
}

/// Prepares a server for the shell's scripts: finds or installs its `heel`
/// and the programs scripts run with.
///
/// It runs when the first command names the server, not when the server is
/// added, so adding a server never waits on it and an unreachable server
/// fails only the commands that name it. Commands that name the server
/// together share the one preparation; a preparation that fails fails the
/// commands waiting on it and is not remembered, so the next command
/// prepares again. A [`PreparedSsh`] is a preparation already done.
pub trait PrepareSsh: Send + Sync + 'static {
    /// Why the preparation failed.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Prepares `server`.
    fn prepare(
        &self,
        server: &SshServer,
    ) -> impl Future<Output = Result<PreparedSsh, Self::Error>> + Send;
}

impl PrepareSsh for PreparedSsh {
    type Error = std::convert::Infallible;

    fn prepare(
        &self,
        _server: &SshServer,
    ) -> impl Future<Output = Result<Self, Self::Error>> + Send {
        std::future::ready(Ok(self.clone()))
    }
}

/// Object-safe twin of [`PrepareSsh`].
trait PrepareSshImpl: Send + Sync {
    fn prepare<'a>(
        &'a self,
        server: &'a SshServer,
    ) -> Pin<Box<dyn Future<Output = Result<PreparedSsh, SshPrepareError>> + Send + 'a>>;
}

impl<T: PrepareSsh> PrepareSshImpl for T {
    fn prepare<'a>(
        &'a self,
        server: &'a SshServer,
    ) -> Pin<Box<dyn Future<Output = Result<PreparedSsh, SshPrepareError>> + Send + 'a>> {
        Box::pin(async move {
            PrepareSsh::prepare(self, server)
                .await
                .map_err(|error| SshPrepareError {
                    server: server.name.clone(),
                    source: Box::new(error),
                })
        })
    }
}

/// A type-erased [`PrepareSsh`].
#[derive(Clone)]
pub struct AnyPrepareSsh(Arc<dyn PrepareSshImpl>);

impl AnyPrepareSsh {
    /// Erases `prepare`.
    #[must_use]
    pub fn new(prepare: impl PrepareSsh) -> Self {
        Self(Arc::new(prepare))
    }
}

impl std::fmt::Debug for AnyPrepareSsh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AnyPrepareSsh")
    }
}

/// One server of a registry's set and what the embedder prepared on it.
/// Removing the server drops them.
#[derive(Debug)]
pub struct SshEnvironment {
    server: SshServer,
    prepare: AnyPrepareSsh,
    prepared: OnceCell<PreparedSsh>,
}

impl SshEnvironment {
    pub(crate) const fn server(&self) -> &SshServer {
        &self.server
    }

    /// What the embedder prepared on the server, preparing it first when no
    /// command has yet. Concurrent callers share one preparation.
    ///
    /// # Errors
    /// Fails with what the preparation reported; the next call prepares
    /// again.
    pub(crate) async fn prepared(&self) -> Result<&PreparedSsh, SshPrepareError> {
        self.prepared
            .get_or_try_init(|| self.prepare.0.prepare(&self.server))
            .await
    }
}

/// The servers a registry offers, shared by every clone of it and changed
/// while it lives.
#[derive(Debug, Clone, Default)]
pub struct SshEnvironments {
    client: SshClient,
    servers: Arc<ArcSwap<Vec<Arc<SshEnvironment>>>>,
}

impl SshEnvironments {
    #[cfg(test)]
    pub(crate) fn with_client(mut self, client: SshClient) -> Self {
        self.client = client;
        self
    }

    pub(crate) const fn client(&self) -> &SshClient {
        &self.client
    }

    pub(crate) fn add(
        &self,
        server: SshServer,
        prepare: impl PrepareSsh,
    ) -> Result<(), SshServerError> {
        let SshServer { name, target } = server;
        let server = SshServer {
            name: name.trim().to_string(),
            target: target.trim().to_string(),
        };
        if server.name.is_empty() || server.target.is_empty() {
            return Err(SshServerError::Blank);
        }
        let environment = Arc::new(SshEnvironment {
            server,
            prepare: AnyPrepareSsh::new(prepare),
            prepared: OnceCell::new(),
        });
        let mut duplicate = false;
        self.servers.rcu(|servers| {
            duplicate = servers
                .iter()
                .any(|existing| existing.server.name == environment.server.name);
            let mut next = Vec::clone(servers);
            if !duplicate {
                next.push(Arc::clone(&environment));
            }
            next
        });
        if duplicate {
            return Err(SshServerError::Duplicate(environment.server.name.clone()));
        }
        Ok(())
    }

    pub(crate) fn remove(&self, name: &str) -> Option<SshServer> {
        let mut removed = None;
        self.servers.rcu(|servers| {
            let (gone, kept): (Vec<_>, Vec<_>) = servers
                .iter()
                .cloned()
                .partition(|environment| environment.server.name == name);
            removed = gone.into_iter().next();
            kept
        });
        removed.map(|environment| environment.server.clone())
    }

    pub(crate) fn list(&self) -> Vec<SshServer> {
        self.servers
            .load()
            .iter()
            .map(|environment| environment.server.clone())
            .collect()
    }

    pub(crate) fn get(&self, name: &str) -> Option<Arc<SshEnvironment>> {
        self.servers
            .load()
            .iter()
            .find(|environment| environment.server.name == name)
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server(name: &str, target: &str) -> SshServer {
        SshServer {
            name: name.to_string(),
            target: target.to_string(),
        }
    }

    fn prepared() -> PreparedSsh {
        PreparedSsh::new(
            None,
            RemoteLaunch::new("/opt/tools/bin", "/opt/tools/bin/bash"),
        )
    }

    #[test]
    fn servers_join_and_leave_a_live_set_every_clone_sees() {
        let set = SshEnvironments::default();
        let clone = set.clone();
        set.add(server(" build ", " builder@build.internal "), prepared())
            .unwrap();
        assert_eq!(
            clone.list(),
            [server("build", "builder@build.internal")],
            "names and targets are trimmed and the clone sees the addition"
        );
        assert!(matches!(
            clone.add(server("build", "other@build.internal"), prepared()),
            Err(SshServerError::Duplicate(name)) if name == "build"
        ));
        assert!(matches!(
            set.add(server("staging", "  "), prepared()),
            Err(SshServerError::Blank)
        ));
        assert_eq!(
            clone.remove("build"),
            Some(server("build", "builder@build.internal"))
        );
        assert_eq!(set.list(), []);
        assert_eq!(set.remove("build"), None);
    }

    /// Counts its preparations, takes its time over each and fails the first
    /// `failures` of them.
    struct Counting {
        runs: Arc<std::sync::atomic::AtomicUsize>,
        failures: usize,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("the server is unreachable")]
    struct Unreachable;

    impl PrepareSsh for Counting {
        type Error = Unreachable;

        async fn prepare(&self, _server: &SshServer) -> Result<PreparedSsh, Unreachable> {
            let run = self.runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            for _ in 0..10 {
                tokio::task::yield_now().await;
            }
            if run < self.failures {
                Err(Unreachable)
            } else {
                Ok(prepared())
            }
        }
    }

    fn environment(failures: usize) -> (Arc<SshEnvironment>, Arc<std::sync::atomic::AtomicUsize>) {
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let set = SshEnvironments::default();
        set.add(
            server("build", "builder@build.internal"),
            Counting {
                runs: Arc::clone(&runs),
                failures,
            },
        )
        .unwrap();
        (set.get("build").unwrap(), runs)
    }

    #[tokio::test]
    async fn a_server_is_prepared_on_its_first_use_and_only_once() {
        let (environment, runs) = environment(0);
        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "adding the server prepares nothing"
        );
        let (first, second) = tokio::join!(environment.prepared(), environment.prepared());
        first.unwrap();
        second.unwrap();
        environment.prepared().await.unwrap();
        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "two concurrent first commands and a later one share one preparation"
        );
    }

    #[tokio::test]
    async fn a_failed_preparation_is_reported_and_retried_by_the_next_command() {
        let (environment, runs) = environment(1);
        let error = environment.prepared().await.unwrap_err();
        assert_eq!(error.server, "build");
        assert!(error.source.to_string().contains("unreachable"), "{error}");
        environment
            .prepared()
            .await
            .expect("the next command prepares again");
        environment.prepared().await.unwrap();
        assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 2);
    }
}
