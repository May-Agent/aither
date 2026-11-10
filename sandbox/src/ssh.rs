//! SSH servers as execution locations: the server set a registry holds while
//! it lives, the client every connection goes through, the first-use state of
//! each server and the OS probe an embedder runs when a server is added.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use arc_swap::ArcSwap;
use async_lock::{Mutex, OnceCell};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tool_runtime::PreparedTools;

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

/// How every SSH connection is made: `ssh` in batch mode with a bounded
/// connect time, the remote command run by `sh` whatever the login shell.
#[derive(Debug, Clone, Default)]
pub struct SshClient {
    /// An `ssh_config(5)` file read instead of the user's; only tests set
    /// one, to reach their own server.
    config: Option<PathBuf>,
}

impl SshClient {
    #[cfg(test)]
    pub(crate) const fn with_config(config: PathBuf) -> Self {
        Self {
            config: Some(config),
        }
    }

    /// `ssh` running `script` under `sh -c` on `server`.
    pub(crate) fn command(&self, server: &SshServer, script: &str) -> std::process::Command {
        let mut command = std::process::Command::new("ssh");
        command.args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]);
        if let Some(config) = &self.config {
            command.arg("-F").arg(config);
        }
        command
            .arg("--")
            .arg(&server.target)
            .arg(shell_words::join(["sh", "-c", script]));
        command
    }

    pub(crate) async fn probe_os(&self, server: &SshServer) -> Result<SshPlatform, SshProbeError> {
        let output = async_process::Command::from(self.command(server, "uname -s -m"))
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

/// A server prepared on its first command: what the embedder prepared on it.
#[derive(Debug, Clone)]
pub struct PreparedSsh {
    /// The `heel` binary on the server, when it has one; a default-mode
    /// command needs it, an `unsafe` one does not.
    pub(crate) heel: Option<String>,
    pub(crate) tools: PreparedTools<PathBuf>,
}

impl PreparedSsh {
    /// The preparation of a server whose `heel` is at the path `heel` on the
    /// server, if it has one, and whose managed tools are `tools`.
    #[must_use]
    pub const fn new(heel: Option<String>, tools: PreparedTools<PathBuf>) -> Self {
        Self { heel, tools }
    }
}

/// One server of a registry's set and what its first use settled: whether
/// commands may run there and the tools prepared there. Removing the server
/// drops both; adding it again starts over.
#[derive(Debug)]
pub struct SshEnvironment {
    server: SshServer,
    /// Whether a session approval was given. Held while a request is
    /// asked, so concurrent commands wait for it.
    approved: Mutex<bool>,
    prepared: OnceCell<PreparedSsh>,
}

/// How far an approval that let a command run on a server reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalReach {
    /// The command that asked.
    Once,
    /// Every command on the server while it stays in the registry.
    Session,
}

impl SshEnvironment {
    pub(crate) const fn server(&self) -> &SshServer {
        &self.server
    }

    /// Settles approval: a session approval is asked once and kept, with
    /// concurrent commands waiting for the one request; a once-only or a
    /// refused request leaves the next command to ask again.
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the guard is held across the request so concurrent commands wait for it"
    )]
    pub(crate) async fn approve<F, Fut, E>(&self, ask: F) -> Result<(), E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<ApprovalReach, E>>,
    {
        let mut approved = self.approved.lock().await;
        if !*approved {
            let reach = ask().await?;
            *approved = reach == ApprovalReach::Session;
        }
        Ok(())
    }

    /// The server's prepared state, produced by `prepare` the first time.
    pub(crate) async fn prepared<F, Fut, E>(&self, prepare: F) -> Result<&PreparedSsh, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<PreparedSsh, E>>,
    {
        self.prepared.get_or_try_init(prepare).await
    }

    #[cfg(test)]
    pub(crate) fn set_prepared(&self, prepared: PreparedSsh) {
        assert!(
            self.prepared.set_blocking(prepared).is_ok(),
            "a test server is prepared once"
        );
    }

    #[cfg(test)]
    pub(crate) fn is_approved(&self) -> bool {
        *self.approved.lock_blocking()
    }

    #[cfg(test)]
    pub(crate) fn is_prepared(&self) -> bool {
        self.prepared.is_initialized()
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

    pub(crate) fn add(&self, server: SshServer) -> Result<(), SshServerError> {
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
            approved: Mutex::new(false),
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

    #[test]
    fn servers_join_and_leave_a_live_set_every_clone_sees() {
        let set = SshEnvironments::default();
        let clone = set.clone();
        set.add(server(" build ", " builder@build.internal "))
            .unwrap();
        assert_eq!(
            clone.list(),
            [server("build", "builder@build.internal")],
            "names and targets are trimmed and the clone sees the addition"
        );
        assert!(matches!(
            clone.add(server("build", "other@build.internal")),
            Err(SshServerError::Duplicate(name)) if name == "build"
        ));
        assert!(matches!(
            set.add(server("staging", "  ")),
            Err(SshServerError::Blank)
        ));
        assert_eq!(
            clone.remove("build"),
            Some(server("build", "builder@build.internal"))
        );
        assert_eq!(set.list(), []);
        assert_eq!(set.remove("build"), None);
    }

    #[tokio::test]
    async fn a_server_added_again_starts_its_first_use_over() {
        let set = SshEnvironments::default();
        set.add(server("build", "builder@build.internal")).unwrap();
        let first = set.get("build").unwrap();
        first
            .approve(|| async { Ok::<_, ()>(ApprovalReach::Session) })
            .await
            .unwrap();
        assert!(first.is_approved());
        set.remove("build");
        set.add(server("build", "builder@build.internal")).unwrap();
        assert!(!set.get("build").unwrap().is_approved());
    }

    #[tokio::test]
    async fn a_refused_approval_is_asked_again() {
        let set = SshEnvironments::default();
        set.add(server("build", "builder@build.internal")).unwrap();
        let environment = set.get("build").unwrap();
        assert_eq!(environment.approve(|| async { Err("no") }).await, Err("no"));
        assert!(!environment.is_approved());
        environment
            .approve(|| async { Ok::<_, &str>(ApprovalReach::Session) })
            .await
            .unwrap();
        // Settled: a later ask is never made.
        environment
            .approve(|| async { Err("asked again") })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn a_once_only_approval_is_not_kept() {
        let set = SshEnvironments::default();
        set.add(server("build", "builder@build.internal")).unwrap();
        let environment = set.get("build").unwrap();
        environment
            .approve(|| async { Ok::<_, &str>(ApprovalReach::Once) })
            .await
            .unwrap();
        assert!(!environment.is_approved());
        assert_eq!(
            environment.approve(|| async { Err("asked again") }).await,
            Err("asked again")
        );
    }
}
