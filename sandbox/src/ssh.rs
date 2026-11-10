//! SSH servers as execution locations: the server set a registry holds while
//! it lives, the client every connection goes through, the lazily prepared
//! state of each server and the OS probe an embedder runs when a server is
//! added.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use arc_swap::ArcSwap;
use futures_lite::io::AsyncWriteExt;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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

    /// `ssh` to `server` with `line` as the remote command line.
    ///
    /// The remote login shell parses `line` before anything else runs, and
    /// shells disagree on quoting, so every line this client sends is made
    /// only of letters, digits and `-`, `.`, `/`, `;`, `>` and spaces.
    fn ssh(&self, server: &SshServer, line: &str) -> std::process::Command {
        let mut command = std::process::Command::new("ssh");
        command.args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]);
        if let Some(config) = &self.config {
            command.arg("-F").arg(config);
        }
        command.arg("--").arg(&server.target).arg(line);
        command
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

/// One server of a registry's set and what the embedder prepared on it.
/// Removing the server drops them.
#[derive(Debug)]
pub struct SshEnvironment {
    server: SshServer,
    prepared: PreparedSsh,
}

impl SshEnvironment {
    pub(crate) const fn server(&self) -> &SshServer {
        &self.server
    }

    pub(crate) const fn prepared(&self) -> &PreparedSsh {
        &self.prepared
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
        prepared: PreparedSsh,
    ) -> Result<(), SshServerError> {
        let SshServer { name, target } = server;
        let server = SshServer {
            name: name.trim().to_string(),
            target: target.trim().to_string(),
        };
        if server.name.is_empty() || server.target.is_empty() {
            return Err(SshServerError::Blank);
        }
        let environment = Arc::new(SshEnvironment { server, prepared });
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
}
