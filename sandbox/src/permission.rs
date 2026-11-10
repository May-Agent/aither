//! Permission handling for terminal execution modes.
//!
//! Different terminal modes have different permission requirements:
//! - `Sandboxed`: No approval needed; outbound network is available by
//!   default, filtered per-domain by the sandbox's network lists and its
//!   [`DomainApprover`](crate::DomainApprover) and audited by higher layers
//! - `Unsafe`: Per-script approval required

use std::future::Future;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::CommandRole;
use crate::script_language::ScriptLanguage;
use crate::ssh::SshServer;

/// Permission mode for terminal execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TerminalMode {
    /// Sandboxed execution without extra host privileges.
    ///
    /// Outbound network is available by default, filtered per-domain through
    /// the configured network policy and always audited.
    #[default]
    Sandboxed,

    /// Unsafe execution without sandbox.
    /// Full system access. Requires per-script approval.
    Unsafe,
}

impl TerminalMode {
    /// Returns whether this mode requires user approval.
    #[must_use]
    pub const fn requires_approval(self) -> bool {
        matches!(self, Self::Unsafe)
    }

    /// Returns whether approval is needed per-script (vs first-use only).
    #[must_use]
    pub const fn requires_per_script_approval(self) -> bool {
        matches!(self, Self::Unsafe)
    }

    /// Returns a human-readable description of this mode.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Sandboxed => "sandboxed",
            Self::Unsafe => "unsafe (no sandbox, full access)",
        }
    }
}

/// Where a command runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CommandLocation<'a> {
    /// The home environment.
    Local,
    /// A server of the registry's SSH set.
    Ssh {
        /// The server's name, as the command gave it as its `location`.
        server: &'a str,
    },
}

/// How far an answer of [`PermissionHandler::check_location`] reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocationApproval {
    /// Commands may not run on the server.
    Refused,
    /// The command that asked runs; the next command on the server asks
    /// again.
    Once,
    /// Commands run on the server for as long as it stays in the registry,
    /// a subagent's included.
    Session,
}

impl<'a> CommandLocation<'a> {
    /// The server's name, or `None` for the home environment.
    #[must_use]
    pub const fn server(self) -> Option<&'a str> {
        match self {
            Self::Local => None,
            Self::Ssh { server } => Some(server),
        }
    }
}

/// A request for a permission beyond the sandbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PermissionRequest<'a> {
    /// A command that needs a mode beyond the sandbox.
    Command {
        /// The mode the command needs.
        mode: TerminalMode,
        /// The script to run.
        script: &'a str,
        /// The language the script is written in.
        language: ScriptLanguage,
        /// Where the command runs.
        location: CommandLocation<'a>,
    },
    /// The first command on an SSH server: whether commands may run there at
    /// all. Approving it lets the terminal connect and prepare the server:
    /// the managed tools in its cache directory and, when it has no `heel`,
    /// the distribution's pinned `heel` for its platform in
    /// `~/.local/bin`. Once approved for the session, the server's later
    /// commands, a subagent's included, are not asked about again while it
    /// stays in the registry; each still passes the admission its mode needs.
    Location {
        /// The server the command names.
        server: &'a SshServer,
        /// Who asks: the main agent or a subagent.
        role: CommandRole,
    },
}

/// How [`PermissionHandler::review`] decided a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Review {
    /// Granted; nothing else is asked.
    Approved,
    /// Refused, with the reason the model reads.
    Denied(String),
    /// Not decided; the request goes on to the handler's own route.
    Passed,
}

/// Trait for handling permission requests.
///
/// Implementors decide whether to allow terminal executions based on mode and script.
pub trait PermissionHandler: Send + Sync {
    /// Checks if the given mode and script are allowed.
    ///
    /// For `Sandboxed` mode, this should always return `Ok(true)`.
    /// For `Unsafe` mode, this should prompt for each script, which is
    /// source in `language` and runs at `location`.
    fn check(
        &self,
        mode: TerminalMode,
        script: &str,
        language: ScriptLanguage,
        location: CommandLocation<'_>,
    ) -> impl Future<Output = Result<bool, PermissionError>> + Send;

    /// Returns whether `check` will suspend waiting for an external approval.
    ///
    /// This is used by higher-level runtimes to surface pause/resume lifecycle
    /// events only when a permission request will actually block.
    fn will_wait_for_approval(
        &self,
        mode: TerminalMode,
        script: &str,
        language: ScriptLanguage,
        location: CommandLocation<'_>,
    ) -> impl Future<Output = bool> + Send {
        let _ = (mode, script, language, location);
        async { false }
    }

    /// Decides `request` before the handler's own route is asked.
    ///
    /// The terminal calls this first for every command that needs a mode
    /// beyond the sandbox and for every first command on a server. Only a
    /// request this passes reaches [`will_wait_for_approval`](Self::will_wait_for_approval)
    /// and [`check`](Self::check), so a request decided here never reports
    /// waiting on anyone.
    ///
    /// The default passes every request.
    fn review(&self, request: &PermissionRequest<'_>) -> impl Future<Output = Review> + Send {
        let _ = request;
        async { Review::Passed }
    }

    /// The handler a subagent's terminal uses, when it must differ from the
    /// one its launcher uses (for example because it labels who is asking).
    ///
    /// `None`, the default, means the subagent shares this handler, so its
    /// requests reach exactly the route the launcher's do.
    fn for_subagent(&self) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }

    /// Whether commands may run on `server` and how far the answer reaches,
    /// asked by the first command that names it and again by every command
    /// after a [`LocationApproval::Once`], after [`review`](Self::review)
    /// passed the [`PermissionRequest::Location`]. `role` says whether a
    /// subagent asks, so the handler can put a subagent's request to its
    /// main agent.
    ///
    /// Default implementation refuses every server (fail-safe).
    fn check_location(
        &self,
        server: &SshServer,
        role: CommandRole,
    ) -> impl Future<Output = Result<LocationApproval, PermissionError>> + Send {
        let _ = (server, role);
        async { Ok(LocationApproval::Refused) }
    }

    /// Whether [`check_location`](Self::check_location) will suspend waiting
    /// for someone's decision, as opposed to answering from a decision it
    /// remembers or a fixed policy. The default is `false`.
    fn will_wait_for_location(
        &self,
        server: &SshServer,
        role: CommandRole,
    ) -> impl Future<Output = bool> + Send {
        let _ = (server, role);
        async { false }
    }
}

/// Error type for permission operations.
#[derive(Debug, thiserror::Error)]
pub enum PermissionError {
    /// Permission was denied by the user.
    #[error("permission denied: {0}")]
    Denied(String),

    /// Permission check was interrupted.
    #[error("permission check interrupted")]
    Interrupted,

    /// Internal error during permission check.
    #[error("permission error: {0}")]
    Internal(#[from] anyhow::Error),
}

/// A simple permission handler that allows all sandboxed operations
/// and denies everything else.
#[derive(Debug, Clone, Copy, Default)]
pub struct DenyUnsafe;

impl PermissionHandler for DenyUnsafe {
    fn check(
        &self,
        mode: TerminalMode,
        _script: &str,
        _language: ScriptLanguage,
        _location: CommandLocation<'_>,
    ) -> impl Future<Output = Result<bool, PermissionError>> + Send {
        std::future::ready(match mode {
            TerminalMode::Sandboxed => Ok(true),
            TerminalMode::Unsafe => Err(PermissionError::Denied(format!(
                "sandbox blocked {} execution; approval is required to escalate this command, but no interactive approval handler is configured",
                mode.description()
            ))),
        })
    }

    fn will_wait_for_approval(
        &self,
        _mode: TerminalMode,
        _script: &str,
        _language: ScriptLanguage,
        _location: CommandLocation<'_>,
    ) -> impl Future<Output = bool> + Send {
        std::future::ready(false)
    }
}

/// A no-op permission handler that allows every execution mode and location.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopPermissionHandler;

impl PermissionHandler for NoopPermissionHandler {
    fn check(
        &self,
        _mode: TerminalMode,
        _script: &str,
        _language: ScriptLanguage,
        _location: CommandLocation<'_>,
    ) -> impl Future<Output = Result<bool, PermissionError>> + Send {
        std::future::ready(Ok(true))
    }

    fn will_wait_for_approval(
        &self,
        _mode: TerminalMode,
        _script: &str,
        _language: ScriptLanguage,
        _location: CommandLocation<'_>,
    ) -> impl Future<Output = bool> + Send {
        std::future::ready(false)
    }

    fn check_location(
        &self,
        _server: &SshServer,
        _role: CommandRole,
    ) -> impl Future<Output = Result<LocationApproval, PermissionError>> + Send {
        std::future::ready(Ok(LocationApproval::Session))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_deny_unsafe() {
        let handler = DenyUnsafe;

        // Sandboxed should be allowed
        assert!(
            handler
                .check(
                    TerminalMode::Sandboxed,
                    "ls",
                    ScriptLanguage::Shell,
                    CommandLocation::Local,
                )
                .await
                .unwrap()
        );

        // Unsafe should be denied
        assert!(
            handler
                .check(
                    TerminalMode::Unsafe,
                    "rm -rf /",
                    ScriptLanguage::Shell,
                    CommandLocation::Local,
                )
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_noop_permission_handler() {
        let handler = NoopPermissionHandler;

        assert!(
            handler
                .check(
                    TerminalMode::Sandboxed,
                    "ls",
                    ScriptLanguage::Shell,
                    CommandLocation::Local,
                )
                .await
                .unwrap()
        );
        assert!(
            handler
                .check(
                    TerminalMode::Unsafe,
                    "rm -rf /",
                    ScriptLanguage::Shell,
                    CommandLocation::Local,
                )
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn only_the_noop_handler_allows_a_location_by_default() {
        let server = SshServer {
            name: "build".to_string(),
            target: "builder@build.internal".to_string(),
        };
        assert_eq!(
            DenyUnsafe
                .check_location(&server, CommandRole::Main)
                .await
                .unwrap(),
            LocationApproval::Refused
        );
        assert_eq!(
            NoopPermissionHandler
                .check_location(&server, CommandRole::Subagent)
                .await
                .unwrap(),
            LocationApproval::Session
        );
    }

    #[test]
    fn a_location_request_names_the_server_and_who_asks() {
        let server = SshServer {
            name: "build".to_string(),
            target: "builder@build.internal".to_string(),
        };
        assert_eq!(
            serde_json::to_value(PermissionRequest::Location {
                server: &server,
                role: CommandRole::Subagent,
            })
            .unwrap(),
            serde_json::json!({
                "kind": "location",
                "server": { "name": "build", "target": "builder@build.internal" },
                "role": "subagent",
            })
        );
    }

    #[test]
    fn a_command_request_names_where_it_runs() {
        let request = |location| PermissionRequest::Command {
            mode: TerminalMode::Unsafe,
            script: "make",
            language: ScriptLanguage::Shell,
            location,
        };
        assert_eq!(
            serde_json::to_value(request(CommandLocation::Local)).unwrap()["location"],
            serde_json::json!({ "kind": "local" })
        );
        assert_eq!(
            serde_json::to_value(request(CommandLocation::Ssh { server: "build" })).unwrap()["location"],
            serde_json::json!({ "kind": "ssh", "server": "build" })
        );
    }
}
