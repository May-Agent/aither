//! What the container and SSH backends run with, as the embedder prepared it.
//!
//! The backends know nothing of how a tool set is chosen, pinned or
//! installed: the embedder hands over the directory its programs are in, the
//! shell to run scripts in and the environment its programs need.

use std::borrow::Cow;

use askama::Template;

use crate::ContainerPath;

#[derive(Template)]
#[template(path = "tool_path.sh", escape = "none")]
struct PathTemplate<'a> {
    bin_dir: Cow<'a, str>,
    exports: Vec<(&'a str, Cow<'a, str>)>,
    script: &'a str,
}

#[derive(Template)]
#[template(path = "managed_shell.sh", escape = "none")]
struct ShellTemplate<'a> {
    shell: Cow<'a, str>,
    arguments: String,
}

fn with_path(
    bin_dir: &str,
    exports: &[(String, String)],
    script: &str,
) -> Result<String, askama::Error> {
    PathTemplate {
        bin_dir: shell_words::quote(bin_dir),
        exports: exports
            .iter()
            .map(|(name, value)| (name.as_str(), shell_words::quote(value)))
            .collect(),
        script,
    }
    .render()
}

/// The arguments with which a managed bash runs `script`: no startup file is
/// read, so a script behaves the same on every host.
#[must_use]
pub fn shell_arguments(script: &str) -> Vec<String> {
    ["--noprofile", "--norc", "-c", script]
        .map(Into::into)
        .into()
}

/// The programs prepared inside a container, for
/// [`Backend::Container`](crate::Backend::Container).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerLaunch {
    bin_dir: ContainerPath,
    shell: ContainerPath,
}

impl ContainerLaunch {
    /// Programs in `bin_dir` inside the container, run by the bash at `shell`.
    #[must_use]
    pub const fn new(bin_dir: ContainerPath, shell: ContainerPath) -> Self {
        Self { bin_dir, shell }
    }

    /// The directory the programs are in.
    #[must_use]
    pub const fn bin_dir(&self) -> &ContainerPath {
        &self.bin_dir
    }

    /// `script` with the programs' directory first on its `PATH`.
    ///
    /// # Errors
    /// Fails when the template cannot render.
    pub fn wrap_script(&self, script: &str) -> Result<String, askama::Error> {
        with_path(self.bin_dir.as_str(), &[], script)
    }

    /// A command line running `script` in the container's bash.
    ///
    /// # Errors
    /// Fails when the template cannot render.
    pub fn shell_command(&self, script: &str) -> Result<String, askama::Error> {
        ShellTemplate {
            shell: shell_words::quote(self.shell.as_str()),
            arguments: shell_words::join(shell_arguments(script)),
        }
        .render()
    }
}

/// The programs prepared on a server, for
/// [`Backend::Ssh`](crate::Backend::Ssh). Paths are on the server and text,
/// because a script names them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteLaunch {
    bin_dir: String,
    shell: String,
    exports: Vec<(String, String)>,
    writable_dir: Option<String>,
}

impl RemoteLaunch {
    /// Programs in `bin_dir` on the server, run by the bash at `shell`.
    #[must_use]
    pub fn new(bin_dir: impl Into<String>, shell: impl Into<String>) -> Self {
        Self {
            bin_dir: bin_dir.into(),
            shell: shell.into(),
            exports: Vec::new(),
            writable_dir: None,
        }
    }

    /// Variables a script starts with, in order.
    #[must_use]
    pub fn with_exports(mut self, exports: impl IntoIterator<Item = (String, String)>) -> Self {
        self.exports.extend(exports);
        self
    }

    /// A directory on the server the sandboxed programs may read, write and
    /// run in, apart from the programs' own.
    #[must_use]
    pub fn with_writable_dir(mut self, dir: impl Into<String>) -> Self {
        self.writable_dir = Some(dir.into());
        self
    }

    /// The directory the programs are in.
    #[must_use]
    pub fn bin_dir(&self) -> &str {
        &self.bin_dir
    }

    /// The bash that runs scripts.
    #[must_use]
    pub fn shell(&self) -> &str {
        &self.shell
    }

    /// The directory the sandboxed programs may write, when there is one.
    #[must_use]
    pub fn writable_dir(&self) -> Option<&str> {
        self.writable_dir.as_deref()
    }

    /// `script` with the programs' directory first on its `PATH`, after the
    /// variables.
    ///
    /// # Errors
    /// Fails when the template cannot render.
    pub fn wrap_script(&self, script: &str) -> Result<String, askama::Error> {
        with_path(&self.bin_dir, &self.exports, script)
    }
}
