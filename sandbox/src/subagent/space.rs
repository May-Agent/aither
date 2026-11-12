//! What the embedder puts in a subagent's space before the run starts.
//!
//! Every subagent works in a space of its own, a fresh directory under its
//! launcher's working directory that it alone can write. An embedder that
//! gives its terminal [`SubagentSpaces`] fills each new space before the
//! subagent's terminal exists, typically with a clone of the repository the
//! launcher works in, and names the working copies it made there and the
//! directories its commands' environment points at. The run fails with the
//! embedder's error when that fails.
//!
//! The space belongs to the launcher: it lies inside the launcher's working
//! directory, stays when the run ends so the launcher can review and take
//! over what the subagent did, and is removed by whoever owns that working
//! directory. Only a space whose preparation failed, which no run ever
//! worked in, is removed by the terminal that created it.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::{StreamExt as _, TryStreamExt as _};

/// How many prepared paths are resolved at once. Each resolution is one
/// filesystem call on the blocking pool, and an embedder names a handful, so
/// the bound matches the import's ref checks rather than any measurement.
const RESOLUTIONS_IN_FLIGHT: usize = 8;

/// Prepares the space each subagent run works in.
///
/// Given to a terminal with
/// [`Shell::with_subagent_spaces`](crate::Shell::with_subagent_spaces),
/// it serves every subagent launched from that terminal and, at any depth,
/// from theirs.
pub trait SubagentSpaces: Send + Sync + 'static {
    /// Why a space could not be prepared.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Fills `run.space` for the run and names the working copies and
    /// environment directories made in it. Each must lie inside the space.
    fn prepare(
        &self,
        run: RunSpace<'_>,
    ) -> impl Future<Output = Result<PreparedSpace, Self::Error>> + Send;
}

/// The space a subagent run is about to start in.
#[derive(Debug, Clone, Copy)]
pub struct RunSpace<'a> {
    /// The run's space: an empty directory, by its canonical path, that only
    /// the run, and the subagents it launches, can write, and its launcher
    /// can read and write.
    pub space: &'a Path,
    /// The launching terminal's working directory, by its canonical path:
    /// the main agent's, or a subagent's space when subagents nest.
    pub launcher: &'a Path,
    /// The directory the launch was asked for from, as the request named it:
    /// where the launching command ran, else the launcher's working
    /// directory.
    pub caller: &'a Path,
}

/// What an embedder made in a run's space.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PreparedSpace {
    working_copies: Vec<PathBuf>,
    env_dirs: Vec<(String, PathBuf)>,
}

impl PreparedSpace {
    /// A space the embedder left empty.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `path`, inside the space, as a working copy the run works in.
    #[must_use]
    pub fn with_working_copy(mut self, path: impl Into<PathBuf>) -> Self {
        self.working_copies.push(path.into());
        self
    }

    /// Records the environment variable `name` as `path`, a directory inside
    /// the space, for every command the run executes: the temporary, home or
    /// cache directory a launcher's environment would otherwise point at a
    /// directory the run cannot write. It overrides the launcher's sandbox
    /// environment and [`EnvSource`](crate::EnvSource), never a command's own
    /// variables, and reaches no subagent the run launches, whose space is
    /// prepared on its own. A container sees `path` as it spells it through
    /// the workspace mount.
    #[must_use]
    pub fn with_env_dir(mut self, name: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        self.env_dirs.push((name.into(), path.into()));
        self
    }

    /// The working copies the run works in, by their canonical paths once
    /// the terminal has accepted the space.
    #[must_use]
    pub fn working_copies(&self) -> &[PathBuf] {
        &self.working_copies
    }

    /// The environment directories, by their canonical paths once the
    /// terminal has accepted the space.
    pub(crate) fn env_dirs(&self) -> &[(String, PathBuf)] {
        &self.env_dirs
    }

    /// Resolves every working copy and environment directory and confirms it
    /// lies inside `space`, which is canonical.
    pub(crate) async fn confine(self, space: &Path) -> Result<Self, super::SubagentGrantError> {
        let working_copies = futures::stream::iter(self.working_copies)
            .map(|path| inside(path, space))
            .buffered(RESOLUTIONS_IN_FLIGHT)
            .try_collect::<Vec<_>>();
        let env_dirs = futures::stream::iter(self.env_dirs)
            .map(|(name, path)| async move {
                Ok::<_, super::SubagentGrantError>((name, inside(path, space).await?))
            })
            .buffered(RESOLUTIONS_IN_FLIGHT)
            .try_collect::<Vec<_>>();
        let (working_copies, env_dirs) =
            futures::future::try_join(working_copies, env_dirs).await?;
        Ok(Self {
            working_copies,
            env_dirs,
        })
    }
}

/// `path` by its canonical path, refused unless it lies inside `space`.
async fn inside(path: PathBuf, space: &Path) -> Result<PathBuf, super::SubagentGrantError> {
    let canonical = super::canonical(&path).await?;
    if !canonical.starts_with(space) {
        return Err(super::SubagentGrantError::PreparedOutsideSpace {
            path: canonical,
            space: space.to_path_buf(),
        });
    }
    Ok(canonical)
}

/// An embedder's failure to prepare a space, whatever its own error type.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct SpacePreparationError(Box<dyn std::error::Error + Send + Sync>);

impl SpacePreparationError {
    /// The embedder's error, when it is an `E`.
    #[must_use]
    pub fn downcast_ref<E: std::error::Error + 'static>(&self) -> Option<&E> {
        self.0.downcast_ref()
    }
}

/// Object-safe twin of [`SubagentSpaces`].
trait SubagentSpacesImpl: Send + Sync {
    fn prepare<'a>(
        &'a self,
        run: RunSpace<'a>,
    ) -> BoxFuture<'a, Result<PreparedSpace, SpacePreparationError>>;
}

impl<T: SubagentSpaces> SubagentSpacesImpl for T {
    fn prepare<'a>(
        &'a self,
        run: RunSpace<'a>,
    ) -> BoxFuture<'a, Result<PreparedSpace, SpacePreparationError>> {
        Box::pin(async move {
            SubagentSpaces::prepare(self, run)
                .await
                .map_err(|error| SpacePreparationError(Box::new(error)))
        })
    }
}

/// A [`SubagentSpaces`] of any type, as a terminal holds it.
#[derive(Clone)]
pub struct AnySubagentSpaces(Arc<dyn SubagentSpacesImpl>);

impl AnySubagentSpaces {
    /// Erases `spaces`.
    #[must_use]
    pub fn new(spaces: impl SubagentSpaces) -> Self {
        Self(Arc::new(spaces))
    }

    /// Prepares `run`'s space.
    pub(crate) async fn prepare(
        &self,
        run: RunSpace<'_>,
    ) -> Result<PreparedSpace, SpacePreparationError> {
        self.0.prepare(run).await
    }
}

impl std::fmt::Debug for AnySubagentSpaces {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnySubagentSpaces").finish_non_exhaustive()
    }
}
