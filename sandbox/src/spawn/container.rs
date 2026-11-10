//! The container backend: a process in a running container.

use std::sync::Arc;

use executor_core::Executor;

use super::ContainerLaunch;
use super::child::{ChildExit, KillHandle, SandboxedChild, Streams, exit_watch};
use super::{EnvSource, Pipe, Sandbox, SpawnError, SpawnSpec};
use crate::shell_session::ContainerExecOutcome;
use crate::terminal::{
    CONTAINER_EXEC_WORKING_DIR, ContainerConfinement, inject_env_exports, wrap_container_script,
};
use crate::{ContainerPath, ContainerShellRuntime};

/// How many output chunks a container exec may run ahead of its reader.
const OUTPUT_BACKLOG: usize = 64;

impl Sandbox {
    /// A container path for the host path `host`, through the workspace
    /// mount.
    fn container_path(
        runtime: &ContainerShellRuntime,
        host: &std::path::Path,
    ) -> Result<ContainerPath, SpawnError> {
        let mount = runtime.workspace_mount().ok_or_else(|| {
            SpawnError::Setup(format!(
                "{} has no counterpart in the container: it declares no workspace mount",
                host.display()
            ))
        })?;
        mount
            .container_path(host)
            .map_err(|error| SpawnError::Setup(error.to_string()))
    }

    /// The sandbox a derived sandbox's process runs in inside the container:
    /// the main agent's processes run unconfined because the container is its
    /// boundary, but a child's keep the per-process sandbox they have on the
    /// host, write access to its own writable grants only.
    fn container_confinement(
        &self,
        runtime: &ContainerShellRuntime,
        working_dir: ContainerPath,
    ) -> Result<Option<ContainerConfinement>, SpawnError> {
        if !self.inner.grants.is_derived() {
            return Ok(None);
        }
        let writable = self
            .inner
            .grants
            .snapshot()
            .writable
            .iter()
            .map(|path| Self::container_path(runtime, path))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(ContainerConfinement {
            working_dir,
            writable,
        }))
    }

    pub(super) fn spawn_container(
        &self,
        runtime: &ContainerShellRuntime,
        launch: &ContainerLaunch,
        spec: SpawnSpec,
    ) -> Result<SandboxedChild, SpawnError> {
        let setup = |error: &dyn std::fmt::Display| SpawnError::Setup(error.to_string());
        let SpawnSpec {
            program,
            args,
            env,
            cwd,
            stdio,
            extra_grants: _,
            ipc: _,
        } = spec;

        let working_dir = match &cwd {
            Some(cwd) => Self::container_path(runtime, cwd)?,
            None => ContainerPath::new(CONTAINER_EXEC_WORKING_DIR)
                .expect("the container working directory is absolute"),
        };
        let confinement = self.container_confinement(runtime, working_dir.clone())?;

        // The exec channel accepts only a script, so the environment travels
        // as exports ahead of it; the spec's lead the source's.
        let script = shell_words::join(std::iter::once(&program).chain(&args));
        let spec_env: Arc<dyn EnvSource> = Arc::new(env);
        let script = inject_env_exports(script, Some(&spec_env)).map_err(|e| setup(&e))?;
        let script =
            inject_env_exports(script, self.inner.env_source.as_ref()).map_err(|e| setup(&e))?;
        let script = launch.wrap_script(&script).map_err(|e| setup(&e))?;
        let script = launch.shell_command(&script).map_err(|e| setup(&e))?;
        let script = wrap_container_script(
            &script,
            &[],
            None,
            self.inner.priority,
            launch.bin_dir(),
            working_dir.as_str(),
            confinement.as_ref(),
        )
        .map_err(|e| setup(&e))?;

        let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
        let (stdin_tx, stdin_rx) = async_channel::unbounded::<Vec<u8>>();
        let (stdout_tx, stdout_rx) = async_channel::bounded::<Vec<u8>>(OUTPUT_BACKLOG);
        let (stderr_tx, stderr_rx) = async_channel::unbounded::<Vec<u8>>();
        let mut streams = Streams::default();
        // A dropped stdin sender is the EOF the process reads.
        if stdio.stdin == Pipe::Piped {
            streams.stdin = Some(stdin_tx);
        }
        // The exec is killed once its stdout has no receiver, so an unpiped
        // stream is read and dropped.
        match stdio.stdout {
            Pipe::Piped => streams.stdout = Some(stdout_rx),
            Pipe::Null => self
                .inner
                .executor
                .spawn(async move { while stdout_rx.recv().await.is_ok() {} })
                .detach(),
        }
        if stdio.stderr == Pipe::Piped {
            streams.stderr = Some(stderr_rx);
        }

        // Dropping the kill sender would kill the exec, so the task that waits
        // for it holds one until it ends.
        let kill = kill_tx.clone();
        let exec = runtime.exec();
        let container_id = runtime.container_id().to_string();
        let working = working_dir.as_str().to_string();
        let (setter, watch) = exit_watch();
        self.inner
            .executor
            .spawn(async move {
                let _kill = kill;
                let outcome = exec
                    .exec_streamed_boxed(
                        &container_id,
                        &script,
                        &working,
                        kill_rx,
                        stdin_rx,
                        stdout_tx,
                    )
                    .await;
                setter.finish(match outcome {
                    Ok(ContainerExecOutcome::Completed { output, .. }) => {
                        if !output.stderr.is_empty() {
                            let _ = stderr_tx.send(output.stderr).await;
                        }
                        ChildExit::Exited(output.status)
                    }
                    Ok(ContainerExecOutcome::Killed) => ChildExit::Killed,
                    Err(error) => ChildExit::Failed(error),
                });
            })
            .detach();

        // The container has no host process: a number out of a fresh uuid
        // identifies the exec.
        let id = u32::from_le_bytes(
            uuid::Uuid::new_v4().as_u128().to_le_bytes()[..4]
                .try_into()
                .expect("uuid byte slice has four bytes"),
        );
        Ok(SandboxedChild::new(
            id,
            streams,
            KillHandle::Switch(kill_tx),
            watch,
        ))
    }
}
