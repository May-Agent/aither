//! The container backend: a process in a running container.

use std::sync::Arc;

use executor_core::Executor;

use heel::Access;

use super::ContainerLaunch;
use super::child::{ChildExit, Descendants, KillHandle, SandboxedChild, Streams, exit_watch};
use super::container_ipc::ContainerIpc;
use super::{
    EnvSource, LaunchInfo, Pipe, Sandbox, SpawnCommand, SpawnError, SpawnGrant, SpawnSpec,
    TreeLifetime, script_reading,
};
use crate::shell_session::{ContainerExecOutcome, ExecLifetime};
use crate::terminal::{
    CONTAINER_EXEC_WORKING_DIR, ContainerConfinement, inject_env_exports,
    start_in_container_directory, wrap_container_script,
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
        // A path is compared by real path, like everything the mount maps:
        // the root as declared, then as it resolves.
        mount
            .container_path(host)
            .or_else(|declared| {
                let resolved = crate::MountRoot::new(
                    std::fs::canonicalize(&mount.host_root).map_err(|_| declared)?,
                    mount.container_root.clone(),
                );
                resolved.container_path(host)
            })
            .map_err(|error| SpawnError::Setup(error.to_string()))
    }

    /// The sandbox a derived sandbox's process runs in inside the container:
    /// the main agent's processes run unconfined because the container is its
    /// boundary, but a child's keep the per-process sandbox they have on the
    /// host, write access to its own writable grants only, and to what the
    /// spawn's own `extra` grants let it write.
    fn container_confinement(
        &self,
        runtime: &ContainerShellRuntime,
        extra: Vec<(ContainerPath, Access)>,
    ) -> Result<Option<ContainerConfinement>, SpawnError> {
        if !self.inner.grants.is_derived() {
            return Ok(None);
        }
        // The confinement is rooted at the sandbox's own directory, which a
        // child's is its space, whatever directory the process starts in.
        let working_dir = Self::container_path(runtime, &self.inner.working_dir)?;
        let mut grants = self
            .inner
            .grants
            .snapshot()
            .writable
            .iter()
            .map(|path| Ok((Self::container_path(runtime, path)?, Access::WRITE)))
            .collect::<Result<Vec<_>, SpawnError>>()?;
        grants.extend(extra);
        Ok(Some(ContainerConfinement {
            working_dir,
            grants,
        }))
    }

    /// The sandbox's space environment as the container spells each
    /// directory, through the workspace mount.
    fn container_space_environment(
        &self,
        runtime: &ContainerShellRuntime,
    ) -> Result<Vec<(String, String)>, SpawnError> {
        self.inner
            .space_environment
            .iter()
            .map(|(name, path)| {
                let path = Self::container_path(runtime, path)?;
                Ok((name.clone(), path.as_str().to_owned()))
            })
            .collect()
    }

    /// `script` as the container's managed shell runs it: the source's
    /// environment exported, then the space's, then the spec's, ahead of
    /// it, the programs' directory first on its `PATH`.
    fn exec_script(
        &self,
        launch: &ContainerLaunch,
        script: String,
        space_env: Vec<(String, String)>,
        env: Vec<(String, String)>,
        info: &LaunchInfo,
    ) -> Result<String, SpawnError> {
        let failed = |error: &dyn std::fmt::Display| SpawnError::Setup(error.to_string());
        let spec_env: Arc<dyn EnvSource> = Arc::new(env);
        let space_env: Arc<dyn EnvSource> = Arc::new(space_env);
        let script = inject_env_exports(script, Some(&spec_env), info).map_err(|e| failed(&e))?;
        let script = inject_env_exports(script, Some(&space_env), info).map_err(|e| failed(&e))?;
        let script = inject_env_exports(script, self.inner.env_source.as_ref(), info)
            .map_err(|e| failed(&e))?;
        let script = launch.wrap_script(&script).map_err(|e| failed(&e))?;
        launch.shell_command(&script).map_err(|e| failed(&e))
    }

    pub(super) fn spawn_container(
        &self,
        runtime: &ContainerShellRuntime,
        launch: &ContainerLaunch,
        spec: SpawnSpec,
    ) -> Result<SandboxedChild, SpawnError> {
        let SpawnSpec {
            command,
            env,
            cwd,
            stdio,
            grants,
            ipc,
            lifetime,
            info,
        } = spec;
        let (script, working_dir, ipc) = self.exec_command(
            runtime,
            launch,
            command,
            env,
            cwd.as_deref(),
            stdio.stdin,
            grants,
            ipc,
            &info,
        )?;

        let (kill_tx, kill_rx) = async_channel::bounded::<()>(1);
        let (stdin_tx, stdin_rx) = async_channel::unbounded::<Vec<u8>>();
        let (stdout_tx, stdout_rx) = async_channel::bounded::<Vec<u8>>(OUTPUT_BACKLOG);
        let (stderr_tx, stderr_rx) = async_channel::unbounded::<Vec<u8>>();
        let mut streams = Streams::default();
        // The exec's attached stream closes with a dropped sender; the script
        // of an unpiped stdin reads the null device instead.
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
        // A background exec lasts until its last process ends and watches its
        // stdin; a foreground one kills what its top-level shell leaves.
        let exec_lifetime = match lifetime {
            TreeLifetime::Foreground => ExecLifetime::Foreground,
            TreeLifetime::Background => {
                let (signal_tx, signal_rx) = async_channel::unbounded();
                if stdio.stdin == Pipe::Piped {
                    streams.stdin_blocked = Some(signal_rx);
                }
                ExecLifetime::Background {
                    stdin_blocked: signal_tx,
                }
            }
        };

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
                        exec_lifetime,
                    )
                    .await;
                if let Some(ipc) = ipc {
                    ipc.stop().await;
                }
                let (exit, descendants) = exit_of(outcome, &stderr_tx).await;
                setter.finish(exit, descendants);
            })
            .detach();

        Ok(SandboxedChild::new(
            exec_id(),
            streams,
            KillHandle::Switch(kill_tx),
            watch,
        ))
    }

    /// The script a container exec runs for the spec's command reading
    /// `stdin`, the directory it runs in, and the listener serving its
    /// commands.
    #[allow(clippy::too_many_arguments)]
    fn exec_command(
        &self,
        runtime: &ContainerShellRuntime,
        launch: &ContainerLaunch,
        command: SpawnCommand,
        env: Vec<(String, String)>,
        cwd: Option<&std::path::Path>,
        stdin: Pipe,
        grants: Vec<SpawnGrant>,
        ipc: Option<heel::IpcRouter>,
        info: &LaunchInfo,
    ) -> Result<(String, ContainerPath, Option<ContainerIpc>), SpawnError> {
        let failed = |error: &dyn std::fmt::Display| SpawnError::Setup(error.to_string());
        let working_dir = match cwd {
            Some(cwd) => Self::container_path(runtime, cwd)?,
            None => ContainerPath::new(CONTAINER_EXEC_WORKING_DIR)
                .expect("the container working directory is absolute"),
        };
        let mut container_grants = Vec::new();
        for grant in grants {
            match grant {
                SpawnGrant::Container { path, access } => container_grants.push((path, access)),
                SpawnGrant::Host(_) => {
                    return Err(SpawnError::Unsupported {
                        backend: "container",
                        what: "a grant on a host path",
                    });
                }
            }
        }
        let confinement = self.container_confinement(runtime, container_grants)?;

        // The exec channel accepts only a script, so the environment travels
        // as exports ahead of it; the source's come first, then the space's,
        // so the spec's win.
        let script = match command {
            SpawnCommand::Script(script) => script,
            SpawnCommand::Program { program, args } => {
                shell_words::join(std::iter::once(&program).chain(&args))
            }
        };
        let script = script_reading(stdin, script)?;
        let space_env = self.container_space_environment(runtime)?;
        let script = self.exec_script(launch, script, space_env, env, info)?;
        // A confined process runs in a sandbox rooted at the sandbox's own
        // directory; the directory it was asked to start in is entered inside.
        let script = match (&confinement, cwd.is_some()) {
            (Some(_), true) => {
                start_in_container_directory(&script, &working_dir).map_err(|e| failed(&e))?
            }
            _ => script,
        };
        let ipc = ipc
            .map(|router| ContainerIpc::serve(&self.inner.executor, router, runtime))
            .transpose()?;
        let script = wrap_container_script(
            &script,
            ipc.as_ref().map(|ipc| &ipc.setup),
            self.inner.priority,
            launch.bin_dir(),
            working_dir.as_str(),
            confinement.as_ref(),
        )
        .map_err(|e| failed(&e))?;
        Ok((script, working_dir, ipc))
    }
}

/// The container has no host process: a number out of a fresh uuid
/// identifies the exec.
fn exec_id() -> u32 {
    u32::from_le_bytes(
        uuid::Uuid::new_v4().as_u128().to_le_bytes()[..4]
            .try_into()
            .expect("uuid byte slice has four bytes"),
    )
}

/// How a finished exec ended the child, and what it killed of the
/// descendants; the exec's stderr is sent on `stderr`.
async fn exit_of(
    outcome: Result<ContainerExecOutcome, String>,
    stderr: &async_channel::Sender<Vec<u8>>,
) -> (ChildExit, Descendants) {
    match outcome {
        Ok(ContainerExecOutcome::Completed { output, killed }) => {
            if !output.stderr.is_empty() {
                let _ = stderr.send(output.stderr).await;
            }
            (
                ChildExit::Exited(output.status),
                Descendants::Killed(killed),
            )
        }
        Ok(ContainerExecOutcome::Killed) => (ChildExit::Killed, Descendants::Killed(Vec::new())),
        Err(error) => (ChildExit::Failed(error), Descendants::Unobserved),
    }
}
