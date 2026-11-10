//! Serving an IPC router to a process in a container.
//!
//! The container reaches the host over its [host route](crate::HostRoute): a
//! listener on the route's address answers the framed requests of the
//! sandboxed `heel ipc` with the router's commands, and shims named after the
//! commands, written by a prelude the exec's shell runs first, make them
//! ordinary programs on the process's `PATH`.

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;

use askama::Template;
use async_channel::{Receiver, Sender};
use async_io::Async;
use executor_core::{AnyExecutor, Executor};
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use heel::IpcRouter;
use heel::ipc::{IpcContext, IpcRequest, IpcResponse};

use super::SpawnError;
use crate::terminal::shell_escape;
use crate::{ContainerPath, ContainerShellRuntime, MountRoot};

/// The longest request body the listener reads.
const MAX_REQUEST: usize = 16 * 1024 * 1024;

#[derive(Template)]
#[template(path = "container_ipc_prelude.sh", escape = "none")]
struct PreludeTemplate<'a> {
    escaped_commands: &'a str,
    ipc_host: &'a str,
    ipc_port: u16,
}

/// What an exec's shell runs around the sandbox of a process.
pub(crate) struct ExecSetup {
    /// Shell run in the exec's shell before and outside the sandbox. It owns
    /// its cleanup and may define shell variables for [`Self::sandbox_args`].
    pub(crate) script: String,
    /// Words appended to the in-container sandbox's `heel run` command line,
    /// each one a shell word the exec's shell expands after the script ran.
    pub(crate) sandbox_args: Vec<String>,
}

/// The setup that makes `commands` programs in a container and points them at
/// the listener on `host:port`.
pub(crate) fn exec_setup(
    commands: &[String],
    host: &str,
    port: u16,
) -> Result<ExecSetup, SpawnError> {
    for name in commands {
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(SpawnError::Setup(format!(
                "the command name {name} cannot be a shim in a container"
            )));
        }
    }
    let escaped_commands = commands
        .iter()
        .map(|name| shell_escape(name))
        .collect::<Vec<_>>()
        .join(" ");
    let script = PreludeTemplate {
        escaped_commands: &escaped_commands,
        ipc_host: host,
        ipc_port: port,
    }
    .render()
    .map_err(|e| SpawnError::Setup(format!("cannot render the container IPC prelude: {e}")))?;
    // The shims are written under the shared temp directory, which stays
    // execute-denied even at `permissive` isolation, so the sandbox is granted
    // them read-and-execute; the prelude writes them before the sandbox
    // starts, so the sandbox itself never writes the directory.
    // `--ipc-endpoint` has heel relay the sandbox's local IPC socket to the
    // listener from outside it, so the sandboxed `heel ipc` needs no TCP
    // access of its own.
    Ok(ExecSetup {
        script,
        sandbox_args: vec![
            "--ipc-endpoint".to_owned(),
            shell_escape(&format!("tcp://{host}:{port}")),
            "--grant".to_owned(),
            "\"$AITHER_IPC_DIR=rx\"".to_owned(),
        ],
    })
}

/// A router served to one process, until the exec ends.
pub(crate) struct ContainerIpc {
    shutdown: Sender<()>,
    #[cfg(test)]
    port: u16,
    pub(super) setup: ExecSetup,
}

impl ContainerIpc {
    /// Serves `router` on the address `runtime`'s host route names and
    /// prepares the shims for its commands.
    pub(crate) fn serve(
        executor: &Arc<AnyExecutor>,
        router: IpcRouter,
        runtime: &ContainerShellRuntime,
    ) -> Result<Self, SpawnError> {
        let failed = |what: &str, error: &dyn std::fmt::Display| {
            SpawnError::Setup(format!("{what}: {error}"))
        };
        let commands = router
            .methods()
            .map(|(name, _)| name.to_owned())
            .collect::<Vec<_>>();
        // The listener executes the router's commands for whoever connects,
        // so it binds only the address the container route names, never a
        // wildcard.
        let listener = TcpListener::bind(SocketAddr::new(runtime.host_route().address(), 0))
            .map_err(|e| failed("cannot bind the container IPC listener", &e))?;
        let port = listener
            .local_addr()
            .map_err(|e| failed("cannot read the container IPC listener's address", &e))?
            .port();
        listener
            .set_nonblocking(true)
            .map_err(|e| failed("cannot make the container IPC listener nonblocking", &e))?;
        let listener = Async::new(listener)
            .map_err(|e| failed("cannot register the container IPC listener", &e))?;

        let setup = exec_setup(&commands, runtime.host_route().name(), port)?;

        let (shutdown, stop) = async_channel::bounded::<()>(1);
        let workspace = runtime.workspace_mount().cloned();
        let router = Arc::new(router);
        let connections = Arc::clone(executor);
        executor
            .spawn(async move {
                if let Err(error) = accept(listener, router, workspace, stop, connections).await {
                    tracing::debug!(%error, "container IPC listener stopped with error");
                }
            })
            .detach();
        Ok(Self {
            shutdown,
            #[cfg(test)]
            port,
            setup,
        })
    }

    /// The port the listener accepted on.
    #[cfg(test)]
    pub(crate) const fn port(&self) -> u16 {
        self.port
    }

    /// Stops serving: the exec ended.
    pub(crate) async fn stop(self) {
        let _ = self.shutdown.send(()).await;
    }
}

enum Event {
    Accept(std::io::Result<(Async<TcpStream>, SocketAddr)>),
    Stop,
}

async fn accept(
    listener: Async<TcpListener>,
    router: Arc<IpcRouter>,
    workspace: Option<MountRoot>,
    stop: Receiver<()>,
    executor: Arc<AnyExecutor>,
) -> Result<(), String> {
    loop {
        let event =
            futures_lite::future::or(async { Event::Accept(listener.accept().await) }, async {
                let _ = stop.recv().await;
                Event::Stop
            })
            .await;
        match event {
            Event::Stop => return Ok(()),
            Event::Accept(Ok((stream, _))) => {
                let router = Arc::clone(&router);
                let workspace = workspace.clone();
                executor
                    .spawn(async move {
                        if let Err(error) = connection(stream, &router, workspace.as_ref()).await {
                            tracing::debug!(%error, "container IPC connection failed");
                        }
                    })
                    .detach();
            }
            Event::Accept(Err(error)) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Event::Accept(Err(error)) => return Err(format!("accept failed: {error}")),
        }
    }
}

async fn connection(
    mut stream: Async<TcpStream>,
    router: &IpcRouter,
    workspace: Option<&MountRoot>,
) -> Result<(), String> {
    loop {
        let mut length = [0_u8; 4];
        match stream.read_exact(&mut length).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(format!("cannot read a request's length: {error}")),
        }
        let length = u32::from_be_bytes(length) as usize;
        if length == 0 || length > MAX_REQUEST {
            return Err(format!("invalid request length {length}"));
        }
        let mut body = vec![0_u8; length];
        stream
            .read_exact(&mut body)
            .await
            .map_err(|e| format!("cannot read a request: {e}"))?;
        let response = match IpcRequest::decode(&body) {
            Ok(request) if request.method.is_empty() => IpcResponse::error("empty method name"),
            Ok(request) => {
                // The caller reports its directory as the container spells
                // it; the host resolves relative paths in the directory the
                // container's is mounted at. One outside the mount has no
                // host spelling and stays as given.
                let working_dir = workspace
                    .and_then(|workspace| {
                        let container = ContainerPath::new(request.working_dir.to_str()?).ok()?;
                        workspace.host_path(&container).ok()
                    })
                    .unwrap_or(request.working_dir);
                match router
                    .handle(
                        &request.method,
                        IpcContext::new(working_dir),
                        request.params,
                    )
                    .await
                {
                    Ok(payload) => IpcResponse::success(payload),
                    Err(error) => IpcResponse::error(&error.to_string()),
                }
            }
            Err(error) => IpcResponse::error(&error.to_string()),
        };
        stream
            .write_all(&response.encode())
            .await
            .map_err(|e| format!("cannot write a response: {e}"))?;
        stream
            .flush()
            .await
            .map_err(|e| format!("cannot flush a response: {e}"))?;
    }
}
