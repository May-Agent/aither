//! Serving an IPC router to a process in a container.
//!
//! The container reaches the host over its [host route](crate::HostRoute): a
//! listener on the route's address answers the framed requests of the
//! sandboxed `heel ipc` with the router's commands, and shims written by
//! `heel shims` from the router's command declarations, which a prelude the
//! exec's shell runs first invokes, make them ordinary programs on the
//! process's `PATH`.

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use askama::Template;
use async_channel::{Receiver, Sender};
use async_io::Async;
use executor_core::{AnyExecutor, Executor};
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use heel::IpcRouter;
use heel::ipc::{IpcCommands, IpcContext, IpcRequest, IpcResponse};
use subtle::ConstantTimeEq as _;

use super::SpawnError;
use crate::terminal::shell_escape;
use crate::{ContainerPath, ContainerShellRuntime, MountRoot};

/// The longest request body the listener reads.
const MAX_REQUEST: usize = 16 * 1024 * 1024;

/// Length in bytes of the hex-encoded 256-bit per-exec token.
const IPC_TOKEN_HEX_LEN: usize = 64;

/// Longest method field `heel-shim` encodes (`u8::MAX`), leaving room for `<token>:`.
const MAX_COMMAND_NAME_LEN: usize = (u8::MAX as usize) - IPC_TOKEN_HEX_LEN - 1;

#[derive(Template)]
#[template(path = "container_ipc_prelude.sh", escape = "none")]
struct PreludeTemplate<'a> {
    escaped_commands: &'a str,
    ipc_token: &'a str,
    ipc_host: &'a str,
    ipc_port: u16,
}

/// What an exec's shell runs around the sandbox of a process.
pub struct ExecSetup {
    /// Shell run in the exec's shell before and outside the sandbox. It owns
    /// its cleanup and may define shell variables for [`Self::sandbox_args`].
    pub(crate) script: String,
    /// Words appended to the in-container sandbox's `heel run` command line,
    /// each one a shell word the exec's shell expands after the script ran.
    pub(crate) sandbox_args: Vec<String>,
}

fn generate_ipc_token() -> Result<String, SpawnError> {
    let mut random = [0_u8; 32];
    getrandom::fill(&mut random)
        .map_err(|e| SpawnError::Setup(format!("cannot generate container IPC token: {e}")))?;
    Ok(hex::encode(random))
}

/// The setup that makes `commands` programs in a container and points them at
/// the listener on `host:port`.
#[cfg(test)]
pub fn exec_setup(commands: &IpcCommands, host: &str, port: u16) -> Result<ExecSetup, SpawnError> {
    let token = generate_ipc_token()?;
    exec_setup_with_token(commands, host, port, &token)
}

fn exec_setup_with_token(
    commands: &IpcCommands,
    host: &str,
    port: u16,
    token: &str,
) -> Result<ExecSetup, SpawnError> {
    let value = serde_json::to_value(commands)
        .map_err(|e| SpawnError::Setup(format!("cannot serialize the IPC commands: {e}")))?;
    if let Some(map) = value.as_object() {
        for name in map.keys() {
            if name.len() > MAX_COMMAND_NAME_LEN || name.contains(':') {
                return Err(SpawnError::Setup(format!(
                    "invalid container IPC command name `{name}`"
                )));
            }
        }
    }
    let commands = serde_json::to_string(&value)
        .map_err(|e| SpawnError::Setup(format!("cannot serialize the IPC commands: {e}")))?;
    let escaped_commands = shell_escape(&commands);
    let script = PreludeTemplate {
        escaped_commands: &escaped_commands,
        ipc_token: token,
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
pub struct ContainerIpc {
    shutdown: Sender<()>,
    #[cfg(test)]
    port: u16,
    #[cfg(test)]
    token: Arc<str>,
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
        // The listener executes the router's commands for whoever connects,
        // so it binds only the address the container route names, never a
        // wildcard, and requires the per-exec token on every request.
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

        let token: Arc<str> = Arc::from(generate_ipc_token()?);
        let setup =
            exec_setup_with_token(router.commands(), runtime.host_route().name(), port, &token)?;

        let (shutdown, stop) = async_channel::bounded::<()>(1);
        let workspace = runtime.workspace_mount().cloned();
        let router = Arc::new(router);
        let connections = Arc::clone(executor);
        let accept_token = Arc::clone(&token);
        executor
            .spawn(async move {
                if let Err(error) =
                    accept(listener, router, workspace, accept_token, stop, connections).await
                {
                    tracing::debug!(%error, "container IPC listener stopped with error");
                }
            })
            .detach();
        Ok(Self {
            shutdown,
            #[cfg(test)]
            port,
            #[cfg(test)]
            token,
            setup,
        })
    }

    /// The port the listener accepted on.
    #[cfg(test)]
    pub(crate) const fn port(&self) -> u16 {
        self.port
    }

    /// The per-exec authentication token the listener expects.
    #[cfg(test)]
    pub(crate) fn token(&self) -> &str {
        &self.token
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
    token: Arc<str>,
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
                let token = Arc::clone(&token);
                executor
                    .spawn(async move {
                        if let Err(error) =
                            connection(stream, &router, workspace.as_ref(), &token).await
                        {
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

fn verify_token(supplied: &str, expected: &str) -> bool {
    supplied.len() == expected.len() && bool::from(supplied.as_bytes().ct_eq(expected.as_bytes()))
}

fn resolve_host_working_dir(
    workspace: Option<&MountRoot>,
    reported: &Path,
) -> Result<PathBuf, &'static str> {
    let workspace = workspace.ok_or("container IPC requires a workspace mount")?;
    let spelled = reported
        .to_str()
        .ok_or("working directory is not valid UTF-8")?;
    let container =
        ContainerPath::new(spelled).map_err(|_| "working directory must be an absolute path")?;
    let mapped = workspace
        .host_path(&container)
        .map_err(|_| "working directory is outside the container workspace mount")?;

    // Reject symlinks inside the workspace that escape the host mount root.
    let canonical_root = dunce::canonicalize(&workspace.host_root)
        .map_err(|_| "container workspace mount root is not canonicalizable")?;
    let mut existing = mapped.as_path();
    while !existing.exists() {
        // A dangling symlink reads as not existing, but a later command
        // would still write through it, outside the workspace.
        if existing
            .symlink_metadata()
            .is_ok_and(|meta| meta.file_type().is_symlink())
        {
            return Err("working directory contains a dangling symlink");
        }
        let Some(parent) = existing.parent() else {
            break;
        };
        existing = parent;
    }
    let canonical_existing =
        dunce::canonicalize(existing).map_err(|_| "working directory is not canonicalizable")?;
    if !canonical_existing.starts_with(&canonical_root) {
        return Err("working directory escapes the container workspace mount");
    }
    Ok(mapped)
}

async fn connection(
    mut stream: Async<TcpStream>,
    router: &IpcRouter,
    workspace: Option<&MountRoot>,
    expected_token: &str,
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
        let mut authorized = true;
        let response = match IpcRequest::decode(&body) {
            Ok(request) => match request.method.split_once(':') {
                Some((supplied_token, method)) if verify_token(supplied_token, expected_token) => {
                    if method.is_empty() {
                        IpcResponse::error("empty method name")
                    } else {
                        match resolve_host_working_dir(workspace, &request.working_dir) {
                            Ok(working_dir) => {
                                match router
                                    .handle(method, IpcContext::new(working_dir), request.params)
                                    .await
                                {
                                    Ok(payload) => IpcResponse::success(payload),
                                    Err(error) => IpcResponse::error(&error.to_string()),
                                }
                            }
                            Err(reason) => IpcResponse::error(reason),
                        }
                    }
                }
                _ => {
                    authorized = false;
                    IpcResponse::error("unauthorized container IPC request")
                }
            },
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
        if !authorized {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A container executor for a runtime whose container never runs.
    struct UnreachableExec;

    impl crate::ContainerExec for UnreachableExec {
        fn exec(
            &self,
            _container_id: &str,
            _script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _lifetime: crate::ExecLifetime,
        ) -> impl std::future::Future<Output = Result<crate::ContainerExecOutcome, String>> + Send
        {
            std::future::ready(Err("the test never execs".to_string()))
        }

        fn exec_streamed(
            &self,
            _container_id: &str,
            _script: &str,
            _working_dir: &str,
            _kill_rx: Receiver<()>,
            _stdin_rx: Receiver<Vec<u8>>,
            _stdout: Sender<Vec<u8>>,
            _lifetime: crate::ExecLifetime,
        ) -> impl std::future::Future<Output = Result<crate::ContainerExecOutcome, String>> + Send
        {
            std::future::ready(Err("the test never execs".to_string()))
        }
    }

    /// An IPC command that answers with its marker and the working directory
    /// it was handed, so a request that reached the live handler through the
    /// mapped mount is unmistakable.
    struct ProbeEchoCommand;

    #[derive(Debug, serde::Deserialize)]
    struct ProbeEchoArgs {
        marker: String,
    }

    impl heel::IpcCommand for ProbeEchoCommand {
        fn name(&self) -> std::borrow::Cow<'static, str> {
            "probe_echo".into()
        }

        type Args = ProbeEchoArgs;
        type Response = String;

        fn handle(
            &self,
            args: Self::Args,
            context: heel::IpcContext,
        ) -> impl std::future::Future<Output = Result<Self::Response, heel::IpcError>> + Send
        {
            std::future::ready(Ok(format!(
                "marker: {} in {}",
                args.marker,
                context.working_dir().display()
            )))
        }
    }

    /// A command shim the rendered prelude writes must reach the live
    /// router: the `heel-shim` helper knows itself by its executable name's
    /// file stem, so the token wrapper renames the binary only to a name
    /// that keeps it. Renaming it `heel-shim-bin` made the helper read a
    /// `heel-shim-bin.json` beside it instead of parsing its command line,
    /// and every IPC command answered `I/O error: No such file or directory`.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn the_prelude_shims_reach_the_router_through_the_real_heel_shim() {
        let work_dir = tempfile::tempdir().expect("a workspace");
        // The directory a shim reports is what `getcwd` answers: the
        // resolved spelling, so the test's mount and working directory are
        // canonical like everything `resolve_host_working_dir` compares.
        let work = dunce::canonicalize(work_dir.path()).expect("the workspace resolves");
        let loopback = std::net::IpAddr::from([127, 0, 0, 1]);
        let runtime = crate::ContainerShellRuntime::new(
            "aither-test-ipc-shim",
            crate::HostRoute::new("127.0.0.1", loopback).expect("loopback is routable"),
            crate::AnyContainerExec::new(Arc::new(UnreachableExec)),
        )
        .with_workspace_mount(crate::MountRoot::new(
            work.clone(),
            ContainerPath::new(work.to_str().expect("the workspace is UTF-8"))
                .expect("the workspace is absolute"),
        ));
        let router = heel::IpcRouter::new().register(ProbeEchoCommand);
        let executor = Arc::new(AnyExecutor::new(executor_core::tokio::TokioGlobal));
        let bridge = ContainerIpc::serve(&executor, router, &runtime).expect("the bridge serves");

        // The prelude is what an exec's shell runs before the script, so
        // running it under `sh` with a real `heel` exercises the shims it
        // writes exactly as a container does.
        let heel = crate::test_heel::binary().await.expect("a test heel");
        let heel_dir = heel
            .as_path()
            .parent()
            .expect("the heel binary has a directory")
            .to_path_buf();
        let path = std::env::join_paths(std::iter::once(heel_dir).chain(std::env::split_paths(
            &std::env::var_os("PATH").unwrap_or_default(),
        )))
        .expect("the PATH joins");
        let output = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(format!(
                "{}\nprobe_echo --marker reached",
                bridge.setup.script
            ))
            .current_dir(&work)
            .env("PATH", path)
            .output()
            .expect("the prelude script runs");
        bridge.stop().await;
        assert!(
            output.status.success(),
            "the shim should reach the router: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            format!("marker: reached in {}\n", work.display())
        );
    }

    /// A symlink inside the workspace whose target does not exist still
    /// reads as absent, but a later command would write through it outside
    /// the mount: it is rejected.
    #[cfg(unix)]
    #[test]
    fn a_dangling_symlink_in_the_workspace_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let host_root = dir.path().join("workspace");
        std::fs::create_dir(&host_root).unwrap();
        let outside = dir.path().join("outside").join("no-such-target");
        std::os::unix::fs::symlink(&outside, host_root.join("escape_link")).unwrap();
        let workspace = MountRoot::new(host_root, ContainerPath::new("/workspace").unwrap());

        assert!(
            resolve_host_working_dir(Some(&workspace), Path::new("/workspace/escape_link"),)
                .is_err(),
            "a dangling symlink must be rejected"
        );
        assert!(
            resolve_host_working_dir(Some(&workspace), Path::new("/workspace/escape_link/subdir"),)
                .is_err(),
            "a dangling symlink mid-path must be rejected"
        );
    }
}
