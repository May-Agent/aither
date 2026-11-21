//! Per-launch IPC multiplexing for a persistent host `heel::Sandbox`.
//!
//! A host `Sandbox` runs one persistent `heel::Sandbox` whose `external_ipc`
//! relays every in-sandbox IPC connection to a loopback [`HostIpcMux`] on the
//! host. Each command launch that carries an [`IpcRouter`] registers its router
//! under a random 256-bit token, writes command shims into a private read+exec
//! directory granted to the persistent sandbox, prepends that directory to the
//! command's `PATH`, and unregisters its token + removes the directory on drop.

use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_channel::{Receiver, Sender};
use async_io::Async;
use executor_core::{AnyExecutor, Executor};
use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use heel::IpcRouter;
use heel::ipc::{IpcContext, IpcRequest, IpcResponse, TcpIpcEndpoint};
use subtle::ConstantTimeEq as _;

use super::SpawnError;

/// Maximum request frame size accepted by the host IPC mux.
const MAX_REQUEST: usize = 16 * 1024 * 1024;

/// Length in bytes of the hex-encoded 256-bit per-launch token.
const IPC_TOKEN_HEX_LEN: usize = 64;

/// Longest command name `heel-shim` can encode (`u8::MAX`) after `<token>:`.
const MAX_COMMAND_NAME_LEN: usize = (u8::MAX as usize) - IPC_TOKEN_HEX_LEN - 1;

/// Connect timeout for heel's local loopback `external_ipc` relay.
const RELAY_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

fn generate_token() -> Result<Arc<str>, SpawnError> {
    let mut random = [0_u8; 32];
    getrandom::fill(&mut random)
        .map_err(|e| SpawnError::Setup(format!("cannot generate host IPC token: {e}")))?;
    Ok(Arc::from(hex::encode(random)))
}

type RouterTable = Arc<Mutex<HashMap<Arc<str>, Arc<IpcRouter>>>>;

/// Loopback IPC multiplexer shared by every command spawned into one persistent
/// host `heel::Sandbox`.
pub struct HostIpcMux {
    endpoint: TcpIpcEndpoint,
    routers: RouterTable,
    shutdown: Sender<()>,
}

impl HostIpcMux {
    /// Starts a loopback listener on `127.0.0.1:0` served on `executor`.
    pub(crate) fn start(executor: &Arc<AnyExecutor>) -> Result<Self, SpawnError> {
        let failed = |what: &str, error: &dyn std::fmt::Display| {
            SpawnError::Setup(format!("{what}: {error}"))
        };
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .map_err(|e| failed("cannot bind host IPC mux listener", &e))?;
        let addr = listener
            .local_addr()
            .map_err(|e| failed("cannot read host IPC mux address", &e))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| failed("cannot set host IPC mux nonblocking", &e))?;
        let listener =
            Async::new(listener).map_err(|e| failed("cannot register host IPC mux", &e))?;

        let routers: RouterTable = Arc::new(Mutex::new(HashMap::new()));
        let (shutdown, stop) = async_channel::bounded::<()>(1);
        let accept_routers = Arc::clone(&routers);
        let connections = Arc::clone(executor);
        executor
            .spawn(async move {
                if let Err(error) = accept(listener, accept_routers, stop, connections).await {
                    tracing::debug!(%error, "host IPC mux listener stopped with error");
                }
            })
            .detach();

        Ok(Self {
            endpoint: TcpIpcEndpoint::new(addr, RELAY_CONNECT_TIMEOUT),
            routers,
            shutdown,
        })
    }

    /// Endpoint passed to `SandboxConfigBuilder::external_ipc`.
    pub(crate) const fn endpoint(&self) -> TcpIpcEndpoint {
        self.endpoint
    }

    /// Materializes shims for `router` in a private temporary directory and
    /// registers `router` on this mux until the returned [`HostIpcLease`] drops.
    pub(crate) async fn register(&self, router: IpcRouter) -> Result<HostIpcLease, SpawnError> {
        let value = serde_json::to_value(router.commands())
            .map_err(|e| SpawnError::Setup(format!("cannot serialize host IPC commands: {e}")))?;
        if let Some(map) = value.as_object() {
            for name in map.keys() {
                if name.len() > MAX_COMMAND_NAME_LEN || name.contains(':') {
                    return Err(SpawnError::Setup(format!(
                        "invalid host IPC command name `{name}`"
                    )));
                }
            }
        }

        let token = generate_token()?;
        let write_token = Arc::clone(&token);
        let commands = router.commands().clone();
        let shim_dir = blocking::unblock(move || -> Result<tempfile::TempDir, SpawnError> {
            let dir = tempfile::Builder::new()
                .prefix("aither-ipc-")
                .tempdir()
                .map_err(|e| SpawnError::Setup(format!("cannot create IPC shim dir: {e}")))?;
            commands
                .write_shims(dir.path())
                .map_err(|e| SpawnError::Setup(format!("cannot write IPC shims: {e}")))?;
            wrap_shim_with_token(dir.path(), &write_token)?;
            Ok(dir)
        })
        .await?;

        self.routers
            .lock()
            .expect("host IPC router table lock is never poisoned")
            .insert(Arc::clone(&token), Arc::new(router));

        Ok(HostIpcLease {
            routers: Arc::clone(&self.routers),
            token,
            shim_dir,
        })
    }
}

impl Drop for HostIpcMux {
    fn drop(&mut self) {
        let _ = self.shutdown.try_send(());
    }
}

/// Keeps one command launch's IPC router and shim directory alive for the
/// lifetime of that command's process tree.
pub struct HostIpcLease {
    routers: RouterTable,
    token: Arc<str>,
    shim_dir: tempfile::TempDir,
}

impl HostIpcLease {
    pub(crate) fn shim_dir(&self) -> &Path {
        self.shim_dir.path()
    }
}

impl Drop for HostIpcLease {
    fn drop(&mut self) {
        self.routers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.token);
    }
}

#[cfg(unix)]
fn wrap_shim_with_token(dir: &Path, token: &str) -> Result<(), SpawnError> {
    use std::os::unix::fs::PermissionsExt as _;

    let shim = dir.join("heel-shim");
    if !shim.is_file() {
        return Ok(());
    }
    let real_dir = dir.join(".real");
    std::fs::create_dir(&real_dir)
        .map_err(|e| SpawnError::Setup(format!("cannot create IPC real shim dir: {e}")))?;
    std::fs::set_permissions(&real_dir, std::fs::Permissions::from_mode(0o755))
        .map_err(|e| SpawnError::Setup(format!("cannot chmod IPC real shim dir: {e}")))?;
    // Keep the executable file name `heel-shim`: `heel-shim` treats any other
    // stem as a Windows hard-linked command and looks for `<stem>.json`.
    let bin = real_dir.join("heel-shim");
    std::fs::rename(&shim, &bin)
        .map_err(|e| SpawnError::Setup(format!("cannot move heel-shim: {e}")))?;
    let wrapper = format!(
        "#!/bin/sh\n__aither_cmd=\"$1\"\nshift\nexec \"${{0%/*}}/.real/heel-shim\" \"{token}:$__aither_cmd\" \"$@\"\n"
    );
    std::fs::write(&shim, wrapper)
        .map_err(|e| SpawnError::Setup(format!("cannot write heel-shim wrapper: {e}")))?;
    std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755))
        .map_err(|e| SpawnError::Setup(format!("cannot chmod heel-shim wrapper: {e}")))?;
    Ok(())
}

#[cfg(not(unix))]
fn wrap_shim_with_token(_dir: &Path, _token: &str) -> Result<(), SpawnError> {
    Ok(())
}

/// Prepends `shim_dir` ahead of `PATH` in `env`.
pub fn prepend_shim_path(env: &mut Vec<(String, String)>, shim_dir: &Path) {
    let base = env
        .iter()
        .rev()
        .find(|(k, _)| k.eq_ignore_ascii_case("PATH"))
        .map(|(_, v)| v.clone())
        .or_else(|| std::env::var("PATH").ok())
        .unwrap_or_default();
    let entries = std::iter::once(PathBuf::from(shim_dir)).chain(std::env::split_paths(&base));
    if let Ok(joined) = std::env::join_paths(entries) {
        env.push(("PATH".to_string(), joined.to_string_lossy().into_owned()));
    }
}

enum Event {
    Accept(std::io::Result<(Async<TcpStream>, SocketAddr)>),
    Stop,
}

async fn accept(
    listener: Async<TcpListener>,
    routers: RouterTable,
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
                let routers = Arc::clone(&routers);
                executor
                    .spawn(async move {
                        if let Err(error) = connection(stream, &routers).await {
                            tracing::debug!(%error, "host IPC mux connection ended with error");
                        }
                    })
                    .detach();
            }
            Event::Accept(Err(error)) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Event::Accept(Err(error)) => return Err(format!("accept failed: {error}")),
        }
    }
}

fn lookup_router(routers: &RouterTable, supplied_token: &str) -> Option<Arc<IpcRouter>> {
    let guard = routers
        .lock()
        .expect("host IPC router table lock is never poisoned");
    guard.iter().find_map(|(token, router)| {
        let matches = token.len() == supplied_token.len()
            && bool::from(token.as_bytes().ct_eq(supplied_token.as_bytes()));
        matches.then(|| Arc::clone(router))
    })
}

async fn connection(mut stream: Async<TcpStream>, routers: &RouterTable) -> Result<(), String> {
    loop {
        let mut length = [0_u8; 4];
        match stream.read_exact(&mut length).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(format!("cannot read request length: {error}")),
        }
        let length = u32::from_be_bytes(length) as usize;
        if length == 0 || length > MAX_REQUEST {
            return Err(format!("invalid request length {length}"));
        }
        let mut body = vec![0_u8; length];
        stream
            .read_exact(&mut body)
            .await
            .map_err(|e| format!("cannot read request body: {e}"))?;

        let mut authorized = true;
        let response = match IpcRequest::decode(&body) {
            Ok(request) => {
                if let Some((supplied_token, method)) = request.method.split_once(':') {
                    if let Some(router) = lookup_router(routers, supplied_token) {
                        if method.is_empty() {
                            IpcResponse::error("empty method name")
                        } else {
                            match router
                                .handle(
                                    method,
                                    IpcContext::new(request.working_dir),
                                    request.params,
                                )
                                .await
                            {
                                Ok(payload) => IpcResponse::success(payload),
                                Err(error) => IpcResponse::error(&error.to_string()),
                            }
                        }
                    } else {
                        authorized = false;
                        IpcResponse::error("unauthorized host IPC request")
                    }
                } else {
                    authorized = false;
                    IpcResponse::error("unauthorized host IPC request")
                }
            }
            Err(error) => IpcResponse::error(&error.to_string()),
        };

        stream
            .write_all(&response.encode())
            .await
            .map_err(|e| format!("cannot write response: {e}"))?;
        stream
            .flush()
            .await
            .map_err(|e| format!("cannot flush response: {e}"))?;
        if !authorized {
            return Ok(());
        }
    }
}
