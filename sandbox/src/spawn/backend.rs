//! Where a [`Sandbox`](crate::Sandbox) runs its processes.

use heel::HeelBinary;

use super::launch::{ContainerLaunch, RemoteLaunch};
use crate::{ContainerShellRuntime, SshServer};

/// Where a sandbox's processes run. The embedder supplies what each backend
/// runs: the verified `heel` binary, the launch of the programs prepared in
/// the container, the server's `heel` and launch.
#[derive(Debug, Clone)]
pub enum Backend {
    /// This machine, under heel's sandbox.
    Host {
        /// The verified `heel` binary the sandbox runs under.
        heel: HeelBinary,
    },
    /// A running container, whose own limits are the boundary.
    Container {
        /// The container: its exec, its id, the workspace mount that maps host
        /// paths into it and the [`HostRoute`](crate::HostRoute) that says
        /// where the host is reachable from inside.
        runtime: ContainerShellRuntime,
        /// The programs prepared inside the container.
        launch: ContainerLaunch,
    },
    /// A remote machine over SSH.
    Ssh {
        /// The server processes run on.
        server: SshServer,
        /// The `heel` binary on the server, as a path there; a confined
        /// process needs it, an unconfined one does not.
        heel: Option<String>,
        /// The programs prepared on the server.
        launch: RemoteLaunch,
    },
}
