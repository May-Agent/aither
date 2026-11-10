//! Where a [`Sandbox`](crate::Sandbox) runs its processes.

use heel::HeelBinary;

use crate::spawn::PreparedSsh;
use crate::{ContainerPath, ContainerShellRuntime, PreparedTools, SshServer};

/// Where a sandbox's processes run. The embedder supplies what each backend
/// needs and pins: the verified `heel` binary, the tools prepared in the
/// container, the server's preparation.
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
        /// The tools prepared at a path inside the container.
        tools: PreparedTools<ContainerPath>,
    },
    /// A remote machine over SSH.
    Ssh {
        /// The server processes run on.
        server: SshServer,
        /// What the embedder prepared on it: its `heel` and its tools.
        prepared: PreparedSsh,
    },
}
