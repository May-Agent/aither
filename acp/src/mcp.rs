//! Connecting the MCP servers a client offers a session.
//!
//! [`SessionSetupParams::mcp_servers`](crate::domain::SessionSetupParams)
//! lists the MCP servers the client wants the agent to speak for it.
//! [`connect_all`] opens each of them concurrently — stdio servers as
//! child processes, HTTP servers over [`HttpTransport`] — and returns the
//! ready [`McpConnection`]s in declaration order.

use aither_mcp::transport::HttpTransport;
use aither_mcp::{McpConnection, McpError};
use async_process::Command;
use executor_core::AnyExecutor;

use crate::domain::McpServer;

/// Why a session's MCP servers could not be connected.
#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    /// The server uses a transport the agent did not advertise.
    #[error(
        "MCP server '{server}' uses transport '{transport}', which this agent does not support"
    )]
    Unsupported {
        /// Name of the MCP server (or its transport tag, when unnamed).
        server: String,
        /// The transport `type` the client declared (`sse`, a custom
        /// `_`-extension, or an unrecognized tag).
        transport: String,
    },
    /// The server did not connect.
    #[error("could not connect MCP server '{server}': {source}")]
    Failed {
        /// Name of the MCP server that failed to connect.
        server: String,
        /// Underlying MCP connection error.
        #[source]
        source: McpError,
    },
}

/// Connects every server concurrently and returns the connections in the
/// order the servers were given.
///
/// # Errors
///
/// Returns [`ConnectError::Unsupported`] if any server uses a transport
/// other than stdio or HTTP (the legacy `sse` transport included), or
/// [`ConnectError::Failed`] if spawning or handshaking with a server fails.
pub async fn connect_all(
    executor: &AnyExecutor,
    servers: &[McpServer],
) -> Result<Vec<McpConnection>, ConnectError> {
    futures_util::future::try_join_all(servers.iter().map(|server| connect(executor, server))).await
}

async fn connect(
    executor: &AnyExecutor,
    server: &McpServer,
) -> Result<McpConnection, ConnectError> {
    let connected = match server {
        McpServer::Stdio(server) => {
            let mut command = Command::new(&server.command);
            command
                .args(&server.args)
                .envs(server.env.iter().map(|var| (&var.name, &var.value)));
            McpConnection::spawn_command(executor, &mut command).await
        }
        McpServer::Http(server) => {
            let transport = server
                .headers
                .iter()
                .fold(HttpTransport::new(&server.url), |transport, header| {
                    transport.with_header(&header.name, &header.value)
                });
            McpConnection::connect(executor, transport, ()).await
        }
        McpServer::Custom(tag, raw) => {
            return Err(ConnectError::Unsupported {
                server: name_of(raw).unwrap_or_else(|| tag.as_str().to_string()),
                transport: tag.as_str().to_string(),
            });
        }
        McpServer::Unknown(tag, raw) => {
            return Err(ConnectError::Unsupported {
                server: name_of(raw).unwrap_or_else(|| tag.clone()),
                transport: tag.clone(),
            });
        }
    };
    connected.map_err(|source| ConnectError::Failed {
        server: server_name(server).to_string(),
        source,
    })
}

/// The display name a server declaration carries.
fn server_name(server: &McpServer) -> &str {
    match server {
        McpServer::Stdio(server) => &server.name,
        McpServer::Http(server) => &server.name,
        McpServer::Custom(tag, _) => tag.as_str(),
        McpServer::Unknown(tag, _) => tag.as_str(),
    }
}

/// The `name` field of a preserved raw server object, when it has one.
fn name_of(raw: &serde_json::Value) -> Option<String> {
    raw.get("name")?.as_str().map(str::to_owned)
}
