//! MCP client implementation.
//!
//! The client connects to MCP servers and provides methods to list and call
//! tools, read resources, and so on. [`McpConnection`] wraps a client together
//! with the tools it discovered, ready to hand to an agent.
//!
//! Each client owns one receive loop (see [`McpClient::connect`]) that is the
//! only reader and writer of its transport. It routes responses to the
//! requests awaiting them, answers the server's `ping`, rejects server
//! requests for capabilities the client does not advertise, and hands
//! notifications to the [`McpHandler`] given at connect time.

mod connection;
mod handler;
#[cfg(all(test, feature = "http"))]
mod tests;
mod toolset;

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use aither_core::llm::tool::CancelSignal;
use arc_swap::ArcSwapOption;
use async_channel::{Receiver, Sender};
use executor_core::{Executor, Task};
use futures_lite::future;
use serde::de::DeserializeOwned;
use tracing::debug;

pub use handler::{McpHandler, ServerNotification};
pub use toolset::{
    McpConnection, McpServerConfig, McpServersConfig, McpTool, McpToolService,
    call_result_to_tool_result,
};

use connection::Command;

use crate::protocol::{
    CallToolParams, CallToolResult, InitializeParams, InitializeResult, JsonRpcNotification,
    JsonRpcRequest, ListToolsResult, McpError, McpToolDefinition, RequestId, Resource,
    ResourceContents, ServerCapabilities, ServerInfo,
};
use crate::transport::BidirectionalTransport;

/// Removes an in-flight request from the receive loop's `pending` map and
/// issues `notifications/cancelled` if its awaiting future is dropped or
/// cancelled before receiving a reply.
struct RequestGuard<'a> {
    commands: &'a Sender<Command>,
    id: Option<RequestId>,
    notify: bool,
    reason: Option<String>,
}

impl Drop for RequestGuard<'_> {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            let _ = self.commands.try_send(Command::Cancel {
                id,
                notify: self.notify,
                reason: self.reason.take(),
            });
        }
    }
}

/// MCP client for connecting to and interacting with MCP servers.
#[derive(Debug)]
pub struct McpClient {
    /// Requests and notifications for the receive loop to write.
    commands: Sender<Command>,
    /// Closes when the receive loop has finished.
    finished: Receiver<()>,
    /// Next request ID; the receive loop draws from it for its own requests.
    next_id: Arc<AtomicI64>,
    /// The server's tools as last listed.
    tools: Arc<ArcSwapOption<Vec<McpToolDefinition>>>,
    server_info: ServerInfo,
    capabilities: ServerCapabilities,
}

impl McpClient {
    /// Connect to an MCP server and perform initialization.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection or initialization fails.
    pub async fn connect<T, H, E>(executor: &E, transport: T, handler: H) -> Result<Self, McpError>
    where
        T: BidirectionalTransport + 'static,
        H: McpHandler,
        E: Executor,
    {
        let (commands, command_rx) = async_channel::unbounded();
        let (alive, finished) = async_channel::bounded(1);
        let next_id = Arc::new(AtomicI64::new(1));
        let tools = Arc::new(ArcSwapOption::empty());
        executor
            .spawn(connection::run(
                transport,
                command_rx,
                commands.downgrade(),
                Arc::new(handler),
                connection::ToolCache {
                    tools: tools.clone(),
                    next_id: next_id.clone(),
                },
                alive,
            ))
            .detach();

        let mut client = Self {
            commands,
            finished,
            next_id,
            tools,
            server_info: ServerInfo {
                name: String::new(),
                version: None,
            },
            capabilities: ServerCapabilities::default(),
        };
        client.initialize().await?;
        Ok(client)
    }

    async fn initialize(&mut self) -> Result<(), McpError> {
        let result: InitializeResult = self
            .request("initialize", Some(InitializeParams::default()))
            .await?;

        debug!(
            "Connected to MCP server: {} v{}",
            result.server_info.name,
            result.server_info.version.as_deref().unwrap_or("unknown")
        );

        self.server_info = result.server_info;
        self.capabilities = result.capabilities;

        let (reply, sent) = async_channel::bounded(1);
        self.commands
            .send(Command::Notification {
                notification: JsonRpcNotification::new("notifications/initialized"),
                reply,
            })
            .await
            .map_err(|_| McpError::ConnectionClosed)?;
        sent.recv().await.map_err(|_| McpError::ConnectionClosed)?
    }

    async fn request<P, R>(&self, method: &str, params: Option<P>) -> Result<R, McpError>
    where
        P: serde::Serialize,
        R: DeserializeOwned,
    {
        self.request_with_cancel(method, params, &CancelSignal::never())
            .await
    }

    async fn request_with_cancel<P, R>(
        &self,
        method: &str,
        params: Option<P>,
        cancel: &CancelSignal,
    ) -> Result<R, McpError>
    where
        P: serde::Serialize,
        R: DeserializeOwned,
    {
        if cancel.is_cancelled() {
            return Err(McpError::Cancelled);
        }

        let raw_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let id = RequestId::Number(raw_id);
        let mut request = JsonRpcRequest::new(id.clone(), method);
        request.params = params.map(serde_json::to_value).transpose()?;

        let (reply, response) = async_channel::bounded(1);
        self.commands
            .try_send(Command::Request { request, reply })
            .map_err(|_| McpError::ConnectionClosed)?;

        // Per the MCP specification, `initialize` must not be cancelled.
        let mut guard = RequestGuard {
            commands: &self.commands,
            id: Some(id),
            notify: method != "initialize",
            reason: None,
        };

        let outcome = future::or(async { Ok(response.recv().await) }, async {
            cancel.cancelled().await;
            Err(McpError::Cancelled)
        })
        .await;

        let received = match outcome {
            Ok(received) => received,
            Err(cancelled) => {
                guard.reason = Some("Request cancelled".to_string());
                return Err(cancelled);
            }
        };

        // The receive loop either answered or failed this request and removed
        // it from `pending`, so no cancel command is needed on drop.
        guard.id = None;

        let response = received.map_err(|_| McpError::ConnectionClosed)??;
        Ok(serde_json::from_value(
            response.into_result().map_err(McpError::JsonRpc)?,
        )?)
    }

    /// Get the server information.
    #[must_use]
    pub const fn server_info(&self) -> &ServerInfo {
        &self.server_info
    }

    /// Get the server capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> &ServerCapabilities {
        &self.capabilities
    }

    /// List available tools from the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>, McpError> {
        let result: ListToolsResult = self.request("tools/list", None::<()>).await?;

        debug!("Listed {} tools", result.tools.len());
        Ok(result.tools)
    }

    /// The server's tools as of the latest listing the client holds.
    #[must_use]
    pub fn tools(&self) -> Arc<Vec<McpToolDefinition>> {
        self.tools.load_full().unwrap_or_default()
    }

    fn seed_tools(&self, tools: Vec<McpToolDefinition>) {
        let tools = Arc::new(tools);
        self.tools
            .rcu(|current| current.clone().or_else(|| Some(tools.clone())));
    }

    /// Call a tool on the server.
    ///
    /// Dropping the returned future before the server answers removes the
    /// pending entry and sends `notifications/cancelled`.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<CallToolResult, McpError> {
        self.call_tool_with_cancel(name, arguments, &CancelSignal::never())
            .await
    }

    /// Call a tool on the server, aborting with `notifications/cancelled` if
    /// `cancel` fires before the response arrives.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::Cancelled`] if `cancel` fires, or an error if the
    /// request fails.
    pub async fn call_tool_with_cancel(
        &self,
        name: &str,
        arguments: serde_json::Value,
        cancel: &CancelSignal,
    ) -> Result<CallToolResult, McpError> {
        let params = CallToolParams {
            name: name.to_string(),
            arguments,
            meta: None,
        };
        let result: CallToolResult = self
            .request_with_cancel("tools/call", Some(params), cancel)
            .await?;

        debug!(
            "Tool {} returned {} content items",
            name,
            result.content.len()
        );
        Ok(result)
    }

    /// List available resources from the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or resources are not supported.
    pub async fn list_resources(&self) -> Result<Vec<Resource>, McpError> {
        #[derive(serde::Deserialize)]
        struct ListResourcesResult {
            resources: Vec<Resource>,
        }

        let result: ListResourcesResult = self.request("resources/list", None::<()>).await?;

        debug!("Listed {} resources", result.resources.len());
        Ok(result.resources)
    }

    /// Read a resource from the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn read_resource(&self, uri: &str) -> Result<ResourceContents, McpError> {
        #[derive(serde::Serialize)]
        struct ReadResourceParams {
            uri: String,
        }

        #[derive(serde::Deserialize)]
        struct ReadResourceResult {
            contents: Vec<ResourceContents>,
        }

        let params = ReadResourceParams {
            uri: uri.to_string(),
        };
        let result: ReadResourceResult = self.request("resources/read", Some(params)).await?;

        result
            .contents
            .into_iter()
            .next()
            .ok_or_else(|| McpError::Transport("No content returned".to_string()))
    }

    /// Close the client connection.
    pub async fn close(&self) {
        self.commands.close();
        let _ = self.finished.recv().await;
    }
}
