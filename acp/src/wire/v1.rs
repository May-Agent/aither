//! ACP v1 wire model.
//!
//! Mechanical port of the legacy `protocol/*` types: v1 field names and
//! shapes are preserved exactly (camelCase serde, `clientInfo`,
//! `agentCapabilities`, untagged stdio `McpServer`, `Other(Value)`
//! update fallback). The [`crate::wire::codec`] module maps these onto
//! the canonical [`crate::domain`] v2 model.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use serde_json::Value;

/// A JSON-RPC request id (`string | number`), as used by v1 elicitation
/// request scopes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    /// A numeric request id.
    Number(i64),
    /// A string request id.
    String(String),
}

impl From<i64> for RequestId {
    fn from(id: i64) -> Self {
        Self::Number(id)
    }
}
impl From<u64> for RequestId {
    fn from(id: u64) -> Self {
        // `Number` is `i64`; a `u64` that exceeds it saturates rather than
        // wrapping into a negative id.
        Self::Number(i64::try_from(id).unwrap_or(i64::MAX))
    }
}
impl From<&str> for RequestId {
    fn from(id: &str) -> Self {
        Self::String(id.to_string())
    }
}
impl From<String> for RequestId {
    fn from(id: String) -> Self {
        Self::String(id)
    }
}

// ===== old protocol/types.rs =====

/// ACP protocol version implemented by this crate.
pub const PROTOCOL_VERSION: u16 = 1;

// =============================================================================
// Initialization
// =============================================================================

/// Initialize request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    /// Protocol version the client supports.
    pub protocol_version: u16,
    /// Client capabilities.
    #[serde(default)]
    pub client_capabilities: ClientCapabilities,
    /// Client information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_info: Option<Implementation>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Initialize response result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    /// Protocol version the agent supports.
    pub protocol_version: u16,
    /// Agent capabilities.
    #[serde(default)]
    pub agent_capabilities: AgentCapabilities,
    /// Agent information.
    #[serde(default)]
    pub agent_info: Option<Implementation>,
    /// Supported authentication methods.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auth_methods: Vec<AuthMethod>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Implementation info (client or agent).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Implementation {
    /// Name.
    pub name: String,
    /// Display title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Version.
    pub version: String,
}

/// Client capabilities.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// File system capabilities.
    #[serde(default)]
    pub fs: FileSystemCapability,
    /// Whether terminal is supported.
    #[serde(default)]
    pub terminal: bool,
    /// What the client offers for `authenticate`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<ClientAuthCapabilities>,
    /// What the client offers for `elicitation/create`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elicitation: Option<ElicitationCapabilities>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Capability keys this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// File system capability.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileSystemCapability {
    /// Can read text files.
    #[serde(default)]
    pub read_text_file: bool,
    /// Can write text files.
    #[serde(default)]
    pub write_text_file: bool,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Agent capabilities.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    /// Whether session loading is supported.
    #[serde(default)]
    pub load_session: bool,
    /// Prompt capabilities.
    #[serde(default)]
    pub prompt_capabilities: PromptCapabilities,
    /// MCP capabilities.
    #[serde(default)]
    pub mcp_capabilities: McpCapabilities,
    /// Session capabilities.
    #[serde(default)]
    pub session_capabilities: SessionCapabilities,
    /// What the agent offers for `authenticate`/`logout`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<AgentAuthCapabilities>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Capability keys this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Prompt capabilities.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptCapabilities {
    /// Image support.
    #[serde(default)]
    pub image: bool,
    /// Audio support.
    #[serde(default)]
    pub audio: bool,
    /// Embedded context support.
    #[serde(default)]
    pub embedded_context: bool,
}

/// MCP capabilities.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpCapabilities {
    /// HTTP MCP support.
    #[serde(default)]
    pub http: bool,
    /// SSE MCP support.
    #[serde(default)]
    pub sse: bool,
}

/// Session capabilities supported by the agent.
///
/// Each flag is an optional empty object: omitted or `null` means the method
/// family is not supported, `{}` means it is.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCapabilities {
    /// Whether the agent supports `session/list`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub list: Option<SessionListCapabilities>,
    /// Whether the agent supports `session/delete`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delete: Option<SessionDeleteCapabilities>,
    /// Whether the agent supports `additionalDirectories` on session
    /// lifecycle requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_directories: Option<SessionAdditionalDirectoriesCapabilities>,
    /// Whether the agent supports `session/resume`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume: Option<SessionResumeCapabilities>,
    /// Whether the agent supports `session/fork`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fork: Option<SessionForkCapabilities>,
    /// Whether the agent supports `session/close`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub close: Option<SessionCloseCapabilities>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Session capability keys this crate does not model (e.g. `subagents`),
    /// preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Capabilities for the `session/list` method.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionListCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Capabilities for the `session/delete` method.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionDeleteCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Capabilities for `additionalDirectories` support on session lifecycle
/// requests.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionAdditionalDirectoriesCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Capabilities for the `session/resume` method.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionResumeCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Capabilities for the `session/fork` method.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionForkCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Capabilities for the `session/close` method.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCloseCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

// =============================================================================
// Session Management
// =============================================================================

/// An MCP server offered to the agent in `session/new`, `session/load`,
/// `session/resume`, or `session/fork`.
///
/// Local stdio servers carry no `type` field; remote servers are tagged
/// `"type": "http"` or `"type": "sse"`. Whether an agent accepts remote
/// servers is negotiated through [`McpCapabilities`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpServer {
    /// A remote server reached over streamable HTTP.
    #[serde(rename = "http")]
    Http(McpServerHttp),
    /// A remote server reached over SSE.
    #[serde(rename = "sse")]
    Sse(McpServerSse),
    /// A local process the agent spawns; sent with no `type` field.
    #[serde(untagged)]
    Stdio(McpServerStdio),
}

impl McpServer {
    /// The server name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Http(server) => &server.name,
            Self::Sse(server) => &server.name,
            Self::Stdio(server) => &server.name,
        }
    }
}

impl From<McpServerStdio> for McpServer {
    fn from(server: McpServerStdio) -> Self {
        Self::Stdio(server)
    }
}

impl From<McpServerHttp> for McpServer {
    fn from(server: McpServerHttp) -> Self {
        Self::Http(server)
    }
}

impl From<McpServerSse> for McpServer {
    fn from(server: McpServerSse) -> Self {
        Self::Sse(server)
    }
}

/// A local MCP server the agent spawns as a child process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerStdio {
    /// Server name.
    pub name: String,
    /// Command to run.
    pub command: String,
    /// Command arguments.
    ///
    /// Always written, even when empty: `args` is required by the schema's
    /// stdio variant, and agents that deserialize `McpServer` as a strict
    /// untagged enum — Devin's does — refuse the object without it.
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment variables.
    ///
    /// Required by the schema for the same reason `args` is.
    #[serde(default)]
    pub env: Vec<EnvVar>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl McpServerStdio {
    /// Build a stdio server spec.
    #[must_use]
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            args: Vec::new(),
            env: Vec::new(),
            meta: None,
        }
    }

    /// Set command arguments.
    #[must_use]
    pub fn args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Set environment variables.
    #[must_use]
    pub fn env(mut self, env: Vec<EnvVar>) -> Self {
        self.env = env;
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// A remote MCP server reached over streamable HTTP.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerHttp {
    /// Server name.
    pub name: String,
    /// Endpoint URL.
    pub url: String,
    /// HTTP headers to send with requests.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub headers: Vec<HttpHeader>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl McpServerHttp {
    /// Build an HTTP server spec.
    #[must_use]
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            headers: Vec::new(),
            meta: None,
        }
    }

    /// Set request headers.
    #[must_use]
    pub fn headers(mut self, headers: Vec<HttpHeader>) -> Self {
        self.headers = headers;
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// A remote MCP server reached over SSE.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerSse {
    /// Server name.
    pub name: String,
    /// Endpoint URL.
    pub url: String,
    /// HTTP headers to send with requests.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub headers: Vec<HttpHeader>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl McpServerSse {
    /// Build an SSE server spec.
    #[must_use]
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            headers: Vec::new(),
            meta: None,
        }
    }

    /// Set request headers.
    #[must_use]
    pub fn headers(mut self, headers: Vec<HttpHeader>) -> Self {
        self.headers = headers;
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// Environment variable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvVar {
    /// Variable name.
    pub name: String,
    /// Variable value.
    pub value: String,
}

/// HTTP header on a remote MCP server request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HttpHeader {
    /// Header name.
    pub name: String,
    /// Header value.
    pub value: String,
}

impl HttpHeader {
    /// Build a header.
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// Create new session request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNewParams {
    /// Working directory. Must be absolute.
    pub cwd: PathBuf,
    /// MCP servers to connect to.
    #[serde(default)]
    pub mcp_servers: Vec<McpServer>,
    /// Additional workspace roots. Each path must be absolute and is only
    /// honoured when the agent advertises
    /// [`SessionCapabilities::additional_directories`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_directories: Vec<PathBuf>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionNewParams {
    /// Build params for a session rooted at `cwd`.
    #[must_use]
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            mcp_servers: Vec::new(),
            additional_directories: Vec::new(),
            meta: None,
        }
    }

    /// Set the MCP servers to connect.
    #[must_use]
    pub fn mcp_servers(mut self, mcp_servers: Vec<McpServer>) -> Self {
        self.mcp_servers = mcp_servers;
        self
    }

    /// Set additional workspace roots.
    #[must_use]
    pub fn additional_directories(mut self, additional_directories: Vec<PathBuf>) -> Self {
        self.additional_directories = additional_directories;
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// Create new session response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNewResult {
    /// Session ID.
    pub session_id: String,
    /// Initial mode state, if the agent supports session modes.
    #[serde(default)]
    pub modes: Option<SessionModeState>,
    /// Initial configuration options, if the agent supports them.
    #[serde(default)]
    pub config_options: Option<Vec<ConfigOption>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Load session request parameters (`session/load`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionLoadParams {
    /// Session ID to load.
    pub session_id: String,
    /// Working directory. Must match the session's `cwd`.
    pub cwd: PathBuf,
    /// MCP servers to connect to.
    #[serde(default)]
    pub mcp_servers: Vec<McpServer>,
    /// Additional workspace roots. Each path must be absolute and must be
    /// re-sent here — the agent does not restore them implicitly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_directories: Vec<PathBuf>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionLoadParams {
    /// Build params loading `session_id` rooted at `cwd`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, cwd: impl Into<PathBuf>) -> Self {
        Self {
            session_id: session_id.into(),
            cwd: cwd.into(),
            mcp_servers: Vec::new(),
            additional_directories: Vec::new(),
            meta: None,
        }
    }

    /// Set the MCP servers to connect.
    #[must_use]
    pub fn mcp_servers(mut self, mcp_servers: Vec<McpServer>) -> Self {
        self.mcp_servers = mcp_servers;
        self
    }

    /// Set additional workspace roots.
    #[must_use]
    pub fn additional_directories(mut self, additional_directories: Vec<PathBuf>) -> Self {
        self.additional_directories = additional_directories;
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// Load session response (`session/load`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionLoadResult {
    /// Initial mode state, if the agent supports session modes.
    #[serde(default)]
    pub modes: Option<SessionModeState>,
    /// Configuration options, if the agent supports them.
    #[serde(default)]
    pub config_options: Option<Vec<ConfigOption>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Resume session request parameters (`session/resume`).
///
/// Unlike `session/load`, resuming restores the session's context inside the
/// agent without replaying its history back as `session/update`
/// notifications — for clients that do not need the transcript re-sent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionResumeParams {
    /// Session ID to resume.
    pub session_id: String,
    /// Working directory. Must match the session's `cwd`.
    pub cwd: PathBuf,
    /// MCP servers to connect to.
    #[serde(default)]
    pub mcp_servers: Vec<McpServer>,
    /// Additional workspace roots. Each path must be absolute and must be
    /// re-sent here — the agent does not restore them implicitly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_directories: Vec<PathBuf>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionResumeParams {
    /// Build params resuming `session_id` rooted at `cwd`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, cwd: impl Into<PathBuf>) -> Self {
        Self {
            session_id: session_id.into(),
            cwd: cwd.into(),
            mcp_servers: Vec::new(),
            additional_directories: Vec::new(),
            meta: None,
        }
    }

    /// Set the MCP servers to connect.
    #[must_use]
    pub fn mcp_servers(mut self, mcp_servers: Vec<McpServer>) -> Self {
        self.mcp_servers = mcp_servers;
        self
    }

    /// Set additional workspace roots.
    #[must_use]
    pub fn additional_directories(mut self, additional_directories: Vec<PathBuf>) -> Self {
        self.additional_directories = additional_directories;
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// Resume session response (`session/resume`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionResumeResult {
    /// Initial mode state, if the agent supports session modes.
    #[serde(default)]
    pub modes: Option<SessionModeState>,
    /// Configuration options, if the agent supports them.
    #[serde(default)]
    pub config_options: Option<Vec<ConfigOption>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Fork session request parameters (`session/fork`).
///
/// Creates a new session seeded with the conversation history of `session_id`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionForkParams {
    /// Source session ID to fork from.
    pub session_id: String,
    /// Working directory for the forked session. Must be absolute.
    pub cwd: PathBuf,
    /// MCP servers to connect to in the forked session.
    #[serde(default)]
    pub mcp_servers: Vec<McpServer>,
    /// Additional workspace roots for the forked session. Each path must be
    /// absolute and is only honoured when the agent advertises
    /// [`SessionCapabilities::additional_directories`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_directories: Vec<PathBuf>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionForkParams {
    /// Build params forking `session_id` into a new session rooted at `cwd`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, cwd: impl Into<PathBuf>) -> Self {
        Self {
            session_id: session_id.into(),
            cwd: cwd.into(),
            mcp_servers: Vec::new(),
            additional_directories: Vec::new(),
            meta: None,
        }
    }

    /// Set the MCP servers to connect.
    #[must_use]
    pub fn mcp_servers(mut self, mcp_servers: Vec<McpServer>) -> Self {
        self.mcp_servers = mcp_servers;
        self
    }

    /// Set additional workspace roots.
    #[must_use]
    pub fn additional_directories(mut self, additional_directories: Vec<PathBuf>) -> Self {
        self.additional_directories = additional_directories;
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// Fork session response (`session/fork`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionForkResult {
    /// Newly created forked session ID.
    pub session_id: String,
    /// Initial mode state, if the agent supports session modes.
    #[serde(default)]
    pub modes: Option<SessionModeState>,
    /// Initial configuration options, if the agent supports them.
    #[serde(default)]
    pub config_options: Option<Vec<ConfigOption>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Set session mode request parameters (`session/set_mode`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSetModeParams {
    /// Session ID.
    pub session_id: String,
    /// Mode to activate.
    pub mode_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionSetModeParams {
    /// Build params activating `mode_id` on `session_id`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, mode_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            mode_id: mode_id.into(),
            meta: None,
        }
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// Set session mode response (`session/set_mode`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSetModeResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Set config option request parameters (`session/set_config_option`).
///
/// The `value` field flattens into the params: a [`SessionConfigValue::Select`]
/// produces `{"value": "id"}` while [`SessionConfigValue::Boolean`] produces
/// `{"type": "boolean", "value": true}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSetConfigOptionParams {
    /// Session ID.
    pub session_id: String,
    /// Config option to set.
    pub config_id: String,
    /// New value for the option.
    #[serde(flatten)]
    pub value: SessionConfigValue,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionSetConfigOptionParams {
    /// Build params setting `config_id` on `session_id`. `value` accepts
    /// `&str`/`String` for `select` options and `bool` for `boolean` options.
    #[must_use]
    pub fn new(
        session_id: impl Into<String>,
        config_id: impl Into<String>,
        value: impl Into<SessionConfigValue>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            config_id: config_id.into(),
            value: value.into(),
            meta: None,
        }
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// Set config option response (`session/set_config_option`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSetConfigOptionResult {
    /// The full set of configuration options and their current values.
    #[serde(default)]
    pub config_options: Vec<ConfigOption>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// New value for a session config option in `session/set_config_option`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SessionConfigValue {
    /// `select` option: the chosen value ID.
    Select {
        /// Selected value ID.
        value: String,
    },
    /// `boolean` option: the new flag state.
    Boolean {
        /// Wire tag; always `"boolean"`.
        #[serde(rename = "type")]
        kind: BooleanConfigTag,
        /// Flag state.
        value: bool,
    },
}

/// Wire tag marking a `boolean` config option write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BooleanConfigTag {
    /// The literal `"boolean"`.
    #[serde(rename = "boolean")]
    Boolean,
}

impl From<String> for SessionConfigValue {
    fn from(value: String) -> Self {
        Self::Select { value }
    }
}

impl From<&str> for SessionConfigValue {
    fn from(value: &str) -> Self {
        Self::Select {
            value: value.to_string(),
        }
    }
}

impl From<bool> for SessionConfigValue {
    fn from(value: bool) -> Self {
        Self::Boolean {
            kind: BooleanConfigTag::Boolean,
            value,
        }
    }
}

/// The set of modes a session can be in and the one currently active.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionModeState {
    /// The currently active mode.
    pub current_mode_id: String,
    /// Modes the agent can operate in.
    #[serde(default)]
    pub available_modes: Vec<SessionMode>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A mode the agent can operate in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMode {
    /// Stable identifier used to select this mode.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Optional human-readable details.
    #[serde(default)]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A session configuration option and its current state.
///
/// `select` options carry `current_value` (a value ID) and `options`;
/// `boolean` options carry a boolean `current_value` and no `options`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigOption {
    /// Unique identifier for the configuration option.
    pub id: String,
    /// Human-readable label for the option.
    pub name: String,
    /// Optional description for the client to display to the user.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional semantic category (e.g. `"mode"`, `"model"`,
    /// `"thought_level"`). Unknown categories are kept verbatim.
    #[serde(default)]
    pub category: Option<String>,
    /// Option type: `"select"`, `"boolean"`, or an agent-defined value.
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    /// Currently selected value.
    #[serde(default)]
    pub current_value: Option<ConfigOptionValue>,
    /// Selectable values for `select` options.
    #[serde(default)]
    pub options: Option<ConfigSelectOptions>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Current value of a [`ConfigOption`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigOptionValue {
    /// Selected value ID for `select` options.
    Selected(String),
    /// Flag state for `boolean` options.
    Toggle(bool),
}

/// Possible values for a `select` config option: a flat list or a list of
/// groups.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigSelectOptions {
    /// Flat list of options with no grouping.
    Flat(Vec<ConfigSelectOption>),
    /// Options grouped under headers.
    Grouped(Vec<ConfigSelectGroup>),
}

/// A possible value for a `select` config option.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigSelectOption {
    /// Unique identifier for this option value.
    pub value: String,
    /// Human-readable label for this option value.
    pub name: String,
    /// Optional description for this option value.
    #[serde(default)]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A group of `select` config options under a header.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigSelectGroup {
    /// Group identifier.
    pub group: String,
    /// Human-readable group label.
    pub name: String,
    /// Options in this group.
    pub options: Vec<ConfigSelectOption>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

// =============================================================================
// Prompt
// =============================================================================

/// Prompt request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptParams {
    /// Session ID.
    pub session_id: String,
    /// Prompt content.
    pub prompt: Vec<ContentBlock>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl PromptParams {
    /// Build params prompting `session_id` with `prompt`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, prompt: Vec<ContentBlock>) -> Self {
        Self {
            session_id: session_id.into(),
            prompt,
            meta: None,
        }
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// Prompt response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptResult {
    /// Reason for stopping.
    pub stop_reason: StopReason,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Stop reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Normal end of turn.
    EndTurn,
    /// Cancelled by user.
    Cancelled,
    /// Error occurred.
    Error,
    /// Max tokens reached.
    MaxTokens,
    /// Max turn requests reached.
    MaxTurnRequests,
    /// The agent refused to respond.
    Refusal,
}

/// Cancel notification parameters (`session/cancel`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCancelParams {
    /// Session ID.
    pub session_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionCancelParams {
    /// Build params cancelling `session_id`'s current turn.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            meta: None,
        }
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

impl From<&str> for SessionCancelParams {
    fn from(session_id: &str) -> Self {
        Self::new(session_id)
    }
}

impl From<String> for SessionCancelParams {
    fn from(session_id: String) -> Self {
        Self::new(session_id)
    }
}

// =============================================================================
// Content Types
// =============================================================================

/// Content block (text, image, audio, resource, resource link).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ContentBlock {
    /// Text content.
    Text(TextContent),
    /// Image content.
    Image(ImageContent),
    /// Audio content.
    Audio(AudioContent),
    /// Resource content.
    Resource(ResourceContent),
    /// A link to a resource without embedding its contents.
    #[serde(rename = "resource_link")]
    ResourceLink(ResourceLink),
}

/// Text content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextContent {
    /// Text value.
    pub text: String,
    /// Annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Image content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64-encoded data. Absent when `uri` carries the image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// URI carrying the image instead of inline `data`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// MIME type.
    pub mime_type: String,
    /// Annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Audio content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioContent {
    /// Base64-encoded data.
    pub data: String,
    /// MIME type.
    pub mime_type: String,
    /// Annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Resource content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceContent {
    /// Resource data.
    pub resource: EmbeddedResource,
    /// Annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Embedded resource: carries either `text` or base64 `blob` contents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddedResource {
    /// Resource URI.
    pub uri: String,
    /// MIME type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Text contents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Base64-encoded binary contents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A link to a resource without embedding its contents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceLink {
    /// Resource URI.
    pub uri: String,
    /// Resource name.
    pub name: String,
    /// Human-readable title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// MIME type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Size in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<i64>,
    /// Annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl ResourceLink {
    /// Build a resource link.
    #[must_use]
    pub fn new(uri: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            name: name.into(),
            title: None,
            description: None,
            mime_type: None,
            size: None,
            annotations: None,
            meta: None,
        }
    }
}

/// Content chunk for streaming.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentChunk {
    /// Content.
    pub content: ContentBlock,
    /// Optional identifier correlating chunks that belong to one message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

// =============================================================================
// Session Updates (Notifications)
// =============================================================================

/// Session update notification parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNotification {
    /// Session ID.
    pub session_id: String,
    /// Update data.
    pub update: SessionUpdate,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl SessionNotification {
    /// Build a `session/update` notification for `session_id`.
    #[must_use]
    pub fn new(session_id: impl Into<String>, update: SessionUpdate) -> Self {
        Self {
            session_id: session_id.into(),
            update,
            meta: None,
            extra: BTreeMap::new(),
        }
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// A session update streamed from the agent in a `session/update`
/// notification.
///
/// Updates whose `sessionUpdate` tag or payload this crate does not model
/// deserialize into [`SessionUpdate::Other`], keeping the raw object, so an
/// unknown update can never fail parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionUpdate {
    /// Agent thinking chunk.
    AgentThoughtChunk(ContentChunk),
    /// Agent message chunk.
    AgentMessageChunk(ContentChunk),
    /// User message chunk (for session replay).
    UserMessageChunk(ContentChunk),
    /// Plan update.
    Plan(Plan),
    /// Tool call initiated.
    ToolCall(ToolCall),
    /// Tool call progress/completion.
    ToolCallUpdate(ToolCallUpdate),
    /// Available commands are ready or have changed.
    AvailableCommandsUpdate(AvailableCommandsUpdate),
    /// The session's current mode changed.
    CurrentModeUpdate(CurrentModeUpdate),
    /// Session configuration options were updated.
    ConfigOptionUpdate(ConfigOptionUpdate),
    /// Session metadata (title, timestamps) changed.
    SessionInfoUpdate(SessionInfoUpdate),
    /// Context window or cost usage changed.
    UsageUpdate(UsageUpdate),
    /// An update this crate does not model; the raw object is preserved.
    Other(Value),
}

/// Serialization view of [`SessionUpdate`] for known variants.
#[derive(Serialize)]
#[serde(tag = "sessionUpdate", rename_all = "snake_case")]
enum SessionUpdateRepr<'a> {
    AgentThoughtChunk(&'a ContentChunk),
    AgentMessageChunk(&'a ContentChunk),
    UserMessageChunk(&'a ContentChunk),
    Plan(&'a Plan),
    ToolCall(&'a ToolCall),
    ToolCallUpdate(&'a ToolCallUpdate),
    AvailableCommandsUpdate(&'a AvailableCommandsUpdate),
    CurrentModeUpdate(&'a CurrentModeUpdate),
    ConfigOptionUpdate(&'a ConfigOptionUpdate),
    SessionInfoUpdate(&'a SessionInfoUpdate),
    UsageUpdate(&'a UsageUpdate),
}

/// Deserialization view of [`SessionUpdate`] for known variants.
#[derive(Deserialize)]
#[serde(tag = "sessionUpdate", rename_all = "snake_case")]
enum SessionUpdateOwned {
    AgentThoughtChunk(ContentChunk),
    AgentMessageChunk(ContentChunk),
    UserMessageChunk(ContentChunk),
    Plan(Plan),
    ToolCall(ToolCall),
    ToolCallUpdate(ToolCallUpdate),
    AvailableCommandsUpdate(AvailableCommandsUpdate),
    CurrentModeUpdate(CurrentModeUpdate),
    ConfigOptionUpdate(ConfigOptionUpdate),
    SessionInfoUpdate(SessionInfoUpdate),
    UsageUpdate(UsageUpdate),
}

impl Serialize for SessionUpdate {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let repr = match self {
            Self::AgentThoughtChunk(chunk) => SessionUpdateRepr::AgentThoughtChunk(chunk),
            Self::AgentMessageChunk(chunk) => SessionUpdateRepr::AgentMessageChunk(chunk),
            Self::UserMessageChunk(chunk) => SessionUpdateRepr::UserMessageChunk(chunk),
            Self::Plan(plan) => SessionUpdateRepr::Plan(plan),
            Self::ToolCall(call) => SessionUpdateRepr::ToolCall(call),
            Self::ToolCallUpdate(update) => SessionUpdateRepr::ToolCallUpdate(update),
            Self::AvailableCommandsUpdate(update) => {
                SessionUpdateRepr::AvailableCommandsUpdate(update)
            }
            Self::CurrentModeUpdate(update) => SessionUpdateRepr::CurrentModeUpdate(update),
            Self::ConfigOptionUpdate(update) => SessionUpdateRepr::ConfigOptionUpdate(update),
            Self::SessionInfoUpdate(update) => SessionUpdateRepr::SessionInfoUpdate(update),
            Self::UsageUpdate(update) => SessionUpdateRepr::UsageUpdate(update),
            Self::Other(raw) => return raw.serialize(serializer),
        };
        repr.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SessionUpdate {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Ok(SessionUpdateOwned::deserialize(&value).map_or_else(
            // Unknown `sessionUpdate` tag or a payload this crate does not
            // model: keep the raw object instead of failing.
            |_| Self::Other(value),
            Self::from,
        ))
    }
}

impl From<SessionUpdateOwned> for SessionUpdate {
    fn from(owned: SessionUpdateOwned) -> Self {
        match owned {
            SessionUpdateOwned::AgentThoughtChunk(chunk) => Self::AgentThoughtChunk(chunk),
            SessionUpdateOwned::AgentMessageChunk(chunk) => Self::AgentMessageChunk(chunk),
            SessionUpdateOwned::UserMessageChunk(chunk) => Self::UserMessageChunk(chunk),
            SessionUpdateOwned::Plan(plan) => Self::Plan(plan),
            SessionUpdateOwned::ToolCall(call) => Self::ToolCall(call),
            SessionUpdateOwned::ToolCallUpdate(update) => Self::ToolCallUpdate(update),
            SessionUpdateOwned::AvailableCommandsUpdate(update) => {
                Self::AvailableCommandsUpdate(update)
            }
            SessionUpdateOwned::CurrentModeUpdate(update) => Self::CurrentModeUpdate(update),
            SessionUpdateOwned::ConfigOptionUpdate(update) => Self::ConfigOptionUpdate(update),
            SessionUpdateOwned::SessionInfoUpdate(update) => Self::SessionInfoUpdate(update),
            SessionUpdateOwned::UsageUpdate(update) => Self::UsageUpdate(update),
        }
    }
}

/// `available_commands_update` payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AvailableCommandsUpdate {
    /// Commands the agent can execute.
    #[serde(default)]
    pub available_commands: Vec<AvailableCommand>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A command the agent can execute.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvailableCommand {
    /// Command name.
    pub name: String,
    /// Human-readable description of what the command does.
    pub description: String,
    /// Input specification, if the command takes input.
    #[serde(default)]
    pub input: Option<AvailableCommandInput>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Input specification for an [`AvailableCommand`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvailableCommandInput {
    /// Hint shown when no input has been provided.
    pub hint: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `current_mode_update` payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CurrentModeUpdate {
    /// The ID of the now-current mode.
    pub current_mode_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `config_option_update` payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigOptionUpdate {
    /// The full set of configuration options and their current values.
    #[serde(default)]
    pub config_options: Vec<ConfigOption>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `session_info_update` payload. All fields are optional to support partial
/// updates.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfoUpdate {
    /// Human-readable title for the session. `null` clears it.
    #[serde(default)]
    pub title: Option<String>,
    /// ISO 8601 timestamp of last activity. `null` clears it.
    #[serde(default)]
    pub updated_at: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `usage_update` payload: context window and cost usage for a session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageUpdate {
    /// Tokens currently in context.
    pub used: u64,
    /// Total context window size in tokens.
    pub size: u64,
    /// Cumulative session cost.
    #[serde(default)]
    pub cost: Option<Cost>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Cost information for a session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// Total cumulative cost for the session.
    pub amount: f64,
    /// ISO 4217 currency code (e.g. `"USD"`).
    pub currency: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

// =============================================================================
// Plan
// =============================================================================

/// Execution plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    /// Plan entries.
    pub entries: Vec<PlanEntry>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Plan entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanEntry {
    /// Entry content.
    pub content: String,
    /// Entry status.
    pub status: PlanEntryStatus,
    /// Priority.
    #[serde(default)]
    pub priority: PlanEntryPriority,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Plan entry status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanEntryStatus {
    /// Not started.
    #[default]
    Pending,
    /// In progress.
    InProgress,
    /// Completed.
    Completed,
}

/// Plan entry priority.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlanEntryPriority {
    /// Low priority.
    Low,
    /// Medium priority.
    #[default]
    Medium,
    /// High priority.
    High,
}

// =============================================================================
// Tool Calls
// =============================================================================

/// Tool call initiated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    /// Unique tool call ID.
    pub tool_call_id: String,
    /// Human-readable title; agents in the wild omit it, so tolerate absence.
    #[serde(default)]
    pub title: String,
    /// Tool kind.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<ToolKind>,
    /// Current status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<ToolCallStatus>,
    /// Content produced.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<ToolCallContent>,
    /// Affected locations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locations: Vec<ToolCallLocation>,
    /// Raw input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_input: Option<Value>,
    /// Raw output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Tool call update.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallUpdate {
    /// Tool call ID.
    pub tool_call_id: String,
    /// Updated status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<ToolCallStatus>,
    /// New content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ToolCallContent>>,
    /// Updated title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Updated kind.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<ToolKind>,
    /// Updated locations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locations: Option<Vec<ToolCallLocation>>,
    /// Raw input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_input: Option<Value>,
    /// Raw output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Tool kind.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// Read operation.
    Read,
    /// Write/edit operation.
    Edit,
    /// Delete operation.
    Delete,
    /// Move/rename operation.
    Move,
    /// Search operation.
    Search,
    /// Execute/run operation.
    Execute,
    /// Internal reasoning or planning.
    Think,
    /// Retrieving external data.
    Fetch,
    /// Switching the session mode.
    SwitchMode,
    /// Other operation.
    #[default]
    Other,
}

/// Tool call status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallStatus {
    /// Pending execution.
    #[default]
    Pending,
    /// In progress.
    InProgress,
    /// Completed successfully.
    Completed,
    /// Failed with error.
    Failed,
}

/// Tool call content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolCallContent {
    /// Regular content.
    Content {
        /// The content block.
        content: ContentBlock,
    },
    /// File diff.
    Diff(Diff),
    /// Terminal reference.
    Terminal {
        /// Terminal ID from terminal/create.
        #[serde(rename = "terminalId")]
        terminal_id: String,
    },
}

/// File diff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Diff {
    /// File path.
    pub path: PathBuf,
    /// Original text (None for new files).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old_text: Option<String>,
    /// New text.
    pub new_text: String,
}

/// Tool call location.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallLocation {
    /// File path.
    pub path: PathBuf,
    /// Line number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
}

// =============================================================================
// Permissions (agent → client requests)
// =============================================================================

/// `session/request_permission` request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestPermissionParams {
    /// Session ID.
    pub session_id: String,
    /// The tool call that needs permission.
    pub tool_call: ToolCall,
    /// Options the user can choose from.
    #[serde(default)]
    pub options: Vec<PermissionOption>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A selectable option in a permission request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionOption {
    /// Unique option ID reported back in the outcome.
    pub option_id: String,
    /// Human-readable label.
    pub name: String,
    /// What the option does.
    pub kind: PermissionOptionKind,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// What a [`PermissionOption`] does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionOptionKind {
    /// Allow this call only.
    AllowOnce,
    /// Allow this call and future ones like it.
    AllowAlways,
    /// Reject this call only.
    RejectOnce,
    /// Reject this call and future ones like it.
    RejectAlways,
}

/// `session/request_permission` response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestPermissionResult {
    /// The outcome of the permission request.
    pub outcome: RequestPermissionOutcome,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Outcome of a permission request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum RequestPermissionOutcome {
    /// The prompt turn was cancelled before the user responded.
    Cancelled,
    /// The user selected one of the provided options.
    Selected {
        /// ID of the selected option.
        #[serde(rename = "optionId")]
        option_id: String,
    },
}

// =============================================================================
// File system (agent → client requests)
// =============================================================================

/// `fs/read_text_file` request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadTextFileParams {
    /// Session ID.
    pub session_id: String,
    /// File path to read.
    pub path: PathBuf,
    /// First line to read (1-based); absent reads from the start.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    /// Maximum number of lines to read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `fs/read_text_file` response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadTextFileResult {
    /// File content.
    pub content: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `fs/write_text_file` request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WriteTextFileParams {
    /// Session ID.
    pub session_id: String,
    /// File path to write.
    pub path: PathBuf,
    /// Full content to write.
    pub content: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `fs/write_text_file` response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteTextFileResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

// =============================================================================
// Terminals (agent → client requests)
// =============================================================================

/// `terminal/create` request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalCreateParams {
    /// Session ID.
    pub session_id: String,
    /// Command to execute.
    pub command: String,
    /// Command arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment variables.
    #[serde(default)]
    pub env: Vec<EnvVar>,
    /// Working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// Truncate retained output to this many bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_byte_limit: Option<u64>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/create` response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalCreateResult {
    /// Terminal ID used by the other `terminal/*` methods.
    pub terminal_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/output` request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalOutputParams {
    /// Session ID.
    pub session_id: String,
    /// Terminal ID from `terminal/create`.
    pub terminal_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/output` response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalOutputResult {
    /// Retained output.
    pub output: String,
    /// Whether the output was truncated at the byte limit.
    pub truncated: bool,
    /// Exit status once the process has exited.
    #[serde(default)]
    pub exit_status: Option<TerminalExitStatus>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/wait_for_exit` request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalWaitForExitParams {
    /// Session ID.
    pub session_id: String,
    /// Terminal ID from `terminal/create`.
    pub terminal_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Exit status of a terminal process; also the `terminal/wait_for_exit`
/// response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalExitStatus {
    /// Exit code, if the process exited normally.
    #[serde(default)]
    pub exit_code: Option<i64>,
    /// Signal that terminated the process, if any.
    #[serde(default)]
    pub signal: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/kill` request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalKillParams {
    /// Session ID.
    pub session_id: String,
    /// Terminal ID from `terminal/create`.
    pub terminal_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/kill` response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalKillResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/release` request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalReleaseParams {
    /// Session ID.
    pub session_id: String,
    /// Terminal ID from `terminal/create`.
    pub terminal_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `terminal/release` response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalReleaseResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

#[cfg(test)]
mod tests_types {
    use super::*;

    /// `audio` content blocks use the ACP `audio` tag and `mimeType` key,
    /// and round-trip back into the same variant.
    #[test]
    fn audio_content_block_serializes_with_audio_tag() {
        let block = ContentBlock::Audio(AudioContent {
            data: "AAAA".to_string(),
            mime_type: "audio/ogg".to_string(),
            annotations: None,
            meta: None,
        });
        let json = serde_json::to_value(&block).expect("serializes");
        assert_eq!(json["type"], "audio");
        assert_eq!(json["mimeType"], "audio/ogg");
        let parsed: ContentBlock = serde_json::from_value(json).expect("deserializes");
        assert!(matches!(parsed, ContentBlock::Audio(_)));
    }

    /// `session/resume` params use camelCase keys and skip empty
    /// `additionalDirectories`; the advertised capability deserializes from
    /// an empty object.
    #[test]
    fn session_resume_wire_format() {
        let params = SessionResumeParams {
            session_id: "s1".to_string(),
            cwd: PathBuf::from("/tmp/chat"),
            mcp_servers: vec![],
            additional_directories: vec![],
            meta: None,
        };
        let json = serde_json::to_value(&params).expect("serializes");
        assert_eq!(json["sessionId"], "s1");
        assert_eq!(json["cwd"], "/tmp/chat");
        assert!(json.get("additionalDirectories").is_none());

        let caps: SessionCapabilities =
            serde_json::from_str(r#"{"resume":{}}"#).expect("deserializes");
        assert!(caps.resume.is_some());
        assert!(caps.close.is_none());
    }

    /// `session/fork` params, result, and capability serialize with camelCase
    /// keys, omit empty optional fields, and preserve flattened extras.
    #[test]
    fn session_fork_wire_format() {
        let minimal = SessionForkParams::new("s1", "/tmp/fork");
        let minimal_json = serde_json::to_value(&minimal).expect("serializes");
        assert_eq!(minimal_json["sessionId"], "s1");
        assert_eq!(minimal_json["cwd"], "/tmp/fork");
        assert_eq!(minimal_json["mcpServers"], serde_json::json!([]));
        assert!(minimal_json.get("additionalDirectories").is_none());
        assert!(minimal_json.get("_meta").is_none());

        let full = SessionForkParams::new("s1", "/tmp/fork")
            .mcp_servers(vec![McpServerStdio::new("fs", "/bin/mcp").into()])
            .additional_directories(vec![PathBuf::from("/tmp/extra")])
            .meta(serde_json::json!({"trace": "t-1"}));
        let full_json = serde_json::to_value(&full).expect("serializes");
        assert_eq!(
            full_json["additionalDirectories"],
            serde_json::json!(["/tmp/extra"])
        );
        assert_eq!(full_json["_meta"]["trace"], "t-1");
        assert_eq!(full_json["mcpServers"][0]["name"], "fs");
        let parsed_params: SessionForkParams =
            serde_json::from_value(full_json).expect("deserializes");
        assert_eq!(parsed_params.session_id, "s1");
        assert_eq!(parsed_params.cwd, PathBuf::from("/tmp/fork"));
        assert_eq!(parsed_params.mcp_servers.len(), 1);
        assert_eq!(
            parsed_params.additional_directories,
            vec![PathBuf::from("/tmp/extra")]
        );

        let result: SessionForkResult = serde_json::from_value(serde_json::json!({
            "sessionId": "s2",
            "modes": {
                "currentModeId": "code",
                "availableModes": [{"id": "code", "name": "Code"}]
            },
            "configOptions": [{
                "id": "model",
                "name": "Model",
                "currentValue": "gpt-5"
            }],
            "_meta": {"source": "s1"},
            "models": {
                "availableModels": [{"modelId": "gpt-5", "name": "GPT-5"}],
                "currentModelId": "gpt-5"
            }
        }))
        .expect("deserializes");
        assert_eq!(result.session_id, "s2");
        assert_eq!(
            result
                .modes
                .as_ref()
                .map(|modes| modes.current_mode_id.as_str()),
            Some("code")
        );
        assert_eq!(result.config_options.as_ref().map(Vec::len), Some(1));
        assert_eq!(result.meta, Some(serde_json::json!({"source": "s1"})));
        assert_eq!(result.extra["models"]["currentModelId"], "gpt-5");
        let round_trip = serde_json::to_value(&result).expect("serializes");
        assert_eq!(round_trip["sessionId"], "s2");
        assert_eq!(round_trip["_meta"]["source"], "s1");
        assert_eq!(round_trip["models"]["currentModelId"], "gpt-5");

        let default_caps =
            serde_json::to_value(SessionCapabilities::default()).expect("serializes");
        assert!(default_caps.get("fork").is_none());

        let caps: SessionCapabilities =
            serde_json::from_str(r#"{"fork":{"_meta":{"v":1}}}"#).expect("deserializes");
        assert_eq!(
            caps.fork.as_ref().and_then(|fork| fork.meta.as_ref()),
            Some(&serde_json::json!({"v": 1}))
        );
        assert!(caps.extra.is_empty());
    }
}

// ===== old protocol/session.rs =====

/// `session/list` request parameters. All filters are optional.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionListParams {
    /// Only list sessions with this working directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// Cursor from a previous `session/list` response's `nextCursor`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionListParams {
    /// Build an unfiltered `session/list` request.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter to sessions with this working directory.
    #[must_use]
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Continue a previous paginated listing.
    #[must_use]
    pub fn cursor(mut self, cursor: impl Into<String>) -> Self {
        self.cursor = Some(cursor.into());
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// `session/list` response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionListResult {
    /// Sessions the agent knows about.
    #[serde(default)]
    pub sessions: Vec<SessionInfo>,
    /// Cursor to pass as [`SessionListParams::cursor`] for the next page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A session returned by `session/list`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfo {
    /// Session ID usable with `session/load`, `session/resume`, or
    /// `session/fork`.
    pub session_id: String,
    /// Working directory the session was created with.
    pub cwd: PathBuf,
    /// Extra workspace roots the session was created or loaded with. These
    /// must be re-sent when loading or resuming — the agent does not
    /// restore them implicitly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_directories: Vec<PathBuf>,
    /// Human-readable title, if the agent tracks one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// ISO 8601 timestamp of last activity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

/// `session/delete` request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionDeleteParams {
    /// Session to delete from the agent's history.
    pub session_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionDeleteParams {
    /// Build params for `session_id`.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            meta: None,
        }
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

impl From<&str> for SessionDeleteParams {
    fn from(session_id: &str) -> Self {
        Self::new(session_id)
    }
}

impl From<String> for SessionDeleteParams {
    fn from(session_id: String) -> Self {
        Self::new(session_id)
    }
}

/// `session/delete` response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionDeleteResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `session/close` request parameters: ends a live session without deleting
/// its history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCloseParams {
    /// Session to close.
    pub session_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl SessionCloseParams {
    /// Build params for `session_id`.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            meta: None,
        }
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

impl From<&str> for SessionCloseParams {
    fn from(session_id: &str) -> Self {
        Self::new(session_id)
    }
}

impl From<String> for SessionCloseParams {
    fn from(session_id: String) -> Self {
        Self::new(session_id)
    }
}

/// `session/close` response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCloseResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

#[cfg(test)]
mod tests_session {
    use super::*;
    use serde_json::json;

    #[test]
    fn list_params_and_result_round_trip() {
        let params = SessionListParams::new().cwd("/repo").cursor("page-2");
        let json = serde_json::to_value(&params).expect("serializes");
        assert_eq!(json["cwd"], "/repo");
        assert_eq!(json["cursor"], "page-2");

        let result: SessionListResult = serde_json::from_value(json!({
            "sessions": [{
                "sessionId": "s1",
                "cwd": "/repo",
                "additionalDirectories": ["/other"],
                "title": "T",
                "updatedAt": "2025-01-01T00:00:00Z",
                "vendor_field": 1,
            }],
            "nextCursor": "page-3",
        }))
        .expect("deserializes");
        assert_eq!(result.sessions[0].session_id, "s1");
        assert_eq!(result.sessions[0].title.as_deref(), Some("T"));
        assert_eq!(result.next_cursor.as_deref(), Some("page-3"));
        // Unknown fields on SessionInfo are preserved verbatim.
        assert_eq!(result.sessions[0].extra["vendor_field"], 1);
        let back = serde_json::to_value(&result.sessions[0]).expect("serializes");
        assert_eq!(back["additionalDirectories"], json!(["/other"]));
        assert_eq!(back["vendor_field"], 1);
    }

    #[test]
    fn delete_and_close_carry_session_id_and_meta() {
        for json in [
            serde_json::to_value(SessionDeleteParams::new("s1").meta(json!({"k": 1})))
                .expect("serializes"),
            serde_json::to_value(SessionCloseParams::new("s1").meta(json!({"k": 1})))
                .expect("serializes"),
        ] {
            assert_eq!(json["sessionId"], "s1");
            assert_eq!(json["_meta"]["k"], 1);
        }
    }
}

// ===== old protocol/auth.rs =====

/// `authenticate` request parameters: runs one of the [`AuthMethod`]s the
/// agent advertised in [`InitializeResult::auth_methods`](InitializeResult::auth_methods).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthenticateParams {
    /// ID of the auth method to run.
    pub method_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl AuthenticateParams {
    /// Build params selecting `method_id`.
    #[must_use]
    pub fn new(method_id: impl Into<String>) -> Self {
        Self {
            method_id: method_id.into(),
            meta: None,
        }
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

impl From<&str> for AuthenticateParams {
    fn from(method_id: &str) -> Self {
        Self::new(method_id)
    }
}

impl From<String> for AuthenticateParams {
    fn from(method_id: String) -> Self {
        Self::new(method_id)
    }
}

/// `authenticate` response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthenticateResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `logout` request parameters. The method takes no fields besides `_meta`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogoutParams {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl LogoutParams {
    /// Build empty logout params.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// `logout` response.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogoutResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// An authentication method the agent advertises in `initialize`.
///
/// `agent` methods — the default kind, sent with no `type` field — are run
/// inside the agent (e.g. a browser flow). `terminal` methods describe a
/// command the client runs in a terminal it owns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AuthMethod {
    /// A method the client runs in its terminal; `"type": "terminal"`.
    #[serde(rename = "terminal")]
    Terminal(AuthMethodTerminal),
    /// A method the agent runs itself; sent with no `type` field.
    #[serde(untagged)]
    Agent(AuthMethodAgent),
}

impl AuthMethod {
    /// The method's unique ID.
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Agent(agent) => &agent.id,
            Self::Terminal(terminal) => &terminal.id,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Agent(agent) => &agent.name,
            Self::Terminal(terminal) => &terminal.name,
        }
    }

    /// Human-readable description, if provided.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        match self {
            Self::Agent(agent) => agent.description.as_deref(),
            Self::Terminal(terminal) => terminal.description.as_deref(),
        }
    }

    /// Extension metadata.
    #[must_use]
    pub const fn meta(&self) -> Option<&Value> {
        match self {
            Self::Agent(agent) => agent.meta.as_ref(),
            Self::Terminal(terminal) => terminal.meta.as_ref(),
        }
    }
}

impl From<AuthMethodAgent> for AuthMethod {
    fn from(method: AuthMethodAgent) -> Self {
        Self::Agent(method)
    }
}

impl From<AuthMethodTerminal> for AuthMethod {
    fn from(method: AuthMethodTerminal) -> Self {
        Self::Terminal(method)
    }
}

/// An auth method the agent runs itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthMethodAgent {
    /// Unique ID passed to `authenticate`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl AuthMethodAgent {
    /// Build an agent-run auth method.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            meta: None,
        }
    }

    /// Set the description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// An auth method the client runs in a terminal it owns: the agent names a
/// program to spawn and the client executes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthMethodTerminal {
    /// Unique ID passed to `authenticate`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Program and arguments to run in the terminal.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// Environment for the spawned program.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub env: HashMap<String, String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// What the client offers for authentication (`clientCapabilities.auth`).
///
/// `terminal` tells the agent it may advertise
/// [`AuthMethod::Terminal`] methods; `_meta` carries vendor flags such as
/// gateway-auth opt-ins.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientAuthCapabilities {
    /// Whether the client can run terminal auth methods.
    #[serde(default)]
    pub terminal: bool,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// What the agent offers for authentication (`agentCapabilities.auth`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentAuthCapabilities {
    /// Whether the agent supports `logout`. `null` and absent both mean no.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logout: Option<LogoutCapabilities>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Marker for `logout` support (`agentCapabilities.auth.logout`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogoutCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

#[cfg(test)]
mod tests_auth {
    use super::*;
    use serde_json::json;

    #[test]
    fn authenticate_params_wire_format() {
        let json =
            serde_json::to_value(AuthenticateParams::new("devin-browser").meta(json!({"k": 1})))
                .expect("serializes");
        assert_eq!(json["methodId"], "devin-browser");
        assert_eq!(json["_meta"]["k"], 1);
    }

    #[test]
    fn agent_auth_method_has_no_type_tag() {
        let method = AuthMethod::from(AuthMethodAgent::new("oauth", "Sign in"));
        let json = serde_json::to_value(&method).expect("serializes");
        assert!(json.get("type").is_none());
        assert_eq!(json["id"], "oauth");

        let parsed: AuthMethod = serde_json::from_value(json).expect("deserializes");
        assert_eq!(parsed.id(), "oauth");
    }

    #[test]
    fn terminal_auth_method_carries_type_tag() {
        let method: AuthMethod = serde_json::from_value(json!({
            "type": "terminal",
            "id": "gh",
            "name": "GitHub CLI",
            "args": ["gh", "auth", "login"],
            "env": {"BROWSER": "none"},
        }))
        .expect("deserializes");
        let AuthMethod::Terminal(terminal) = &method else {
            panic!("expected terminal method");
        };
        assert_eq!(terminal.id, "gh");
        assert_eq!(terminal.args, ["gh", "auth", "login"]);
        assert_eq!(terminal.env["BROWSER"], "none");
    }

    #[test]
    fn agent_auth_capabilities_round_trip() {
        let caps: AgentAuthCapabilities =
            serde_json::from_value(json!({"logout": {}})).expect("deserializes");
        assert!(caps.logout.is_some());
        let caps: AgentAuthCapabilities = serde_json::from_value(json!({})).expect("deserializes");
        assert!(caps.logout.is_none());
    }
}

// ===== old protocol/elicitation.rs =====

/// `elicitation/create` request parameters.
///
/// `scope` (a `sessionId`/`toolCallId` pair, or a `requestId`) is flattened
/// into [`ElicitationMode`]'s variants, matching the wire layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationCreateParams {
    /// How the client should collect the input: a form or a URL visit.
    #[serde(flatten)]
    pub mode: ElicitationMode,
    /// Human-readable explanation of what is being asked and why.
    pub message: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl ElicitationCreateParams {
    /// Build a request with the given mode and message.
    #[must_use]
    pub fn new(mode: impl Into<ElicitationMode>, message: impl Into<String>) -> Self {
        Self {
            mode: mode.into(),
            message: message.into(),
            meta: None,
        }
    }

    /// The scope this elicitation belongs to.
    #[must_use]
    pub const fn scope(&self) -> &ElicitationScope {
        self.mode.scope()
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// The interaction the client should run for an elicitation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ElicitationMode {
    /// Render `requestedSchema` as a form and collect the values.
    Form(ElicitationFormMode),
    /// Open `url` and wait for the matching `elicitation/complete`.
    Url(ElicitationUrlMode),
    /// A mode this crate does not model; the raw fields are preserved.
    #[serde(untagged)]
    Other(OtherElicitationMode),
}

impl ElicitationMode {
    /// The scope this elicitation belongs to.
    #[must_use]
    pub const fn scope(&self) -> &ElicitationScope {
        match self {
            Self::Form(form) => &form.scope,
            Self::Url(url) => &url.scope,
            Self::Other(other) => &other.scope,
        }
    }
}

impl From<ElicitationFormMode> for ElicitationMode {
    fn from(mode: ElicitationFormMode) -> Self {
        Self::Form(mode)
    }
}

impl From<ElicitationUrlMode> for ElicitationMode {
    fn from(mode: ElicitationUrlMode) -> Self {
        Self::Url(mode)
    }
}

/// `form` mode: the client renders [`ElicitationSchema`] and returns the
/// collected values in [`ElicitationAcceptAction::content`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationFormMode {
    /// What the elicitation belongs to.
    #[serde(flatten)]
    pub scope: ElicitationScope,
    /// JSON Schema describing the values to collect.
    pub requested_schema: ElicitationSchema,
}

impl ElicitationFormMode {
    /// Build a form-mode request for `scope`.
    #[must_use]
    pub fn new(scope: impl Into<ElicitationScope>, requested_schema: ElicitationSchema) -> Self {
        Self {
            scope: scope.into(),
            requested_schema,
        }
    }
}

/// `url` mode: the client opens `url` and waits for `elicitation/complete`
/// with the matching `elicitation_id`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationUrlMode {
    /// What the elicitation belongs to.
    #[serde(flatten)]
    pub scope: ElicitationScope,
    /// ID the agent later echoes in `elicitation/complete`.
    pub elicitation_id: String,
    /// URL to open.
    pub url: String,
}

impl ElicitationUrlMode {
    /// Build a url-mode request for `scope`.
    #[must_use]
    pub fn new(
        scope: impl Into<ElicitationScope>,
        elicitation_id: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        Self {
            scope: scope.into(),
            elicitation_id: elicitation_id.into(),
            url: url.into(),
        }
    }
}

/// An elicitation mode this crate does not model. The raw fields are
/// preserved so a client can still forward them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OtherElicitationMode {
    /// The unrecognized `mode` tag.
    pub mode: String,
    /// The scope, if it uses one of the known scope shapes.
    #[serde(flatten)]
    pub scope: ElicitationScope,
    /// Every remaining field of the mode object.
    #[serde(flatten)]
    pub fields: BTreeMap<String, Value>,
}

const KNOWN_ELICITATION_MODES: &[&str] = &["form", "url"];

impl<'de> Deserialize<'de> for OtherElicitationMode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut fields = BTreeMap::<String, Value>::deserialize(deserializer)?;
        let mode = fields
            .remove("mode")
            .ok_or_else(|| D::Error::missing_field("mode"))?;
        let Value::String(mode) = mode else {
            return Err(D::Error::custom("`mode` must be a string"));
        };
        if KNOWN_ELICITATION_MODES.contains(&mode.as_str()) {
            return Err(D::Error::custom(format!(
                "known elicitation mode `{mode}` did not match its schema"
            )));
        }
        let scope = serde_json::from_value(Value::Object(fields.clone().into_iter().collect()))
            .map_err(D::Error::custom)?;
        // Scope fields belong to the scope, not to the mode's own extras.
        fields.remove("sessionId");
        fields.remove("toolCallId");
        fields.remove("requestId");
        Ok(Self {
            mode,
            scope,
            fields,
        })
    }
}

/// What an elicitation belongs to: a session (optionally narrowed to one
/// tool call) or another in-flight request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ElicitationScope {
    /// Scoped to a session, optionally to one tool call inside it.
    Session(ElicitationSessionScope),
    /// Scoped to a specific request by its JSON-RPC ID.
    Request(ElicitationRequestScope),
}

impl From<ElicitationSessionScope> for ElicitationScope {
    fn from(scope: ElicitationSessionScope) -> Self {
        Self::Session(scope)
    }
}

impl From<ElicitationRequestScope> for ElicitationScope {
    fn from(scope: ElicitationRequestScope) -> Self {
        Self::Request(scope)
    }
}

/// Session scope for an elicitation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationSessionScope {
    /// Session the elicitation belongs to.
    pub session_id: String,
    /// Tool call the elicitation belongs to, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ElicitationSessionScope {
    /// Build a session scope.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            tool_call_id: None,
        }
    }

    /// Narrow the scope to one tool call.
    #[must_use]
    pub fn tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }
}

/// Request scope for an elicitation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationRequestScope {
    /// ID of the request this elicitation refines.
    pub request_id: RequestId,
}

impl ElicitationRequestScope {
    /// Build a request scope.
    #[must_use]
    pub fn new(request_id: impl Into<RequestId>) -> Self {
        Self {
            request_id: request_id.into(),
        }
    }
}

/// `elicitation/create` response: the user's answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElicitationCreateResult {
    /// What the user did.
    #[serde(flatten)]
    pub action: ElicitationAction,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl ElicitationCreateResult {
    /// Build a response for `action`.
    #[must_use]
    pub fn new(action: impl Into<ElicitationAction>) -> Self {
        Self {
            action: action.into(),
            meta: None,
        }
    }

    /// `accept` response carrying the collected form values.
    #[must_use]
    pub fn accept(content: BTreeMap<String, ElicitationContentValue>) -> Self {
        Self::new(ElicitationAcceptAction {
            content: Some(content),
        })
    }

    /// `decline` response: the user refused to provide input.
    #[must_use]
    pub fn decline() -> Self {
        Self::new(ElicitationAction::Decline)
    }

    /// `cancel` response: the user dismissed the interaction.
    #[must_use]
    pub fn cancel() -> Self {
        Self::new(ElicitationAction::Cancel)
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// How the user answered an elicitation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ElicitationAction {
    /// The user provided input; `content` carries the collected values for
    /// form mode (absent for url mode, where the agent already has them).
    Accept(ElicitationAcceptAction),
    /// The user refused to provide input.
    Decline,
    /// The user dismissed the interaction.
    Cancel,
    /// An action this crate does not model; the raw fields are preserved.
    #[serde(untagged)]
    Other(OtherElicitationAction),
}

impl From<ElicitationAcceptAction> for ElicitationAction {
    fn from(action: ElicitationAcceptAction) -> Self {
        Self::Accept(action)
    }
}

/// The `accept` action payload.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ElicitationAcceptAction {
    /// Values collected for a `form` elicitation, keyed by property name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<BTreeMap<String, ElicitationContentValue>>,
}

/// An elicitation action this crate does not model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OtherElicitationAction {
    /// The unrecognized `action` tag.
    pub action: String,
    /// Every remaining field of the action object.
    #[serde(flatten)]
    pub fields: BTreeMap<String, Value>,
}

const KNOWN_ELICITATION_ACTIONS: &[&str] = &["accept", "decline", "cancel"];

impl<'de> Deserialize<'de> for OtherElicitationAction {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut fields = BTreeMap::<String, Value>::deserialize(deserializer)?;
        let action = fields
            .remove("action")
            .ok_or_else(|| D::Error::missing_field("action"))?;
        let Value::String(action) = action else {
            return Err(D::Error::custom("`action` must be a string"));
        };
        if KNOWN_ELICITATION_ACTIONS.contains(&action.as_str()) {
            return Err(D::Error::custom(format!(
                "known elicitation action `{action}` did not match its schema"
            )));
        }
        Ok(Self { action, fields })
    }
}

/// One collected form value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ElicitationContentValue {
    /// A text value.
    String(String),
    /// An integer value.
    Integer(i64),
    /// A non-integer numeric value.
    Number(f64),
    /// A flag value.
    Boolean(bool),
    /// Selected entries of a multi-select.
    StringArray(Vec<String>),
}

impl From<String> for ElicitationContentValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for ElicitationContentValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<i64> for ElicitationContentValue {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for ElicitationContentValue {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<bool> for ElicitationContentValue {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl From<Vec<String>> for ElicitationContentValue {
    fn from(value: Vec<String>) -> Self {
        Self::StringArray(value)
    }
}

/// `elicitation/complete` notification parameters: the agent tells the
/// client a URL-mode elicitation is finished.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationCompleteParams {
    /// ID from the original `url`-mode `elicitation/create`.
    pub elicitation_id: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// The JSON Schema for a `form` elicitation. Only `"type": "object"` is
/// defined.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationSchema {
    /// Schema kind; always `"object"`.
    #[serde(rename = "type", default)]
    pub kind: ElicitationSchemaType,
    /// Title to render above the form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Properties to collect, keyed by name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub properties: BTreeMap<String, ElicitationPropertySchema>,
    /// Names of required properties.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    /// Human-readable description of the form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl ElicitationSchema {
    /// Build an empty object schema.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a property; `required` marks it mandatory.
    #[must_use]
    pub fn property(
        mut self,
        name: impl Into<String>,
        schema: impl Into<ElicitationPropertySchema>,
        required: bool,
    ) -> Self {
        let name = name.into();
        if required {
            self.required
                .get_or_insert_with(Vec::new)
                .push(name.clone());
        }
        self.properties.insert(name, schema.into());
        self
    }

    /// Set the title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set extension metadata.
    #[must_use]
    pub fn meta(mut self, meta: Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// The only schema kind ACP elicitation defines.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElicitationSchemaType {
    /// An object with named properties.
    #[default]
    Object,
}

/// Schema for one form property.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ElicitationPropertySchema {
    /// A text field.
    String(StringPropertySchema),
    /// A non-integer numeric field.
    Number(NumberPropertySchema),
    /// An integer numeric field.
    Integer(IntegerPropertySchema),
    /// A checkbox.
    Boolean(BooleanPropertySchema),
    /// A multi-select.
    Array(MultiSelectPropertySchema),
    /// A property kind this crate does not model; the raw fields are
    /// preserved.
    #[serde(untagged)]
    Other(OtherElicitationPropertySchema),
}

impl From<StringPropertySchema> for ElicitationPropertySchema {
    fn from(schema: StringPropertySchema) -> Self {
        Self::String(schema)
    }
}

impl From<NumberPropertySchema> for ElicitationPropertySchema {
    fn from(schema: NumberPropertySchema) -> Self {
        Self::Number(schema)
    }
}

impl From<IntegerPropertySchema> for ElicitationPropertySchema {
    fn from(schema: IntegerPropertySchema) -> Self {
        Self::Integer(schema)
    }
}

impl From<BooleanPropertySchema> for ElicitationPropertySchema {
    fn from(schema: BooleanPropertySchema) -> Self {
        Self::Boolean(schema)
    }
}

impl From<MultiSelectPropertySchema> for ElicitationPropertySchema {
    fn from(schema: MultiSelectPropertySchema) -> Self {
        Self::Array(schema)
    }
}

/// Schema for a text property.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StringPropertySchema {
    /// Human-readable label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Minimum text length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_length: Option<u32>,
    /// Maximum text length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<u32>,
    /// Regex the value must match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    /// Well-known text format (email, URI, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<ElicitationStringFormat>,
    /// Value used when the user does not fill the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    /// Fixed set of allowed values.
    #[serde(rename = "enum", default, skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    /// Allowed values with titles and descriptions.
    #[serde(rename = "oneOf", default, skip_serializing_if = "Option::is_none")]
    pub one_of: Option<Vec<ElicitationEnumOption>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// One labelled choice in a [`StringPropertySchema::one_of`] list or a
/// [`MultiSelectItems::Titled`] item set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElicitationEnumOption {
    /// The value reported when this option is selected.
    #[serde(rename = "const")]
    pub value: String,
    /// Human-readable label.
    pub title: String,
    /// Human-readable help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Well-known text formats for a string property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ElicitationStringFormat {
    /// An email address.
    Email,
    /// A URI.
    Uri,
    /// A date (`YYYY-MM-DD`).
    Date,
    /// A date-time (`YYYY-MM-DDThh:mm:ss`).
    DateTime,
}

/// Schema for a non-integer numeric property.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NumberPropertySchema {
    /// Human-readable label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Minimum value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    /// Maximum value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
    /// Default value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<f64>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Schema for an integer property.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IntegerPropertySchema {
    /// Human-readable label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Minimum value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum: Option<i64>,
    /// Maximum value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum: Option<i64>,
    /// Default value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<i64>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Schema for a boolean property.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BooleanPropertySchema {
    /// Human-readable label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Default state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<bool>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Schema for a multi-select property.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MultiSelectPropertySchema {
    /// Human-readable label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable help text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Minimum selections.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_items: Option<u64>,
    /// Maximum selections.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_items: Option<u64>,
    /// The selectable items.
    pub items: MultiSelectItems,
    /// Pre-selected values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Vec<String>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// The item set of a multi-select property: a bare value list
/// (`"type": "string", "enum": [...]`) or a titled list (`"anyOf": [...]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MultiSelectItems {
    /// A bare list of selectable values.
    String(StringMultiSelectItems),
    /// Selectable values with titles.
    #[serde(untagged)]
    Titled(TitledMultiSelectItems),
    /// An item set this crate does not model; the raw fields are preserved.
    #[serde(untagged)]
    Other(OtherMultiSelectItems),
}

/// A bare `{"type": "string", "enum": [...]}` item set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StringMultiSelectItems {
    /// Selectable values.
    #[serde(rename = "enum")]
    pub values: Vec<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A titled `{"anyOf": [...]}` item set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TitledMultiSelectItems {
    /// Selectable options.
    #[serde(rename = "anyOf")]
    pub options: Vec<ElicitationEnumOption>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// An item-set shape this crate does not model.
///
/// The `fields` bag holds raw JSON, so equality is not derivable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OtherMultiSelectItems {
    /// The unrecognized `type` tag.
    #[serde(rename = "type")]
    pub kind: String,
    /// Every remaining field of the item set.
    #[serde(flatten)]
    pub fields: BTreeMap<String, Value>,
}

const KNOWN_MULTI_SELECT_ITEM_TYPES: &[&str] = &["string"];

impl<'de> Deserialize<'de> for OtherMultiSelectItems {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut fields = BTreeMap::<String, Value>::deserialize(deserializer)?;
        let kind = fields
            .remove("type")
            .ok_or_else(|| D::Error::missing_field("type"))?;
        let Value::String(kind) = kind else {
            return Err(D::Error::custom("`type` must be a string"));
        };
        if KNOWN_MULTI_SELECT_ITEM_TYPES.contains(&kind.as_str()) {
            return Err(D::Error::custom(format!(
                "known multi-select item type `{kind}` did not match its schema"
            )));
        }
        Ok(Self { kind, fields })
    }
}

/// A property kind this crate does not model.
///
/// The `fields` bag holds raw JSON, so equality is not derivable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OtherElicitationPropertySchema {
    /// The unrecognized `type` tag.
    #[serde(rename = "type")]
    pub kind: String,
    /// Every remaining field of the schema.
    #[serde(flatten)]
    pub fields: BTreeMap<String, Value>,
}

const KNOWN_PROPERTY_TYPES: &[&str] = &["string", "number", "integer", "boolean", "array"];

impl<'de> Deserialize<'de> for OtherElicitationPropertySchema {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut fields = BTreeMap::<String, Value>::deserialize(deserializer)?;
        let kind = fields
            .remove("type")
            .ok_or_else(|| D::Error::missing_field("type"))?;
        let Value::String(kind) = kind else {
            return Err(D::Error::custom("`type` must be a string"));
        };
        if KNOWN_PROPERTY_TYPES.contains(&kind.as_str()) {
            return Err(D::Error::custom(format!(
                "known elicitation property type `{kind}` did not match its schema"
            )));
        }
        Ok(Self { kind, fields })
    }
}

/// What the client offers for elicitation (`clientCapabilities.elicitation`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElicitationCapabilities {
    /// Whether the client can render forms. `{}` advertises support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub form: Option<ElicitationFormCapabilities>,
    /// Whether the client can open URLs for url-mode elicitation. `{}`
    /// advertises support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<ElicitationUrlCapabilities>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Marker for form elicitation support.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElicitationFormCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Marker for URL elicitation support.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElicitationUrlCapabilities {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

#[cfg(test)]
mod tests_elicitation {
    use super::*;
    use serde_json::json;

    /// A form-mode create request flattens mode and scope into the params.
    #[test]
    fn form_create_params_wire_format() {
        let params = ElicitationCreateParams::new(
            ElicitationFormMode::new(
                ElicitationSessionScope::new("s1").tool_call_id("tc-9"),
                ElicitationSchema::new().property(
                    "name",
                    StringPropertySchema {
                        title: Some("Name".to_string()),
                        ..StringPropertySchema::default()
                    },
                    true,
                ),
            ),
            "Who are you?",
        );
        let json = serde_json::to_value(&params).expect("serializes");
        assert_eq!(json["mode"], "form");
        assert_eq!(json["sessionId"], "s1");
        assert_eq!(json["toolCallId"], "tc-9");
        assert_eq!(json["requestedSchema"]["type"], "object");
        assert_eq!(json["requestedSchema"]["required"], json!(["name"]));

        let parsed: ElicitationCreateParams = serde_json::from_value(json).expect("deserializes");
        let ElicitationMode::Form(form) = &parsed.mode else {
            panic!("expected form mode");
        };
        let ElicitationScope::Session(scope) = &form.scope else {
            panic!("expected session scope");
        };
        assert_eq!(scope.session_id, "s1");
        assert_eq!(scope.tool_call_id.as_deref(), Some("tc-9"));
    }

    /// A url-mode create request carries the elicitation ID and URL.
    #[test]
    fn url_create_params_wire_format() {
        let params = ElicitationCreateParams::new(
            ElicitationUrlMode::new(
                ElicitationRequestScope::new(RequestId::from(7_i64)),
                "elic-1",
                "http://localhost:8787/auth",
            ),
            "Sign in",
        );
        let json = serde_json::to_value(&params).expect("serializes");
        assert_eq!(json["mode"], "url");
        assert_eq!(json["requestId"], 7);
        assert_eq!(json["elicitationId"], "elic-1");

        let parsed: ElicitationCreateParams = serde_json::from_value(json).expect("deserializes");
        let ElicitationMode::Url(url) = &parsed.mode else {
            panic!("expected url mode");
        };
        assert_eq!(url.elicitation_id, "elic-1");
        let ElicitationScope::Request(scope) = &url.scope else {
            panic!("expected request scope");
        };
        assert_eq!(scope.request_id, RequestId::from(7_i64));
    }

    /// An unknown mode lands in `Other` with every field preserved.
    #[test]
    fn unknown_mode_preserved() {
        let params: ElicitationCreateParams = serde_json::from_value(json!({
            "mode": "voice",
            "sessionId": "s1",
            "message": "Say it",
            "pitch": "high",
        }))
        .expect("deserializes");
        let ElicitationMode::Other(other) = &params.mode else {
            panic!("expected Other mode");
        };
        assert_eq!(other.mode, "voice");
        let ElicitationScope::Session(scope) = &other.scope else {
            panic!("expected session scope");
        };
        assert_eq!(scope.session_id, "s1");
        assert_eq!(other.fields["pitch"], "high");
        let back = serde_json::to_value(&params).expect("serializes");
        assert_eq!(back["mode"], "voice");
        assert_eq!(back["pitch"], "high");
    }

    /// Results flatten the action; unknown actions land in `Other`.
    #[test]
    fn result_actions_wire_format() {
        let result =
            ElicitationCreateResult::accept(BTreeMap::from([("age".to_string(), 42_i64.into())]));
        let json = serde_json::to_value(&result).expect("serializes");
        assert_eq!(json["action"], "accept");
        assert_eq!(json["content"]["age"], 42);

        let parsed: ElicitationCreateResult =
            serde_json::from_value(json!({"action": "decline"})).expect("deserializes");
        assert!(matches!(parsed.action, ElicitationAction::Decline));

        let parsed: ElicitationCreateResult =
            serde_json::from_value(json!({"action": "defer", "eta": 5})).expect("deserializes");
        let ElicitationAction::Other(other) = &parsed.action else {
            panic!("expected Other action");
        };
        assert_eq!(other.action, "defer");
        assert_eq!(other.fields["eta"], 5);
    }

    /// Property schemas tag by `type`; unknown kinds land in `Other`.
    #[test]
    fn property_schema_variants() {
        let schema: ElicitationPropertySchema = serde_json::from_value(json!({
            "type": "array",
            "items": {"type": "string", "enum": ["a", "b"]},
            "minItems": 1,
        }))
        .expect("deserializes");
        let ElicitationPropertySchema::Array(multi) = &schema else {
            panic!("expected array schema");
        };
        let MultiSelectItems::String(items) = &multi.items else {
            panic!("expected string items");
        };
        assert_eq!(items.values, ["a", "b"]);

        let schema: ElicitationPropertySchema =
            serde_json::from_value(json!({"type": "color", "swatches": 3})).expect("deserializes");
        let ElicitationPropertySchema::Other(other) = &schema else {
            panic!("expected Other schema");
        };
        assert_eq!(other.kind, "color");
        assert_eq!(other.fields["swatches"], 3);
    }

    /// `elicitation/complete` carries just the elicitation ID.
    #[test]
    fn complete_params_wire_format() {
        let params: ElicitationCompleteParams =
            serde_json::from_value(json!({"elicitationId": "e1"})).expect("deserializes");
        assert_eq!(params.elicitation_id, "e1");
        assert_eq!(
            serde_json::to_value(&params).expect("serializes"),
            json!({"elicitationId": "e1"})
        );
    }
}
