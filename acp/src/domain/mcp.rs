//! MCP server descriptors carried in session setup params.
//!
//! [`McpServer`] is an open union tagged `type`: `stdio` and `http`. The
//! v1-only `sse` transport deliberately has no domain variant — the v1
//! codec maps it through the [`McpServer::Unknown`] arm instead.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::open::{ExtIdent, open_union_serde};

/// An MCP server a client may provide to the session (tagged `type`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpServer {
    /// `{"type":"stdio","name","command","args","env"}`.
    Stdio(McpServerStdio),
    /// `{"type":"http","name","url","headers"}`.
    Http(McpServerHttp),
    /// `_`-prefixed extension transport, preserved verbatim.
    Custom(ExtIdent, Value),
    /// Unrecognized transport (including v1 `sse`), preserved verbatim.
    Unknown(String, Value),
}

open_union_serde!(McpServer, "type", [
    Stdio => "stdio",
    Http => "http",
]);

/// A stdio MCP server (`{"type":"stdio","name","command","args","env"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerStdio {
    /// Server name.
    pub name: String,
    /// Executable path or name.
    pub command: String,
    /// Process arguments. Always emitted for v1 compatibility.
    pub args: Vec<String>,
    /// Environment variables. Always emitted for v1 compatibility.
    pub env: Vec<EnvVar>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// An HTTP MCP server (`{"type":"http","name","url","headers"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerHttp {
    /// Server name.
    pub name: String,
    /// Base URL.
    pub url: String,
    /// Request headers.
    #[serde(default)]
    pub headers: Vec<HttpHeader>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// An environment variable (`{"name","value"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvVar {
    /// Variable name.
    pub name: String,
    /// Variable value.
    pub value: String,
}

/// An HTTP header (`{"name","value"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HttpHeader {
    /// Header name.
    pub name: String,
    /// Header value.
    pub value: String,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn mcp_server_variants() {
        let raw = json!({"type": "stdio", "name": "fs", "command": "npx", "args": ["-y", "mcp-fs"], "env": []});
        let s: McpServer = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(s, McpServer::Stdio(_)));
        assert_eq!(serde_json::to_value(&s).expect("ser"), raw);

        let raw = json!({"type": "http", "name": "web", "url": "http://x", "headers": []});
        let s: McpServer = serde_json::from_value(raw).expect("de");
        assert!(matches!(s, McpServer::Http(_)));

        // v1 legacy "sse" lands in Unknown and round-trips.
        let raw = json!({"type": "sse", "name": "old", "url": "http://x"});
        let s: McpServer = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(s, McpServer::Unknown(ref t, _) if t == "sse"));
        assert_eq!(serde_json::to_value(&s).expect("ser"), raw);
    }
}
