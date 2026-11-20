//! `initialize` capabilities and peer info.
//!
//! v2 capability support markers are `Option<EmptyObject>`: omitted means
//! unsupported, `{}` means supported.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::patch::EmptyObject;

/// `{"name","version","title"?}` — who the peer is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Implementation {
    /// Program name.
    pub name: String,
    /// Human-readable display title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Program version.
    pub version: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Implementation {
    /// `{"name","version"}` with no title.
    #[must_use]
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            title: None,
            version: version.into(),
            meta: None,
            extra: BTreeMap::new(),
        }
    }
}

/// Capabilities the client advertises at `initialize`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    /// Authentication-related abilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<ClientAuthCaps>,
    /// Elicitation abilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elicitation: Option<ElicitCaps>,
    /// Client-side session abilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<ClientSessionCaps>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Client auth abilities (`{"terminal": {}}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientAuthCaps {
    /// Can run terminal auth methods.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal: Option<EmptyObject>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Elicitation abilities (`{"form": {}?, "url": {}?}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitCaps {
    /// Can render form elicitations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub form: Option<EmptyObject>,
    /// Can open URL elicitations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<EmptyObject>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Client-side session abilities (`{"promptSuggestions": {}}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientSessionCaps {
    /// Client can offer prompt suggestions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_suggestions: Option<EmptyObject>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Capabilities the agent advertises at `initialize`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    /// Session-related abilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<SessionCaps>,
    /// Authentication abilities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<Value>,
    /// Configured model/provider support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub providers: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Agent session abilities (`{"prompt","mcp","delete",...}`).
///
/// Every member is an `Option<EmptyObject>` or a nested capability object;
/// absence means unsupported.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCaps {
    /// Accepted prompt content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PromptCaps>,
    /// Supported MCP transports.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp: Option<McpCaps>,
    /// Supports `session/delete`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delete: Option<EmptyObject>,
    /// Supports `additionalDirectories`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_directories: Option<EmptyObject>,
    /// Supports system prompts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<SystemPromptCaps>,
    /// Supports `session/fork`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fork: Option<EmptyObject>,
    /// Supports session archival.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archive: Option<EmptyObject>,
    /// Supports configuration options.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<EmptyObject>,
    /// Supports status updates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<EmptyObject>,
    /// Supports session titles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<EmptyObject>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Prompt content the agent accepts (`{"image"?:{},"audio"?:{},"embeddedContext"?}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptCaps {
    /// Accepts image content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<EmptyObject>,
    /// Accepts audio content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<EmptyObject>,
    /// Accepts embedded resource content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedded_context: Option<EmptyObject>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// MCP transports the agent accepts (`{"stdio"?:{},"http"?:{},"acp"?:{}}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpCaps {
    /// Accepts stdio MCP servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stdio: Option<EmptyObject>,
    /// Accepts HTTP MCP servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http: Option<EmptyObject>,
    /// Accepts ACP-as-MCP child sessions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acp: Option<EmptyObject>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// System-prompt abilities (`{"mode"?:{}}`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemPromptCaps {
    /// Supports selecting a system-prompt mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<EmptyObject>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The `initialize` handshake payload (agent side), used by the wire
/// codecs for version detection.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InitResult {
    /// Negotiated protocol version.
    pub protocol_version: u16,
    /// Agent capabilities.
    pub capabilities: AgentCapabilities,
    /// Agent implementation info.
    pub info: Option<Implementation>,
    /// Advertised auth methods.
    pub auth_methods: Vec<super::auth::AuthMethod>,
    /// Extension metadata.
    pub meta: Option<Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn caps_markers_serialize_as_empty_objects() {
        let caps = SessionCaps {
            delete: Some(EmptyObject),
            ..SessionCaps::default()
        };
        let v = serde_json::to_value(&caps).expect("ser");
        assert_eq!(v, json!({"delete": {}}));
        let back: SessionCaps = serde_json::from_value(v).expect("de");
        assert!(back.delete.is_some());
        assert!(back.fork.is_none());
    }
}
