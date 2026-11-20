//! Authentication methods and login params.
//!
//! [`AuthMethod`] describes one way a client can authenticate
//! (`{"methodId","name","description"?,` kind fields`}`). Its `kind` is an
//! open union tagged `type`: `agent` (agent-driven flow) or `terminal`
//! (run a command), preserving custom/unknown kinds verbatim.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ids::AuthMethodId;
use super::mcp::EnvVar;
use super::open::Open;

/// An authentication method the agent advertises in `initialize`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthMethod {
    /// Method id used by `auth/login` and `auth/logout`.
    pub method_id: AuthMethodId,
    /// Display name.
    pub name: String,
    /// Optional description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// How the method authenticates (flattened `{"type":...}` union).
    #[serde(flatten)]
    pub kind: Open<AuthKind>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// How an [`AuthMethod`] works (tagged `type`); wrapped in [`Open`] at
/// the field to preserve custom/unknown kinds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuthKind {
    /// `{"type":"agent"}` — the agent runs the flow itself.
    Agent(AgentAuth),
    /// `{"type":"terminal","args","env"}` — the client runs a command.
    Terminal(TerminalAuth),
}

impl super::open::Openable for AuthKind {
    const TAG: Option<&'static str> = Some("type");
    const KNOWN: &'static [&'static str] = &["agent", "terminal"];
}

/// `{"type":"agent"}` — marker for agent-driven login.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentAuth {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `{"type":"terminal","args","env"}` — the client runs a login command
/// with argument vector `args` and environment `env`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalAuth {
    /// Command argument vector (the executable is `args[0]`).
    pub args: Vec<String>,
    /// Environment for the command.
    #[serde(default)]
    pub env: Vec<EnvVar>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `auth/login` params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthLoginParams {
    /// Method to log in with.
    pub method_id: AuthMethodId,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `auth/logout` params.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthLogoutParams {
    /// Method to log out of; absent means the current login.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method_id: Option<AuthMethodId>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn auth_method_agent_and_terminal() {
        let raw = json!({"methodId": "m", "name": "Login", "type": "agent"});
        let m: AuthMethod = serde_json::from_value(raw).expect("de");
        assert!(matches!(m.kind, Open::Known(AuthKind::Agent(_))));
        assert_eq!(serde_json::to_value(&m).expect("ser")["type"], "agent");

        let raw = json!({"methodId": "t", "name": "CLI", "type": "terminal",
            "args": ["gh", "auth"], "env": [{"name": "GH_TOKEN", "value": "x"}]});
        let m: AuthMethod = serde_json::from_value(raw).expect("de");
        assert!(
            matches!(m.kind, Open::Known(AuthKind::Terminal(ref t)) if t.args == ["gh", "auth"])
        );

        let raw = json!({"methodId": "x", "name": "X", "type": "_sso", "url": "u"});
        let m: AuthMethod = serde_json::from_value(raw).expect("de");
        assert!(matches!(m.kind, Open::Custom(_, _)));
        assert_eq!(serde_json::to_value(&m).expect("ser")["type"], "_sso");
    }
}
