//! Slash/available commands.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::open::{ExtIdent, Open, open_union_serde};

/// An available command (open union tagged `type`; `command` is the only
/// known kind).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AvailableCommand {
    /// `{"type":"command","name","description","input"?}`.
    Command(Command),
    /// `_`-prefixed extension command, preserved verbatim.
    Custom(ExtIdent, Value),
    /// Unrecognized command type, preserved verbatim.
    Unknown(String, Value),
}

open_union_serde!(AvailableCommand, "type", [
    Command => "command",
]);

/// `{"type":"command","name","description","input"?}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Command {
    /// Command name (without the leading `/`).
    pub name: String,
    /// What the command does.
    pub description: String,
    /// Expected input shape, when the command takes arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Open<CommandInput>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Expected command input (tagged `type`; only `text` is known).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CommandInput {
    /// `{"type":"text","hint"?}` — free-form text argument.
    Text(TextInput),
}

impl super::open::Openable for CommandInput {
    const TAG: Option<&'static str> = Some("type");
    const KNOWN: &'static [&'static str] = &["text"];
}

/// `{"type":"text","hint"?}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextInput {
    /// Placeholder describing the expected text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// An update carrying the current available commands
/// (`{"sessionUpdate":"available_commands_update","availableCommands":[...]}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AvailableCommandsUpdate {
    /// The full current command list.
    pub available_commands: Vec<AvailableCommand>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}
