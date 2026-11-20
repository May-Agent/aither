//! Session configuration options.
//!
//! A [`ConfigOption`] is a named setting the client can render and change
//! via `session/set_config_option`. `kind` is the option's value schema
//! (`select` or `boolean`, preserving custom/unknown kinds); `category`
//! groups options (`mode`, `model`, `model_config`, `thought_level`,
//! custom/unknown).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ids::{ConfigGroupId, ConfigId, ConfigValueId, SessionId};
use super::open::{ExtIdent, Open, open_tags, open_union_serde};

/// Category a [`ConfigOption`] belongs to (bare string on the wire).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigCategory {
    /// Session operating mode (drives `current_mode_update` on v1 peers).
    Mode,
    /// Model selection.
    Model,
    /// Provider/model-specific sub-option.
    ModelConfig,
    /// Reasoning-effort selection.
    ThoughtLevel,
}
open_tags!(ConfigCategory; None => ["mode", "model", "model_config", "thought_level"]);

/// A single configuration option
/// (`{"configId","name","category"?,` kind fields`...}`).
///
/// `kind` is flattened into the option object on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigOption {
    /// Option id.
    pub config_id: ConfigId,
    /// Display name.
    pub name: String,
    /// Optional description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Option category.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<Open<ConfigCategory>>,
    /// Option value schema and current value.
    #[serde(flatten)]
    pub kind: ConfigKind,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ConfigOption {
    /// `true` when this option's category is `mode`.
    #[must_use]
    pub const fn is_mode(&self) -> bool {
        matches!(&self.category, Some(Open::Known(ConfigCategory::Mode)))
    }
}

/// The value schema of a [`ConfigOption`] (tagged `type`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigKind {
    /// `{"type":"select","currentValue","options"}`.
    Select(SelectConfig),
    /// `{"type":"boolean","currentValue":bool}`.
    Boolean(BooleanConfig),
    /// `_`-prefixed extension kind, preserved verbatim.
    Custom(ExtIdent, Value),
    /// Unrecognized kind, preserved verbatim.
    Unknown(String, Value),
}

open_union_serde!(ConfigKind, "type", [
    Select => "select",
    Boolean => "boolean",
]);

/// A select-style option (`{"type":"select","currentValue","options"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectConfig {
    /// Current value id.
    pub current_value: ConfigValueId,
    /// Offered values, flat or grouped.
    pub options: SelectOptions,
}

/// A boolean option (`{"type":"boolean","currentValue":bool}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BooleanConfig {
    /// Current value.
    pub current_value: bool,
}

/// Selectable values: flat or grouped (untagged on the wire).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SelectOptions {
    /// A flat list of values.
    Ungrouped(Vec<SelectOption>),
    /// Named groups of values.
    Grouped(Vec<SelectGroup>),
}

/// One selectable value (`{"value","name","description"?}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectOption {
    /// Value id (what `currentValue`/`value` carry).
    pub value: ConfigValueId,
    /// Display name.
    pub name: String,
    /// Optional description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A named group of [`SelectOption`]s
/// (`{"groupId","name","options"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectGroup {
    /// Group id.
    pub group_id: ConfigGroupId,
    /// Display name.
    pub name: String,
    /// The group's values.
    pub options: Vec<SelectOption>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A new value for a configuration option (tagged `type`, flattened into
/// `session/set_config_option` params).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigValue {
    /// `{"type":"id","value":"<value-id>"}` — selects a value by id.
    Id(ConfigValueById),
    /// `{"type":"boolean","value":bool}`.
    Boolean(ConfigValueBoolean),
    /// `_`-prefixed extension value shape, preserved verbatim.
    Custom(ExtIdent, Value),
    /// Unrecognized value shape, preserved verbatim.
    Unknown(String, Value),
}

open_union_serde!(ConfigValue, "type", [
    Id => "id",
    Boolean => "boolean",
]);

/// `{"type":"id","value":"<value-id>"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigValueById {
    /// The selected value id.
    pub value: ConfigValueId,
}

/// `{"type":"boolean","value":bool}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigValueBoolean {
    /// The new value.
    pub value: bool,
}

/// `session/set_config_option` params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetConfigParams {
    /// Target session.
    pub session_id: SessionId,
    /// Option to change.
    pub config_id: ConfigId,
    /// New value (flattened `{"type":...,"value":...}`).
    #[serde(flatten)]
    pub value: ConfigValue,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `session/set_config_option` result.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetConfigResult {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// An update carrying the current set of configuration options
/// (`{"sessionUpdate":"config_option_update","configOptions":[...]}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigOptionUpdate {
    /// The full current option list (replaces, not merges).
    pub config_options: Vec<ConfigOption>,
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
    fn config_option_select_and_boolean_round_trip() {
        let raw = json!({
            "configId": "mode", "name": "Mode", "category": "mode",
            "type": "select", "currentValue": "code",
            "options": [{"value": "code", "name": "Code"}]
        });
        let o: ConfigOption = serde_json::from_value(raw).expect("de");
        assert!(o.is_mode());
        assert!(matches!(o.kind, ConfigKind::Select(_)));
        let back = serde_json::to_value(&o).expect("ser");
        assert_eq!(back["type"], "select");
        assert_eq!(back["currentValue"], "code");

        let raw =
            json!({"configId": "diff", "name": "Diff", "type": "boolean", "currentValue": true});
        let o: ConfigOption = serde_json::from_value(raw).expect("de");
        assert!(matches!(o.kind, ConfigKind::Boolean(_)));

        // Unknown kind preserved.
        let raw = json!({"configId": "x", "name": "X", "type": "_dial", "v": 1});
        let o: ConfigOption = serde_json::from_value(raw).expect("de");
        assert!(matches!(o.kind, ConfigKind::Custom(_, _)));
        let back = serde_json::to_value(&o).expect("ser");
        assert_eq!(back["type"], "_dial");
    }

    #[test]
    fn config_value_tagged() {
        let v: ConfigValue =
            serde_json::from_value(json!({"type": "id", "value": "x"})).expect("de");
        assert!(matches!(v, ConfigValue::Id(_)));
        let v: ConfigValue =
            serde_json::from_value(json!({"type": "boolean", "value": true})).expect("de");
        assert!(matches!(v, ConfigValue::Boolean(_)));
        let raw = json!({"type": "spin", "value": 3});
        let v: ConfigValue = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(v, ConfigValue::Unknown(ref t, _) if t == "spin"));
        assert_eq!(serde_json::to_value(&v).expect("ser"), raw);
    }
}
