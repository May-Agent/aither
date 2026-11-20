//! Elicitations: agent-initiated structured input requests.
//!
//! v2 shapes only: `elicitation/create` carries an [`ElicitMode`] (`form`
//! or `url`, flattened `mode` union), a `message`, and `_meta`. The v1
//! `scope` fields and the `pattern` string-schema field have no domain
//! representation (the v1 codec maps scope itself).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ids::ElicitationId;
use super::open::Open;

/// `elicitation/create` params.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitCreate {
    /// How the input is collected (flattened `{"mode":...}` union).
    #[serde(flatten)]
    pub mode: Open<ElicitMode>,
    /// Human-readable prompt explaining what input is needed.
    pub message: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// How an elicitation collects input (tagged `mode`); wrapped in
/// [`Open`] at the field to preserve custom/unknown modes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ElicitMode {
    /// `{"mode":"form","requestedSchema"}` — render a schema-driven form.
    Form(FormMode),
    /// `{"mode":"url","elicitationId","url"}` — send the user to a URL.
    Url(UrlMode),
}

impl super::open::Openable for ElicitMode {
    const TAG: Option<&'static str> = Some("mode");
    const KNOWN: &'static [&'static str] = &["form", "url"];
}

/// `{"mode":"form","requestedSchema"}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FormMode {
    /// The JSON-schema-ish object describing the form fields.
    pub requested_schema: ElicitationSchema,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `{"mode":"url","elicitationId","url"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UrlMode {
    /// Id the completion will echo.
    pub elicitation_id: ElicitationId,
    /// URL the client should open.
    pub url: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The schema of a form elicitation
/// (`{"type":"object","properties","required"?,...}`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitationSchema {
    /// Always `"object"` on the wire.
    #[serde(rename = "type", default = "default_object")]
    pub schema_type: String,
    /// Form title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Property schemas by name.
    #[serde(default)]
    pub properties: BTreeMap<String, Open<ElicitProperty>>,
    /// Required property names.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
    /// Description shown with the form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

fn default_object() -> String {
    "object".to_string()
}

/// A form-field schema (tagged `type`); wrapped in [`Open`] in
/// [`ElicitationSchema::properties`] to preserve custom/unknown types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ElicitProperty {
    /// `{"type":"string",...}`.
    String(StringProperty),
    /// `{"type":"number",...}`.
    Number(NumberProperty),
    /// `{"type":"integer",...}`.
    Integer(IntegerProperty),
    /// `{"type":"boolean",...}`.
    Boolean(BooleanProperty),
    /// `{"type":"array",...}`.
    Array(ArrayProperty),
}

impl super::open::Openable for ElicitProperty {
    const TAG: Option<&'static str> = Some("type");
    const KNOWN: &'static [&'static str] = &["string", "number", "integer", "boolean", "array"];
}

/// Common form-field fields shared by the scalar property schemas.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PropertyCommon {
    /// Field title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Field description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `{"type":"string",...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StringProperty {
    /// Common fields.
    #[serde(flatten)]
    pub common: PropertyCommon,
    /// Minimum length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_length: Option<u64>,
    /// Maximum length.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_length: Option<u64>,
    /// Format hint (`"email"`, `"uri"`, `"date"`, `"date_time"`, custom).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Open<StringFormat>>,
    /// Default value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    /// Closed value set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    /// Named alternatives (`{"const","title"}` objects).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub one_of: Option<Vec<EnumOption>>,
}

/// String format hint (bare string).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StringFormat {
    /// An email address.
    Email,
    /// A URI.
    Uri,
    /// A date.
    Date,
    /// A date and time.
    DateTime,
    /// A hostname.
    Hostname,
    /// An IPv4 address.
    Ipv4,
    /// An IPv6 address.
    Ipv6,
}

impl super::open::Openable for StringFormat {
    const TAG: Option<&'static str> = None;
    const KNOWN: &'static [&'static str] = &[
        "email",
        "uri",
        "date",
        "date_time",
        "hostname",
        "ipv4",
        "ipv6",
    ];
}

/// A named enum alternative (`{"const","title"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnumOption {
    /// The value.
    #[serde(rename = "const")]
    pub const_value: String,
    /// Display title.
    pub title: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `{"type":"number",...}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NumberProperty {
    /// Common fields.
    #[serde(flatten)]
    pub common: PropertyCommon,
    /// Minimum value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    /// Maximum value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
    /// Default value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<f64>,
}

/// `{"type":"integer",...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IntegerProperty {
    /// Common fields.
    #[serde(flatten)]
    pub common: PropertyCommon,
    /// Minimum value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum: Option<i64>,
    /// Maximum value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum: Option<i64>,
    /// Default value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<i64>,
}

/// `{"type":"boolean",...}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BooleanProperty {
    /// Common fields.
    #[serde(flatten)]
    pub common: PropertyCommon,
    /// Default value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<bool>,
}

/// `{"type":"array",...}` — multi-select item schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArrayProperty {
    /// Common fields.
    #[serde(flatten)]
    pub common: PropertyCommon,
    /// Item schema (string or titled options).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub items: Option<Value>,
    /// Minimum items.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_items: Option<u64>,
    /// Maximum items.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_items: Option<u64>,
    /// Default value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Vec<String>>,
}

/// `elicitation/create` result: the completed action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitResult {
    /// The user's action (flattened `{"action":...}` union).
    #[serde(flatten)]
    pub action: Open<ElicitAction>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// What the user did with an elicitation (tagged `action`); wrapped in
/// [`Open`] in [`ElicitResult`] to preserve custom/unknown actions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ElicitAction {
    /// `{"action":"accept","content"?}` — submitted values.
    Accept(AcceptAction),
    /// `{"action":"decline"}` — explicitly declined.
    Decline(DeclineAction),
    /// `{"action":"cancel"}` — dismissed without answering.
    Cancel(CancelAction),
}

impl super::open::Openable for ElicitAction {
    const TAG: Option<&'static str> = Some("action");
    const KNOWN: &'static [&'static str] = &["accept", "decline", "cancel"];
}

/// `{"action":"accept","content"?}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcceptAction {
    /// Submitted field values by name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<BTreeMap<String, ElicitContentValue>>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `{"action":"decline"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeclineAction {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// `{"action":"cancel"}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelAction {
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A submitted field value (untagged scalar/array on the wire).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ElicitContentValue {
    /// A string.
    String(String),
    /// An integer.
    Integer(i64),
    /// A number.
    Number(f64),
    /// A boolean.
    Boolean(bool),
    /// A multi-select selection.
    Array(Vec<String>),
}

/// `elicitation/complete` notification params.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElicitComplete {
    /// The elicitation that finished.
    pub elicitation_id: ElicitationId,
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
    fn create_modes_and_result() {
        let raw =
            json!({"mode": "url", "elicitationId": "e1", "url": "http://x", "message": "login"});
        let c: ElicitCreate = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(c.mode, Open::Known(ElicitMode::Url(_))));
        assert_eq!(serde_json::to_value(&c).expect("ser"), raw);

        let raw = json!({"mode": "form", "requestedSchema": {"type": "object", "properties": {"name": {"type": "string"}}}, "message": "hi"});
        let c: ElicitCreate = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(c.mode, Open::Known(ElicitMode::Form(_))));
        assert_eq!(serde_json::to_value(&c).expect("ser"), raw);

        let raw = json!({"action": "accept", "content": {"name": "x", "tags": ["a"]}});
        let r: ElicitResult = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(r.action, Open::Known(ElicitAction::Accept(_))));
        assert_eq!(serde_json::to_value(&r).expect("ser"), raw);

        let raw = json!({"action": "_defer", "until": "x"});
        let r: ElicitResult = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(r.action, Open::Custom(_, _)));
        assert_eq!(serde_json::to_value(&r).expect("ser"), raw);
    }
}
