//! Usage and cost accounting.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A context-usage update (`{"sessionUpdate":"usage_update",...}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageUpdate {
    /// Tokens of context used.
    pub used: u64,
    /// Context window size; optional on the domain model (required on the
    /// v2 schema — missing values tolerate `0` through the v1 codec).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Monetary cost of usage so far.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<Cost>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Monetary cost (`{"amount","currency"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cost {
    /// Amount in `currency`'s smallest unit or a decimal string.
    pub amount: Value,
    /// Currency code (e.g. `"USD"`).
    pub currency: String,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Token usage attached to a finished turn (`PromptFinished.usage`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    /// Sum of all token types.
    #[serde(default)]
    pub total_tokens: u64,
    /// Input tokens.
    #[serde(default)]
    pub input_tokens: u64,
    /// Output tokens.
    #[serde(default)]
    pub output_tokens: u64,
    /// Thought/reasoning tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thought_tokens: Option<u64>,
    /// Cache read tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_read_tokens: Option<u64>,
    /// Cache write tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_write_tokens: Option<u64>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}
