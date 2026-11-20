//! `_session/steering` — inject a prompt into a running turn.
//!
//! A provider-neutral convention: an agent advertises steering support
//! with a `supported: true` flag under `_meta.steering` on its
//! `initialize` result. Steering lets the client add content to a turn
//! that is already running, rather than waiting for it to finish. On ACP
//! v2 a concurrent `session/prompt` is the native equivalent; this method
//! is the v1 lane.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::domain::open::string_enum;
use crate::domain::{ContentBlock, ExtMethod, SessionId};

/// The steering extension method.
pub const METHOD: ExtMethod = ExtMethod::new("_session/steering");

/// `initialize` result `_meta` key carrying the steering support object.
pub const CAPABILITY: &str = "steering";

/// Whether the agent advertised steering support
/// (`meta.steering.supported` on the `initialize` result).
#[must_use]
pub fn advertised(meta: &Option<Value>) -> bool {
    meta.as_ref()
        .and_then(|meta| meta.get(CAPABILITY))
        .and_then(|cap| cap.get("supported"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

string_enum! {
    /// How the agent applied a steering prompt.
    pub enum SteeringOutcome {
        /// The prompt was added to the running turn.
        Injected = "injected",
        /// The prompt could not be merged, so the agent started a new turn.
        StartedNewTurn = "startedNewTurn",
        /// The steering request was rejected.
        Failed = "failed",
    }
}

/// `_session/steering` request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SteeringParams {
    /// Session whose running turn should receive the prompt.
    pub session_id: SessionId,
    /// Prompt content to inject.
    pub prompt: Vec<ContentBlock>,
}

impl SteeringParams {
    /// Build steering params for `session_id`.
    #[must_use]
    pub fn new(session_id: impl Into<SessionId>, prompt: Vec<ContentBlock>) -> Self {
        Self {
            session_id: session_id.into(),
            prompt,
        }
    }
}

/// `_session/steering` result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SteeringResult {
    /// How the agent applied the prompt.
    pub outcome: SteeringOutcome,
    /// Fields this crate does not model, preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

// TODO(wave-3): `steer(&Client, &SessionId, Vec<ContentBlock>)` — a thin
// `Client::ext_request(&METHOD, &SteeringParams)` wrapper returning
// `Result<SteeringOutcome, ClientError>` — once `client::Client` lands.

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// Steering support is a `supported: true` flag under `_meta.steering`.
    #[test]
    fn advertised_flag() {
        let meta = Some(json!({"steering": {"supported": true}}));
        assert!(advertised(&meta));
        let meta = Some(json!({"steering": {"supported": false}}));
        assert!(!advertised(&meta));
        assert!(!advertised(&None));
        assert!(!advertised(&Some(json!({"other": {}}))));
    }

    /// Params carry the prompt content blocks verbatim.
    #[test]
    fn params_wire_format() {
        let params = SteeringParams::new("s1", vec![ContentBlock::text("keep going")]);
        let json = serde_json::to_value(&params).expect("serializes");
        assert_eq!(json["sessionId"], "s1");
        assert_eq!(json["prompt"][0]["type"], "text");
        assert_eq!(json["prompt"][0]["text"], "keep going");
    }

    /// Outcomes tag in camelCase; unknown outcomes survive round-tripping.
    #[test]
    fn outcome_wire_format() {
        let result: SteeringResult =
            serde_json::from_value(json!({"outcome": "startedNewTurn"})).expect("deserializes");
        assert_eq!(result.outcome, SteeringOutcome::StartedNewTurn);

        let result: SteeringResult =
            serde_json::from_value(json!({"outcome": "queued"})).expect("deserializes");
        assert_eq!(result.outcome, SteeringOutcome::Other("queued".to_string()));
        assert_eq!(
            serde_json::to_value(&result).expect("serializes")["outcome"],
            "queued"
        );
    }
}
