//! Stateful ACP v1 ↔ v2 (domain) translation.
//!
//! [`V1Codec`] is per-session: it remembers tool calls seen so far (first
//! `tool_call_update` downgrades to v1 `tool_call`), synthesized message
//! ids for the id-less v1 chunk lanes, the configuration-option cache used
//! to detect the mode category, and the accumulated tool-call content used
//! to flatten `tool_call_content_chunk` into v1 `tool_call_update`s.
//!
//! All conversions keep unknown/custom values wherever v1 has a fallback
//! (`v1::SessionUpdate::Other`, v1 `Other*` elicitation variants).

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::{Value, json};

use crate::domain::{self, Open, Patch};

use super::v1;

/// Default `configId` used for the synthesized mode option when a v1 peer
/// reports `modes` without an explicit config option id.
pub const DEFAULT_MODE_OPTION_ID: &str = "mode";

/// Negotiated wire protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// ACP v1.
    V1,
    /// ACP v2.
    V2,
}

/// Which v1 method an outbound `session/set_config_option` maps to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V1SetConfig {
    /// The option is mode-category → `session/set_mode`.
    SetMode(v1::SessionSetModeParams),
    /// Any other option → `session/set_config_option`.
    SetConfig(v1::SessionSetConfigOptionParams),
    /// The domain value shape has no v1 representation; carries the raw
    /// flattened value for the caller to fail with `-32602`.
    Unsupported(Value),
}

/// Which v1 method a domain resume maps to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V1Resume {
    /// `replayFrom: start` → `session/load`.
    Load(v1::SessionLoadParams),
    /// No replay → `session/resume`.
    Resume(v1::SessionResumeParams),
    /// A cursor v1 cannot express; raw for `-32602`.
    UnsupportedCursor(Value),
}

/// Result of translating a domain [`domain::SetupResult`] for a v1 peer:
/// the fields to merge into the v1 `session/*` response plus follow-up
/// `session/update` notifications.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct V1SetupWire {
    /// `modes` for the response (synthesized from the mode option).
    pub modes: Option<v1::SessionModeState>,
    /// `configOptions` for the response.
    pub config_options: Vec<v1::ConfigOption>,
    /// `session/update` notifications to send after the response
    /// (e.g. `available_commands_update`, which v1 cannot put in the
    /// response itself).
    pub pending_updates: Vec<v1::SessionUpdate>,
}

/// Per-session v1↔v2 translator. See module docs.
#[derive(Debug, Default)]
pub struct V1Codec {
    seen_tool_calls: HashSet<String>,
    msg_ids: HashMap<&'static str, String>,
    config_options: BTreeMap<String, domain::ConfigOption>,
    mode_option_id: Option<String>,
    tool_content: BTreeMap<String, Vec<v1::ToolCallContent>>,
}

impl V1Codec {
    /// A codec with no state (e.g. for a freshly negotiated session).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn msg_id(&mut self, kind: &'static str) -> String {
        self.msg_ids
            .entry(kind)
            .or_insert_with(|| domain::MessageId::new().into_string())
            .clone()
    }

    fn cache_options(&mut self, options: &[domain::ConfigOption]) {
        for option in options {
            if option.is_mode() && self.mode_option_id.is_none() {
                self.mode_option_id = Some(option.config_id.to_string());
            }
            self.config_options
                .insert(option.config_id.to_string(), option.clone());
        }
    }

    /// Domain session update → zero or more v1 updates.
    ///
    /// v2-only updates (state, terminal, subagent, session messages,
    /// compaction, notice, replay) drop — v1 has no lane for them.
    pub fn update_to_v1(&mut self, update: &domain::SessionUpdate) -> Vec<v1::SessionUpdate> {
        use domain::SessionUpdate as U;
        match update {
            U::AgentThoughtChunk(c) => vec![v1::SessionUpdate::AgentThoughtChunk(chunk_to_v1(c))],
            U::AgentMessageChunk(c) => vec![v1::SessionUpdate::AgentMessageChunk(chunk_to_v1(c))],
            U::UserMessageChunk(c) => vec![v1::SessionUpdate::UserMessageChunk(chunk_to_v1(c))],
            U::AgentThought(m) => self.message_to_v1("thought", m),
            U::AgentMessage(m) => self.message_to_v1("agent", m),
            U::UserMessage(m) => self.message_to_v1("user", m),
            U::ToolCallUpdate(u) => vec![self.tool_call_update_to_v1(u)],
            U::ToolCallContentChunk(chunk) => {
                let all = self
                    .tool_content
                    .entry(chunk.tool_call_id.to_string())
                    .or_default();
                if let Some(c) = tool_content_to_v1(&chunk.content) {
                    all.push(c);
                }
                vec![v1::SessionUpdate::ToolCallUpdate(v1::ToolCallUpdate {
                    tool_call_id: chunk.tool_call_id.to_string(),
                    status: None,
                    content: Some(all.clone()),
                    title: None,
                    kind: None,
                    locations: None,
                    raw_input: None,
                    raw_output: None,
                    meta: chunk.meta.clone(),
                })]
            }
            U::PlanUpdate(p) => match &p.plan {
                Open::Known(domain::Plan::Items(items)) => {
                    vec![v1::SessionUpdate::Plan(v1::Plan {
                        entries: items.entries.iter().map(plan_entry_to_v1).collect(),
                        meta: p.meta.clone(),
                    })]
                }
                Open::Custom(_, raw) | Open::Unknown(_, raw) => {
                    vec![v1::SessionUpdate::Other(raw.clone())]
                }
            },
            U::ConfigOptionUpdate(u) => self.config_option_update_to_v1(u),
            U::CurrentModeUpdate(u) => vec![v1::SessionUpdate::CurrentModeUpdate(
                v1::CurrentModeUpdate {
                    current_mode_id: u.current_mode_id.to_string(),
                    meta: u.meta.clone(),
                },
            )],
            U::AvailableCommandsUpdate(u) => vec![v1::SessionUpdate::AvailableCommandsUpdate(
                v1::AvailableCommandsUpdate {
                    available_commands: u
                        .available_commands
                        .iter()
                        .filter_map(command_to_v1)
                        .collect(),
                    meta: u.meta.clone(),
                },
            )],
            U::SessionInfoUpdate(u) => {
                vec![v1::SessionUpdate::SessionInfoUpdate(
                    v1::SessionInfoUpdate {
                        title: u.title.clone().into_option(),
                        updated_at: u.updated_at.clone().into_option(),
                        meta: u.meta.clone().into_option(),
                    },
                )]
            }
            U::UsageUpdate(u) => vec![v1::SessionUpdate::UsageUpdate(v1::UsageUpdate {
                used: u.used,
                size: u.size.unwrap_or(0),
                cost: u.cost.as_ref().map(cost_to_v1),
                meta: u.meta.clone(),
            })],
            U::Custom(_, raw) | U::Unknown(_, raw) => vec![v1::SessionUpdate::Other(raw.clone())],
            // v2-only lanes: state, terminals, subagents, session messages,
            // compaction, notices, replay.
            U::StateUpdate(_)
            | U::TerminalUpdate(_)
            | U::TerminalOutputChunk(_)
            | U::SubagentUpdate(_)
            | U::SessionMessage(_)
            | U::SessionMessageChunk(_)
            | U::CompactionUpdate(_)
            | U::CompactionSummaryChunk(_)
            | U::Notice(_)
            | U::ReplayComplete(_) => Vec::new(),
        }
    }

    /// Whole user/agent/thought message upsert → one v1 content chunk per
    /// block, all sharing the lane's synthesized message id.
    fn message_to_v1(&mut self, kind: &'static str, m: &domain::Message) -> Vec<v1::SessionUpdate> {
        let id = self.msg_id(kind);
        m.content.as_option().map_or_else(Vec::new, |blocks| {
            blocks
                .iter()
                .filter_map(content_to_v1)
                .map(|content| {
                    let chunk = v1::ContentChunk {
                        content,
                        message_id: Some(id.clone()),
                        meta: m.meta.clone(),
                    };
                    match kind {
                        "thought" => v1::SessionUpdate::AgentThoughtChunk(chunk),
                        "agent" => v1::SessionUpdate::AgentMessageChunk(chunk),
                        _ => v1::SessionUpdate::UserMessageChunk(chunk),
                    }
                })
                .collect()
        })
    }

    /// Domain tool-call upsert → v1 `tool_call` (first sighting of the id)
    /// or `tool_call_update`.
    pub fn tool_call_update_to_v1(&mut self, u: &domain::ToolCallUpdate) -> v1::SessionUpdate {
        if self.seen_tool_calls.insert(u.tool_call_id.to_string()) {
            v1::SessionUpdate::ToolCall(v1::ToolCall {
                tool_call_id: u.tool_call_id.to_string(),
                title: u.title.clone().into_option().unwrap_or_default(),
                kind: u.kind.as_option().map(tool_kind_to_v1),
                status: u.status.as_option().and_then(tool_status_to_v1),
                content: u
                    .content
                    .clone()
                    .into_option()
                    .map_or_default(|cs| cs.iter().filter_map(tool_content_to_v1).collect()),
                locations: u
                    .locations
                    .clone()
                    .into_option()
                    .map_or_default(|ls| ls.iter().map(location_to_v1).collect()),
                raw_input: u.raw_input.clone().into_option(),
                raw_output: u.raw_output.clone().into_option(),
                meta: u.meta.clone().into_option(),
            })
        } else {
            v1::SessionUpdate::ToolCallUpdate(v1::ToolCallUpdate {
                tool_call_id: u.tool_call_id.to_string(),
                status: u.status.as_option().and_then(tool_status_to_v1),
                content: u
                    .content
                    .clone()
                    .into_option()
                    .map(|cs| cs.iter().filter_map(tool_content_to_v1).collect()),
                title: u.title.clone().into_option(),
                kind: u.kind.as_option().map(tool_kind_to_v1),
                locations: u
                    .locations
                    .clone()
                    .into_option()
                    .map(|ls| ls.iter().map(location_to_v1).collect()),
                raw_input: u.raw_input.clone().into_option(),
                raw_output: u.raw_output.clone().into_option(),
                meta: u.meta.clone().into_option(),
            })
        }
    }

    /// v1 session update → domain update. Message ids are synthesized
    /// per chunk lane.
    pub fn update_from_v1(&mut self, update: v1::SessionUpdate) -> domain::SessionUpdate {
        use v1::SessionUpdate as U;
        match update {
            U::AgentThoughtChunk(c) => {
                domain::SessionUpdate::AgentThoughtChunk(self.chunk_to_domain(c, "thought"))
            }
            U::AgentMessageChunk(c) => {
                domain::SessionUpdate::AgentMessageChunk(self.chunk_to_domain(c, "agent"))
            }
            U::UserMessageChunk(c) => {
                domain::SessionUpdate::UserMessageChunk(self.chunk_to_domain(c, "user"))
            }
            U::Plan(p) => domain::SessionUpdate::PlanUpdate(domain::PlanUpdate {
                plan: Open::Known(domain::Plan::Items(domain::PlanItems {
                    plan_id: domain::PlanId::from("default"),
                    entries: p.entries.iter().map(plan_entry_to_domain).collect(),
                    meta: p.meta.clone(),
                    extra: BTreeMap::new(),
                })),
                meta: p.meta,
                extra: BTreeMap::new(),
            }),
            U::ToolCall(c) => domain::SessionUpdate::ToolCallUpdate(tool_call_to_domain(&c)),
            U::ToolCallUpdate(u) => {
                domain::SessionUpdate::ToolCallUpdate(tool_call_patch_to_domain(u))
            }
            U::AvailableCommandsUpdate(u) => {
                domain::SessionUpdate::AvailableCommandsUpdate(domain::AvailableCommandsUpdate {
                    available_commands: u
                        .available_commands
                        .iter()
                        .map(command_to_domain)
                        .collect(),
                    meta: u.meta,
                    extra: BTreeMap::new(),
                })
            }
            U::CurrentModeUpdate(u) => {
                domain::SessionUpdate::CurrentModeUpdate(domain::CurrentModeUpdate {
                    current_mode_id: u.current_mode_id.into(),
                    meta: u.meta,
                    extra: BTreeMap::new(),
                })
            }
            U::ConfigOptionUpdate(u) => {
                domain::SessionUpdate::ConfigOptionUpdate(domain::ConfigOptionUpdate {
                    config_options: u
                        .config_options
                        .iter()
                        .map(|o| self.config_option_to_domain(o))
                        .collect(),
                    meta: u.meta,
                    extra: BTreeMap::new(),
                })
            }
            U::SessionInfoUpdate(u) => {
                domain::SessionUpdate::SessionInfoUpdate(domain::SessionInfoUpdate {
                    title: u.title.map_or(Patch::Unchanged, Patch::Set),
                    updated_at: u.updated_at.map_or(Patch::Unchanged, Patch::Set),
                    meta: u.meta.map_or(Patch::Unchanged, Patch::Set),
                    extra: BTreeMap::new(),
                })
            }
            U::UsageUpdate(u) => domain::SessionUpdate::UsageUpdate(domain::UsageUpdate {
                used: u.used,
                size: Some(u.size),
                cost: u.cost.as_ref().map(cost_to_domain),
                meta: u.meta,
                extra: BTreeMap::new(),
            }),
            U::Other(raw) => {
                let tag = raw
                    .get("sessionUpdate")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if let Ok(id) = domain::ExtIdent::try_new(tag.clone()) {
                    domain::SessionUpdate::Custom(id, raw)
                } else {
                    domain::SessionUpdate::Unknown(tag, raw)
                }
            }
        }
    }

    fn chunk_to_domain(&mut self, c: v1::ContentChunk, kind: &'static str) -> domain::ContentChunk {
        let message_id = c
            .message_id
            .clone()
            .unwrap_or_else(|| self.msg_id(kind))
            .into();
        domain::ContentChunk {
            message_id,
            content: content_to_domain(&c.content),
            meta: c.meta,
            extra: BTreeMap::new(),
        }
    }

    /// v1 config option → domain option (kind reconstructed from the flat
    /// `type`/`currentValue`/`options` fields; unknown kinds preserved).
    pub fn config_option_to_domain(&self, o: &v1::ConfigOption) -> domain::ConfigOption {
        let kind = match (o.kind.as_deref(), &o.current_value) {
            (Some("boolean"), _) | (_, Some(v1::ConfigOptionValue::Toggle(_))) => {
                let current = match &o.current_value {
                    Some(v1::ConfigOptionValue::Toggle(b)) => *b,
                    _ => false,
                };
                domain::ConfigKind::Boolean(domain::BooleanConfig {
                    current_value: current,
                })
            }
            (Some("select") | None, _) => {
                let current_value = match &o.current_value {
                    Some(v1::ConfigOptionValue::Selected(s)) => s.clone(),
                    _ => String::new(),
                };
                let options = o.options.as_ref().map_or(
                    domain::SelectOptions::Ungrouped(Vec::new()),
                    select_options_to_domain,
                );
                domain::ConfigKind::Select(domain::SelectConfig {
                    current_value: current_value.into(),
                    options,
                })
            }
            (Some(other), _) => {
                let raw = json!({
                    "type": other,
                    "currentValue": o.current_value,
                    "options": o.options,
                });
                match domain::ExtIdent::try_new(other.to_string()) {
                    Ok(id) => domain::ConfigKind::Custom(id, raw),
                    Err(_) => domain::ConfigKind::Unknown(other.to_string(), raw),
                }
            }
        };
        domain::ConfigOption {
            config_id: o.id.clone().into(),
            name: o.name.clone(),
            description: o.description.clone(),
            category: o.category.clone().map(config_category),
            kind,
            meta: o.meta.clone(),
            extra: BTreeMap::new(),
        }
    }

    /// Domain config option → flat v1 option.
    #[must_use]
    pub fn config_option_to_v1(&self, o: &domain::ConfigOption) -> v1::ConfigOption {
        let (kind, current_value, options) = match &o.kind {
            domain::ConfigKind::Select(sel) => (
                Some("select".to_string()),
                Some(v1::ConfigOptionValue::Selected(
                    sel.current_value.to_string(),
                )),
                Some(select_options_to_v1(&sel.options)),
            ),
            domain::ConfigKind::Boolean(b) => (
                Some("boolean".to_string()),
                Some(v1::ConfigOptionValue::Toggle(b.current_value)),
                None,
            ),
            domain::ConfigKind::Custom(_, raw) | domain::ConfigKind::Unknown(_, raw) => {
                let current = raw.get("currentValue").cloned();
                let options = raw.get("options").cloned();
                let kind = raw.get("type").and_then(Value::as_str).map(str::to_string);
                (
                    kind,
                    current.and_then(|v| serde_json::from_value(v).ok()),
                    options.and_then(|v| serde_json::from_value(v).ok()),
                )
            }
        };
        v1::ConfigOption {
            id: o.config_id.to_string(),
            name: o.name.clone(),
            description: o.description.clone(),
            category: o.category.as_ref().map(|c| match c {
                Open::Known(domain::ConfigCategory::Mode) => "mode".to_string(),
                Open::Known(domain::ConfigCategory::Model) => "model".to_string(),
                Open::Known(domain::ConfigCategory::ModelConfig) => "model_config".to_string(),
                Open::Known(domain::ConfigCategory::ThoughtLevel) => "thought_level".to_string(),
                Open::Custom(id, _) => id.to_string(),
                Open::Unknown(tag, _) => tag.clone(),
            }),
            kind,
            current_value,
            options,
            meta: o.meta.clone(),
        }
    }

    fn config_option_update_to_v1(
        &mut self,
        u: &domain::ConfigOptionUpdate,
    ) -> Vec<v1::SessionUpdate> {
        self.cache_options(&u.config_options);
        let mut out = vec![v1::SessionUpdate::ConfigOptionUpdate(
            v1::ConfigOptionUpdate {
                config_options: u
                    .config_options
                    .iter()
                    .map(|o| self.config_option_to_v1(o))
                    .collect(),
                meta: u.meta.clone(),
            },
        )];
        // A mode-category select also drives v1 `current_mode_update`.
        for option in &u.config_options {
            if option.is_mode()
                && let domain::ConfigKind::Select(sel) = &option.kind
            {
                out.push(v1::SessionUpdate::CurrentModeUpdate(
                    v1::CurrentModeUpdate {
                        current_mode_id: sel.current_value.to_string(),
                        meta: u.meta.clone(),
                    },
                ));
            }
        }
        out
    }

    /// Domain setup result → v1 response fields + follow-up updates.
    /// `modes` is synthesized from the mode-category option; v1 keeps all
    /// options in `configOptions` too.
    pub fn setup_result_to_v1(&mut self, r: &domain::SetupResult) -> V1SetupWire {
        self.cache_options(&r.config_options);
        let modes = r
            .config_options
            .iter()
            .find(|o| o.is_mode())
            .and_then(|o| match &o.kind {
                domain::ConfigKind::Select(sel) => Some(v1::SessionModeState {
                    current_mode_id: sel.current_value.to_string(),
                    available_modes: match &sel.options {
                        domain::SelectOptions::Ungrouped(opts) => opts
                            .iter()
                            .map(|o| v1::SessionMode {
                                id: o.value.to_string(),
                                name: o.name.clone(),
                                description: o.description.clone(),
                                meta: o.meta.clone(),
                            })
                            .collect(),
                        domain::SelectOptions::Grouped(groups) => groups
                            .iter()
                            .flat_map(|g| g.options.iter())
                            .map(|o| v1::SessionMode {
                                id: o.value.to_string(),
                                name: o.name.clone(),
                                description: o.description.clone(),
                                meta: o.meta.clone(),
                            })
                            .collect(),
                    },
                    meta: None,
                }),
                _ => None,
            });
        let mut pending = Vec::new();
        if !r.available_commands.is_empty() {
            pending.push(v1::SessionUpdate::AvailableCommandsUpdate(
                v1::AvailableCommandsUpdate {
                    available_commands: r
                        .available_commands
                        .iter()
                        .filter_map(command_to_v1)
                        .collect(),
                    meta: None,
                },
            ));
        }
        V1SetupWire {
            modes,
            config_options: r
                .config_options
                .iter()
                .map(|o| self.config_option_to_v1(o))
                .collect(),
            pending_updates: pending,
        }
    }

    /// v1 `session/new`-style result → domain setup result (modes become
    /// a synthesized mode-category option prepended to `config_options`).
    pub fn setup_result_from_v1(&mut self, r: &v1::SessionNewResult) -> domain::SetupResult {
        let mut config_options = Vec::new();
        if let Some(modes) = &r.modes {
            self.mode_option_id = Some(DEFAULT_MODE_OPTION_ID.to_string());
            config_options.push(domain::ConfigOption {
                config_id: DEFAULT_MODE_OPTION_ID.into(),
                name: "Mode".to_string(),
                description: None,
                category: Some(Open::Known(domain::ConfigCategory::Mode)),
                kind: domain::ConfigKind::Select(domain::SelectConfig {
                    current_value: modes.current_mode_id.clone().into(),
                    options: domain::SelectOptions::Ungrouped(
                        modes
                            .available_modes
                            .iter()
                            .map(|m| domain::SelectOption {
                                value: m.id.clone().into(),
                                name: m.name.clone(),
                                description: m.description.clone(),
                                meta: m.meta.clone(),
                            })
                            .collect(),
                    ),
                }),
                meta: modes.meta.clone(),
                extra: BTreeMap::new(),
            });
        }
        if let Some(options) = &r.config_options {
            config_options.extend(options.iter().map(|o| self.config_option_to_domain(o)));
        }
        self.cache_options(&config_options);
        domain::SetupResult {
            session_id: r.session_id.clone().into(),
            config_options,
            available_commands: Vec::new(),
            meta: r.meta.clone(),
            extra: r.extra.clone(),
        }
    }

    /// Domain `session/set_config_option` → v1 method params. The
    /// mode-category option maps to `session/set_mode`.
    #[must_use]
    pub fn set_config_to_v1(&self, p: &domain::SetConfigParams) -> V1SetConfig {
        let is_mode = self
            .config_options
            .get(&p.config_id.to_string())
            .is_some_and(domain::ConfigOption::is_mode)
            || self.mode_option_id.as_deref() == Some(p.config_id.as_str());
        if is_mode {
            let mode_id = match &p.value {
                domain::ConfigValue::Id(v) => v.value.to_string(),
                domain::ConfigValue::Boolean(b) => b.value.to_string(),
                domain::ConfigValue::Custom(_, raw) | domain::ConfigValue::Unknown(_, raw) => {
                    return V1SetConfig::Unsupported(raw.clone());
                }
            };
            return V1SetConfig::SetMode(v1::SessionSetModeParams {
                session_id: p.session_id.to_string(),
                mode_id,
                meta: p.meta.clone(),
            });
        }
        let value = match &p.value {
            domain::ConfigValue::Id(v) => v1::SessionConfigValue::Select {
                value: v.value.to_string(),
            },
            domain::ConfigValue::Boolean(b) => v1::SessionConfigValue::Boolean {
                kind: v1::BooleanConfigTag::Boolean,
                value: b.value,
            },
            domain::ConfigValue::Custom(_, raw) | domain::ConfigValue::Unknown(_, raw) => {
                return V1SetConfig::Unsupported(raw.clone());
            }
        };
        V1SetConfig::SetConfig(v1::SessionSetConfigOptionParams {
            session_id: p.session_id.to_string(),
            config_id: p.config_id.to_string(),
            value,
            meta: p.meta.clone(),
        })
    }

    /// v1 `session/set_mode` → domain `session/set_config_option` on the
    /// mode-category option (or [`DEFAULT_MODE_OPTION_ID`]).
    #[must_use]
    pub fn set_mode_to_domain(&self, p: &v1::SessionSetModeParams) -> domain::SetConfigParams {
        domain::SetConfigParams {
            session_id: p.session_id.clone().into(),
            config_id: self
                .mode_option_id
                .clone()
                .unwrap_or_else(|| DEFAULT_MODE_OPTION_ID.to_string())
                .into(),
            value: domain::ConfigValue::Id(domain::ConfigValueById {
                value: p.mode_id.clone().into(),
            }),
            meta: p.meta.clone(),
            extra: BTreeMap::new(),
        }
    }

    /// v1 `session/set_config_option` → domain params.
    #[must_use]
    pub fn set_config_to_domain(
        &self,
        p: &v1::SessionSetConfigOptionParams,
    ) -> domain::SetConfigParams {
        let value = match &p.value {
            v1::SessionConfigValue::Select { value } => {
                domain::ConfigValue::Id(domain::ConfigValueById {
                    value: value.clone().into(),
                })
            }
            v1::SessionConfigValue::Boolean { value, .. } => {
                domain::ConfigValue::Boolean(domain::ConfigValueBoolean { value: *value })
            }
        };
        domain::SetConfigParams {
            session_id: p.session_id.clone().into(),
            config_id: p.config_id.clone().into(),
            value,
            meta: p.meta.clone(),
            extra: BTreeMap::new(),
        }
    }

    /// Domain auth method → v1 method (`methodId`→`id`, terminal `env`
    /// vector→map; custom/unknown kinds degrade to an agent method).
    #[must_use]
    pub fn auth_method_to_v1(&self, m: &domain::AuthMethod) -> v1::AuthMethod {
        match m.kind.as_known() {
            Some(domain::AuthKind::Terminal(t)) => {
                v1::AuthMethod::Terminal(v1::AuthMethodTerminal {
                    id: m.method_id.to_string(),
                    name: m.name.clone(),
                    description: m.description.clone(),
                    args: t.args.clone(),
                    env: t
                        .env
                        .iter()
                        .map(|e| (e.name.clone(), e.value.clone()))
                        .collect(),
                    meta: m.meta.clone(),
                })
            }
            _ => v1::AuthMethod::Agent(v1::AuthMethodAgent {
                id: m.method_id.to_string(),
                name: m.name.clone(),
                description: m.description.clone(),
                meta: m.meta.clone(),
            }),
        }
    }

    /// v1 auth method → domain (`id`→`methodId`, env map→vector).
    #[must_use]
    pub fn auth_method_to_domain(&self, m: &v1::AuthMethod) -> domain::AuthMethod {
        match m {
            v1::AuthMethod::Terminal(t) => domain::AuthMethod {
                method_id: t.id.clone().into(),
                name: t.name.clone(),
                description: t.description.clone(),
                kind: Open::Known(domain::AuthKind::Terminal(domain::TerminalAuth {
                    args: t.args.clone(),
                    env: t
                        .env
                        .iter()
                        .map(|(k, v)| domain::EnvVar {
                            name: k.clone(),
                            value: v.clone(),
                        })
                        .collect(),
                    meta: None,
                    extra: BTreeMap::new(),
                })),
                meta: t.meta.clone(),
                extra: BTreeMap::new(),
            },
            v1::AuthMethod::Agent(a) => domain::AuthMethod {
                method_id: a.id.clone().into(),
                name: a.name.clone(),
                description: a.description.clone(),
                kind: Open::Known(domain::AuthKind::Agent(domain::AgentAuth {
                    meta: None,
                    extra: BTreeMap::new(),
                })),
                meta: a.meta.clone(),
                extra: BTreeMap::new(),
            },
        }
    }

    /// Domain permission request → v1 params: the subject becomes a full
    /// v1 `toolCall` (command subjects synthesize an Execute-kind call).
    #[must_use]
    pub fn permission_to_v1(&self, p: &domain::PermissionRequest) -> v1::RequestPermissionParams {
        let tool_call = match &p.subject {
            Some(Open::Known(domain::PermissionSubject::ToolCall(s))) => {
                v1_tool_call_from_update(&s.tool_call)
            }
            Some(Open::Known(domain::PermissionSubject::Command(c))) => v1::ToolCall {
                tool_call_id: c
                    .tool_call_id
                    .clone()
                    .unwrap_or_else(domain::ToolCallId::new)
                    .to_string(),
                title: c.command.clone(),
                kind: Some(v1::ToolKind::Execute),
                status: None,
                content: Vec::new(),
                locations: c
                    .cwd
                    .iter()
                    .map(|cwd| v1::ToolCallLocation {
                        path: cwd.clone(),
                        line: None,
                    })
                    .collect(),
                raw_input: Some(json!({"command": c.command, "cwd": c.cwd})),
                raw_output: None,
                meta: c.meta.clone(),
            },
            Some(Open::Custom(_, raw) | Open::Unknown(_, raw)) => v1::ToolCall {
                tool_call_id: domain::ToolCallId::new().to_string(),
                title: p.title.clone(),
                kind: Some(v1::ToolKind::Other),
                status: None,
                content: Vec::new(),
                locations: Vec::new(),
                raw_input: Some(raw.clone()),
                raw_output: None,
                meta: p.meta.clone(),
            },
            None => v1::ToolCall {
                tool_call_id: domain::ToolCallId::new().to_string(),
                title: p.title.clone(),
                kind: None,
                status: None,
                content: Vec::new(),
                locations: Vec::new(),
                raw_input: None,
                raw_output: None,
                meta: p.meta.clone(),
            },
        };
        v1::RequestPermissionParams {
            session_id: p.session_id.to_string(),
            tool_call,
            options: p
                .options
                .iter()
                .map(|o| v1::PermissionOption {
                    option_id: o.option_id.to_string(),
                    name: o.name.clone(),
                    kind: permission_kind_to_v1(&o.kind),
                    meta: o.meta.clone(),
                })
                .collect(),
            meta: p.meta.clone(),
        }
    }

    /// v1 permission request → domain (the v1 `toolCall` becomes a
    /// `tool_call` subject carrying a fully-set [`domain::ToolCallUpdate`]).
    #[must_use]
    pub fn permission_to_domain(
        &self,
        p: &v1::RequestPermissionParams,
    ) -> domain::PermissionRequest {
        domain::PermissionRequest {
            session_id: p.session_id.clone().into(),
            title: p.tool_call.title.clone(),
            description: None,
            subject: Some(Open::Known(domain::PermissionSubject::ToolCall(Box::new(
                domain::ToolCallSubject {
                    tool_call: tool_call_update_from_snapshot(&p.tool_call),
                    meta: None,
                    extra: BTreeMap::new(),
                },
            )))),
            options: p
                .options
                .iter()
                .map(|o| domain::PermissionOption {
                    option_id: o.option_id.clone().into(),
                    name: o.name.clone(),
                    kind: Open::Known(permission_kind_to_domain(o.kind)),
                    meta: o.meta.clone(),
                    extra: BTreeMap::new(),
                })
                .collect(),
            meta: p.meta.clone(),
            extra: BTreeMap::new(),
        }
    }

    /// v1 permission outcome → domain.
    #[must_use]
    pub fn outcome_to_domain(&self, o: &v1::RequestPermissionOutcome) -> domain::PermissionOutcome {
        match o {
            v1::RequestPermissionOutcome::Cancelled => {
                domain::PermissionOutcome::Cancelled(domain::CancelledOutcome { meta: None })
            }
            v1::RequestPermissionOutcome::Selected { option_id } => {
                domain::PermissionOutcome::Selected(domain::SelectedOutcome {
                    option_id: option_id.clone().into(),
                    meta: None,
                })
            }
        }
    }

    /// Domain permission outcome → v1 (unknown outcomes map to
    /// `cancelled`, the only safe v1 fallback).
    #[must_use]
    pub fn outcome_to_v1(&self, o: &domain::PermissionOutcome) -> v1::RequestPermissionOutcome {
        match o {
            domain::PermissionOutcome::Selected(s) => v1::RequestPermissionOutcome::Selected {
                option_id: s.option_id.to_string(),
            },
            _ => v1::RequestPermissionOutcome::Cancelled,
        }
    }

    /// Domain MCP server → v1 descriptor (stdio drops the `type` tag and
    /// always emits `args`/`env`; v1 `sse` recovered from `Unknown`).
    #[must_use]
    pub fn mcp_server_to_v1(&self, s: &domain::McpServer) -> Option<v1::McpServer> {
        match s {
            domain::McpServer::Stdio(d) => Some(v1::McpServer::Stdio(v1::McpServerStdio {
                name: d.name.clone(),
                command: d.command.clone(),
                args: d.args.clone(),
                env: d
                    .env
                    .iter()
                    .map(|e| v1::EnvVar {
                        name: e.name.clone(),
                        value: e.value.clone(),
                    })
                    .collect(),
                meta: d.meta.clone(),
            })),
            domain::McpServer::Http(d) => Some(v1::McpServer::Http(v1::McpServerHttp {
                name: d.name.clone(),
                url: d.url.clone(),
                headers: d
                    .headers
                    .iter()
                    .map(|h| v1::HttpHeader {
                        name: h.name.clone(),
                        value: h.value.clone(),
                    })
                    .collect(),
                meta: d.meta.clone(),
            })),
            domain::McpServer::Custom(_, raw) | domain::McpServer::Unknown(_, raw) => {
                serde_json::from_value::<v1::McpServer>(raw.clone()).ok()
            }
        }
    }

    /// v1 MCP server → domain (stdio gains `"type":"stdio"`; `sse` lands
    /// in [`domain::McpServer::Unknown`]).
    #[must_use]
    pub fn mcp_server_to_domain(&self, s: &v1::McpServer) -> domain::McpServer {
        match s {
            v1::McpServer::Stdio(d) => domain::McpServer::Stdio(domain::McpServerStdio {
                name: d.name.clone(),
                command: d.command.clone(),
                args: d.args.clone(),
                env: d
                    .env
                    .iter()
                    .map(|e| domain::EnvVar {
                        name: e.name.clone(),
                        value: e.value.clone(),
                    })
                    .collect(),
                meta: d.meta.clone(),
                extra: BTreeMap::new(),
            }),
            v1::McpServer::Http(d) => domain::McpServer::Http(domain::McpServerHttp {
                name: d.name.clone(),
                url: d.url.clone(),
                headers: d
                    .headers
                    .iter()
                    .map(|h| domain::HttpHeader {
                        name: h.name.clone(),
                        value: h.value.clone(),
                    })
                    .collect(),
                meta: d.meta.clone(),
                extra: BTreeMap::new(),
            }),
            v1::McpServer::Sse(d) => domain::McpServer::Unknown(
                "sse".to_string(),
                json!({"type": "sse", "name": d.name, "url": d.url, "headers": d.headers, "_meta": d.meta}),
            ),
        }
    }

    /// Domain resume params → v1 method choice (`replayFrom` decides
    /// `session/load` vs `session/resume`).
    #[must_use]
    pub fn resume_to_v1(&self, p: &domain::ResumeParams) -> V1Resume {
        let mcp_servers = p
            .setup
            .mcp_servers
            .iter()
            .filter_map(|s| self.mcp_server_to_v1(s))
            .collect();
        match &p.replay_from {
            None => V1Resume::Resume(v1::SessionResumeParams {
                session_id: p.session_id.to_string(),
                cwd: p.setup.cwd.clone(),
                mcp_servers,
                additional_directories: p.setup.additional_directories.clone(),
                meta: p.setup.meta.clone(),
            }),
            Some(Open::Known(domain::ReplayCursor::Start(_))) => {
                V1Resume::Load(v1::SessionLoadParams {
                    session_id: p.session_id.to_string(),
                    cwd: p.setup.cwd.clone(),
                    mcp_servers,
                    additional_directories: p.setup.additional_directories.clone(),
                    meta: p.setup.meta.clone(),
                })
            }
            Some(Open::Custom(_, raw) | Open::Unknown(_, raw)) => {
                V1Resume::UnsupportedCursor(raw.clone())
            }
        }
    }

    /// v1 `session/load` → domain resume with `replayFrom: start`.
    #[must_use]
    pub fn load_to_domain(&self, p: &v1::SessionLoadParams) -> domain::ResumeParams {
        domain::ResumeParams {
            setup: self.setup_to_domain(
                &p.cwd,
                &p.mcp_servers,
                &p.additional_directories,
                p.meta.clone(),
            ),
            session_id: p.session_id.clone().into(),
            replay_from: Some(Open::Known(domain::ReplayCursor::Start(
                domain::ReplayStart { meta: None },
            ))),
        }
    }

    /// v1 `session/resume` → domain resume without replay.
    #[must_use]
    pub fn resume_to_domain(&self, p: &v1::SessionResumeParams) -> domain::ResumeParams {
        domain::ResumeParams {
            setup: self.setup_to_domain(
                &p.cwd,
                &p.mcp_servers,
                &p.additional_directories,
                p.meta.clone(),
            ),
            session_id: p.session_id.clone().into(),
            replay_from: None,
        }
    }

    /// v1 `session/new` → domain setup params.
    #[must_use]
    pub fn new_to_domain(&self, p: &v1::SessionNewParams) -> domain::SessionSetupParams {
        self.setup_to_domain(
            &p.cwd,
            &p.mcp_servers,
            &p.additional_directories,
            p.meta.clone(),
        )
    }

    /// Domain setup params → v1 `session/new` params.
    #[must_use]
    pub fn setup_to_v1(&self, s: &domain::SessionSetupParams) -> v1::SessionNewParams {
        v1::SessionNewParams {
            cwd: s.cwd.clone(),
            mcp_servers: s
                .mcp_servers
                .iter()
                .filter_map(|m| self.mcp_server_to_v1(m))
                .collect(),
            additional_directories: s.additional_directories.clone(),
            meta: s.meta.clone(),
        }
    }

    fn setup_to_domain(
        &self,
        cwd: &std::path::Path,
        mcp: &[v1::McpServer],
        additional: &[std::path::PathBuf],
        meta: Option<Value>,
    ) -> domain::SessionSetupParams {
        domain::SessionSetupParams {
            cwd: cwd.to_path_buf(),
            additional_directories: additional.to_vec(),
            mcp_servers: mcp.iter().map(|s| self.mcp_server_to_domain(s)).collect(),
            system_prompt: None,
            meta,
            extra: BTreeMap::new(),
        }
    }

    /// v1 `session/prompt` params → domain.
    #[must_use]
    pub fn prompt_to_domain(&self, p: &v1::PromptParams) -> domain::PromptParams {
        domain::PromptParams {
            session_id: p.session_id.clone().into(),
            prompt: p.prompt.iter().map(content_to_domain).collect(),
            idempotency_key: None,
            meta: p.meta.clone(),
            extra: BTreeMap::new(),
        }
    }

    /// Domain `session/prompt` params → v1 (idempotency key dropped).
    #[must_use]
    pub fn prompt_to_v1(&self, p: &domain::PromptParams) -> v1::PromptParams {
        v1::PromptParams {
            session_id: p.session_id.to_string(),
            prompt: p.prompt.iter().filter_map(content_to_v1).collect(),
            meta: p.meta.clone(),
        }
    }

    /// v1 prompt result → domain turn-end signal.
    #[must_use]
    pub fn prompt_result_to_domain(&self, r: &v1::PromptResult) -> domain::PromptFinished {
        domain::PromptFinished {
            stop_reason: Open::Known(stop_reason_to_domain(r.stop_reason)),
            usage: None,
            meta: r.meta.clone(),
        }
    }

    /// Domain turn-end → v1 prompt result.
    #[must_use]
    pub fn prompt_finished_to_v1(&self, r: &domain::PromptFinished) -> v1::PromptResult {
        v1::PromptResult {
            stop_reason: stop_reason_to_v1(&r.stop_reason),
            meta: r.meta.clone(),
        }
    }

    /// v1 client capabilities → domain.
    #[must_use]
    pub fn client_caps_to_domain(&self, c: &v1::ClientCapabilities) -> domain::ClientCapabilities {
        let mut extra = c.extra.clone();
        extra.insert(
            "fs".to_string(),
            json!({"readTextFile": c.fs.read_text_file, "writeTextFile": c.fs.write_text_file}),
        );
        extra.insert("terminal".to_string(), json!(c.terminal));
        domain::ClientCapabilities {
            auth: c.auth.as_ref().map(|a| domain::ClientAuthCaps {
                terminal: a.terminal.then_some(domain::EmptyObject),
                meta: a.meta.clone(),
                extra: BTreeMap::new(),
            }),
            elicitation: c.elicitation.as_ref().map(|e| domain::ElicitCaps {
                form: e.form.as_ref().map(|f| {
                    let _ = f;
                    domain::EmptyObject
                }),
                url: e.url.as_ref().map(|u| {
                    let _ = u;
                    domain::EmptyObject
                }),
                meta: e.meta.clone(),
                extra: BTreeMap::new(),
            }),
            session: None,
            meta: c.meta.clone(),
            extra,
        }
    }

    /// Domain client capabilities → v1 (`fs`/`terminal` recovered from
    /// `extra`, else `false`).
    #[must_use]
    pub fn client_caps_to_v1(&self, c: &domain::ClientCapabilities) -> v1::ClientCapabilities {
        let fs = c.extra.get("fs").cloned().unwrap_or_else(|| json!({}));
        v1::ClientCapabilities {
            fs: v1::FileSystemCapability {
                read_text_file: fs
                    .get("readTextFile")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                write_text_file: fs
                    .get("writeTextFile")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                meta: None,
            },
            terminal: c
                .extra
                .get("terminal")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            auth: c.auth.as_ref().map(|a| v1::ClientAuthCapabilities {
                terminal: a.terminal.is_some(),
                meta: a.meta.clone(),
            }),
            elicitation: c.elicitation.as_ref().map(|e| v1::ElicitationCapabilities {
                form: e
                    .form
                    .as_ref()
                    .map(|_| v1::ElicitationFormCapabilities { meta: None }),
                url: e
                    .url
                    .as_ref()
                    .map(|_| v1::ElicitationUrlCapabilities { meta: None }),
                meta: e.meta.clone(),
            }),
            meta: c.meta.clone(),
            extra: c
                .extra
                .iter()
                .filter(|(k, _)| *k != "fs" && *k != "terminal")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }

    /// v1 agent capabilities → domain.
    #[must_use]
    pub fn agent_caps_to_domain(&self, c: &v1::AgentCapabilities) -> domain::AgentCapabilities {
        let p = &c.prompt_capabilities;
        let m = &c.mcp_capabilities;
        let s = &c.session_capabilities;
        domain::AgentCapabilities {
            session: Some(domain::SessionCaps {
                prompt: Some(domain::PromptCaps {
                    image: p.image.then_some(domain::EmptyObject),
                    audio: p.audio.then_some(domain::EmptyObject),
                    embedded_context: p.embedded_context.then_some(domain::EmptyObject),
                    meta: None,
                    extra: BTreeMap::new(),
                }),
                mcp: Some(domain::McpCaps {
                    stdio: Some(domain::EmptyObject),
                    http: m.http.then_some(domain::EmptyObject),
                    acp: None,
                    meta: None,
                    extra: BTreeMap::new(),
                }),
                delete: s.delete.as_ref().map(|_| domain::EmptyObject),
                additional_directories: s
                    .additional_directories
                    .as_ref()
                    .map(|_| domain::EmptyObject),
                system_prompt: None,
                fork: s.fork.as_ref().map(|_| domain::EmptyObject),
                archive: None,
                options: None,
                status: None,
                title: None,
                meta: s.meta.clone(),
                extra: {
                    let mut extra = s.extra.clone();
                    if let Some(v) = s.list.as_ref() {
                        extra.insert("list".into(), json!(v));
                    }
                    if let Some(v) = s.resume.as_ref() {
                        extra.insert("resume".into(), json!(v));
                    }
                    if let Some(v) = s.close.as_ref() {
                        extra.insert("close".into(), json!(v));
                    }
                    if c.load_session {
                        extra.insert("loadSession".into(), json!(true));
                    }
                    extra
                },
            }),
            auth: c
                .auth
                .as_ref()
                .map(|a| json!({"logout": a.logout.is_some()})),
            providers: None,
            meta: c.meta.clone(),
            extra: c.extra.clone(),
        }
    }

    /// Domain agent capabilities → v1.
    #[must_use]
    pub fn agent_caps_to_v1(&self, c: &domain::AgentCapabilities) -> v1::AgentCapabilities {
        let s = c.session.as_ref();
        let p = s.and_then(|s| s.prompt.as_ref());
        let m = s.and_then(|s| s.mcp.as_ref());
        let truthy = |v: Option<&Value>| v.and_then(|v| v.as_object()).is_some();
        v1::AgentCapabilities {
            load_session: s.is_some_and(|s| {
                s.extra
                    .get("loadSession")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                    || s.fork.is_some()
                    || s.delete.is_some()
            }) || s.is_some(),
            prompt_capabilities: v1::PromptCapabilities {
                image: p.is_some_and(|p| p.image.is_some()),
                audio: p.is_some_and(|p| p.audio.is_some()),
                embedded_context: p.is_some_and(|p| p.embedded_context.is_some()),
            },
            mcp_capabilities: v1::McpCapabilities {
                http: m.is_some_and(|m| m.http.is_some()),
                sse: false,
            },
            session_capabilities: v1::SessionCapabilities {
                list: s
                    .and_then(|s| s.extra.get("list"))
                    .filter(|v| truthy(Some(v)))
                    .map(|_| v1::SessionListCapabilities::default()),
                delete: s
                    .and_then(|s| s.delete.as_ref())
                    .map(|_| v1::SessionDeleteCapabilities::default()),
                additional_directories: s
                    .and_then(|s| s.additional_directories.as_ref())
                    .map(|_| v1::SessionAdditionalDirectoriesCapabilities::default()),
                resume: s
                    .and_then(|s| s.extra.get("resume"))
                    .filter(|v| truthy(Some(v)))
                    .map(|_| v1::SessionResumeCapabilities::default()),
                fork: s
                    .and_then(|s| s.fork.as_ref())
                    .map(|_| v1::SessionForkCapabilities::default()),
                close: s
                    .and_then(|s| s.extra.get("close"))
                    .filter(|v| truthy(Some(v)))
                    .map(|_| v1::SessionCloseCapabilities::default()),
                meta: s.and_then(|s| s.meta.clone()),
                extra: s.map_or_else(BTreeMap::new, |s| {
                    s.extra
                        .iter()
                        .filter(|(k, _)| {
                            !matches!(k.as_str(), "list" | "resume" | "close" | "loadSession")
                        })
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect()
                }),
            },
            auth: c.auth.as_ref().map(|a| v1::AgentAuthCapabilities {
                logout: a
                    .get("logout")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                    .then(v1::LogoutCapabilities::default),
                meta: None,
            }),
            meta: c.meta.clone(),
            extra: c.extra.clone(),
        }
    }

    /// v1 initialize result → domain init result.
    #[must_use]
    pub fn initialize_result_to_domain(&self, r: &v1::InitializeResult) -> domain::InitResult {
        domain::InitResult {
            protocol_version: r.protocol_version,
            capabilities: self.agent_caps_to_domain(&r.agent_capabilities),
            info: r.agent_info.clone().map(|i| domain::Implementation {
                name: i.name,
                title: i.title,
                version: i.version,
                meta: None,
                extra: BTreeMap::new(),
            }),
            auth_methods: r
                .auth_methods
                .iter()
                .map(|m| self.auth_method_to_domain(m))
                .collect(),
            meta: r.meta.clone(),
        }
    }

    /// Domain init result → v1 initialize result.
    #[must_use]
    pub fn initialize_result_to_v1(&self, r: &domain::InitResult) -> v1::InitializeResult {
        v1::InitializeResult {
            protocol_version: 1,
            agent_capabilities: self.agent_caps_to_v1(&r.capabilities),
            agent_info: r.info.clone().map(|i| v1::Implementation {
                name: i.name,
                title: i.title,
                version: i.version,
            }),
            auth_methods: r
                .auth_methods
                .iter()
                .map(|m| self.auth_method_to_v1(m))
                .collect(),
            meta: r.meta.clone(),
        }
    }

    /// v1 `session/cancel` params → domain.
    #[must_use]
    pub fn cancel_to_domain(&self, p: &v1::SessionCancelParams) -> domain::CancelParams {
        domain::CancelParams {
            session_id: p.session_id.clone().into(),
            meta: p.meta.clone(),
            extra: BTreeMap::new(),
        }
    }

    /// Domain elicit request → v1 `elicitation/create` (scope synthesized
    /// as a session scope; unknown modes preserved via value round-trip).
    ///
    /// # Errors
    ///
    /// Returns the raw mode object when a `Custom`/`Unknown` elicitation
    /// mode cannot be deserialized as any v1 `elicitation/create` mode
    /// even with the synthesized scope merged in.
    pub fn elicit_to_v1(
        &self,
        e: &domain::ElicitCreate,
        session_id: &domain::SessionId,
    ) -> Result<v1::ElicitationCreateParams, Value> {
        let scope = v1::ElicitationScope::Session(v1::ElicitationSessionScope {
            session_id: session_id.to_string(),
            tool_call_id: None,
        });
        let mode = match &e.mode {
            Open::Known(domain::ElicitMode::Form(f)) => {
                v1::ElicitationMode::Form(v1::ElicitationFormMode {
                    scope,
                    requested_schema: schema_to_v1(&f.requested_schema),
                })
            }
            Open::Known(domain::ElicitMode::Url(u)) => {
                v1::ElicitationMode::Url(v1::ElicitationUrlMode {
                    scope,
                    elicitation_id: u.elicitation_id.to_string(),
                    url: u.url.clone(),
                })
            }
            Open::Custom(_, raw) | Open::Unknown(_, raw) => {
                let mut raw = raw.clone();
                if let Value::Object(map) = &mut raw {
                    map.insert(
                        "scope".to_string(),
                        serde_json::to_value(&scope).unwrap_or(Value::Null),
                    );
                    if let Ok(mode) =
                        serde_json::from_value::<v1::ElicitationMode>(Value::Object(map.clone()))
                    {
                        return Ok(v1::ElicitationCreateParams {
                            mode,
                            message: e.message.clone(),
                            meta: e.meta.clone(),
                        });
                    }
                }
                return Err(raw);
            }
        };
        Ok(v1::ElicitationCreateParams {
            mode,
            message: e.message.clone(),
            meta: e.meta.clone(),
        })
    }

    /// v1 `elicitation/create` → domain (scope dropped).
    #[must_use]
    pub fn elicit_to_domain(&self, p: &v1::ElicitationCreateParams) -> domain::ElicitCreate {
        let mode = match &p.mode {
            v1::ElicitationMode::Form(f) => {
                Open::Known(domain::ElicitMode::Form(domain::FormMode {
                    requested_schema: schema_to_domain(&f.requested_schema),
                    meta: None,
                    extra: BTreeMap::new(),
                }))
            }
            v1::ElicitationMode::Url(u) => Open::Known(domain::ElicitMode::Url(domain::UrlMode {
                elicitation_id: u.elicitation_id.clone().into(),
                url: u.url.clone(),
                meta: None,
                extra: BTreeMap::new(),
            })),
            v1::ElicitationMode::Other(o) => {
                let raw = serde_json::to_value(o).unwrap_or(Value::Null);
                match domain::ExtIdent::try_new(o.mode.clone()) {
                    Ok(id) => Open::Custom(id, raw),
                    Err(_) => Open::Unknown(o.mode.clone(), raw),
                }
            }
        };
        domain::ElicitCreate {
            mode,
            message: p.message.clone(),
            meta: p.meta.clone(),
            extra: BTreeMap::new(),
        }
    }

    /// v1 elicitation result → domain.
    #[must_use]
    pub fn elicit_result_to_domain(&self, r: &v1::ElicitationCreateResult) -> domain::ElicitResult {
        domain::ElicitResult {
            action: action_to_domain(&r.action),
            meta: r.meta.clone(),
            extra: BTreeMap::new(),
        }
    }

    /// Domain elicit result → v1.
    #[must_use]
    pub fn elicit_result_to_v1(&self, r: &domain::ElicitResult) -> v1::ElicitationCreateResult {
        v1::ElicitationCreateResult {
            action: action_to_v1(&r.action),
            meta: r.meta.clone(),
        }
    }
}

// ===== shared helpers =====

fn chunk_to_v1(c: &domain::ContentChunk) -> v1::ContentChunk {
    v1::ContentChunk {
        content: content_to_v1(&c.content).unwrap_or(v1::ContentBlock::Text(v1::TextContent {
            text: String::new(),
            annotations: None,
            meta: None,
        })),
        message_id: Some(c.message_id.to_string()),
        meta: c.meta.clone(),
    }
}

fn content_to_v1(b: &domain::ContentBlock) -> Option<v1::ContentBlock> {
    let v = serde_json::to_value(b).ok()?;
    // v2 embedded resources nest under "resource"; v1 flattens them.
    if let domain::ContentBlock::Resource(r) = b {
        let mut res = serde_json::to_value(&r.resource).ok()?;
        if let Value::Object(m) = &mut res
            && let Some(meta) = &r.meta
        {
            m.insert("_meta".into(), meta.clone());
        }
        let _ = &v;
        return serde_json::from_value(res).ok();
    }
    serde_json::from_value(v).ok()
}

fn content_to_domain(b: &v1::ContentBlock) -> domain::ContentBlock {
    let v = serde_json::to_value(b).unwrap_or(Value::Null);
    if let v1::ContentBlock::Resource(_) = b {
        let meta = v.get("_meta").cloned();
        return domain::ContentBlock::Resource(domain::EmbeddedResource {
            resource: serde_json::from_value(v).unwrap_or(domain::ResourceContents::Text(
                domain::TextResourceContents {
                    uri: String::new(),
                    text: String::new(),
                    mime_type: None,
                    meta: None,
                },
            )),
            annotations: None,
            meta,
            extra: BTreeMap::new(),
        });
    }
    serde_json::from_value(v)
        .unwrap_or_else(|_| domain::ContentBlock::Unknown("malformed".to_string(), Value::Null))
}

fn location_to_v1(l: &domain::Location) -> v1::ToolCallLocation {
    v1::ToolCallLocation {
        path: l.path.clone(),
        line: l.line,
    }
}

const fn tool_kind_to_v1(k: &Open<domain::ToolKind>) -> v1::ToolKind {
    use domain::ToolKind as K;
    match k {
        Open::Known(k) => match k {
            K::Read => v1::ToolKind::Read,
            K::Edit => v1::ToolKind::Edit,
            K::Delete => v1::ToolKind::Delete,
            K::Move => v1::ToolKind::Move,
            K::Search => v1::ToolKind::Search,
            K::Execute => v1::ToolKind::Execute,
            K::Think => v1::ToolKind::Think,
            K::Fetch => v1::ToolKind::Fetch,
            K::SwitchMode => v1::ToolKind::SwitchMode,
            K::Other => v1::ToolKind::Other,
        },
        Open::Custom(..) | Open::Unknown(..) => v1::ToolKind::Other,
    }
}

const fn tool_kind_to_domain(k: v1::ToolKind) -> Open<domain::ToolKind> {
    use domain::ToolKind as K;
    Open::Known(match k {
        v1::ToolKind::Read => K::Read,
        v1::ToolKind::Edit => K::Edit,
        v1::ToolKind::Delete => K::Delete,
        v1::ToolKind::Move => K::Move,
        v1::ToolKind::Search => K::Search,
        v1::ToolKind::Execute => K::Execute,
        v1::ToolKind::Think => K::Think,
        v1::ToolKind::Fetch => K::Fetch,
        v1::ToolKind::SwitchMode => K::SwitchMode,
        v1::ToolKind::Other => K::Other,
    })
}

const fn tool_status_to_v1(s: &Open<domain::ToolCallStatus>) -> Option<v1::ToolCallStatus> {
    use domain::ToolCallStatus as S;
    Some(match s {
        Open::Known(s) => match s {
            S::Pending => v1::ToolCallStatus::Pending,
            // v1 has no `paused`; in_progress is the closest live state.
            S::InProgress | S::Paused => v1::ToolCallStatus::InProgress,
            S::Completed => v1::ToolCallStatus::Completed,
            // v1 has no `cancelled`; failed is the closest terminal state.
            S::Failed | S::Cancelled => v1::ToolCallStatus::Failed,
        },
        Open::Custom(..) | Open::Unknown(..) => return None,
    })
}

const fn tool_status_to_domain(s: v1::ToolCallStatus) -> Open<domain::ToolCallStatus> {
    use domain::ToolCallStatus as S;
    Open::Known(match s {
        v1::ToolCallStatus::Pending => S::Pending,
        v1::ToolCallStatus::InProgress => S::InProgress,
        v1::ToolCallStatus::Completed => S::Completed,
        v1::ToolCallStatus::Failed => S::Failed,
    })
}

fn tool_content_to_v1(c: &domain::ToolCallContent) -> Option<v1::ToolCallContent> {
    use domain::ToolCallContent as C;
    Some(match c {
        C::Content(d) => v1::ToolCallContent::Content {
            content: content_to_v1(&d.content)?,
        },
        C::Terminal(t) => v1::ToolCallContent::Terminal {
            terminal_id: t.terminal_id.to_string(),
        },
        C::Diff(d) => {
            // v2 structured diff → v1 legacy text diff (best effort:
            // path from the first change, texts from the patch).
            let path = d
                .changes
                .iter()
                .find_map(|c| {
                    if let Open::Known(op) = &c.operation {
                        match op {
                            domain::DiffOperation::Add(p)
                            | domain::DiffOperation::Delete(p)
                            | domain::DiffOperation::Modify(p) => Some(p.path.clone()),
                            domain::DiffOperation::Move(p) | domain::DiffOperation::Copy(p) => {
                                Some(p.path.clone())
                            }
                        }
                    } else {
                        None
                    }
                })
                .unwrap_or_default();
            v1::ToolCallContent::Diff(v1::Diff {
                path,
                old_text: d
                    .extra
                    .get("oldText")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                new_text: d
                    .patch
                    .as_ref()
                    .map(|p| p.text.clone())
                    .or_else(|| {
                        d.extra
                            .get("newText")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .unwrap_or_default(),
            })
        }
        C::Custom(_, raw) | C::Unknown(_, raw) => {
            serde_json::from_value::<v1::ToolCallContent>(raw.clone()).ok()?
        }
    })
}

fn tool_content_to_domain(c: &v1::ToolCallContent) -> domain::ToolCallContent {
    use domain::ToolCallContent as C;
    match c {
        v1::ToolCallContent::Content { content } => C::Content(domain::Content {
            content: content_to_domain(content),
            meta: None,
            extra: BTreeMap::new(),
        }),
        v1::ToolCallContent::Terminal { terminal_id } => C::Terminal(domain::TerminalRef {
            terminal_id: terminal_id.clone().into(),
            meta: None,
            extra: BTreeMap::new(),
        }),
        v1::ToolCallContent::Diff(d) => {
            // v1 legacy diff {path, oldText, newText} → v2 structured
            // diff; the text pair rides in `extra` for round-tripping.
            let mut extra = BTreeMap::new();
            if let Some(old) = &d.old_text {
                extra.insert("oldText".into(), json!(old));
            }
            extra.insert("newText".into(), json!(d.new_text));
            C::Diff(domain::Diff {
                changes: vec![domain::DiffChange {
                    operation: Open::Known(domain::DiffOperation::Modify(domain::DiffPathChange {
                        path: d.path.clone(),
                    })),
                    file_type: None,
                    mime_type: None,
                    meta: None,
                    extra: BTreeMap::new(),
                }],
                patch: None,
                meta: None,
                extra,
            })
        }
    }
}

fn plan_entry_to_v1(e: &domain::PlanEntry) -> v1::PlanEntry {
    v1::PlanEntry {
        content: e.content.clone(),
        status: match &e.status {
            Some(Open::Known(domain::PlanEntryStatus::InProgress)) => {
                v1::PlanEntryStatus::InProgress
            }
            // v1 has no `cancelled`; completed is the closest terminal state.
            Some(Open::Known(
                domain::PlanEntryStatus::Completed | domain::PlanEntryStatus::Cancelled,
            ))
            | None => v1::PlanEntryStatus::Completed,
            // Pending and unrecognized values stay pending.
            _ => v1::PlanEntryStatus::Pending,
        },
        priority: match &e.priority {
            Some(Open::Known(domain::PlanEntryPriority::High)) => v1::PlanEntryPriority::High,
            Some(Open::Known(domain::PlanEntryPriority::Low)) => v1::PlanEntryPriority::Low,
            // Medium and unrecognized values stay medium.
            _ => v1::PlanEntryPriority::Medium,
        },
        meta: e.meta.clone(),
    }
}

fn plan_entry_to_domain(e: &v1::PlanEntry) -> domain::PlanEntry {
    domain::PlanEntry {
        content: e.content.clone(),
        status: Some(Open::Known(match e.status {
            v1::PlanEntryStatus::Pending => domain::PlanEntryStatus::Pending,
            v1::PlanEntryStatus::InProgress => domain::PlanEntryStatus::InProgress,
            v1::PlanEntryStatus::Completed => domain::PlanEntryStatus::Completed,
        })),
        priority: Some(Open::Known(match e.priority {
            v1::PlanEntryPriority::High => domain::PlanEntryPriority::High,
            v1::PlanEntryPriority::Medium => domain::PlanEntryPriority::Medium,
            v1::PlanEntryPriority::Low => domain::PlanEntryPriority::Low,
        })),
        meta: e.meta.clone(),
        extra: BTreeMap::new(),
    }
}

fn cost_to_v1(c: &domain::Cost) -> v1::Cost {
    v1::Cost {
        amount: c.amount.as_f64().unwrap_or(0.0),
        currency: c.currency.clone(),
        meta: c.meta.clone(),
    }
}

fn cost_to_domain(c: &v1::Cost) -> domain::Cost {
    domain::Cost {
        amount: json!(c.amount),
        currency: c.currency.clone(),
        meta: c.meta.clone(),
        extra: BTreeMap::new(),
    }
}

fn select_options_to_domain(o: &v1::ConfigSelectOptions) -> domain::SelectOptions {
    match o {
        v1::ConfigSelectOptions::Flat(opts) => domain::SelectOptions::Ungrouped(
            opts.iter()
                .map(|o| domain::SelectOption {
                    value: o.value.clone().into(),
                    name: o.name.clone(),
                    description: o.description.clone(),
                    meta: o.meta.clone(),
                })
                .collect(),
        ),
        v1::ConfigSelectOptions::Grouped(groups) => domain::SelectOptions::Grouped(
            groups
                .iter()
                .map(|g| domain::SelectGroup {
                    group_id: g.group.clone().into(),
                    name: g.name.clone(),
                    options: g
                        .options
                        .iter()
                        .map(|o| domain::SelectOption {
                            value: o.value.clone().into(),
                            name: o.name.clone(),
                            description: o.description.clone(),
                            meta: o.meta.clone(),
                        })
                        .collect(),
                    meta: g.meta.clone(),
                })
                .collect(),
        ),
    }
}

fn select_options_to_v1(o: &domain::SelectOptions) -> v1::ConfigSelectOptions {
    match o {
        domain::SelectOptions::Ungrouped(opts) => v1::ConfigSelectOptions::Flat(
            opts.iter()
                .map(|o| v1::ConfigSelectOption {
                    value: o.value.to_string(),
                    name: o.name.clone(),
                    description: o.description.clone(),
                    meta: o.meta.clone(),
                })
                .collect(),
        ),
        domain::SelectOptions::Grouped(groups) => v1::ConfigSelectOptions::Grouped(
            groups
                .iter()
                .map(|g| v1::ConfigSelectGroup {
                    group: g.group_id.to_string(),
                    name: g.name.clone(),
                    options: g
                        .options
                        .iter()
                        .map(|o| v1::ConfigSelectOption {
                            value: o.value.to_string(),
                            name: o.name.clone(),
                            description: o.description.clone(),
                            meta: o.meta.clone(),
                        })
                        .collect(),
                    meta: g.meta.clone(),
                })
                .collect(),
        ),
    }
}

fn config_category(s: String) -> Open<domain::ConfigCategory> {
    use domain::ConfigCategory as C;
    match s.as_str() {
        "mode" => Open::Known(C::Mode),
        "model" => Open::Known(C::Model),
        "model_config" => Open::Known(C::ModelConfig),
        "thought_level" => Open::Known(C::ThoughtLevel),
        _ => match domain::ExtIdent::try_new(s.clone()) {
            Ok(id) => Open::Custom(id, Value::String(s)),
            Err(_) => Open::Unknown(s.clone(), Value::String(s)),
        },
    }
}

const fn permission_kind_to_v1(k: &Open<domain::PermissionOptionKind>) -> v1::PermissionOptionKind {
    use domain::PermissionOptionKind as K;
    match k {
        Open::Known(K::AllowOnce) => v1::PermissionOptionKind::AllowOnce,
        Open::Known(K::AllowAlways) => v1::PermissionOptionKind::AllowAlways,
        Open::Known(K::RejectAlways) => v1::PermissionOptionKind::RejectAlways,
        // Unknown kinds are not safe to grant; deny.
        _ => v1::PermissionOptionKind::RejectOnce,
    }
}

const fn permission_kind_to_domain(k: v1::PermissionOptionKind) -> domain::PermissionOptionKind {
    use domain::PermissionOptionKind as K;
    match k {
        v1::PermissionOptionKind::AllowOnce => K::AllowOnce,
        v1::PermissionOptionKind::AllowAlways => K::AllowAlways,
        v1::PermissionOptionKind::RejectOnce => K::RejectOnce,
        v1::PermissionOptionKind::RejectAlways => K::RejectAlways,
    }
}

const fn stop_reason_to_domain(r: v1::StopReason) -> domain::StopReason {
    use domain::StopReason as S;
    match r {
        v1::StopReason::EndTurn => S::EndTurn,
        v1::StopReason::Cancelled => S::Cancelled,
        v1::StopReason::Error => S::Error,
        v1::StopReason::MaxTokens => S::MaxTokens,
        v1::StopReason::MaxTurnRequests => S::MaxTurnRequests,
        v1::StopReason::Refusal => S::Refusal,
    }
}

const fn stop_reason_to_v1(r: &Open<domain::StopReason>) -> v1::StopReason {
    use domain::StopReason as S;
    match r {
        Open::Known(S::Cancelled) => v1::StopReason::Cancelled,
        Open::Known(S::Error) => v1::StopReason::Error,
        Open::Known(S::MaxTokens) => v1::StopReason::MaxTokens,
        Open::Known(S::MaxTurnRequests) => v1::StopReason::MaxTurnRequests,
        Open::Known(S::Refusal) => v1::StopReason::Refusal,
        // EndTurn and unrecognized reasons report a normal end.
        _ => v1::StopReason::EndTurn,
    }
}

fn command_to_v1(c: &domain::AvailableCommand) -> Option<v1::AvailableCommand> {
    match c {
        domain::AvailableCommand::Command(d) => Some(v1::AvailableCommand {
            name: d.name.clone(),
            description: d.description.clone(),
            input: d.input.as_ref().and_then(|i| match i {
                Open::Known(domain::CommandInput::Text(t)) => Some(v1::AvailableCommandInput {
                    hint: t.hint.clone().unwrap_or_default(),
                    meta: t.meta.clone(),
                }),
                _ => None,
            }),
            meta: d.meta.clone(),
        }),
        domain::AvailableCommand::Custom(_, raw) | domain::AvailableCommand::Unknown(_, raw) => {
            serde_json::from_value(raw.clone()).ok()
        }
    }
}

fn command_to_domain(c: &v1::AvailableCommand) -> domain::AvailableCommand {
    domain::AvailableCommand::Command(domain::Command {
        name: c.name.clone(),
        description: c.description.clone(),
        input: c.input.as_ref().map(|i| {
            Open::Known(domain::CommandInput::Text(domain::TextInput {
                hint: Some(i.hint.clone()),
                meta: i.meta.clone(),
            }))
        }),
        meta: c.meta.clone(),
        extra: BTreeMap::new(),
    })
}

fn v1_tool_call_from_update(u: &domain::ToolCallUpdate) -> v1::ToolCall {
    v1::ToolCall {
        tool_call_id: u.tool_call_id.to_string(),
        title: u.title.clone().into_option().unwrap_or_default(),
        kind: u.kind.as_option().map(tool_kind_to_v1),
        status: u.status.as_option().and_then(tool_status_to_v1),
        content: u
            .content
            .clone()
            .into_option()
            .map_or_default(|cs| cs.iter().filter_map(tool_content_to_v1).collect()),
        locations: u
            .locations
            .clone()
            .into_option()
            .map_or_default(|ls| ls.iter().map(location_to_v1).collect()),
        raw_input: u.raw_input.clone().into_option(),
        raw_output: u.raw_output.clone().into_option(),
        meta: u.meta.clone().into_option(),
    }
}

fn tool_call_update_from_snapshot(c: &v1::ToolCall) -> domain::ToolCallUpdate {
    domain::ToolCallUpdate {
        tool_call_id: c.tool_call_id.clone().into(),
        name: Patch::Unchanged,
        title: Patch::Set(c.title.clone()),
        kind: c
            .kind
            .map_or(Patch::Unchanged, |k| Patch::Set(tool_kind_to_domain(k))),
        status: c
            .status
            .map_or(Patch::Unchanged, |s| Patch::Set(tool_status_to_domain(s))),
        content: Patch::Set(c.content.iter().map(tool_content_to_domain).collect()),
        locations: Patch::Set(
            c.locations
                .iter()
                .map(|l| domain::Location {
                    path: l.path.clone(),
                    line: l.line,
                    meta: None,
                })
                .collect(),
        ),
        raw_input: c.raw_input.clone().map_or(Patch::Unchanged, Patch::Set),
        raw_output: c.raw_output.clone().map_or(Patch::Unchanged, Patch::Set),
        capabilities: Patch::Unchanged,
        meta: c.meta.clone().map_or(Patch::Unchanged, Patch::Set),
        extra: BTreeMap::new(),
    }
}

fn tool_call_to_domain(c: &v1::ToolCall) -> domain::ToolCallUpdate {
    tool_call_update_from_snapshot(c)
}

fn tool_call_patch_to_domain(u: v1::ToolCallUpdate) -> domain::ToolCallUpdate {
    domain::ToolCallUpdate {
        tool_call_id: u.tool_call_id.into(),
        name: Patch::Unchanged,
        title: u.title.map_or(Patch::Unchanged, Patch::Set),
        kind: u
            .kind
            .map_or(Patch::Unchanged, |k| Patch::Set(tool_kind_to_domain(k))),
        status: u
            .status
            .map_or(Patch::Unchanged, |s| Patch::Set(tool_status_to_domain(s))),
        content: u.content.map_or(Patch::Unchanged, |cs| {
            Patch::Set(cs.iter().map(tool_content_to_domain).collect())
        }),
        locations: u.locations.map_or(Patch::Unchanged, |ls| {
            Patch::Set(
                ls.iter()
                    .map(|l| domain::Location {
                        path: l.path.clone(),
                        line: l.line,
                        meta: None,
                    })
                    .collect(),
            )
        }),
        raw_input: u.raw_input.map_or(Patch::Unchanged, Patch::Set),
        raw_output: u.raw_output.map_or(Patch::Unchanged, Patch::Set),
        capabilities: Patch::Unchanged,
        meta: u.meta.map_or(Patch::Unchanged, Patch::Set),
        extra: BTreeMap::new(),
    }
}

fn schema_to_v1(s: &domain::ElicitationSchema) -> v1::ElicitationSchema {
    serde_json::to_value(s)
        .ok()
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default()
}

fn schema_to_domain(s: &v1::ElicitationSchema) -> domain::ElicitationSchema {
    serde_json::to_value(s)
        .ok()
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_else(|| domain::ElicitationSchema {
            schema_type: "object".to_string(),
            title: None,
            properties: BTreeMap::new(),
            required: Vec::new(),
            description: None,
            meta: None,
            extra: BTreeMap::new(),
        })
}

fn action_to_domain(a: &v1::ElicitationAction) -> Open<domain::ElicitAction> {
    match a {
        v1::ElicitationAction::Accept(d) => {
            Open::Known(domain::ElicitAction::Accept(domain::AcceptAction {
                content: d.content.as_ref().map(|m| {
                    m.iter()
                        .map(|(k, v)| (k.clone(), content_value_to_domain(v)))
                        .collect()
                }),
                meta: None,
            }))
        }
        v1::ElicitationAction::Decline => {
            Open::Known(domain::ElicitAction::Decline(domain::DeclineAction {
                meta: None,
            }))
        }
        v1::ElicitationAction::Cancel => {
            Open::Known(domain::ElicitAction::Cancel(domain::CancelAction {
                meta: None,
            }))
        }
        v1::ElicitationAction::Other(o) => {
            let raw = serde_json::to_value(o).unwrap_or(Value::Null);
            match domain::ExtIdent::try_new(o.action.clone()) {
                Ok(id) => Open::Custom(id, raw),
                Err(_) => Open::Unknown(o.action.clone(), raw),
            }
        }
    }
}

fn action_to_v1(a: &Open<domain::ElicitAction>) -> v1::ElicitationAction {
    match a {
        Open::Known(domain::ElicitAction::Accept(d)) => {
            v1::ElicitationAction::Accept(v1::ElicitationAcceptAction {
                content: d.content.as_ref().map(|m| {
                    m.iter()
                        .map(|(k, v)| (k.clone(), content_value_to_v1(v)))
                        .collect()
                }),
            })
        }
        Open::Known(domain::ElicitAction::Decline(_)) => v1::ElicitationAction::Decline,
        Open::Known(domain::ElicitAction::Cancel(_)) => v1::ElicitationAction::Cancel,
        Open::Custom(_, raw) | Open::Unknown(_, raw) => {
            serde_json::from_value(raw.clone()).unwrap_or(v1::ElicitationAction::Cancel)
        }
    }
}

fn content_value_to_domain(v: &v1::ElicitationContentValue) -> domain::ElicitContentValue {
    match v {
        v1::ElicitationContentValue::String(s) => domain::ElicitContentValue::String(s.clone()),
        v1::ElicitationContentValue::Integer(i) => domain::ElicitContentValue::Integer(*i),
        v1::ElicitationContentValue::Number(n) => domain::ElicitContentValue::Number(*n),
        v1::ElicitationContentValue::Boolean(b) => domain::ElicitContentValue::Boolean(*b),
        v1::ElicitationContentValue::StringArray(a) => domain::ElicitContentValue::Array(a.clone()),
    }
}

fn content_value_to_v1(v: &domain::ElicitContentValue) -> v1::ElicitationContentValue {
    match v {
        domain::ElicitContentValue::String(s) => v1::ElicitationContentValue::String(s.clone()),
        domain::ElicitContentValue::Integer(i) => v1::ElicitationContentValue::Integer(*i),
        domain::ElicitContentValue::Number(n) => v1::ElicitationContentValue::Number(*n),
        domain::ElicitContentValue::Boolean(b) => v1::ElicitationContentValue::Boolean(*b),
        domain::ElicitContentValue::Array(a) => v1::ElicitationContentValue::StringArray(a.clone()),
    }
}

// ===== free-standing negotiation helpers =====

/// Build the dual-shape `initialize` request params a client sends to an
/// unknown-version agent: both `capabilities`/`info` (v2) and
/// `clientCapabilities`/`clientInfo` (v1) in one object.
#[must_use]
pub fn initialize_params_dual(
    capabilities: &domain::ClientCapabilities,
    info: &domain::Implementation,
    meta: Option<Value>,
) -> Value {
    let codec = V1Codec::new();
    let mut obj = serde_json::Map::new();
    obj.insert("protocolVersion".into(), json!(2));
    obj.insert(
        "capabilities".into(),
        serde_json::to_value(capabilities).unwrap_or_else(|_| json!({})),
    );
    obj.insert(
        "info".into(),
        serde_json::to_value(info).unwrap_or_else(|_| json!({})),
    );
    obj.insert(
        "clientCapabilities".into(),
        serde_json::to_value(codec.client_caps_to_v1(capabilities)).unwrap_or_else(|_| json!({})),
    );
    obj.insert(
        "clientInfo".into(),
        serde_json::to_value(info).unwrap_or_else(|_| json!({})),
    );
    if let Some(meta) = meta {
        obj.insert("_meta".into(), meta);
    }
    Value::Object(obj)
}

/// Detect which protocol version an `initialize` *response* used:
/// `capabilities` + `info` means v2, `agentCapabilities` means v1.
#[must_use]
pub fn detect_initialize(value: &Value) -> Protocol {
    if value.get("capabilities").is_some() {
        Protocol::V2
    } else if value.get("agentCapabilities").is_some() {
        Protocol::V1
    } else {
        Protocol::V2
    }
}

/// Parse an `initialize` response into the domain [`domain::InitResult`],
/// dispatching on [`detect_initialize`].
///
/// # Errors
///
/// Returns an error when `value` does not deserialize as the detected
/// protocol version's `initialize` result shape.
pub fn initialize_result_detect(value: &Value) -> Result<domain::InitResult, serde_json::Error> {
    match detect_initialize(value) {
        Protocol::V2 => {
            let r: super::v2::InitializeResponse = serde_json::from_value(value.clone())?;
            Ok(domain::InitResult {
                protocol_version: r.protocol_version,
                capabilities: r.capabilities,
                info: r.info,
                auth_methods: r.auth_methods,
                meta: r.meta,
            })
        }
        Protocol::V1 => {
            let r: v1::InitializeResult = serde_json::from_value(value.clone())?;
            Ok(V1Codec::new().initialize_result_to_domain(&r))
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn domain_update(raw: Value) -> domain::SessionUpdate {
        serde_json::from_value(raw).expect("domain update")
    }

    #[test]
    fn first_tool_call_becomes_tool_call_then_update() {
        let mut codec = V1Codec::new();
        let raw = json!({"sessionUpdate": "tool_call_update", "toolCallId": "t1",
            "title": "Run", "status": "in_progress"});
        let u = domain_update(raw);
        let out = codec.update_to_v1(&u);
        assert!(matches!(out[0], v1::SessionUpdate::ToolCall(_)));

        let raw =
            json!({"sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed"});
        let u = domain_update(raw);
        let out = codec.update_to_v1(&u);
        assert!(matches!(out[0], v1::SessionUpdate::ToolCallUpdate(_)));
    }

    #[test]
    fn config_option_mode_emits_current_mode_update() {
        let mut codec = V1Codec::new();
        let raw = json!({"sessionUpdate": "config_option_update", "configOptions": [{
            "configId": "mode", "name": "Mode", "category": "mode",
            "type": "select", "currentValue": "code",
            "options": [{"value": "code", "name": "Code"}]
        }]});
        let u = domain_update(raw);
        let out = codec.update_to_v1(&u);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], v1::SessionUpdate::ConfigOptionUpdate(_)));
        match &out[1] {
            v1::SessionUpdate::CurrentModeUpdate(m) => assert_eq!(m.current_mode_id, "code"),
            other => panic!("expected current_mode_update, got {other:?}"),
        }
    }

    #[test]
    fn load_maps_to_replay_start() {
        let codec = V1Codec::new();
        let p = v1::SessionLoadParams {
            session_id: "s1".into(),
            cwd: "/tmp".into(),
            mcp_servers: vec![],
            additional_directories: vec![],
            meta: None,
        };
        let d = codec.load_to_domain(&p);
        assert!(matches!(
            d.replay_from,
            Some(Open::Known(domain::ReplayCursor::Start(_)))
        ));
        let v1::SessionResumeParams { .. } = v1::SessionResumeParams {
            session_id: "s1".into(),
            cwd: "/tmp".into(),
            mcp_servers: vec![],
            additional_directories: vec![],
            meta: None,
        };
        let back = codec.resume_to_v1(&d);
        assert!(matches!(back, V1Resume::Load(_)));

        let d2 = codec.resume_to_domain(&v1::SessionResumeParams {
            session_id: "s1".into(),
            cwd: "/tmp".into(),
            mcp_servers: vec![],
            additional_directories: vec![],
            meta: None,
        });
        assert!(d2.replay_from.is_none());
        assert!(matches!(codec.resume_to_v1(&d2), V1Resume::Resume(_)));
    }

    #[test]
    fn auth_method_id_rename() {
        let codec = V1Codec::new();
        let m: domain::AuthMethod = serde_json::from_value(json!({
            "methodId": "gh", "name": "GitHub", "type": "terminal",
            "args": ["gh", "auth"], "env": [{"name": "T", "value": "v"}]
        }))
        .expect("de");
        let v = codec.auth_method_to_v1(&m);
        let v1::AuthMethod::Terminal(t) = &v else {
            panic!("expected terminal")
        };
        assert_eq!(t.id, "gh");
        assert_eq!(t.env.get("T").map(String::as_str), Some("v"));

        let back = codec.auth_method_to_domain(&v);
        assert_eq!(back.method_id.as_str(), "gh");
        assert!(matches!(
            back.kind,
            Open::Known(domain::AuthKind::Terminal(_))
        ));
    }

    #[test]
    fn set_config_mode_routes_to_set_mode() {
        let mut codec = V1Codec::new();
        let update = domain_update(
            json!({"sessionUpdate": "config_option_update", "configOptions": [{
                "configId": "mode", "name": "Mode", "category": "mode",
                "type": "select", "currentValue": "code",
                "options": [{"value": "code", "name": "Code"}]
            }]}),
        );
        let _ = codec.update_to_v1(&update); // populates the cache
        let p = domain::SetConfigParams {
            session_id: "s".into(),
            config_id: "mode".into(),
            value: domain::ConfigValue::Id(domain::ConfigValueById {
                value: "plan".into(),
            }),
            meta: None,
            extra: BTreeMap::new(),
        };
        match codec.set_config_to_v1(&p) {
            V1SetConfig::SetMode(m) => assert_eq!(m.mode_id, "plan"),
            other => panic!("expected SetMode, got {other:?}"),
        }
        let back = codec.set_mode_to_domain(&v1::SessionSetModeParams {
            session_id: "s".into(),
            mode_id: "plan".into(),
            meta: None,
        });
        assert_eq!(back.config_id.as_str(), "mode");
    }

    #[test]
    fn v1_update_upgrades_tool_call_and_chunks() {
        let mut codec = V1Codec::new();
        let raw = json!({"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "Run",
            "kind": "execute", "status": "in_progress", "content": [{"type": "content", "content": {"type": "text", "text": "hi"}}]});
        let u = codec.update_from_v1(serde_json::from_value(raw).expect("v1"));
        let domain::SessionUpdate::ToolCallUpdate(t) = &u else {
            panic!("expected tool_call_update")
        };
        assert!(matches!(
            t.status,
            Patch::Set(Open::Known(domain::ToolCallStatus::InProgress))
        ));
        assert!(matches!(
            t.kind,
            Patch::Set(Open::Known(domain::ToolKind::Execute))
        ));

        let raw = json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "hi"}});
        let u = codec.update_from_v1(serde_json::from_value(raw).expect("v1"));
        let domain::SessionUpdate::AgentMessageChunk(c) = &u else {
            panic!("expected chunk")
        };
        assert_ne!(c.message_id.as_str(), "");
        // stable across chunks of the same lane
        let u2 = codec.update_from_v1(serde_json::from_value(json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "more"}})).expect("v1"));
        let domain::SessionUpdate::AgentMessageChunk(c2) = &u2 else {
            panic!("expected chunk")
        };
        assert_eq!(c.message_id, c2.message_id);
    }

    #[test]
    fn initialize_detect() {
        let v2 = json!({"protocolVersion": 2, "capabilities": {}, "info": {"name": "a", "version": "1"}});
        assert_eq!(detect_initialize(&v2), Protocol::V2);
        let v1 = json!({"protocolVersion": 1, "agentCapabilities": {"loadSession": true, "promptCapabilities": {"image": false, "audio": false, "embeddedContext": false}, "mcpCapabilities": {"http": false, "sse": false}, "sessionCapabilities": {}}, "agentInfo": {"name": "a", "version": "1"}});
        assert_eq!(detect_initialize(&v1), Protocol::V1);
        let parsed = initialize_result_detect(&v1).expect("parse");
        assert_eq!(parsed.protocol_version, 1);
        assert!(parsed.capabilities.session.is_some());
    }

    #[test]
    fn prompt_result_maps_stop_reason() {
        let codec = V1Codec::new();
        let v1r = v1::PromptResult {
            stop_reason: v1::StopReason::EndTurn,
            meta: None,
        };
        let d = codec.prompt_result_to_domain(&v1r);
        assert_eq!(d.stop_reason, Open::Known(domain::StopReason::EndTurn));
        let back = codec.prompt_finished_to_v1(&d);
        assert_eq!(back.stop_reason, v1::StopReason::EndTurn);
    }

    #[test]
    fn mcp_stdio_type_stripped_and_sse_preserved() {
        let codec = V1Codec::new();
        let s: domain::McpServer = serde_json::from_value(
            json!({"type": "stdio", "name": "n", "command": "c", "args": [], "env": []}),
        )
        .expect("de");
        let v = codec.mcp_server_to_v1(&s).expect("mapped");
        let raw = serde_json::to_value(&v).expect("ser");
        assert!(raw.get("type").is_none());
        assert!(raw.get("args").is_some());
        assert!(raw.get("env").is_some());

        let v1sse = v1::McpServer::Sse(v1::McpServerSse {
            name: "x".into(),
            url: "u".into(),
            headers: vec![],
            meta: None,
        });
        let d = codec.mcp_server_to_domain(&v1sse);
        assert!(matches!(d, domain::McpServer::Unknown(ref t, _) if t == "sse"));
    }

    #[test]
    fn permission_subject_maps_to_tool_call() {
        let codec = V1Codec::new();
        let p: domain::PermissionRequest = serde_json::from_value(json!({
            "sessionId": "s", "title": "Run ls",
            "subject": {"type": "command", "command": "ls", "cwd": "/tmp"},
            "options": [{"optionId": "ok", "name": "Allow", "kind": "allow_once"}]
        }))
        .expect("de");
        let v = codec.permission_to_v1(&p);
        assert_eq!(v.tool_call.kind, Some(v1::ToolKind::Execute));
        assert_eq!(v.tool_call.title, "ls");

        let back = codec.permission_to_domain(&v);
        assert!(matches!(
            back.subject,
            Some(Open::Known(domain::PermissionSubject::ToolCall(_)))
        ));
    }
}
