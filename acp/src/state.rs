//! Client-side session entity reducer.
//!
//! [`SessionReducer`] folds the [`SessionUpdate`]s a session produces into
//! a materialized [`SessionState`] plus ergonomic [`SessionEvent`] deltas
//! a UI can render without knowing the patch mechanics. It is
//! protocol-version neutral: it consumes canonical [`crate::domain`]
//! updates, so v1 peers work through the codec
//! ([`crate::wire::V1Codec`]) exactly like v2 peers.
//!
//! Patch rules, applied uniformly: a chunk for an unknown entity id
//! creates the entity with defaults, then appends; a `Set` field
//! replaces, `Clear` empties, `Unchanged` keeps; `tool_call_update`'s
//! `content` replaces accumulated chunks while `tool_call_content_chunk`
//! appends.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::domain::{
    AvailableCommand, BackgroundItem, CompactionId, CompactionUpdate, ConfigKind, ConfigOption,
    ConfigOptionUpdate, ConfigValueId, ContentBlock, ForegroundState, Message, MessageId, Notice,
    Open, Patch, Plan, PlanId, PlanUpdate, SessionId, SessionInfoUpdate, SessionMessage,
    SessionUpdate, StateUpdate, StopReason, SubagentUpdate, TerminalId, TerminalOutput,
    TerminalOutputChunk, TerminalUpdate, ToolCall, ToolCallContentChunk, ToolCallId,
    ToolCallUpdate, UsageUpdate,
};

/// Which message stream a stored [`Message`] belongs to.
///
/// Message ids are only unique within a stream, so this is half of the
/// [`SessionState::messages`] key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MessageKind {
    /// `user_message`/`user_message_chunk`.
    User,
    /// `agent_message`/`agent_message_chunk`.
    Agent,
    /// `agent_thought`/`agent_thought_chunk` — agent reasoning.
    Thought,
    /// `session_message`/`session_message_chunk` — session-directed
    /// messages exchanged between sessions.
    SessionDirected,
}

/// Title/updated-at reported by `session_info_update`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionInfoState {
    /// Session title, when reported.
    pub title: Option<String>,
    /// RFC 3339 last-updated timestamp, when reported.
    pub updated_at: Option<String>,
}

/// A session's materialized state, reduced from its updates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionState {
    /// Messages keyed by `(kind, message_id)` — message ids are only
    /// unique within a [`MessageKind`]. Chunks append content
    /// (consecutive `text` blocks concatenate into one); whole-message
    /// updates replace or clear content.
    pub messages: BTreeMap<(MessageKind, MessageId), Message>,
    /// Tool calls by id, as materialized snapshots. A chunk or update
    /// for an unknown id creates the call with defaults first.
    pub tool_calls: BTreeMap<ToolCallId, ToolCall>,
    /// Plans by id; every `plan_update` replaces that plan's entries.
    pub plans: BTreeMap<PlanId, Plan>,
    /// Agent-owned terminals by id. Stored updates carry every field ever
    /// reported as [`Patch::Set`]; [`Patch::Unchanged`] means never
    /// reported (a `Clear` reads the same). `terminal_output_chunk`s
    /// append to `output.data`.
    pub terminals: BTreeMap<TerminalId, TerminalUpdate>,
    /// Session config options (whole-list replace).
    pub config_options: Vec<ConfigOption>,
    /// Session available commands (whole-list replace).
    pub available_commands: Vec<AvailableCommand>,
    /// The active mode's value id — the current value of the
    /// mode-category [`ConfigOption`], or the id carried by a legacy v1
    /// `current_mode_update`.
    pub current_mode: Option<ConfigValueId>,
    /// Foreground execution state, from `state_update`.
    pub foreground: Option<Open<ForegroundState>>,
    /// Why the last turn ended, when reported.
    pub stop_reason: Option<Open<StopReason>>,
    /// Work still running after the turn (`state_update.background`).
    pub background: Vec<BackgroundItem>,
    /// Title/updated-at from `session_info_update`.
    pub session_info: SessionInfoState,
    /// Latest context-usage report.
    pub usage: Option<UsageUpdate>,
    /// Subagent sessions by session id; stored updates carry every field
    /// reported so far as [`Patch::Set`].
    pub subagents: BTreeMap<SessionId, SubagentUpdate>,
    /// `true` while replayed history is being applied — from
    /// [`SessionReducer::begin_replay`] until a `replay_complete`
    /// update arrives.
    pub replaying: bool,
    /// `true` once a `replay_complete` update arrived.
    pub replay_done: bool,
}

/// An ergonomic delta emitted by [`SessionReducer::apply`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionEvent {
    /// Message content changed.
    MessageDelta {
        /// The affected message.
        message_id: MessageId,
        /// The message stream.
        kind: MessageKind,
        /// The appended content item for `*_chunk` updates; `None` for
        /// whole-message updates.
        delta: Option<ContentBlock>,
        /// The update's text, when it carried any: a chunk's appended
        /// text, or a whole-message `Set`'s concatenated text. `None`
        /// for non-textual updates and unchanged content.
        text: Option<String>,
        /// The message's full accumulated content after the update.
        full: Vec<ContentBlock>,
    },
    /// A tool call was created or patched; carries the materialized
    /// snapshot.
    ToolCall(ToolCall),
    /// A plan's entries were replaced.
    Plan {
        /// The replaced plan.
        plan_id: PlanId,
        /// The new plan content.
        plan: Plan,
    },
    /// `available_commands` was replaced.
    CommandsChanged,
    /// `config_options` was replaced.
    ConfigChanged,
    /// The session mode changed — the mode-category config option's new
    /// current value, or a legacy v1 `current_mode_update` id.
    ModeChanged(ConfigValueId),
    /// The foreground execution state changed.
    ForegroundChanged {
        /// The new state.
        state: Open<ForegroundState>,
    },
    /// The turn ended (`state_update` reached `idle`).
    TurnEnded {
        /// Why the turn ended, when reported.
        stop_reason: Option<Open<StopReason>>,
        /// The JSON-RPC error object carried by `error` stops.
        error: Option<Value>,
    },
    /// A terminal was created or patched; carries the materialized
    /// update.
    TerminalUpsert(TerminalUpdate),
    /// A terminal produced output.
    TerminalOutput {
        /// The producing terminal.
        terminal_id: TerminalId,
        /// The appended output data.
        data: String,
    },
    /// A subagent session was created or patched.
    Subagent(SubagentUpdate),
    /// `session_info` changed.
    InfoChanged,
    /// Context usage changed.
    Usage(UsageUpdate),
    /// A transient notice arrived.
    Notice(Notice),
    /// A `compaction_update` arrived — status, summary, and error
    /// patches all land here (named for the first update of each
    /// compaction).
    CompactionStarted {
        /// The compaction being reported.
        compaction_id: CompactionId,
        /// The reported patch.
        update: CompactionUpdate,
    },
    /// A chunk of compaction summary arrived.
    CompactionSummary {
        /// The compaction the summary belongs to.
        compaction_id: CompactionId,
        /// The summary content item.
        content: ContentBlock,
    },
    /// Replay finished (`replay_complete`).
    ReplayComplete,
    /// An update the reducer does not model — extension
    /// (`SessionUpdate::Custom`) and unrecognized
    /// (`SessionUpdate::Unknown`) updates land here, as do
    /// non-`items` plan payloads.
    Other(SessionUpdate),
}

/// Folds [`SessionUpdate`]s into a [`SessionState`] and emits
/// [`SessionEvent`]s.
///
/// Order of `apply` calls defines the state; the reducer tolerates
/// chunks arriving before the update that would have created the
/// entity (the entity is then created with defaults).
#[derive(Debug, Default)]
pub struct SessionReducer {
    state: SessionState,
}

impl SessionReducer {
    /// An empty reducer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The materialized session state.
    #[must_use]
    pub const fn state(&self) -> &SessionState {
        &self.state
    }

    /// Marks the start of a replay: updates applied until the next
    /// `replay_complete` update are history being replayed, not live
    /// output. [`SessionState::replaying`] stays `true` meanwhile.
    pub const fn begin_replay(&mut self) {
        self.state.replaying = true;
    }

    /// Applies one update, returning the events it produced.
    pub fn apply(&mut self, update: &SessionUpdate) -> Vec<SessionEvent> {
        match update {
            SessionUpdate::UserMessageChunk(chunk) => {
                vec![self.message_chunk(MessageKind::User, &chunk.message_id, &chunk.content)]
            }
            SessionUpdate::AgentMessageChunk(chunk) => {
                vec![self.message_chunk(MessageKind::Agent, &chunk.message_id, &chunk.content)]
            }
            SessionUpdate::AgentThoughtChunk(chunk) => {
                vec![self.message_chunk(MessageKind::Thought, &chunk.message_id, &chunk.content)]
            }
            SessionUpdate::UserMessage(message) => {
                vec![self.whole_message(MessageKind::User, message)]
            }
            SessionUpdate::AgentMessage(message) => {
                vec![self.whole_message(MessageKind::Agent, message)]
            }
            SessionUpdate::AgentThought(message) => {
                vec![self.whole_message(MessageKind::Thought, message)]
            }
            SessionUpdate::SessionMessage(message) => vec![self.session_message(message)],
            SessionUpdate::SessionMessageChunk(chunk) => vec![self.message_chunk(
                MessageKind::SessionDirected,
                &chunk.message_id,
                &chunk.content,
            )],
            SessionUpdate::StateUpdate(state) => self.state_update(state),
            SessionUpdate::ToolCallUpdate(tool_call) => vec![self.tool_call_update(tool_call)],
            SessionUpdate::ToolCallContentChunk(chunk) => {
                vec![self.tool_call_content_chunk(chunk)]
            }
            SessionUpdate::TerminalUpdate(terminal) => vec![self.terminal_update(terminal)],
            SessionUpdate::TerminalOutputChunk(chunk) => {
                vec![self.terminal_output_chunk(chunk)]
            }
            SessionUpdate::PlanUpdate(plan) => self.plan_update(plan, update),
            SessionUpdate::SubagentUpdate(subagent) => vec![self.subagent_update(subagent)],
            SessionUpdate::AvailableCommandsUpdate(commands) => {
                self.state
                    .available_commands
                    .clone_from(&commands.available_commands);
                vec![SessionEvent::CommandsChanged]
            }
            SessionUpdate::ConfigOptionUpdate(config) => self.config_update(config),
            SessionUpdate::SessionInfoUpdate(info) => self.info_update(info),
            SessionUpdate::UsageUpdate(usage) => {
                self.state.usage = Some(usage.clone());
                vec![SessionEvent::Usage(usage.clone())]
            }
            SessionUpdate::CompactionUpdate(compaction) => {
                vec![SessionEvent::CompactionStarted {
                    compaction_id: compaction.compaction_id.clone(),
                    update: compaction.clone(),
                }]
            }
            SessionUpdate::CompactionSummaryChunk(chunk) => {
                vec![SessionEvent::CompactionSummary {
                    compaction_id: chunk.compaction_id.clone(),
                    content: chunk.content.clone(),
                }]
            }
            SessionUpdate::Notice(notice) => vec![SessionEvent::Notice(notice.clone())],
            SessionUpdate::ReplayComplete(_) => {
                self.state.replaying = false;
                self.state.replay_done = true;
                vec![SessionEvent::ReplayComplete]
            }
            SessionUpdate::CurrentModeUpdate(mode) => {
                let mode_id = mode.current_mode_id.clone();
                self.state.current_mode = Some(mode_id.clone());
                vec![SessionEvent::ModeChanged(mode_id)]
            }
            SessionUpdate::Custom(..) | SessionUpdate::Unknown(..) => {
                vec![SessionEvent::Other(update.clone())]
            }
        }
    }

    /// The stored message for `(kind, id)`, creating it with defaults on
    /// first sight.
    fn entry_message(&mut self, kind: MessageKind, id: &MessageId) -> &mut Message {
        self.state
            .messages
            .entry((kind, id.clone()))
            .or_insert_with(|| Message {
                message_id: id.clone(),
                content: Patch::Set(Vec::new()),
                meta: None,
                extra: BTreeMap::new(),
            })
    }

    /// A chunk appends its content item to the stored message.
    fn message_chunk(
        &mut self,
        kind: MessageKind,
        id: &MessageId,
        content: &ContentBlock,
    ) -> SessionEvent {
        let message = self.entry_message(kind, id);
        let mut stored = message.content.as_option().cloned().unwrap_or_default();
        append_content(&mut stored, content);
        message.content = Patch::Set(stored);
        SessionEvent::MessageDelta {
            message_id: id.clone(),
            kind,
            delta: Some(content.clone()),
            text: content.as_text().map(str::to_owned),
            full: message.content.as_option().cloned().unwrap_or_default(),
        }
    }

    /// A whole-message update replaces, clears, or keeps stored content
    /// per its `content` patch.
    fn whole_message(&mut self, kind: MessageKind, update: &Message) -> SessionEvent {
        let entry = self.entry_message(kind, &update.message_id);
        let text = match &update.content {
            Patch::Set(blocks) => text_of(blocks),
            Patch::Clear | Patch::Unchanged => None,
        };
        match &update.content {
            Patch::Set(blocks) => entry.content = Patch::Set(blocks.clone()),
            Patch::Clear => entry.content = Patch::Set(Vec::new()),
            Patch::Unchanged => {}
        }
        if update.meta.is_some() {
            entry.meta.clone_from(&update.meta);
        }
        entry.extra.extend(update.extra.clone());
        SessionEvent::MessageDelta {
            message_id: update.message_id.clone(),
            kind,
            delta: None,
            text,
            full: entry.content.as_option().cloned().unwrap_or_default(),
        }
    }

    /// A `session_message` is a whole-message update in the
    /// session-directed stream; sender/recipient id patches are not
    /// retained in the stored [`Message`].
    fn session_message(&mut self, update: &SessionMessage) -> SessionEvent {
        let entry = self.entry_message(MessageKind::SessionDirected, &update.message_id);
        let text = match &update.content {
            Patch::Set(blocks) => text_of(blocks),
            Patch::Clear | Patch::Unchanged => None,
        };
        match &update.content {
            Patch::Set(blocks) => entry.content = Patch::Set(blocks.clone()),
            Patch::Clear => entry.content = Patch::Set(Vec::new()),
            Patch::Unchanged => {}
        }
        entry.meta = update.meta.clone().apply_to(entry.meta.take());
        entry.extra.extend(update.extra.clone());
        SessionEvent::MessageDelta {
            message_id: update.message_id.clone(),
            kind: MessageKind::SessionDirected,
            delta: None,
            text,
            full: entry.content.as_option().cloned().unwrap_or_default(),
        }
    }

    /// The stored tool call for `id`, creating it with defaults on first
    /// sight.
    fn tool_call_entry(&mut self, id: &ToolCallId) -> &mut ToolCall {
        self.state
            .tool_calls
            .entry(id.clone())
            .or_insert_with(|| ToolCall {
                tool_call_id: id.clone(),
                ..ToolCall::default()
            })
    }

    /// A `tool_call_update` patches the materialized call.
    fn tool_call_update(&mut self, update: &ToolCallUpdate) -> SessionEvent {
        let entry = self.tool_call_entry(&update.tool_call_id);
        patch_option(&mut entry.name, &update.name);
        patch_default(&mut entry.title, &update.title);
        patch_option(&mut entry.kind, &update.kind);
        patch_option(&mut entry.status, &update.status);
        patch_vec(&mut entry.content, &update.content);
        patch_vec(&mut entry.locations, &update.locations);
        patch_option(&mut entry.raw_input, &update.raw_input);
        patch_option(&mut entry.raw_output, &update.raw_output);
        patch_option(&mut entry.capabilities, &update.capabilities);
        patch_option(&mut entry.meta, &update.meta);
        entry.extra.extend(update.extra.clone());
        SessionEvent::ToolCall(entry.clone())
    }

    /// A `tool_call_content_chunk` appends to the call's content.
    fn tool_call_content_chunk(&mut self, chunk: &ToolCallContentChunk) -> SessionEvent {
        let entry = self.tool_call_entry(&chunk.tool_call_id);
        entry.content.push(chunk.content.clone());
        SessionEvent::ToolCall(entry.clone())
    }

    /// The stored terminal for `id`, creating it with defaults on first
    /// sight.
    fn terminal_entry(&mut self, id: &TerminalId) -> &mut TerminalUpdate {
        self.state
            .terminals
            .entry(id.clone())
            .or_insert_with(|| TerminalUpdate::new(id.clone()))
    }

    /// A `terminal_update` merges its patch fields into the stored
    /// update.
    fn terminal_update(&mut self, update: &TerminalUpdate) -> SessionEvent {
        let entry = self.terminal_entry(&update.terminal_id);
        merge_patch(&mut entry.pid, &update.pid);
        merge_patch(&mut entry.command, &update.command);
        merge_patch(&mut entry.args, &update.args);
        merge_patch(&mut entry.cwd, &update.cwd);
        merge_patch(&mut entry.output, &update.output);
        merge_patch(&mut entry.exit_status, &update.exit_status);
        merge_patch(&mut entry.size, &update.size);
        merge_patch(&mut entry.capabilities, &update.capabilities);
        merge_patch(&mut entry.meta, &update.meta);
        entry.extra.extend(update.extra.clone());
        SessionEvent::TerminalUpsert(entry.clone())
    }

    /// A `terminal_output_chunk` appends to the terminal's output.
    fn terminal_output_chunk(&mut self, chunk: &TerminalOutputChunk) -> SessionEvent {
        let entry = self.terminal_entry(&chunk.terminal_id);
        let mut output = match std::mem::replace(&mut entry.output, Patch::Unchanged) {
            Patch::Set(output) => output,
            Patch::Clear | Patch::Unchanged => TerminalOutput::default(),
        };
        output.data.push_str(&chunk.data);
        output.byte_count = Some(output.byte_count.unwrap_or(0) + chunk.data.len() as u64);
        if chunk.truncated == Some(true) {
            output.truncated = Some(true);
        }
        entry.output = Patch::Set(output);
        SessionEvent::TerminalOutput {
            terminal_id: chunk.terminal_id.clone(),
            data: chunk.data.clone(),
        }
    }

    /// A `plan_update` replaces the plan's entries. Custom/unknown plan
    /// payloads carry no `plan_id`, so they pass through as
    /// [`SessionEvent::Other`].
    fn plan_update(&mut self, update: &PlanUpdate, original: &SessionUpdate) -> Vec<SessionEvent> {
        match &update.plan {
            Open::Known(Plan::Items(items)) => {
                let plan_id = items.plan_id.clone();
                let plan = Plan::Items(items.clone());
                self.state.plans.insert(plan_id.clone(), plan.clone());
                vec![SessionEvent::Plan { plan_id, plan }]
            }
            Open::Custom(..) | Open::Unknown(..) => vec![SessionEvent::Other(original.clone())],
        }
    }

    /// A `subagent_update` merges its patch fields into the stored
    /// update.
    fn subagent_update(&mut self, update: &SubagentUpdate) -> SessionEvent {
        let entry = self
            .state
            .subagents
            .entry(update.session_id.clone())
            .or_insert_with(|| SubagentUpdate {
                session_id: update.session_id.clone(),
                ..SubagentUpdate::default()
            });
        merge_patch(&mut entry.title, &update.title);
        merge_patch(&mut entry.description, &update.description);
        merge_patch(&mut entry.capabilities, &update.capabilities);
        merge_patch(&mut entry.state, &update.state);
        merge_patch(&mut entry.cwd, &update.cwd);
        merge_patch(
            &mut entry.additional_directories,
            &update.additional_directories,
        );
        merge_patch(&mut entry.meta, &update.meta);
        entry.extra.extend(update.extra.clone());
        SessionEvent::Subagent(entry.clone())
    }

    /// A `state_update` tracks the foreground state and detects turn
    /// end: `idle` after a non-idle state, or any `idle` carrying a
    /// `stop_reason`/`error`, emits [`SessionEvent::TurnEnded`].
    fn state_update(&mut self, update: &StateUpdate) -> Vec<SessionEvent> {
        let mut events = Vec::new();
        // `was_known` distinguishes "never saw a foreground" from "saw
        // `idle`" — a bare `idle` update while no foreground was ever
        // reported (e.g. an agent that announces `idle` at `session/new`)
        // is informational, not a turn end.
        let was_known = self.state.foreground.is_some();
        let was_idle = matches!(
            self.state.foreground,
            Some(Open::Known(ForegroundState::Idle))
        );
        let is_idle = matches!(update.state, Open::Known(ForegroundState::Idle));
        if self.state.foreground.as_ref() != Some(&update.state) {
            self.state.foreground = Some(update.state.clone());
            events.push(SessionEvent::ForegroundChanged {
                state: update.state.clone(),
            });
        }
        self.state.background.clone_from(&update.background);
        if update.stop_reason.is_some() {
            self.state.stop_reason.clone_from(&update.stop_reason);
        }
        // A turn ends when the foreground *leaves* a known non-idle
        // state for `idle`, or whenever an `idle` carries a turn-end
        // payload (`stop_reason`/`error` — a rejected prompt ends
        // without ever reporting `running`). Bare initial or repeated
        // `idle`s never produce a second `TurnEnded`.
        if is_idle
            && ((was_known && !was_idle) || update.stop_reason.is_some() || update.error.is_some())
        {
            events.push(SessionEvent::TurnEnded {
                stop_reason: update.stop_reason.clone(),
                error: update.error.clone(),
            });
        }
        events
    }

    /// A `config_option_update` replaces the option list and extracts
    /// the mode-category option's current value as the session mode.
    fn config_update(&mut self, update: &ConfigOptionUpdate) -> Vec<SessionEvent> {
        self.state.config_options.clone_from(&update.config_options);
        let mut events = vec![SessionEvent::ConfigChanged];
        let mode = update
            .config_options
            .iter()
            .find(|option| option.is_mode())
            .and_then(|option| match &option.kind {
                ConfigKind::Select(select) => Some(select.current_value.clone()),
                _ => None,
            });
        if let Some(mode) = mode
            && self.state.current_mode.as_ref() != Some(&mode)
        {
            self.state.current_mode = Some(mode.clone());
            events.push(SessionEvent::ModeChanged(mode));
        }
        events
    }

    /// A `session_info_update` patches `session_info`.
    fn info_update(&mut self, update: &SessionInfoUpdate) -> Vec<SessionEvent> {
        let changed =
            update.meta.is_changed() || update.title.is_changed() || update.updated_at.is_changed();
        patch_option(&mut self.state.session_info.title, &update.title);
        patch_option(&mut self.state.session_info.updated_at, &update.updated_at);
        if changed {
            vec![SessionEvent::InfoChanged]
        } else {
            Vec::new()
        }
    }
}

/// Concatenated text of `content`, `None` when it has no text blocks.
fn text_of(content: &[ContentBlock]) -> Option<String> {
    let mut text = String::new();
    let mut any = false;
    for block in content {
        if let Some(t) = block.as_text() {
            text.push_str(t);
            any = true;
        }
    }
    any.then_some(text)
}

/// Appends `block` to `content`, concatenating consecutive `text` blocks
/// into one.
fn append_content(content: &mut Vec<ContentBlock>, block: &ContentBlock) {
    if let (Some(ContentBlock::Text(last)), ContentBlock::Text(chunk)) = (content.last_mut(), block)
    {
        last.text.push_str(&chunk.text);
        return;
    }
    content.push(block.clone());
}

/// Applies `patch` to an `Option` field (`Set`→`Some`, `Clear`→`None`).
fn patch_option<T: Clone>(stored: &mut Option<T>, patch: &Patch<T>) {
    match patch {
        Patch::Set(value) => *stored = Some(value.clone()),
        Patch::Clear => *stored = None,
        Patch::Unchanged => {}
    }
}

/// Applies `patch` to a `Vec` field (`Set`→replace, `Clear`→empty).
fn patch_vec<T: Clone>(stored: &mut Vec<T>, patch: &Patch<Vec<T>>) {
    match patch {
        Patch::Set(values) => stored.clone_from(values),
        Patch::Clear => stored.clear(),
        Patch::Unchanged => {}
    }
}

/// Applies `patch` to a defaulted field (`Set`→replace, `Clear`→default).
fn patch_default<T: Clone + Default>(stored: &mut T, patch: &Patch<T>) {
    match patch {
        Patch::Set(value) => *stored = value.clone(),
        Patch::Clear => *stored = T::default(),
        Patch::Unchanged => {}
    }
}

/// Merges `incoming` into a stored patch field, keeping only reported
/// values: `Set`→`Set`, `Clear`→`Unchanged` (a cleared field reads the
/// same as never-reported in a materialized snapshot).
fn merge_patch<T: Clone>(stored: &mut Patch<T>, incoming: &Patch<T>) {
    match incoming {
        Patch::Set(value) => *stored = Patch::Set(value.clone()),
        Patch::Clear => *stored = Patch::Unchanged,
        Patch::Unchanged => {}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::domain::{Content, ContentChunk, ToolCallContent};

    fn chunk(id: &str, text: &str) -> SessionUpdate {
        SessionUpdate::AgentMessageChunk(ContentChunk::new(id, text))
    }

    fn message_content(state: &SessionState, kind: MessageKind, id: &str) -> Vec<ContentBlock> {
        state
            .messages
            .get(&(kind, MessageId::from(id)))
            .and_then(|message| message.content.as_option().cloned())
            .unwrap_or_default()
    }

    #[test]
    fn chunk_before_update_creates_then_concatenates() {
        let mut reducer = SessionReducer::new();
        let events = reducer.apply(&chunk("m1", "hello"));
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            SessionEvent::MessageDelta {
                kind: MessageKind::Agent,
                delta: Some(_),
                ..
            }
        ));
        reducer.apply(&chunk("m1", " world"));
        let content = message_content(reducer.state(), MessageKind::Agent, "m1");
        // Consecutive text chunks concatenate into one block.
        assert_eq!(content.len(), 1);
        assert_eq!(content[0].as_text(), Some("hello world"));

        // A whole-message update replaces content.
        reducer.apply(&SessionUpdate::AgentMessage(Message::new(
            "m1",
            vec!["done".into()],
        )));
        let content = message_content(reducer.state(), MessageKind::Agent, "m1");
        assert_eq!(content[0].as_text(), Some("done"));

        // `content: null` clears.
        let update: Message =
            serde_json::from_value(json!({"messageId": "m1", "content": null})).expect("de");
        reducer.apply(&SessionUpdate::AgentMessage(update));
        assert_eq!(
            message_content(reducer.state(), MessageKind::Agent, "m1"),
            []
        );

        // The same id in another stream is a different entity.
        reducer.apply(&SessionUpdate::UserMessageChunk(ContentChunk::new(
            "m1", "mine",
        )));
        assert_eq!(
            message_content(reducer.state(), MessageKind::User, "m1")[0].as_text(),
            Some("mine")
        );
        assert_eq!(
            reducer.state().messages.len(),
            2,
            "(kind, id) keying keeps streams apart"
        );
    }

    #[test]
    fn tool_call_patch_clear_set_and_content_accumulation() {
        let mut reducer = SessionReducer::new();

        // A chunk for an unknown id creates the call with defaults first.
        let content_chunk = ToolCallContentChunk {
            tool_call_id: "t1".into(),
            content: ToolCallContent::Content(Content {
                content: ContentBlock::text("line 1"),
                meta: None,
                extra: BTreeMap::new(),
            }),
            meta: None,
            extra: BTreeMap::new(),
        };
        let events = reducer.apply(&SessionUpdate::ToolCallContentChunk(content_chunk));
        assert!(matches!(events[0], SessionEvent::ToolCall(_)));
        assert_eq!(
            reducer.state().tool_calls[&ToolCallId::from("t1")]
                .content
                .len(),
            1
        );

        // Set fields patch in.
        let update: ToolCallUpdate = serde_json::from_value(json!({
            "toolCallId": "t1", "title": "Run", "name": "bash",
            "kind": "execute", "status": "in_progress",
        }))
        .expect("de");
        reducer.apply(&SessionUpdate::ToolCallUpdate(update));
        let call = &reducer.state().tool_calls[&ToolCallId::from("t1")];
        assert_eq!(call.title, "Run");
        assert_eq!(call.name.as_deref(), Some("bash"));
        assert_eq!(
            call.status,
            Some(Open::Known(crate::domain::ToolCallStatus::InProgress))
        );

        // `content` in an update replaces accumulated chunks.
        let update: ToolCallUpdate = serde_json::from_value(json!({
            "toolCallId": "t1",
            "content": [{"type": "content", "content": {"type": "text", "text": "only"}}],
        }))
        .expect("de");
        reducer.apply(&SessionUpdate::ToolCallUpdate(update));
        assert_eq!(
            reducer.state().tool_calls[&ToolCallId::from("t1")]
                .content
                .len(),
            1
        );

        // null clears: title back to default, name to None.
        let update: ToolCallUpdate =
            serde_json::from_value(json!({"toolCallId": "t1", "title": null, "name": null}))
                .expect("de");
        reducer.apply(&SessionUpdate::ToolCallUpdate(update));
        let call = &reducer.state().tool_calls[&ToolCallId::from("t1")];
        assert_eq!(call.title, "");
        assert_eq!(call.name, None);
    }

    #[test]
    fn plan_update_replaces_entries() {
        let mut reducer = SessionReducer::new();
        let apply = |r: &mut SessionReducer, entries: serde_json::Value| {
            r.apply(
                &serde_json::from_value::<SessionUpdate>(json!({
                    "sessionUpdate": "plan_update",
                    "plan": {"type": "items", "planId": "p1", "entries": entries},
                }))
                .expect("de"),
            )
        };
        let events = apply(&mut reducer, json!([{"content": "a"}, {"content": "b"}]));
        assert!(matches!(events[0], SessionEvent::Plan { .. }));
        let Plan::Items(items) = &reducer.state().plans[&PlanId::from("p1")];
        assert_eq!(items.entries.len(), 2);

        apply(&mut reducer, json!([{"content": "only"}]));
        let Plan::Items(items) = &reducer.state().plans[&PlanId::from("p1")];
        assert_eq!(items.entries.len(), 1);
        assert_eq!(items.entries[0].content, "only");
    }

    #[test]
    fn config_update_extracts_mode() {
        let mut reducer = SessionReducer::new();
        let update: SessionUpdate = serde_json::from_value(json!({
            "sessionUpdate": "config_option_update",
            "configOptions": [
                {"configId": "mode", "name": "Mode", "category": "mode",
                 "type": "select", "currentValue": "plan",
                 "options": [{"value": "code", "name": "Code"}, {"value": "plan", "name": "Plan"}]},
                {"configId": "verbose", "name": "Verbose", "type": "boolean", "currentValue": true},
            ],
        }))
        .expect("de");
        let events = reducer.apply(&update);
        assert!(events.contains(&SessionEvent::ConfigChanged));
        assert!(events.contains(&SessionEvent::ModeChanged("plan".into())));
        assert_eq!(
            reducer.state().current_mode,
            Some(ConfigValueId::from("plan"))
        );

        // Re-applying the same mode does not emit another ModeChanged.
        let events = reducer.apply(&update);
        assert!(events.contains(&SessionEvent::ConfigChanged));
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, SessionEvent::ModeChanged(_)))
        );

        // Legacy v1 `current_mode_update` maps onto the same state.
        let update: SessionUpdate = serde_json::from_value(json!({
            "sessionUpdate": "current_mode_update", "currentModeId": "code",
        }))
        .expect("de");
        let events = reducer.apply(&update);
        assert_eq!(events, vec![SessionEvent::ModeChanged("code".into())]);
        assert_eq!(
            reducer.state().current_mode,
            Some(ConfigValueId::from("code"))
        );
    }

    #[test]
    fn idle_state_update_ends_turn() {
        let mut reducer = SessionReducer::new();
        let state_update = |value: serde_json::Value| {
            let mut object = json!({"sessionUpdate": "state_update"});
            object
                .as_object_mut()
                .expect("object")
                .extend(value.as_object().expect("object").clone());
            serde_json::from_value::<SessionUpdate>(object).expect("de")
        };

        let events = reducer.apply(&state_update(json!({"state": "running"})));
        assert_eq!(
            events,
            vec![SessionEvent::ForegroundChanged {
                state: Open::Known(ForegroundState::Running)
            }]
        );

        let events = reducer.apply(&state_update(
            json!({"state": "idle", "stopReason": "end_turn"}),
        ));
        assert_eq!(
            events,
            vec![
                SessionEvent::ForegroundChanged {
                    state: Open::Known(ForegroundState::Idle)
                },
                SessionEvent::TurnEnded {
                    stop_reason: Some(Open::Known(StopReason::EndTurn)),
                    error: None,
                },
            ]
        );

        // A second bare `idle` is not another turn end.
        let events = reducer.apply(&state_update(json!({"state": "idle"})));
        assert_eq!(events, []);

        // But an `idle` carrying an error stops the turn again.
        let events = reducer.apply(&state_update(json!({
            "state": "idle", "stopReason": "error",
            "error": {"code": -32000, "message": "boom"},
        })));
        assert_eq!(events.len(), 1);
        let SessionEvent::TurnEnded { error, .. } = &events[0] else {
            panic!("expected TurnEnded");
        };
        assert_eq!(error.as_ref().expect("error")["code"], -32000);
    }

    #[test]
    fn replay_flags() {
        let mut reducer = SessionReducer::new();
        assert!(!reducer.state().replaying);
        assert!(!reducer.state().replay_done);

        reducer.begin_replay();
        assert!(reducer.state().replaying);
        // Updates still apply normally while replaying.
        reducer.apply(&chunk("m1", "history"));

        let update: SessionUpdate =
            serde_json::from_value(json!({"sessionUpdate": "replay_complete"})).expect("de");
        let events = reducer.apply(&update);
        assert_eq!(events, vec![SessionEvent::ReplayComplete]);
        assert!(!reducer.state().replaying);
        assert!(reducer.state().replay_done);
    }

    #[test]
    fn terminal_output_chunk_creates_and_appends() {
        let mut reducer = SessionReducer::new();
        let chunk = |data: &str| TerminalOutputChunk {
            terminal_id: "t1".into(),
            data: data.to_string(),
            truncated: None,
            meta: None,
            extra: BTreeMap::new(),
        };
        let events = reducer.apply(&SessionUpdate::TerminalOutputChunk(chunk("ab")));
        assert!(matches!(events[0], SessionEvent::TerminalOutput { .. }));
        reducer.apply(&SessionUpdate::TerminalOutputChunk(chunk("cd")));
        let Patch::Set(output) = &reducer.state().terminals[&TerminalId::from("t1")].output else {
            panic!("expected output");
        };
        assert_eq!(output.data, "abcd");
        assert_eq!(output.byte_count, Some(4));

        // A terminal_update merges fields without touching output.
        let update: TerminalUpdate =
            serde_json::from_value(json!({"terminalId": "t1", "command": "ls"})).expect("de");
        reducer.apply(&SessionUpdate::TerminalUpdate(update));
        let stored = &reducer.state().terminals[&TerminalId::from("t1")];
        assert!(matches!(stored.command, Patch::Set(ref c) if c == "ls"));
        assert!(matches!(stored.output, Patch::Set(ref o) if o.data == "abcd"));
    }

    #[test]
    fn custom_updates_pass_through_as_other() {
        let mut reducer = SessionReducer::new();
        let update: SessionUpdate =
            serde_json::from_value(json!({"sessionUpdate": "_acme/thing", "n": 1})).expect("de");
        let events = reducer.apply(&update);
        assert_eq!(events, vec![SessionEvent::Other(update)]);
    }

    #[test]
    fn subagent_update_merges() {
        let mut reducer = SessionReducer::new();
        let update: SessionUpdate = serde_json::from_value(json!({
            "sessionUpdate": "subagent_update",
            "sessionId": "sub-1", "title": "researcher",
        }))
        .expect("de");
        let events = reducer.apply(&update);
        assert!(matches!(events[0], SessionEvent::Subagent(_)));
        let stored = &reducer.state().subagents[&SessionId::from("sub-1")];
        assert!(matches!(stored.title, Patch::Set(ref t) if t == "researcher"));
    }
}
