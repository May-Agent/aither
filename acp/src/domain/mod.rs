//! Canonical ACP domain model — v2 wire shapes.
//!
//! Every type here serializes as the ACP v2 wire format (camelCase keys,
//! `_meta`, tagged open unions). ACP v1 wire shapes and the stateful
//! v1↔v2 translator live in [`crate::wire`].
//!
//! Domain intentionally has no dependency on `crate::rpc`: where a
//! payload can carry a JSON-RPC error object (e.g.
//! [`update::StateUpdate::error`]) it is represented as
//! `serde_json::Value`.

pub mod auth;
pub mod command;
pub mod config;
pub mod content;
pub mod elicitation;
pub mod ids;
pub mod init;
pub mod mcp;
pub mod open;
pub mod patch;
pub mod permission;
pub mod session;
pub mod terminal;
pub mod tool;
pub mod update;
pub mod usage;

pub use auth::{AgentAuth, AuthKind, AuthLoginParams, AuthLogoutParams, AuthMethod, TerminalAuth};
pub use command::{AvailableCommand, AvailableCommandsUpdate, Command, CommandInput, TextInput};
pub use config::{
    BooleanConfig, ConfigCategory, ConfigKind, ConfigOption, ConfigOptionUpdate, ConfigValue,
    ConfigValueBoolean, ConfigValueById, SelectConfig, SelectGroup, SelectOption, SelectOptions,
    SetConfigParams, SetConfigResult,
};
pub use content::{
    Annotations, AudioContent, BlobResourceContents, ContentBlock, ContentChunk, EmbeddedResource,
    Icon, ImageContent, Message, ResourceContents, ResourceLink, Role, TextContent,
    TextResourceContents,
};
pub use elicitation::{
    AcceptAction, ArrayProperty, BooleanProperty, CancelAction, DeclineAction, ElicitAction,
    ElicitComplete, ElicitContentValue, ElicitCreate, ElicitMode, ElicitProperty, ElicitResult,
    ElicitationSchema, EnumOption, FormMode, IntegerProperty, NumberProperty, PropertyCommon,
    StringFormat, StringProperty, UrlMode,
};
pub use ids::{
    AuthMethodId, CompactionId, ConfigGroupId, ConfigValueId, ElicitationId, MessageId,
    PermissionOptionId, PlanId, SessionId, TerminalId, ToolCallId,
};
pub use init::{
    AgentCapabilities, ClientAuthCaps, ClientCapabilities, ClientSessionCaps, ElicitCaps,
    Implementation, InitResult, McpCaps, PromptCaps, SessionCaps, SystemPromptCaps,
};
pub use mcp::{EnvVar, HttpHeader, McpServer, McpServerHttp, McpServerStdio};
pub use open::{ExtIdent, ExtIdentError, ExtMethod, Open, Openable};
pub use patch::{EmptyObject, MetaPatch, Patch};
pub use permission::{
    CancelledOutcome, CommandSubject, PermissionOption, PermissionOptionKind, PermissionOutcome,
    PermissionRequest, PermissionResult, PermissionSubject, SelectedOutcome, SubjectPatch,
    ToolCallSubject,
};
pub use session::{
    CancelParams, CurrentModeUpdate, ForkParams, ModeSystemPrompt, NewSessionParams,
    PromptAccepted, PromptFinished, PromptParams, ReplayCursor, ReplayStart, ResumeParams,
    SessionEndParams, SessionInfo, SessionInfoUpdate, SessionListParams, SessionNotification,
    SessionPage, SessionSetupParams, SetupResult, SystemPrompt,
};
pub use terminal::{
    TerminalCaps, TerminalExitStatus, TerminalOutput, TerminalOutputChunk, TerminalSize,
    TerminalUpdate,
};
pub use tool::{
    Content, Diff, DiffChange, DiffFileType, DiffOperation, DiffPatch, DiffPatchFormat,
    DiffPathChange, DiffPathPairChange, Location, TerminalRef, ToolCall, ToolCallCaps,
    ToolCallContent, ToolCallContentChunk, ToolCallStatus, ToolCallUpdate, ToolKind,
};
pub use update::{
    BackgroundItem, BackgroundSubagent, BackgroundTerminal, BackgroundToolCall, CompactionStatus,
    CompactionSummaryChunk, CompactionUpdate, ForegroundState, Notice, NoticeSeverity, Plan,
    PlanEntry, PlanEntryPriority, PlanEntryStatus, PlanItems, PlanUpdate, ReplayComplete,
    SessionMessage, SessionMessageChunk, SessionUpdate, StateUpdate, StopReason, SubagentUpdate,
};
pub use usage::{Cost, Usage, UsageUpdate};
