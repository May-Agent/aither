//! Aither Agent Framework
//!
//! An ergonomic, minimal-core agent framework for building autonomous AI agents.
//! The agent handles the tool-use loop internally with a simple API.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use aither_agent::Agent;
//!
//! // Simple one-shot query
//! let agent = Agent::new(llm);
//! let response = agent.query("What is 2+2?").await?;
//!
//! // With tools and hooks
//! let agent = Agent::builder(llm)
//!     .tool(FileSystemTool::read_only("."))
//!     .hook(LoggingHook)
//!     .build();
//!
//! let response = agent.query("List all Rust files").await?;
//! ```
//!
//! # Features
//!
//! - **Ergonomic API**: Simple `query()` for one-shot tasks, `run()` to handle each event as it happens
//! - **Internal Loop**: Tool execution handled automatically
//! - **Trait-based Hooks**: Intercept operations at compile time
//! - **Context Management**: One policy, a cache-aware window, async compaction
//! - **Generic over LLM**: Works with any `LanguageModel` implementation
//!
//! The crate carries no product policy: no default system prompt, no terminal
//! tool and no built-in subagent types. The embedder supplies its own
//! instructions, tools and policies.

// Core modules
mod agent;
mod builder;
mod cache_stats;
#[cfg(test)]
mod call_identity_tests;
mod checkpoint;
mod clock;
mod compaction_state;
mod config;
mod connectivity;
pub mod context;
mod context_window;
mod effort;
mod error;
mod event;
mod fork;
mod hook;
#[cfg(test)]
mod idle_compaction_tests;
#[cfg(test)]
mod midstream_retry_tests;
mod model_group;
mod request_parameters;
#[cfg(test)]
mod test_support;
mod tools;
mod warmth;

// Re-export tool crates
#[cfg(feature = "command")]
pub use aither_command as command;
#[cfg(feature = "filesystem")]
pub use aither_fs as filesystem;
#[cfg(feature = "webfetch")]
pub use aither_webfetch as webfetch;
#[cfg(feature = "websearch")]
pub use aither_websearch as websearch;

// Public API
pub use agent::{Agent, CompactResult, Run};
pub use builder::AgentBuilder;
pub use cache_stats::CacheStats;
pub use checkpoint::AgentCheckpoint;
pub use clock::{AnyClock, Clock, Instant, ManualClock, SystemClock};
pub use config::AgentConfig;
pub use context::{
    AnyContextSource, AnyEventSource, AnyFileReferences, AnyLiveSource, AnyStateSource,
    AttachmentCatalogEntry, AttachmentRef, Block, CompactionError, CompletedCompaction, Context,
    ContextBudget, ContextCheckpoint, ContextFile, ContextItem, ContextPolicy, ContextSource,
    EventSource, ExternalContent, ExternalSlot, FileReferences, Handoff, HandoffDocument,
    HandoffPrompt, HandoffSchema, InstructionSlot, InvalidHandoff, KeptAttachment, LineRange,
    LiveChange, LiveContent, LiveSource, LiveUpdate, PrefixBlock, PrefixChain, PrefixRewrites,
    PromptCache, PromptPrefix, ReassemblyReport, Restoration, SkillCatalogEntry, Sourced,
    SourcedSlot, StateChange, StateSource, SystemLayout,
};
pub use context_window::{ContextWindowMetrics, ContextWindowPhase, ContextWindowSnapshot};
pub use error::AgentError;
pub use event::{AgentEvent, CompactionOutcome, TimedEvent};
pub use fork::{ForkSnapshot, ForkSource};
pub use hook::{
    AnyStopHook, CheckpointContext, CheckpointReason, HCons, Hook, PostToolAction, PreToolAction,
    StopContext, StopDecision, StopHook, StopReason, ToolResultContext, ToolUseContext,
    TurnBoundaryAction, TurnBoundaryContext,
};
pub use tools::AgentTools;
pub use warmth::{CacheState, CacheWarmth};

// Model groups for budget tracking and fallback
pub use model_group::{Budget, BudgetedModel, ModelGroup, ModelGroupError};

// Re-export core tool trait for convenience
pub use aither_attachments::{CacheEntry, FileCache};
pub use aither_core::llm::Tool;
