//! Hooks: where the embedder decides.
//!
//! Events observe; hooks are the control points. One [`Hooks`] value holds
//! the embedder's decisions at every point the loop offers one, and each
//! method has a default that changes nothing, so an implementation overrides
//! only what it decides.
//!
//! # Example
//!
//! ```rust,ignore
//! struct ReadOnly;
//!
//! impl Hooks for ReadOnly {
//!     async fn before_tool(&self, call: &ToolInvocation<'_>) -> ToolDecision {
//!         if call.name == "write" {
//!             ToolDecision::Deny("this agent only reads".into())
//!         } else {
//!             ToolDecision::Allow
//!         }
//!     }
//! }
//!
//! let agent = Agent::builder(llm).hooks(ReadOnly).build();
//! ```

use std::time::Duration;

use aither_core::llm::tool::ApprovalRequest;
use aither_core::llm::{CallIdentity, ToolResult};

use crate::context::ContextBlock;
use crate::event::RunId;

/// What the loop knows as it assembles a request.
#[derive(Debug)]
#[non_exhaustive]
pub struct RequestCx<'a> {
    /// The run the request belongs to.
    pub run: &'a RunId,
    /// The turn the request starts, counting from 1.
    pub turn: usize,
    /// Number of recent conversation messages currently in memory.
    pub message_count: usize,
}

/// A tool call the model made, before it runs.
#[derive(Debug)]
#[non_exhaustive]
pub struct ToolInvocation<'a> {
    /// The call and the agent making it.
    pub call: &'a CallIdentity,
    /// The tool.
    pub name: &'a str,
    /// The arguments as the model made them.
    pub arguments: &'a serde_json::Value,
    /// The turn the call belongs to.
    pub turn: usize,
    /// Number of recent conversation messages currently in memory.
    pub message_count: usize,
}

/// What [`Hooks::before_tool`] decides for a tool call.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ToolDecision {
    /// Run the call as the model made it.
    Allow,
    /// Do not run it; the model reads `reason` as the call's result.
    Deny(String),
    /// Run it with these arguments instead.
    Rewrite(serde_json::Value),
    /// Run it if the agent's [`Approver`](aither_core::llm::tool::Approver)
    /// approves `request`; a denial reaches the model like
    /// [`Deny`](Self::Deny). Without an approver the run fails.
    Ask(ApprovalRequest),
    /// End the run with this reason as its error.
    Abort(String),
}

/// A tool call after it ran.
///
/// The error is the one the tool returned, typed, so a hook can downcast it.
#[derive(Debug)]
#[non_exhaustive]
pub struct ToolOutcome {
    /// The call and the agent that made it.
    pub call: CallIdentity,
    /// The tool.
    pub name: String,
    /// The arguments the call ran with.
    pub arguments: serde_json::Value,
    /// What the tool returned. A call a hook denied is an `Ok` error result.
    pub result: Result<ToolResult, aither_core::Error>,
    /// How long the tool ran.
    pub duration: Duration,
}

impl ToolOutcome {
    /// What the model reads of the call.
    #[must_use]
    pub fn for_model(&self) -> ToolResult {
        match &self.result {
            Ok(result) => result.clone(),
            Err(error) => ToolResult::error(format!("Error: {error}")),
        }
    }
}

/// What [`Hooks::after_tool`] decides for a finished tool call.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AfterTool {
    /// The model reads what the tool returned.
    Keep,
    /// The model reads this instead.
    Replace(ToolResult),
    /// The model reads what the tool returned, and the conversation gains
    /// this context as a durable block after the turn's results.
    Append(String),
    /// End the run with this reason as its error.
    Abort(String),
}

/// The model ended its turn with no tool call.
#[derive(Debug)]
#[non_exhaustive]
pub struct StopCx<'a> {
    /// The model's reply.
    pub final_text: &'a str,
    /// The turn, counting from 1.
    pub turn: usize,
    /// Number of recent conversation messages currently in memory.
    pub message_count: usize,
}

/// What [`Hooks::on_stop`] decides when the model ends its turn.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StopDecision {
    /// The run ends.
    End,
    /// The run goes on for another turn; the model reads `reason` as a
    /// system reminder.
    Continue {
        /// Why the run is not over.
        reason: String,
    },
}

/// A turn that called tools has ended.
#[derive(Debug)]
#[non_exhaustive]
pub struct TurnBoundaryContext<'a> {
    /// The text the model wrote in the turn.
    pub assistant_text: &'a str,
    /// The turn, counting from 1.
    pub turn: usize,
    /// Number of recent conversation messages currently in memory.
    pub message_count: usize,
    /// The turn's tool calls and what became of them, in completion order.
    pub outcomes: &'a [ToolOutcome],
}

/// What [`Hooks::on_turn_boundary`] decides at a turn's end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TurnBoundaryAction {
    /// The run goes on.
    Continue,
    /// The run ends here, such as to let a waiting message in.
    EndTurn,
}

/// The embedder's decisions at the loop's control points.
pub trait Hooks: Send + Sync + 'static {
    /// Blocks to put into the conversation before a request is sent.
    fn before_request(
        &self,
        cx: &RequestCx<'_>,
    ) -> impl std::future::Future<Output = Vec<ContextBlock>> + Send {
        let _ = cx;
        async { Vec::new() }
    }

    /// Whether a tool call runs, and how.
    fn before_tool(
        &self,
        call: &ToolInvocation<'_>,
    ) -> impl std::future::Future<Output = ToolDecision> + Send {
        let _ = call;
        async { ToolDecision::Allow }
    }

    /// What the model reads of a finished tool call.
    fn after_tool(
        &self,
        outcome: &ToolOutcome,
    ) -> impl std::future::Future<Output = AfterTool> + Send {
        let _ = outcome;
        async { AfterTool::Keep }
    }

    /// Whether the run ends when the model ends its turn without a tool
    /// call.
    fn on_stop(&self, cx: &StopCx<'_>) -> impl std::future::Future<Output = StopDecision> + Send {
        let _ = cx;
        async { StopDecision::End }
    }

    /// Whether the run goes on after a turn that called tools.
    fn on_turn_boundary(
        &self,
        cx: &TurnBoundaryContext<'_>,
    ) -> impl std::future::Future<Output = TurnBoundaryAction> + Send {
        let _ = cx;
        async { TurnBoundaryAction::Continue }
    }
}

impl Hooks for () {}
