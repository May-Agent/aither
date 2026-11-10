//! Who made a tool call: the call's id and the agent behind it.
//!
//! The agent loop attaches a [`CallIdentity`] to the [`ToolContext`](super::ToolContext)
//! of every call it dispatches, so a tool, and any command the tool runs on its
//! behalf, can attribute what it did to the exact call and agent.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{self, Display};

/// The id a model provider gave one tool call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct ToolCallId(String);

impl ToolCallId {
    /// Wraps a provider-assigned tool call id.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The id as the provider spelled it.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for ToolCallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A subagent: its type and the call that started it.
///
/// The starting call's identity makes the subagent unique across nested
/// subagents and ties it to the call that launched it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SubagentIdentity {
    name: String,
    started_by: Option<Box<CallIdentity>>,
}

impl SubagentIdentity {
    /// A subagent of type `name` started by `started_by`; `None` when the
    /// dispatcher of the starting call attached no identity.
    #[must_use]
    pub fn new(name: impl Into<String>, started_by: Option<CallIdentity>) -> Self {
        Self {
            name: name.into(),
            started_by: started_by.map(Box::new),
        }
    }

    /// The subagent type it runs as.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The call that started this subagent, if its dispatcher identified it.
    #[must_use]
    pub fn started_by(&self) -> Option<&CallIdentity> {
        self.started_by.as_deref()
    }
}

/// The agent making a tool call.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AgentIdentity {
    /// The main agent.
    Main,
    /// A subagent, at any nesting depth.
    Subagent(SubagentIdentity),
}

/// What identifies one tool call: its id and the agent that made it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CallIdentity {
    call: ToolCallId,
    agent: AgentIdentity,
}

impl CallIdentity {
    /// The identity of call `call` made by `agent`.
    #[must_use]
    pub const fn new(call: ToolCallId, agent: AgentIdentity) -> Self {
        Self { call, agent }
    }

    /// The call's id.
    #[must_use]
    pub const fn call(&self) -> &ToolCallId {
        &self.call
    }

    /// The agent that made the call.
    #[must_use]
    pub const fn agent(&self) -> &AgentIdentity {
        &self.agent
    }
}

/// Where an agent stands in the tree of agents a run started: empty for the
/// main agent, the subagent's name beneath its parent's path otherwise.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct AgentPath(Vec<String>);

impl AgentPath {
    /// The main agent.
    #[must_use]
    pub const fn main() -> Self {
        Self(Vec::new())
    }

    /// The path of the agent `identity` names, beneath the agent that
    /// started it.
    #[must_use]
    pub fn of(identity: &AgentIdentity) -> Self {
        match identity {
            AgentIdentity::Main => Self::main(),
            AgentIdentity::Subagent(subagent) => {
                let mut path = subagent
                    .started_by()
                    .map_or_else(Self::main, |call| Self::of(call.agent()));
                path.0.push(subagent.name().to_string());
                path
            }
        }
    }

    /// The path of a child named `name` beneath this agent.
    #[must_use]
    pub fn child(&self, name: impl Into<String>) -> Self {
        let mut path = self.clone();
        path.0.push(name.into());
        path
    }

    /// The names from the main agent down to this one.
    #[must_use]
    pub fn segments(&self) -> &[String] {
        &self.0
    }

    /// Whether this is the main agent.
    #[must_use]
    pub const fn is_main(&self) -> bool {
        self.0.is_empty()
    }
}

impl Display for AgentPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("main");
        }
        f.write_str(&self.0.join("/"))
    }
}
