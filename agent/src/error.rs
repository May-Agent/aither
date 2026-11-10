//! Agent error types.

use core::fmt;
use std::borrow::Cow;

use aither_core::llm::{Classify, Failure};

/// Errors that can occur during agent execution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AgentError {
    /// LLM returned an error.
    Llm {
        /// The provider's error message.
        message: String,
        /// How the provider error classifies: network loss, transient fault,
        /// quota exhaustion or permanent failure, and the requested retry
        /// delay.
        failure: Failure,
    },

    /// Tool execution failed.
    ToolExecution {
        /// Name of the tool that failed.
        name: String,
        /// The underlying error message.
        error: String,
    },

    /// Maximum iterations exceeded without completing the task.
    MaxIterations {
        /// The iteration limit that was exceeded.
        limit: usize,
    },

    /// A hook rejected the operation.
    HookRejected {
        /// Name of the hook that rejected.
        hook: Cow<'static, str>,
        /// Reason for rejection.
        reason: String,
    },

    /// Tool not found.
    ToolNotFound {
        /// Name of the missing tool.
        name: String,
    },

    /// Configuration error.
    Config(String),

    /// A tool call needed an approval and none could be had: the agent has
    /// no approver, the approver failed, or the run was cancelled while
    /// waiting.
    Approval(std::sync::Arc<aither_core::llm::tool::ApprovalError>),

    /// The run was cancelled.
    ///
    /// A cancelled [`Run`](crate::Run) ends with
    /// [`RunOutcome::Cancelled`](crate::RunOutcome::Cancelled) and `drive`
    /// returns normally; this is what [`Agent::query`](crate::Agent::query),
    /// which returns a reply, reports.
    Cancelled,

    /// The run paused for a reason that needs the embedder; see
    /// [`RunPause`](crate::RunPause).
    /// [`Agent::query`](crate::Agent::query) reports it, since it has no
    /// reply to return.
    Paused(crate::RunPause),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Llm { message, .. } => write!(f, "LLM error: {message}"),
            Self::ToolExecution { name, error } => {
                write!(f, "tool '{name}' failed: {error}")
            }
            Self::MaxIterations { limit } => {
                write!(f, "exceeded maximum iterations ({limit})")
            }
            Self::HookRejected { hook, reason } => {
                write!(f, "hook '{hook}' rejected: {reason}")
            }
            Self::ToolNotFound { name } => {
                write!(f, "tool '{name}' not found")
            }
            Self::Config(msg) => write!(f, "configuration error: {msg}"),
            Self::Approval(error) => write!(f, "{error}"),
            Self::Cancelled => f.write_str("the run was cancelled"),
            Self::Paused(pause) => write!(f, "the run paused: {pause:?}"),
        }
    }
}

impl std::error::Error for AgentError {}

impl AgentError {
    /// Captures a provider error with its classification.
    pub fn from_model_error<E: std::error::Error + Classify>(error: &E) -> Self {
        Self::Llm {
            message: error.to_string(),
            failure: error.failure(),
        }
    }

    /// An LLM-side error that originates in this process, never from a
    /// provider, and so is permanent.
    pub fn llm_local(message: impl Into<String>) -> Self {
        Self::Llm {
            message: message.into(),
            failure: Failure::permanent(),
        }
    }

    /// How the provider failure classifies, or `None` when this error is not
    /// a provider failure.
    ///
    /// Retry policy belongs to the caller: this says whether a retry can help
    /// ([`Failure::is_retryable`]) and what delay the upstream asked for.
    #[must_use]
    pub const fn failure(&self) -> Option<&Failure> {
        match self {
            Self::Llm { failure, .. } => Some(failure),
            _ => None,
        }
    }
}

impl From<anyhow::Error> for AgentError {
    fn from(error: anyhow::Error) -> Self {
        Self::llm_local(error.to_string())
    }
}

impl From<String> for AgentError {
    fn from(error: String) -> Self {
        Self::llm_local(error)
    }
}

impl From<crate::context::CompactionError> for AgentError {
    fn from(error: crate::context::CompactionError) -> Self {
        use crate::context::CompactionError;
        let failure = match &error {
            CompactionError::Provider { failure, .. } => *failure,
            CompactionError::Invalid { .. } | CompactionError::Empty => Failure::permanent(),
        };
        Self::Llm {
            message: error.to_string(),
            failure,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use aither_core::llm::{Classify, Failure, FailureKind};

    use super::AgentError;

    #[derive(Debug)]
    struct Overloaded;

    impl std::fmt::Display for Overloaded {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("overloaded")
        }
    }

    impl std::error::Error for Overloaded {}

    impl Classify for Overloaded {
        fn failure(&self) -> Failure {
            Failure::transient().with_retry_after(Some(Duration::from_secs(5)))
        }
    }

    #[test]
    fn provider_failure_is_carried() {
        let error = AgentError::from_model_error(&Overloaded);
        assert_eq!(error.to_string(), "LLM error: overloaded");
        let failure = error.failure().expect("an LLM error carries a failure");
        assert_eq!(failure.kind, FailureKind::Transient);
        assert_eq!(failure.retry_after, Some(Duration::from_secs(5)));
    }

    #[test]
    fn non_llm_errors_have_no_failure() {
        assert!(AgentError::Config("x".into()).failure().is_none());
        assert_eq!(
            AgentError::llm_local("x").failure().map(|f| f.kind),
            Some(FailureKind::Permanent)
        );
    }
}

/// The serialized form of an [`AgentError`].
///
/// An approval error holds a boxed error that has no wire form, so it travels
/// as its message and returns as an error carrying that message.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
enum AgentErrorRepr {
    Llm {
        message: String,
        failure: Failure,
    },
    ToolExecution {
        name: String,
        error: String,
    },
    MaxIterations {
        limit: usize,
    },
    HookRejected {
        hook: String,
        reason: String,
    },
    ToolNotFound {
        name: String,
    },
    Config(String),
    ApprovalNoApprover {
        kind: aither_core::llm::tool::ApprovalKind,
    },
    ApprovalFailed {
        kind: aither_core::llm::tool::ApprovalKind,
        message: String,
    },
    ApprovalCancelled,
    Cancelled,
    Paused(crate::RunPause),
}

#[cfg(feature = "serde")]
impl From<&AgentError> for AgentErrorRepr {
    fn from(error: &AgentError) -> Self {
        use aither_core::llm::tool::ApprovalError;
        match error {
            AgentError::Llm { message, failure } => Self::Llm {
                message: message.clone(),
                failure: *failure,
            },
            AgentError::ToolExecution { name, error } => Self::ToolExecution {
                name: name.clone(),
                error: error.clone(),
            },
            AgentError::MaxIterations { limit } => Self::MaxIterations { limit: *limit },
            AgentError::HookRejected { hook, reason } => Self::HookRejected {
                hook: hook.to_string(),
                reason: reason.clone(),
            },
            AgentError::ToolNotFound { name } => Self::ToolNotFound { name: name.clone() },
            AgentError::Config(message) => Self::Config(message.clone()),
            AgentError::Approval(approval) => match &**approval {
                ApprovalError::NoApprover { kind } => {
                    Self::ApprovalNoApprover { kind: kind.clone() }
                }
                ApprovalError::Failed { kind, failure } => Self::ApprovalFailed {
                    kind: kind.clone(),
                    message: std::error::Error::source(failure)
                        .map_or_else(|| failure.to_string(), ToString::to_string),
                },
                ApprovalError::Cancelled => Self::ApprovalCancelled,
            },
            AgentError::Cancelled => Self::Cancelled,
            AgentError::Paused(pause) => Self::Paused(*pause),
        }
    }
}

#[cfg(feature = "serde")]
impl From<AgentErrorRepr> for AgentError {
    fn from(repr: AgentErrorRepr) -> Self {
        use aither_core::llm::tool::{ApprovalError, ApproverFailure};
        let approval = |error| Self::Approval(std::sync::Arc::new(error));
        match repr {
            AgentErrorRepr::Llm { message, failure } => Self::Llm { message, failure },
            AgentErrorRepr::ToolExecution { name, error } => Self::ToolExecution { name, error },
            AgentErrorRepr::MaxIterations { limit } => Self::MaxIterations { limit },
            AgentErrorRepr::HookRejected { hook, reason } => Self::HookRejected {
                hook: hook.into(),
                reason,
            },
            AgentErrorRepr::ToolNotFound { name } => Self::ToolNotFound { name },
            AgentErrorRepr::Config(message) => Self::Config(message),
            AgentErrorRepr::ApprovalNoApprover { kind } => {
                approval(ApprovalError::NoApprover { kind })
            }
            AgentErrorRepr::ApprovalFailed { kind, message } => approval(ApprovalError::Failed {
                kind,
                failure: ApproverFailure::new(std::io::Error::other(message)),
            }),
            AgentErrorRepr::ApprovalCancelled => approval(ApprovalError::Cancelled),
            AgentErrorRepr::Cancelled => Self::Cancelled,
            AgentErrorRepr::Paused(pause) => Self::Paused(pause),
        }
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for AgentError {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        AgentErrorRepr::from(self).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for AgentError {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        AgentErrorRepr::deserialize(deserializer).map(Self::from)
    }
}
