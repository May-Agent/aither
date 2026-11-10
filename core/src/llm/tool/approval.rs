//! Asking the embedder whether an action may go ahead.
//!
//! A tool that is about to do something its embedder must approve first,
//! such as running a command outside the sandbox, builds an
//! [`ApprovalRequest`] and awaits [`ToolContext::approve`](super::ToolContext::approve).
//! The request reaches the [`Approver`] the dispatcher attached with
//! [`ToolContext::with_approver`](super::ToolContext::with_approver), so the
//! tool needs no agent loop: a host that builds a [`ToolContext`](super::ToolContext)
//! itself attaches its own approver the same way the loop does.
//!
//! ```
//! use aither_core::llm::{
//!     Approval, ApprovalKind, ApprovalRequest, Approver, AnyApprover, ToolContext,
//! };
//!
//! struct DenyAll;
//!
//! impl Approver for DenyAll {
//!     async fn approve(&self, request: ApprovalRequest) -> Approval {
//!         Approval::Denied {
//!             reason: format!("`{}` is not allowed here", request.subject()),
//!         }
//!     }
//! }
//!
//! # tokio_test::block_on(async {
//! let cx = ToolContext::new().with_approver(AnyApprover::new(DenyAll));
//! let request = ApprovalRequest::new(ApprovalKind::new("unsandboxed"), "rm -rf target");
//! let approval = cx.approve(request).await.unwrap();
//! assert!(!approval.is_approved());
//! # });
//! ```

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use core::fmt::{self, Debug, Display};
use core::future::Future;
use core::pin::Pin;

use super::CallIdentity;

/// What kind of action a request asks about, named by whoever builds the
/// request.
///
/// An approver groups and remembers decisions by it: May records its
/// approvals per escalation mode (sandboxed, unsafe, SSH, publish), the
/// Claude Agent SDK hands `canUseTool` the tool name, and an ACP permission
/// request carries the tool call's kind.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ApprovalKind(Cow<'static, str>);

impl ApprovalKind {
    /// A kind with the given name.
    #[must_use]
    pub fn new(name: impl Into<Cow<'static, str>>) -> Self {
        Self(name.into())
    }

    /// The kind's name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for ApprovalKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One action awaiting approval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalRequest {
    kind: ApprovalKind,
    subject: String,
    reason: Option<String>,
    call: Option<CallIdentity>,
}

impl ApprovalRequest {
    /// A request to approve `subject`, an action of `kind`.
    ///
    /// `subject` is exactly what would go ahead, as the person deciding
    /// should see it: the script to run, the text to publish, a tool call's
    /// arguments.
    #[must_use]
    pub fn new(kind: ApprovalKind, subject: impl Into<String>) -> Self {
        Self {
            kind,
            subject: subject.into(),
            reason: None,
            call: None,
        }
    }

    /// Says why the action needs approval, such as the rule that stopped it
    /// (the `decisionReason` the Claude Agent SDK hands `canUseTool`).
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Attributes the request to the tool call that raised it.
    ///
    /// [`ToolContext::approve`](super::ToolContext::approve) attaches its
    /// call's identity itself; this is for a request raised outside a tool
    /// call's context, such as one the agent loop raises before the call
    /// runs.
    #[must_use]
    pub fn with_call(mut self, call: CallIdentity) -> Self {
        self.call = Some(call);
        self
    }

    /// The kind of action asked about.
    #[must_use]
    pub const fn kind(&self) -> &ApprovalKind {
        &self.kind
    }

    /// The action that would go ahead.
    #[must_use]
    pub fn subject(&self) -> &str {
        &self.subject
    }

    /// Why the action needs approval, when the requester said.
    #[must_use]
    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }

    /// The tool call, and through it the agent, that raised the request.
    ///
    /// An approver attributes the request by it, so a subagent's request can
    /// be put to the person differently from the main agent's (May labels
    /// each request with its agent, the Claude Agent SDK passes `toolUseID`
    /// and `agentID`). `None` for a request raised outside any call, such as
    /// a host running a command with no agent loop.
    #[must_use]
    pub const fn call(&self) -> Option<&CallIdentity> {
        self.call.as_ref()
    }
}

/// An approver's answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Approval {
    /// The action may go ahead.
    Approved,
    /// The action must not go ahead.
    Denied {
        /// Why, worded for the model, which reads it in place of the action's
        /// result.
        reason: String,
    },
}

impl Approval {
    /// Whether the action may go ahead.
    #[must_use]
    pub const fn is_approved(&self) -> bool {
        matches!(self, Self::Approved)
    }
}

/// Decides approval requests for an embedder.
///
/// It may answer at once from a policy or a remembered decision, or wait for
/// a person. Dropping the returned future abandons the request: that is how
/// a cancelled call withdraws its question.
pub trait Approver: Send + Sync + 'static {
    /// The answer to `request`.
    fn approve(&self, request: ApprovalRequest) -> impl Future<Output = Approval> + Send;
}

/// Object-safe twin of [`Approver`], so [`AnyApprover`] can hold any approver.
trait ApproverImpl: Send + Sync {
    fn approve(
        &self,
        request: ApprovalRequest,
    ) -> Pin<Box<dyn Future<Output = Approval> + Send + '_>>;
}

impl<A: Approver> ApproverImpl for A {
    fn approve(
        &self,
        request: ApprovalRequest,
    ) -> Pin<Box<dyn Future<Output = Approval> + Send + '_>> {
        Box::pin(Approver::approve(self, request))
    }
}

/// Any [`Approver`], shared: every call a dispatcher runs carries a clone of
/// the same approver.
#[derive(Clone)]
pub struct AnyApprover(Arc<dyn ApproverImpl>);

impl AnyApprover {
    /// Wraps `approver`.
    #[must_use]
    pub fn new(approver: impl Approver) -> Self {
        Self(Arc::new(approver))
    }
}

impl Approver for AnyApprover {
    fn approve(&self, request: ApprovalRequest) -> impl Future<Output = Approval> + Send {
        self.0.approve(request)
    }
}

impl Debug for AnyApprover {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyApprover").finish_non_exhaustive()
    }
}

/// Why [`ToolContext::approve`](super::ToolContext::approve) produced no
/// answer.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ApprovalError {
    /// The call's dispatcher attached no approver, so nobody can approve the
    /// action; it must not go ahead.
    #[error(
        "approval for `{kind}` was requested, but this tool call's context has no approver \
         (its dispatcher did not call `ToolContext::with_approver`)"
    )]
    NoApprover {
        /// The kind of the request nobody could answer.
        kind: ApprovalKind,
    },
    /// The call was cancelled before the approver answered.
    #[error("the tool call was cancelled while waiting for approval")]
    Cancelled,
}
