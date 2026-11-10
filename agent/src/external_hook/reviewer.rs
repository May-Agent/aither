//! Reviewers: external hooks on the permission path.

use aither_core::llm::AgentIdentity;
use aither_sandbox::{
    DomainApprover, DomainRequest, Interpreter, SandboxMode, ScriptCheck, ScriptDecision,
    ScriptPolicy, ScriptRefusal,
};
use serde::{Deserialize, Serialize};

use super::lifecycle::ExternalHooks;
use super::live::Run;
use super::runner::{HookCommand, HookRole};

/// A reviewer as the host configures it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewerConfig {
    /// Names the reviewer in the denials it raises and in diagnostics.
    pub name: String,
    /// What to run and for how long.
    #[serde(flatten)]
    pub command: HookCommand,
}

/// A network domain a sandbox's lists leave open, as a reviewer reads it.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DomainSubject<'a> {
    /// A connection to `domain` on `port`.
    Domain {
        /// The host being connected to.
        domain: &'a str,
        /// The port being connected to.
        port: u16,
    },
}

/// Where a reviewed script runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScriptLocation<'a> {
    /// The home environment.
    Local,
    /// A configured SSH server.
    Ssh {
        /// The server's name, as the script gave it as its `location`.
        server: &'a str,
    },
}

/// A script that crosses the sandbox boundary, as a reviewer reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CommandSubject<'a> {
    /// A command that runs outside the sandbox or on a server.
    Command {
        /// Whether the command runs inside the sandbox.
        sandbox: SandboxMode,
        /// The script to run.
        script: &'a str,
        /// The interpreter the script is written for.
        interpreter: Interpreter,
        /// Where the command runs.
        location: ScriptLocation<'a>,
    },
}

/// What a reviewer is asked about: a script that crosses the sandbox
/// boundary, or a network domain the sandbox's lists leave open.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(untagged)]
pub enum ReviewSubject<'a> {
    /// A script that runs outside the sandbox or on a server.
    Command(CommandSubject<'a>),
    /// A network domain.
    Domain(DomainSubject<'a>),
}

/// How the reviewers decided a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Review {
    /// Granted; nothing else is asked.
    Approved,
    /// Refused, with the reason the model reads.
    Denied(String),
    /// Not decided; the request goes on to the next route.
    Passed,
}

/// What a reviewer reads on stdin.
///
/// The role of the agent that asked, and the request, which is either a command escalation (`"kind":"command"` with
/// `sandbox`, `script`, its `interpreter` and the `location` it runs at, `{"kind":"local"}` or
/// `{"kind":"ssh","server":…}`) or a network domain (`"kind":"domain"` with `domain`
/// and `port`).
#[derive(Debug, Serialize)]
pub struct ReviewRequest<'a> {
    /// The agent whose request it is: the main agent or a subagent.
    pub role: HookRole,
    /// What is being asked for.
    #[serde(flatten)]
    pub request: ReviewSubject<'a>,
}

/// What a reviewer prints on stdout. Printing nothing is a pass. Any field
/// beyond the decision's own makes the answer invalid (the variants are
/// struct variants so that serde enforces this).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReviewAnswer {
    /// Grant the request without asking anyone else.
    Approve {},
    /// Refuse the request; the reason reaches the model.
    Deny {
        /// Why the request is refused.
        reason: String,
    },
    /// Leave the request to the next reviewer.
    Pass {},
}

impl ExternalHooks {
    /// Runs the reviewers in order until one approves or denies.
    ///
    /// A reviewer that fails to run, runs past its deadline or prints an
    /// invalid answer is replaced by its previous version when it has one;
    /// when no working version is left the user is asked to fix it. Where the
    /// user cannot be asked the failure is reported through `tracing` and
    /// counts as a pass, so a failure never grants anything.
    pub(super) async fn review(&self, role: HookRole, request: ReviewSubject<'_>) -> Review {
        let request = ReviewRequest { role, request };
        let set = self.live().set();
        for reviewer in &set.reviewers {
            let answer = self
                .live()
                .run_hook(
                    self.user(),
                    reviewer,
                    &request,
                    |answer: Option<ReviewAnswer>| Ok(answer.unwrap_or(ReviewAnswer::Pass {})),
                )
                .await;
            match answer {
                Run::Answered(ReviewAnswer::Approve {}) => return Review::Approved,
                Run::Answered(ReviewAnswer::Deny { reason }) => {
                    return Review::Denied(self.notices().reviewer_denial(
                        &crate::ReviewerDenial {
                            reviewer: &reviewer.config.name,
                            reason: &reason,
                        },
                    ));
                }
                Run::Answered(ReviewAnswer::Pass {}) | Run::Gone => {}
                Run::Failed(failure) => tracing::warn!(
                    reviewer = %reviewer.config.name,
                    %failure,
                    "permission reviewer failed; it counts as a pass"
                ),
            }
        }
        Review::Passed
    }
}

/// The host's reviewers as a [`ScriptPolicy`].
///
/// Every script that crosses the sandbox boundary, unsandboxed or on a
/// server, goes to the reviewers in order: the first to approve allows it
/// without asking the approver, the first to deny refuses it with the reason
/// the model reads, and a script every reviewer passes is left to the other
/// policies and the approver. Scripts inside the sandbox at home are not
/// reviewed; network domains are reviewed by [`ReviewedDomains`].
///
/// A reviewer reads the role of the agent that made the call, so one policy
/// serves the main agent and its subagents. The reviewers are those of the
/// [`ExternalHooks`] scope the policy is made with, so they follow the
/// host's live changes like every other hook.
#[derive(Debug, Clone)]
pub struct ReviewedScripts {
    hooks: ExternalHooks,
}

impl ReviewedScripts {
    /// The reviewers of `hooks` as a script policy.
    #[must_use]
    pub const fn new(hooks: ExternalHooks) -> Self {
        Self { hooks }
    }
}

impl ScriptPolicy for ReviewedScripts {
    type State = ();

    async fn check(&self, script: &ScriptCheck<'_>, (): &()) -> ScriptDecision {
        if script.sandbox == SandboxMode::Sandboxed && script.location.is_none() {
            return ScriptDecision::Pass;
        }
        let role = script
            .call
            .map_or(self.hooks.role(), |call| match call.agent() {
                AgentIdentity::Main => HookRole::Main,
                AgentIdentity::Subagent(_) => HookRole::Subagent,
            });
        let subject =
            ReviewSubject::Command(CommandSubject::Command {
                sandbox: script.sandbox,
                script: script.script,
                interpreter: script.interpreter,
                location: script.location.map_or(ScriptLocation::Local, |server| {
                    ScriptLocation::Ssh { server }
                }),
            });
        match self.hooks.review(role, subject).await {
            Review::Approved => ScriptDecision::Allow,
            Review::Denied(reason) => ScriptDecision::Refuse(ScriptRefusal::new(reason)),
            Review::Passed => ScriptDecision::Pass,
        }
    }
}

/// A [`DomainApprover`] with the host's reviewers in front of `inner`.
///
/// The first reviewer to approve a domain grants it without reaching
/// `inner`, the first to deny refuses it, and a domain every reviewer passes
/// goes to `inner`. Only domains the sandbox's network lists leave open reach
/// it, so a listed host is never reviewed. Whether an approval is remembered
/// is `inner`'s policy; a reviewer's approval is not passed to it.
#[derive(Debug, Clone)]
pub struct ReviewedDomains<A> {
    inner: A,
    hooks: ExternalHooks,
}

impl<A> ReviewedDomains<A> {
    /// The main agent's approver: the reviewers of `hooks` in front of
    /// `inner`.
    #[must_use]
    pub const fn new(inner: A, hooks: ExternalHooks) -> Self {
        Self { inner, hooks }
    }
}

impl<A: Clone> ReviewedDomains<A> {
    /// The same approver for a subagent, its reviewers labelled with the
    /// subagent role.
    #[must_use]
    pub fn for_subagent(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            hooks: self.hooks.for_subagent(),
        }
    }
}

impl<A: DomainApprover> DomainApprover for ReviewedDomains<A> {
    async fn approve(&self, request: &DomainRequest) -> bool {
        let subject = ReviewSubject::Domain(DomainSubject::Domain {
            domain: &request.host,
            port: request.port,
        });
        match self.hooks.review(self.hooks.role(), subject).await {
            Review::Approved => true,
            Review::Denied(_) => false,
            Review::Passed => self.inner.approve(request).await,
        }
    }
}
