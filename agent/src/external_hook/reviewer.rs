//! Reviewers: external hooks on the permission path.

use std::sync::Arc;

use aither_sandbox::permission::{
    CommandLocation, LocationApproval, PermissionError, PermissionHandler, PermissionRequest,
    Review, TerminalMode,
};
use aither_sandbox::{CommandRole, DomainApprover, DomainRequest, ScriptLanguage, SshServer};
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

/// What a reviewer is asked about: a permission beyond the sandbox, or a
/// network domain the sandbox's lists leave open.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(untagged)]
pub enum ReviewSubject<'a> {
    /// A command escalation or an SSH location.
    Permission(PermissionRequest<'a>),
    /// A network domain.
    Domain(DomainSubject<'a>),
}

/// What a reviewer reads on stdin.
///
/// The role of the agent that asked, and the request, which is either a command escalation (`"kind":"command"` with
/// `mode`, `script`, its `language` and the `location` it runs at, `{"kind":"local"}` or
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
    async fn review(&self, request: ReviewSubject<'_>) -> Review {
        let request = ReviewRequest {
            role: self.role(),
            request,
        };
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

/// A permission handler with the host's reviewers in front of the existing
/// route.
///
/// The reviewers are the handler's [`review`](PermissionHandler::review) step,
/// which the terminal runs before it asks `inner` anything: the first
/// reviewer to approve grants the request without reaching `inner`, the first
/// to deny refuses it, and a request every reviewer passes goes to `inner`
/// unchanged, including its wait query. A command that needs a mode beyond
/// the sandbox is reviewed; network domains are reviewed by
/// [`ReviewedDomains`].
/// A subagent's terminal gets [`for_subagent`](PermissionHandler::for_subagent),
/// the same reviewers labelled with the subagent role, over the same `inner`
/// route.
///
/// The reviewers are those of the [`ExternalHooks`] scope the handler is made
/// with, so they follow the host's live changes like every other hook.
#[derive(Debug)]
pub struct ReviewedPermission<P> {
    inner: Arc<P>,
    hooks: ExternalHooks,
}

impl<P> ReviewedPermission<P> {
    /// The main agent's handler: the reviewers of `hooks` in front of
    /// `inner`.
    #[must_use]
    pub fn new(inner: P, hooks: ExternalHooks) -> Self {
        Self {
            inner: Arc::new(inner),
            hooks,
        }
    }
}

impl<P: PermissionHandler + 'static> PermissionHandler for ReviewedPermission<P> {
    /// Runs the chain; the first reviewer to decide ends it.
    async fn review(&self, request: &PermissionRequest<'_>) -> Review {
        self.hooks.review(ReviewSubject::Permission(*request)).await
    }

    async fn check(
        &self,
        mode: TerminalMode,
        script: &str,
        language: ScriptLanguage,
        location: CommandLocation<'_>,
    ) -> Result<bool, PermissionError> {
        self.inner.check(mode, script, language, location).await
    }

    fn will_wait_for_approval(
        &self,
        mode: TerminalMode,
        script: &str,
        language: ScriptLanguage,
        location: CommandLocation<'_>,
    ) -> impl Future<Output = bool> + Send {
        self.inner
            .will_wait_for_approval(mode, script, language, location)
    }

    fn for_subagent(&self) -> Option<Self> {
        Some(Self {
            inner: Arc::clone(&self.inner),
            hooks: self.hooks.for_subagent(),
        })
    }

    fn check_location(
        &self,
        server: &SshServer,
        role: CommandRole,
    ) -> impl Future<Output = Result<LocationApproval, PermissionError>> + Send {
        self.inner.check_location(server, role)
    }

    fn will_wait_for_location(
        &self,
        server: &SshServer,
        role: CommandRole,
    ) -> impl Future<Output = bool> + Send {
        self.inner.will_wait_for_location(server, role)
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
        match self.hooks.review(subject).await {
            Review::Approved => true,
            Review::Denied(_) => false,
            Review::Passed => self.inner.approve(request).await,
        }
    }
}
