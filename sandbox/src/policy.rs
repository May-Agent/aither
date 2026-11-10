//! Script policies: the embedder's rules on what a script may run.
//!
//! The sandbox decides where a script runs and what it can reach; whether a
//! script should run at all — a toolchain preference, a ban on killing by
//! name, a rule against polling, an approval for a server — is the
//! embedder's. Every [`ScriptPolicy`] registered on the
//! [`ToolRegistryBuilder`] sees each script before it runs, parsed when it
//! is shell, with the directory it starts in, the call it came from and the
//! policy's own state for the agent that runs it, and decides [`Pass`],
//! [`Allow`], [`Ask`] or [`Refuse`]. With no policy the sandbox refuses
//! nothing, and a script that stays inside the sandbox on the home
//! environment runs.
//!
//! A script that crosses the sandbox boundary — it runs unsandboxed or on a
//! remote location — runs only after an [`Allow`] or an approved [`Ask`];
//! with only [`Pass`]es it is asked.
//!
//! [`ToolRegistryBuilder`]: crate::ToolRegistryBuilder
//! [`Pass`]: ScriptDecision::Pass
//! [`Allow`]: ScriptDecision::Allow
//! [`Ask`]: ScriptDecision::Ask
//! [`Refuse`]: ScriptDecision::Refuse

use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use aither_core::llm::tool::{ApprovalRequest, CallIdentity};

use crate::container::ContainerPath;
use crate::permission::TerminalMode;
use crate::script_language::ScriptLanguage;
use crate::shell_syntax::{ShellParseError, ShellScript};
use crate::terminal::DirectoryResetReason;

/// How a script came to the terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptOrigin {
    /// A `terminal` call the caller waits on.
    Foreground,
    /// A `terminal` call that detaches the script (`background: true`).
    Background,
    /// A [`TerminalRunner::run`](crate::TerminalRunner::run) from a
    /// terminal-bound command: the embedder's own script, waited on.
    Run,
}

/// Whether a script runs inside the sandbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SandboxMode {
    /// Inside the sandbox.
    Sandboxed,
    /// Outside it, with the access of the user who runs the agent.
    Unsandboxed,
}

impl From<TerminalMode> for SandboxMode {
    fn from(mode: TerminalMode) -> Self {
        match mode {
            TerminalMode::Sandboxed => Self::Sandboxed,
            TerminalMode::Unsafe => Self::Unsandboxed,
        }
    }
}

/// A directory a script starts or ends in, as the backend that runs it
/// names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptDir {
    /// A directory of the host the agent runs on.
    Host(PathBuf),
    /// A directory in the container the script runs in.
    Container {
        /// The directory as the script sees it.
        inside: ContainerPath,
        /// The host side of that directory, for resolving what the script
        /// names against the host's files.
        host: PathBuf,
    },
    /// The script runs on a remote server, whose directories the sandbox
    /// does not follow.
    Remote {
        /// The name of the server, as the call gave it as its `location`.
        server: String,
    },
}

impl ScriptDir {
    /// The directory on the host the script's relative paths resolve
    /// against; `None` for a script that runs on a remote server.
    #[must_use]
    pub fn host(&self) -> Option<&std::path::Path> {
        match self {
            Self::Host(dir) | Self::Container { host: dir, .. } => Some(dir),
            Self::Remote { .. } => None,
        }
    }
}

/// A script about to run, as a [`ScriptPolicy`] sees it.
#[derive(Debug, Clone, Copy)]
pub struct ScriptCheck<'a> {
    /// The id the script's task gets when it runs; the [`ScriptEnd`] that
    /// reports how it ended carries the same id.
    pub task_id: &'a str,
    /// The script as written.
    pub script: &'a str,
    /// The language it is written in.
    pub language: ScriptLanguage,
    /// The script parsed as shell: `None` for a script in another language,
    /// an error when it is not valid shell.
    pub shell: Option<Result<&'a ShellScript, &'a ShellParseError>>,
    /// How the script came to the terminal.
    pub origin: ScriptOrigin,
    /// Whether the script runs inside the sandbox.
    pub sandbox: SandboxMode,
    /// The configured server the script runs on; `None` for the home
    /// environment.
    pub location: Option<&'a str>,
    /// The directory the script starts in, fixed when the script was
    /// admitted: for a foreground script of the main agent the directory the
    /// shell carried from the previous one, for a run the directory it
    /// names, otherwise the terminal's own directory. The launch starts in
    /// the same directory.
    pub start: &'a ScriptDir,
    /// The tool call that carries the script and the agent that made it,
    /// when the caller identified one.
    pub call: Option<&'a CallIdentity>,
}

/// How a script a [`ScriptPolicy`] was asked about ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptOutcome {
    /// It never started: a policy, the command gate or an approval refused
    /// it, or its launch could not be prepared.
    NotRun,
    /// It started but could not be run to an exit status.
    Failed,
    /// It exited with `code`.
    Exited {
        /// The exit status.
        code: i32,
    },
}

/// What a script did to the directory the shell carries to the next
/// foreground script.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Carry {
    /// The shell ended in another directory, which the next foreground
    /// script starts in.
    Moved,
    /// The directory the shell ended in is not carried; the next script
    /// starts where this one did.
    Reset(DirectoryResetReason),
    /// The shell ended where the script started.
    Unchanged,
    /// The script carries nothing: it ran in the background, failed to
    /// start or was refused, or the terminal carries no directory.
    NotCarried,
}

/// The end of a script a [`ScriptPolicy`] was asked about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScriptEnd<'a> {
    /// The task id the [`ScriptCheck`] carried.
    pub task_id: &'a str,
    /// How it ended.
    pub outcome: ScriptOutcome,
    /// The directory the terminal's next foreground script starts in now
    /// this one has ended, settled under the lock that updates the carried
    /// directory: the end of one script is the start of the next, whatever
    /// calls run in parallel.
    pub end: &'a ScriptDir,
    /// What the script did to the carried directory.
    pub carry: Carry,
}

/// A script a [`ScriptPolicy`] refused; the model reads [`Self::text`] as
/// the call's result.
///
/// Tools return it as their error, so an agent loop tells a policy's
/// refusal from any other failure by downcasting.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{text}")]
pub struct ScriptRefusal {
    text: String,
}

impl ScriptRefusal {
    /// A refusal the model reads as `text`.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    /// The text the model reads.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// What a [`ScriptPolicy`] decided about one script.
///
/// When several policies decide, the strictest wins: [`Refuse`], then
/// [`Ask`], then [`Allow`], then [`Pass`].
///
/// [`Pass`]: Self::Pass
/// [`Allow`]: Self::Allow
/// [`Ask`]: Self::Ask
/// [`Refuse`]: Self::Refuse
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptDecision {
    /// The policy has no opinion on the script.
    Pass,
    /// The policy approves the script: one that crosses the sandbox
    /// boundary runs without being asked.
    Allow,
    /// The script runs only if the approver approves `request`.
    Ask(ApprovalRequest),
    /// The script does not run; the model reads the refusal in its place.
    Refuse(ScriptRefusal),
}

impl ScriptDecision {
    const fn strictness(&self) -> u8 {
        match self {
            Self::Pass => 0,
            Self::Allow => 1,
            Self::Ask(_) => 2,
            Self::Refuse(_) => 3,
        }
    }

    /// Folds `decisions`, in registration order, into the strictest; of
    /// equally strict ones the first stands, so the model reads the first
    /// refusal and the approver is asked the first question.
    #[must_use]
    pub fn strictest(decisions: impl IntoIterator<Item = Self>) -> Self {
        decisions
            .into_iter()
            .fold(Self::Pass, |strictest, decision| {
                if decision.strictness() > strictest.strictness() {
                    decision
                } else {
                    strictest
                }
            })
    }
}

/// Decides whether a script may run, before it does.
///
/// Every script of the `terminal` tool, the main agent's and every
/// subagent's, and every [`TerminalRunner::run`](crate::TerminalRunner::run),
/// is put to every policy of its registry ahead of the command gate and the
/// approval it needs. The policies of one registry decide concurrently and
/// the strictest decision wins.
///
/// A policy keeps what it remembers in its [`State`](Self::State): every
/// agent's terminal, each subagent's included, starts with a fresh one and
/// shares it with nobody else. Calls run in parallel, so the state is shared
/// by reference and the policy chooses how to synchronise it.
pub trait ScriptPolicy: Send + Sync + 'static {
    /// What the policy remembers for one agent.
    type State: Default + Send + Sync + 'static;

    /// Decides on `script`.
    fn check(
        &self,
        script: &ScriptCheck<'_>,
        state: &Self::State,
    ) -> impl Future<Output = ScriptDecision> + Send;

    /// Told once how every script put to [`Self::check`] ended, whichever
    /// policy decided what: refused, never started, failed or exited. The
    /// result of a foreground script reaches the caller only after this
    /// returns, so a state it records holds for the agent's next call; a
    /// background script's end is told after its call returned.
    fn ended(&self, end: &ScriptEnd<'_>, state: &Self::State) -> impl Future<Output = ()> + Send {
        let _ = (end, state);
        std::future::ready(())
    }
}

/// Object-safe twin of [`ScriptPolicy`]: the policy, before it is bound to
/// an agent.
trait ScriptPolicyImpl: Send + Sync {
    fn bind(self: Arc<Self>) -> Arc<dyn BoundPolicyImpl>;
}

impl<T: ScriptPolicy> ScriptPolicyImpl for T {
    fn bind(self: Arc<Self>) -> Arc<dyn BoundPolicyImpl> {
        Arc::new(Bound {
            policy: self,
            state: T::State::default(),
        })
    }
}

/// A policy with one agent's state.
struct Bound<T: ScriptPolicy> {
    policy: Arc<T>,
    state: T::State,
}

trait BoundPolicyImpl: Send + Sync {
    fn check<'a>(
        &'a self,
        script: &'a ScriptCheck<'a>,
    ) -> Pin<Box<dyn Future<Output = ScriptDecision> + Send + 'a>>;

    fn ended<'a>(&'a self, end: &'a ScriptEnd<'a>)
    -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

impl<T: ScriptPolicy> BoundPolicyImpl for Bound<T> {
    fn check<'a>(
        &'a self,
        script: &'a ScriptCheck<'a>,
    ) -> Pin<Box<dyn Future<Output = ScriptDecision> + Send + 'a>> {
        Box::pin(self.policy.check(script, &self.state))
    }

    fn ended<'a>(
        &'a self,
        end: &'a ScriptEnd<'a>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(self.policy.ended(end, &self.state))
    }
}

/// A type-erased [`ScriptPolicy`].
#[derive(Clone)]
pub struct AnyScriptPolicy(Arc<dyn ScriptPolicyImpl>);

impl AnyScriptPolicy {
    /// Erases `policy`.
    #[must_use]
    pub fn new(policy: impl ScriptPolicy) -> Self {
        Self(Arc::new(policy))
    }
}

impl fmt::Debug for AnyScriptPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyScriptPolicy")
    }
}

/// A registry's policies, each bound to the state of the agent the registry
/// serves.
#[derive(Clone, Default)]
pub struct ScriptPolicies {
    policies: Arc<[AnyScriptPolicy]>,
    bound: Arc<[Arc<dyn BoundPolicyImpl>]>,
}

impl ScriptPolicies {
    /// `policies`, bound to a fresh agent.
    #[must_use]
    pub fn new(policies: Vec<AnyScriptPolicy>) -> Self {
        let policies: Arc<[AnyScriptPolicy]> = policies.into();
        Self {
            bound: Self::bind(&policies),
            policies,
        }
    }

    fn bind(policies: &[AnyScriptPolicy]) -> Arc<[Arc<dyn BoundPolicyImpl>]> {
        policies
            .iter()
            .map(|policy| policy.0.clone().bind())
            .collect()
    }

    /// The same policies, bound to another agent with fresh state.
    #[must_use]
    pub fn for_agent(&self) -> Self {
        Self {
            policies: self.policies.clone(),
            bound: Self::bind(&self.policies),
        }
    }

    /// Whether no policy is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bound.is_empty()
    }

    /// Puts `script` to every policy; the strictest decision wins, the
    /// first of equally strict ones in registration order.
    pub async fn check(&self, script: &ScriptCheck<'_>) -> ScriptDecision {
        let decisions =
            futures::future::join_all(self.bound.iter().map(|policy| policy.check(script))).await;
        ScriptDecision::strictest(decisions)
    }

    /// Tells every policy how a checked script ended, and where the
    /// terminal's next foreground script starts.
    pub async fn ended(
        &self,
        task_id: &str,
        outcome: ScriptOutcome,
        end: &ScriptDir,
        carry: Carry,
    ) {
        if self.is_empty() {
            return;
        }
        let end = ScriptEnd {
            task_id,
            outcome,
            end,
            carry,
        };
        futures::future::join_all(self.bound.iter().map(|policy| policy.ended(&end))).await;
    }
}

impl fmt::Debug for ScriptPolicies {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScriptPolicies")
            .field("policies", &self.policies.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use aither_core::llm::tool::ApprovalKind;

    use super::*;

    /// Refuses every script after the first `allowed` its agent ran.
    struct Quota {
        allowed: usize,
    }

    impl ScriptPolicy for Quota {
        type State = AtomicUsize;

        fn check(
            &self,
            _script: &ScriptCheck<'_>,
            state: &AtomicUsize,
        ) -> impl Future<Output = ScriptDecision> + Send {
            std::future::ready(if state.fetch_add(1, Ordering::SeqCst) < self.allowed {
                ScriptDecision::Allow
            } else {
                ScriptDecision::Refuse(ScriptRefusal::new("over quota"))
            })
        }
    }

    /// Decides the same way on every script.
    struct Always(ScriptDecision);

    impl ScriptPolicy for Always {
        type State = ();

        fn check(
            &self,
            _script: &ScriptCheck<'_>,
            (): &(),
        ) -> impl Future<Output = ScriptDecision> + Send {
            std::future::ready(self.0.clone())
        }
    }

    fn refuse(text: &str) -> ScriptDecision {
        ScriptDecision::Refuse(ScriptRefusal::new(text))
    }

    fn ask(subject: &str) -> ScriptDecision {
        ScriptDecision::Ask(ApprovalRequest::new(ApprovalKind::new("test"), subject))
    }

    fn check(task_id: &str) -> ScriptCheck<'_> {
        static START: std::sync::LazyLock<ScriptDir> =
            std::sync::LazyLock::new(|| ScriptDir::Host(PathBuf::from("/")));
        ScriptCheck {
            task_id,
            script: "true",
            language: ScriptLanguage::Shell,
            shell: None,
            origin: ScriptOrigin::Foreground,
            sandbox: SandboxMode::Sandboxed,
            location: None,
            start: &START,
            call: None,
        }
    }

    fn policies(decisions: impl IntoIterator<Item = ScriptDecision>) -> ScriptPolicies {
        ScriptPolicies::new(
            decisions
                .into_iter()
                .map(|decision| AnyScriptPolicy::new(Always(decision)))
                .collect(),
        )
    }

    #[tokio::test]
    async fn each_agent_binds_fresh_state() {
        let main = ScriptPolicies::new(vec![AnyScriptPolicy::new(Quota { allowed: 1 })]);
        let view = main.clone();
        let subagent = main.for_agent();
        assert_eq!(main.check(&check("a")).await, ScriptDecision::Allow);
        assert_eq!(main.check(&check("b")).await, refuse("over quota"));
        assert!(
            matches!(view.check(&check("c")).await, ScriptDecision::Refuse(_)),
            "a view of the main agent's registry shares its state"
        );
        assert_eq!(
            subagent.check(&check("d")).await,
            ScriptDecision::Allow,
            "the subagent's state is its own"
        );
    }

    #[tokio::test]
    async fn the_strictest_decision_wins() {
        let order = [
            ScriptDecision::Pass,
            ScriptDecision::Allow,
            ask("first question"),
            refuse("first refusal"),
        ];
        // Whatever the registration order, the stricter of two stands.
        for (weak, strong) in order
            .iter()
            .enumerate()
            .flat_map(|(i, weak)| order[i + 1..].iter().map(move |strong| (weak, strong)))
        {
            for pair in [[weak, strong], [strong, weak]] {
                let policies = policies(pair.into_iter().cloned());
                assert_eq!(&policies.check(&check("a")).await, strong);
            }
        }
        assert_eq!(
            policies([ScriptDecision::Allow, ScriptDecision::Pass])
                .check(&check("b"))
                .await,
            ScriptDecision::Allow
        );
    }

    #[tokio::test]
    async fn the_first_of_equally_strict_decisions_in_registration_order_stands() {
        let policies = policies([
            ScriptDecision::Allow,
            ask("first question"),
            ask("second question"),
            refuse("first refusal"),
            refuse("second refusal"),
        ]);
        assert_eq!(policies.check(&check("a")).await, refuse("first refusal"));
        let asking = super::ScriptPolicies::new(vec![
            AnyScriptPolicy::new(Always(ask("first question"))),
            AnyScriptPolicy::new(Always(ask("second question"))),
        ]);
        assert_eq!(asking.check(&check("b")).await, ask("first question"));
        assert!(ScriptPolicies::default().is_empty());
        assert_eq!(
            ScriptPolicies::default().check(&check("c")).await,
            ScriptDecision::Pass
        );
    }
}
