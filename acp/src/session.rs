//! ACP session management.
//!
//! Each session represents an active conversation with the agent.

use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use aither_agent::Agent;
use aither_core::LanguageModel;
use aither_core::llm::tool::{CancelSignal, Canceller, ToolDefinition};
use uuid::Uuid;

use crate::SessionUpdateMapper;
use crate::protocol::{McpServer, SessionUpdate};

#[derive(Debug, Default)]
struct CancellationState {
    next_id: u64,
    pre_cancelled: bool,
    last_cancelled: bool,
    active: Vec<PromptScope>,
}

#[derive(Debug, Clone)]
struct PromptScope {
    id: u64,
    canceller: Arc<Canceller>,
}

/// Scoped cancellation token for a single [`AcpSession`] prompt turn.
///
/// Created before the turn begins so a concurrent `session/cancel`
/// notification or [`Cancellation::cancel`] call cancels this exact turn
/// without any start-of-turn `reset()` wiping out the request.
#[derive(Debug)]
pub struct PromptCancelGuard {
    inner: Arc<Mutex<CancellationState>>,
    id: u64,
    canceller: Arc<Canceller>,
}

impl PromptCancelGuard {
    pub(crate) fn is_cancelled(&self) -> bool {
        self.canceller.is_cancelled()
    }

    fn signal(&self) -> CancelSignal {
        self.canceller.signal()
    }
}

impl Drop for PromptCancelGuard {
    fn drop(&mut self) {
        if let Ok(mut state) = self.inner.lock() {
            state.active.retain(|scope| scope.id != self.id);
        }
    }
}

/// A session's cancellation coordinator, shared between the session and the
/// server's `session/cancel` handler.
///
/// Each prompt turn registers its own [`Canceller`] scope rather than resetting
/// a shared boolean at turn start, eliminating the race where a concurrent
/// `session/cancel` arrives right before `prompt` begins.
#[derive(Debug, Clone)]
pub struct Cancellation {
    inner: Arc<Mutex<CancellationState>>,
}

impl Cancellation {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(CancellationState::default())),
        }
    }

    /// Requests that the running turn end (or, if called directly before a turn
    /// starts, marks the impending turn cancelled).
    pub fn cancel(&self) {
        let scopes = {
            let mut state = self.inner.lock().expect("cancellation mutex poisoned");
            state.last_cancelled = true;
            if state.active.is_empty() {
                state.pre_cancelled = true;
                Vec::new()
            } else {
                state.active.clone()
            }
        };
        for scope in scopes {
            scope.canceller.cancel();
        }
    }

    /// Cancels only turns currently in flight on this session, ignoring late
    /// `session/cancel` notifications that arrive after a turn has already
    /// finished so they cannot poison a subsequent turn.
    pub(crate) fn cancel_active(&self) {
        let scopes = {
            let mut state = self.inner.lock().expect("cancellation mutex poisoned");
            if state.active.is_empty() {
                return;
            }
            state.last_cancelled = true;
            state.active.clone()
        };
        for scope in scopes {
            scope.canceller.cancel();
        }
    }

    /// Registers a new prompt scope before awaiting session resources or model
    /// streaming.
    pub(crate) fn begin_prompt(&self) -> PromptCancelGuard {
        let canceller = Arc::new(Canceller::default());
        let (id, was_pre_cancelled) = {
            let mut state = self.inner.lock().expect("cancellation mutex poisoned");
            state.next_id += 1;
            let id = state.next_id;
            let pre_cancelled = std::mem::take(&mut state.pre_cancelled);
            state.last_cancelled = pre_cancelled;
            state.active.push(PromptScope {
                id,
                canceller: Arc::clone(&canceller),
            });
            drop(state);
            (id, pre_cancelled)
        };
        if was_pre_cancelled {
            canceller.cancel();
        }
        PromptCancelGuard {
            inner: Arc::clone(&self.inner),
            id,
            canceller,
        }
    }

    fn is_cancelled(&self) -> bool {
        let state = self.inner.lock().expect("cancellation mutex poisoned");
        state.last_cancelled
            || state
                .active
                .iter()
                .any(|scope| scope.canceller.is_cancelled())
    }

    /// Clears a previous cancellation record when no turn is active.
    fn reset(&self) {
        let mut state = self.inner.lock().expect("cancellation mutex poisoned");
        state.pre_cancelled = false;
        if state.active.is_empty() {
            state.last_cancelled = false;
        }
    }
}

/// An active ACP session.
///
/// Each session wraps a conversation context and can process prompts.
#[derive(Debug)]
pub struct AcpSession<LLM: LanguageModel + 'static> {
    id: String,
    cwd: PathBuf,
    mcp_servers: Vec<McpServer>,
    cancelled: Cancellation,
    mapper: SessionUpdateMapper,
    agent: Agent<LLM>,
}

impl<LLM: LanguageModel + 'static> AcpSession<LLM> {
    /// Create a new session that drives the given agent.
    #[must_use]
    pub fn new(cwd: PathBuf, mcp_servers: Vec<McpServer>, agent: Agent<LLM>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            cwd,
            mcp_servers,
            cancelled: Cancellation::new(),
            mapper: SessionUpdateMapper::default(),
            agent,
        }
    }

    /// Attaches a custom [`SessionUpdateMapper`] configured with tool metadata.
    #[must_use]
    pub fn with_mapper(mut self, mapper: SessionUpdateMapper) -> Self {
        self.mapper = mapper;
        self
    }

    /// Populates the session's [`SessionUpdateMapper`] with optional display
    /// metadata read from `definitions`.
    #[must_use]
    pub fn with_tool_definitions<'a>(
        mut self,
        definitions: impl IntoIterator<Item = &'a ToolDefinition>,
    ) -> Self {
        self.mapper = self.mapper.with_definitions(definitions);
        self
    }

    /// Get the session ID.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the working directory.
    #[must_use]
    pub const fn cwd(&self) -> &PathBuf {
        &self.cwd
    }

    /// Get MCP server specifications.
    #[must_use]
    pub fn mcp_servers(&self) -> &[McpServer] {
        &self.mcp_servers
    }

    /// Check if the session's current or most recent turn was cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.is_cancelled()
    }

    /// Cancel the current operation.
    pub fn stop(&self) {
        self.cancelled.cancel();
    }

    /// Reset cancellation flag.
    pub fn reset(&self) {
        self.cancelled.reset();
    }

    /// The shared cancellation coordinator for this session.
    ///
    /// The server tracks one per session so a `session/cancel` notification
    /// can reach a session even while its prompt turn holds it out of the
    /// session map.
    pub(crate) fn cancellation(&self) -> Cancellation {
        self.cancelled.clone()
    }

    /// Run the agent over `prompt`, reporting progress through `on_update`.
    ///
    /// Each agent event is translated into the corresponding ACP session update
    /// as it arrives, so the client sees text and tool activity while the turn
    /// is still running. Returns when the turn completes, the client cancels
    /// via [`Self::stop`], or the agent errors.
    ///
    /// # Errors
    ///
    /// Returns the agent's error message if the turn could not be completed.
    pub async fn prompt<F>(&mut self, prompt: &str, on_update: F) -> Result<(), String>
    where
        F: FnMut(SessionUpdate),
    {
        let guard = self.cancelled.begin_prompt();
        self.prompt_with_guard(&guard, prompt, on_update).await
    }

    pub(crate) async fn prompt_with_guard<F>(
        &mut self,
        guard: &PromptCancelGuard,
        prompt: &str,
        mut on_update: F,
    ) -> Result<(), String>
    where
        F: FnMut(SessionUpdate),
    {
        let signal = guard.signal();
        // A cancelled turn ends like a completed one.
        let _ = self
            .agent
            .run(prompt, [])
            .drive(
                (&mut self.mapper, &mut on_update),
                signal.cancelled(),
                |(mapper, on_update), timed| async move {
                    if let Some(update) = mapper.map(&timed) {
                        on_update(update);
                    }
                    ControlFlow::Continue((mapper, on_update))
                },
            )
            .await
            .map_err(|err| err.to_string())?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;
    use aither_core::llm::{Event, LLMRequest, model::Profile};
    use futures_lite::Stream;

    #[derive(Debug)]
    struct NeverError;

    impl std::fmt::Display for NeverError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("never")
        }
    }

    impl std::error::Error for NeverError {}

    impl aither_core::llm::Classify for NeverError {
        fn failure(&self) -> aither_core::llm::Failure {
            aither_core::llm::Failure::permanent()
        }
    }

    /// A provider whose stream never yields unless `fast` is set.
    #[derive(Clone, Default)]
    struct SwitchableModel {
        fast: Arc<AtomicBool>,
    }

    impl LanguageModel for SwitchableModel {
        type Error = NeverError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, NeverError>> + Send {
            let fast = self.fast.load(Ordering::Acquire);
            futures_lite::stream::once_future(async move {
                if fast {
                    Ok(Event::Text("ok".to_string()))
                } else {
                    std::future::pending::<Result<Event, NeverError>>().await
                }
            })
        }

        fn profile(&self) -> impl Future<Output = Result<Profile, NeverError>> + Send {
            std::future::ready(Ok(Profile::new(
                "switch",
                "test",
                "switch",
                "Switchable model",
                128_000,
            )))
        }
    }

    #[tokio::test]
    async fn cancelling_ends_a_turn_blocked_on_a_provider_that_never_yields() {
        let mut session = AcpSession::new(
            PathBuf::from("."),
            Vec::new(),
            Agent::new(SwitchableModel::default()),
        );
        let cancellation = session.cancellation();

        let turn = tokio::spawn(async move {
            let result = session.prompt("hello", |_| {}).await;
            (session, result)
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        cancellation.cancel();

        let (session, result) = tokio::time::timeout(std::time::Duration::from_secs(5), turn)
            .await
            .expect("the cancelled turn must return")
            .expect("turn task");
        assert_eq!(result, Ok(()));
        assert!(session.is_cancelled());
    }

    #[tokio::test]
    async fn cancel_before_prompt_polls_is_not_wiped_and_subsequent_turn_succeeds() {
        let model = SwitchableModel::default();
        let mut session =
            AcpSession::new(PathBuf::from("."), Vec::new(), Agent::new(model.clone()));
        let cancellation = session.cancellation();

        // Cancel BEFORE `prompt` even begins polling — previously `self.reset()`
        // at the top of `prompt` clobbered this cancellation and hung forever.
        cancellation.cancel();
        let first = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            session.prompt("blocked", |_| {}),
        )
        .await
        .expect("pre-cancelled prompt must return immediately");
        assert_eq!(first, Ok(()));
        assert!(session.is_cancelled());

        // The next turn starts with a fresh scope and completes normally.
        model.fast.store(true, Ordering::Release);
        let mut chunks = Vec::new();
        let second = session
            .prompt("follow-up", |update| chunks.push(update))
            .await;
        assert_eq!(second, Ok(()));
        assert!(!session.is_cancelled());
        assert!(!chunks.is_empty());
    }
}
