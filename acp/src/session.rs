//! ACP session management.
//!
//! Each session represents an active conversation with the agent.

use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use aither_agent::Agent;
use aither_core::LanguageModel;
use uuid::Uuid;

use crate::SessionUpdateMapper;
use crate::protocol::{McpServer, SessionUpdate};

/// A session's cancellation request, shared between the session and the
/// server's `session/cancel` handler.
///
/// The flag records that the turn was cancelled; the channel wakes a turn
/// that is waiting, so cancellation does not depend on the turn producing
/// another event.
#[derive(Debug, Clone)]
pub struct Cancellation {
    flag: Arc<AtomicBool>,
    wake: (async_channel::Sender<()>, async_channel::Receiver<()>),
}

impl Cancellation {
    fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            wake: async_channel::bounded(1),
        }
    }

    /// Requests that the running turn end.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
        // A full channel already holds a pending wake-up.
        let _ = self.wake.0.try_send(());
    }

    fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// Clears a previous request before a new turn starts.
    fn reset(&self) {
        self.flag.store(false, Ordering::SeqCst);
        while self.wake.1.try_recv().is_ok() {}
    }

    /// Resolves once the turn is cancelled.
    async fn requested(&self) {
        let _ = self.wake.1.recv().await;
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

    /// Check if the session is cancelled.
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

    /// The shared cancellation request for this session.
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
    pub async fn prompt<F>(&mut self, prompt: &str, mut on_update: F) -> Result<(), String>
    where
        F: FnMut(SessionUpdate),
    {
        self.reset();

        let cancelled = &self.cancelled;
        // A cancelled turn ends like a completed one.
        let _ = self
            .agent
            .run(prompt, [])
            .drive(
                (&mut self.mapper, &mut on_update),
                cancelled.requested(),
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

    /// A provider whose stream never yields.
    #[derive(Clone)]
    struct HungModel;

    impl LanguageModel for HungModel {
        type Error = NeverError;

        fn respond(
            &self,
            _request: LLMRequest,
        ) -> impl Stream<Item = Result<Event, NeverError>> + Send {
            futures_lite::stream::pending()
        }

        fn profile(&self) -> impl Future<Output = Result<Profile, NeverError>> + Send {
            std::future::ready(Ok(Profile::new(
                "hung",
                "test",
                "hung",
                "Hung model",
                128_000,
            )))
        }
    }

    #[tokio::test]
    async fn cancelling_ends_a_turn_blocked_on_a_provider_that_never_yields() {
        let mut session = AcpSession::new(PathBuf::from("."), Vec::new(), Agent::new(HungModel));
        let cancellation = session.cancellation();

        let turn = tokio::spawn(async move {
            let result = session.prompt("hello", |_| {}).await;
            (session, result)
        });
        // Let the turn reach the provider before cancelling it.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        cancellation.cancel();

        let (session, result) = tokio::time::timeout(std::time::Duration::from_secs(5), turn)
            .await
            .expect("the cancelled turn must return")
            .expect("turn task");
        assert_eq!(result, Ok(()));
        assert!(session.is_cancelled());
    }
}
