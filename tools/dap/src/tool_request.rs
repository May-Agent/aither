//! Request/response channel between [`crate::DebugTool`] and the host broker
//! that owns live debug sessions.

use async_channel::{Receiver, Sender};

/// A request carrying tool arguments and a oneshot response sender.
#[derive(Debug)]
pub struct ToolRequest<Args, Response> {
    /// Tool arguments provided by the agent.
    pub args: Args,
    response_tx: Sender<Response>,
}

impl<Args, Response> ToolRequest<Args, Response> {
    /// Responds to the request without blocking the host.
    ///
    /// # Errors
    ///
    /// Returns the response back if the requester already stopped waiting.
    pub fn respond(self, response: Response) -> Result<(), async_channel::TrySendError<Response>> {
        self.response_tx.try_send(response)
    }
}

/// Why a [`ToolRequestBroker::request`] call failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ToolRequestError {
    /// The host stopped listening for requests.
    #[error("debug tool request channel is unavailable")]
    Unavailable,
    /// The host dropped the request without answering.
    #[error("debug tool request was cancelled")]
    Cancelled,
}

/// Sends requests to the host broker and awaits its responses.
#[derive(Debug, Clone)]
pub struct ToolRequestBroker<Args, Response> {
    tx: Sender<ToolRequest<Args, Response>>,
}

impl<Args, Response> ToolRequestBroker<Args, Response> {
    /// Sends a request and awaits the host's response.
    ///
    /// # Errors
    ///
    /// Returns [`ToolRequestError::Unavailable`] if the queue was dropped or
    /// [`ToolRequestError::Cancelled`] if the host dropped the request unanswered.
    pub async fn request(&self, args: Args) -> Result<Response, ToolRequestError> {
        let (response_tx, response_rx) = async_channel::bounded(1);
        self.tx
            .send(ToolRequest { args, response_tx })
            .await
            .map_err(|_| ToolRequestError::Unavailable)?;

        response_rx
            .recv()
            .await
            .map_err(|_| ToolRequestError::Cancelled)
    }
}

/// Pending requests consumed by the host broker.
#[derive(Debug)]
pub struct ToolRequestQueue<Args, Response> {
    rx: Receiver<ToolRequest<Args, Response>>,
}

impl<Args, Response> ToolRequestQueue<Args, Response> {
    /// Awaits the next request, or returns `None` when all brokers are dropped.
    pub async fn next(&self) -> Option<ToolRequest<Args, Response>> {
        self.rx.recv().await.ok()
    }

    /// Returns `true` if no requests are queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rx.is_empty()
    }
}

/// Creates a new request broker/queue pair.
#[must_use]
pub fn channel<Args, Response>() -> (
    ToolRequestBroker<Args, Response>,
    ToolRequestQueue<Args, Response>,
) {
    let (tx, rx) = async_channel::unbounded();
    (ToolRequestBroker { tx }, ToolRequestQueue { rx })
}
