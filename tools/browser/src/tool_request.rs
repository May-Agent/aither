//! Request/response channel between the browser tool and the host that
//! drives the browser.

use async_channel::{Receiver, Sender};

/// A request carrying tool arguments and a response sender.
#[derive(Debug)]
pub struct ToolRequest<Args, Response> {
    /// Tool arguments provided by the agent.
    pub args: Args,
    response_tx: Sender<Response>,
}

impl<Args, Response> ToolRequest<Args, Response> {
    /// Respond to the request without blocking the host.
    ///
    /// # Errors
    ///
    /// Returns the response back if the requester has already given up waiting.
    pub fn respond(self, response: Response) -> Result<(), async_channel::TrySendError<Response>> {
        self.response_tx.try_send(response)
    }
}

/// Sends requests to the host and awaits its responses.
#[derive(Debug, Clone)]
pub struct ToolRequestBroker<Args, Response> {
    tx: Sender<ToolRequest<Args, Response>>,
}

impl<Args, Response> ToolRequestBroker<Args, Response> {
    /// Send a request and await the response.
    ///
    /// # Errors
    ///
    /// Returns an error if the queue has been dropped, which means no host is
    /// listening for requests any more, or if the host dropped the request
    /// without responding.
    pub async fn request(&self, args: Args) -> aither_core::Result<Response> {
        let (response_tx, response_rx) = async_channel::bounded(1);
        self.tx
            .send(ToolRequest { args, response_tx })
            .await
            .map_err(|_| aither_core::Error::msg("Tool request channel is unavailable."))?;

        response_rx
            .recv()
            .await
            .map_err(|_| aither_core::Error::msg("Tool request was cancelled."))
    }
}

/// Pending requests, consumed by the host.
#[derive(Debug)]
pub struct ToolRequestQueue<Args, Response> {
    rx: Receiver<ToolRequest<Args, Response>>,
}

impl<Args, Response> ToolRequestQueue<Args, Response> {
    /// Await the next request.
    pub async fn next(&self) -> Option<ToolRequest<Args, Response>> {
        self.rx.recv().await.ok()
    }

    /// Returns true if there are no pending requests.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rx.is_empty()
    }
}

/// Create a new request broker/queue pair.
#[must_use]
pub fn channel<Args, Response>() -> (
    ToolRequestBroker<Args, Response>,
    ToolRequestQueue<Args, Response>,
) {
    let (tx, rx) = async_channel::unbounded();
    (ToolRequestBroker { tx }, ToolRequestQueue { rx })
}
