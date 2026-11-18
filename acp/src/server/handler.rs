//! Handler trait for client-to-agent ACP traffic.

use std::future::Future;

use aither_mcp::protocol::{JsonRpcError, JsonRpcNotification};
use serde_json::Value;
use tracing::debug;

use super::ClientPeer;
use crate::protocol::{
    AuthenticateParams, AuthenticateResult, ExtMethod, InitializeParams, InitializeResult,
    LogoutParams, LogoutResult, PromptParams, PromptResult, SessionCancelParams,
    SessionCloseParams, SessionCloseResult, SessionDeleteParams, SessionDeleteResult,
    SessionListParams, SessionListResult, SessionLoadParams, SessionLoadResult, SessionNewParams,
    SessionNewResult, SessionResumeParams, SessionResumeResult, SessionSetConfigOptionParams,
    SessionSetConfigOptionResult, SessionSetModeParams, SessionSetModeResult,
};

/// Handles traffic the client (editor) initiates toward the agent.
///
/// Implementors answer `initialize` and whichever session, authentication,
/// prompt, configuration, and extension methods the agent supports. Every
/// method receives a [`ClientPeer`] handle that can stream `session/update`
/// notifications and issue correlated agent-to-client requests (such as
/// `session/request_permission`, `fs/*`, `terminal/*`, and
/// `elicitation/create`) while the handler's future is in flight.
///
/// Inbound request and notification futures are polled concurrently by
/// [`serve`](super::serve), so a long-running [`prompt`](Self::prompt) awaiting
/// [`ClientPeer::request_permission`] never blocks outbound updates, client
/// responses, or mid-turn [`cancel`](Self::cancel) notifications.
///
/// Optional request methods default to reporting JSON-RPC `Method not found`,
/// and optional notification methods default to a no-op.
pub trait AgentHandler: Send + Sync + 'static {
    /// Handle the `initialize` handshake request.
    fn initialize(
        &self,
        peer: &ClientPeer,
        params: Option<InitializeParams>,
    ) -> impl Future<Output = Result<InitializeResult, JsonRpcError>> + Send;

    /// Handle an `authenticate` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn authenticate(
        &self,
        _peer: &ClientPeer,
        _params: AuthenticateParams,
    ) -> impl Future<Output = Result<AuthenticateResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("authenticate")) }
    }

    /// Handle a `logout` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn logout(
        &self,
        _peer: &ClientPeer,
        _params: LogoutParams,
    ) -> impl Future<Output = Result<LogoutResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("logout")) }
    }

    /// Handle a `session/new` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn new_session(
        &self,
        _peer: &ClientPeer,
        _params: SessionNewParams,
    ) -> impl Future<Output = Result<SessionNewResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("session/new")) }
    }

    /// Handle a `session/load` request.
    ///
    /// Implementations that support loading replay historical messages via
    /// [`ClientPeer::session_update`] before returning [`SessionLoadResult`].
    /// The default responds with a JSON-RPC method-not-found error.
    fn load_session(
        &self,
        _peer: &ClientPeer,
        _params: SessionLoadParams,
    ) -> impl Future<Output = Result<SessionLoadResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("session/load")) }
    }

    /// Handle a `session/resume` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn resume_session(
        &self,
        _peer: &ClientPeer,
        _params: SessionResumeParams,
    ) -> impl Future<Output = Result<SessionResumeResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("session/resume")) }
    }

    /// Handle a `session/list` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn list_sessions(
        &self,
        _peer: &ClientPeer,
        _params: SessionListParams,
    ) -> impl Future<Output = Result<SessionListResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("session/list")) }
    }

    /// Handle a `session/close` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn close_session(
        &self,
        _peer: &ClientPeer,
        _params: SessionCloseParams,
    ) -> impl Future<Output = Result<SessionCloseResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("session/close")) }
    }

    /// Handle a `session/delete` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn delete_session(
        &self,
        _peer: &ClientPeer,
        _params: SessionDeleteParams,
    ) -> impl Future<Output = Result<SessionDeleteResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("session/delete")) }
    }

    /// Handle a `session/prompt` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn prompt(
        &self,
        _peer: &ClientPeer,
        _params: PromptParams,
    ) -> impl Future<Output = Result<PromptResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("session/prompt")) }
    }

    /// Handle a `session/set_mode` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn set_mode(
        &self,
        _peer: &ClientPeer,
        _params: SessionSetModeParams,
    ) -> impl Future<Output = Result<SessionSetModeResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("session/set_mode")) }
    }

    /// Handle a `session/set_config_option` request.
    ///
    /// The default responds with a JSON-RPC method-not-found error.
    fn set_config_option(
        &self,
        _peer: &ClientPeer,
        _params: SessionSetConfigOptionParams,
    ) -> impl Future<Output = Result<SessionSetConfigOptionResult, JsonRpcError>> + Send {
        async { Err(JsonRpcError::method_not_found("session/set_config_option")) }
    }

    /// Handle a client-to-agent request for a `_`-prefixed extension method.
    ///
    /// The default responds with a JSON-RPC method-not-found error, as ACP
    /// requires for unrecognized extension requests.
    fn ext_request(
        &self,
        _peer: &ClientPeer,
        method: ExtMethod,
        _params: Option<Value>,
    ) -> impl Future<Output = Result<Value, JsonRpcError>> + Send {
        async move { Err(JsonRpcError::method_not_found(method.as_str())) }
    }

    /// Handle a `session/cancel` notification.
    ///
    /// Dispatched concurrently while any active [`prompt`](Self::prompt) future
    /// is running. The default ignores the notification.
    fn cancel(
        &self,
        _peer: &ClientPeer,
        _params: SessionCancelParams,
    ) -> impl Future<Output = ()> + Send {
        std::future::ready(())
    }

    /// Handle a `notifications/initialized` notification from the client.
    ///
    /// The default ignores the notification.
    fn initialized(&self, _peer: &ClientPeer) -> impl Future<Output = ()> + Send {
        std::future::ready(())
    }

    /// Handle a `_`-prefixed extension notification from the client.
    ///
    /// The default ignores the notification, as ACP requires for unrecognized
    /// extension notifications.
    fn ext_notify(
        &self,
        _peer: &ClientPeer,
        method: ExtMethod,
        _params: Option<Value>,
    ) -> impl Future<Output = ()> + Send {
        debug!(method = %method, "ignoring client extension notification");
        std::future::ready(())
    }

    /// Handle any other client-to-agent notification this crate does not model.
    ///
    /// The default ignores the notification.
    fn notification(
        &self,
        _peer: &ClientPeer,
        notification: JsonRpcNotification,
    ) -> impl Future<Output = ()> + Send {
        debug!(method = %notification.method, "ignoring client notification");
        std::future::ready(())
    }
}
