//! A session's MCP servers are connected and their tools reach its agent.

use std::sync::atomic::{AtomicUsize, Ordering};

use aither_acp::aither::AitherHost;
use aither_acp::client::{Client, SessionEvent, TurnOutcome};
use aither_acp::domain::{
    EnvVar, Implementation, McpServer, McpServerStdio, Open, StopReason, ToolCallStatus,
};
use aither_acp::server::Server;
use aither_agent::Agent;
use aither_core::LanguageModel;
use aither_core::llm::{Classify, Event, Failure, LLMRequest, model::Profile};
use futures_lite::{Stream, StreamExt, stream};

#[derive(Debug)]
struct ModelError;

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("model error")
    }
}

impl std::error::Error for ModelError {}

impl Classify for ModelError {
    fn failure(&self) -> Failure {
        Failure::permanent()
    }
}

/// Calls `ping` on its first request and ends the turn on the next.
#[derive(Clone, Default)]
struct PingsOnce {
    requests: std::sync::Arc<AtomicUsize>,
}

impl LanguageModel for PingsOnce {
    type Error = ModelError;

    fn respond(
        &self,
        _request: LLMRequest,
    ) -> impl Stream<Item = Result<Event, ModelError>> + Send {
        let event = if self.requests.fetch_add(1, Ordering::SeqCst) == 0 {
            Event::tool_call("call-1", "ping", serde_json::json!({}))
        } else {
            Event::Text("done".to_string())
        };
        stream::once(Ok(event))
    }

    fn profile(&self) -> impl std::future::Future<Output = Result<Profile, ModelError>> + Send {
        std::future::ready(Ok(Profile::new("ping", "test", "ping", "Ping", 128_000)))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_session_agent_can_call_the_tools_of_the_mcp_servers_it_was_opened_with() {
    let (client_t, agent_t) = aither_acp::rpc::Memory::pair();
    let server = Server::new(
        AitherHost::new(
            executor_core::tokio::TokioGlobal,
            |setup: aither_acp::aither::SessionSetup| async move {
                let builder = setup.mcp.into_iter().fold(
                    Agent::builder(PingsOnce::default()),
                    aither_agent::AgentBuilder::mcp,
                );
                Ok(builder.build())
            },
        ),
        Implementation::new("test", "0"),
    );
    let (_agent_peer, serve) = server.serve(agent_t);
    // The server must be driven for `connect`'s handshake to resolve.
    tokio::spawn(serve);
    let (client, conn) = Client::builder(Implementation::new("test-client", "0"))
        .connect(client_t)
        .await
        .expect("initialize");
    let _drive = tokio::spawn(conn);

    let script = format!(
        "{}/tests/fixtures/ping_mcp_server.sh",
        env!("CARGO_MANIFEST_DIR")
    );
    let session = client
        .new_session(std::env::temp_dir())
        .mcp_server(McpServer::Stdio(McpServerStdio {
            name: "ping".to_string(),
            command: "sh".to_string(),
            args: vec![script],
            env: Vec::<EnvVar>::new(),
            meta: None,
            extra: std::collections::BTreeMap::new(),
        }))
        .open()
        .await
        .expect("session/new");

    let mut events = session.subscribe();
    let outcome = session.prompt_run("ping").await.expect("prompt");
    let TurnOutcome::Finished { stop_reason, .. } = outcome else {
        panic!("expected a finished turn, got {outcome:?}");
    };
    assert_eq!(stop_reason, Some(Open::Known(StopReason::EndTurn)));

    // The ping tool call completed; its result text is the server's `pong`.
    let mut calls = Vec::new();
    while let Ok(Some(event)) =
        tokio::time::timeout(std::time::Duration::from_millis(20), events.next()).await
    {
        calls.push(event);
    }
    let finished = calls.iter().find_map(|event| match event {
        SessionEvent::ToolCall(call)
            if call.status == Some(Open::Known(ToolCallStatus::Completed)) =>
        {
            Some(call)
        }
        _ => None,
    });
    let finished = finished.expect("the agent called a tool");
    let text = finished
        .content
        .iter()
        .filter_map(|content| match content {
            aither_acp::domain::ToolCallContent::Content(c) => {
                c.content.as_text().map(str::to_owned)
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(text, "pong");
}
