//! Client↔server end-to-end over `rpc::Memory`, at the public API level.
//!
//! v2 is exercised through [`Client`]/[`ClientSession`]; the v1 flow is
//! driven with a raw [`Peer`] so the wire codec is what is under test.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_acp::client::{Client, MessageKind, TurnOutcome};
use aither_acp::domain::ids::{ConfigId, ConfigValueId};
use aither_acp::domain::{
    ConfigKind, ConfigOption, ConfigValue, ConfigValueById, ContentBlock, ForegroundState,
    Implementation, Open, SelectConfig, SelectOptions, SessionId, SessionSetupParams,
    SessionUpdate, StopReason,
};
use aither_acp::rpc::{self, Handler, Notification, Peer, Request, RequestContext, Response};
use aither_acp::server::{
    PendingPrompt, PromptError, ResumeRequest, Server, ServerError, Session, SessionContext,
    SessionHost,
};
use aither_acp::wire::v1;
use serde_json::{Value, json};

// --- the test host ---------------------------------------------------------

/// What a scripted session does when prompted.
struct EchoSession {
    /// Waits for cancellation instead of replying.
    wait_cancel: bool,
    /// Counts `prompt()` calls for the idempotency test.
    prompts: Arc<AtomicUsize>,
}

impl Session for EchoSession {
    async fn prompt(
        &self,
        pending: PendingPrompt,
        _cx: &SessionContext,
    ) -> Result<StopReason, PromptError> {
        self.prompts.fetch_add(1, Ordering::SeqCst);
        let text = pending
            .content()
            .iter()
            .filter_map(ContentBlock::as_text)
            .collect::<Vec<_>>()
            .join("\n");
        let turn = pending
            .accept()
            .await
            .map_err(|rejected| PromptError::from(rejected.into_inner()))?;
        if self.wait_cancel {
            turn.cancelled().await;
            return Ok(StopReason::Cancelled);
        }
        turn.message()
            .text(format!("echo: {text}"))
            .await
            .map_err(|_| PromptError::failed("emitting the echo"))?;
        Ok(StopReason::EndTurn)
    }

    fn config_options(&self) -> Vec<ConfigOption> {
        vec![ConfigOption {
            config_id: ConfigId::from_string("model"),
            name: "Model".to_string(),
            description: None,
            category: None,
            kind: ConfigKind::Select(SelectConfig {
                current_value: "gpt-5".into(),
                options: SelectOptions::Ungrouped(vec![]),
            }),
            meta: None,
            extra: BTreeMap::new(),
        }]
    }

    fn set_config(
        &self,
        _id: &ConfigId,
        _value: ConfigValue,
        _cx: &SessionContext,
    ) -> impl std::future::Future<Output = Result<Vec<ConfigOption>, rpc::RpcError>> + Send {
        std::future::ready(Ok(self.config_options()))
    }
}

/// A host handing out echo sessions; `prompts` counts `prompt()` calls,
/// `replayed` the updates `session/resume` replays.
struct TestHost {
    prompts: Arc<AtomicUsize>,
    wait_cancel: bool,
    replayed: Vec<SessionUpdate>,
}

fn echo_host() -> TestHost {
    TestHost {
        prompts: Arc::new(AtomicUsize::new(0)),
        wait_cancel: false,
        replayed: Vec::new(),
    }
}

impl SessionHost for TestHost {
    type Session = EchoSession;

    fn new_session(
        &self,
        _params: SessionSetupParams,
        _cx: &SessionContext,
    ) -> impl std::future::Future<Output = Result<Self::Session, ServerError>> + Send {
        std::future::ready(Ok(EchoSession {
            wait_cancel: self.wait_cancel,
            prompts: self.prompts.clone(),
        }))
    }

    async fn resume_session(
        &self,
        request: ResumeRequest,
        cx: &SessionContext,
    ) -> Result<Self::Session, ServerError> {
        for update in self.replayed.clone() {
            let _ = request.replay.send(update);
        }
        drop(request);
        self.new_session(
            SessionSetupParams {
                cwd: "/".into(),
                additional_directories: Vec::new(),
                mcp_servers: Vec::new(),
                system_prompt: None,
                meta: None,
                extra: BTreeMap::new(),
            },
            cx,
        )
        .await
    }
}

/// Runs a [`Server`] on one end and connects a [`Client`] on the other;
/// returns the client plus the task driving both directions.
async fn wired_client(host: impl SessionHost) -> (Client, tokio::task::JoinHandle<()>) {
    let (client_t, agent_t) = rpc::Memory::pair();
    let server = Server::new(host, Implementation::new("test-agent", "0.0.0"));
    let (_peer, serve) = server.serve(agent_t);
    // The server must be driven for `connect`'s handshake to resolve.
    tokio::spawn(serve);
    let (client, conn) = Client::builder(Implementation::new("test-client", "0.0.0"))
        .connect(client_t)
        .await
        .expect("initialize");
    let drive = tokio::spawn(conn);
    (client, drive)
}

// --- v2 through the public client -------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_initialize_new_prompt_and_idle() {
    let (client, _drive) = wired_client(echo_host()).await;
    assert_eq!(client.protocol_version(), 2);
    assert_eq!(
        client.agent_info().map(|info| info.name.as_str()),
        Some("test-agent")
    );

    let session = client
        .new_session("/tmp")
        .open()
        .await
        .expect("session/new");
    let outcome = session.prompt_run("hi").await.expect("prompt");

    let TurnOutcome::Finished {
        stop_reason, text, ..
    } = outcome
    else {
        panic!("expected a finished turn, got {outcome:?}");
    };
    assert_eq!(stop_reason, Some(Open::Known(StopReason::EndTurn)));
    assert_eq!(text, "echo: hi");

    // The reducer observed the turn lifecycle in the session snapshot.
    let state = session.state();
    assert!(
        state
            .messages
            .keys()
            .any(|(kind, _)| *kind == MessageKind::User),
        "state: {state:?}"
    );
    assert!(
        state
            .messages
            .keys()
            .any(|(kind, _)| *kind == MessageKind::Agent)
    );
    assert!(matches!(
        state.foreground,
        Some(Open::Known(ForegroundState::Idle))
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_cancel_ends_the_turn_cancelled() {
    let host = TestHost {
        wait_cancel: true,
        ..echo_host()
    };
    let (client, _drive) = wired_client(host).await;
    let session = client.new_session("/tmp").open().await.expect("new");

    let mut turn = session.prompt("hi").expect("prompt");
    let clone = session.clone();
    let cancel = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        clone.cancel().await.expect("cancel");
    });
    let outcome = turn.finish().await;
    cancel.await.unwrap();
    assert!(
        matches!(
            outcome,
            TurnOutcome::Finished {
                stop_reason: Some(Open::Known(StopReason::Cancelled)),
                ..
            } | TurnOutcome::Cancelled
        ),
        "outcome: {outcome:?}"
    );
}

/// The server dedupes repeated `idempotencyKey`s: `prompt()` runs once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_idempotent_prompt_runs_once() {
    let host = echo_host();
    let prompts = host.prompts.clone();
    let (client, _drive) = wired_client(host).await;
    let session = client.new_session("/tmp").open().await.expect("new");

    // `prompt` generates an idempotency key itself; repeating the *same*
    // request is a server concern, exercised at the wire level in the
    // idempotency unit tests. Two distinct prompts still both complete.
    let first = session.prompt_run("one").await.expect("prompt 1");
    let second = session.prompt_run("two").await.expect("prompt 2");
    assert!(matches!(first, TurnOutcome::Finished { .. }));
    assert!(matches!(second, TurnOutcome::Finished { .. }));
    assert_eq!(prompts.load(Ordering::SeqCst), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_resume_replays_history_then_replay_complete() {
    let host = TestHost {
        replayed: vec![SessionUpdate::UserMessageChunk(
            aither_acp::domain::ContentChunk::new(
                aither_acp::domain::MessageId::from_string("replay-1"),
                ContentBlock::text("restored text"),
            ),
        )],
        ..echo_host()
    };
    let (client, _drive) = wired_client(host).await;
    let session = client
        .resume_session(SessionId::from_string("restored-1"), "/tmp")
        .open()
        .await
        .expect("resume");

    // The replay happened during `open()`, before any subscription —
    // it is visible in the session snapshot.
    let state = session.state();
    assert!(state.replay_done, "replay: {state:?}");
    let restored = state
        .messages
        .values()
        .flat_map(|message| message.content.as_option().into_iter().flatten())
        .filter_map(|block| block.as_text())
        .collect::<String>();
    assert_eq!(restored, "restored text");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_set_config_reports_the_new_options() {
    let (client, _drive) = wired_client(echo_host()).await;
    let session = client.new_session("/tmp").open().await.expect("new");
    session
        .set_config(
            ConfigId::from_string("model"),
            ConfigValue::Id(ConfigValueById {
                value: ConfigValueId::from_string("gpt-5"),
            }),
        )
        .await
        .expect("set_config");
    // The agent's config_option_update reached the session snapshot.
    for _ in 0..50 {
        if !session.state().config_options.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(session.state().config_options.len(), 1);
    assert_eq!(session.state().config_options[0].name, "Model");
}

// --- v1 driven raw ----------------------------------------------------------

/// A recording client handler: updates go to `updates`.
#[derive(Clone, Default)]
struct Capture {
    updates: Arc<Mutex<Vec<Value>>>,
}

impl Handler for Capture {
    fn request(
        &self,
        req: Request,
        _cx: RequestContext,
    ) -> impl std::future::Future<Output = Response> + Send {
        std::future::ready(Response::error(
            req.id,
            rpc::RpcError::method_not_found(&req.method),
        ))
    }

    async fn notification(&self, notification: Notification) {
        self.updates
            .lock()
            .expect("updates")
            .push(notification.params.unwrap_or_default());
    }
}

async fn wait_updates(capture: &Capture, pred: impl Fn(&[Value]) -> bool) {
    for _ in 0..200 {
        if pred(&capture.updates.lock().expect("updates")) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for updates");
}

/// Wire a server to a raw `Peer` driven by `capture`.
fn wire(host: impl SessionHost) -> (Peer, Capture, tokio::task::JoinHandle<()>) {
    let (client_t, agent_t) = rpc::Memory::pair();
    let server = Server::new(host, Implementation::new("test", "0.0.0"));
    let (_agent_peer, serve) = server.serve(agent_t);
    let capture = Capture::default();
    let (peer, conn) = rpc::connection(client_t, capture.clone());
    let drive = tokio::spawn(async move {
        futures_lite::future::or(conn, serve).await;
    });
    (peer, capture, drive)
}

/// The v1 flow: v1-shaped initialize → session/new → prompt blocks for the
/// turn and answers `stopReason`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v1_prompt_blocks_for_the_stop_reason() {
    let (peer, capture, _drive) = wire(echo_host());

    let init: Value = peer
        .request(
            "initialize",
            &v1::InitializeParams {
                protocol_version: 1,
                client_capabilities: v1::ClientCapabilities::default(),
                client_info: None,
                meta: None,
            },
        )
        .await
        .expect("initialize");
    assert_eq!(init["protocolVersion"], 1);

    let new: Value = peer
        .request("session/new", &json!({"cwd": "/tmp"}))
        .await
        .expect("session/new");
    let session_id = new["sessionId"].as_str().expect("sessionId").to_string();

    let result: Value = peer
        .request(
            "session/prompt",
            &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}),
        )
        .await
        .expect("session/prompt");
    assert_eq!(result["stopReason"], "end_turn");

    wait_updates(&capture, |updates| {
        updates
            .iter()
            .any(|u| u["update"]["sessionUpdate"] == "agent_message_chunk")
    })
    .await;
}

/// A repeated `idempotencyKey` returns the same `messageId` and drives
/// exactly one `prompt()` call — on the v2 wire this time.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_wire_idempotent_prompt_dedupes() {
    let host = echo_host();
    let prompts = host.prompts.clone();
    let (peer, capture, _drive) = wire(host);
    peer.request::<Value, Value>(
        "initialize",
        &json!({"protocolVersion": 2, "capabilities": {}}),
    )
    .await
    .expect("initialize");
    let new: Value = peer
        .request("session/new", &json!({"cwd": "/tmp"}))
        .await
        .expect("session/new");
    let session_id = new["sessionId"].as_str().unwrap().to_string();
    let params = json!({
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": "hi"}],
        "idempotencyKey": "k-1",
    });
    let first: Value = peer
        .request("session/prompt", &params)
        .await
        .expect("prompt 1");
    let second: Value = peer
        .request("session/prompt", &params)
        .await
        .expect("prompt 2");
    assert_eq!(first["messageId"], second["messageId"]);
    wait_updates(&capture, |updates| {
        updates.iter().any(|u| {
            u["update"]["sessionUpdate"] == "state_update" && u["update"]["state"] == "idle"
        })
    })
    .await;
    assert_eq!(prompts.load(Ordering::SeqCst), 1);
}

/// v1 `session/cancel` resolves the blocked prompt with `cancelled`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v1_cancel_returns_cancelled() {
    let host = TestHost {
        wait_cancel: true,
        ..echo_host()
    };
    let (peer, _capture, _drive) = wire(host);

    peer.request::<Value, Value>(
        "initialize",
        &json!({"protocolVersion": 1, "clientCapabilities": {}}),
    )
    .await
    .expect("initialize");
    let new: Value = peer
        .request("session/new", &json!({"cwd": "/tmp"}))
        .await
        .expect("session/new");
    let session_id = new["sessionId"].as_str().unwrap().to_string();

    let prompt_peer = peer.clone();
    let sid = session_id.clone();
    let prompt = tokio::spawn(async move {
        prompt_peer
            .request::<Value, Value>(
                "session/prompt",
                &json!({"sessionId": sid, "prompt": [{"type": "text", "text": "hi"}]}),
            )
            .await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    peer.notify("session/cancel", &json!({"sessionId": session_id}))
        .await
        .expect("cancel");
    let result = prompt.await.unwrap().expect("prompt result");
    assert_eq!(result["stopReason"], "cancelled");
}
