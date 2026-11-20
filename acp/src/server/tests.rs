//! Server-side tests over the `Memory` transport (both protocol
//! versions).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};

use crate::domain::{
    ContentBlock, Implementation, Open, SessionSetupParams, SessionUpdate, StopReason,
};
use crate::rpc::{self, Handler, Notification, Peer, Request, RequestContext, Response};
use crate::wire::v1;

use super::Server;
use super::context::SessionContext;
use super::error::{PromptError, ServerError};
use super::host::{ResumeRequest, SessionHost};
use super::prompt::PendingPrompt;
use super::session::Session;

/// A client handler recording inbound notifications as raw values.
#[derive(Clone, Default)]
struct Capture {
    updates: Arc<Mutex<Vec<Value>>>,
    requests: Arc<Mutex<Vec<Value>>>,
}

impl Handler for Capture {
    fn request(&self, req: Request, _cx: RequestContext) -> impl Future<Output = Response> + Send {
        self.requests
            .lock()
            .expect("requests")
            .push(json!({"method": req.method, "params": req.params}));
        std::future::ready(Response::error(
            req.id,
            rpc::RpcError::method_not_found(&req.method),
        ))
    }

    async fn notification(&self, n: Notification) {
        self.updates
            .lock()
            .expect("updates")
            .push(json!({"method": n.method, "params": n.params}));
    }
}

impl Capture {
    fn session_updates(&self) -> Vec<Value> {
        self.updates
            .lock()
            .expect("updates")
            .iter()
            .filter(|u| u["method"] == "session/update")
            .cloned()
            .collect()
    }

    fn update_kinds(&self) -> Vec<String> {
        self.session_updates()
            .iter()
            .filter_map(|u| {
                u["params"]["update"]["sessionUpdate"]
                    .as_str()
                    .map(String::from)
            })
            .collect()
    }

    /// Poll until `pred` sees enough updates (bounded wait).
    async fn wait_for(&self, pred: impl Fn(&[Value]) -> bool) {
        for _ in 0..200 {
            let updates = self.updates.lock().expect("updates").clone();
            if pred(&updates) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for updates: {:?}", self.updates);
    }
}

/// The test session: `prompt` echoes the prompt text and ends the turn.
struct EchoSession {
    prompts: AtomicUsize,
    /// When set, prompt waits on cancellation instead of replying.
    wait_cancel: bool,
    /// When set, only the *first* prompt waits on cancellation.
    first_cancel: bool,
    /// When set, the first prompt blocks until released.
    gate: Option<async_channel::Receiver<()>>,
    prompt_count: Arc<AtomicUsize>,
}

impl Session for EchoSession {
    async fn prompt(
        &self,
        pending: PendingPrompt,
        _cx: &SessionContext,
    ) -> Result<StopReason, PromptError> {
        let index = self.prompts.fetch_add(1, Ordering::SeqCst);
        self.prompt_count.fetch_add(1, Ordering::SeqCst);
        let text = pending
            .content()
            .iter()
            .find_map(|b| match b {
                ContentBlock::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let turn = pending
            .accept()
            .await
            .map_err(|e| PromptError::from(e.into_inner()))?;
        if self.wait_cancel {
            turn.cancelled().await;
            return Ok(StopReason::Cancelled);
        }
        // First-turn gating for the FIFO test: `first_cancel` sessions
        // wait on cancellation only for their first prompt; `gate`
        // sessions block until released.
        if self.first_cancel && index == 0 {
            turn.cancelled().await;
            return Ok(StopReason::Cancelled);
        }
        if index == 0
            && let Some(gate) = &self.gate
        {
            let _ = gate.recv().await;
        }
        let stream = turn.message();
        stream.text(format!("echo: {text}")).await.ok();
        Ok(StopReason::EndTurn)
    }
}

/// The test host counting `prompt()` calls per session.
struct TestHost {
    prompt_count: Arc<AtomicUsize>,
    wait_cancel: bool,
    /// Passed to [`EchoSession::first_cancel`].
    first_cancel: bool,
    /// First-prompt release channel (`Some` for the FIFO test).
    gate: Option<(async_channel::Sender<()>, async_channel::Receiver<()>)>,
    /// Updates a resumed session replays.
    replayed: Mutex<Vec<SessionUpdate>>,
    sessions: AtomicUsize,
}

impl TestHost {
    /// A plain echo host.
    fn echo() -> Self {
        Self {
            prompt_count: Arc::new(AtomicUsize::new(0)),
            wait_cancel: false,
            first_cancel: false,
            gate: None,
            replayed: Mutex::new(Vec::new()),
            sessions: AtomicUsize::new(0),
        }
    }

    /// A host whose every session waits on cancellation.
    fn cancelling() -> Self {
        Self {
            wait_cancel: true,
            ..Self::echo()
        }
    }
}

impl SessionHost for TestHost {
    type Session = EchoSession;

    fn new_session(
        &self,
        _params: SessionSetupParams,
        _cx: &SessionContext,
    ) -> impl Future<Output = Result<Self::Session, ServerError>> + Send {
        self.sessions.fetch_add(1, Ordering::SeqCst);
        std::future::ready(Ok(EchoSession {
            prompts: AtomicUsize::new(0),
            wait_cancel: self.wait_cancel,
            first_cancel: self.first_cancel,
            gate: self.gate.as_ref().map(|(_, rx)| rx.clone()),
            prompt_count: self.prompt_count.clone(),
        }))
    }

    async fn resume_session(
        &self,
        req: ResumeRequest,
        cx: &SessionContext,
    ) -> Result<Self::Session, ServerError> {
        let updates = std::mem::take(&mut *self.replayed.lock().expect("replayed"));
        for update in updates {
            let _ = req.replay.send(update);
        }
        drop(req);
        self.new_session(
            crate::domain::SessionSetupParams {
                cwd: std::path::PathBuf::from("/"),
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

/// Wire a server to a capturing client over `Memory`; returns the
/// client peer plus the capture handle.
fn wire(
    host: impl SessionHost,
) -> (
    Peer,
    Capture,
    std::pin::Pin<Box<dyn Future<Output = ()> + Send>>,
) {
    let (client_t, agent_t) = rpc::Memory::pair();
    let server = Server::new(host, Implementation::new("test", "0.0.0"));
    let (_agent_peer, serve) = server.serve(agent_t);
    let capture = Capture::default();
    let (peer, conn) = rpc::connection(client_t, capture.clone());
    let both = async move {
        futures_lite::future::or(conn, serve).await;
    };
    (peer, capture, Box::pin(both))
}

fn setup_params() -> Value {
    json!({"cwd": "/"})
}

use std::future::Future;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn initialize_v2_negotiates_v2() {
    let (peer, _cap, conn) = wire(TestHost::echo());
    tokio::spawn(conn);
    let result: Value = peer
        .request::<Value, Value>(
            "initialize",
            &json!({"protocolVersion": 2, "capabilities": {}, "info": {"name": "c", "version": "0"}}),
        )
        .await
        .expect("initialize");
    assert_eq!(result["protocolVersion"], 2);
    assert_eq!(result["info"]["name"], "test");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn initialize_v1_negotiates_v1() {
    let (peer, _cap, conn) = wire(TestHost::echo());
    tokio::spawn(conn);
    let result: Value = peer
        .request::<_, Value>(
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
    assert_eq!(result["protocolVersion"], 1);
}

/// v2 flow: initialize → session/new → session/prompt → `messageId`
/// immediately + user/agent/idle updates; turn completion reaches the
/// client as a `state_update` `idle`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_prompt_streams_and_finishes() {
    let (peer, cap, conn) = wire(TestHost::echo());
    tokio::spawn(conn);
    let init: Value = peer
        .request(
            "initialize",
            &json!({"protocolVersion": 2, "capabilities": {}, "info": {"name": "c", "version": "0"}}),
        )
        .await
        .expect("initialize");
    assert_eq!(init["protocolVersion"], 2);
    let new: Value = peer
        .request("session/new", &setup_params())
        .await
        .expect("session/new");
    let session_id = new["sessionId"].as_str().expect("sessionId").to_string();
    let prompt: Value = peer
        .request(
            "session/prompt",
            &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}),
        )
        .await
        .expect("session/prompt");
    assert!(prompt["messageId"].as_str().is_some());
    cap.wait_for(|u| {
        u.iter().any(|x| {
            x["params"]["update"]["sessionUpdate"] == "state_update"
                && x["params"]["update"]["state"] == "idle"
        })
    })
    .await;
    let kinds = cap.update_kinds();
    assert!(
        kinds.contains(&"user_message".to_string()),
        "kinds: {kinds:?}"
    );
    assert!(kinds.contains(&"agent_message_chunk".to_string()));
    assert!(kinds.contains(&"state_update".to_string()));
}

/// v1 flow: initialize → session/new → session/prompt blocks for the
/// turn result and returns `stopReason: "end_turn"`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v1_prompt_returns_stop_reason() {
    let (peer, cap, conn) = wire(TestHost::echo());
    tokio::spawn(conn);
    let init: Value = peer
        .request::<_, Value>(
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
        .request("session/new", &setup_params())
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
    // v1 saw the chunks (sessionUpdate *_message_chunk).
    let kinds = cap.update_kinds();
    assert!(
        kinds.contains(&"user_message_chunk".to_string()),
        "kinds: {kinds:?}"
    );
    assert!(kinds.contains(&"agent_message_chunk".to_string()));
}

/// `session/cancel` resolves the turn with `stopReason: cancelled`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_cancel_resolves_cancelled() {
    let (peer, cap, conn) = wire(TestHost::cancelling());
    tokio::spawn(conn);
    peer.request::<Value, Value>(
        "initialize",
        &json!({"protocolVersion": 2, "capabilities": {}}),
    )
    .await
    .expect("initialize");
    let new: Value = peer
        .request("session/new", &setup_params())
        .await
        .expect("session/new");
    let session_id = new["sessionId"].as_str().expect("sessionId").to_string();
    let prompt: Value = peer
        .request(
            "session/prompt",
            &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}),
        )
        .await
        .expect("session/prompt");
    assert!(prompt["messageId"].is_string());
    peer.notify("session/cancel", &json!({"sessionId": session_id}))
        .await
        .expect("cancel");
    cap.wait_for(|u| {
        u.iter().any(|x| {
            x["params"]["update"]["sessionUpdate"] == "state_update"
                && x["params"]["update"]["stopReason"] == "cancelled"
        })
    })
    .await;
}

/// A repeated `idempotencyKey` returns the same `messageId` and drives
/// exactly one `prompt()` call.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_idempotent_prompt_dedupes() {
    let prompt_count = Arc::new(AtomicUsize::new(0));
    let (peer, cap, conn) = wire(TestHost {
        prompt_count: prompt_count.clone(),
        wait_cancel: false,
        first_cancel: false,
        gate: None,
        replayed: Mutex::new(Vec::new()),
        sessions: AtomicUsize::new(0),
    });
    tokio::spawn(conn);
    peer.request::<Value, Value>(
        "initialize",
        &json!({"protocolVersion": 2, "capabilities": {}}),
    )
    .await
    .expect("initialize");
    let new: Value = peer
        .request("session/new", &setup_params())
        .await
        .expect("session/new");
    let session_id = new["sessionId"].as_str().expect("sessionId").to_string();
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
    cap.wait_for(|u| {
        u.iter().any(|x| {
            x["params"]["update"]["sessionUpdate"] == "state_update"
                && x["params"]["update"]["state"] == "idle"
        })
    })
    .await;
    assert_eq!(prompt_count.load(Ordering::SeqCst), 1);
}

/// `session/close` removes the session; subsequent prompts fail
/// with `session not found` (-32002).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_close_then_prompt_fails() {
    let (peer, _cap, conn) = wire(TestHost::echo());
    tokio::spawn(conn);
    peer.request::<Value, Value>(
        "initialize",
        &json!({"protocolVersion": 2, "capabilities": {}}),
    )
    .await
    .expect("initialize");
    let new: Value = peer
        .request("session/new", &setup_params())
        .await
        .expect("session/new");
    let session_id = new["sessionId"].as_str().expect("sessionId").to_string();
    peer.request::<Value, Value>("session/close", &json!({"sessionId": session_id}))
        .await
        .expect("session/close");
    let err = peer
        .request::<Value, Value>(
            "session/prompt",
            &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}),
        )
        .await
        .expect_err("prompt should fail");
    match err {
        rpc::PeerError::Rejected(e) => assert_eq!(e.code, -32002, "err: {e}"),
        other => panic!("expected Rejected, got {other}"),
    }
}

/// `session/resume` replays stored updates and ends with
/// `replay_complete` before returning.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_resume_replays() {
    let replayed = vec![SessionUpdate::Notice(crate::domain::Notice {
        severity: Open::Known(crate::domain::NoticeSeverity::Info),
        title: "restored".to_string(),
        description: None,
        meta: None,
        extra: BTreeMap::new(),
    })];
    let (peer, cap, conn) = wire(TestHost {
        prompt_count: Arc::new(AtomicUsize::new(0)),
        wait_cancel: false,
        first_cancel: false,
        gate: None,
        replayed: Mutex::new(replayed),
        sessions: AtomicUsize::new(0),
    });
    tokio::spawn(conn);
    peer.request::<Value, Value>(
        "initialize",
        &json!({"protocolVersion": 2, "capabilities": {}}),
    )
    .await
    .expect("initialize");
    let result: Value = peer
        .request(
            "session/resume",
            &json!({
                "sessionId": "restored-1",
                "cwd": "/",
            }),
        )
        .await
        .expect("session/resume");
    assert_eq!(result["sessionId"], "restored-1");
    let kinds = cap.update_kinds();
    assert!(kinds.contains(&"notice".to_string()), "kinds: {kinds:?}");
    assert!(
        kinds.contains(&"replay_complete".to_string()),
        "kinds: {kinds:?}"
    );
}

/// Two prompts on one session run in FIFO order: while the first turn
/// is gated open, the second queues; releasing the gate lets prompt 1
/// finish and prompt 2 run next.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v2_prompts_run_in_fifo_order() {
    let prompt_count = Arc::new(AtomicUsize::new(0));
    let (gate_tx, gate_rx) = async_channel::unbounded();
    let (peer, cap, conn) = wire(TestHost {
        prompt_count: prompt_count.clone(),
        wait_cancel: false,
        first_cancel: false,
        gate: Some((gate_tx.clone(), gate_rx)),
        replayed: Mutex::new(Vec::new()),
        sessions: AtomicUsize::new(0),
    });
    tokio::spawn(conn);
    peer.request::<Value, Value>(
        "initialize",
        &json!({"protocolVersion": 2, "capabilities": {}}),
    )
    .await
    .expect("initialize");
    let new: Value = peer
        .request("session/new", &setup_params())
        .await
        .expect("session/new");
    let session_id = new["sessionId"].as_str().expect("sessionId").to_string();
    // Prompt 1 occupies the turn (its prompt() blocks on the gate);
    // prompt 2 must queue behind it.
    let first: Value = peer
        .request(
            "session/prompt",
            &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "one"}]}),
        )
        .await
        .expect("prompt 1");
    cap.wait_for(|u| {
        u.iter().any(|x| {
            x["params"]["update"]["sessionUpdate"] == "state_update"
                && x["params"]["update"]["state"] == "running"
        })
    })
    .await;
    let second: Value = peer
        .request(
            "session/prompt",
            &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "two"}]}),
        )
        .await
        .expect("prompt 2");
    assert_ne!(first["messageId"], second["messageId"]);
    // Still only one turn has run while prompt 1 holds the slot.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(prompt_count.load(Ordering::SeqCst), 1);
    // Release prompt 1: it ends the turn, then prompt 2 must run.
    gate_tx.send(()).await.expect("release");
    cap.wait_for(|u| {
        u.iter()
            .filter(|x| {
                x["params"]["update"]["sessionUpdate"] == "state_update"
                    && x["params"]["update"]["state"] == "idle"
            })
            .count()
            >= 2
    })
    .await;
    assert_eq!(prompt_count.load(Ordering::SeqCst), 2);
    // FIFO order: two user_message emissions and two end_turn idles.
    let updates = cap.session_updates();
    let mut user_messages = 0;
    let mut idle_reasons = Vec::new();
    for u in updates {
        match u["params"]["update"]["sessionUpdate"].as_str() {
            Some("user_message") => user_messages += 1,
            Some("state_update") if u["params"]["update"]["state"] == "idle" => {
                idle_reasons.push(
                    u["params"]["update"]["stopReason"]
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
                );
            }
            _ => {}
        }
    }
    assert_eq!(user_messages, 2);
    assert_eq!(
        idle_reasons,
        vec!["end_turn".to_string(), "end_turn".to_string()]
    );
}

/// A non-`initialize` request ahead of the handshake is answered
/// `-32600`, and `initialize` must be the literal first request — a
/// later one is rejected too. A second `initialize` on a clean
/// connection is `-32600` as well.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn requests_before_initialize_and_duplicate_initialize_are_rejected() {
    // On this connection `session/new` runs ahead of the handshake.
    let (peer, _cap, conn) = wire(TestHost::echo());
    tokio::spawn(conn);
    match peer
        .request::<Value, Value>("session/new", &setup_params())
        .await
    {
        Err(rpc::PeerError::Rejected(e)) => assert_eq!(e.code(), rpc::INVALID_REQUEST),
        other => panic!("expected -32600, got {other:?}"),
    }
    // `initialize` is only ever the first request: even now it is -32600.
    match peer
        .request::<Value, Value>(
            "initialize",
            &json!({"protocolVersion": 2, "capabilities": {}}),
        )
        .await
    {
        Err(rpc::PeerError::Rejected(e)) => assert_eq!(e.code(), rpc::INVALID_REQUEST),
        other => panic!("expected -32600 for a late initialize, got {other:?}"),
    }
    drop(peer);

    // On a clean connection a repeated initialize is rejected.
    let (peer, _cap, conn) = wire(TestHost::echo());
    tokio::spawn(conn);
    peer.request::<Value, Value>(
        "initialize",
        &json!({"protocolVersion": 2, "capabilities": {}}),
    )
    .await
    .expect("initialize");
    match peer
        .request::<Value, Value>(
            "initialize",
            &json!({"protocolVersion": 2, "capabilities": {}}),
        )
        .await
    {
        Err(rpc::PeerError::Rejected(e)) => assert_eq!(e.code(), rpc::INVALID_REQUEST),
        other => panic!("expected -32600, got {other:?}"),
    }
}

/// `session/cancel` also fails prompts still waiting in the session's
/// queue, not just the running turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v1_cancel_fails_queued_prompts() {
    let (gate_tx, gate_rx) = async_channel::bounded::<()>(1);
    let prompt_count = Arc::new(AtomicUsize::new(0));
    let (peer, _cap, conn) = wire(TestHost {
        prompt_count: prompt_count.clone(),
        wait_cancel: true,
        first_cancel: false,
        gate: Some((gate_tx, gate_rx)),
        replayed: Mutex::new(Vec::new()),
        sessions: AtomicUsize::new(0),
    });
    tokio::spawn(conn);

    peer.request::<_, Value>(
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
    let new: Value = peer
        .request("session/new", &setup_params())
        .await
        .expect("session/new");
    let session_id = new["sessionId"].as_str().expect("sessionId").to_string();

    // Prompt 1 parks on cancellation; prompt 2 sits behind it in the
    // session queue.
    let first = tokio::spawn({
        let peer = peer.clone();
        let session_id = session_id.clone();
        async move {
            peer.request::<Value, Value>(
                "session/prompt",
                &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "one"}]}),
            )
            .await
        }
    });
    cap_wait_until_one(&prompt_count).await;
    let second = tokio::spawn({
        let peer = peer.clone();
        let session_id = session_id.clone();
        async move {
            peer.request::<Value, Value>(
                "session/prompt",
                &json!({"sessionId": session_id, "prompt": [{"type": "text", "text": "two"}]}),
            )
            .await
        }
    });
    // Let prompt 2 land in the queue, then cancel the session.
    tokio::time::sleep(Duration::from_millis(50)).await;
    peer.notify("session/cancel", &json!({"sessionId": session_id}))
        .await
        .expect("cancel");

    // The queued prompt fails `request cancelled` (-32800); the running
    // prompt reports `stopReason: cancelled`.
    match second.await.expect("second") {
        Err(rpc::PeerError::Rejected(e)) => assert_eq!(e.code(), rpc::REQUEST_CANCELLED),
        other => panic!("expected -32800 for the queued prompt, got {other:?}"),
    }
    let first = first.await.expect("first").expect("first prompt response");
    assert_eq!(first["stopReason"], "cancelled");
}

/// Wait until the host has taken one prompt.
async fn cap_wait_until_one(count: &Arc<AtomicUsize>) {
    for _ in 0..200 {
        if count.load(Ordering::SeqCst) >= 1 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for the first prompt");
}
