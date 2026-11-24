//! Client unit tests: init detection, demux/reducer, turn ending.

use std::collections::BTreeMap;

use serde_json::json;

use crate::domain::{
    ConfigValueId, ContentBlock, CurrentModeUpdate, ForegroundState, MessageId, Open,
    ReplayComplete, SessionId, SessionUpdate, StopReason, update::StateUpdate,
};
use crate::wire::codec::{Protocol, detect_initialize};

use super::handle::SessionRegistry;
use crate::state::{MessageKind, SessionEvent, SessionReducer};

#[test]
fn detect_v2_and_v1_shapes() {
    assert_eq!(
        detect_initialize(&json!({"protocolVersion": 2, "capabilities": {}})),
        Protocol::V2
    );
    assert_eq!(
        detect_initialize(&json!({"protocolVersion": 1, "agentCapabilities": {}})),
        Protocol::V1
    );
}

fn idle_update(stop: Option<Open<StopReason>>) -> SessionUpdate {
    SessionUpdate::StateUpdate(StateUpdate {
        state: Open::Known(ForegroundState::Idle),
        stop_reason: stop,
        error: None,
        background: Vec::new(),
        meta: None,
        extra: BTreeMap::new(),
    })
}

fn foreground_update() -> SessionUpdate {
    SessionUpdate::StateUpdate(StateUpdate {
        state: Open::Known(ForegroundState::Running),
        stop_reason: None,
        error: None,
        background: Vec::new(),
        meta: None,
        extra: BTreeMap::new(),
    })
}

#[test]
fn reducer_turn_end_only_on_idle_transition() {
    let mut r = SessionReducer::new();
    // A bare `idle` while none was ever seen is informational, not a
    // turn end.
    let events = r.apply(&idle_update(None));
    assert!(
        events
            .iter()
            .all(|e| !matches!(e, SessionEvent::TurnEnded { .. }))
    );
    // Running -> idle: TurnEnded, carrying the update's stop reason.
    r.apply(&foreground_update());
    let events = r.apply(&idle_update(Some(Open::Known(StopReason::EndTurn))));
    assert!(events.iter().any(|e| matches!(
        e,
        SessionEvent::TurnEnded {
            stop_reason: Some(Open::Known(StopReason::EndTurn)),
            ..
        }
    )));
    // A second *bare* idle while already idle: no duplicate TurnEnded.
    let events = r.apply(&idle_update(None));
    assert!(
        events
            .iter()
            .all(|e| !matches!(e, SessionEvent::TurnEnded { .. }))
    );
}

#[test]
fn reducer_chunk_creates_message_and_mode_update() {
    let mut r = SessionReducer::new();
    let chunk = SessionUpdate::AgentMessageChunk(crate::domain::ContentChunk::new(
        "m1",
        ContentBlock::text("hi"),
    ));
    let events = r.apply(&chunk);
    assert!(matches!(
        events.first(),
        Some(SessionEvent::MessageDelta {
            kind: MessageKind::Agent,
            text: Some(t),
            ..
        }) if t == "hi"
    ));
    assert!(
        r.state()
            .messages
            .contains_key(&(MessageKind::Agent, MessageId::from("m1")))
    );

    let events = r.apply(&SessionUpdate::CurrentModeUpdate(CurrentModeUpdate {
        current_mode_id: ConfigValueId::from("code"),
        meta: None,
        extra: BTreeMap::new(),
    }));
    assert!(matches!(
        events.first(),
        Some(SessionEvent::ModeChanged(v)) if v.as_str() == "code"
    ));
    assert_eq!(
        r.state().current_mode.as_ref().map(ConfigValueId::as_str),
        Some("code")
    );
}

#[test]
fn demux_routes_updates_to_the_right_cell() {
    let registry = SessionRegistry::new();
    let s1 = SessionId::from("s1");
    let s2 = SessionId::from("s2");
    let c1 = registry.get_or_create(s1.clone());
    let rx2 = registry.get_or_create(s2).subscribe();

    registry.apply(&crate::domain::SessionNotification {
        session_id: s1,
        update: idle_update(Some(Open::Known(StopReason::EndTurn))),
        meta: None,
        extra: BTreeMap::new(),
    });

    // s1's cell saw the update; s2's subscriber did not.
    assert!(c1.reducer.lock().unwrap().state().foreground.is_some());
    assert!(rx2.is_empty());
}

#[tokio::test]
async fn subscribe_snapshot_then_live_events() {
    let registry = SessionRegistry::new();
    let cell = registry.get_or_create(SessionId::from("s"));
    cell.apply(&foreground_update());
    let (tx, rx) = async_channel::unbounded::<SessionEvent>();
    drop(tx);
    let sub = cell.subscribe();
    cell.mark_replay_complete();
    assert!(cell.state().replay_done);
    let event = sub.recv().await.expect("event");
    assert!(matches!(event, SessionEvent::ReplayComplete));
    drop(rx);
}

#[test]
fn v1_turn_end_update_synthesizes_idle() {
    let finished = crate::domain::PromptFinished {
        stop_reason: Open::Known(StopReason::EndTurn),
        usage: None,
        meta: None,
    };
    let update = SessionUpdate::StateUpdate(super::connection::v1_turn_end_update(&finished));
    let mut r = SessionReducer::new();
    r.apply(&foreground_update());
    let events = r.apply(&update);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SessionEvent::TurnEnded { .. }))
    );
}

#[test]
fn replay_complete_marks_state() {
    let mut r = SessionReducer::new();
    r.begin_replay();
    let events = r.apply(&SessionUpdate::ReplayComplete(ReplayComplete {
        meta: None,
        extra: BTreeMap::new(),
    }));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SessionEvent::ReplayComplete))
    );
    assert!(r.state().replay_done);
}

/// The in-process direct lane end to end: a `SessionHost` served by
/// `Client::in_process` — initialize, `session/new`, `session/prompt`,
/// the `running`/`idle` transition, and a finished turn outcome — all
/// without a transport or a serialized message.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn in_process_initialize_session_prompt_idle() {
    use crate::domain::SessionSetupParams;
    use crate::server::{
        PendingPrompt, PromptError, ServerError, Session, SessionContext, SessionHost,
    };

    /// A session that echoes the prompt back and ends the turn.
    struct Echo;
    impl Session for Echo {
        async fn prompt(
            &self,
            pending: PendingPrompt,
            _cx: &SessionContext,
        ) -> Result<StopReason, PromptError> {
            let text = pending
                .content()
                .iter()
                .filter_map(ContentBlock::as_text)
                .collect::<String>();
            let turn = pending
                .accept()
                .await
                .map_err(|rejected| PromptError::from(rejected.into_inner()))?;
            turn.message()
                .text(format!("echo: {text}"))
                .await
                .map_err(|_| PromptError::failed("echo failed"))?;
            Ok(StopReason::EndTurn)
        }
    }

    struct EchoHost;
    impl SessionHost for EchoHost {
        type Session = Echo;
        fn new_session(
            &self,
            _params: SessionSetupParams,
            _cx: &SessionContext,
        ) -> impl std::future::Future<Output = Result<Echo, ServerError>> + Send {
            std::future::ready(Ok(Echo))
        }
    }

    let (client, conn) =
        crate::client::Client::builder(crate::domain::Implementation::new("test-client", "0.0.0"))
            .in_process(EchoHost)
            .await
            .expect("in_process initialize");
    // The drive future must be polled for the turn scheduler to run.
    tokio::spawn(conn);

    assert_eq!(client.protocol_version(), 2);
    let session = client
        .new_session("/tmp")
        .open()
        .await
        .expect("session/new");
    let outcome = session.prompt_run("hi").await.expect("prompt");
    let crate::client::TurnOutcome::Finished {
        stop_reason, text, ..
    } = outcome
    else {
        panic!("expected a finished turn, got {outcome:?}");
    };
    assert_eq!(stop_reason, Some(Open::Known(StopReason::EndTurn)));
    assert_eq!(text, "echo: hi");
    assert!(matches!(
        session.state().foreground,
        Some(Open::Known(ForegroundState::Idle))
    ));
    assert!(
        session
            .state()
            .messages
            .values()
            .any(|m| m.content.as_option().is_some_and(|c| !c.is_empty()))
    );
}

type AuthLog = std::sync::Arc<std::sync::Mutex<Vec<crate::server::AuthRequest>>>;
type NotifyLog = std::sync::Arc<std::sync::Mutex<Vec<(String, Option<serde_json::Value>)>>>;

struct DirectEchoSession;

impl crate::server::Session for DirectEchoSession {
    async fn prompt(
        &self,
        pending: crate::server::PendingPrompt,
        _cx: &crate::server::SessionContext,
    ) -> Result<StopReason, crate::server::PromptError> {
        let text = pending
            .content()
            .iter()
            .filter_map(ContentBlock::as_text)
            .collect::<String>();
        let turn = pending
            .accept()
            .await
            .map_err(|rejected| crate::server::PromptError::from(rejected.into_inner()))?;
        turn.message()
            .text(format!("echo: {text}"))
            .await
            .map_err(|_| crate::server::PromptError::failed("echo failed"))?;
        Ok(StopReason::EndTurn)
    }
}

struct DirectEchoHost;

impl crate::server::SessionHost for DirectEchoHost {
    type Session = DirectEchoSession;

    fn new_session(
        &self,
        _params: crate::domain::SessionSetupParams,
        _cx: &crate::server::SessionContext,
    ) -> impl std::future::Future<Output = Result<DirectEchoSession, crate::server::ServerError>> + Send
    {
        std::future::ready(Ok(DirectEchoSession))
    }
}

fn configured_test_server() -> (crate::server::Server, AuthLog, NotifyLog) {
    use std::sync::{Arc, Mutex};

    use crate::domain::{AgentAuth, AuthKind, AuthMethod, AuthMethodId, ExtMethod, Implementation};
    use crate::rpc::RpcError;
    use crate::server::{AuthConfig, AuthRequest, Server};

    let auth_log: AuthLog = Arc::new(Mutex::new(Vec::new()));
    let auth_log_cb = auth_log.clone();
    let notify_log: NotifyLog = Arc::new(Mutex::new(Vec::new()));
    let notify_log_cb = notify_log.clone();

    let mut agent_impl = Implementation::new("configured-agent", "2.3.4");
    agent_impl.title = Some("Configured Agent".to_string());

    let auth_method = AuthMethod {
        method_id: AuthMethodId::from("oauth"),
        name: "OAuth".to_string(),
        description: Some("Sign in with OAuth".to_string()),
        kind: Open::Known(AuthKind::Agent(AgentAuth::default())),
        meta: None,
        extra: BTreeMap::new(),
    };

    let server = Server::builder(DirectEchoHost, agent_impl)
        .auth(AuthConfig::new(vec![auth_method], move |req| {
            let log = auth_log_cb.clone();
            async move {
                match &req {
                    AuthRequest::Login(login) if login.method_id.as_str() == "oauth" => {
                        log.lock().expect("auth_log").push(req);
                        Ok(())
                    }
                    AuthRequest::Login(login) => Err(RpcError::invalid_params(format!(
                        "unknown auth method {}",
                        login.method_id
                    ))),
                    AuthRequest::Logout(Some(id)) if id.as_str() == "cannot-logout" => {
                        Err(RpcError::invalid_params("logout refused"))
                    }
                    AuthRequest::Logout(_) => {
                        log.lock().expect("auth_log").push(req);
                        Ok(())
                    }
                }
            }
        }))
        .ext(move |method: ExtMethod, params| {
            let notifications = notify_log_cb.clone();
            async move {
                match method.as_str() {
                    "_demo/echo" => Ok(json!({
                        "method": method.as_str(),
                        "params": params,
                    })),
                    "_demo/notify" => {
                        notifications
                            .lock()
                            .expect("notify_log")
                            .push((method.as_str().to_string(), params));
                        Ok(json!({}))
                    }
                    "_demo/fail" => Err(RpcError::invalid_params("bad ext params")),
                    other => Err(RpcError::method_not_found(other)),
                }
            }
        })
        .unknown_handler(Arc::new(|method, params| {
            let method = method.to_string();
            Box::pin(async move {
                Ok(json!({
                    "unknown": method,
                    "params": params,
                }))
            })
        }))
        .build();

    (server, auth_log, notify_log)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn in_process_server_preserves_metadata_auth_and_prompting() {
    use crate::domain::{AuthMethodId, Implementation};
    use crate::rpc;

    let (server, auth_log, _notify_log) = configured_test_server();
    let (client, conn) =
        crate::client::Client::builder(Implementation::new("test-client", "0.0.0"))
            .in_process_server(server)
            .await
            .expect("in_process_server initialize");
    let driver = tokio::spawn(conn);

    assert_eq!(client.protocol_version(), 2);
    let info = client.agent_info().expect("agent info");
    assert_eq!(info.name, "configured-agent");
    assert_eq!(info.version, "2.3.4");
    assert_eq!(info.title.as_deref(), Some("Configured Agent"));
    assert_eq!(client.auth_methods().len(), 1);
    assert_eq!(client.auth_methods()[0].method_id.as_str(), "oauth");

    client.authenticate("oauth").await.expect("auth login");
    client
        .logout(Some(AuthMethodId::from("oauth")))
        .await
        .expect("auth logout specific");
    client.logout(None).await.expect("auth logout default");

    let login_err = client
        .authenticate("bad-method")
        .await
        .expect_err("unknown auth method must fail");
    assert!(
        matches!(login_err, super::ClientError::Rejected(ref e) if e.code() == rpc::INVALID_PARAMS),
        "expected invalid params, got {login_err:?}"
    );
    let logout_err = client
        .logout(Some(AuthMethodId::from("cannot-logout")))
        .await
        .expect_err("refused logout must fail");
    assert!(
        matches!(logout_err, super::ClientError::Rejected(ref e) if e.code() == rpc::INVALID_PARAMS),
        "expected invalid params, got {logout_err:?}"
    );
    assert_eq!(auth_log.lock().expect("auth_log").len(), 3);

    let session = client
        .new_session("/tmp")
        .open()
        .await
        .expect("session/new");
    let outcome = session.prompt_run("hello").await.expect("prompt");
    assert!(matches!(
        outcome,
        crate::client::TurnOutcome::Finished { ref text, .. } if text == "echo: hello"
    ));

    client.close().await;
    driver.await.expect("driver exited cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn in_process_server_preserves_ext_and_unknown_handler() {
    use crate::domain::{ExtMethod, Implementation};
    use crate::rpc;

    let (server, _auth_log, notify_log) = configured_test_server();
    let core = server.core().clone();
    let (client, conn) =
        crate::client::Client::builder(Implementation::new("test-client", "0.0.0"))
            .in_process_server(server)
            .await
            .expect("in_process_server initialize");
    let driver = tokio::spawn(conn);

    let echo = ExtMethod::try_new("_demo/echo").expect("ext method");
    let reply = client
        .ext_request(&echo, &json!({"value": 42}))
        .await
        .expect("ext_request");
    assert_eq!(reply["method"], "_demo/echo");
    assert_eq!(reply["params"]["value"], 42);

    let fail = ExtMethod::try_new("_demo/fail").expect("ext method");
    let ext_err = client
        .ext_request(&fail, &json!({}))
        .await
        .expect_err("ext_request failure");
    assert!(
        matches!(ext_err, super::ClientError::Rejected(ref e) if e.code() == rpc::INVALID_PARAMS && e.message.contains("bad ext params")),
        "expected invalid params, got {ext_err:?}"
    );

    let notify = ExtMethod::try_new("_demo/notify").expect("ext method");
    client
        .ext_notify(&notify, &json!({"ping": true}))
        .await
        .expect("ext_notify");
    {
        let recorded = notify_log.lock().expect("notify_log");
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].0, "_demo/notify");
        assert_eq!(recorded[0].1, Some(json!({"ping": true})));
    }

    let notify_err = client
        .ext_notify(&fail, &json!({}))
        .await
        .expect_err("ext_notify failure");
    assert!(
        matches!(notify_err, super::ClientError::Rejected(ref e) if e.code() == rpc::INVALID_PARAMS),
        "expected invalid params, got {notify_err:?}"
    );

    assert!(core.initialized());
    let unknown_reply = core
        .unknown_request("legacy/ping", Some(json!({"ok": true})))
        .await
        .expect("unknown_handler preserved");
    assert_eq!(unknown_reply["unknown"], "legacy/ping");
    assert_eq!(unknown_reply["params"]["ok"], true);

    client.close().await;
    driver.await.expect("driver exited cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn in_process_server_ext_handler_unconfigured_and_double_init_errors() {
    use std::sync::Arc;

    use crate::domain::{ExtMethod, Implementation};
    use crate::rpc;
    use crate::server::Server;

    // `.ext_handler(...)` works while unconfigured `.auth(...)` rejects with METHOD_NOT_FOUND.
    let server = Server::builder(DirectEchoHost, Implementation::new("ext-only", "0.1.0"))
        .ext_handler(Arc::new(|method, _params| {
            Box::pin(async move { Ok(json!({ "handled": method.as_str() })) })
        }))
        .build();
    let (client, conn) =
        crate::client::Client::builder(Implementation::new("test-client", "0.0.0"))
            .in_process_server(server)
            .await
            .expect("in_process_server");
    let driver = tokio::spawn(conn);

    assert_eq!(client.auth_methods(), []);
    let auth_err = client
        .authenticate("oauth")
        .await
        .expect_err("unconfigured auth login must fail");
    assert!(
        matches!(auth_err, super::ClientError::Rejected(ref e) if e.code() == rpc::METHOD_NOT_FOUND),
        "expected method not found, got {auth_err:?}"
    );
    let logout_err = client
        .logout(None)
        .await
        .expect_err("unconfigured auth logout must fail");
    assert!(
        matches!(logout_err, super::ClientError::Rejected(ref e) if e.code() == rpc::METHOD_NOT_FOUND),
        "expected method not found, got {logout_err:?}"
    );
    let ping = ExtMethod::try_new("_ext/ping").expect("ext method");
    let reply = client
        .ext_request(&ping, &json!({}))
        .await
        .expect("ext_handler");
    assert_eq!(reply["handled"], "_ext/ping");
    client.close().await;
    driver.await.expect("driver exited");

    // A bare Server with no ext handler rejects ext_request with METHOD_NOT_FOUND.
    let bare = Server::new(DirectEchoHost, Implementation::new("bare", "0.1.0"));
    let (bare_client, bare_conn) =
        crate::client::Client::builder(Implementation::new("test-client", "0.0.0"))
            .in_process_server(bare)
            .await
            .expect("bare in_process_server");
    let bare_driver = tokio::spawn(bare_conn);
    let ext_err = bare_client
        .ext_request(&ping, &json!({}))
        .await
        .expect_err("unconfigured ext must fail");
    assert!(
        matches!(ext_err, super::ClientError::Rejected(ref e) if e.code() == rpc::METHOD_NOT_FOUND),
        "expected method not found, got {ext_err:?}"
    );
    bare_client.close().await;
    bare_driver.await.expect("bare driver exited");

    // An already-initialized Server rejects a second initialize with INVALID_REQUEST.
    let already_init = Server::new(DirectEchoHost, Implementation::new("reused", "0.1.0"));
    already_init
        .core()
        .gate_request("initialize")
        .expect("first gate");
    let init_err = crate::client::Client::builder(Implementation::new("test-client", "0.0.0"))
        .in_process_server(already_init)
        .await
        .expect_err("already initialized server must reject");
    assert!(
        matches!(init_err, super::ClientError::Rejected(ref e) if e.code() == rpc::INVALID_REQUEST),
        "expected invalid request, got {init_err:?}"
    );
}
