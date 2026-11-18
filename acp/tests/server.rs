//! Integration tests pairing `serve` / `AgentHandler` / `ClientPeer` with
//! `AcpClient` / `ClientHandler` over in-memory duplex and NDJSON stream
//! transports.

use std::collections::BTreeMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use aither_acp::{
    AcpClient, AgentCapabilities, AgentHandler, AuthenticateParams, AuthenticateResult,
    ClientError, ClientHandler, ClientPeer, ConfigOption, ConfigOptionValue, ContentBlock,
    ContentChunk, ElicitationAction, ElicitationCompleteParams, ElicitationCreateParams,
    ElicitationCreateResult, ElicitationFormMode, ElicitationMode, ElicitationSchema,
    ElicitationSessionScope, ElicitationUrlMode, ExtMethod, Implementation, InitializeParams,
    InitializeResult, LogoutParams, LogoutResult, McpServerStdio, PROTOCOL_VERSION,
    PermissionOption, PermissionOptionKind, PromptParams, PromptResult, ReadTextFileParams,
    ReadTextFileResult, RequestPermissionOutcome, RequestPermissionParams, RequestPermissionResult,
    ServerError, SessionAdditionalDirectoriesCapabilities, SessionCancelParams,
    SessionCapabilities, SessionCloseParams, SessionCloseResult, SessionConfigValue,
    SessionDeleteParams, SessionDeleteResult, SessionForkCapabilities, SessionForkParams,
    SessionForkResult, SessionInfo, SessionListParams, SessionListResult, SessionLoadParams,
    SessionLoadResult, SessionMode, SessionModeState, SessionNewParams, SessionNewResult,
    SessionNotification, SessionResumeParams, SessionResumeResult, SessionSetConfigOptionParams,
    SessionSetConfigOptionResult, SessionSetModeParams, SessionSetModeResult, SessionUpdate,
    StopReason, TerminalCreateParams, TerminalCreateResult, TerminalExitStatus, TerminalKillParams,
    TerminalKillResult, TerminalOutputParams, TerminalOutputResult, TerminalReleaseParams,
    TerminalReleaseResult, TerminalWaitForExitParams, TextContent, ToolCall, ToolCallStatus,
    WriteTextFileParams, WriteTextFileResult, serve,
};
use aither_mcp::protocol::{
    ErrorCode, JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
};
use aither_mcp::transport::{BidirectionalTransport, DuplexTransport, StreamTransport};
use futures_lite::future::or;
use futures_lite::io::{AsyncRead, AsyncWrite, BufReader};
use serde_json::{Value, json};

fn text_block(text: &str) -> ContentBlock {
    ContentBlock::Text(TextContent {
        text: text.to_string(),
        annotations: None,
        meta: None,
    })
}

fn message_chunk(text: &str) -> SessionUpdate {
    SessionUpdate::AgentMessageChunk(ContentChunk {
        content: text_block(text),
        message_id: None,
        meta: None,
    })
}

fn user_chunk(text: &str) -> SessionUpdate {
    SessionUpdate::UserMessageChunk(ContentChunk {
        content: text_block(text),
        message_id: None,
        meta: None,
    })
}

fn prompt_first_text(params: &PromptParams) -> &str {
    params
        .prompt
        .first()
        .and_then(|block| match block {
            ContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .unwrap_or_default()
}

// =============================================================================
// Test 1: Full ACP lifecycle (initialize, auth, sessions, modes, config)
// =============================================================================

struct LifecycleAgent {
    events: Mutex<Vec<String>>,
    initialized_rx: (async_channel::Sender<()>, async_channel::Receiver<()>),
}

impl LifecycleAgent {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            events: Mutex::new(Vec::new()),
            initialized_rx: async_channel::bounded(1),
        })
    }

    fn record(&self, event: impl Into<String>) {
        self.events.lock().expect("events").push(event.into());
    }
}

struct SharedHandler<H>(Arc<H>);

impl AgentHandler for SharedHandler<LifecycleAgent> {
    fn initialize(
        &self,
        _peer: &ClientPeer,
        params: Option<InitializeParams>,
    ) -> impl Future<Output = Result<InitializeResult, JsonRpcError>> + Send {
        let client_name = params
            .and_then(|p| p.client_info)
            .map_or_else(|| "none".to_string(), |info| info.name);
        self.0.record(format!("initialize:{client_name}"));
        std::future::ready(Ok(InitializeResult {
            protocol_version: PROTOCOL_VERSION,
            agent_capabilities: AgentCapabilities {
                load_session: true,
                ..AgentCapabilities::default()
            },
            agent_info: Some(Implementation {
                name: "lifecycle-agent".to_string(),
                title: Some("Lifecycle Agent".to_string()),
                version: "0.1.0".to_string(),
            }),
            auth_methods: vec![],
            meta: None,
        }))
    }

    fn initialized(&self, _peer: &ClientPeer) -> impl Future<Output = ()> + Send {
        self.0.record("initialized");
        let _ = self.0.initialized_rx.0.try_send(());
        std::future::ready(())
    }

    fn authenticate(
        &self,
        _peer: &ClientPeer,
        params: AuthenticateParams,
    ) -> impl Future<Output = Result<AuthenticateResult, JsonRpcError>> + Send {
        self.0.record(format!("authenticate:{}", params.method_id));
        std::future::ready(Ok(AuthenticateResult::default()))
    }

    fn logout(
        &self,
        _peer: &ClientPeer,
        _params: LogoutParams,
    ) -> impl Future<Output = Result<LogoutResult, JsonRpcError>> + Send {
        self.0.record("logout");
        std::future::ready(Ok(LogoutResult::default()))
    }

    fn new_session(
        &self,
        _peer: &ClientPeer,
        params: SessionNewParams,
    ) -> impl Future<Output = Result<SessionNewResult, JsonRpcError>> + Send {
        self.0
            .record(format!("new_session:{}", params.cwd.display()));
        std::future::ready(Ok(SessionNewResult {
            session_id: "sess-1".to_string(),
            modes: Some(SessionModeState {
                current_mode_id: "default".to_string(),
                available_modes: vec![
                    SessionMode {
                        id: "default".to_string(),
                        name: "Default".to_string(),
                        description: None,
                        meta: None,
                    },
                    SessionMode {
                        id: "bypass".to_string(),
                        name: "Bypass".to_string(),
                        description: None,
                        meta: None,
                    },
                ],
                meta: None,
            }),
            config_options: None,
            meta: None,
            extra: BTreeMap::new(),
        }))
    }

    fn load_session(
        &self,
        peer: &ClientPeer,
        params: SessionLoadParams,
    ) -> impl Future<Output = Result<SessionLoadResult, JsonRpcError>> + Send {
        let peer = peer.clone();
        let state = self.0.clone();
        async move {
            state.record(format!("load_session:{}", params.session_id));
            peer.session_update(SessionNotification::new(
                &params.session_id,
                user_chunk("earlier question"),
            ))
            .await
            .map_err(|err| JsonRpcError::internal_error(err.to_string()))?;
            peer.session_update(SessionNotification::new(
                &params.session_id,
                message_chunk("earlier answer"),
            ))
            .await
            .map_err(|err| JsonRpcError::internal_error(err.to_string()))?;
            Ok(SessionLoadResult::default())
        }
    }

    fn resume_session(
        &self,
        _peer: &ClientPeer,
        params: SessionResumeParams,
    ) -> impl Future<Output = Result<SessionResumeResult, JsonRpcError>> + Send {
        self.0
            .record(format!("resume_session:{}", params.session_id));
        std::future::ready(Ok(SessionResumeResult::default()))
    }

    fn list_sessions(
        &self,
        _peer: &ClientPeer,
        params: SessionListParams,
    ) -> impl Future<Output = Result<SessionListResult, JsonRpcError>> + Send {
        self.0.record("list_sessions");
        let cwd = params.cwd.unwrap_or_else(|| PathBuf::from("/workspace"));
        std::future::ready(Ok(SessionListResult {
            sessions: vec![SessionInfo {
                session_id: "sess-cjk".to_string(),
                cwd,
                additional_directories: Vec::new(),
                // CJK coverage: Chinese, Japanese, and Korean together.
                title: Some("会话标题 / セッション / 세션".to_string()),
                updated_at: Some("2026-01-01T00:00:00Z".to_string()),
                meta: None,
                extra: BTreeMap::new(),
            }],
            next_cursor: None,
            meta: None,
        }))
    }

    fn set_mode(
        &self,
        _peer: &ClientPeer,
        params: SessionSetModeParams,
    ) -> impl Future<Output = Result<SessionSetModeResult, JsonRpcError>> + Send {
        self.0
            .record(format!("set_mode:{}:{}", params.session_id, params.mode_id));
        std::future::ready(Ok(SessionSetModeResult::default()))
    }

    fn set_config_option(
        &self,
        _peer: &ClientPeer,
        params: SessionSetConfigOptionParams,
    ) -> impl Future<Output = Result<SessionSetConfigOptionResult, JsonRpcError>> + Send {
        let current_value = match params.value {
            SessionConfigValue::Select { value } => ConfigOptionValue::Selected(value),
            SessionConfigValue::Boolean { value, .. } => ConfigOptionValue::Toggle(value),
        };
        self.0
            .record(format!("set_config_option:{}", params.config_id));
        std::future::ready(Ok(SessionSetConfigOptionResult {
            config_options: vec![ConfigOption {
                id: params.config_id,
                name: "Option".to_string(),
                description: None,
                category: None,
                kind: None,
                current_value: Some(current_value),
                options: None,
                meta: None,
            }],
            meta: None,
        }))
    }

    fn close_session(
        &self,
        _peer: &ClientPeer,
        params: SessionCloseParams,
    ) -> impl Future<Output = Result<SessionCloseResult, JsonRpcError>> + Send {
        self.0
            .record(format!("close_session:{}", params.session_id));
        std::future::ready(Ok(SessionCloseResult::default()))
    }

    fn delete_session(
        &self,
        _peer: &ClientPeer,
        params: SessionDeleteParams,
    ) -> impl Future<Output = Result<SessionDeleteResult, JsonRpcError>> + Send {
        self.0
            .record(format!("delete_session:{}", params.session_id));
        std::future::ready(Ok(SessionDeleteResult::default()))
    }
}

#[derive(Default)]
struct RecordingClient {
    updates: Mutex<Vec<SessionNotification>>,
    completions: Mutex<Vec<String>>,
    notifications: Mutex<Vec<JsonRpcNotification>>,
}

impl ClientHandler for RecordingClient {
    fn session_update(&self, notification: SessionNotification) -> impl Future<Output = ()> + Send {
        self.updates.lock().expect("updates").push(notification);
        std::future::ready(())
    }

    fn request_permission(
        &self,
        _params: RequestPermissionParams,
    ) -> impl Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send {
        std::future::ready(Ok(RequestPermissionResult {
            outcome: RequestPermissionOutcome::Cancelled,
            meta: None,
        }))
    }

    fn elicitation_complete(
        &self,
        params: ElicitationCompleteParams,
    ) -> impl Future<Output = ()> + Send {
        self.completions
            .lock()
            .expect("completions")
            .push(params.elicitation_id);
        std::future::ready(())
    }

    fn notification(&self, notification: JsonRpcNotification) -> impl Future<Output = ()> + Send {
        self.notifications
            .lock()
            .expect("notifications")
            .push(notification);
        std::future::ready(())
    }
}

async fn exercise_session_controls(client: &AcpClient<RecordingClient>) {
    client
        .set_mode(SessionSetModeParams::new("sess-1", "bypass"))
        .await
        .expect("session/set_mode");

    let updated_select = client
        .set_config_option(SessionSetConfigOptionParams::new(
            "sess-1",
            "model",
            "claude-sonnet",
        ))
        .await
        .expect("set select config");
    assert_eq!(
        updated_select[0].current_value,
        Some(ConfigOptionValue::Selected("claude-sonnet".to_string()))
    );

    let updated_bool = client
        .set_config_option(SessionSetConfigOptionParams::new(
            "sess-1", "thinking", true,
        ))
        .await
        .expect("set bool config");
    assert_eq!(
        updated_bool[0].current_value,
        Some(ConfigOptionValue::Toggle(true))
    );

    client.close_session("sess-1").await.expect("session/close");
    client
        .delete_session("sess-old")
        .await
        .expect("session/delete");
}

#[tokio::test]
async fn lifecycle_initialize_auth_sessions_modes_and_config_options() {
    let (client_end, server_end) = DuplexTransport::pair();
    let state = LifecycleAgent::new();
    let (_peer, server_driver) = serve(server_end, SharedHandler(state.clone()));
    let server_task = tokio::spawn(server_driver);

    let (client, client_driver) = AcpClient::connect(client_end, RecordingClient::default());
    let client = client.with_client_info(Implementation {
        name: "zed-test".to_string(),
        title: None,
        version: "1.0.0".to_string(),
    });
    let client_task = tokio::spawn(client_driver);

    let init = client.initialize().await.expect("initialize");
    assert_eq!(init.protocol_version, PROTOCOL_VERSION);
    assert!(init.agent_capabilities.load_session);
    assert_eq!(
        init.agent_info.as_ref().map(|i| i.name.as_str()),
        Some("lifecycle-agent")
    );

    client.authenticate("oauth").await.expect("authenticate");
    client.logout(LogoutParams::new()).await.expect("logout");
    let _: LogoutResult = client
        .request("logout", &Value::Null)
        .await
        .expect("logout with null params");

    let session = client
        .new_session(SessionNewParams::new("/tmp/project"))
        .await
        .expect("session/new");
    assert_eq!(session.session_id, "sess-1");

    // Replayed updates must arrive at the client BEFORE `load_session` resolves.
    client
        .load_session(SessionLoadParams::new("sess-old", "/tmp/project"))
        .await
        .expect("session/load");
    assert_eq!(
        client.handler().updates.lock().expect("updates").len(),
        2,
        "load_session updates must flush before SessionLoadResult"
    );

    client
        .resume_session(SessionResumeParams::new("sess-old", "/tmp/project"))
        .await
        .expect("session/resume");

    let listed = client
        .list_sessions(SessionListParams::new().cwd("/tmp/project"))
        .await
        .expect("session/list");
    assert_eq!(listed.sessions.len(), 1);
    assert_eq!(
        listed.sessions[0].title.as_deref(),
        Some("会话标题 / セッション / 세션")
    );

    exercise_session_controls(&client).await;

    client.close();
    client_task.await.expect("client task");
    server_task.await.expect("server task");

    let events = state.events.lock().expect("events").clone();
    assert_eq!(
        events,
        vec![
            "initialize:zed-test",
            "authenticate:oauth",
            "logout",
            "logout",
            "new_session:/tmp/project",
            "load_session:sess-old",
            "resume_session:sess-old",
            "list_sessions",
            "set_mode:sess-1:bypass",
            "set_config_option:model",
            "set_config_option:thinking",
            "close_session:sess-1",
            "delete_session:sess-old",
        ]
    );
}

// =============================================================================
// Test 2: Minimal handler defaults (method_not_found + invalid_params + no-op)
// =============================================================================

struct MinimalHandler;

impl AgentHandler for MinimalHandler {
    fn initialize(
        &self,
        _peer: &ClientPeer,
        _params: Option<InitializeParams>,
    ) -> impl Future<Output = Result<InitializeResult, JsonRpcError>> + Send {
        std::future::ready(Ok(InitializeResult {
            protocol_version: PROTOCOL_VERSION,
            agent_capabilities: AgentCapabilities::default(),
            agent_info: None,
            auth_methods: vec![],
            meta: None,
        }))
    }
}

fn assert_method_not_found<T: std::fmt::Debug>(result: Result<T, ClientError>) {
    match result {
        Err(ClientError::JsonRpc(error)) => {
            assert_eq!(error.code, ErrorCode::METHOD_NOT_FOUND, "{error:?}");
        }
        other => panic!("expected METHOD_NOT_FOUND, got {other:?}"),
    }
}

#[tokio::test]
async fn default_agent_handler_methods_report_method_not_found_and_ignore_notifications() {
    let (client_end, server_end) = DuplexTransport::pair();
    let (_peer, server_driver) = serve(server_end, MinimalHandler);
    let server_task = tokio::spawn(server_driver);
    let (client, client_driver) = AcpClient::connect(client_end, RecordingClient::default());
    let client_task = tokio::spawn(client_driver);

    client.initialize().await.expect("initialize");

    assert_method_not_found(client.authenticate("x").await);
    assert_method_not_found(client.logout(LogoutParams::new()).await);
    assert_method_not_found(client.new_session(SessionNewParams::new("/tmp")).await);
    assert_method_not_found(
        client
            .load_session(SessionLoadParams::new("s1", "/tmp"))
            .await,
    );
    assert_method_not_found(
        client
            .resume_session(SessionResumeParams::new("s1", "/tmp"))
            .await,
    );
    assert_method_not_found(
        client
            .fork_session(SessionForkParams::new("s1", "/tmp"))
            .await,
    );
    assert_method_not_found(client.list_sessions(SessionListParams::new()).await);
    assert_method_not_found(client.close_session("s1").await);
    assert_method_not_found(client.delete_session("s1").await);
    assert_method_not_found(
        client
            .prompt(PromptParams::new("s1", vec![text_block("hi")]))
            .await,
    );
    assert_method_not_found(
        client
            .set_mode(SessionSetModeParams::new("s1", "plan"))
            .await,
    );
    assert_method_not_found(
        client
            .set_config_option(SessionSetConfigOptionParams::new("s1", "k", "v"))
            .await,
    );
    assert_method_not_found(
        client
            .ext_request::<_, Value>(&ExtMethod::new("_custom/unknown"), &json!({}))
            .await,
    );
    assert_method_not_found(
        client
            .request::<_, Value>("unknown/method", &json!({}))
            .await,
    );
    assert_method_not_found(aither_acp::vendor::codex::set_model(&client, "s1", "gpt-5").await);
    assert_method_not_found(aither_acp::vendor::codex::authentication_status(&client).await);
    assert_method_not_found(aither_acp::vendor::codex::authentication_logout(&client).await);

    // Malformed `initialize` params surface as INVALID_PARAMS.
    match client
        .request::<_, Value>("initialize", &json!({"protocolVersion": "not-a-u16"}))
        .await
    {
        Err(ClientError::JsonRpc(error)) => assert_eq!(error.code, ErrorCode::INVALID_PARAMS),
        other => panic!("expected INVALID_PARAMS, got {other:?}"),
    }

    // Notifications with default handlers are ignored cleanly without breaking
    // subsequent requests.
    client
        .cancel(SessionCancelParams::new("s1"))
        .await
        .expect("cancel");
    client
        .ext_notify(&ExtMethod::new("_custom/ping"), &json!({"ok": true}))
        .await
        .expect("ext_notify");
    client
        .initialize()
        .await
        .expect("initialize after notifications");

    client.close();
    client_task.await.expect("client task");
    server_task.await.expect("server task");
}

// =============================================================================
// Test 3: Live streaming updates + mid-turn `request_permission` round-trip
// =============================================================================

struct StreamingPermissionAgent {
    chunk_seen_rx: async_channel::Receiver<()>,
}

impl AgentHandler for StreamingPermissionAgent {
    fn initialize(
        &self,
        _peer: &ClientPeer,
        _params: Option<InitializeParams>,
    ) -> impl Future<Output = Result<InitializeResult, JsonRpcError>> + Send {
        std::future::ready(Ok(InitializeResult {
            protocol_version: PROTOCOL_VERSION,
            agent_capabilities: AgentCapabilities::default(),
            agent_info: None,
            auth_methods: vec![],
            meta: None,
        }))
    }

    fn prompt(
        &self,
        peer: &ClientPeer,
        params: PromptParams,
    ) -> impl Future<Output = Result<PromptResult, JsonRpcError>> + Send {
        let peer = peer.clone();
        let chunk_seen_rx = self.chunk_seen_rx.clone();
        async move {
            // Stream chunk 1 and wait for the client to confirm receipt BEFORE
            // continuing. If `serve` buffered updates until `prompt` returned,
            // this await would deadlock.
            peer.session_update(SessionNotification::new(
                &params.session_id,
                message_chunk("before-permission:"),
            ))
            .await
            .map_err(|e| JsonRpcError::internal_error(e.to_string()))?;

            chunk_seen_rx
                .recv()
                .await
                .expect("client must observe first chunk mid-turn");

            let decision = peer
                .request_permission(RequestPermissionParams {
                    session_id: params.session_id.clone(),
                    tool_call: ToolCall {
                        tool_call_id: "tc-42".to_string(),
                        title: "Run tests".to_string(),
                        kind: None,
                        status: Some(ToolCallStatus::Pending),
                        content: vec![],
                        locations: vec![],
                        raw_input: Some(json!({"cmd": "cargo nextest run"})),
                        raw_output: None,
                        meta: None,
                    },
                    options: vec![PermissionOption {
                        option_id: "allow-once".to_string(),
                        name: "Allow once".to_string(),
                        kind: PermissionOptionKind::AllowOnce,
                        meta: None,
                    }],
                    meta: None,
                })
                .await
                .map_err(|e| JsonRpcError::internal_error(e.to_string()))?;

            let chosen = match decision.outcome {
                RequestPermissionOutcome::Selected { option_id } => option_id,
                RequestPermissionOutcome::Cancelled => "cancelled".to_string(),
            };

            peer.session_update(SessionNotification::new(
                &params.session_id,
                message_chunk(&chosen),
            ))
            .await
            .map_err(|e| JsonRpcError::internal_error(e.to_string()))?;

            Ok(PromptResult {
                stop_reason: StopReason::EndTurn,
                meta: None,
            })
        }
    }
}

struct ApprovingClient {
    updates: Mutex<Vec<String>>,
    chunk_seen_tx: async_channel::Sender<()>,
    permissions: Mutex<Vec<RequestPermissionParams>>,
}

impl ClientHandler for ApprovingClient {
    fn session_update(&self, notification: SessionNotification) -> impl Future<Output = ()> + Send {
        if let SessionUpdate::AgentMessageChunk(chunk) = notification.update
            && let ContentBlock::Text(text) = chunk.content
        {
            self.updates.lock().expect("updates").push(text.text);
            let _ = self.chunk_seen_tx.try_send(());
        }
        std::future::ready(())
    }

    fn request_permission(
        &self,
        params: RequestPermissionParams,
    ) -> impl Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send {
        let option_id = params.options[0].option_id.clone();
        self.permissions.lock().expect("permissions").push(params);
        async move {
            Ok(RequestPermissionResult {
                outcome: RequestPermissionOutcome::Selected { option_id },
                meta: None,
            })
        }
    }
}

#[tokio::test]
async fn prompt_streams_live_updates_and_round_trips_mid_turn_permission() {
    let (chunk_seen_tx, chunk_seen_rx) = async_channel::bounded(2);
    let (client_end, server_end) = DuplexTransport::pair();
    let (_peer, server_driver) = serve(server_end, StreamingPermissionAgent { chunk_seen_rx });
    let server_task = tokio::spawn(server_driver);

    let (client, client_driver) = AcpClient::connect(
        client_end,
        ApprovingClient {
            updates: Mutex::new(Vec::new()),
            chunk_seen_tx,
            permissions: Mutex::new(Vec::new()),
        },
    );
    let client_task = tokio::spawn(client_driver);

    let result = tokio::time::timeout(
        Duration::from_secs(5),
        client.prompt(PromptParams::new("sess-live", vec![text_block("go")])),
    )
    .await
    .expect("prompt must not deadlock")
    .expect("prompt succeeded");

    assert_eq!(result.stop_reason, StopReason::EndTurn);
    assert_eq!(
        client.handler().updates.lock().expect("updates").as_slice(),
        ["before-permission:", "allow-once"]
    );
    assert_eq!(
        client.handler().permissions.lock().expect("permissions")[0]
            .tool_call
            .tool_call_id,
        "tc-42"
    );

    client.close();
    client_task.await.expect("client task");
    server_task.await.expect("server task");
}

// =============================================================================
// Test 4: Callbacks (`fs`, `terminal`, `elicitation`, `ext`) + failure paths
// =============================================================================

type ExtNotificationLog = Arc<Mutex<Vec<(String, Option<Value>)>>>;

async fn exercise_fs_and_terminal_callbacks(peer: &ClientPeer, sid: &str) {
    let read = peer
        .read_text_file(ReadTextFileParams {
            session_id: sid.to_string(),
            path: PathBuf::from("/workspace/readme.txt"),
            line: Some(1),
            limit: Some(10),
            meta: None,
        })
        .await
        .expect("read_text_file");
    assert_eq!(read.content, "你好 / こんにちは / 안녕하세요");

    peer.write_text_file(WriteTextFileParams {
        session_id: sid.to_string(),
        path: PathBuf::from("/workspace/out.txt"),
        content: read.content,
        meta: None,
    })
    .await
    .expect("write_text_file");

    let created = peer
        .terminal_create(TerminalCreateParams {
            session_id: sid.to_string(),
            command: "echo".to_string(),
            args: vec!["ok".to_string()],
            env: vec![],
            cwd: Some(PathBuf::from("/workspace")),
            output_byte_limit: Some(4096),
            meta: None,
        })
        .await
        .expect("terminal_create");
    assert_eq!(created.terminal_id, "term-1");

    let output = peer
        .terminal_output(TerminalOutputParams {
            session_id: sid.to_string(),
            terminal_id: created.terminal_id.clone(),
            meta: None,
        })
        .await
        .expect("terminal_output");
    assert_eq!(output.output, "ok\n");
    assert!(!output.truncated);

    let exit = peer
        .terminal_wait_for_exit(TerminalWaitForExitParams {
            session_id: sid.to_string(),
            terminal_id: created.terminal_id.clone(),
            meta: None,
        })
        .await
        .expect("terminal_wait_for_exit");
    assert_eq!(exit.exit_code, Some(0));

    peer.terminal_kill(TerminalKillParams {
        session_id: sid.to_string(),
        terminal_id: created.terminal_id.clone(),
        meta: None,
    })
    .await
    .expect("terminal_kill");

    peer.terminal_release(TerminalReleaseParams {
        session_id: sid.to_string(),
        terminal_id: created.terminal_id,
        meta: None,
    })
    .await
    .expect("terminal_release");
}

async fn exercise_elicitation_and_ext_callbacks(peer: &ClientPeer, sid: &str) {
    let form_res = peer
        .elicitation_create(ElicitationCreateParams::new(
            ElicitationFormMode::new(ElicitationSessionScope::new(sid), ElicitationSchema::new()),
            "Provide details",
        ))
        .await
        .expect("elicitation form");
    assert!(matches!(form_res.action, ElicitationAction::Accept(_)));

    let _url_res = peer
        .elicitation_create(ElicitationCreateParams::new(
            ElicitationUrlMode::new(
                ElicitationSessionScope::new(sid),
                "elic-99",
                "https://example.com/auth",
            ),
            "Complete sign-in",
        ))
        .await
        .expect("elicitation url");

    peer.elicitation_complete(ElicitationCompleteParams {
        elicitation_id: "elic-99".to_string(),
        meta: None,
    })
    .await
    .expect("elicitation_complete");

    let ext_val = peer
        .ext_request(ExtMethod::new("_bridge/probe"), Some(json!({"ping": 1})))
        .await
        .expect("ext_request with params");
    assert_eq!(ext_val["pong"], 1);

    let ext_none = peer
        .ext_request(&ExtMethod::new("_bridge/empty"), None)
        .await
        .expect("ext_request without params");
    assert_eq!(ext_none["empty"], true);

    peer.ext_notify(ExtMethod::new("_bridge/event"), Some(json!({"step": 2})))
        .await
        .expect("ext_notify with params");
    peer.ext_notify(&ExtMethod::new("_bridge/bare"), None)
        .await
        .expect("ext_notify without params");
    peer.notify(JsonRpcNotification::with_params(
        "vendor/custom_notice",
        json!({"ok": true}),
    ))
    .await
    .expect("raw notify");
}

struct CallbackAgent {
    ext_notifications: ExtNotificationLog,
    raw_notifications: Arc<Mutex<Vec<String>>>,
}

impl AgentHandler for CallbackAgent {
    fn initialize(
        &self,
        _peer: &ClientPeer,
        _params: Option<InitializeParams>,
    ) -> impl Future<Output = Result<InitializeResult, JsonRpcError>> + Send {
        std::future::ready(Ok(InitializeResult {
            protocol_version: PROTOCOL_VERSION,
            agent_capabilities: AgentCapabilities::default(),
            agent_info: None,
            auth_methods: vec![],
            meta: None,
        }))
    }

    fn prompt(
        &self,
        peer: &ClientPeer,
        params: PromptParams,
    ) -> impl Future<Output = Result<PromptResult, JsonRpcError>> + Send {
        let peer = peer.clone();
        let mode = prompt_first_text(&params).to_string();
        async move {
            let sid = &params.session_id;
            if mode == "client_error" {
                let err = peer
                    .read_text_file(ReadTextFileParams {
                        session_id: sid.clone(),
                        path: PathBuf::from("/denied"),
                        line: None,
                        limit: None,
                        meta: None,
                    })
                    .await
                    .expect_err("client should reject /denied");
                assert!(matches!(err, ServerError::JsonRpc(_)), "got {err:?}");
                return Ok(PromptResult {
                    stop_reason: StopReason::EndTurn,
                    meta: None,
                });
            }

            exercise_fs_and_terminal_callbacks(&peer, sid).await;
            exercise_elicitation_and_ext_callbacks(&peer, sid).await;

            Ok(PromptResult {
                stop_reason: StopReason::EndTurn,
                meta: None,
            })
        }
    }

    fn ext_request(
        &self,
        _peer: &ClientPeer,
        method: ExtMethod,
        params: Option<Value>,
    ) -> impl Future<Output = Result<Value, JsonRpcError>> + Send {
        std::future::ready(Ok(json!({
            "method": method.as_str(),
            "params": params,
        })))
    }

    fn ext_notify(
        &self,
        _peer: &ClientPeer,
        method: ExtMethod,
        params: Option<Value>,
    ) -> impl Future<Output = ()> + Send {
        self.ext_notifications
            .lock()
            .expect("ext_notifications")
            .push((method.as_str().to_string(), params));
        std::future::ready(())
    }

    fn notification(
        &self,
        _peer: &ClientPeer,
        notification: JsonRpcNotification,
    ) -> impl Future<Output = ()> + Send {
        self.raw_notifications
            .lock()
            .expect("raw_notifications")
            .push(notification.method);
        std::future::ready(())
    }
}

#[derive(Default)]
struct FullCallbackClient {
    written_files: Mutex<Vec<(PathBuf, String)>>,
    completions: Mutex<Vec<String>>,
    notifications: Mutex<Vec<(String, Option<Value>)>>,
    elicit_modes: Mutex<Vec<String>>,
}

impl ClientHandler for FullCallbackClient {
    fn session_update(
        &self,
        _notification: SessionNotification,
    ) -> impl Future<Output = ()> + Send {
        std::future::ready(())
    }

    fn request_permission(
        &self,
        _params: RequestPermissionParams,
    ) -> impl Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send {
        std::future::ready(Ok(RequestPermissionResult {
            outcome: RequestPermissionOutcome::Cancelled,
            meta: None,
        }))
    }

    fn read_text_file(
        &self,
        params: ReadTextFileParams,
    ) -> impl Future<Output = Result<ReadTextFileResult, JsonRpcError>> + Send {
        let result = if params.path == *"/denied" {
            Err(JsonRpcError::invalid_params("access denied"))
        } else {
            Ok(ReadTextFileResult {
                content: "你好 / こんにちは / 안녕하세요".to_string(),
                meta: None,
            })
        };
        std::future::ready(result)
    }

    fn write_text_file(
        &self,
        params: WriteTextFileParams,
    ) -> impl Future<Output = Result<WriteTextFileResult, JsonRpcError>> + Send {
        self.written_files
            .lock()
            .expect("written_files")
            .push((params.path, params.content));
        std::future::ready(Ok(WriteTextFileResult::default()))
    }

    fn terminal_create(
        &self,
        _params: TerminalCreateParams,
    ) -> impl Future<Output = Result<TerminalCreateResult, JsonRpcError>> + Send {
        std::future::ready(Ok(TerminalCreateResult {
            terminal_id: "term-1".to_string(),
            meta: None,
        }))
    }

    fn terminal_output(
        &self,
        _params: TerminalOutputParams,
    ) -> impl Future<Output = Result<TerminalOutputResult, JsonRpcError>> + Send {
        std::future::ready(Ok(TerminalOutputResult {
            output: "ok\n".to_string(),
            truncated: false,
            exit_status: None,
            meta: None,
        }))
    }

    fn terminal_wait_for_exit(
        &self,
        _params: TerminalWaitForExitParams,
    ) -> impl Future<Output = Result<TerminalExitStatus, JsonRpcError>> + Send {
        std::future::ready(Ok(TerminalExitStatus {
            exit_code: Some(0),
            signal: None,
            meta: None,
        }))
    }

    fn terminal_kill(
        &self,
        _params: TerminalKillParams,
    ) -> impl Future<Output = Result<TerminalKillResult, JsonRpcError>> + Send {
        std::future::ready(Ok(TerminalKillResult::default()))
    }

    fn terminal_release(
        &self,
        _params: TerminalReleaseParams,
    ) -> impl Future<Output = Result<TerminalReleaseResult, JsonRpcError>> + Send {
        std::future::ready(Ok(TerminalReleaseResult::default()))
    }

    fn elicitation_create(
        &self,
        params: ElicitationCreateParams,
    ) -> impl Future<Output = Result<ElicitationCreateResult, JsonRpcError>> + Send {
        let label = match &params.mode {
            ElicitationMode::Form(_) => "form",
            ElicitationMode::Url(_) => "url",
            ElicitationMode::Other(_) => "other",
        };
        self.elicit_modes
            .lock()
            .expect("elicit_modes")
            .push(label.to_string());
        std::future::ready(Ok(ElicitationCreateResult::accept(BTreeMap::new())))
    }

    fn elicitation_complete(
        &self,
        params: ElicitationCompleteParams,
    ) -> impl Future<Output = ()> + Send {
        self.completions
            .lock()
            .expect("completions")
            .push(params.elicitation_id);
        std::future::ready(())
    }

    fn ext_request(
        &self,
        method: ExtMethod,
        params: Option<Value>,
    ) -> impl Future<Output = Result<Value, JsonRpcError>> + Send {
        let reply = if method.as_str() == "_bridge/empty" {
            json!({"empty": params.is_none()})
        } else {
            json!({"pong": params.and_then(|p| p.get("ping").cloned())})
        };
        std::future::ready(Ok(reply))
    }

    fn notification(&self, notification: JsonRpcNotification) -> impl Future<Output = ()> + Send {
        self.notifications
            .lock()
            .expect("notifications")
            .push((notification.method, notification.params));
        std::future::ready(())
    }
}

fn assert_callback_records(
    client: &AcpClient<FullCallbackClient>,
    ext_notifications: &ExtNotificationLog,
    raw_notifications: &Arc<Mutex<Vec<String>>>,
) {
    assert_eq!(
        client
            .handler()
            .written_files
            .lock()
            .expect("written")
            .as_slice(),
        [(
            PathBuf::from("/workspace/out.txt"),
            "你好 / こんにちは / 안녕하세요".to_string()
        )]
    );
    assert_eq!(
        client
            .handler()
            .elicit_modes
            .lock()
            .expect("elicit_modes")
            .as_slice(),
        ["form", "url"]
    );
    assert_eq!(
        client
            .handler()
            .completions
            .lock()
            .expect("completions")
            .as_slice(),
        ["elic-99"]
    );

    let client_notifs = client
        .handler()
        .notifications
        .lock()
        .expect("notifications")
        .clone();
    assert_eq!(client_notifs.len(), 3);
    assert_eq!(client_notifs[0].0, "_bridge/event");
    assert_eq!(client_notifs[1], ("_bridge/bare".to_string(), None));
    assert_eq!(client_notifs[2].0, "vendor/custom_notice");

    assert_eq!(
        ext_notifications.lock().expect("ext_notifications")[0].0,
        "_editor/focus"
    );
    assert!(
        raw_notifications
            .lock()
            .expect("raw_notifications")
            .is_empty()
    );
}

#[tokio::test]
async fn callbacks_cover_fs_terminal_elicitation_and_extensions_both_directions() {
    let ext_notifications = Arc::new(Mutex::new(Vec::new()));
    let raw_notifications = Arc::new(Mutex::new(Vec::new()));
    let (client_end, server_end) = DuplexTransport::pair();
    let (peer, server_driver) = serve(
        server_end,
        CallbackAgent {
            ext_notifications: ext_notifications.clone(),
            raw_notifications: raw_notifications.clone(),
        },
    );
    let server_task = tokio::spawn(server_driver);

    let (client, client_driver) = AcpClient::connect(client_end, FullCallbackClient::default());
    let client_task = tokio::spawn(client_driver);

    client.initialize().await.expect("initialize");

    let res = client
        .prompt(PromptParams::new("sess-cb", vec![text_block("all")]))
        .await
        .expect("prompt all");
    assert_eq!(res.stop_reason, StopReason::EndTurn);

    let err_res = client
        .prompt(PromptParams::new(
            "sess-cb",
            vec![text_block("client_error")],
        ))
        .await
        .expect("prompt client_error");
    assert_eq!(err_res.stop_reason, StopReason::EndTurn);

    let echoed: Value = client
        .ext_request(&ExtMethod::new("_editor/state"), &json!({"openFiles": 3}))
        .await
        .expect("client ext_request");
    assert_eq!(echoed["method"], "_editor/state");
    assert_eq!(echoed["params"]["openFiles"], 3);

    client
        .ext_notify(&ExtMethod::new("_editor/focus"), &json!({"focused": true}))
        .await
        .expect("client ext_notify");

    let outside_read = peer
        .read_text_file(ReadTextFileParams {
            session_id: "sess-cb".to_string(),
            path: PathBuf::from("/workspace/outside.txt"),
            line: None,
            limit: None,
            meta: None,
        })
        .await
        .expect("peer read outside handler");
    assert_eq!(outside_read.content, "你好 / こんにちは / 안녕하세요");

    client.initialize().await.expect("barrier initialize");
    assert_callback_records(&client, &ext_notifications, &raw_notifications);

    client.close();
    client_task.await.expect("client task");
    server_task.await.expect("server task");
}

#[tokio::test]
async fn malformed_client_response_surfaces_as_protocol_error() {
    let (mut raw_client, server_end) = DuplexTransport::pair();
    let (peer, server_driver) = serve(server_end, MinimalHandler);
    let server_task = tokio::spawn(server_driver);

    let client_responder = tokio::spawn(async move {
        if let Ok(Some(JsonRpcMessage::Request(req))) = raw_client.recv().await {
            // `ReadTextFileResult` expects `{"content": "<string>"}`; send an
            // incompatible shape to exercise `ServerError::Protocol`.
            raw_client
                .respond(JsonRpcResponse::success(req.id, json!({"wrongField": 123})))
                .await
                .expect("respond");
        }
        raw_client
    });

    let err = peer
        .read_text_file(ReadTextFileParams {
            session_id: "s1".to_string(),
            path: PathBuf::from("/tmp/x"),
            line: None,
            limit: None,
            meta: None,
        })
        .await
        .expect_err("malformed response must fail");
    assert!(matches!(err, ServerError::Protocol(_)), "got {err:?}");

    let mut raw_client = client_responder.await.expect("responder");
    aither_mcp::transport::Transport::close(&mut raw_client)
        .await
        .expect("close");
    drop(raw_client);
    server_task.await.expect("server task");
}

// =============================================================================
// Test 5: Mid-turn `session/cancel` interrupts `prompt` blocked on permission
// =============================================================================

struct CancellablePermissionAgent {
    cancel_signal: (
        async_channel::Sender<String>,
        async_channel::Receiver<String>,
    ),
}

impl AgentHandler for CancellablePermissionAgent {
    fn initialize(
        &self,
        _peer: &ClientPeer,
        _params: Option<InitializeParams>,
    ) -> impl Future<Output = Result<InitializeResult, JsonRpcError>> + Send {
        std::future::ready(Ok(InitializeResult {
            protocol_version: PROTOCOL_VERSION,
            agent_capabilities: AgentCapabilities::default(),
            agent_info: None,
            auth_methods: vec![],
            meta: None,
        }))
    }

    fn prompt(
        &self,
        peer: &ClientPeer,
        params: PromptParams,
    ) -> impl Future<Output = Result<PromptResult, JsonRpcError>> + Send {
        let peer = peer.clone();
        let cancel_rx = self.cancel_signal.1.clone();
        async move {
            let perm_fut = peer.request_permission(RequestPermissionParams {
                session_id: params.session_id.clone(),
                tool_call: ToolCall {
                    tool_call_id: "tc-hang".to_string(),
                    title: "rm -rf build".to_string(),
                    kind: None,
                    status: Some(ToolCallStatus::Pending),
                    content: vec![],
                    locations: vec![],
                    raw_input: None,
                    raw_output: None,
                    meta: None,
                },
                options: vec![PermissionOption {
                    option_id: "allow".to_string(),
                    name: "Allow".to_string(),
                    kind: PermissionOptionKind::AllowOnce,
                    meta: None,
                }],
                meta: None,
            });

            let stop_reason = or(
                async move {
                    let _ = perm_fut.await;
                    StopReason::EndTurn
                },
                async move {
                    while let Ok(cancelled_id) = cancel_rx.recv().await {
                        if cancelled_id == params.session_id {
                            return StopReason::Cancelled;
                        }
                    }
                    StopReason::Error
                },
            )
            .await;

            Ok(PromptResult {
                stop_reason,
                meta: None,
            })
        }
    }

    fn cancel(
        &self,
        _peer: &ClientPeer,
        params: SessionCancelParams,
    ) -> impl Future<Output = ()> + Send {
        let _ = self.cancel_signal.0.try_send(params.session_id);
        std::future::ready(())
    }
}

struct HangingPermissionClient {
    permission_entered_tx: async_channel::Sender<()>,
    never_rx: async_channel::Receiver<()>,
}

impl ClientHandler for HangingPermissionClient {
    fn session_update(
        &self,
        _notification: SessionNotification,
    ) -> impl Future<Output = ()> + Send {
        std::future::ready(())
    }

    fn request_permission(
        &self,
        _params: RequestPermissionParams,
    ) -> impl Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send {
        let entered = self.permission_entered_tx.clone();
        let never_rx = self.never_rx.clone();
        async move {
            let _ = entered.send(()).await;
            let _ = never_rx.recv().await;
            Ok(RequestPermissionResult {
                outcome: RequestPermissionOutcome::Cancelled,
                meta: None,
            })
        }
    }
}

#[tokio::test]
async fn mid_turn_cancel_interrupts_prompt_blocked_on_permission_without_deadlock() {
    let (perm_entered_tx, perm_entered_rx) = async_channel::bounded(1);
    let (_never_tx, never_rx) = async_channel::bounded(1);
    let (client_end, server_end) = DuplexTransport::pair();

    let (_peer, server_driver) = serve(
        server_end,
        CancellablePermissionAgent {
            cancel_signal: async_channel::unbounded(),
        },
    );
    let server_task = tokio::spawn(server_driver);

    let (client, client_driver) = AcpClient::connect(
        client_end,
        HangingPermissionClient {
            permission_entered_tx: perm_entered_tx,
            never_rx,
        },
    );
    let client_task = tokio::spawn(client_driver);

    let prompt_future = client
        .start_prompt(&PromptParams::new("sess-cancel", vec![text_block("run")]))
        .await
        .expect("start_prompt");
    let prompt_handle = tokio::spawn(prompt_future);

    // Wait until `prompt` has sent `session/request_permission` and the client
    // handler is parked on `never_rx`.
    tokio::time::timeout(Duration::from_secs(5), perm_entered_rx.recv())
        .await
        .expect("permission request must reach client")
        .expect("channel open");

    client
        .cancel(SessionCancelParams::new("sess-cancel"))
        .await
        .expect("send session/cancel");

    let result = tokio::time::timeout(Duration::from_secs(5), prompt_handle)
        .await
        .expect("mid-turn session/cancel must not head-of-line block")
        .expect("prompt task panicked")
        .expect("prompt result");
    assert_eq!(result.stop_reason, StopReason::Cancelled);

    client.close();
    client_task.await.expect("client task");
    server_task.await.expect("server task");
}

// =============================================================================
// Test 6: Transport close fails all pending `ClientPeer` requests immediately
// =============================================================================

struct SpawnedPeerRequestAgent {
    outcome_tx: async_channel::Sender<Result<RequestPermissionResult, ServerError>>,
}

impl AgentHandler for SpawnedPeerRequestAgent {
    fn initialize(
        &self,
        _peer: &ClientPeer,
        _params: Option<InitializeParams>,
    ) -> impl Future<Output = Result<InitializeResult, JsonRpcError>> + Send {
        std::future::ready(Ok(InitializeResult {
            protocol_version: PROTOCOL_VERSION,
            agent_capabilities: AgentCapabilities::default(),
            agent_info: None,
            auth_methods: vec![],
            meta: None,
        }))
    }

    fn prompt(
        &self,
        peer: &ClientPeer,
        params: PromptParams,
    ) -> impl Future<Output = Result<PromptResult, JsonRpcError>> + Send {
        let peer = peer.clone();
        let outcome_tx = self.outcome_tx.clone();
        // Spawn an independent task holding a cloned `ClientPeer` (mirroring an
        // external runtime bridge's session task) AND await inside `prompt`.
        tokio::spawn(async move {
            let res = peer
                .request_permission(RequestPermissionParams {
                    session_id: params.session_id,
                    tool_call: ToolCall {
                        tool_call_id: "tc-drop".to_string(),
                        title: "dropped".to_string(),
                        kind: None,
                        status: None,
                        content: vec![],
                        locations: vec![],
                        raw_input: None,
                        raw_output: None,
                        meta: None,
                    },
                    options: vec![],
                    meta: None,
                })
                .await;
            let _ = outcome_tx.send(res).await;
        });
        std::future::pending()
    }
}

struct HangingFsAndPermissionClient {
    entered_tx: async_channel::Sender<&'static str>,
    never_rx: async_channel::Receiver<()>,
}

impl ClientHandler for HangingFsAndPermissionClient {
    fn session_update(
        &self,
        _notification: SessionNotification,
    ) -> impl Future<Output = ()> + Send {
        std::future::ready(())
    }

    fn request_permission(
        &self,
        _params: RequestPermissionParams,
    ) -> impl Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send {
        let entered_tx = self.entered_tx.clone();
        let never_rx = self.never_rx.clone();
        async move {
            let _ = entered_tx.send("perm").await;
            let _ = never_rx.recv().await;
            Err(JsonRpcError::internal_error("unreachable"))
        }
    }

    fn read_text_file(
        &self,
        _params: ReadTextFileParams,
    ) -> impl Future<Output = Result<ReadTextFileResult, JsonRpcError>> + Send {
        let entered_tx = self.entered_tx.clone();
        let never_rx = self.never_rx.clone();
        async move {
            let _ = entered_tx.send("fs").await;
            let _ = never_rx.recv().await;
            Err(JsonRpcError::internal_error("unreachable"))
        }
    }
}

#[tokio::test]
async fn transport_close_fails_all_pending_client_peer_requests_immediately() {
    let (entered_tx, entered_rx) = async_channel::bounded(2);
    let (_never_tx, never_rx) = async_channel::bounded(1);
    let (outcome_tx, outcome_rx) = async_channel::bounded(1);

    let (client_end, server_end) = DuplexTransport::pair();
    let (peer, server_driver) = serve(server_end, SpawnedPeerRequestAgent { outcome_tx });
    let server_task = tokio::spawn(server_driver);

    let (client, client_driver) = AcpClient::connect(
        client_end,
        HangingFsAndPermissionClient {
            entered_tx,
            never_rx,
        },
    );
    let client_task = tokio::spawn(client_driver);

    let _prompt_resp = client
        .start_prompt(&PromptParams::new("sess-drop", vec![text_block("go")]))
        .await
        .expect("start_prompt");

    let peer_clone = peer.clone();
    let fs_task = tokio::spawn(async move {
        peer_clone
            .read_text_file(ReadTextFileParams {
                session_id: "sess-drop".to_string(),
                path: PathBuf::from("/tmp/pending"),
                line: None,
                limit: None,
                meta: None,
            })
            .await
    });

    // Wait for both agent-to-client requests to be in flight on the wire.
    for _ in 0..2 {
        tokio::time::timeout(Duration::from_secs(5), entered_rx.recv())
            .await
            .expect("request must reach client")
            .expect("entered_rx open");
    }

    // Drop the client connection abruptly while both requests are unanswered.
    client.close();
    client_task.await.expect("client task");

    let perm_outcome = tokio::time::timeout(Duration::from_secs(5), outcome_rx.recv())
        .await
        .expect("pending request_permission must fail immediately")
        .expect("outcome_rx");
    assert!(
        matches!(perm_outcome, Err(ServerError::Closed)),
        "expected ServerError::Closed, got {perm_outcome:?}"
    );

    let fs_outcome = tokio::time::timeout(Duration::from_secs(5), fs_task)
        .await
        .expect("pending read_text_file must fail immediately")
        .expect("fs_task panicked");
    assert!(
        matches!(fs_outcome, Err(ServerError::Closed)),
        "expected ServerError::Closed, got {fs_outcome:?}"
    );

    tokio::time::timeout(Duration::from_secs(5), server_task)
        .await
        .expect("server driver must terminate on EOF")
        .expect("server task panicked");

    // Any subsequent calls on surviving `ClientPeer` clones also fail immediately.
    assert!(matches!(
        peer.session_update(SessionNotification::new("s", message_chunk("late")))
            .await,
        Err(ServerError::Closed)
    ));
    assert!(matches!(
        peer.read_text_file(ReadTextFileParams {
            session_id: "s".to_string(),
            path: PathBuf::from("/tmp/late"),
            line: None,
            limit: None,
            meta: None,
        })
        .await,
        Err(ServerError::Closed)
    ));
}

// =============================================================================
// Test 7: Byte-level LF-only NDJSON framing over `StreamTransport` with
// unescaped U+2028, U+2029, embedded `\n`, and CJK (Chinese/Japanese/Korean)
// =============================================================================

struct ChannelReader {
    rx: Pin<Box<async_channel::Receiver<Vec<u8>>>>,
    pending: Vec<u8>,
    offset: usize,
}

impl AsyncRead for ChannelReader {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        let this = &mut *self;
        while this.offset >= this.pending.len() {
            match futures_lite::stream::Stream::poll_next(this.rx.as_mut(), cx) {
                Poll::Ready(Some(chunk)) => {
                    this.pending = chunk;
                    this.offset = 0;
                }
                Poll::Ready(None) => return Poll::Ready(Ok(0)),
                Poll::Pending => return Poll::Pending,
            }
        }
        let available = &this.pending[this.offset..];
        let count = available.len().min(buf.len());
        buf[..count].copy_from_slice(&available[..count]);
        this.offset += count;
        Poll::Ready(Ok(count))
    }
}

struct TeeWriter {
    tx: async_channel::Sender<Vec<u8>>,
    wire_log: Arc<Mutex<Vec<u8>>>,
}

impl AsyncWrite for TeeWriter {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }
        self.wire_log
            .lock()
            .expect("wire_log")
            .extend_from_slice(buf);
        self.tx
            .try_send(buf.to_vec())
            .map(|()| buf.len())
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe closed"))
            .into()
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        self.tx.close();
        Poll::Ready(Ok(()))
    }
}

fn byte_pipe() -> (BufReader<ChannelReader>, TeeWriter, Arc<Mutex<Vec<u8>>>) {
    let (tx, rx) = async_channel::unbounded();
    let wire_log = Arc::new(Mutex::new(Vec::new()));
    (
        BufReader::new(ChannelReader {
            rx: Box::pin(rx),
            pending: Vec::new(),
            offset: 0,
        }),
        TeeWriter {
            tx,
            wire_log: wire_log.clone(),
        },
        wire_log,
    )
}

struct EchoNdjsonAgent;

impl AgentHandler for EchoNdjsonAgent {
    fn initialize(
        &self,
        _peer: &ClientPeer,
        _params: Option<InitializeParams>,
    ) -> impl Future<Output = Result<InitializeResult, JsonRpcError>> + Send {
        std::future::ready(Ok(InitializeResult {
            protocol_version: PROTOCOL_VERSION,
            agent_capabilities: AgentCapabilities::default(),
            agent_info: None,
            auth_methods: vec![],
            meta: None,
        }))
    }

    fn prompt(
        &self,
        peer: &ClientPeer,
        params: PromptParams,
    ) -> impl Future<Output = Result<PromptResult, JsonRpcError>> + Send {
        let peer = peer.clone();
        let echoed = prompt_first_text(&params).to_string();
        async move {
            let file = peer
                .read_text_file(ReadTextFileParams {
                    session_id: params.session_id.clone(),
                    path: PathBuf::from("/tmp/ndjson"),
                    line: None,
                    limit: None,
                    meta: None,
                })
                .await
                .map_err(|e| JsonRpcError::internal_error(e.to_string()))?;

            peer.session_update(SessionNotification::new(
                &params.session_id,
                message_chunk(&format!("{echoed}|{content}", content = file.content)),
            ))
            .await
            .map_err(|e| JsonRpcError::internal_error(e.to_string()))?;

            Ok(PromptResult {
                stop_reason: StopReason::EndTurn,
                meta: None,
            })
        }
    }
}

struct NdjsonClient {
    payload: String,
    updates: Mutex<Vec<String>>,
}

impl ClientHandler for NdjsonClient {
    fn session_update(&self, notification: SessionNotification) -> impl Future<Output = ()> + Send {
        if let SessionUpdate::AgentMessageChunk(chunk) = notification.update
            && let ContentBlock::Text(text) = chunk.content
        {
            self.updates.lock().expect("updates").push(text.text);
        }
        std::future::ready(())
    }

    fn request_permission(
        &self,
        _params: RequestPermissionParams,
    ) -> impl Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send {
        std::future::ready(Ok(RequestPermissionResult {
            outcome: RequestPermissionOutcome::Cancelled,
            meta: None,
        }))
    }

    fn read_text_file(
        &self,
        _params: ReadTextFileParams,
    ) -> impl Future<Output = Result<ReadTextFileResult, JsonRpcError>> + Send {
        std::future::ready(Ok(ReadTextFileResult {
            content: self.payload.clone(),
            meta: None,
        }))
    }
}

#[tokio::test]
async fn ndjson_stream_transport_preserves_u2028_u2029_and_cjk_in_both_directions() {
    // Contains Chinese, Japanese, Korean, unescaped U+2028 (LINE SEPARATOR),
    // unescaped U+2029 (PARAGRAPH SEPARATOR), and an embedded LF (`\n`).
    let tricky = "中文\u{2028}日本語\u{2029}한국어\nsecond line";

    let (server_reader, client_writer, c2s_bytes) = byte_pipe();
    let (client_reader, server_writer, s2c_bytes) = byte_pipe();

    let client_transport = StreamTransport::new(client_reader, client_writer);
    let server_transport = StreamTransport::new(server_reader, server_writer);

    let (peer, server_driver) = serve(server_transport, EchoNdjsonAgent);
    let server_task = tokio::spawn(server_driver);

    let (client, client_driver) = AcpClient::connect(
        client_transport,
        NdjsonClient {
            payload: tricky.to_string(),
            updates: Mutex::new(Vec::new()),
        },
    );
    let client_task = tokio::spawn(client_driver);

    client.initialize().await.expect("initialize");
    let result = client
        .prompt(PromptParams::new("sess-ndjson", vec![text_block(tricky)]))
        .await
        .expect("prompt");
    assert_eq!(result.stop_reason, StopReason::EndTurn);

    let expected_combined = format!("{tricky}|{tricky}");
    assert_eq!(
        client.handler().updates.lock().expect("updates").as_slice(),
        [expected_combined]
    );

    // Verify raw NDJSON framing: every line separated by `\n` parses as a
    // complete JSON object while preserving raw UTF-8 U+2028 / U+2029 inside
    // JSON string literals.
    let c2s_raw = std::str::from_utf8(&c2s_bytes.lock().expect("c2s"))
        .expect("utf8")
        .to_string();
    let s2c_raw = std::str::from_utf8(&s2c_bytes.lock().expect("s2c"))
        .expect("utf8")
        .to_string();
    for raw in [c2s_raw, s2c_raw] {
        assert!(
            raw.contains('\u{2028}') && raw.contains('\u{2029}'),
            "wire bytes must carry unescaped U+2028 and U+2029: {raw:?}"
        );
        for line in raw.split('\n').filter(|line| !line.is_empty()) {
            serde_json::from_str::<Value>(line)
                .unwrap_or_else(|e| panic!("NDJSON frame split mid-message: {e}: {line:?}"));
        }
    }

    peer.close();
    client.close();
    client_task.await.expect("client task");
    server_task.await.expect("server task");
}

// =============================================================================
// Test 8: `session/fork` round-trip with causal `session/update` replay
// =============================================================================

struct ForkingAgent {
    observed: Arc<Mutex<Vec<SessionForkParams>>>,
}

impl AgentHandler for ForkingAgent {
    fn initialize(
        &self,
        _peer: &ClientPeer,
        _params: Option<InitializeParams>,
    ) -> impl Future<Output = Result<InitializeResult, JsonRpcError>> + Send {
        std::future::ready(Ok(InitializeResult {
            protocol_version: PROTOCOL_VERSION,
            agent_capabilities: AgentCapabilities {
                session_capabilities: SessionCapabilities {
                    fork: Some(SessionForkCapabilities::default()),
                    additional_directories: Some(
                        SessionAdditionalDirectoriesCapabilities::default(),
                    ),
                    ..SessionCapabilities::default()
                },
                ..AgentCapabilities::default()
            },
            agent_info: None,
            auth_methods: vec![],
            meta: None,
        }))
    }

    fn fork_session(
        &self,
        peer: &ClientPeer,
        params: SessionForkParams,
    ) -> impl Future<Output = Result<SessionForkResult, JsonRpcError>> + Send {
        let peer = peer.clone();
        let observed = self.observed.clone();
        async move {
            let forked_id = format!("{}-fork-1", params.session_id);
            observed.lock().expect("observed").push(params);

            peer.session_update(SessionNotification::new(
                &forked_id,
                user_chunk("replayed question"),
            ))
            .await
            .map_err(|err| JsonRpcError::internal_error(err.to_string()))?;
            peer.session_update(SessionNotification::new(
                &forked_id,
                message_chunk("replayed answer"),
            ))
            .await
            .map_err(|err| JsonRpcError::internal_error(err.to_string()))?;

            Ok(SessionForkResult {
                session_id: forked_id,
                modes: Some(SessionModeState {
                    current_mode_id: "code".to_string(),
                    available_modes: vec![SessionMode {
                        id: "code".to_string(),
                        name: "Code".to_string(),
                        description: None,
                        meta: None,
                    }],
                    meta: None,
                }),
                config_options: Some(vec![ConfigOption {
                    id: "model".to_string(),
                    name: "Model".to_string(),
                    description: None,
                    category: None,
                    kind: None,
                    current_value: Some(ConfigOptionValue::Selected("gpt-5".to_string())),
                    options: None,
                    meta: None,
                }]),
                meta: Some(json!({"forked": true})),
                extra: BTreeMap::from([(
                    aither_acp::vendor::codex::MODELS.to_string(),
                    json!({
                        "availableModels": [{"modelId": "gpt-5", "name": "GPT-5"}],
                        "currentModelId": "gpt-5",
                    }),
                )]),
            })
        }
    }
}

#[tokio::test]
async fn fork_session_replays_updates_before_result_and_preserves_fields() {
    let observed = Arc::new(Mutex::new(Vec::new()));
    let (client_end, server_end) = DuplexTransport::pair();
    let (_peer, server_driver) = serve(
        server_end,
        ForkingAgent {
            observed: observed.clone(),
        },
    );
    let server_task = tokio::spawn(server_driver);
    let (client, client_driver) = AcpClient::connect(client_end, RecordingClient::default());
    let client_task = tokio::spawn(client_driver);

    let init = client.initialize().await.expect("initialize");
    assert!(init.agent_capabilities.session_capabilities.fork.is_some());

    let forked = client
        .fork_session(
            SessionForkParams::new("sess-parent", "/tmp/forked")
                .mcp_servers(vec![McpServerStdio::new("fs", "/bin/echo").into()])
                .additional_directories(vec![PathBuf::from("/tmp/shared")])
                .meta(json!({"origin": "test"})),
        )
        .await
        .expect("session/fork");

    assert_eq!(forked.session_id, "sess-parent-fork-1");
    assert_eq!(
        forked.modes.as_ref().map(|m| m.current_mode_id.as_str()),
        Some("code")
    );
    assert_eq!(forked.config_options.as_ref().map(Vec::len), Some(1));
    assert_eq!(forked.meta, Some(json!({"forked": true})));
    let models = aither_acp::vendor::codex::models(&forked.extra).expect("codex models extra");
    assert_eq!(models.current_model_id.as_deref(), Some("gpt-5"));

    // Both replayed `session/update` notifications must arrive before
    // `SessionForkResult` resolves.
    {
        let updates = client.handler().updates.lock().expect("updates");
        assert_eq!(
            updates.len(),
            2,
            "fork_session updates must flush before SessionForkResult"
        );
        assert_eq!(updates[0].session_id, "sess-parent-fork-1");
        assert_eq!(updates[1].session_id, "sess-parent-fork-1");
    }

    {
        let recorded = observed.lock().expect("observed");
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].session_id, "sess-parent");
        assert_eq!(recorded[0].cwd, PathBuf::from("/tmp/forked"));
        assert_eq!(recorded[0].mcp_servers.len(), 1);
        assert_eq!(recorded[0].mcp_servers[0].name(), "fs");
        assert_eq!(
            recorded[0].additional_directories,
            vec![PathBuf::from("/tmp/shared")]
        );
        assert_eq!(recorded[0].meta, Some(json!({"origin": "test"})));
    }

    // Malformed `session/fork` params report `INVALID_PARAMS`.
    match client
        .request::<_, Value>("session/fork", &json!({"cwd": "/tmp/missing-session-id"}))
        .await
    {
        Err(ClientError::JsonRpc(error)) => assert_eq!(error.code, ErrorCode::INVALID_PARAMS),
        other => panic!("expected INVALID_PARAMS, got {other:?}"),
    }

    client.close();
    client_task.await.expect("client task");
    server_task.await.expect("server task");
}

// =============================================================================
// Test 9: `AgentHandler::request` fallback answers legacy Codex vendor methods
// =============================================================================

struct LegacyCodexAgent {
    selected_models: Arc<Mutex<Vec<(String, String)>>>,
    authenticated_email: Arc<Mutex<Option<String>>>,
}

impl AgentHandler for LegacyCodexAgent {
    fn initialize(
        &self,
        _peer: &ClientPeer,
        _params: Option<InitializeParams>,
    ) -> impl Future<Output = Result<InitializeResult, JsonRpcError>> + Send {
        std::future::ready(Ok(InitializeResult {
            protocol_version: PROTOCOL_VERSION,
            agent_capabilities: AgentCapabilities::default(),
            agent_info: None,
            auth_methods: vec![],
            meta: None,
        }))
    }

    fn request(
        &self,
        peer: &ClientPeer,
        request: JsonRpcRequest,
    ) -> impl Future<Output = Result<Value, JsonRpcError>> + Send {
        let peer = peer.clone();
        let selected_models = self.selected_models.clone();
        let authenticated_email = self.authenticated_email.clone();
        async move {
            match request.method.as_str() {
                aither_acp::vendor::codex::SET_MODEL_METHOD => {
                    let params = request
                        .params
                        .ok_or_else(|| JsonRpcError::invalid_params("missing params"))?;
                    let session_id = params
                        .get("sessionId")
                        .and_then(Value::as_str)
                        .ok_or_else(|| JsonRpcError::invalid_params("missing sessionId"))?
                        .to_string();
                    let model_id = params
                        .get("modelId")
                        .and_then(Value::as_str)
                        .ok_or_else(|| JsonRpcError::invalid_params("missing modelId"))?
                        .to_string();
                    selected_models
                        .lock()
                        .expect("selected_models")
                        .push((session_id.clone(), model_id.clone()));
                    peer.session_update(SessionNotification::new(
                        &session_id,
                        message_chunk(&format!("model:{model_id}")),
                    ))
                    .await
                    .map_err(|err| JsonRpcError::internal_error(err.to_string()))?;
                    Ok(json!({}))
                }
                aither_acp::vendor::codex::AUTHENTICATION_STATUS_METHOD => {
                    let email = authenticated_email
                        .lock()
                        .expect("authenticated_email")
                        .clone();
                    let status = email.map_or(
                        aither_acp::vendor::codex::AuthenticationStatus::Unauthenticated,
                        |email| aither_acp::vendor::codex::AuthenticationStatus::ChatGpt { email },
                    );
                    serde_json::to_value(status)
                        .map_err(|err| JsonRpcError::internal_error(err.to_string()))
                }
                aither_acp::vendor::codex::AUTHENTICATION_LOGOUT_METHOD => {
                    *authenticated_email.lock().expect("authenticated_email") = None;
                    Ok(json!({}))
                }
                other => Err(JsonRpcError::method_not_found(other)),
            }
        }
    }
}

#[tokio::test]
async fn agent_handler_request_fallback_serves_legacy_codex_vendor_methods() {
    let selected_models = Arc::new(Mutex::new(Vec::new()));
    let authenticated_email = Arc::new(Mutex::new(Some("dev@example.com".to_string())));

    let (client_end, server_end) = DuplexTransport::pair();
    let (_peer, server_driver) = serve(
        server_end,
        LegacyCodexAgent {
            selected_models: selected_models.clone(),
            authenticated_email,
        },
    );
    let server_task = tokio::spawn(server_driver);
    let (client, client_driver) = AcpClient::connect(client_end, RecordingClient::default());
    let client_task = tokio::spawn(client_driver);

    client.initialize().await.expect("initialize");

    let initial_status = aither_acp::vendor::codex::authentication_status(&client)
        .await
        .expect("authentication/status");
    assert!(matches!(
        initial_status,
        aither_acp::vendor::codex::AuthenticationStatus::ChatGpt { ref email }
            if email == "dev@example.com"
    ));

    aither_acp::vendor::codex::set_model(&client, "sess-codex", "gpt-5")
        .await
        .expect("session/set_model");
    assert_eq!(
        selected_models.lock().expect("selected_models").as_slice(),
        [("sess-codex".to_string(), "gpt-5".to_string())]
    );
    assert_eq!(
        client.handler().updates.lock().expect("updates").len(),
        1,
        "session/update queued inside AgentHandler::request must flush before response"
    );

    aither_acp::vendor::codex::authentication_logout(&client)
        .await
        .expect("authentication/logout");

    let post_logout_status = aither_acp::vendor::codex::authentication_status(&client)
        .await
        .expect("authentication/status after logout");
    assert!(matches!(
        post_logout_status,
        aither_acp::vendor::codex::AuthenticationStatus::Unauthenticated
    ));

    assert_method_not_found(
        client
            .request::<_, Value>("legacy/unrecognized", &json!({}))
            .await,
    );

    client.close();
    client_task.await.expect("client task");
    server_task.await.expect("server task");
}
