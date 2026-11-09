//! Commands on SSH servers: the `location` argument, first-use approval
//! through the permission handler, lazy preparation, both modes, the live
//! server set and the OS probe, against a private `sshd` on loopback.

use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

use crate::ssh::SshClient;
use crate::{CommandRole, SshPlatform, SshProbeError, SshServer};

use super::*;

/// Records the location requests it answers and approves `unsafe`
/// commands; it says it waits on someone for every location.
#[derive(Default)]
struct LocationHandler {
    deny: AtomicBool,
    asked: Mutex<Vec<(String, CommandRole)>>,
    unsafe_checks: AtomicUsize,
}

impl LocationHandler {
    fn asked(&self) -> Vec<(String, CommandRole)> {
        self.asked.lock().unwrap().clone()
    }
}

impl PermissionHandler for LocationHandler {
    fn check(
        &self,
        mode: TerminalMode,
        _script: &str,
        _language: ScriptLanguage,
    ) -> impl std::future::Future<Output = Result<bool, PermissionError>> + Send {
        if mode == TerminalMode::Unsafe {
            self.unsafe_checks.fetch_add(1, AtomicOrdering::SeqCst);
        }
        std::future::ready(Ok(true))
    }

    fn check_location(
        &self,
        server: &SshServer,
        role: CommandRole,
    ) -> impl std::future::Future<Output = Result<bool, PermissionError>> + Send {
        self.asked.lock().unwrap().push((server.name.clone(), role));
        std::future::ready(Ok(!self.deny.load(AtomicOrdering::SeqCst)))
    }

    fn will_wait_for_location(
        &self,
        _server: &SshServer,
        _role: CommandRole,
    ) -> impl std::future::Future<Output = bool> + Send {
        std::future::ready(true)
    }
}

type Terminal = TerminalTool<LocationHandler, executor_core::tokio::TokioGlobal, Configured>;

#[derive(Template)]
#[template(path = "test_sshd_config.txt", escape = "none")]
struct SshdConfig<'a> {
    port: u16,
    host_key: &'a str,
    authorized_keys: &'a str,
    pid_file: &'a str,
    home: &'a str,
}

#[derive(Template)]
#[template(path = "test_ssh_config.txt", escape = "none")]
struct ClientConfig<'a> {
    alias: &'a str,
    port: u16,
    user: &'a str,
    identity: &'a str,
}

const ALIAS: &str = "fixture";

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn text(path: &Path) -> &str {
    path.to_str().expect("a temporary path is UTF-8")
}

fn stdout_of(program: &str, args: &[&str]) -> String {
    let output = std::process::Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("{program} runs: {error}"));
    assert!(output.status.success(), "{program} {args:?} failed");
    String::from_utf8(output.stdout).unwrap().trim().to_string()
}

fn keygen(path: &Path) {
    stdout_of(
        "ssh-keygen",
        &["-q", "-t", "ed25519", "-N", "", "-f", text(path)],
    );
}

/// A client config whose alias reaches `port` on loopback as this user.
fn client_config(dir: &Path, port: u16, identity: &Path) -> SshClient {
    let path = dir.join("ssh_config");
    let user = stdout_of("id", &["-un"]);
    let config = ClientConfig {
        alias: ALIAS,
        port,
        user: &user,
        identity: text(identity),
    }
    .render()
    .unwrap();
    std::fs::write(&path, config).unwrap();
    SshClient::with_config(path)
}

/// A client whose alias reaches a loopback port nothing listens on, so a
/// connection is refused at once.
fn refusing_client(dir: &Path) -> SshClient {
    client_config(dir, free_port(), &dir.join("no_key"))
}

/// An `sshd` of this user on a loopback port, whose sessions have a home of
/// their own, holding the pinned `heel` unless started without it. Killed
/// when dropped.
struct Sshd {
    dir: tempfile::TempDir,
    child: std::process::Child,
    client: SshClient,
}

impl Sshd {
    async fn start() -> Self {
        Self::start_with_heel(true).await
    }

    async fn start_with_heel(has_heel: bool) -> Self {
        let sshd = Path::new("/usr/sbin/sshd");
        assert!(
            sshd.is_file(),
            "the SSH tests run a private OpenSSH server from /usr/sbin/sshd"
        );
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        keygen(&root.join("host_key"));
        keygen(&root.join("client_key"));
        std::fs::copy(root.join("client_key.pub"), root.join("authorized_keys")).unwrap();
        let home = root.join("home");
        std::fs::create_dir_all(home.join(".local/bin")).unwrap();
        if has_heel {
            let heel = crate::test_heel::pinned().await.unwrap();
            std::os::unix::fs::symlink(heel.as_path(), home.join(".local/bin/heel")).unwrap();
        }

        let port = free_port();
        let config = SshdConfig {
            port,
            host_key: text(&root.join("host_key")),
            authorized_keys: text(&root.join("authorized_keys")),
            pid_file: text(&root.join("sshd.pid")),
            home: text(&home),
        }
        .render()
        .unwrap();
        std::fs::write(root.join("sshd_config"), config).unwrap();
        let log = std::fs::File::create(root.join("sshd.log")).unwrap();
        let child = std::process::Command::new(sshd)
            .arg("-f")
            .arg(root.join("sshd_config"))
            .args(["-D", "-e"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .unwrap();
        let client = client_config(&root, port, &root.join("client_key"));
        let fixture = Self { dir, child, client };
        for _ in 0..200 {
            if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
                return fixture;
            }
            async_io::Timer::after(Duration::from_millis(25)).await;
        }
        panic!(
            "sshd did not listen: {}",
            std::fs::read_to_string(root.join("sshd.log")).unwrap_or_default()
        );
    }

    fn home(&self) -> PathBuf {
        std::fs::canonicalize(self.dir.path()).unwrap().join("home")
    }

    fn server() -> SshServer {
        SshServer {
            name: "fixture".to_string(),
            target: ALIAS.to_string(),
        }
    }
}

impl Drop for Sshd {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

async fn terminal(dir: &Path, registry: ShellSessionRegistry, role: CommandRole) -> Terminal {
    let tool = TerminalTool::new_exact(
        dir,
        LocationHandler::default(),
        executor_core::tokio::TokioGlobal,
    )
    .await
    .expect("terminal tool should initialize")
    .adopt_prepared_tools(registry)
    .await
    .expect("prepared tools should be adopted");
    let commands = ToolRegistryBuilder::new().build(tool.output_store().clone());
    let commands = match role {
        CommandRole::Main => commands,
        CommandRole::Subagent => commands.for_subagent(None, None),
    };
    tool.with_registry(Arc::new(commands))
}

fn args(script: &str, mode: TerminalExecutionMode, location: &str) -> TerminalArgs {
    TerminalArgs {
        description: "run on the server".to_string(),
        script: script.to_string(),
        language: ScriptLanguage::Shell,
        mode,
        location: Some(location.to_string()),
        expect: OutputFormat::Text,
        resolution: MediaResolution::Auto,
        max_lines: 50,
        raw: true,
        timeout: 60,
        background: false,
    }
}

/// A local home prepared by hand, reaching servers through `client`.
async fn home_registry(
    client: SshClient,
) -> (
    crate::tool_runtime::tests::UnlockingTempDir,
    ShellSessionRegistry,
) {
    let (cache, registry) = prepared_test_registry().await;
    (cache, registry.with_ssh_client(client))
}

#[tokio::test(flavor = "multi_thread")]
async fn the_probe_reports_the_servers_platform_without_preparing_it() {
    let sshd = Sshd::start().await;
    let platform = sshd.client.probe_os(&Sshd::server()).await.unwrap();
    assert_eq!(
        platform,
        SshPlatform {
            os: stdout_of("uname", &["-s"]),
            arch: stdout_of("uname", &["-m"]),
        }
    );
    assert!(
        !sshd.home().join(".cache/aither").exists(),
        "the probe installs nothing"
    );

    let unreachable = tempfile::tempdir().unwrap();
    assert!(matches!(
        refusing_client(unreachable.path())
            .probe_os(&Sshd::server())
            .await,
        Err(SshProbeError::Failed { .. })
    ));
}

/// The first command on a server added to a live registry asks once and
/// prepares the server; later commands, `unsafe` ones included, run there
/// without asking again; a removed server is no location.
#[tokio::test(flavor = "multi_thread")]
async fn a_server_added_live_is_approved_once_prepared_lazily_and_runs_both_modes() {
    let sshd = Sshd::start().await;
    let artifacts = tempfile::tempdir().unwrap();
    let distribution = script_distribution(artifacts.path()).await;
    let cache = crate::tool_runtime::tests::unlocking_tempdir();
    let registry = local_registry()
        .with_ssh_client(sshd.client.clone())
        .initialize_tools(&distribution, cache.path())
        .await
        .expect("the home prepares without touching any server");
    let dir = tempfile::tempdir().unwrap();
    let terminal = terminal(dir.path(), registry, CommandRole::Main).await;
    terminal
        .shell_sessions()
        .add_ssh_server(Sshd::server())
        .unwrap();
    assert!(
        !sshd.home().join(".cache/aither").exists(),
        "added, not prepared"
    );

    let script = "echo \"remote=$AITHER_FIXTURE_BASH\"";
    let sandboxed = terminal
        .call(
            args(script, TerminalExecutionMode::Default, "fixture"),
            ToolContext::new(),
        )
        .await
        .expect("a default-mode command runs on the server");
    let sandboxed = render_terminal_result(&sandboxed);
    assert!(!sandboxed.contains("Exit code:"), "{sandboxed}");
    // The managed bash ran, from the tools its first command prepared there.
    assert!(sandboxed.contains("remote=1.0.0"), "{sandboxed}");
    assert!(
        sshd.home().join(".cache/aither/tools").is_dir(),
        "prepared on the server"
    );
    assert_eq!(
        terminal.permission_handler.asked(),
        [("fixture".to_string(), CommandRole::Main)]
    );
    let events = terminal.permission_receiver().take_pending();
    assert_eq!(
        events
            .iter()
            .map(|event| (
                event.stage,
                event.location.as_deref(),
                event.script.as_str()
            ))
            .collect::<Vec<_>>(),
        [
            (PermissionEventStage::Waiting, Some("fixture"), script),
            (PermissionEventStage::Resolved, Some("fixture"), script),
        ]
    );

    let direct = terminal
        .call(
            args(
                "echo \"direct=$AITHER_FIXTURE_BASH home=$HOME\"",
                TerminalExecutionMode::Unsafe,
                " fixture ",
            ),
            ToolContext::new(),
        )
        .await
        .expect("an unsafe command runs on the server");
    let direct = render_terminal_result(&direct);
    // Outside `heel`, the session's environment reaches the command.
    assert!(
        direct.contains(&format!("direct=1.0.0 home={}", sshd.home().display())),
        "{direct}"
    );
    assert_eq!(terminal.permission_handler.asked().len(), 1, "asked once");
    assert_eq!(
        terminal
            .permission_handler
            .unsafe_checks
            .load(AtomicOrdering::SeqCst),
        1,
        "unsafe still passes its own admission"
    );
    // A server's command is a remote job: its local `ssh` process stands for
    // work whose memory is never judged here.
    terminal
        .run_background(args(
            "echo remote-job",
            TerminalExecutionMode::Default,
            "fixture",
        ))
        .await
        .expect("a background command starts on the server");
    let places = terminal
        .job_registry()
        .list()
        .await
        .into_iter()
        .filter(|job| job.script == "echo remote-job")
        .map(|job| job.place)
        .collect::<Vec<_>>();
    assert_eq!(places, [crate::JobPlace::Remote]);
    assert_eq!(terminal.job_memory().snapshot().await.unwrap().jobs, []);
    assert!(terminal.container_probe().is_none());
    terminal.job_registry().kill_running().await;
}

/// A server without `heel` gets the distribution's pinned one for its own
/// target on its first command, which then runs in default mode.
#[tokio::test(flavor = "multi_thread")]
async fn a_server_without_heel_gets_the_pinned_one_installed() {
    let sshd = Sshd::start_with_heel(false).await;
    let artifacts = tempfile::tempdir().unwrap();
    let distribution = script_distribution(artifacts.path()).await;
    let cache = crate::tool_runtime::tests::unlocking_tempdir();
    let registry = local_registry()
        .with_ssh_client(sshd.client.clone())
        .initialize_tools(&distribution, cache.path())
        .await
        .expect("the home prepares without touching any server");
    let dir = tempfile::tempdir().unwrap();
    let terminal = terminal(dir.path(), registry, CommandRole::Main).await;
    terminal
        .shell_sessions()
        .add_ssh_server(Sshd::server())
        .unwrap();
    let installed = sshd.home().join(".local/bin/heel");
    assert!(!installed.exists());

    let output = terminal
        .call(
            args("echo installed", TerminalExecutionMode::Default, "fixture"),
            ToolContext::new(),
        )
        .await
        .expect("a default-mode command runs under the installed heel");
    assert!(render_terminal_result(&output).contains("installed"));
    assert!(installed.is_file(), "heel was installed on the server");
}

/// A refused first command is asked again by the next one and never
/// connects to the server.
#[tokio::test(flavor = "multi_thread")]
async fn a_denied_server_is_asked_again_and_never_prepared() {
    let unreachable = tempfile::tempdir().unwrap();
    let (_cache, registry) = home_registry(refusing_client(unreachable.path())).await;
    registry.add_ssh_server(Sshd::server()).unwrap();
    let environment = registry.ssh_environments().get("fixture").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let terminal = terminal(dir.path(), registry, CommandRole::Main).await;
    terminal
        .permission_handler
        .deny
        .store(true, AtomicOrdering::SeqCst);
    for _ in 0..2 {
        let denied = terminal
            .call(
                args("true", TerminalExecutionMode::Default, "fixture"),
                ToolContext::new(),
            )
            .await
            .expect_err("a denied server runs nothing");
        assert!(
            denied
                .to_string()
                .contains("running commands on `fixture` was not approved"),
            "{denied}"
        );
    }
    assert_eq!(terminal.permission_handler.asked().len(), 2);
    assert!(!environment.is_approved());
    assert!(!environment.is_prepared());
}

/// A subagent's first command reaches the handler as the subagent's, and
/// the approval holds for the main agent sharing the registry.
#[tokio::test(flavor = "multi_thread")]
async fn a_subagent_asks_as_a_subagent_and_its_approval_is_shared() {
    let unreachable = tempfile::tempdir().unwrap();
    let (_cache, registry) = home_registry(refusing_client(unreachable.path())).await;
    let tools = registry.local_tools().unwrap().clone();
    let registry = registry.with_test_ssh(Sshd::server(), Some("/opt/heel".into()), tools);
    let dir = tempfile::tempdir().unwrap();
    let subagent = terminal(dir.path(), registry.clone(), CommandRole::Subagent).await;
    let main = terminal(dir.path(), registry, CommandRole::Main).await;
    // The connection is refused; only the questions put matter here.
    let _ = subagent
        .call(
            args("true", TerminalExecutionMode::Default, "fixture"),
            ToolContext::new(),
        )
        .await;
    let _ = main
        .call(
            args("true", TerminalExecutionMode::Default, "fixture"),
            ToolContext::new(),
        )
        .await;
    assert_eq!(
        subagent.permission_handler.asked(),
        [("fixture".to_string(), CommandRole::Subagent)]
    );
    assert_eq!(main.permission_handler.asked(), []);
}

/// A server without `heel` runs `unsafe` commands only.
#[tokio::test(flavor = "multi_thread")]
async fn default_mode_needs_the_servers_heel() {
    let unreachable = tempfile::tempdir().unwrap();
    let (_cache, registry) = home_registry(refusing_client(unreachable.path())).await;
    let tools = registry.local_tools().unwrap().clone();
    let registry = registry.with_test_ssh(Sshd::server(), None, tools);
    let dir = tempfile::tempdir().unwrap();
    let terminal = terminal(dir.path(), registry, CommandRole::Main).await;
    let refused = terminal
        .call(
            args("true", TerminalExecutionMode::Default, "fixture"),
            ToolContext::new(),
        )
        .await
        .expect_err("no sandbox there");
    assert!(
        refused
            .to_string()
            .contains("default mode runs under `heel`, which `fixture` does not have"),
        "{refused}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_removed_or_unknown_location_is_refused_and_an_empty_one_is_home() {
    let (_cache, registry) = prepared_test_registry().await;
    let tools = registry.local_tools().unwrap().clone();
    let registry = registry.with_test_ssh(Sshd::server(), Some("/opt/heel".into()), tools);
    let dir = tempfile::tempdir().unwrap();
    let terminal = terminal(dir.path(), registry, CommandRole::Main).await;
    assert_eq!(
        terminal.shell_sessions().remove_ssh_server("fixture"),
        Some(Sshd::server())
    );
    for name in ["fixture", "staging"] {
        let refused = terminal
            .call(
                args("true", TerminalExecutionMode::Default, name),
                ToolContext::new(),
            )
            .await
            .expect_err("no such server");
        assert!(
            refused
                .to_string()
                .contains(&format!("no SSH server named `{name}` is configured")),
            "{refused}"
        );
    }
    let home = terminal
        .call(
            args("echo home", TerminalExecutionMode::Default, ""),
            ToolContext::new(),
        )
        .await
        .expect("an empty location is the home environment");
    assert!(render_terminal_result(&home).contains("home"));
    assert_eq!(terminal.permission_handler.asked(), []);
}

#[tokio::test]
async fn the_public_probe_reports_an_unreachable_server() {
    let error = crate::probe_ssh_os(&SshServer {
        name: "nowhere".to_string(),
        target: "ssh://nobody@127.0.0.1:1".to_string(),
    })
    .await
    .expect_err("nothing answers");
    assert!(matches!(error, SshProbeError::Failed { .. }), "{error}");
}

/// The pool bounds this machine's jobs: a server's command runs outside it,
/// while a local command of the same terminal still draws from it.
#[cfg(feature = "jobserver")]
#[tokio::test(flavor = "multi_thread")]
async fn a_servers_command_runs_outside_the_job_pool_and_a_local_one_inside() {
    let sshd = Sshd::start().await;
    let artifacts = tempfile::tempdir().unwrap();
    let distribution = script_distribution(artifacts.path()).await;
    let cache = crate::tool_runtime::tests::unlocking_tempdir();
    let registry = local_registry()
        .with_ssh_client(sshd.client.clone())
        .initialize_tools(&distribution, cache.path())
        .await
        .expect("the home prepares");
    let dir = tempfile::tempdir().unwrap();
    let pool_dir = tempfile::tempdir().unwrap();
    let pool = crate::jobserver::JobPool::create_fifo(pool_dir.path().join("pool.fifo"), 2)
        .expect("a pool");
    let terminal = terminal(dir.path(), registry, CommandRole::Main)
        .await
        .with_jobserver_pool(pool, pool_dir.path().join("commands"))
        .expect("the pool attaches");
    terminal
        .shell_sessions()
        .add_ssh_server(Sshd::server())
        .unwrap();
    let script = "printenv MAKEFLAGS || echo jobs=none";

    let remote = terminal
        .call(
            args(script, TerminalExecutionMode::Default, "fixture"),
            ToolContext::new(),
        )
        .await
        .expect("a server's command runs while a pool is configured");
    let remote = render_terminal_result(&remote);
    assert!(remote.contains("jobs=none"), "{remote}");

    let mut local = args(script, TerminalExecutionMode::Default, "fixture");
    local.location = None;
    let local = terminal
        .call(local, ToolContext::new())
        .await
        .expect("a local command runs");
    let local = render_terminal_result(&local);
    assert!(local.contains("--jobserver-auth=fifo:"), "{local}");
}
