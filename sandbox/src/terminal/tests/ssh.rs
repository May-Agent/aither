//! Commands on SSH servers: the `location` argument, the approval a script
//! on a server is asked for, lazy preparation, both modes, the live server
//! set and the OS probe, against a private `sshd` on loopback.

use std::process::Stdio;

use crate::ssh::SshClient;
use crate::{SshPlatform, SshProbeError, SshServer};

use super::*;

/// Records the directory and location every script it checks starts in.
#[derive(Clone, Default)]
struct RecordStarts(Arc<Mutex<Vec<Start>>>);

/// The location and starting directory of one checked script.
type Start = (Option<String>, ScriptDir);

impl RecordStarts {
    fn seen(&self) -> Vec<Start> {
        self.0.lock().unwrap().clone()
    }
}

impl crate::ScriptPolicy for RecordStarts {
    type State = ();

    fn check(
        &self,
        script: &crate::ScriptCheck<'_>,
        (): &(),
    ) -> impl Future<Output = ScriptDecision> + Send {
        self.0
            .lock()
            .unwrap()
            .push((script.location.map(str::to_string), script.start.clone()));
        std::future::ready(ScriptDecision::Pass)
    }
}

type Terminal = Shell<executor_core::tokio::TokioGlobal, Configured>;

#[derive(Template)]
#[template(path = "test_sshd_config.txt", escape = "none")]
struct SshdConfig<'a> {
    port: u16,
    host_key: &'a str,
    authorized_keys: &'a str,
    pid_file: &'a str,
    home: &'a str,
    login_shell: Option<&'a str>,
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
        Self::start_with(has_heel, None).await
    }

    /// A server whose every command line is parsed by `login_shell`, a
    /// program path, instead of the shell of this user's account.
    async fn start_with_login_shell(login_shell: &str) -> Self {
        Self::start_with(false, Some(login_shell)).await
    }

    async fn start_with(has_heel: bool, login_shell: Option<&str>) -> Self {
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
            login_shell,
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

    /// How many connections the server has authenticated so far, read from
    /// its own log: `sshd` records each accepted connection at its default
    /// verbosity.
    fn accepted_connections(&self) -> usize {
        std::fs::read_to_string(self.dir.path().join("sshd.log"))
            .unwrap()
            .lines()
            .filter(|line| line.contains("Accepted publickey"))
            .count()
    }
}

impl Drop for Sshd {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The host backend a shell's home is, under the pinned `heel`.
async fn host_home() -> crate::Backend {
    crate::Backend::Host {
        heel: crate::test_heel::pinned().await.expect("the pinned heel"),
    }
}

/// A terminal in `dir` over `registry` that asks `approver` for what needs
/// approval.
async fn terminal(dir: &Path, registry: ShellSessionRegistry, approver: &Approvals) -> Terminal {
    terminal_with(dir, registry, approver, RecordStarts::default()).await
}

async fn terminal_with(
    dir: &Path,
    registry: ShellSessionRegistry,
    approver: &Approvals,
    policy: RecordStarts,
) -> Terminal {
    let tool = Shell::new_exact(
        dir,
        executor_core::tokio::TokioGlobal,
        crate::limits::fixture::limits(),
        crate::output_policy::fixture::policy(),
        crate::test_wording::TestWording::any(),
        crate::test_wording::job_tools(),
    )
    .await
    .expect("terminal tool should initialize")
    .with_approver(approver.clone())
    .adopt_prepared_tools(registry)
    .await
    .expect("prepared tools should be adopted");
    let mut commands = ToolRegistryBuilder::new();
    commands.script_policy(policy);
    let commands = Arc::new(commands.build(tool.output_store().clone()));
    tool.with_registry(commands)
}

fn args(script: &str, mode: SandboxMode, location: &str) -> ShellArgs {
    ShellArgs {
        description: "run on the server".to_string(),
        script: script.to_string(),
        interpreter: Interpreter::Shell,
        sandbox: mode,
        location: Some(location.to_string()),
        resolution: MediaResolution::Auto,
        max_lines: Some(50),
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
        Err(SshProbeError::Upload(_))
    ));
}

/// A command on a server added to a live registry is asked about before
/// the server is prepared, as every command on a server is; the policy sees
/// it start on the server, and both modes run there.
#[tokio::test(flavor = "multi_thread")]
async fn a_server_added_live_is_asked_prepared_lazily_and_runs_both_modes() {
    let sshd = Sshd::start().await;
    let artifacts = tempfile::tempdir().unwrap();
    let distribution = script_distribution(artifacts.path()).await;
    let cache = crate::tool_runtime::tests::unlocking_tempdir();
    let registry = local_registry()
        .with_ssh_client(sshd.client.clone())
        .initialize_tools(&host_home().await, &distribution, cache.path())
        .await
        .expect("the home prepares without touching any server");
    let dir = tempfile::tempdir().unwrap();
    let approvals = Approvals::default();
    let starts = RecordStarts::default();
    let terminal = terminal_with(dir.path(), registry, &approvals, starts.clone()).await;
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
            args(script, SandboxMode::Sandboxed, "fixture"),
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
    // Only the policy's pass was given, so the shell asked its approver.
    let asked = approvals.asked();
    assert_eq!(asked.len(), 1);
    assert_eq!(asked[0].kind().as_str(), "remote");
    assert_eq!(asked[0].subject(), script);
    assert_eq!(
        starts.seen(),
        [(
            Some("fixture".to_string()),
            ScriptDir::Remote {
                server: "fixture".to_string()
            }
        )]
    );

    let direct = terminal
        .call(
            args(
                "echo \"direct=$AITHER_FIXTURE_BASH home=$HOME\"",
                SandboxMode::Unsandboxed,
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
    let asked = approvals.asked();
    assert_eq!(asked.len(), 2, "no approval is remembered between scripts");
    assert_eq!(asked[1].kind().as_str(), "unsandboxed");
    // A server's command is a remote job: its local `ssh` process stands for
    // work whose memory is never judged here.
    terminal
        .run_background(args("echo remote-job", SandboxMode::Sandboxed, "fixture"))
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
        .initialize_tools(&host_home().await, &distribution, cache.path())
        .await
        .expect("the home prepares without touching any server");
    let dir = tempfile::tempdir().unwrap();
    let terminal = terminal(dir.path(), registry, &Approvals::default()).await;
    terminal
        .shell_sessions()
        .add_ssh_server(Sshd::server())
        .unwrap();
    let installed = sshd.home().join(".local/bin/heel");
    assert!(!installed.exists());

    let output = terminal
        .call(
            args("echo installed", SandboxMode::Sandboxed, "fixture"),
            ToolContext::new(),
        )
        .await
        .expect("a default-mode command runs under the installed heel");
    assert!(render_terminal_result(&output).contains("installed"));
    assert!(installed.is_file(), "heel was installed on the server");
}

/// A denied script on a server is asked again by the next one, never
/// connects to the server, and the reason the approver gave reaches the
/// caller.
#[tokio::test(flavor = "multi_thread")]
async fn a_denied_script_on_a_server_never_prepares_it_and_carries_the_reason() {
    let unreachable = tempfile::tempdir().unwrap();
    let (_cache, registry) = home_registry(refusing_client(unreachable.path())).await;
    registry.add_ssh_server(Sshd::server()).unwrap();
    let environment = registry.ssh_environments().get("fixture").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let approvals = Approvals::denying("fixture is off limits");
    let terminal = terminal(dir.path(), registry, &approvals).await;
    for _ in 0..2 {
        let denied = terminal
            .call(
                args("true", SandboxMode::Sandboxed, "fixture"),
                ToolContext::new(),
            )
            .await
            .expect_err("a denied script runs nothing");
        assert!(
            denied.to_string().contains("fixture is off limits"),
            "{denied}"
        );
    }
    assert_eq!(approvals.asked().len(), 2);
    assert!(!environment.is_prepared());
}

/// A script on a server that only passes the policies needs an approver,
/// and one a policy allows asks nobody.
#[tokio::test(flavor = "multi_thread")]
async fn a_remote_script_with_only_passes_needs_an_approver_and_an_allow_asks_nobody() {
    let unreachable = tempfile::tempdir().unwrap();
    let (_cache, registry) = home_registry(refusing_client(unreachable.path())).await;
    let tools = registry.local_tools().unwrap().clone();
    let registry = registry.with_test_ssh(Sshd::server(), Some("/opt/heel".into()), tools);
    let dir = tempfile::tempdir().unwrap();

    // No policy decided, and no approver exists: the script is not run.
    let tool = Shell::new_exact(
        dir.path(),
        executor_core::tokio::TokioGlobal,
        crate::limits::fixture::limits(),
        crate::output_policy::fixture::policy(),
        crate::test_wording::TestWording::any(),
        crate::test_wording::job_tools(),
    )
    .await
    .unwrap()
    .adopt_prepared_tools(registry.clone())
    .await
    .unwrap();
    let commands = Arc::new(ToolRegistryBuilder::new().build(tool.output_store().clone()));
    let tool = tool.with_registry(commands);
    let error = tool
        .call(
            args("true", SandboxMode::Sandboxed, "fixture"),
            ToolContext::new(),
        )
        .await
        .expect_err("a remote script with only passes needs an approver");
    assert!(error.to_string().contains("no approver"), "{error}");

    // A policy's allow settles it without the approver; the connection is
    // refused, so the call fails later, and not for want of approval.
    let tool = Shell::new_exact(
        dir.path(),
        executor_core::tokio::TokioGlobal,
        crate::limits::fixture::limits(),
        crate::output_policy::fixture::policy(),
        crate::test_wording::TestWording::any(),
        crate::test_wording::job_tools(),
    )
    .await
    .unwrap()
    .adopt_prepared_tools(registry)
    .await
    .unwrap();
    let mut policies = ToolRegistryBuilder::new();
    policies.script_policy(Decide(ScriptDecision::Allow));
    let commands = Arc::new(policies.build(tool.output_store().clone()));
    let tool = tool.with_registry(commands);
    let result = tool
        .call(
            args("true", SandboxMode::Sandboxed, "fixture"),
            ToolContext::new(),
        )
        .await;
    assert!(
        result
            .as_ref()
            .err()
            .is_none_or(|error| !error.to_string().contains("no approver")),
        "{result:?}"
    );
}

/// A policy sees the server an `unsafe` command is located on, or the home
/// when it names none.
#[tokio::test(flavor = "multi_thread")]
async fn a_policy_sees_the_location_of_an_unsafe_command() {
    let unreachable = tempfile::tempdir().unwrap();
    let (_cache, registry) = home_registry(refusing_client(unreachable.path())).await;
    let tools = registry.local_tools().unwrap().clone();
    let registry = registry.with_test_ssh(Sshd::server(), None, tools);
    let dir = tempfile::tempdir().unwrap();
    let starts = RecordStarts::default();
    let terminal = terminal_with(dir.path(), registry, &Approvals::default(), starts.clone()).await;
    // The connection is refused; only the policy's view matters here.
    let _ = terminal
        .call(
            args("true", SandboxMode::Unsandboxed, "fixture"),
            ToolContext::new(),
        )
        .await;
    terminal
        .call(
            args("true", SandboxMode::Unsandboxed, ""),
            ToolContext::new(),
        )
        .await
        .expect("an unsafe command runs at home");
    let locations = starts
        .seen()
        .into_iter()
        .map(|(location, _)| location)
        .collect::<Vec<_>>();
    assert_eq!(locations, [Some("fixture".to_string()), None]);
}

/// A server without `heel` runs `unsafe` commands only.
#[tokio::test(flavor = "multi_thread")]
async fn default_mode_needs_the_servers_heel() {
    let unreachable = tempfile::tempdir().unwrap();
    let (_cache, registry) = home_registry(refusing_client(unreachable.path())).await;
    let tools = registry.local_tools().unwrap().clone();
    let registry = registry.with_test_ssh(Sshd::server(), None, tools);
    let dir = tempfile::tempdir().unwrap();
    let terminal = terminal(dir.path(), registry, &Approvals::default()).await;
    let refused = terminal
        .call(
            args("true", SandboxMode::Sandboxed, "fixture"),
            ToolContext::new(),
        )
        .await
        .expect_err("no sandbox there");
    assert_eq!(refused.to_string(), "no remote sandbox: fixture");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_removed_or_unknown_location_is_refused_and_an_empty_one_is_home() {
    let (_cache, registry) = prepared_test_registry().await;
    let tools = registry.local_tools().unwrap().clone();
    let registry = registry.with_test_ssh(Sshd::server(), Some("/opt/heel".into()), tools);
    let dir = tempfile::tempdir().unwrap();
    let approvals = Approvals::default();
    let terminal = terminal(dir.path(), registry, &approvals).await;
    assert_eq!(
        terminal.shell_sessions().remove_ssh_server("fixture"),
        Some(Sshd::server())
    );
    for name in ["fixture", "staging"] {
        let refused = terminal
            .call(
                args("true", SandboxMode::Sandboxed, name),
                ToolContext::new(),
            )
            .await
            .expect_err("no such server");
        assert_eq!(refused.to_string(), format!("unknown location: {name}"));
    }
    let home = terminal
        .call(
            args("echo home", SandboxMode::Sandboxed, ""),
            ToolContext::new(),
        )
        .await
        .expect("an empty location is the home environment");
    assert!(render_terminal_result(&home).contains("home"));
    assert_eq!(
        approvals.asked(),
        [],
        "the home environment needs no approval"
    );
}

#[tokio::test]
async fn the_public_probe_reports_an_unreachable_server() {
    let error = crate::probe_ssh_os(&SshServer {
        name: "nowhere".to_string(),
        target: "ssh://nobody@127.0.0.1:1".to_string(),
    })
    .await
    .expect_err("nothing answers");
    assert!(matches!(error, SshProbeError::Upload(_)), "{error}");
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
        .initialize_tools(&host_home().await, &distribution, cache.path())
        .await
        .expect("the home prepares");
    let dir = tempfile::tempdir().unwrap();
    let pool_dir = tempfile::tempdir().unwrap();
    let pool = crate::jobserver::JobPool::create_fifo(pool_dir.path().join("pool.fifo"), 2)
        .expect("a pool");
    let terminal = terminal(dir.path(), registry, &Approvals::default())
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
            args(script, SandboxMode::Sandboxed, "fixture"),
            ToolContext::new(),
        )
        .await
        .expect("a server's command runs while a pool is configured");
    let remote = render_terminal_result(&remote);
    assert!(remote.contains("jobs=none"), "{remote}");

    let mut local = args(script, SandboxMode::Sandboxed, "fixture");
    local.location = None;
    let local = terminal
        .call(local, ToolContext::new())
        .await
        .expect("a local command runs");
    let local = render_terminal_result(&local);
    assert!(local.contains("--jobserver-auth=fifo:"), "{local}");
}

/// A process spawned on a server streams its output under the server's
/// `heel`, and `kill` ends the remote tree.
#[tokio::test(flavor = "multi_thread")]
async fn spawn_on_a_server_streams_output_and_kill_ends_the_remote_tree() {
    use crate::SpawnSpec;

    let sshd = Sshd::start().await;
    let (_cache, registry) = prepared_test_registry().await;
    let tools = registry.local_tools().unwrap().clone();
    let dir = tempfile::tempdir().unwrap();
    let heel = sshd.home().join(".local/bin/heel");
    let sandbox = Sandbox::new(
        crate::Backend::Ssh {
            server: Sshd::server(),
            heel: Some(text(&heel).to_string()),
            launch: tools.remote_launch().unwrap(),
        },
        executor_core::tokio::TokioGlobal,
        dir.path(),
    )
    .with_ssh_client(sshd.client.clone());

    let mut child = sandbox
        .spawn(SpawnSpec::new("/bin/echo").args(["from-the-server"]))
        .await
        .expect("the process starts under the server's heel");
    let stdout = child.take_stdout().unwrap();
    let stderr = child.take_stderr().unwrap();
    let mut printed = Vec::new();
    while let Ok(chunk) = stdout.recv().await {
        printed.extend(chunk);
    }
    let mut failed = Vec::new();
    while let Ok(chunk) = stderr.recv().await {
        failed.extend(chunk);
    }
    let exit = child.wait().await;
    assert_eq!(
        String::from_utf8_lossy(&printed),
        "from-the-server\n",
        "{exit:?}: {}",
        String::from_utf8_lossy(&failed)
    );
    assert!(exit.success());

    // The server's login shell parses the command line first, so the job is
    // a file rather than a quoted argument.
    let job = sshd.home().join("job.sh");
    std::fs::write(&job, "sleep 300 &\necho $!\nwait\n").unwrap();
    let unconfined = sandbox.unconfined();
    let mut child = unconfined
        .spawn(SpawnSpec::new("/bin/sh").args([text(&job)]))
        .await
        .expect("the process starts");
    let stdout = child.take_stdout().unwrap();
    let remote = tokio::time::timeout(Duration::from_secs(60), stdout.recv())
        .await
        .expect("the descendant reports its id")
        .expect("stdout is open");
    let remote = String::from_utf8(remote).unwrap().trim().to_string();
    let alive = |pid: &str| {
        std::process::Command::new("kill")
            .args(["-0", pid])
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success()
    };
    assert!(alive(&remote), "the remote descendant runs");
    child.kill().await.expect("the tree is killed");
    tokio::time::timeout(Duration::from_secs(60), child.wait())
        .await
        .expect("the killed process is waited for");
    tokio::time::timeout(Duration::from_secs(60), async {
        while alive(&remote) {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the remote descendant ends with the client");
}

/// A script spawned on a server is quoted once for its login shell, which
/// parses the command line first: a double-quoted script survives a login
/// shell that does not read single quotes the way POSIX does.
#[tokio::test(flavor = "multi_thread")]
async fn a_script_spawned_on_a_server_runs_in_the_managed_shell() {
    use crate::SpawnSpec;

    let sshd = Sshd::start().await;
    let (_cache, registry) = prepared_test_registry().await;
    let tools = registry.local_tools().unwrap().clone();
    let dir = tempfile::tempdir().unwrap();
    let heel = sshd.home().join(".local/bin/heel");
    let sandbox = Sandbox::new(
        crate::Backend::Ssh {
            server: Sshd::server(),
            heel: Some(text(&heel).to_string()),
            launch: tools.remote_launch().unwrap(),
        },
        executor_core::tokio::TokioGlobal,
        dir.path(),
    )
    .with_ssh_client(sshd.client.clone());

    let mut child = sandbox
        .spawn(SpawnSpec::script("echo \"from-the-script\""))
        .await
        .expect("the script starts under the server's heel");
    let stdout = child.take_stdout().unwrap();
    let stderr = child.take_stderr().unwrap();
    let mut printed = Vec::new();
    while let Ok(chunk) = stdout.recv().await {
        printed.extend(chunk);
    }
    let mut failed = Vec::new();
    while let Ok(chunk) = stderr.recv().await {
        failed.extend(chunk);
    }
    let exit = child.wait().await;
    assert_eq!(
        String::from_utf8_lossy(&printed),
        "from-the-script\n",
        "{exit:?}: {}",
        String::from_utf8_lossy(&failed)
    );
    assert!(exit.success());
    // The processes on the server are out of sight, and its stdin has no watch.
    assert_eq!(
        child.descendants_killed().await,
        crate::Descendants::Unobserved
    );
    assert!(child.take_stdin_blocked().is_none());
}

/// Text every quoting convention treats differently, in the scripts of the
/// login-shell tests: spaces, both quotes, backslashes, `$`, command
/// substitution, newlines and Chinese, Japanese and Korean.
const AWKWARD: &str = "a b 'single' \"double\" \\back\\\\slash \\' $HOME ${x} $(id) `id` !! ;|&<>\n\
    \tindented\n\n中文：你好，世界\n日本語：こんにちは、世界\n한국어: 안녕하세요, 세계";

/// Runs a script that prints its own path, then [`AWKWARD`] from a quoted
/// here-document, through a server whose login shell is `login_shell`: the
/// text must arrive unchanged and the script's file must be gone.
async fn assert_script_arrives_unchanged(login_shell: &str) {
    let sshd = Sshd::start_with_login_shell(login_shell).await;
    let script = format!("printf '%s\\n' \"$0\"\ncat <<'AITHER_END'\n{AWKWARD}\nAITHER_END\n");
    let command = sshd.client.command(&Sshd::server(), &script).await.unwrap();
    let output = async_process::Command::from(command)
        .stdin(Stdio::null())
        .output()
        .await
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{login_shell}: {stderr}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    let (path, body) = stdout.split_once('\n').unwrap();
    assert_eq!(body, format!("{AWKWARD}\n"), "{login_shell}");
    assert!(
        !Path::new(path).exists(),
        "{login_shell}: the script deletes its own file {path}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_script_reaches_sh_unchanged_under_a_posix_login_shell() {
    assert_script_arrives_unchanged("/bin/sh").await;
}

/// Needs `fish` installed: skipped, loudly, where it is not.
#[tokio::test(flavor = "multi_thread")]
async fn a_script_reaches_sh_unchanged_under_a_fish_login_shell() {
    let fish = [
        "/opt/homebrew/bin/fish",
        "/usr/local/bin/fish",
        "/usr/bin/fish",
    ]
    .into_iter()
    .find(|path| Path::new(path).is_file());
    let Some(fish) = fish else {
        eprintln!("SKIPPED: fish is not installed, so the fish login shell case did not run");
        return;
    };
    assert_script_arrives_unchanged(fish).await;
}

/// The job's standard input is its own: the script travels another way.
#[tokio::test(flavor = "multi_thread")]
async fn the_commands_stdin_is_free_for_the_scripts_input() {
    let sshd = Sshd::start().await;
    let command = sshd.client.command(&Sshd::server(), "cat").await.unwrap();
    let mut child = async_process::Command::from(command)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    futures_lite::io::AsyncWriteExt::write_all(&mut stdin, "输入 'x'\n".as_bytes())
        .await
        .unwrap();
    drop(stdin);
    let output = child.output().await.unwrap();
    assert_eq!(output.stdout, "输入 'x'\n".as_bytes());
}

async fn run_script(client: &SshClient, script: &str) -> String {
    let command = client.command(&Sshd::server(), script).await.unwrap();
    let output = async_process::Command::from(command)
        .stdin(Stdio::null())
        .output()
        .await
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap()
}

/// The server's own log counts the connections: one script is an upload and
/// a run, and none of them, sequential or concurrent, opens a second one.
#[tokio::test(flavor = "multi_thread")]
async fn every_command_to_a_server_shares_one_connection() {
    let sshd = Sshd::start().await;
    assert_eq!(run_script(&sshd.client, "echo one").await, "one\n");
    assert_eq!(
        sshd.accepted_connections(),
        1,
        "the upload and the run of one script share a connection"
    );
    assert_eq!(run_script(&sshd.client, "echo two").await, "two\n");
    assert_eq!(sshd.accepted_connections(), 1);

    let fresh = Sshd::start().await;
    let scripts = ["echo a", "echo b", "echo c", "echo d", "echo e", "echo f"];
    let outputs = futures::future::join_all(
        scripts
            .iter()
            .map(|script| run_script(&fresh.client, script)),
    )
    .await;
    assert_eq!(outputs, ["a\n", "b\n", "c\n", "d\n", "e\n", "f\n"]);
    assert_eq!(
        fresh.accepted_connections(),
        1,
        "concurrent first uses start one master"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn dropping_the_client_ends_its_master_and_removes_its_sockets() {
    let sshd = Sshd::start().await;
    let command = sshd
        .client
        .command(&Sshd::server(), "echo up")
        .await
        .unwrap();
    drop(command);
    let socket_dir = {
        let ssh = sshd.client.ssh(&Sshd::server(), "true");
        let path = ssh
            .get_args()
            .find_map(|arg| arg.to_str()?.strip_prefix("ControlPath="))
            .unwrap()
            .to_string();
        PathBuf::from(path).parent().unwrap().to_path_buf()
    };
    assert!(socket_dir.is_dir());
    let Sshd { client, .. } = &sshd;
    let client = client.clone();
    drop(sshd);
    drop(client);
    assert!(!socket_dir.exists(), "the socket directory is removed");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_script_that_cannot_be_uploaded_runs_nothing_and_says_why() {
    let unreachable = tempfile::tempdir().unwrap();
    let error = refusing_client(unreachable.path())
        .command(&Sshd::server(), "echo ran")
        .await
        .unwrap_err();
    assert!(
        matches!(error, crate::ssh::SshUploadError::Failed { .. }),
        "{error}"
    );
}
