//! Spawning on the host and container backends: streams, exit, tree kill,
//! live and derived grants, and the environment layers.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_channel::{Receiver, Sender};
use executor_core::tokio::TokioGlobal;

use super::{Backend, ChildExit, GrantSnapshot, Pipe, Sandbox, SandboxedChild, SpawnSpec, Stdio};
use crate::shell_session::ContainerExecOutcome;
use crate::{AnyContainerExec, ContainerExec, ContainerShellRuntime, HostRoute};

/// The bound on any wait for a process or a stream: a hang fails the test.
const BOUND: Duration = Duration::from_secs(60);

fn canonical_tempdir() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let path = std::fs::canonicalize(dir.path()).expect("a temporary directory resolves");
    (dir, path)
}

async fn host_sandbox(working_dir: &Path) -> Sandbox {
    let heel = crate::test_heel::pinned()
        .await
        .expect("the pinned heel is available");
    Sandbox::new(Backend::Host { heel }, TokioGlobal, working_dir)
}

/// Everything a stream carried until it closed.
async fn drain(stream: Receiver<Vec<u8>>) -> String {
    let mut bytes = Vec::new();
    while let Ok(chunk) = stream.recv().await {
        bytes.extend(chunk);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Runs `script` in `sh` to its end and returns how it ended and what it
/// printed on each stream.
async fn run(sandbox: &Sandbox, script: &str) -> (ChildExit, String, String) {
    run_spec(sandbox, SpawnSpec::new("/bin/sh").args(["-c", script])).await
}

async fn run_spec(sandbox: &Sandbox, spec: SpawnSpec) -> (ChildExit, String, String) {
    tokio::time::timeout(BOUND, async {
        let mut child = sandbox.spawn(spec).await.expect("the process starts");
        let stdout = drain(child.take_stdout().expect("stdout is piped"));
        let stderr = drain(child.take_stderr().expect("stderr is piped"));
        let (out, err) = tokio::join!(stdout, stderr);
        (child.wait().await, out, err)
    })
    .await
    .expect("the process ends within its bound")
}

fn exit_code(exit: &ChildExit) -> Option<i32> {
    match exit {
        ChildExit::Exited(status) => status.code(),
        ChildExit::Killed | ChildExit::Failed(_) => None,
    }
}

/// Whether a process with this id still exists.
fn alive(pid: &str) -> bool {
    std::process::Command::new("kill")
        .args(["-0", pid])
        .stderr(std::process::Stdio::null())
        .status()
        .expect("kill runs")
        .success()
}

#[tokio::test]
async fn a_host_process_streams_its_output_and_reports_its_exit() {
    let (_work, work) = canonical_tempdir();
    let sandbox = host_sandbox(&work).await;
    let (exit, out, err) = run(&sandbox, "echo out; echo err >&2; exit 3").await;
    assert_eq!(exit_code(&exit), Some(3), "{exit:?}");
    assert_eq!(out, "out\n");
    assert_eq!(err, "err\n");
}

#[tokio::test]
async fn a_process_reads_what_is_sent_to_its_stdin_until_it_is_dropped() {
    let (_work, work) = canonical_tempdir();
    let sandbox = host_sandbox(&work).await;
    let spec = SpawnSpec::new("/bin/cat").stdio(Stdio {
        stdin: Pipe::Piped,
        ..Stdio::default()
    });
    let (exit, out) = tokio::time::timeout(BOUND, async {
        let mut child = sandbox.spawn(spec).await.expect("the process starts");
        let stdin = child.take_stdin().expect("stdin is piped");
        let stdout = child.take_stdout().expect("stdout is piped");
        stdin.send(b"fed ".to_vec()).await.unwrap();
        stdin.send(b"through".to_vec()).await.unwrap();
        drop(stdin);
        (child.wait().await, drain(stdout).await)
    })
    .await
    .expect("cat ends once its input closes");
    assert!(exit.success(), "{exit:?}");
    assert_eq!(out, "fed through");
}

#[tokio::test]
async fn wait_answers_the_same_before_and_after_the_exit_and_to_every_caller() {
    let (_work, work) = canonical_tempdir();
    let sandbox = host_sandbox(&work).await;
    let mut child = sandbox
        .spawn(SpawnSpec::new("/bin/sh").args(["-c", "exit 7"]))
        .await
        .unwrap();
    let (first, second) = tokio::join!(child.wait(), child.wait());
    assert_eq!(first, second);
    assert_eq!(exit_code(&first), Some(7));
    assert_eq!(
        child.wait().await,
        first,
        "a late wait gets the same answer"
    );
    drop((child.take_stdout(), child.take_stderr()));
}

#[tokio::test]
async fn kill_ends_the_whole_process_tree() {
    let (_work, work) = canonical_tempdir();
    let sandbox = host_sandbox(&work).await;
    let mut child = sandbox
        .spawn(SpawnSpec::new("/bin/sh").args(["-c", "sleep 300 & echo $!; wait"]))
        .await
        .expect("the process starts");
    let stdout = child.take_stdout().unwrap();
    let grandchild = tokio::time::timeout(BOUND, stdout.recv())
        .await
        .expect("the descendant reports its id")
        .expect("stdout is open");
    let grandchild = String::from_utf8(grandchild).unwrap().trim().to_string();
    assert!(alive(&grandchild), "the descendant runs");

    child.kill().await.expect("the tree is killed");
    let exit = tokio::time::timeout(BOUND, child.wait())
        .await
        .expect("the killed process is waited for");
    assert!(!exit.success(), "{exit:?}");
    tokio::time::timeout(BOUND, async {
        while alive(&grandchild) {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the descendant is gone with its parent");
    child
        .kill()
        .await
        .expect("killing an ended tree does nothing");
}

/// A grant changed through the sandbox's handle applies to the next spawn
/// and to none already running.
#[tokio::test]
async fn live_grants_apply_to_the_next_spawn() {
    let (_work, work) = canonical_tempdir();
    let (_target, target) = canonical_tempdir();
    let file = target.join("written");
    let script = format!("touch {}", file.display());
    let sandbox = host_sandbox(&work).await;

    let (exit, ..) = run(&sandbox, &script).await;
    assert!(!exit.success(), "nothing grants the directory yet");
    assert!(!file.exists());

    sandbox.grants().add_writable(&target);
    let (exit, ..) = run(&sandbox, &script).await;
    assert!(exit.success(), "{exit:?}");
    assert!(file.exists());

    std::fs::remove_file(&file).unwrap();
    sandbox.grants().revoke(&target);
    let (exit, ..) = run(&sandbox, &script).await;
    assert!(!exit.success(), "the revoked grant no longer applies");
    assert!(!file.exists());
}

/// A writable directory executes what it holds; a readable one does not.
#[tokio::test]
async fn writable_directories_execute_and_readable_ones_do_not() {
    use std::os::unix::fs::PermissionsExt as _;
    let (_work, work) = canonical_tempdir();
    let (_writable, writable) = canonical_tempdir();
    let (_readable, readable) = canonical_tempdir();
    for dir in [&writable, &readable] {
        let script = dir.join("tool.sh");
        std::fs::write(&script, "#!/bin/sh\necho ran\n").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let sandbox = host_sandbox(&work).await;
    sandbox.grants().add_writable(&writable);
    sandbox.grants().add_readable(&readable);

    let (exit, out, _) = run_spec(
        &sandbox,
        SpawnSpec::new(writable.join("tool.sh").to_str().unwrap()),
    )
    .await;
    assert!(exit.success(), "{exit:?}");
    assert_eq!(out, "ran\n");
    let (_, out, _) = run(
        &sandbox,
        &format!("cat {}", readable.join("tool.sh").display()),
    )
    .await;
    assert!(out.contains("echo ran"), "a readable path reads: {out}");
    let spec = SpawnSpec::new(readable.join("tool.sh").to_str().unwrap());
    match sandbox.spawn(spec).await {
        Err(_) => {}
        Ok(mut child) => {
            drop((child.take_stdout(), child.take_stderr()));
            assert!(
                !child.wait().await.success(),
                "a readable path does not execute"
            );
        }
    }
}

/// A sandbox derived from another never writes what only the parent may.
#[tokio::test]
async fn a_derived_sandbox_cannot_widen_its_parent() {
    let (_work, work) = canonical_tempdir();
    let (_parents, parents) = canonical_tempdir();
    let (_owns, owns) = canonical_tempdir();
    let sandbox = host_sandbox(&work).await;
    sandbox.grants().add_writable(&parents);
    let child = sandbox.derived(GrantSnapshot {
        writable: vec![owns.clone()],
        ..GrantSnapshot::default()
    });

    let touch = |dir: &Path| format!("touch {}", dir.join("f").display());
    let (exit, ..) = run(&child, &touch(&owns)).await;
    assert!(
        exit.success(),
        "the derived sandbox writes its own: {exit:?}"
    );
    let (exit, ..) = run(&child, &touch(&parents)).await;
    assert!(
        !exit.success(),
        "the parent's writable path is not the child's"
    );
    assert!(!parents.join("f").exists());
    let (exit, ..) = run(&sandbox, &touch(&parents)).await;
    assert!(exit.success(), "the parent keeps its own: {exit:?}");
    assert!(parents.join("f").exists());
}

/// The spec's environment leads the source's, which leads the sandbox's.
#[tokio::test]
async fn environment_layers_resolve_spec_over_source_over_sandbox() {
    let (_work, work) = canonical_tempdir();
    let pair = |name: &str, value: &str| (name.to_string(), value.to_string());
    let sandbox = host_sandbox(&work)
        .await
        .with_environment(vec![pair("LAYER_A", "sandbox"), pair("LAYER_B", "sandbox")])
        .with_env_source(vec![pair("LAYER_B", "source"), pair("LAYER_C", "source")]);
    let spec = SpawnSpec::new("/bin/sh")
        .args(["-c", "echo $LAYER_A $LAYER_B $LAYER_C"])
        .env([pair("LAYER_C", "spec")]);
    let (exit, out, _) = run_spec(&sandbox, spec).await;
    assert!(exit.success(), "{exit:?}");
    assert_eq!(out, "sandbox source spec\n");
}

/// A container exec that records its script, prints once and then waits for
/// the kill switch.
struct HangingExec {
    scripts: Arc<Mutex<Vec<String>>>,
}

impl ContainerExec for HangingExec {
    fn exec(
        &self,
        _container_id: &str,
        _script: &str,
        _working_dir: &str,
        _kill_rx: Receiver<()>,
        _stdin_rx: Receiver<Vec<u8>>,
        _lifetime: crate::ExecLifetime,
    ) -> impl std::future::Future<Output = Result<ContainerExecOutcome, String>> + Send {
        std::future::ready(Err("the spawn layer streams its execs".into()))
    }

    fn exec_streamed(
        &self,
        _container_id: &str,
        script: &str,
        _working_dir: &str,
        kill_rx: Receiver<()>,
        _stdin_rx: Receiver<Vec<u8>>,
        stdout: Sender<Vec<u8>>,
    ) -> impl std::future::Future<Output = Result<ContainerExecOutcome, String>> + Send {
        self.scripts.lock().unwrap().push(script.to_string());
        async move {
            let _ = stdout.send(b"started\n".to_vec()).await;
            let _ = kill_rx.recv().await;
            Ok(ContainerExecOutcome::Killed)
        }
    }
}

fn container_sandbox(scripts: &Arc<Mutex<Vec<String>>>, working_dir: &Path) -> Sandbox {
    let runtime = ContainerShellRuntime::new(
        "aither-test-spawn",
        HostRoute::new(
            "host.docker.internal",
            std::net::IpAddr::from([127, 0, 0, 1]),
        )
        .expect("loopback is routable"),
        AnyContainerExec::new(Arc::new(HangingExec {
            scripts: scripts.clone(),
        })),
    );
    let tools =
        crate::tool_runtime::tests::container_tools(&crate::tool_runtime::tests::image_tools());
    Sandbox::new(
        Backend::Container { runtime, tools },
        TokioGlobal,
        working_dir,
    )
}

#[tokio::test]
async fn a_container_process_streams_and_kill_ends_its_exec() {
    let (_work, work) = canonical_tempdir();
    let scripts = Arc::new(Mutex::new(Vec::new()));
    let pair = |name: &str, value: &str| (name.to_string(), value.to_string());
    let sandbox = container_sandbox(&scripts, &work)
        .with_env_source(vec![pair("FROM_SOURCE", "source-value")]);
    let spec = SpawnSpec::new("long-job")
        .args(["--flag"])
        .env([pair("FROM_SPEC", "spec-value")]);
    let mut child: SandboxedChild = sandbox.spawn(spec).await.expect("the exec starts");
    let stdout = child.take_stdout().unwrap();
    let first = tokio::time::timeout(BOUND, stdout.recv())
        .await
        .expect("the exec prints")
        .expect("stdout is open");
    assert_eq!(first, b"started\n");

    let script = scripts.lock().unwrap().join("\n");
    assert!(script.contains("long-job --flag"), "{script}");
    assert!(
        script.contains("FROM_SPEC") && script.contains("spec-value"),
        "{script}"
    );
    assert!(
        script.contains("FROM_SOURCE") && script.contains("source-value"),
        "{script}"
    );

    child.kill().await.expect("the exec is killed");
    let exit = tokio::time::timeout(BOUND, child.wait())
        .await
        .expect("the killed exec is waited for");
    assert_eq!(exit, ChildExit::Killed);
    assert_eq!(child.wait().await, ChildExit::Killed);
}
