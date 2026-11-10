//! Building a [`Shell`] on a [`Sandbox`].

use std::sync::Arc;

use executor_core::AnyExecutor;

use super::{Interpreter, Shell, ShellToolSettings, TerminalError};
use crate::output_treatment::AnyTreatOutput;
use crate::{
    AnyIpcProvider, AnyScriptPolicy, AnyShellWording, HostLaunch, IpcProvider, OutputPolicy,
    Sandbox, ScriptPolicy, ShellEvents, ShellLimits, ShellWording, TreatOutput,
};

/// A value the builder has not been given yet.
#[derive(Debug, Clone, Copy)]
pub struct Missing;

/// A value the builder was given.
#[derive(Debug, Clone)]
pub struct Given<T>(T);

/// What a shell is built from besides its sandbox.
pub struct ShellParts {
    pub(crate) interpreters: Vec<Interpreter>,
    pub(crate) limits: ShellLimits,
    pub(crate) output: OutputPolicy,
    pub(crate) wording: AnyShellWording,
    pub(crate) tools: ShellToolSettings,
    pub(crate) events: Option<ShellEvents>,
    pub(crate) policies: Vec<AnyScriptPolicy>,
    pub(crate) ipc: Option<AnyIpcProvider>,
    pub(crate) treat_output: Option<AnyTreatOutput>,
    pub(crate) host_shell: Option<HostLaunch>,
}

/// Builds a [`Shell`] on the [`Sandbox`] its scripts run in.
///
/// The interpreters, limits, output policy, wording and tool identities have
/// no defaults: [`build`](Self::build) exists once every one is given.
#[derive(Debug)]
pub struct ShellBuilder<I, L, O, W, T> {
    sandbox: Sandbox,
    interpreters: I,
    limits: L,
    output: O,
    wording: W,
    tools: T,
    events: Option<ShellEvents>,
    policies: Vec<AnyScriptPolicy>,
    ipc: Option<AnyIpcProvider>,
    treat_output: Option<AnyTreatOutput>,
    host_shell: Option<HostLaunch>,
}

impl Shell<Arc<AnyExecutor>> {
    /// Starts a shell whose scripts run in `sandbox`, a sandbox on a
    /// container: its working directory, live grants, environment source,
    /// network policy, priority and backend are the shell's, and the shell
    /// is the one the container's launch names. A sandbox on the host takes
    /// [`builder_on_host`](Self::builder_on_host).
    #[must_use]
    pub const fn builder(
        sandbox: Sandbox,
    ) -> ShellBuilder<Missing, Missing, Missing, Missing, Missing> {
        ShellBuilder {
            sandbox,
            interpreters: Missing,
            limits: Missing,
            output: Missing,
            wording: Missing,
            tools: Missing,
            events: None,
            policies: Vec::new(),
            ipc: None,
            treat_output: None,
            host_shell: None,
        }
    }

    /// Starts a shell whose scripts run in `sandbox`, a sandbox on the host,
    /// under the shell `host_shell` names. A host sandbox spawns programs and
    /// names no shell of its own, so the shell is an input of the shell that
    /// needs one.
    #[must_use]
    pub const fn builder_on_host(
        sandbox: Sandbox,
        host_shell: HostLaunch,
    ) -> ShellBuilder<Missing, Missing, Missing, Missing, Missing> {
        ShellBuilder {
            sandbox,
            interpreters: Missing,
            limits: Missing,
            output: Missing,
            wording: Missing,
            tools: Missing,
            events: None,
            policies: Vec::new(),
            ipc: None,
            treat_output: None,
            host_shell: Some(host_shell),
        }
    }
}

impl<I, L, O, W, T> ShellBuilder<I, L, O, W, T> {
    /// Lets every script of the shell, and of the subagent shells it
    /// launches, call the commands `provider` serves back on the host. The
    /// provider is asked once per process the shell starts on the host or in
    /// a container. A process on an SSH server has no way back to this
    /// machine, so the shell asks for none there. Without a provider no
    /// script reaches a command of the embedder.
    #[must_use]
    pub fn ipc(mut self, provider: impl IpcProvider) -> Self {
        self.ipc = Some(AnyIpcProvider::new(provider));
        self
    }

    /// Puts every script to `policy`, after the policies given before it.
    /// Each agent's shell gets state of its own from the policy's
    /// definition, see [`ScriptPolicies::for_agent`](crate::ScriptPolicies).
    #[must_use]
    pub fn policy(mut self, policy: impl ScriptPolicy) -> Self {
        self.policies.push(AnyScriptPolicy::new(policy));
        self
    }

    /// Lets `treat` choose, before each script of the shell and of the
    /// subagent shells it launches runs, whether its output reaches the model
    /// verbatim or cleaned as command output; it replaces one given before.
    /// Without it every script's output is cleaned as command output, see
    /// [`OutputTreatment`](crate::OutputTreatment).
    #[must_use]
    pub fn treat_output(mut self, treat: impl TreatOutput) -> Self {
        self.treat_output = Some(AnyTreatOutput::new(treat));
        self
    }

    /// Raises the shell's events, and those of the subagent shells it
    /// launches, on `events`, which other shells may share. Without it the
    /// shell has a handle of its own, read with [`Shell::events`].
    #[must_use]
    pub fn events(mut self, events: ShellEvents) -> Self {
        self.events = Some(events);
        self
    }

    /// The interpreters a script may run under, and the ones the model's
    /// `interpreter` argument lists.
    #[must_use]
    pub fn interpreters(
        self,
        interpreters: impl IntoIterator<Item = Interpreter>,
    ) -> ShellBuilder<Given<Vec<Interpreter>>, L, O, W, T> {
        ShellBuilder {
            sandbox: self.sandbox,
            interpreters: Given(interpreters.into_iter().collect()),
            limits: self.limits,
            output: self.output,
            wording: self.wording,
            tools: self.tools,
            events: self.events,
            policies: self.policies,
            ipc: self.ipc,
            treat_output: self.treat_output,
            host_shell: self.host_shell,
        }
    }

    /// The time bounds of every script.
    #[must_use]
    pub fn limits(self, limits: ShellLimits) -> ShellBuilder<I, Given<ShellLimits>, O, W, T> {
        ShellBuilder {
            sandbox: self.sandbox,
            interpreters: self.interpreters,
            limits: Given(limits),
            output: self.output,
            wording: self.wording,
            tools: self.tools,
            events: self.events,
            policies: self.policies,
            ipc: self.ipc,
            treat_output: self.treat_output,
            host_shell: self.host_shell,
        }
    }

    /// The budgets of what a script's output returns inline.
    #[must_use]
    pub fn output(self, output: OutputPolicy) -> ShellBuilder<I, L, Given<OutputPolicy>, W, T> {
        ShellBuilder {
            sandbox: self.sandbox,
            interpreters: self.interpreters,
            limits: self.limits,
            output: Given(output),
            wording: self.wording,
            tools: self.tools,
            events: self.events,
            policies: self.policies,
            ipc: self.ipc,
            treat_output: self.treat_output,
            host_shell: self.host_shell,
        }
    }

    /// What words every text the model reads.
    #[must_use]
    pub fn wording(
        self,
        wording: impl ShellWording,
    ) -> ShellBuilder<I, L, O, Given<AnyShellWording>, T> {
        ShellBuilder {
            sandbox: self.sandbox,
            interpreters: self.interpreters,
            limits: self.limits,
            output: self.output,
            wording: Given(AnyShellWording::new(wording)),
            tools: self.tools,
            events: self.events,
            policies: self.policies,
            ipc: self.ipc,
            treat_output: self.treat_output,
            host_shell: self.host_shell,
        }
    }

    /// What the shell tool and the job control tools are named and
    /// described as.
    #[must_use]
    pub fn tools(
        self,
        tools: ShellToolSettings,
    ) -> ShellBuilder<I, L, O, W, Given<ShellToolSettings>> {
        ShellBuilder {
            sandbox: self.sandbox,
            interpreters: self.interpreters,
            limits: self.limits,
            output: self.output,
            wording: self.wording,
            tools: Given(tools),
            events: self.events,
            policies: self.policies,
            ipc: self.ipc,
            treat_output: self.treat_output,
            host_shell: self.host_shell,
        }
    }
}

impl
    ShellBuilder<
        Given<Vec<Interpreter>>,
        Given<ShellLimits>,
        Given<OutputPolicy>,
        Given<AnyShellWording>,
        Given<ShellToolSettings>,
    >
{
    /// Builds the shell.
    ///
    /// # Errors
    /// Returns an error when the working directory or the output store cannot
    /// be created, when the sandbox runs its processes on an SSH server (a
    /// server is a location a script names, not a shell's home), or when the
    /// constructor does not fit the sandbox's backend: a host sandbox built
    /// with [`Shell::builder`], or a container sandbox built with
    /// [`Shell::builder_on_host`].
    pub async fn build(self) -> Result<Shell<Arc<AnyExecutor>>, TerminalError> {
        let Self {
            sandbox,
            interpreters: Given(interpreters),
            limits: Given(limits),
            output: Given(output),
            wording: Given(wording),
            tools: Given(tools),
            events,
            policies,
            ipc,
            treat_output,
            host_shell,
        } = self;
        let executor = sandbox.executor();
        Shell::from_parts(
            sandbox,
            executor,
            ShellParts {
                interpreters,
                limits,
                output,
                wording,
                tools,
                events,
                policies,
                ipc,
                treat_output,
                host_shell,
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use aither_core::llm::Tool;

    use super::*;
    use crate::spawn::Grants;
    use crate::{Backend, test_budgets, test_wording};

    async fn shell(
        directory: &std::path::Path,
        grants: Grants,
        interpreters: &[Interpreter],
    ) -> Shell<Arc<AnyExecutor>> {
        let heel = crate::test_heel::binary().await.expect("a heel for tests");
        let sandbox = Sandbox::new(
            Backend::Host { heel },
            executor_core::tokio::TokioGlobal,
            directory,
        )
        .with_grants(grants);
        Shell::builder_on_host(sandbox, HostLaunch::new("/bin/bash"))
            .interpreters(interpreters.iter().copied())
            .limits(test_budgets::limits())
            .output(test_budgets::output_policy())
            .wording(test_wording::TestWording)
            .tools(test_wording::job_tools())
            .build()
            .await
            .expect("the shell builds")
    }

    async fn ipc_shell(
        directory: &std::path::Path,
        provider: crate::spawn::ipc_fixture::Recording,
    ) -> Shell<Arc<AnyExecutor>> {
        let heel = crate::test_heel::binary().await.expect("a heel for tests");
        let sandbox = Sandbox::new(
            Backend::Host { heel },
            executor_core::tokio::TokioGlobal,
            directory,
        );
        Shell::builder_on_host(sandbox, HostLaunch::new("/bin/bash"))
            .interpreters([Interpreter::Shell])
            .limits(test_budgets::limits())
            .output(test_budgets::output_policy())
            .wording(test_wording::TestWording)
            .tools(test_wording::job_tools())
            .ipc(provider)
            .build()
            .await
            .expect("the shell builds")
    }

    /// Keeps the output of a script that reads a file with `cat` verbatim.
    struct CatIsVerbatim;

    impl TreatOutput for CatIsVerbatim {
        fn treatment(&self, script: &crate::ScriptCheck<'_>) -> crate::OutputTreatment {
            if script.script.starts_with("cat ") {
                crate::OutputTreatment::Verbatim
            } else {
                crate::OutputTreatment::Command
            }
        }
    }

    async fn treated_shell(
        directory: &std::path::Path,
        treat: Option<CatIsVerbatim>,
    ) -> Shell<Arc<AnyExecutor>> {
        let heel = crate::test_heel::binary().await.expect("a heel for tests");
        let sandbox = Sandbox::new(
            Backend::Host { heel },
            executor_core::tokio::TokioGlobal,
            directory,
        );
        let builder = Shell::builder_on_host(sandbox, HostLaunch::new("/bin/bash"))
            .interpreters([Interpreter::Shell])
            .limits(test_budgets::limits())
            .output(test_budgets::output_policy())
            .wording(test_wording::TestWording)
            .tools(test_wording::job_tools());
        let builder = match treat {
            Some(treat) => builder.treat_output(treat),
            None => builder,
        };
        builder.build().await.expect("the shell builds")
    }

    async fn run_in(
        shell: &Shell<Arc<AnyExecutor>>,
        dir: &std::path::Path,
        script: &str,
    ) -> crate::ScriptResult {
        let result = shell
            .run(crate::Script::new(
                script,
                dir,
                std::time::Duration::from_secs(30),
            ))
            .await
            .expect("the script runs");
        assert!(
            matches!(result.outcome, crate::RunOutcome::Exited { code: 0 }),
            "{result:?}"
        );
        result
    }

    /// A file whose blank-line runs command-output cleaning collapses.
    const NOTES: &str = "# Notes\n\n\n\nfirst\n\n\nsecond\n";

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_shell_without_a_treatment_cleans_every_output_as_command_output() {
        let directory = tempfile::tempdir().unwrap();
        let start = directory.path().canonicalize().unwrap();
        std::fs::write(start.join("notes.md"), NOTES).unwrap();
        let shell = treated_shell(&start, None).await;
        let read = run_in(&shell, &start, "cat notes.md").await;
        assert_ne!(stdout_of(&read), NOTES, "the blank runs collapse");
        assert!(
            stdout_of(&read).contains("blank lines collapsed"),
            "{read:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn the_treatment_keeps_a_read_verbatim_and_cleans_command_output() {
        let directory = tempfile::tempdir().unwrap();
        let start = directory.path().canonicalize().unwrap();
        std::fs::write(start.join("notes.md"), NOTES).unwrap();
        let shell = treated_shell(&start, Some(CatIsVerbatim)).await;

        let read = run_in(&shell, &start, "cat notes.md").await;
        assert_eq!(
            stdout_of(&read),
            NOTES,
            "the file reaches the model as it is"
        );

        let command = run_in(&shell, &start, "printf '# Notes\\n\\n\\n\\nfirst\\n'").await;
        assert!(
            stdout_of(&command).contains("blank lines collapsed"),
            "{command:?}"
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_verbatim_output_over_the_budget_stores_the_whole_text() {
        let directory = tempfile::tempdir().unwrap();
        let start = directory.path().canonicalize().unwrap();
        let lines = test_budgets::output_policy().lines(None) * 2;
        let long = (0..lines)
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n\n\n");
        std::fs::write(start.join("long.md"), &long).unwrap();
        let shell = treated_shell(&start, Some(CatIsVerbatim)).await;

        let read = run_in(&shell, &start, "cat long.md").await;
        let aither_core::llm::CommandStream::Stored { file, preview, .. } = &read.stdout else {
            panic!("output over the line budget is stored: {read:?}");
        };
        assert!(!preview.is_empty(), "the model reads a preview");
        assert_eq!(
            std::fs::read_to_string(&file.path).unwrap(),
            long,
            "the stored file holds the whole text, blank runs included"
        );
    }

    fn stdout_of(result: &crate::ScriptResult) -> &str {
        match &result.stdout {
            aither_core::llm::CommandStream::Inline { text } => text,
            other => panic!("a short answer is inline: {other:?}"),
        }
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_script_calls_the_command_the_providers_router_serves() {
        let directory = tempfile::tempdir().unwrap();
        let start = directory.path().canonicalize().unwrap();
        let provider = crate::spawn::ipc_fixture::Recording::default();
        let shell = ipc_shell(&start, provider.clone()).await;
        let result = shell
            .run(crate::Script::new(
                "probe",
                &start,
                std::time::Duration::from_secs(30),
            ))
            .await
            .expect("the script runs");
        assert_eq!(
            stdout_of(&result).trim(),
            crate::spawn::ipc_fixture::ANSWER,
            "{result:?}"
        );
        let launches = provider.launches.lock().unwrap();
        assert_eq!(launches.len(), 1, "one process, one question");
        assert!(launches[0].task_id.is_some(), "{:?}", launches[0]);
        assert_eq!(launches[0].mode, crate::SandboxMode::Sandboxed);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_shell_without_a_provider_serves_no_command() {
        let directory = tempfile::tempdir().unwrap();
        let start = directory.path().canonicalize().unwrap();
        let shell = shell(&start, Grants::new(), &[Interpreter::Shell]).await;
        // The `;` makes the script one the managed shell runs: a bare
        // `probe` is a direct command, which Linux's supervisor refuses to
        // start at all and macOS reports as an exit.
        let result = shell
            .run(crate::Script::new(
                "probe;",
                &start,
                std::time::Duration::from_secs(30),
            ))
            .await
            .expect("the script runs");
        assert!(
            matches!(result.outcome, crate::RunOutcome::Exited { code: 127 }),
            "the shell finds no `probe`: {result:?}"
        );
    }

    #[tokio::test]
    async fn the_shell_is_built_on_the_sandboxs_directory_and_live_grants() {
        let directory = tempfile::tempdir().unwrap();
        let grants = Grants::new();
        let shell = shell(directory.path(), grants.clone(), &[Interpreter::Shell]).await;
        assert_eq!(
            shell.working_dir,
            dunce::simplified(&directory.path().canonicalize().unwrap())
        );
        let granted = directory.path().join("granted");
        grants.add_writable(&granted);
        assert_eq!(
            shell.sandbox.grants().snapshot().writable,
            grants.snapshot().writable,
            "the shell sees the grants the sandbox's handle changes"
        );
    }

    #[tokio::test]
    async fn the_tool_schema_lists_exactly_the_configured_interpreters() {
        let directory = tempfile::tempdir().unwrap();
        for (configured, listed) in [
            (vec![Interpreter::Shell], serde_json::json!(["shell"])),
            (
                vec![Interpreter::Shell, Interpreter::Python],
                serde_json::json!(["shell", "python"]),
            ),
        ] {
            let shell = shell(directory.path(), Grants::new(), &configured).await;
            let tool = shell.tool();
            assert_eq!(tool.name(), "terminal");
            let schema = tool.arguments_schema().to_value();
            assert_eq!(schema["$defs"]["Interpreter"]["enum"], listed);
        }
    }

    #[tokio::test]
    async fn every_argument_is_described_by_the_identity_alone() {
        let directory = tempfile::tempdir().unwrap();
        let shell = shell(directory.path(), Grants::new(), &[Interpreter::Shell]).await;
        let schema = shell.tool().arguments_schema().to_value();
        let described = test_wording::argument_descriptions();
        let properties = schema["properties"].as_object().expect("an object schema");
        assert!(!properties.is_empty());
        for (name, property) in properties {
            assert_eq!(
                property["description"].as_str(),
                Some(&*described[name.as_str()]),
                "{name}"
            );
        }
        assert!(schema.get("description").is_none(), "{schema}");
        assert!(
            schema["$defs"]["Interpreter"].get("description").is_none(),
            "{schema}"
        );
    }

    #[tokio::test]
    async fn a_script_in_an_interpreter_the_shell_lacks_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let shell = shell(directory.path(), Grants::new(), &[Interpreter::Shell]).await;
        let arguments: super::super::ShellArgs = serde_json::from_value(serde_json::json!({
            "description": "print",
            "script": "print(1)",
            "interpreter": "python",
            "timeout": 5
        }))
        .unwrap();
        let error = shell
            .tool()
            .call(arguments, aither_core::llm::ToolContext::new())
            .await
            .expect_err("python is not configured");
        assert!(error.to_string().contains("not configured"), "{error}");
    }

    #[tokio::test]
    async fn a_host_sandbox_built_without_its_host_shell_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let heel = crate::test_heel::binary().await.expect("a heel for tests");
        let sandbox = Sandbox::new(
            Backend::Host { heel },
            executor_core::tokio::TokioGlobal,
            directory.path(),
        );
        let error = Shell::builder(sandbox)
            .interpreters([Interpreter::Shell])
            .limits(test_budgets::limits())
            .output(test_budgets::output_policy())
            .wording(test_wording::TestWording)
            .tools(test_wording::job_tools())
            .build()
            .await
            .expect_err("a host shell needs its program");
        assert!(error.to_string().contains("host shell"), "{error}");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn the_shells_sandbox_spawns_a_program_in_the_shells_directory() {
        let directory = tempfile::tempdir().unwrap();
        let shell = shell(directory.path(), Grants::new(), &[Interpreter::Shell]).await;
        let mut child = shell
            .sandbox()
            .spawn(crate::SpawnSpec::new("/bin/pwd"))
            .await
            .expect("the process starts");
        let stdout = child.take_stdout().expect("stdout is piped");
        let mut printed = Vec::new();
        while let Ok(chunk) = stdout.recv().await {
            printed.extend(chunk);
        }
        child.wait().await;
        let printed = String::from_utf8(printed).unwrap();
        assert_eq!(
            std::path::Path::new(printed.trim()),
            shell.sandbox().working_dir()
        );
    }
}
