//! Building a [`Shell`] on a [`Sandbox`].

use std::sync::Arc;

use executor_core::AnyExecutor;

use super::{Interpreter, Shell, ShellToolSettings, TerminalError, Unconfigured};
use crate::{AnyShellWording, OutputPolicy, Sandbox, ShellLimits, ShellWording};

/// A value the builder has not been given yet.
#[derive(Debug, Clone, Copy)]
pub struct Missing;

/// A value the builder was given.
#[derive(Debug, Clone)]
pub struct Given<T>(T);

/// What a shell is built from besides its sandbox.
pub(crate) struct ShellParts {
    pub(crate) interpreters: Vec<Interpreter>,
    pub(crate) limits: ShellLimits,
    pub(crate) output: OutputPolicy,
    pub(crate) wording: AnyShellWording,
    pub(crate) tools: ShellToolSettings,
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
}

impl Shell<Arc<AnyExecutor>, Unconfigured> {
    /// Starts a shell whose scripts run in `sandbox`: its working directory,
    /// live grants, environment source, network policy, priority and
    /// backend are the shell's.
    #[must_use]
    pub fn builder(sandbox: Sandbox) -> ShellBuilder<Missing, Missing, Missing, Missing, Missing> {
        ShellBuilder {
            sandbox,
            interpreters: Missing,
            limits: Missing,
            output: Missing,
            wording: Missing,
            tools: Missing,
        }
    }
}

impl<I, L, O, W, T> ShellBuilder<I, L, O, W, T> {
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
    /// be created, or when the sandbox runs its processes on an SSH server:
    /// a server is a location a script names, not a shell's home.
    pub async fn build(self) -> Result<Shell<Arc<AnyExecutor>, Unconfigured>, TerminalError> {
        let Self {
            sandbox,
            interpreters: Given(interpreters),
            limits: Given(limits),
            output: Given(output),
            wording: Given(wording),
            tools: Given(tools),
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
    ) -> Shell<Arc<AnyExecutor>, Unconfigured> {
        let heel = crate::test_heel::pinned().await.expect("the pinned heel");
        let sandbox = Sandbox::new(
            Backend::Host { heel },
            executor_core::tokio::TokioGlobal,
            directory,
        )
        .with_grants(grants);
        Shell::builder(sandbox)
            .interpreters(interpreters.iter().copied())
            .limits(test_budgets::limits())
            .output(test_budgets::output_policy())
            .wording(test_wording::TestWording)
            .tools(test_wording::job_tools())
            .build()
            .await
            .expect("the shell builds")
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
            let registry = crate::ToolRegistryBuilder::new().build(shell.output_store().clone());
            let tool = shell.with_registry(Arc::new(registry)).tool();
            assert_eq!(tool.name(), "terminal");
            let schema = tool.arguments_schema().to_value();
            assert_eq!(schema["$defs"]["Interpreter"]["enum"], listed);
        }
    }

    #[tokio::test]
    async fn every_argument_is_described_by_the_identity_alone() {
        let directory = tempfile::tempdir().unwrap();
        let shell = shell(directory.path(), Grants::new(), &[Interpreter::Shell]).await;
        let registry = crate::ToolRegistryBuilder::new().build(shell.output_store().clone());
        let schema = shell
            .with_registry(Arc::new(registry))
            .tool()
            .arguments_schema()
            .to_value();
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
        let registry = crate::ToolRegistryBuilder::new().build(shell.output_store().clone());
        let shell = shell.with_registry(Arc::new(registry));
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
}
