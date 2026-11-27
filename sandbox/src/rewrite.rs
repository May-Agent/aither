//! Static shell script rewriting: an embedder hook that transforms a shell
//! script before execution when `raw` is `false` and optionally attaches a
//! natural-language notice to the command's result.
//!
//! When an embedder configures a [`RewriteScript`] on [`crate::ShellBuilder`],
//! every [`Interpreter::Shell`](crate::Interpreter::Shell) run with
//! `raw == false` passes through [`RewriteScript::rewrite`] after
//! [`ScriptPolicy`](crate::ScriptPolicy) checks accept the original script and
//! before [`TreatOutput`](crate::TreatOutput) and
//! [`PlanPipelines`](crate::PlanPipelines) inspect the final script.

use std::fmt;
use std::sync::Arc;

use crate::ScriptCheck;

/// The outcome of statically rewriting a shell script before execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptRewrite {
    /// Replacement shell script executed in place of the original.
    pub script: String,
    /// Optional natural-language notice surfaced alongside the command's result.
    pub notice: Option<String>,
}

/// Statically rewrites a shell script before execution when `raw` is `false`.
///
/// Configured on [`ShellBuilder::rewrite_script`](crate::ShellBuilder::rewrite_script)
/// and shared with every subagent shell derived from that shell. Invoked only
/// for [`Interpreter::Shell`](crate::Interpreter::Shell) commands when `raw` is
/// `false`, after [`ScriptPolicy`](crate::ScriptPolicy) checks accept the
/// original script. When [`Self::rewrite`] returns `Some`, the shell re-parses
/// the replacement script once for [`TreatOutput`](crate::TreatOutput) and
/// [`PlanPipelines`](crate::PlanPipelines), executes the replacement script,
/// and attaches any [`ScriptRewrite::notice`] to the command's tool result.
pub trait RewriteScript: Send + Sync + 'static {
    /// Returns a [`ScriptRewrite`] to execute in place of `script`, or `None`
    /// to execute `script` unchanged.
    fn rewrite(&self, script: &ScriptCheck<'_>) -> Option<ScriptRewrite>;
}

/// A type-erased [`RewriteScript`], shared by a shell and the subagent shells
/// derived from it.
#[derive(Clone)]
pub struct AnyRewriteScript(Arc<dyn RewriteScript>);

impl AnyRewriteScript {
    /// Erases `rewriter`.
    #[must_use]
    pub fn new(rewriter: impl RewriteScript) -> Self {
        Self(Arc::new(rewriter))
    }

    /// Evaluates the rewriter on `script`.
    #[must_use]
    pub fn rewrite(&self, script: &ScriptCheck<'_>) -> Option<ScriptRewrite> {
        self.0.rewrite(script)
    }
}

impl fmt::Debug for AnyRewriteScript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyRewriteScript")
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use aither_core::llm::{CommandOutcome, CommandStream, Tool, ToolContext};
    use executor_core::AnyExecutor;

    use super::*;
    use crate::{
        Backend, HostLaunch, Interpreter, OutputTreatment, PlanPipelines, Sandbox, SandboxMode,
        ScriptDecision, ScriptDir, ScriptOrigin, ScriptPolicy, ScriptRefusal, Shell, ShellArgs,
        StageDirective, StageSyntax, SubagentGrant, TreatOutput, test_budgets, test_wording,
    };

    struct EchoRewriter {
        calls: Arc<AtomicUsize>,
    }

    impl RewriteScript for EchoRewriter {
        fn rewrite(&self, script: &ScriptCheck<'_>) -> Option<ScriptRewrite> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if script.script.trim() == "echo original" {
                Some(ScriptRewrite {
                    script: "echo rewritten".to_owned(),
                    notice: Some("Note: command was rewritten before execution.".to_owned()),
                })
            } else if script.script.trim() == "sleep 60" {
                Some(ScriptRewrite {
                    script: "echo bg-rewritten; sleep 60".to_owned(),
                    notice: Some("Note: background command was rewritten.".to_owned()),
                })
            } else {
                None
            }
        }
    }

    #[test]
    fn any_rewriter_delegates_and_formats_debug() {
        let calls = Arc::new(AtomicUsize::new(0));
        let erased = AnyRewriteScript::new(EchoRewriter {
            calls: calls.clone(),
        });
        assert_eq!(format!("{erased:?}"), "AnyRewriteScript");
        let dir = ScriptDir::Host(PathBuf::from("/"));
        let check = ScriptCheck {
            launch: crate::LaunchId::from_raw(1),
            script: "echo original",
            interpreter: Interpreter::Shell,
            shell: None,
            origin: ScriptOrigin::Foreground,
            sandbox: SandboxMode::Sandboxed,
            location: None,
            start: &dir,
            call: None,
        };
        let cloned = erased;
        assert_eq!(
            cloned.rewrite(&check),
            Some(ScriptRewrite {
                script: "echo rewritten".to_owned(),
                notice: Some("Note: command was rewritten before execution.".to_owned()),
            })
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[cfg(unix)]
    fn shell_args(script: &str, raw: bool, background: bool) -> ShellArgs {
        serde_json::from_value(serde_json::json!({
            "description": "test script rewrite",
            "script": script,
            "raw": raw,
            "background": background,
            "timeout": 30,
        }))
        .expect("valid ShellArgs")
    }

    #[cfg(unix)]
    async fn build_shell(
        dir: &std::path::Path,
        rewriter: impl RewriteScript,
    ) -> Shell<Arc<AnyExecutor>> {
        let heel = crate::test_heel::binary().await.expect("a heel for tests");
        let sandbox = Sandbox::new(
            Backend::Host { heel },
            executor_core::tokio::TokioGlobal,
            dir,
        );
        Shell::builder_on_host(sandbox, HostLaunch::new("/bin/bash"))
            .interpreters([Interpreter::Shell, Interpreter::Python])
            .limits(test_budgets::limits())
            .output(test_budgets::output_policy())
            .wording(test_wording::TestWording)
            .tools(test_wording::job_tools())
            .rewrite_script(rewriter)
            .build()
            .await
            .expect("shell builds")
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn raw_false_executes_rewritten_script_and_renders_notice() {
        let dir = tempfile::tempdir().unwrap();
        let start = dir.path().canonicalize().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let shell = build_shell(
            &start,
            EchoRewriter {
                calls: calls.clone(),
            },
        )
        .await;

        let result = shell
            .tool()
            .call(
                shell_args("echo original", false, false),
                ToolContext::new(),
            )
            .await
            .expect("foreground call succeeds");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let command = result.command().expect("command result");
        assert_eq!(command.outcome, CommandOutcome::Exited { code: 0 });
        assert_eq!(
            command.stdout,
            CommandStream::Inline {
                text: "rewritten\n".to_owned(),
            }
        );
        assert_eq!(
            command.notice.as_deref(),
            Some("Note: command was rewritten before execution.")
        );
        assert_eq!(
            result.render_for_model(),
            "rewritten\nNote: command was rewritten before execution.\n"
        );

        let detached = shell
            .tool()
            .call(shell_args("sleep 60", false, true), ToolContext::new())
            .await
            .expect("background call succeeds");
        let bg_command = detached.command().expect("background command result");
        let CommandOutcome::Running { pid, .. } = &bg_command.outcome else {
            panic!("expected running background job: {bg_command:?}");
        };
        assert_eq!(
            bg_command.notice.as_deref(),
            Some("Note: background command was rewritten.")
        );
        let rendered_bg = detached.render_for_model();
        assert!(
            rendered_bg.contains("Started in background with PID "),
            "{rendered_bg}"
        );
        assert!(
            rendered_bg.ends_with("Note: background command was rewritten.\n"),
            "{rendered_bg}"
        );
        assert!(shell.jobs().kill(*pid).await);
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn raw_true_skips_rewriter_and_executes_original_without_notice() {
        let dir = tempfile::tempdir().unwrap();
        let start = dir.path().canonicalize().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let shell = build_shell(
            &start,
            EchoRewriter {
                calls: calls.clone(),
            },
        )
        .await;

        let result = shell
            .tool()
            .call(shell_args("echo original", true, false), ToolContext::new())
            .await
            .expect("raw call succeeds");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "raw: true must skip RewriteScript"
        );
        let command = result.command().expect("command result");
        assert_eq!(
            command.stdout,
            CommandStream::Inline {
                text: "original\n".to_owned(),
            }
        );
        assert_eq!(command.notice, None);
        assert_eq!(result.render_for_model(), "original\n");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn returning_none_executes_original_script_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let start = dir.path().canonicalize().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let shell = build_shell(
            &start,
            EchoRewriter {
                calls: calls.clone(),
            },
        )
        .await;

        let result = shell
            .tool()
            .call(
                shell_args("echo untouched", false, false),
                ToolContext::new(),
            )
            .await
            .expect("unchanged call succeeds");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let command = result.command().expect("command result");
        assert_eq!(
            command.stdout,
            CommandStream::Inline {
                text: "untouched\n".to_owned(),
            }
        );
        assert_eq!(command.notice, None);
        assert_eq!(result.render_for_model(), "untouched\n");
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn subagent_shell_inherits_rewriter() {
        let dir = tempfile::tempdir().unwrap();
        let start = dir.path().canonicalize().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let parent = build_shell(
            &start,
            EchoRewriter {
                calls: calls.clone(),
            },
        )
        .await;
        let subagent = parent
            .subagent(SubagentGrant::new(), None, "worker")
            .await
            .expect("subagent shell derives");

        let result = subagent
            .tool()
            .call(
                shell_args("echo original", false, false),
                ToolContext::new(),
            )
            .await
            .expect("subagent call succeeds");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let command = result.command().expect("command result");
        assert_eq!(
            command.stdout,
            CommandStream::Inline {
                text: "rewritten\n".to_owned(),
            }
        );
        assert_eq!(
            command.notice.as_deref(),
            Some("Note: command was rewritten before execution.")
        );
        assert_eq!(
            result.render_for_model(),
            "rewritten\nNote: command was rewritten before execution.\n"
        );
    }

    #[cfg(unix)]
    struct RecordingPolicy {
        seen: Arc<Mutex<Vec<String>>>,
    }

    #[cfg(unix)]
    impl ScriptPolicy for RecordingPolicy {
        type State = ();

        fn check(
            &self,
            script: &ScriptCheck<'_>,
            (): &(),
        ) -> impl std::future::Future<Output = ScriptDecision> + Send {
            self.seen
                .lock()
                .unwrap()
                .push(script.script.trim().to_owned());
            let decision = if script.script.trim() == "forbidden" {
                ScriptDecision::Refuse(ScriptRefusal::new("blocked by policy"))
            } else {
                ScriptDecision::Pass
            };
            std::future::ready(decision)
        }
    }

    #[cfg(unix)]
    struct MultiRewriter {
        calls: Arc<AtomicUsize>,
    }

    #[cfg(unix)]
    impl RewriteScript for MultiRewriter {
        fn rewrite(&self, script: &ScriptCheck<'_>) -> Option<ScriptRewrite> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match script.script.trim() {
                "forbidden" => Some(ScriptRewrite {
                    script: "echo bypassed".to_owned(),
                    notice: Some("should not run".to_owned()),
                }),
                "read_notes" => Some(ScriptRewrite {
                    script: "cat notes.md".to_owned(),
                    notice: Some("Rewrote read_notes to cat notes.md.".to_owned()),
                }),
                "run_pipe" => Some(ScriptRewrite {
                    script: "printf 'alpha\\n' | cat".to_owned(),
                    notice: None,
                }),
                _ => None,
            }
        }
    }

    #[cfg(unix)]
    struct CatIsVerbatim {
        seen: Arc<Mutex<Vec<(String, bool)>>>,
    }

    #[cfg(unix)]
    impl TreatOutput for CatIsVerbatim {
        fn treatment(&self, script: &ScriptCheck<'_>) -> OutputTreatment {
            let parsed_ok = matches!(script.shell, Some(Ok(_)));
            self.seen
                .lock()
                .unwrap()
                .push((script.script.to_owned(), parsed_ok));
            if script.script.starts_with("cat ") {
                OutputTreatment::Verbatim
            } else {
                OutputTreatment::Command
            }
        }
    }

    #[cfg(unix)]
    struct RecordingPlanner {
        seen_stages: Arc<Mutex<Vec<Vec<Option<String>>>>>,
    }

    #[cfg(unix)]
    impl PlanPipelines for RecordingPlanner {
        fn plan_pipeline(
            &self,
            _script: &ScriptCheck<'_>,
            stages: &[StageSyntax],
        ) -> Option<Vec<StageDirective>> {
            self.seen_stages
                .lock()
                .unwrap()
                .push(stages.iter().map(|s| s.program.clone()).collect());
            None
        }
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn policies_see_original_and_treatment_and_pipelines_see_rewritten() {
        const NOTES: &str = "# Notes\n\n\n\nfirst\n";

        let dir = tempfile::tempdir().unwrap();
        let start = dir.path().canonicalize().unwrap();
        std::fs::write(start.join("notes.md"), NOTES).unwrap();

        let policy_seen = Arc::new(Mutex::new(Vec::new()));
        let rewrite_calls = Arc::new(AtomicUsize::new(0));
        let treatment_seen = Arc::new(Mutex::new(Vec::new()));
        let planner_stages = Arc::new(Mutex::new(Vec::new()));

        let heel = crate::test_heel::binary().await.expect("a heel for tests");
        let sandbox = Sandbox::new(
            Backend::Host { heel },
            executor_core::tokio::TokioGlobal,
            &start,
        );
        let shell = Shell::builder_on_host(sandbox, HostLaunch::new("/bin/bash"))
            .interpreters([Interpreter::Shell, Interpreter::Python])
            .limits(test_budgets::limits())
            .output(test_budgets::output_policy())
            .wording(test_wording::TestWording)
            .tools(test_wording::job_tools())
            .policy(RecordingPolicy {
                seen: policy_seen.clone(),
            })
            .rewrite_script(MultiRewriter {
                calls: rewrite_calls.clone(),
            })
            .treat_output(CatIsVerbatim {
                seen: treatment_seen.clone(),
            })
            .plan_pipelines(RecordingPlanner {
                seen_stages: planner_stages.clone(),
            })
            .build()
            .await
            .expect("shell builds");

        // 1. Policy refuses the original script before the rewriter is called.
        let refused = shell
            .tool()
            .call(shell_args("forbidden", false, false), ToolContext::new())
            .await
            .expect_err("policy must refuse original script");
        assert!(
            refused.to_string().contains("blocked by policy"),
            "{refused}"
        );
        assert_eq!(rewrite_calls.load(Ordering::SeqCst), 0);

        // 2. Policy sees original, rewriter transforms, TreatOutput sees re-parsed rewritten script.
        let notes = shell
            .tool()
            .call(shell_args("read_notes", false, false), ToolContext::new())
            .await
            .expect("rewritten read_notes succeeds");
        let notes_cmd = notes.command().expect("command result");
        assert_eq!(
            notes_cmd.stdout,
            CommandStream::Inline {
                text: NOTES.to_owned(),
            },
            "TreatOutput must see rewritten `cat notes.md` and keep blank runs verbatim"
        );
        assert_eq!(
            notes_cmd.notice.as_deref(),
            Some("Rewrote read_notes to cat notes.md.")
        );
        assert_eq!(
            *treatment_seen.lock().unwrap(),
            vec![("cat notes.md".to_owned(), true)]
        );

        // 3. PlanPipelines sees the rewritten pipeline stages.
        let piped = shell
            .tool()
            .call(shell_args("run_pipe", false, false), ToolContext::new())
            .await
            .expect("rewritten pipeline succeeds");
        let piped_cmd = piped.command().expect("command result");
        assert_eq!(
            piped_cmd.stdout,
            CommandStream::Inline {
                text: "alpha\n".to_owned(),
            }
        );
        assert_eq!(piped_cmd.notice, None);
        assert_eq!(
            *planner_stages.lock().unwrap(),
            vec![vec![Some("printf".to_owned()), Some("cat".to_owned())]]
        );

        // 4. Policy saw the original scripts in every call.
        assert_eq!(
            *policy_seen.lock().unwrap(),
            vec![
                "forbidden".to_owned(),
                "read_notes".to_owned(),
                "run_pipe".to_owned(),
            ]
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn non_shell_interpreter_skips_rewriter() {
        let dir = tempfile::tempdir().unwrap();
        let start = dir.path().canonicalize().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let shell = build_shell(
            &start,
            EchoRewriter {
                calls: calls.clone(),
            },
        )
        .await;

        let python_args: ShellArgs = serde_json::from_value(serde_json::json!({
            "description": "python run",
            "script": "print('py')",
            "interpreter": "python",
            "raw": false,
            "timeout": 30,
        }))
        .expect("valid python ShellArgs");
        let _ = shell.tool().call(python_args, ToolContext::new()).await;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "non-shell interpreter must never invoke RewriteScript"
        );
    }
}
