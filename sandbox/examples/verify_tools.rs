#![doc = "Verify prepared baseline commands through the real terminal sandbox."]

use std::path::PathBuf;
use std::time::Duration;

use aither_core::llm::{Tool, ToolContext};
use aither_sandbox::{
    AnyShellWording, CommandBlocked, CommandKilled, FoldedLines, JobInputSent, JobKillAsked,
    JobRead, JobToolSettings, JsonAsTsv, MediaBounds, MonitorRefusal, MonitorStarted,
    OmittedLines, OutputLimits, OutputPolicy, OutputSaved, RepeatedLines, ReservedForMain,
    ShellArgs, ShellLimits, ShellWording, StoredBinaryPart, Shell, ToolDistribution,
    ToolIdentity, ToolRegistryBuilder, ToolTarget,
};

/// The wording of this example's shell: aither ships none, and the example's
/// only reader is the acceptance script that checks its output.
struct ExampleWording;

impl ShellWording for ExampleWording {
    fn job_read(&self, read: &JobRead) -> String {
        match read {
            JobRead::Unchanged { task_id } => format!("task {task_id} has no new output"),
            JobRead::Delta(delta) => String::from_utf8_lossy(&delta.bytes).into_owned(),
        }
    }

    fn job_input(&self, sent: &JobInputSent<'_>) -> String {
        format!("input written to {}", sent.task_id)
    }

    fn job_kill(&self, kill: &JobKillAsked<'_>) -> String {
        format!("kill of {}: {}", kill.task_id, kill.killed)
    }

    fn command_blocked(&self, blocked: &CommandBlocked) -> String {
        format!("blocked: {}", blocked.reason)
    }

    fn monitor_refusal(&self, refusal: &MonitorRefusal) -> String {
        format!("monitor refused: {refusal:?}")
    }

    fn monitor_started(&self, started: &MonitorStarted<'_>) -> String {
        format!("monitoring {}", started.task_id)
    }

    fn reserved_for_main(&self, reserved: &ReservedForMain<'_>) -> String {
        format!("{} is reserved for the main agent", reserved.command)
    }

    fn command_killed(&self, killed: &CommandKilled<'_>) -> String {
        format!("killed: {killed:?}")
    }

    fn output_saved(&self, saved: &OutputSaved<'_>) -> String {
        format!(
            "{}\n[{} lines saved to {}]",
            saved.preview, saved.line_count, saved.path
        )
    }

    fn stored_binary_part(&self, part: &StoredBinaryPart<'_>) -> String {
        format!("[{} stored at {}]", part.mime, part.path)
    }

    fn json_as_tsv(&self, json: &JsonAsTsv<'_>) -> String {
        format!("{}\n[JSON saved to {}]", json.tsv, json.path)
    }

    fn folded_code(&self, folded: &str) -> String {
        folded.to_owned()
    }

    fn blank_lines(&self, count: usize) -> String {
        format!("[{count} blank lines]")
    }

    fn folded_lines(&self, folded: &FoldedLines<'_>) -> String {
        format!(
            "{}  [{} lines folded]",
            folded.indent, folded.count
        )
    }

    fn omitted_lines(&self, omitted: &OmittedLines) -> String {
        format!(
            "[lines {}-{} of {} omitted]",
            omitted.range.first, omitted.range.last, omitted.total
        )
    }

    fn repeated_lines(&self, repeated: &RepeatedLines) -> String {
        format!("[repeated {} times]", repeated.count)
    }
}

fn identity(name: &'static str, description: &'static str) -> ToolIdentity {
    ToolIdentity {
        name: name.into(),
        description: description.into(),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let (Some(manifest), Some(cache), Some(workspace), None) =
        (args.next(), args.next(), args.next(), args.next())
    else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: verify_tools <manifest.json> <cache-directory> <workspace-directory>",
        )
        .into());
    };
    let pinned: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest)?)?;
    let mut distribution = ToolDistribution::from_path(&PathBuf::from(manifest)).await?;
    // CI hands its job token down so release downloads are authenticated.
    if let Ok(token) = std::env::var("GITHUB_TOKEN") {
        distribution = distribution.with_github_token(token);
    }
    let limits = ShellLimits::new(
        Duration::from_secs(10),
        Duration::from_secs(120 * 60),
        Duration::from_secs(40 * 60),
        1024 * 1024 * 1024,
        Duration::from_secs(120),
    );
    let output = OutputPolicy::new(OutputLimits {
        inline_bytes: 16_000,
        preview_bytes: 16_000,
        default_lines: 200,
        ceiling_lines: 800,
        read_bytes: 32 * 1024,
        media: MediaBounds {
            low: 512,
            medium: 1024,
            high: 2048,
        },
        job_preview_bytes: 50,
    });
    let terminal = Shell::new_exact(
        PathBuf::from(workspace),
        executor_core::tokio::TokioGlobal,
        limits,
        output,
        AnyShellWording::new(ExampleWording),
        JobToolSettings {
            read: identity("job_read", "Read a job's output."),
            input: identity("job_input", "Write to a job's stdin."),
            kill: identity("job_kill", "Kill a job."),
        },
    )
    .await?
    .initialize_tools(&distribution, &PathBuf::from(cache))
    .await?;
    let registry = ToolRegistryBuilder::new().build(terminal.output_store().clone());
    let terminal = terminal.with_registry(std::sync::Arc::new(registry));
    let arguments: ShellArgs = serde_json::from_value(serde_json::json!({
        "description": "Verify prepared search, JSON and edit commands",
        "script": include_str!("../templates/verify_baseline_tools.sh"),
        "raw": true,
        "timeout": 120
    }))?;
    let result = terminal.call(arguments, ToolContext::new()).await?;
    let output = result.render_for_model();
    assert!(output.contains("baseline acceptance passed"), "{output}");
    // `uv --version` and `python3 --version` report the manifest's pins.
    let target = ToolTarget::native()?;
    if target.provides_python() {
        let pin_of = |entries: &str, key: &str, matches: &dyn Fn(&serde_json::Value) -> bool| {
            pinned[entries]
                .as_array()
                .and_then(|entries| entries.iter().find(|entry| matches(entry)))
                .and_then(|entry| entry[key].as_str())
                .map(str::to_owned)
        };
        let on_target = |entry: &serde_json::Value| {
            entry["target"]["os"] == target.os() && entry["target"]["arch"] == target.arch()
        };
        let uv = pin_of("artifacts", "version", &|entry| {
            entry["tool"] == "uv" && on_target(entry)
        })
        .expect("the manifest pins uv for the host");
        let python =
            pin_of("python", "version", &on_target).expect("the manifest pins python for the host");
        assert!(
            output.contains(&format!("uv {uv} ")),
            "uv {uv} missing: {output}"
        );
        assert!(
            output.contains(&format!("Python {python}")),
            "Python {python} missing: {output}"
        );
    }
    assert!(!output.contains("Exit code:"), "{output}");
    assert!(!output.contains("Started in background"), "{output}");
    Ok(())
}
