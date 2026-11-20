//! Cleaning a context a compaction failed on.
//!
//! A compaction replays the whole window, so what makes it fail is usually
//! what makes the window big or noisy: system alerts and reminders that
//! accumulated, and tool outputs of tens of thousands of tokens. Recovery
//! removes the first and moves the second out of the window — to files the
//! agent can still read, leaving the path and a short head in place — so the
//! synchronous rerun summarises a context the summariser can cope with.

use std::path::Path;

use askama::Template;

use super::reassembly::truncate_on_char_boundary;
use super::window::Context;

/// What recovery took out of the window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    /// System alerts and reminders removed.
    pub alerts_removed: usize,
    /// Tool outputs moved to files.
    pub outputs_spilled: usize,
}

/// Default subdirectory name used by [`AgentBuilder::restore_dir`](crate::AgentBuilder::restore_dir).
pub const DEFAULT_SPILL_SUBDIR: &str = "outputs";

/// Maximum length of the sanitized tool-call ID segment in a spilled filename.
const MAX_ID_STEM_LEN: usize = 64;

#[derive(Template)]
#[template(path = "spilled_tool_output.txt", escape = "none")]
struct SpilledToolOutput<'a> {
    head: &'a str,
    bytes: usize,
    path: &'a str,
    absolute_path: &'a str,
}

/// How much of a context recovery may move out of the window.
#[derive(Debug, Clone, Copy)]
pub struct SpillLimits {
    /// Bytes a tool output may hold before it is moved.
    pub threshold: usize,
    /// Bytes of a moved output that stay in place.
    pub head: usize,
}

/// Removes the alerts and reminders numbered below `fork_point` (the fork
/// point of the compaction about to run) from `context`, and moves the tool
/// outputs over `limits.threshold` into `spill_dir`.
///
/// With no spill directory there is nowhere the agent could read a file from,
/// so outputs stay where they are and only the alerts go.
///
/// # Errors
///
/// Returns the I/O error of the first file that could not be written; the
/// output it held is left in the window.
pub async fn clean_for_recovery(
    context: &mut Context,
    fork_point: u64,
    spill_dir: Option<&Path>,
    first_root: Option<&Path>,
    limits: SpillLimits,
) -> std::io::Result<RecoveryReport> {
    let alerts_removed = context.prune_system_messages_below(fork_point);
    let mut outputs_spilled = 0;
    if let Some(dir) = spill_dir {
        for output in context.long_tool_outputs(limits.threshold) {
            async_fs::create_dir_all(dir).await?;
            let name = format!(
                "compaction-{}-{}.txt",
                output.seq,
                file_stem(&output.tool_call_id)
            );
            let absolute = dir.join(&name);
            async_fs::write(&absolute, output.content.as_bytes()).await?;
            let relative = display_spill_path(&absolute, dir, first_root, &name);
            let replacement = SpilledToolOutput {
                head: &truncate_on_char_boundary(&output.content, limits.head),
                bytes: output.content.len(),
                path: &relative,
                absolute_path: &absolute.to_string_lossy(),
            }
            .render()
            .expect("spilled tool output template must render");
            context.replace_tool_output(output.seq, replacement);
            outputs_spilled += 1;
        }
    }
    Ok(RecoveryReport {
        alerts_removed,
        outputs_spilled,
    })
}

/// Spills non-kept tool outputs at or above `since_seq` exceeding
/// `limits.threshold` into `spill_dir`, returning `(seq, replacement)` pairs
/// without mutating `context` so callers can commit atomically after awaits.
pub async fn prepare_post_fork_spills(
    context: &Context,
    since_seq: u64,
    spill_dir: Option<&Path>,
    first_root: Option<&Path>,
    limits: SpillLimits,
) -> std::io::Result<Vec<(u64, String)>> {
    let Some(dir) = spill_dir else {
        return Ok(Vec::new());
    };
    let outputs = context.long_tool_outputs_since(since_seq, limits.threshold);
    if outputs.is_empty() {
        return Ok(Vec::new());
    }
    async_fs::create_dir_all(dir).await?;
    let mut replacements = Vec::with_capacity(outputs.len());
    for output in outputs {
        let name = format!(
            "compaction-{}-{}.txt",
            output.seq,
            file_stem(&output.tool_call_id)
        );
        let absolute = dir.join(&name);
        async_fs::write(&absolute, output.content.as_bytes()).await?;
        let relative = display_spill_path(&absolute, dir, first_root, &name);
        let replacement = SpilledToolOutput {
            head: &truncate_on_char_boundary(&output.content, limits.head),
            bytes: output.content.len(),
            path: &relative,
            absolute_path: &absolute.to_string_lossy(),
        }
        .render()
        .expect("spilled tool output template must render");
        replacements.push((output.seq, replacement));
    }
    Ok(replacements)
}

fn display_spill_path(
    absolute: &Path,
    spill_dir: &Path,
    first_root: Option<&Path>,
    name: &str,
) -> String {
    if let Some(root) = first_root
        && let Ok(rel) = absolute.strip_prefix(root)
    {
        return rel.to_string_lossy().replace('\\', "/");
    }
    if let Some(dir_name) = spill_dir.file_name().and_then(|segment| segment.to_str()) {
        return format!("{dir_name}/{name}");
    }
    name.to_string()
}

/// A tool call id as a bounded file name stem: ids come from providers and may
/// hold characters or lengths a path cannot.
fn file_stem(id: &str) -> String {
    let stem: String = id
        .bytes()
        .take(MAX_ID_STEM_LEN)
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_' {
                char::from(byte)
            } else {
                '_'
            }
        })
        .collect();
    if stem.is_empty() {
        "call".to_string()
    } else {
        stem
    }
}

#[cfg(test)]
mod tests {
    use aither_core::llm::Message;

    use super::*;

    fn limits() -> SpillLimits {
        SpillLimits {
            threshold: 100,
            head: 10,
        }
    }

    #[tokio::test]
    async fn long_outputs_move_to_files_and_alerts_go() {
        let dir = tempfile::tempdir().expect("tempdir");
        let spill = dir.path().join(DEFAULT_SPILL_SUBDIR);
        let mut context = Context::new();
        context.push(Message::user("run it"));
        context.push(Message::system("<system-reminder>stale</system-reminder>"));
        let seq = context.push(Message::tool("call/1", "é".repeat(200)));
        context.push(Message::tool("call-2", "short"));
        context.mark_prefix_sent();

        let report = clean_for_recovery(
            &mut context,
            u64::MAX,
            Some(&spill),
            Some(dir.path()),
            limits(),
        )
        .await
        .expect("recovery");

        assert_eq!(
            report,
            RecoveryReport {
                alerts_removed: 1,
                outputs_spilled: 1
            }
        );
        let expected_name = format!("compaction-{seq}-call_1.txt");
        let spilled = spill.join(&expected_name);
        assert_eq!(
            std::fs::read_to_string(&spilled).expect("spill file"),
            "é".repeat(200)
        );
        let messages = context.conversation_messages();
        assert_eq!(messages.len(), 3);
        let stub = messages[1].content();
        assert!(stub.starts_with("ééééé\n"), "short head kept: {stub}");
        assert!(stub.contains(&format!("outputs/{expected_name}")));
        assert!(stub.contains(&*spilled.to_string_lossy()));
        assert_eq!(messages[2].content(), "short");
    }

    #[tokio::test]
    async fn colliding_or_overlong_ids_spill_to_distinct_bounded_filenames() {
        let spill = tempfile::tempdir().expect("spill dir");
        let mut context = Context::new();
        let long_id = format!("call/{}", "x".repeat(400));
        let first_seq = context.push(Message::tool(&long_id, "a".repeat(200)));
        let second_seq = context.push(Message::tool(
            format!("call:{}", "x".repeat(400)),
            "b".repeat(200),
        ));

        let report = clean_for_recovery(&mut context, u64::MAX, Some(spill.path()), None, limits())
            .await
            .expect("recovery");
        assert_eq!(report.outputs_spilled, 2);

        let stem = format!("call_{}", "x".repeat(MAX_ID_STEM_LEN - 5));
        let first_file = spill
            .path()
            .join(format!("compaction-{first_seq}-{stem}.txt"));
        let second_file = spill
            .path()
            .join(format!("compaction-{second_seq}-{stem}.txt"));
        assert_eq!(
            std::fs::read_to_string(&first_file).expect("first spill"),
            "a".repeat(200)
        );
        assert_eq!(
            std::fs::read_to_string(&second_file).expect("second spill"),
            "b".repeat(200)
        );
    }

    #[tokio::test]
    async fn only_alerts_below_the_fork_point_go() {
        let mut context = Context::new();
        context.push(Message::system("old alert"));
        context.mark_prefix_sent();
        let fork_point = context.fork_point();
        context.push(Message::system("new alert"));

        let report = clean_for_recovery(&mut context, fork_point, None, None, limits())
            .await
            .expect("recovery");

        assert_eq!(report.alerts_removed, 1);
        let kept: Vec<_> = context.recent().iter().map(|m| m.content()).collect();
        assert_eq!(kept, ["new alert"]);
    }

    #[tokio::test]
    async fn an_alert_no_request_carried_survives_recovery() {
        let mut context = Context::new();
        context.push(Message::system("delivered alert"));
        context.mark_prefix_sent();
        context.push(Message::system("undelivered alert"));

        let report = clean_for_recovery(&mut context, u64::MAX, None, None, limits())
            .await
            .expect("recovery");

        assert_eq!(report.alerts_removed, 1);
        let kept: Vec<_> = context.recent().iter().map(|m| m.content()).collect();
        assert_eq!(kept, ["undelivered alert"]);
    }

    #[tokio::test]
    async fn without_a_spill_directory_only_the_alerts_go() {
        let mut context = Context::new();
        context.push(Message::system("alert"));
        context.push(Message::tool("call-1", "x".repeat(500)));
        context.mark_prefix_sent();

        let report = clean_for_recovery(&mut context, u64::MAX, None, None, limits())
            .await
            .expect("recovery");

        assert_eq!(report.alerts_removed, 1);
        assert_eq!(report.outputs_spilled, 0);
        assert_eq!(context.conversation_messages()[0].content().len(), 500);
    }
}
