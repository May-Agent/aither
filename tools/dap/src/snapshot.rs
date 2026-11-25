//! Stopped-state snapshot tables indexing transient DAP frame and variable handles.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::DapError;
use crate::protocol::{StackFrame, StoppedEventBody, Thread};

/// Number of context lines shown above and below the stopped source line.
const SOURCE_CONTEXT_RADIUS: u32 = 3;

/// Maximum source file size (2 MiB) read for inline stop-context snippets.
const MAX_SNIPPET_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// A single source line rendered in a stop snippet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLine {
    /// 1-based line number.
    pub line: u32,
    /// Whether this line is the active stopped line.
    pub current: bool,
    /// Line text without trailing newline.
    pub text: String,
}

/// Source snippet around a stopped location.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSnippet {
    /// Source file path.
    pub path: String,
    /// Lines around the stopped line.
    pub lines: Vec<SourceLine>,
}

/// Summary of one thread in the debug target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadSummary {
    /// Adapter thread identifier.
    pub id: i64,
    /// Thread display name.
    pub name: String,
    /// Whether this thread is the current stopped thread.
    pub active: bool,
}

/// Indexed stack frame summary exposed to callers with small integer index `0, 1, 2...`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexedFrame {
    /// Small 0-based frame index (`--frame <index>`).
    pub index: usize,
    /// Function or frame label name.
    pub name: String,
    /// Source file path, if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// 1-based line number (0 if unavailable).
    pub line: u32,
    /// 1-based column number (0 if unavailable).
    pub column: u32,
}

/// Summary of a stopped debug target rendered to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopSummary {
    /// Stop reason reported by the adapter (`breakpoint`, `step`, `exception`, `entry`, `pause`).
    pub reason: String,
    /// Optional stop description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Optional detail text (such as an exception message).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Stopped thread ID, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<i64>,
    /// Client-assigned breakpoint IDs matched at this stop.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hit_breakpoint_ids: Vec<u32>,
    /// Top frame summary, if stack frames were retrieved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_frame: Option<IndexedFrame>,
    /// Source-line context around `top_frame`, when readable on disk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_snippet: Option<SourceSnippet>,
}

/// Stack snapshot returned by `stack`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackSnapshot {
    /// Thread ID whose stack was captured.
    pub thread_id: i64,
    /// Indexed stack frames (0 = innermost).
    pub frames: Vec<IndexedFrame>,
    /// Total frame count reported by the adapter, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_frames: Option<usize>,
}

/// One variable entry rendered to the model with optional small integer expand handle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariableEntry {
    /// Variable name.
    pub name: String,
    /// Formatted value string.
    pub value: String,
    /// Optional type name.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub value_type: Option<String>,
    /// Small 1-based handle for `vars --expand <handle>` when the value has children.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expand: Option<usize>,
}

/// Variables grouped under a named scope (such as `Locals` or `Arguments`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeVariables {
    /// Scope display name.
    pub name: String,
    /// Variables in this scope.
    pub variables: Vec<VariableEntry>,
}

/// Result of `vars` (either scopes for a frame or one-level expansion of a handle).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub enum VariablesView {
    /// Top-level scopes for a stack frame.
    Frame {
        /// 0-based frame index inspected.
        frame: usize,
        /// Frame function name.
        frame_name: String,
        /// Scopes and their direct variables.
        scopes: Vec<ScopeVariables>,
    },
    /// One-level child expansion of a variable handle.
    Expanded {
        /// 1-based variable handle that was expanded.
        handle: usize,
        /// Display label of the expanded parent variable.
        label: String,
        /// Direct child variables.
        variables: Vec<VariableEntry>,
    },
}

/// Result of `eval`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalResult {
    /// Evaluated expression.
    pub expression: String,
    /// Formatted value string.
    pub value: String,
    /// Optional result type.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub value_type: Option<String>,
    /// Small 1-based handle for `vars --expand <handle>` when expandable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expand: Option<usize>,
}

#[derive(Debug, Clone)]
struct FrameRecord {
    dap_id: i64,
    summary: IndexedFrame,
}

#[derive(Debug, Clone)]
struct VariableHandleRecord {
    dap_ref: i64,
    label: String,
}

/// Cached stopped-state tables for the current stop.
///
/// Created when execution stops and automatically dropped whenever execution
/// resumes so stale DAP `frameId` and `variablesReference` values can never be reused.
#[derive(Debug)]
pub struct StoppedSnapshot {
    /// Summary of the stop event and top frame context.
    pub summary: StopSummary,
    /// Active threads observed at this stop.
    pub threads: Vec<ThreadSummary>,
    /// Thread ID whose stack frames are currently indexed.
    pub stack_thread_id: Option<i64>,
    frames: Vec<FrameRecord>,
    total_frames: Option<usize>,
    /// 1-based handle -> DAP `variablesReference` + display label.
    handles: Vec<VariableHandleRecord>,
    /// Reverse index from DAP `variablesReference` -> 1-based handle so repeated
    /// inspections return stable handles within the same stop.
    ref_to_handle: HashMap<i64, usize>,
}

impl StoppedSnapshot {
    /// Builds a stopped snapshot from a `stopped` event, thread list, stack frames,
    /// client-resolved hit breakpoint IDs, and optional source snippet.
    #[must_use]
    pub fn build(
        event: &StoppedEventBody,
        threads: &[Thread],
        stack_thread_id: Option<i64>,
        raw_frames: &[StackFrame],
        total_frames: Option<usize>,
        hit_breakpoint_ids: Vec<u32>,
        source_snippet: Option<SourceSnippet>,
    ) -> Self {
        let active_thread = event.thread_id.or(stack_thread_id);
        let thread_summaries: Vec<ThreadSummary> = threads
            .iter()
            .map(|thread| ThreadSummary {
                id: thread.id,
                name: thread.name.clone(),
                active: active_thread == Some(thread.id),
            })
            .collect();

        let frames: Vec<FrameRecord> = raw_frames
            .iter()
            .enumerate()
            .map(|(index, frame)| FrameRecord {
                dap_id: frame.id,
                summary: IndexedFrame {
                    index,
                    name: frame.name.clone(),
                    file: frame
                        .source
                        .as_ref()
                        .and_then(|src| src.path.clone().or_else(|| src.name.clone())),
                    line: frame.line,
                    column: frame.column,
                },
            })
            .collect();

        let top_frame = frames.first().map(|record| record.summary.clone());

        let summary = StopSummary {
            reason: event.reason.clone(),
            description: event.description.clone(),
            text: event.text.clone(),
            thread_id: active_thread,
            hit_breakpoint_ids,
            top_frame,
            source_snippet,
        };

        Self {
            summary,
            threads: thread_summaries,
            stack_thread_id: active_thread,
            frames,
            total_frames,
            handles: Vec::new(),
            ref_to_handle: HashMap::new(),
        }
    }

    /// Replaces the indexed frame table (for example when `stack --thread-id`
    /// inspects a different thread while stopped) and clears handles bound to the
    /// previous frame table.
    pub fn replace_frames(
        &mut self,
        thread_id: i64,
        raw_frames: &[StackFrame],
        total_frames: Option<usize>,
    ) {
        self.stack_thread_id = Some(thread_id);
        self.total_frames = total_frames;
        self.frames = raw_frames
            .iter()
            .enumerate()
            .map(|(index, frame)| FrameRecord {
                dap_id: frame.id,
                summary: IndexedFrame {
                    index,
                    name: frame.name.clone(),
                    file: frame
                        .source
                        .as_ref()
                        .and_then(|src| src.path.clone().or_else(|| src.name.clone())),
                    line: frame.line,
                    column: frame.column,
                },
            })
            .collect();
    }

    /// Returns the indexed stack snapshot, optionally truncated to `limit` frames.
    #[must_use]
    pub fn stack_snapshot(&self, limit: Option<usize>) -> StackSnapshot {
        let mut frames: Vec<IndexedFrame> = self.frames.iter().map(|r| r.summary.clone()).collect();
        if let Some(limit) = limit {
            frames.truncate(limit);
        }
        StackSnapshot {
            thread_id: self.stack_thread_id.unwrap_or_default(),
            frames,
            total_frames: self.total_frames,
        }
    }

    /// Resolves a 0-based frame index into `(dap_frame_id, frame_name)`.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::InvalidFrameIndex`] when `index` is out of range.
    pub fn resolve_frame(&self, index: usize) -> Result<(i64, &str), DapError> {
        let record = self.frames.get(index).ok_or(DapError::InvalidFrameIndex {
            index,
            count: self.frames.len(),
        })?;
        Ok((record.dap_id, &record.summary.name))
    }

    /// Registers a non-zero DAP `variablesReference` under a small 1-based handle
    /// index, reusing an existing handle if the reference was already indexed.
    pub fn register_variable_ref(
        &mut self,
        dap_ref: i64,
        label: impl Into<String>,
    ) -> Option<usize> {
        if dap_ref <= 0 {
            return None;
        }
        if let Some(&existing) = self.ref_to_handle.get(&dap_ref) {
            return Some(existing);
        }
        let handle = self.handles.len() + 1;
        self.handles.push(VariableHandleRecord {
            dap_ref,
            label: label.into(),
        });
        self.ref_to_handle.insert(dap_ref, handle);
        Some(handle)
    }

    /// Resolves a 1-based variable handle into `(dap_variables_reference, label)`.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::InvalidVariableHandle`] when `handle` is 0 or unknown.
    pub fn resolve_variable_handle(&self, handle: usize) -> Result<(i64, &str), DapError> {
        let idx = handle
            .checked_sub(1)
            .ok_or(DapError::InvalidVariableHandle(handle))?;
        let record = self
            .handles
            .get(idx)
            .ok_or(DapError::InvalidVariableHandle(handle))?;
        Ok((record.dap_ref, &record.label))
    }
}

/// Reads a bounded source-line context snippet around `line` from `path`, returning
/// `None` if the file does not exist, is too large, or cannot be read as UTF-8.
pub async fn read_source_snippet(path: &str, line: u32) -> Option<SourceSnippet> {
    if line == 0 {
        return None;
    }
    let file_path = Path::new(path);
    let metadata = async_fs::metadata(file_path).await.ok()?;
    if !metadata.is_file() || metadata.len() > MAX_SNIPPET_FILE_BYTES {
        return None;
    }
    let content = async_fs::read_to_string(file_path).await.ok()?;
    let start_line = line.saturating_sub(SOURCE_CONTEXT_RADIUS).max(1);
    let end_line = line.saturating_add(SOURCE_CONTEXT_RADIUS);

    let mut lines = Vec::new();
    for (idx, text) in content.lines().enumerate() {
        let Ok(line_no) = u32::try_from(idx + 1) else {
            break;
        };
        if line_no < start_line {
            continue;
        }
        if line_no > end_line {
            break;
        }
        lines.push(SourceLine {
            line: line_no,
            current: line_no == line,
            text: text.to_owned(),
        });
    }

    if lines.is_empty() {
        None
    } else {
        Some(SourceSnippet {
            path: path.to_owned(),
            lines,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Source;

    #[test]
    fn indexes_frames_and_variable_handles() {
        let event = StoppedEventBody {
            reason: "breakpoint".to_owned(),
            thread_id: Some(7),
            ..StoppedEventBody::default()
        };
        let threads = vec![Thread {
            id: 7,
            name: "main".to_owned(),
        }];
        let frames = vec![
            StackFrame {
                id: 9001,
                name: "inner".to_owned(),
                source: Some(Source {
                    path: Some("/src/lib.rs".to_owned()),
                    ..Source::default()
                }),
                line: 42,
                column: 5,
            },
            StackFrame {
                id: 9002,
                name: "outer".to_owned(),
                source: None,
                line: 10,
                column: 1,
            },
        ];

        let mut snap =
            StoppedSnapshot::build(&event, &threads, Some(7), &frames, Some(2), vec![1], None);
        assert_eq!(snap.resolve_frame(0).unwrap(), (9001, "inner"));
        assert_eq!(snap.resolve_frame(1).unwrap(), (9002, "outer"));
        assert!(matches!(
            snap.resolve_frame(2),
            Err(DapError::InvalidFrameIndex { index: 2, count: 2 })
        ));

        assert_eq!(snap.register_variable_ref(0, "scalar"), None);
        let h1 = snap.register_variable_ref(500, "state").unwrap();
        let h2 = snap.register_variable_ref(501, "items").unwrap();
        assert_eq!(h1, 1);
        assert_eq!(h2, 2);
        // Re-registering the same DAP variablesReference returns the same handle.
        assert_eq!(snap.register_variable_ref(500, "state"), Some(1));
        assert_eq!(snap.resolve_variable_handle(1).unwrap(), (500, "state"));
        assert!(matches!(
            snap.resolve_variable_handle(0),
            Err(DapError::InvalidVariableHandle(0))
        ));
    }

    #[tokio::test]
    async fn reads_source_context_around_stopped_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.rs");
        std::fs::write(
            &path,
            "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\n",
        )
        .unwrap();

        let snippet = read_source_snippet(path.to_str().unwrap(), 5)
            .await
            .unwrap();
        assert_eq!(snippet.lines.first().unwrap().line, 2);
        assert_eq!(snippet.lines.last().unwrap().line, 8);
        assert!(snippet.lines.iter().any(|l| l.line == 5 && l.current));
    }
}
