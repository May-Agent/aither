//! Client-indexed breakpoint management reconciling DAP's replace-all semantics.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::DapError;
use crate::protocol::{
    Breakpoint, BreakpointEventBody, FunctionBreakpoint, SetBreakpointsArguments,
    SetFunctionBreakpointsArguments, Source, SourceBreakpoint,
};

/// Kind and location of a managed breakpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum BreakpointTarget {
    /// Source file and requested/resolved line location.
    Source {
        /// Source file path.
        file: String,
        /// Requested 1-based line number.
        requested_line: u32,
        /// Adapter-resolved 1-based line number, when verified or adjusted.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resolved_line: Option<u32>,
        /// Optional 1-based column number.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        column: Option<u32>,
    },
    /// Function-name breakpoint.
    Function {
        /// Function name.
        name: String,
        /// Optional resolved source file path reported by the adapter.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resolved_file: Option<String>,
        /// Optional resolved 1-based line number reported by the adapter.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        resolved_line: Option<u32>,
    },
}

/// A client-tracked breakpoint with monotonic ID and adapter verification state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManagedBreakpoint {
    /// Monotonic client-assigned breakpoint ID (used by `unbreak <id>`).
    pub id: u32,
    /// Adapter-assigned breakpoint ID, when provided by the adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_id: Option<i64>,
    /// Target file/line or function name.
    pub target: BreakpointTarget,
    /// Optional conditional expression.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    /// Optional hit count condition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hit_condition: Option<String>,
    /// Optional logpoint message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_message: Option<String>,
    /// Whether the adapter verified and bound the breakpoint.
    pub verified: bool,
    /// Optional verification diagnostic message from the adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Specification for adding a source breakpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceBreakpointSpec {
    /// Source file path.
    pub file: String,
    /// Requested 1-based line number.
    pub line: u32,
    /// Optional 1-based column number.
    pub column: Option<u32>,
    /// Optional conditional expression.
    pub condition: Option<String>,
    /// Optional hit count condition.
    pub hit_condition: Option<String>,
    /// Optional logpoint message.
    pub log_message: Option<String>,
}

/// Specification for adding a function breakpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionBreakpointSpec {
    /// Function name.
    pub name: String,
    /// Optional conditional expression.
    pub condition: Option<String>,
    /// Optional hit count condition.
    pub hit_condition: Option<String>,
}

/// Which DAP request must be sent after mutating [`BreakpointStore`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BreakpointSync {
    /// Re-send the complete source breakpoint list for one file.
    Source {
        /// Canonical key in the store.
        key: PathBuf,
        /// Full `setBreakpoints` request arguments.
        args: SetBreakpointsArguments,
    },
    /// Re-send the complete function breakpoint list.
    Function {
        /// Full `setFunctionBreakpoints` request arguments.
        args: SetFunctionBreakpointsArguments,
    },
}

/// Tracks per-file source breakpoints and function breakpoints under monotonic client IDs.
///
/// DAP's `setBreakpoints` replaces all breakpoints for a given source file on every
/// call. [`BreakpointStore`] maintains the full per-file sets so callers can add
/// (`break`) and remove (`unbreak <id>`) breakpoints incrementally while always
/// sending the complete list for that file to the adapter.
#[derive(Debug, Default)]
pub struct BreakpointStore {
    next_id: u32,
    by_file: BTreeMap<PathBuf, Vec<ManagedBreakpoint>>,
    functions: Vec<ManagedBreakpoint>,
}

impl BreakpointStore {
    /// Creates an empty breakpoint store with IDs starting at `1`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next_id: 1,
            by_file: BTreeMap::new(),
            functions: Vec::new(),
        }
    }

    /// Returns total count of tracked source and function breakpoints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_file.values().map(Vec::len).sum::<usize>() + self.functions.len()
    }

    /// Returns `true` if no breakpoints are currently registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns all managed breakpoints sorted by monotonic client ID.
    #[must_use]
    pub fn list(&self) -> Vec<ManagedBreakpoint> {
        let mut items: Vec<ManagedBreakpoint> = self
            .by_file
            .values()
            .flat_map(|list| list.iter().cloned())
            .chain(self.functions.iter().cloned())
            .collect();
        items.sort_by_key(|bp| bp.id);
        items
    }

    /// Looks up a single breakpoint by client ID.
    #[must_use]
    pub fn get(&self, id: u32) -> Option<&ManagedBreakpoint> {
        self.by_file
            .values()
            .flat_map(|list| list.iter())
            .chain(self.functions.iter())
            .find(|bp| bp.id == id)
    }

    /// Inserts a source breakpoint and builds the full `setBreakpoints` request
    /// payload for its file.
    pub fn add_source(&mut self, spec: SourceBreakpointSpec) -> (u32, BreakpointSync) {
        let id = self.allocate_id();
        let key = normalize_file_key(&spec.file);
        let entry = ManagedBreakpoint {
            id,
            adapter_id: None,
            target: BreakpointTarget::Source {
                file: spec.file,
                requested_line: spec.line,
                resolved_line: None,
                column: spec.column,
            },
            condition: spec.condition,
            hit_condition: spec.hit_condition,
            log_message: spec.log_message,
            verified: false,
            message: None,
        };
        self.by_file.entry(key.clone()).or_default().push(entry);
        let sync = self.source_sync(&key);
        (id, sync)
    }

    /// Inserts a function breakpoint and builds the full `setFunctionBreakpoints`
    /// request payload.
    pub fn add_function(&mut self, spec: FunctionBreakpointSpec) -> (u32, BreakpointSync) {
        let id = self.allocate_id();
        let entry = ManagedBreakpoint {
            id,
            adapter_id: None,
            target: BreakpointTarget::Function {
                name: spec.name,
                resolved_file: None,
                resolved_line: None,
            },
            condition: spec.condition,
            hit_condition: spec.hit_condition,
            log_message: None,
            verified: false,
            message: None,
        };
        self.functions.push(entry);
        let sync = self.function_sync();
        (id, sync)
    }

    /// Removes the breakpoint with client ID `id` and returns the removed entry
    /// along with the sync request needed to update the adapter.
    ///
    /// # Errors
    ///
    /// Returns [`DapError::BreakpointNotFound`] if `id` is not registered.
    pub fn remove(&mut self, id: u32) -> Result<(ManagedBreakpoint, BreakpointSync), DapError> {
        let found = self.by_file.iter().find_map(|(key, list)| {
            list.iter()
                .position(|bp| bp.id == id)
                .map(|pos| (key.clone(), pos))
        });
        if let Some((key, pos)) = found
            && let Some(list) = self.by_file.get_mut(&key)
        {
            let removed = list.remove(pos);
            if list.is_empty() {
                self.by_file.remove(&key);
            }
            let file_str = match &removed.target {
                BreakpointTarget::Source { file, .. } => file.clone(),
                BreakpointTarget::Function { .. } => key.display().to_string(),
            };
            let sync = self.source_sync_with_fallback(&key, &file_str);
            return Ok((removed, sync));
        }

        if let Some(pos) = self.functions.iter().position(|bp| bp.id == id) {
            let removed = self.functions.remove(pos);
            let sync = self.function_sync();
            return Ok((removed, sync));
        }

        Err(DapError::BreakpointNotFound(id))
    }

    /// Re-inserts a previously removed breakpoint (for rolling back failed removals).
    pub fn restore(&mut self, bp: ManagedBreakpoint) {
        match &bp.target {
            BreakpointTarget::Source { file, .. } => {
                let key = normalize_file_key(file);
                let list = self.by_file.entry(key).or_default();
                list.push(bp);
                list.sort_by_key(|entry| entry.id);
            }
            BreakpointTarget::Function { .. } => {
                self.functions.push(bp);
                self.functions.sort_by_key(|entry| entry.id);
            }
        }
    }

    /// Removes a newly added breakpoint by client ID without building a sync
    /// (for rolling back failed additions).
    pub fn rollback_add(&mut self, id: u32) {
        let mut empty_key = None;
        for (key, list) in &mut self.by_file {
            if let Some(pos) = list.iter().position(|bp| bp.id == id) {
                list.remove(pos);
                if list.is_empty() {
                    empty_key = Some(key.clone());
                }
                break;
            }
        }
        if let Some(key) = empty_key {
            self.by_file.remove(&key);
        }
        if let Some(pos) = self.functions.iter().position(|bp| bp.id == id) {
            self.functions.remove(pos);
        }
    }

    /// Applies the adapter's `setBreakpoints` response for `key`.
    pub fn apply_source_response(&mut self, key: &Path, responses: &[Breakpoint]) {
        let Some(entries) = self.by_file.get_mut(key) else {
            return;
        };
        for (entry, reported) in entries.iter_mut().zip(responses.iter()) {
            apply_adapter_breakpoint(entry, reported);
        }
    }

    /// Applies the adapter's `setFunctionBreakpoints` response.
    pub fn apply_function_response(&mut self, responses: &[Breakpoint]) {
        for (entry, reported) in self.functions.iter_mut().zip(responses.iter()) {
            apply_adapter_breakpoint(entry, reported);
        }
    }

    /// Reconciles an asynchronous DAP `breakpoint` event matched by `adapter_id`.
    pub fn apply_event(&mut self, event: &BreakpointEventBody) {
        let Some(adapter_id) = event.breakpoint.id else {
            return;
        };
        if event.reason == "removed" {
            self.by_file.retain(|_, list| {
                list.retain(|bp| bp.adapter_id != Some(adapter_id));
                !list.is_empty()
            });
            self.functions
                .retain(|bp| bp.adapter_id != Some(adapter_id));
            return;
        }
        for entry in self
            .by_file
            .values_mut()
            .flat_map(|list| list.iter_mut())
            .chain(self.functions.iter_mut())
        {
            if entry.adapter_id == Some(adapter_id) {
                apply_adapter_breakpoint(entry, &event.breakpoint);
            }
        }
    }

    /// Resolves adapter breakpoint IDs from a `stopped` event into client-assigned IDs.
    #[must_use]
    pub fn resolve_client_ids(&self, adapter_ids: &[i64]) -> Vec<u32> {
        let mut resolved = Vec::new();
        for &adapter_id in adapter_ids {
            if let Some(bp) = self
                .by_file
                .values()
                .flat_map(|list| list.iter())
                .chain(self.functions.iter())
                .find(|bp| bp.adapter_id == Some(adapter_id))
            {
                resolved.push(bp.id);
            }
        }
        resolved
    }

    const fn allocate_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    fn source_sync(&self, key: &Path) -> BreakpointSync {
        let fallback = key.display().to_string();
        self.source_sync_with_fallback(key, &fallback)
    }

    fn source_sync_with_fallback(&self, key: &Path, fallback_path: &str) -> BreakpointSync {
        let entries = self.by_file.get(key).map_or(&[][..], Vec::as_slice);
        let source_path = entries
            .first()
            .and_then(|bp| match &bp.target {
                BreakpointTarget::Source { file, .. } => Some(file.as_str()),
                BreakpointTarget::Function { .. } => None,
            })
            .unwrap_or(fallback_path)
            .to_owned();
        let source_name = Path::new(&source_path)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned());
        let breakpoints = entries
            .iter()
            .filter_map(|bp| match &bp.target {
                BreakpointTarget::Source {
                    requested_line,
                    column,
                    ..
                } => Some(SourceBreakpoint {
                    line: *requested_line,
                    column: *column,
                    condition: bp.condition.clone(),
                    hit_condition: bp.hit_condition.clone(),
                    log_message: bp.log_message.clone(),
                }),
                BreakpointTarget::Function { .. } => None,
            })
            .collect();
        BreakpointSync::Source {
            key: key.to_path_buf(),
            args: SetBreakpointsArguments {
                source: Source {
                    name: source_name,
                    path: Some(source_path),
                    source_reference: None,
                },
                breakpoints,
            },
        }
    }

    fn function_sync(&self) -> BreakpointSync {
        let breakpoints = self
            .functions
            .iter()
            .filter_map(|bp| match &bp.target {
                BreakpointTarget::Function { name, .. } => Some(FunctionBreakpoint {
                    name: name.clone(),
                    condition: bp.condition.clone(),
                    hit_condition: bp.hit_condition.clone(),
                }),
                BreakpointTarget::Source { .. } => None,
            })
            .collect();
        BreakpointSync::Function {
            args: SetFunctionBreakpointsArguments { breakpoints },
        }
    }
}

fn apply_adapter_breakpoint(entry: &mut ManagedBreakpoint, reported: &Breakpoint) {
    if reported.id.is_some() {
        entry.adapter_id = reported.id;
    }
    entry.verified = reported.verified;
    entry.message.clone_from(&reported.message);
    match &mut entry.target {
        BreakpointTarget::Source {
            resolved_line,
            column,
            ..
        } => {
            if reported.line.is_some() {
                *resolved_line = reported.line;
            }
            if reported.column.is_some() {
                *column = reported.column;
            }
        }
        BreakpointTarget::Function {
            resolved_file,
            resolved_line,
            ..
        } => {
            if let Some(path) = reported.source.as_ref().and_then(|s| s.path.as_ref()) {
                *resolved_file = Some(path.clone());
            }
            if reported.line.is_some() {
                *resolved_line = reported.line;
            }
        }
    }
}

fn normalize_file_key(file: &str) -> PathBuf {
    let path = Path::new(file);
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_add_and_remove_sends_full_file_list() {
        let mut store = BreakpointStore::new();
        let (id1, sync1) = store.add_source(SourceBreakpointSpec {
            file: "/tmp/example.rs".to_owned(),
            line: 10,
            column: None,
            condition: None,
            hit_condition: None,
            log_message: None,
        });
        assert_eq!(id1, 1);
        let BreakpointSync::Source { key, args } = sync1 else {
            panic!("expected source sync");
        };
        assert_eq!(args.breakpoints.len(), 1);
        assert_eq!(args.breakpoints[0].line, 10);
        store.apply_source_response(
            &key,
            &[Breakpoint {
                id: Some(100),
                verified: true,
                line: Some(12),
                ..Breakpoint::default()
            }],
        );

        let (id2, sync2) = store.add_source(SourceBreakpointSpec {
            file: "/tmp/example.rs".to_owned(),
            line: 25,
            column: None,
            condition: Some("x > 0".to_owned()),
            hit_condition: None,
            log_message: None,
        });
        assert_eq!(id2, 2);
        let BreakpointSync::Source {
            key: key2,
            args: args2,
        } = sync2
        else {
            panic!("expected source sync");
        };
        assert_eq!(key, key2);
        assert_eq!(args2.breakpoints.len(), 2);
        assert_eq!(args2.breakpoints[0].line, 10);
        assert_eq!(args2.breakpoints[1].line, 25);

        let (removed, sync3) = store.remove(id1).unwrap();
        assert_eq!(removed.id, 1);
        let BreakpointSync::Source { args: args3, .. } = sync3 else {
            panic!("expected source sync");
        };
        assert_eq!(args3.breakpoints.len(), 1);
        assert_eq!(args3.breakpoints[0].line, 25);

        let (_, sync4) = store.remove(id2).unwrap();
        let BreakpointSync::Source { args: args4, .. } = sync4 else {
            panic!("expected source sync");
        };
        assert_eq!(args4.breakpoints, Vec::new());
        assert!(store.is_empty());
    }

    #[test]
    fn remove_unknown_id_returns_not_found() {
        let mut store = BreakpointStore::new();
        assert!(matches!(
            store.remove(99),
            Err(DapError::BreakpointNotFound(99))
        ));
    }
}
