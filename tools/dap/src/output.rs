//! Bounded ring buffer for target `output` events (`stdout`, `stderr`, `console`).

use std::collections::VecDeque;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Stream category for target debug output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum OutputCategory {
    /// Include stdout, stderr, and console output.
    #[default]
    All,
    /// Target standard output.
    Stdout,
    /// Target standard error.
    Stderr,
    /// Adapter or runtime console output.
    Console,
}

impl OutputCategory {
    /// Parses a DAP `output` event category string; returns `None` for non-user streams
    /// such as `telemetry`.
    #[must_use]
    pub fn from_dap(category: Option<&str>) -> Option<Self> {
        match category.unwrap_or("console") {
            "stdout" => Some(Self::Stdout),
            "stderr" => Some(Self::Stderr),
            "telemetry" => None,
            _ => Some(Self::Console),
        }
    }

    /// Returns a short display label for templates.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Stdout => "stdout",
            Self::Stderr => "stderr",
            Self::Console => "console",
        }
    }

    const fn matches(self, entry: Self) -> bool {
        match self {
            Self::All => true,
            _ => matches!(
                (self, entry),
                (Self::Stdout, Self::Stdout)
                    | (Self::Stderr, Self::Stderr)
                    | (Self::Console, Self::Console)
            ),
        }
    }
}

/// One captured output chunk in the bounded ring buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputEntry {
    /// Stream category (`stdout`, `stderr`, or `console`).
    pub category: OutputCategory,
    /// Captured UTF-8 text.
    pub text: String,
}

/// Snapshot returned by reading the output ring buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputSnapshot {
    /// Category filter applied.
    pub category: OutputCategory,
    /// Matching output entries in chronological order.
    pub entries: Vec<OutputEntry>,
    /// Number of oldest entries evicted due to capacity limits.
    pub dropped_entries: usize,
    /// Number of oldest bytes evicted due to capacity limits.
    pub dropped_bytes: usize,
}

/// Capacity bounds for [`OutputBuffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputBufferConfig {
    /// Maximum number of retained entries (default: 512).
    pub max_entries: usize,
    /// Maximum total retained bytes across all entries (default: 64 KiB).
    pub max_bytes: usize,
}

impl Default for OutputBufferConfig {
    fn default() -> Self {
        Self {
            // Retain enough recent chunks for typical test/debug runs while keeping
            // prompt-injected output strictly bounded.
            max_entries: 512,
            max_bytes: 64 * 1024,
        }
    }
}

/// Bounded FIFO ring buffer capturing target output events.
#[derive(Debug)]
pub struct OutputBuffer {
    config: OutputBufferConfig,
    entries: VecDeque<OutputEntry>,
    total_bytes: usize,
    dropped_entries: usize,
    dropped_bytes: usize,
}

impl Default for OutputBuffer {
    fn default() -> Self {
        Self::new(OutputBufferConfig::default())
    }
}

impl OutputBuffer {
    /// Creates an output buffer with the given capacity bounds.
    #[must_use]
    pub const fn new(config: OutputBufferConfig) -> Self {
        Self {
            config,
            entries: VecDeque::new(),
            total_bytes: 0,
            dropped_entries: 0,
            dropped_bytes: 0,
        }
    }

    /// Appends an output chunk, evicting oldest entries when over entry or byte limits.
    pub fn push(&mut self, category: OutputCategory, mut text: String) {
        if text.is_empty() || self.config.max_entries == 0 || self.config.max_bytes == 0 {
            return;
        }

        if text.len() > self.config.max_bytes {
            let keep_from = truncate_tail_utf8(&text, self.config.max_bytes);
            self.dropped_bytes = self.dropped_bytes.saturating_add(keep_from);
            text.drain(..keep_from);
        }

        self.total_bytes = self.total_bytes.saturating_add(text.len());
        self.entries.push_back(OutputEntry { category, text });

        while self.entries.len() > self.config.max_entries
            || self.total_bytes > self.config.max_bytes
        {
            if let Some(evicted) = self.entries.pop_front() {
                self.total_bytes = self.total_bytes.saturating_sub(evicted.text.len());
                self.dropped_entries = self.dropped_entries.saturating_add(1);
                self.dropped_bytes = self.dropped_bytes.saturating_add(evicted.text.len());
            } else {
                break;
            }
        }
    }

    /// Reads entries matching `category`, optionally taking only the last `tail`
    /// entries and optionally clearing the buffer afterward.
    pub fn snapshot(
        &mut self,
        category: OutputCategory,
        tail: Option<usize>,
        clear: bool,
    ) -> OutputSnapshot {
        let mut entries: Vec<OutputEntry> = self
            .entries
            .iter()
            .filter(|entry| category.matches(entry.category))
            .cloned()
            .collect();

        if let Some(limit) = tail
            && entries.len() > limit
        {
            let start = entries.len() - limit;
            entries.drain(..start);
        }

        let snapshot = OutputSnapshot {
            category,
            entries,
            dropped_entries: self.dropped_entries,
            dropped_bytes: self.dropped_bytes,
        };

        if clear {
            self.clear();
        }

        snapshot
    }

    /// Clears all retained output and resets dropped counters.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_bytes = 0;
        self.dropped_entries = 0;
        self.dropped_bytes = 0;
    }
}

/// Finds the smallest valid UTF-8 character boundary in `s` such that
/// `s[boundary..].len() <= max_bytes`.
const fn truncate_tail_utf8(s: &str, max_bytes: usize) -> usize {
    let target = s.len().saturating_sub(max_bytes);
    let mut idx = target;
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evicts_oldest_entries_when_byte_limit_exceeded() {
        let mut buf = OutputBuffer::new(OutputBufferConfig {
            max_entries: 10,
            max_bytes: 12,
        });
        buf.push(OutputCategory::Stdout, "hello\n".to_owned());
        buf.push(OutputCategory::Stderr, "world\n".to_owned());
        buf.push(OutputCategory::Stdout, "tail!\n".to_owned());

        let snap = buf.snapshot(OutputCategory::All, None, false);
        assert_eq!(snap.dropped_entries, 1);
        assert_eq!(snap.dropped_bytes, 6);
        assert_eq!(snap.entries.len(), 2);
        assert_eq!(snap.entries[0].category, OutputCategory::Stderr);
        assert_eq!(snap.entries[0].text, "world\n");
        assert_eq!(snap.entries[1].text, "tail!\n");
    }

    #[test]
    fn preserves_utf8_boundaries_when_single_entry_exceeds_limit() {
        let mut buf = OutputBuffer::new(OutputBufferConfig {
            max_entries: 4,
            max_bytes: 5,
        });
        buf.push(OutputCategory::Stdout, "你好世界".to_owned());
        let snap = buf.snapshot(OutputCategory::Stdout, None, true);
        assert_eq!(snap.entries.len(), 1);
        assert_eq!(snap.entries[0].text, "界");
        assert!(snap.dropped_bytes > 0);
        assert_eq!(
            buf.snapshot(OutputCategory::All, None, false).entries,
            Vec::new()
        );
    }
}
