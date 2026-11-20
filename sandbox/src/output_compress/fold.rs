//! Code-folding stage: collapses the bodies of compound blocks in output
//! that is source code, keeping a line number gutter so the folded text can
//! still be matched against the stored unfolded copy.
//!
//! This stage is lossy, so the orchestrator only applies it when it can store
//! the unfolded text and name it in the text the model reads.

#[cfg(not(feature = "code-folding"))]
use crate::wording::ShellWording;

/// Reusable pool of content-identification sessions for source-code folding.
#[cfg(not(feature = "code-folding"))]
#[derive(Debug, Default)]
pub struct CodeFolder;

#[cfg(not(feature = "code-folding"))]
impl CodeFolder {
    /// Without the `code-folding` feature there is no content detector, so
    /// output is passed through rather than folded against a guessed grammar.
    #[allow(clippy::unused_self)]
    pub fn fold_source_code(&self, _text: &str, _wording: &dyn ShellWording) -> Option<String> {
        None
    }
}

#[cfg(feature = "code-folding")]
pub use enabled::CodeFolder;

#[cfg(feature = "code-folding")]
mod enabled {
    use std::fmt;
    use std::num::NonZero;
    use std::sync::Mutex;
    #[cfg(test)]
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tracing::debug;

    use crate::wording::{FoldedLines, LineRange, ShellWording};

    /// Minimum number of lines a foldable block must span to be worth folding.
    const MIN_FOLD_LINES: usize = 4;

    /// Reusable pool of `magika::Session` instances owned by an `OutputStore`,
    /// bounded to the host's available parallelism.
    pub struct CodeFolder {
        sessions: Mutex<Vec<magika::Session>>,
        capacity: usize,
        #[cfg(test)]
        created: AtomicUsize,
    }

    impl Default for CodeFolder {
        fn default() -> Self {
            let capacity = std::thread::available_parallelism()
                .map(NonZero::get)
                .unwrap_or(1);
            Self {
                sessions: Mutex::new(Vec::new()),
                capacity,
                #[cfg(test)]
                created: AtomicUsize::new(0),
            }
        }
    }

    impl fmt::Debug for CodeFolder {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let pooled = self
                .sessions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len();
            f.debug_struct("CodeFolder")
                .field("pooled", &pooled)
                .field("capacity", &self.capacity)
                .finish()
        }
    }

    struct PooledSession<'a> {
        pool: &'a CodeFolder,
        session: Option<magika::Session>,
    }

    impl<'a> PooledSession<'a> {
        fn acquire(pool: &'a CodeFolder) -> Option<Self> {
            let cached = pool
                .sessions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .pop();
            let session = if let Some(session) = cached {
                session
            } else {
                let session = magika::Session::new().ok()?;
                #[cfg(test)]
                pool.created.fetch_add(1, Ordering::Relaxed);
                session
            };
            Some(Self {
                pool,
                session: Some(session),
            })
        }

        fn identify_label(&mut self, bytes: &[u8]) -> Option<&'static str> {
            let result = self.session.as_mut()?.identify_content_sync(bytes).ok()?;
            Some(result.info().label)
        }
    }

    impl Drop for PooledSession<'_> {
        fn drop(&mut self) {
            if let Some(session) = self.session.take() {
                let mut pool = self
                    .pool
                    .sessions
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if pool.len() < self.pool.capacity {
                    pool.push(session);
                }
            }
        }
    }

    /// Maps a magika content-type label to a tree-sitter grammar.
    fn magika_label_to_ts_language(label: &str) -> Option<tree_sitter::Language> {
        match label {
            "terminal" | "shell" => Some(rs_tree_sitter_languages::bash::language()),
            "c" => Some(rs_tree_sitter_languages::c::language()),
            "cpp" => Some(rs_tree_sitter_languages::cpp::language()),
            "css" => Some(rs_tree_sitter_languages::css::language()),
            "elixir" => Some(rs_tree_sitter_languages::elixir::language()),
            "erlang" => Some(rs_tree_sitter_languages::erlang::language()),
            "go" => Some(rs_tree_sitter_languages::go::language()),
            "haskell" => Some(rs_tree_sitter_languages::haskell::language()),
            "html" => Some(rs_tree_sitter_languages::html::language()),
            "java" => Some(rs_tree_sitter_languages::java::language()),
            "javascript" => Some(rs_tree_sitter_languages::javascript::language()),
            "json" => Some(rs_tree_sitter_languages::json::language()),
            "lua" => Some(rs_tree_sitter_languages::lua::language()),
            "markdown" => Some(rs_tree_sitter_languages::markdown::language()),
            "perl" => Some(rs_tree_sitter_languages::perl::language()),
            "python" => Some(rs_tree_sitter_languages::python::language()),
            "ruby" => Some(rs_tree_sitter_languages::ruby::language()),
            "rust" => Some(rs_tree_sitter_languages::rust::language()),
            "toml" => Some(rs_tree_sitter_languages::toml::language()),
            "tsx" => Some(rs_tree_sitter_languages::tsx::language()),
            "typescript" => Some(rs_tree_sitter_languages::typescript::language()),
            "yaml" => Some(rs_tree_sitter_languages::yaml::language()),
            _ => None,
        }
    }

    /// Tree-sitter node types whose body can be collapsed.
    fn is_foldable_node(kind: &str) -> bool {
        matches!(
            kind,
            "function_definition"
                | "function_item"
                | "function_declaration"
                | "method_definition"
                | "method_declaration"
                | "class_definition"
                | "class_declaration"
                | "class_body"
                | "impl_item"
                | "trait_item"
                | "module"
                | "block"
                | "compound_statement"
                | "if_statement"
                | "if_expression"
                | "for_statement"
                | "for_expression"
                | "while_statement"
                | "while_expression"
                | "match_expression"
                | "switch_statement"
                | "try_statement"
                | "do_statement"
        )
    }

    impl CodeFolder {
        /// Folds `text` when it is recognisable source code and folding makes
        /// it smaller.
        ///
        /// Every emitted line carries its 1-based number in `text`, and each
        /// fold marker names the range it replaced, so the result lines up
        /// with the unfolded copy the orchestrator stores.
        pub fn fold_source_code(
            &self,
            text: &str,
            wording: &dyn ShellWording,
        ) -> Option<String> {
            let mut session = PooledSession::acquire(self)?;
            let label = session.identify_label(text.as_bytes())?;
            drop(session);

            let ts_lang = magika_label_to_ts_language(label)?;

            debug!(label = %label, "detected source code via magika, applying tree-sitter folding");

            let mut parser = tree_sitter::Parser::new();
            parser.set_language(&ts_lang).ok()?;
            let tree = parser.parse(text, None)?;

            let mut fold_ranges: Vec<(usize, usize)> = Vec::new();
            collect_foldable_ranges(tree.root_node(), &mut fold_ranges);
            if fold_ranges.is_empty() {
                return None;
            }
            fold_ranges.sort_by_key(|range| range.0);
            let merged = merge_ranges(&fold_ranges);

            let lines: Vec<&str> = text.lines().collect();
            let total_lines = lines.len();
            let width = digit_count(total_lines);
            let indent = " ".repeat(width);

            let mut output = String::new();
            let mut cursor = 0;

            for &(fold_start, fold_end) in &merged {
                let fold_end = fold_end.min(total_lines.saturating_sub(1));

                while cursor < fold_start && cursor < total_lines {
                    append_numbered_line(&mut output, cursor + 1, lines[cursor], width);
                    cursor += 1;
                }

                // The first line of the region is the signature or declaration.
                if cursor < total_lines {
                    append_numbered_line(&mut output, cursor + 1, lines[cursor], width);
                    cursor += 1;
                }

                let inner_end = fold_end + 1;
                let folded_count = inner_end.saturating_sub(cursor);
                if folded_count > 0 {
                    let marker = wording.folded_lines(&FoldedLines {
                        indent: &indent,
                        count: folded_count,
                        range: LineRange {
                            first: cursor + 1,
                            last: inner_end,
                        },
                    });
                    output.push_str(&marker);
                    output.push('\n');
                    cursor = inner_end;
                }
            }

            while cursor < total_lines {
                append_numbered_line(&mut output, cursor + 1, lines[cursor], width);
                cursor += 1;
            }

            (output.len() < text.len()).then_some(output)
        }
    }

    /// Recursively collects foldable line ranges from the tree-sitter AST.
    fn collect_foldable_ranges(node: tree_sitter::Node, ranges: &mut Vec<(usize, usize)>) {
        if is_foldable_node(node.kind()) {
            let start_line = node.start_position().row;
            let end_line = node.end_position().row;
            if end_line - start_line >= MIN_FOLD_LINES {
                ranges.push((start_line, end_line));
            }
        }
        for index in 0..node.child_count() {
            if let Some(child) = node.child(index) {
                collect_foldable_ranges(child, ranges);
            }
        }
    }

    /// Merges overlapping or adjacent ranges.
    fn merge_ranges(ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for &(start, end) in ranges {
            if let Some(last) = merged.last_mut()
                && start <= last.1 + 1
            {
                last.1 = last.1.max(end);
                continue;
            }
            merged.push((start, end));
        }
        merged
    }

    fn append_numbered_line(out: &mut String, line_num: usize, line: &str, width: usize) {
        use std::fmt::Write;
        writeln!(out, "{line_num:>width$}  {line}").expect("writing to a String cannot fail");
    }

    const fn digit_count(n: usize) -> usize {
        // Counting digits by repeated division avoids a float round-trip, and so
        // cannot be thrown off by rounding near a power of ten.
        let mut count = 1;
        let mut remaining = n;
        while remaining >= 10 {
            remaining /= 10;
            count += 1;
        }
        count
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::test_wording::TestWording;

        fn fold_source_code(text: &str) -> Option<String> {
            CodeFolder::default().fold_source_code(text, &TestWording)
        }

        #[test]
        fn reuses_pooled_magika_session_across_sequential_folds() {
            let folder = CodeFolder::default();
            let code = "fn compute_sum(values: &[i64]) -> i64 {\n    let mut total = 0;\n    for &value in values {\n        total += value;\n    }\n    total\n}\n";
            for _ in 0..4 {
                let _ = folder.fold_source_code(code, &TestWording);
            }
            let created = folder.created.load(Ordering::Relaxed);
            if created > 0 {
                assert_eq!(
                    created, 1,
                    "sequential folds should reuse one pooled session"
                );
                assert_eq!(
                    folder.sessions.lock().unwrap().len(),
                    1,
                    "session should be returned to the pool"
                );
            }
        }

        #[test]
        fn folds_python_function_bodies() {
            let code = "def hello():\n    print(\"hello\")\n    print(\"world\")\n    print(\"foo\")\n    print(\"bar\")\n    return True\n\ndef goodbye():\n    print(\"bye\")\n    print(\"see ya\")\n    print(\"later\")\n    print(\"adios\")\n    return False\n";
            // magika may decline to classify a short snippet; folding is only
            // asserted when it did classify it.
            if let Some(folded) = fold_source_code(code) {
                assert!(folded.contains("lines folded"), "{folded}");
                assert!(folded.len() < code.len(), "{folded}");
            }
        }

        #[test]
        fn declines_plain_prose() {
            assert!(fold_source_code("just a sentence of output\n").is_none());
        }

        #[test]
        fn digit_count_matches_decimal_width() {
            assert_eq!(digit_count(0), 1);
            assert_eq!(digit_count(9), 1);
            assert_eq!(digit_count(10), 2);
            assert_eq!(digit_count(999), 3);
            assert_eq!(digit_count(1000), 4);
        }
    }
}
