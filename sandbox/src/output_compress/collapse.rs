//! Blank-run collapsing stage.

use crate::wording::ShellWording;

/// A line counts as blank only when it has no characters at all.
///
/// Lines made of spaces are left alone on purpose: in a unified diff every
/// context line carries a leading space, so a blank line of the patched file
/// reaches us as `" "`. Treating that as blank would drop context lines and
/// the patch would stop applying.
const fn is_blank(line: &str) -> bool {
    line.is_empty()
}

/// Replaces every run of two or more blank lines with one marker line that
/// says how many blank lines it stands for.
///
/// A lone blank line is always kept, so paragraphs, hunk separators and the
/// blank context lines of a patch survive. The marker is one line of the
/// returned text, and the returned text is what gets stored, so every line
/// number downstream (previews, omitted ranges) counts the marker as one
/// line and matches the stored file. Whether the text ends with a line
/// terminator is preserved: lines keep the newline they arrived with, and
/// none is added to a last line that had none.
#[must_use]
pub fn collapse_blank_runs(text: &str, wording: &dyn ShellWording) -> String {
    let mut out = String::with_capacity(text.len());
    let mut run = 0_usize;
    let flush = |out: &mut String, run: usize| match run {
        0 => {}
        1 => out.push('\n'),
        count => {
            out.push_str(&wording.blank_lines(count));
            out.push('\n');
        }
    };
    for line in text.split_inclusive('\n') {
        if is_blank(line.strip_suffix('\n').unwrap_or(line)) {
            run += 1;
            continue;
        }
        flush(&mut out, run);
        run = 0;
        out.push_str(line);
    }
    flush(&mut out, run);
    out
}

#[cfg(test)]
mod tests {
    use crate::test_wording::TestWording;

    fn collapse_blank_runs(text: &str) -> String {
        super::collapse_blank_runs(text, &TestWording)
    }

    #[test]
    fn collapses_a_run_to_one_marker_naming_its_count() {
        let collapsed = collapse_blank_runs("alpha\n\n\n\nbeta\n");
        let lines: Vec<&str> = collapsed.lines().collect();
        assert_eq!(lines.len(), 3, "{collapsed}");
        assert_eq!(lines[0], "alpha");
        assert!(
            lines[1].starts_with("[... 3 blank lines collapsed"),
            "{collapsed}"
        );
        assert!(lines[1].contains("not part of it"), "{collapsed}");
        assert_eq!(lines[2], "beta");
    }

    #[test]
    fn a_run_of_two_is_marked_too() {
        let collapsed = collapse_blank_runs("a\n\n\nb\n");
        assert!(
            collapsed.contains("[... 2 blank lines collapsed"),
            "{collapsed}"
        );
    }

    #[test]
    fn line_numbers_after_a_run_count_the_marker_as_one_line() {
        let collapsed = collapse_blank_runs("a\n\n\n\n\nb\nc\n");
        let number = collapsed
            .lines()
            .position(|line| line == "c")
            .expect("c present")
            + 1;
        assert_eq!(number, 4, "{collapsed}");
    }

    #[test]
    fn keeps_a_single_blank_line() {
        assert_eq!(collapse_blank_runs("alpha\n\nbeta\n"), "alpha\n\nbeta\n");
    }

    #[test]
    fn keeps_space_only_lines() {
        // A diff context line for a blank source line is a single space.
        assert_eq!(collapse_blank_runs(" \n \nx\n"), " \n \nx\n");
    }

    #[test]
    fn collapses_a_trailing_run() {
        let collapsed = collapse_blank_runs("alpha\n\n\n");
        assert!(
            collapsed.starts_with("alpha\n[... 2 blank lines collapsed")
                && collapsed.ends_with("...]\n"),
            "{collapsed}"
        );
    }

    #[test]
    fn empty_input_stays_empty() {
        assert_eq!(collapse_blank_runs(""), "");
    }

    #[test]
    fn keeps_a_missing_final_newline() {
        assert!(!collapse_blank_runs("alpha\n\n\nbeta").ends_with('\n'));
        assert_eq!(collapse_blank_runs("abc"), "abc");
    }
}
