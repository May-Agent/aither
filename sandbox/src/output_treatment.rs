//! Output treatments: the embedder's judgement of what a script's output is.
//!
//! The output pipeline cleans what a command prints for the model: it renders
//! the bytes as a terminal shows them, then collapses runs of blank lines and
//! turns a JSON array of records into TSV. That fits command output, but a
//! script that reads a file prints the file, and cleaning it hands the model
//! text the file does not hold. Which scripts produce content of that kind is
//! the embedder's judgement, so the [`ShellBuilder`]
//! takes a [`TreatOutput`] that sees each script before it runs and chooses
//! its [`OutputTreatment`]. Without one every script's output is treated as
//! [`Command`](OutputTreatment::Command) output.

use std::fmt;
use std::sync::Arc;

use crate::ScriptCheck;

/// What the shell does to a foreground script's text output before the
/// model reads it.
///
/// Either way the output is rendered as a terminal shows it: escape
/// sequences are consumed and `\r` resolves to the last frame of each line.
/// When output exceeds the inline budget, [`Command`](Self::Command) and
/// [`VerbatimFoldable`](Self::VerbatimFoldable) replace head+tail truncation
/// with structural code folding for recognisable source code, while
/// [`Verbatim`](Self::Verbatim) never folds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutputTreatment {
    /// Command output: after rendering, each run of blank lines collapses
    /// into one marker line, a JSON array of records becomes TSV whose
    /// original is written to disk and named, and cut source code is folded.
    #[default]
    Command,
    /// Whole-file content printed verbatim (such as an un-ranged `cat` of a
    /// large file): blank lines and JSON are kept verbatim, while cut source
    /// code is folded instead of head+tail truncated.
    VerbatimFoldable,
    /// Targeted content the script printed as it is (such as a ranged file
    /// read or search): nothing after rendering changes the text, and cut
    /// output uses head+tail preview rather than code folding.
    Verbatim,
}

impl OutputTreatment {
    /// Whether cut source code under this treatment is folded instead of
    /// head+tail truncated.
    #[must_use]
    pub const fn allows_folding(self) -> bool {
        matches!(self, Self::Command | Self::VerbatimFoldable)
    }
}

/// Chooses the [`OutputTreatment`] of every script of a shell, before it
/// runs.
///
/// Every script of the `terminal` tool, the main agent's and every
/// subagent's, and every [`Shell::run`](crate::Shell::run), is put to it
/// with the same view its [`ScriptPolicy`](crate::ScriptPolicy)s see, the
/// shell parsed once for all of them.
pub trait TreatOutput: Send + Sync + 'static {
    /// The treatment of `script`'s output.
    fn treatment(&self, script: &ScriptCheck<'_>) -> OutputTreatment;
}

/// A type-erased [`TreatOutput`], shared by a shell and the subagent shells
/// derived from it.
#[derive(Clone)]
pub struct AnyTreatOutput(Arc<dyn TreatOutput>);

impl AnyTreatOutput {
    /// Erases `treat`.
    pub fn new(treat: impl TreatOutput) -> Self {
        Self(Arc::new(treat))
    }

    /// The treatment of `script`'s output.
    pub fn treatment(&self, script: &ScriptCheck<'_>) -> OutputTreatment {
        self.0.treatment(script)
    }
}

impl fmt::Debug for AnyTreatOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AnyTreatOutput")
    }
}

#[cfg(test)]
mod tests {
    use super::OutputTreatment;

    #[test]
    fn allows_folding_matches_treatment_policy() {
        assert!(OutputTreatment::Command.allows_folding());
        assert!(OutputTreatment::VerbatimFoldable.allows_folding());
        assert!(!OutputTreatment::Verbatim.allows_folding());
    }
}
