//! What a shell inlines, cuts and reads back: the embedder's output budgets.

use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::output::MediaResolution;

/// The pixel bounding boxes of the [`MediaResolution`] levels that resize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaBounds {
    /// The bounding box of [`MediaResolution::Low`], in pixels.
    pub low: u32,
    /// The bounding box of [`MediaResolution::Medium`], in pixels.
    pub medium: u32,
    /// The bounding box of [`MediaResolution::High`], in pixels.
    pub high: u32,
}

impl MediaBounds {
    /// The bounding box a level resizes to; `None` keeps the source media.
    #[must_use]
    pub const fn bound(self, resolution: MediaResolution) -> Option<u32> {
        match resolution {
            MediaResolution::Auto | MediaResolution::Native => None,
            MediaResolution::Low => Some(self.low),
            MediaResolution::Medium => Some(self.medium),
            MediaResolution::High => Some(self.high),
        }
    }
}

/// Every budget of the output a shell hands the model.
///
/// There is no `Default`: the values are the embedder's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputLimits {
    /// The most bytes of a text output delivered inline; a larger output is
    /// saved to a file.
    pub inline_bytes: usize,
    /// The most bytes of the preview a saved text output keeps in the result.
    pub preview_bytes: usize,
    /// The inline line budget of a call that does not state `max_lines`.
    pub default_lines: u32,
    /// The most lines a call may ask to inline.
    pub ceiling_lines: u32,
    /// The bytes a read of a job returns when the call states no `max_bytes`.
    pub read_bytes: usize,
    /// The pixel bounds of the resizing media levels.
    pub media: MediaBounds,
    /// The most bytes of a job's script a job listing shows.
    pub job_preview_bytes: usize,
}

/// The live handle to a shell's [`OutputLimits`].
///
/// Every clone shares the values, so an update reaches the shells and tools
/// already running.
#[derive(Debug, Clone)]
pub struct OutputPolicy(Arc<ArcSwap<OutputLimits>>);

impl OutputPolicy {
    /// A policy holding `limits`.
    #[must_use]
    pub fn new(limits: OutputLimits) -> Self {
        Self(Arc::new(ArcSwap::from_pointee(limits)))
    }

    /// The limits in force now.
    #[must_use]
    pub fn get(&self) -> OutputLimits {
        **self.0.load()
    }

    /// Replaces the limits for every holder of this handle.
    pub fn set(&self, limits: OutputLimits) {
        self.0.store(Arc::new(limits));
    }

    /// The inline line budget for a call asking for `requested` lines: the
    /// default when it asks for none, never more than the ceiling.
    #[must_use]
    pub fn lines(&self, requested: Option<u32>) -> usize {
        let limits = self.get();
        requested
            .unwrap_or(limits.default_lines)
            .min(limits.ceiling_lines) as usize
    }
}

#[cfg(any(test, feature = "test-budgets"))]
pub mod fixture {
    use super::{MediaBounds, OutputLimits, OutputPolicy};

    /// The limits tests use, whose numbers stand for nothing.
    pub const LIMITS: OutputLimits = OutputLimits {
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
    };

    /// A policy holding [`LIMITS`].
    pub fn policy() -> OutputPolicy {
        OutputPolicy::new(LIMITS)
    }
}
