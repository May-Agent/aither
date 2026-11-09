//! The single context-management policy.
//!
//! Every context decision the agent makes — when to fork a compaction, when
//! to stop and wait for one, when an idle gap is long enough that rebuilding
//! the window is cheaper than keeping it, and how much of a handoff may be
//! spent on restored detail — is a threshold on this one struct. There is no
//! second knob, no per-level strategy and no preservation config: a single
//! policy keeps the behaviour predictable and the prompt cache analysable.
//!
//! Compaction points are token counts. The room left below the window is
//! the handoff reserve: what the compaction request's instruction and the
//! handoff the model writes need — a size that does not grow with the
//! window, so it is a fixed reserve, not a fraction. A ceiling still caps
//! the pause point: on a million-token model even the full window stops
//! being useful well before it is full.

use std::time::Duration;

use num_traits::ToPrimitive as _;

/// The share of the auto-compact point a fork may start from.
const FORK_HEADROOM: f64 = 0.9;

/// Thresholds that govern context compaction and pausing.
///
/// The window is the calibrated one of the [`ContextBudget`]: it subtracts
/// what the response and the tool surface need from the model's raw context
/// length. From it the policy derives three points, each in tokens:
///
/// - the *pause point*, `min(window - handoff_reserve, compact_ceiling)`,
///   where the loop waits for the compaction instead of sending more;
/// - the *start point*, [`start_distance`](Self::start_distance) before it,
///   where a background compaction is forked;
/// - the *warn point*, [`warn_distance`](Self::warn_distance) before it,
///   where the model is told to save its work and compact itself.
///
/// The distances are fixed counts, so a window too small for them (a start
/// distance past the point itself) forks from the first request; a host with
/// such a window sets distances that fit it.
///
/// # Example
///
/// ```
/// use aither_agent::ContextPolicy;
///
/// let policy = ContextPolicy::default();
/// assert_eq!(policy.pause_point(1_000_000), 400_000);
/// assert_eq!(policy.pause_point(200_000), 180_000);
/// assert!(policy.start_point(200_000) < policy.pause_point(200_000));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextPolicy {
    /// Tokens kept free below the window: the room the compaction
    /// request's instruction and the handoff the model writes need.
    ///
    /// A fixed count rather than a fraction — the handoff is about the same
    /// size whatever the window — so a large window is not robbed of a fifth
    /// of it and a small one is not under-reserved. The pause point is
    /// `min(window - handoff_reserve, compact_ceiling)`, capped by
    /// [`compact_ceiling`](Self::compact_ceiling), and every compaction
    /// threshold sits at a fixed distance below it.
    pub handoff_reserve: usize,

    /// Most tokens the conversation may hold before it is compacted,
    /// whatever the window: a long context stops being useful well before a
    /// large window is full.
    pub compact_ceiling: usize,

    /// Tokens before the pause point at which a background compaction is
    /// forked.
    ///
    /// The fork happens at a request boundary and runs concurrently with the
    /// next model turns, so this distance buys the turns needed to finish the
    /// handoff without stalling the run. It summarises everything before the
    /// fork; the turn keeps running.
    pub start_distance: usize,

    /// Tokens before the pause point at which the model is told once
    /// to save its work and call `compact` itself.
    ///
    /// Larger than [`start_distance`](Self::start_distance), so the model has
    /// the turns in between to reach a clean point before the harness forks a
    /// compaction of its own. Only an agent with the `compact` tool raises
    /// the alert.
    pub warn_distance: usize,

    /// Fraction of the window a restored-context block may occupy.
    ///
    /// After compaction the agent re-reads the files and documents the
    /// handoff names. This caps that restoration so a handoff cannot
    /// reintroduce the pressure it was supposed to relieve.
    pub restore_budget: f32,

    /// How long before the cache lifetime ends an idle session is compacted.
    ///
    /// The compaction then reads the conversation from the warm cache, and
    /// the session resumes on a much smaller prefix instead of paying to write
    /// the whole context again. Counted back from the end of the
    /// conversation's [`CacheLifetime`](super::CacheLifetime): six minutes
    /// compacts at 54 minutes under a one-hour lifetime.
    pub idle_compact_before_expiry: Duration,

    /// Tokens below which a context is not worth compacting on request.
    ///
    /// Re-reading a short context cold costs little, while the handoff
    /// replaces the verbatim conversation and loses its detail, so idle
    /// compaction and the model's own `compact` call leave a context under
    /// this floor alone. Automatic compaction near the window's limit is not
    /// governed by it.
    pub compaction_floor: usize,

    /// Tokens a tool output may hold before failure recovery moves it to a
    /// file, leaving the path and a short head in its place.
    pub spill_threshold: usize,

    /// Tokens of a spilled tool output that stay in place as its head.
    pub spill_head: usize,

    /// How far below the warn point an endpoint count is worth its round trip.
    ///
    /// Models that count through a provider endpoint (`TokenCounting::
    /// Endpoint`) spend a network call on every measurement, so the count is
    /// taken only once the estimate is this close to
    /// [`warn_point`](Self::warn_point). Local counting is cheap enough to
    /// run on every assembly and ignores this.
    pub count_distance: usize,

    /// Tokens of the user's own messages that survive a compaction verbatim.
    ///
    /// The user's words are the authoritative record of what was asked,
    /// decided and ruled out, and a retelling drifts from them. The newest
    /// messages are kept up to this budget; the handoff covers the rest.
    /// `ask_user` exchanges are kept whole whatever their size and do not
    /// count against it.
    pub user_message_budget: usize,
}

impl Default for ContextPolicy {
    fn default() -> Self {
        Self {
            handoff_reserve: 20_000,
            compact_ceiling: 400_000,
            start_distance: 50_000,
            warn_distance: 80_000,
            restore_budget: 0.15,
            idle_compact_before_expiry: Duration::from_secs(6 * 60),
            compaction_floor: 20_000,
            count_distance: 30_000,
            spill_threshold: 4_000,
            spill_head: 200,
            user_message_budget: 20_000,
        }
    }
}

impl ContextPolicy {
    /// Whether a context holding `tokens` is under the
    /// [`compaction_floor`](Self::compaction_floor).
    #[must_use]
    pub const fn below_floor(&self, tokens: usize) -> bool {
        tokens < self.compaction_floor
    }

    /// The token count at which a background compaction is forked.
    #[must_use]
    pub fn start_point(&self, window: u32) -> usize {
        self.pause_point(window).saturating_sub(self.start_distance)
    }

    /// The token count at which the model is told to save its work and call
    /// `compact`.
    #[must_use]
    pub fn warn_point(&self, window: u32) -> usize {
        self.pause_point(window).saturating_sub(self.warn_distance)
    }

    /// Whether the window holds enough to warn the model of the coming
    /// compaction. A window of unknown size never warns.
    #[must_use]
    pub fn should_warn(&self, tokens: usize, window: u32) -> bool {
        window > 0 && tokens >= self.warn_point(window)
    }

    /// Whether `tokens` is close enough to the warn point that an endpoint
    /// token count is worth its round trip — within
    /// [`count_distance`](Self::count_distance) of it. A window of unknown
    /// size is never counted.
    #[must_use]
    pub fn should_count_endpoint(&self, tokens: usize, window: u32) -> bool {
        window > 0 && tokens >= self.warn_point(window).saturating_sub(self.count_distance)
    }

    /// The token count at which the loop waits for compaction:
    /// `min(window - handoff_reserve, compact_ceiling)`.
    #[must_use]
    pub fn pause_point(&self, window: u32) -> usize {
        usize::try_from(window)
            .unwrap_or(usize::MAX)
            .saturating_sub(self.handoff_reserve)
            .min(self.compact_ceiling)
    }

    /// Whether a window of `window` tokens leaves any room below the pause
    /// point. One that does not would pause before its first request.
    #[must_use]
    pub fn admits(&self, window: u32) -> bool {
        self.pause_point(window) > 0
    }

    /// Whether the window holds enough to fork a background compaction.
    ///
    /// A budget with no window yet never reports it: thresholds must not fire
    /// on a window the agent has not learned the size of.
    #[must_use]
    pub fn should_compact(&self, tokens: usize, window: u32) -> bool {
        window > 0 && tokens >= self.start_point(window)
    }

    /// Whether the window is down to the handoff reserve, so the loop must
    /// wait for a compaction.
    #[must_use]
    pub fn should_pause(&self, tokens: usize, window: u32) -> bool {
        window > 0 && tokens >= self.pause_point(window)
    }

    /// Whether a fork may start from a context holding `tokens`: it is
    /// refused within a tenth of the pause point, where the main agent and
    /// the fork would each go on to compact the same context.
    #[must_use]
    pub fn allows_fork(&self, tokens: usize, window: u32) -> bool {
        let limit = to_f64(self.pause_point(window)) * FORK_HEADROOM;
        to_f64(tokens) < limit
    }
}

/// Widens a token count for ratio arithmetic, saturating rather than wrapping.
fn to_f64(tokens: usize) -> f64 {
    tokens.to_f64().unwrap_or(f64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_endpoint_count_is_gated_on_the_warn_point() {
        let policy = ContextPolicy {
            count_distance: 30_000,
            ..ContextPolicy::default()
        };
        // 1M window → pause point 400k (ceiling) → warn point 320k;
        // endpoint counting starts at 290k estimated.
        assert!(!policy.should_count_endpoint(100_000, 1_000_000));
        assert!(policy.should_count_endpoint(290_000, 1_000_000));
        assert!(policy.should_count_endpoint(320_000, 1_000_000));
        // An unknown window is never counted.
        assert!(!policy.should_count_endpoint(1_000_000, 0));
    }

    #[test]
    fn the_pause_point_is_the_lesser_of_window_minus_reserve_and_the_ceiling() {
        let policy = ContextPolicy::default();
        assert_eq!(policy.pause_point(1_000_000), 400_000);
        assert_eq!(policy.pause_point(200_000), 180_000);
        assert_eq!(policy.pause_point(500_000), 400_000);
        assert_eq!(policy.pause_point(0), 0);
    }

    #[test]
    fn the_reserve_and_the_ceiling_are_configurable() {
        let policy = ContextPolicy {
            handoff_reserve: 50_000,
            compact_ceiling: 100_000,
            ..ContextPolicy::default()
        };
        assert_eq!(policy.pause_point(1_000_000), 100_000);
        assert_eq!(policy.pause_point(100_000), 50_000);
    }

    #[test]
    fn default_points_are_ordered() {
        let policy = ContextPolicy::default();
        for window in [100_000, 200_000, 1_000_000] {
            assert!(policy.warn_point(window) < policy.start_point(window));
            assert!(policy.start_point(window) < policy.pause_point(window));
            assert!(policy.pause_point(window) < window as usize);
        }
    }

    #[test]
    fn the_start_and_warn_points_sit_their_distances_before_the_pause_point() {
        let policy = ContextPolicy::default();
        assert_eq!(policy.start_point(1_000_000), 350_000);
        assert_eq!(policy.warn_point(1_000_000), 320_000);
        assert_eq!(policy.start_point(200_000), 130_000);
        assert_eq!(policy.warn_point(200_000), 100_000);
    }

    #[test]
    fn thresholds_are_inclusive() {
        let policy = ContextPolicy::default();
        assert!(policy.should_compact(130_000, 200_000));
        assert!(!policy.should_compact(129_999, 200_000));
        assert!(policy.should_pause(180_000, 200_000));
        assert!(!policy.should_pause(179_999, 200_000));
    }

    #[test]
    fn a_window_of_unknown_size_never_crosses_a_threshold() {
        let policy = ContextPolicy::default();
        assert!(!policy.should_compact(1_000_000, 0));
        assert!(!policy.should_pause(1_000_000, 0));
    }

    #[test]
    fn the_floor_defaults_to_twenty_thousand_tokens_and_is_exclusive_below() {
        let policy = ContextPolicy::default();
        assert!(policy.below_floor(19_999));
        assert!(!policy.below_floor(20_000));
    }

    #[test]
    fn a_fork_is_refused_within_a_tenth_of_the_compaction_point() {
        let policy = ContextPolicy::default();
        assert!(policy.allows_fork(161_999, 200_000));
        assert!(!policy.allows_fork(162_000, 200_000));
        assert!(!policy.allows_fork(175_000, 200_000));
    }

    /// A window no larger than the handoff reserve leaves nothing below
    /// the pause point, so it is refused the way a window without that room
    /// has always been.
    #[test]
    fn a_window_no_larger_than_the_reserve_is_not_admitted() {
        let policy = ContextPolicy::default();
        let smallest = u32::try_from(policy.handoff_reserve).expect("fits u32");
        assert!(!policy.admits(smallest));
        assert!(policy.admits(smallest + 1));
    }
}
