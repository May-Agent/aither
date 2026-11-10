//! Cancelling a tool call from outside it.
//!
//! Dropping a call's future already stops the work that future drives. A
//! tool whose work outlives the future it returned, such as a process it
//! spawned or a job it moved to the background, needs to hear about the
//! cancellation instead: whoever dispatches the call keeps a [`Canceller`]
//! and hands the matching [`CancelSignal`] to the call through
//! [`ToolContext::with_cancellation`](super::ToolContext::with_cancellation).
//! The pair follows the `AbortController`/`AbortSignal` split of the web
//! platform, which the Claude Agent SDK passes to its tools and to
//! `canUseTool`.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};

use event_listener::Event;

#[derive(Debug, Default)]
struct Shared {
    cancelled: AtomicBool,
    event: Event,
}

/// The side of a cancellation that fires it.
///
/// Dropping it does not cancel: a call whose dispatcher goes away without
/// cancelling is torn down by dropping its future, not through its signal.
#[derive(Debug, Default)]
pub struct Canceller {
    shared: Arc<Shared>,
}

impl Canceller {
    /// A canceller that has not fired.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A signal that fires when this canceller does. Every signal of one
    /// canceller fires together.
    #[must_use]
    pub fn signal(&self) -> CancelSignal {
        CancelSignal {
            shared: Some(Arc::clone(&self.shared)),
        }
    }

    /// Fires every signal of this canceller. Firing again does nothing.
    pub fn cancel(&self) {
        if !self.shared.cancelled.swap(true, Ordering::AcqRel) {
            self.shared.event.notify(usize::MAX);
        }
    }

    /// Whether [`cancel`](Self::cancel) has been called.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.shared.cancelled.load(Ordering::Acquire)
    }
}

/// The side of a cancellation a tool observes.
///
/// Cheap to clone, so a tool can hand it to every part of its work that
/// must stop when the call is cancelled.
#[derive(Debug, Clone, Default)]
pub struct CancelSignal {
    /// `None` for a signal no canceller can fire.
    shared: Option<Arc<Shared>>,
}

impl CancelSignal {
    /// A signal that never fires, for a dispatcher that does not cancel its
    /// calls from outside.
    #[must_use]
    pub const fn never() -> Self {
        Self { shared: None }
    }

    /// Whether the call has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.shared
            .as_ref()
            .is_some_and(|shared| shared.cancelled.load(Ordering::Acquire))
    }

    /// Resolves once the call is cancelled, at once when it already is; a
    /// signal that [never](Self::never) fires never resolves.
    pub async fn cancelled(&self) {
        let Some(shared) = &self.shared else {
            return core::future::pending().await;
        };
        loop {
            if shared.cancelled.load(Ordering::Acquire) {
                return;
            }
            let listener = shared.event.listen();
            // A cancel between the first check and `listen` notified nobody.
            if shared.cancelled.load(Ordering::Acquire) {
                return;
            }
            listener.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signals_fire_with_their_canceller() {
        let canceller = Canceller::new();
        let early = canceller.signal();
        assert!(!early.is_cancelled());
        canceller.cancel();
        canceller.cancel();
        assert!(canceller.is_cancelled());
        assert!(early.is_cancelled());
        assert!(canceller.signal().is_cancelled());
        assert!(!CancelSignal::never().is_cancelled());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancelled_resolves_for_a_waiting_signal() {
        let canceller = Canceller::new();
        let signal = canceller.signal();
        let waiter = tokio::spawn(async move { signal.cancelled().await });
        tokio::task::yield_now().await;
        canceller.cancel();
        waiter.await.expect("waiter finished");
        // Resolves at once once fired.
        canceller.signal().cancelled().await;
    }
}
