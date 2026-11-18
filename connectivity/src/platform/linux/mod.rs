//! Linux: rtnetlink for reachability, logind for sleep and wake.
//!
//! - rtnetlink groups and route dumps: <https://man7.org/linux/man-pages/man7/rtnetlink.7.html>
//! - logind `PrepareForSleep`:
//!   <https://www.freedesktop.org/software/systemd/man/latest/org.freedesktop.login1.html>
//!
//! Both watchers run on one dedicated thread that stops when the [`Monitor`]
//! drops.

mod decode;
mod logind;
mod route;

use std::thread::JoinHandle;
use std::time::Duration;

use async_io::Timer;
use futures_lite::future;

use crate::{ConnectivityError, hub::Sink};

/// Backoff delays between watcher reconnection attempts before giving up.
const RECONNECT_DELAYS: &[Duration] = &[
    Duration::from_millis(100),
    Duration::from_millis(500),
    Duration::from_secs(1),
    Duration::from_secs(2),
    Duration::from_secs(5),
];

/// Why [`race_watchers`] returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExitReason {
    Stopped,
    RoutesExited,
    PowerExited,
}

/// Watches reachability and power state until dropped.
pub struct Monitor {
    stop: async_channel::Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl Monitor {
    pub fn start(sink: Sink) -> Result<Self, ConnectivityError> {
        let (stop, stopped) = async_channel::bounded::<()>(1);
        let (ready_tx, ready_rx) = async_channel::bounded(1);
        let thread = std::thread::Builder::new()
            .name("aither-connectivity".into())
            .spawn(move || run(&sink, &stopped, &ready_tx))
            .map_err(|error| ConnectivityError::Register {
                monitor: "linux watcher thread",
                reason: error.to_string(),
            })?;
        // The thread reports whether the route and system bus watchers came up.
        match ready_rx.recv_blocking() {
            Ok(Ok(())) => Ok(Self {
                stop,
                thread: Some(thread),
            }),
            Ok(Err(error)) => {
                let _ = thread.join();
                Err(error)
            }
            Err(_) => Err(ConnectivityError::Register {
                monitor: "linux watcher thread",
                reason: "the thread exited before reporting".into(),
            }),
        }
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        self.stop.close();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The watcher thread: runs both watchers until `stopped` closes or a watcher
/// fails past its reconnection budget.
#[allow(
    clippy::disallowed_methods,
    reason = "this is a dedicated OS thread, nothing else runs on it"
)]
fn run(
    sink: &Sink,
    stopped: &async_channel::Receiver<()>,
    ready: &async_channel::Sender<Result<(), ConnectivityError>>,
) {
    future::block_on(async {
        let routes = match route::Watcher::open(sink).await {
            Ok(routes) => routes,
            Err(error) => {
                let _ = ready.send(Err(error)).await;
                return;
            }
        };
        let logind = match logind::Watcher::connect().await {
            Ok(logind) => logind,
            Err(error) => {
                let _ = ready.send(Err(error)).await;
                return;
            }
        };
        let _ = ready.send(Ok(())).await;

        let routes_fut = watch_routes_with_reconnect(routes, sink);
        let power_fut = watch_power_with_reconnect(logind, sink);
        match race_watchers(routes_fut, power_fut, stopped).await {
            ExitReason::Stopped => {}
            ExitReason::RoutesExited => {
                tracing::error!("rtnetlink watcher exited and could not reconnect");
            }
            ExitReason::PowerExited => {
                tracing::error!("logind watcher exited and could not reconnect");
            }
        }
    });
}

/// Runs `routes` and `power` concurrently, returning as soon as `stopped`
/// closes or either watcher exits.
#[allow(clippy::future_not_send)]
async fn race_watchers<R, P>(
    routes: R,
    power: P,
    stopped: &async_channel::Receiver<()>,
) -> ExitReason
where
    R: Future<Output = ()>,
    P: Future<Output = ()>,
{
    let watchers = future::or(
        async {
            routes.await;
            ExitReason::RoutesExited
        },
        async {
            power.await;
            ExitReason::PowerExited
        },
    );
    // `stopped` only ever closes; nothing is sent on it.
    future::or(
        async {
            let _ = stopped.recv().await;
            ExitReason::Stopped
        },
        watchers,
    )
    .await
}

#[allow(clippy::future_not_send)]
async fn watch_routes_with_reconnect(initial: route::Watcher, sink: &Sink) {
    run_with_reconnect(
        "rtnetlink",
        Some(initial),
        RECONNECT_DELAYS,
        |watcher| async move {
            let _ = watcher.watch(sink).await;
        },
        || async { route::Watcher::open(sink).await.map(Some) },
        Timer::after,
    )
    .await;
}

#[allow(clippy::future_not_send)]
async fn watch_power_with_reconnect(initial: Option<logind::Watcher>, sink: &Sink) {
    if initial.is_none() {
        tracing::info!("no system bus on this machine, sleep and wake are not tracked");
        future::pending::<()>().await;
        return;
    }

    run_with_reconnect(
        "logind",
        initial,
        RECONNECT_DELAYS,
        |watcher| async move {
            watcher.watch(sink).await;
        },
        logind::Watcher::connect,
        Timer::after,
    )
    .await;
}

/// Drives a watcher and reconnects with `delays` if it exits; parks forever if
/// `reconnect` returns `Ok(None)` (watcher permanently absent on this machine),
/// and returns if reconnection fails after exhausting `delays`.
#[allow(clippy::future_not_send)]
async fn run_with_reconnect<W, E, Watch, WatchFut, Reconnect, ReconnectFut, Sleep, SleepFut>(
    name: &'static str,
    mut current: Option<W>,
    delays: &[Duration],
    mut watch: Watch,
    mut reconnect: Reconnect,
    mut sleep: Sleep,
) where
    E: std::fmt::Display,
    Watch: FnMut(W) -> WatchFut,
    WatchFut: Future<Output = ()>,
    Reconnect: FnMut() -> ReconnectFut,
    ReconnectFut: Future<Output = Result<Option<W>, E>>,
    Sleep: FnMut(Duration) -> SleepFut,
    SleepFut: Future,
{
    loop {
        if let Some(watcher) = current.take() {
            watch(watcher).await;
            tracing::warn!(
                watcher = name,
                "connectivity watcher exited; attempting reconnect"
            );
        }

        let mut reconnected = false;
        for &delay in delays {
            let _ = sleep(delay).await;
            match reconnect().await {
                Ok(Some(watcher)) => {
                    tracing::info!(watcher = name, "connectivity watcher reconnected");
                    current = Some(watcher);
                    reconnected = true;
                    break;
                }
                Ok(None) => {
                    tracing::info!(
                        watcher = name,
                        "connectivity watcher unavailable on this host"
                    );
                    future::pending::<()>().await;
                    return;
                }
                Err(error) => {
                    tracing::warn!(watcher = name, %error, "connectivity watcher reconnect failed");
                }
            }
        }

        if !reconnected {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[tokio::test]
    async fn early_exit_route_watcher_does_not_hang_when_power_is_pending() {
        let (_stop, stopped) = async_channel::bounded::<()>(1);
        let reason = race_watchers(async {}, future::pending::<()>(), &stopped).await;
        assert_eq!(reason, ExitReason::RoutesExited);
    }

    #[tokio::test]
    async fn early_exit_power_watcher_does_not_hang_when_routes_is_pending() {
        let (_stop, stopped) = async_channel::bounded::<()>(1);
        let reason = race_watchers(future::pending::<()>(), async {}, &stopped).await;
        assert_eq!(reason, ExitReason::PowerExited);
    }

    #[tokio::test]
    async fn stopped_channel_terminates_pending_watchers() {
        let (stop, stopped) = async_channel::bounded::<()>(1);
        stop.close();
        let reason =
            race_watchers(future::pending::<()>(), future::pending::<()>(), &stopped).await;
        assert_eq!(reason, ExitReason::Stopped);
    }

    #[tokio::test]
    async fn reconnect_loop_recovers_once_then_exits_when_reconnect_exhausted() {
        let watch_calls = Arc::new(AtomicUsize::new(0));
        let reconnect_calls = Arc::new(AtomicUsize::new(0));
        let watch_calls_ref = Arc::clone(&watch_calls);
        let reconnect_calls_ref = Arc::clone(&reconnect_calls);

        run_with_reconnect(
            "test",
            Some(1usize),
            &[Duration::ZERO, Duration::ZERO],
            move |_watcher| {
                let calls = Arc::clone(&watch_calls_ref);
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                }
            },
            move || {
                let calls = Arc::clone(&reconnect_calls_ref);
                async move {
                    let n = calls.fetch_add(1, Ordering::SeqCst);
                    if n == 0 {
                        Ok(Some(2usize))
                    } else {
                        Err("connection refused")
                    }
                }
            },
            |_| async {},
        )
        .await;

        assert_eq!(watch_calls.load(Ordering::SeqCst), 2);
        assert_eq!(reconnect_calls.load(Ordering::SeqCst), 3);
    }
}
