//! Records the sleep and offline intervals each running command overlaps.
//!
//! One task per terminal owns every piece of state: it alone reads the
//! connectivity changes and the commands' begin and end requests, so nothing
//! is shared or locked.

#[cfg(not(feature = "connectivity"))]
pub use disabled::Outages;
#[cfg(feature = "connectivity")]
pub use enabled::Outages;

#[cfg(not(feature = "connectivity"))]
mod disabled {
    use aither_core::llm::Interruption;

    use crate::LaunchId;

    /// Without the `connectivity` feature nothing is recorded.
    #[derive(Clone)]
    pub struct Outages;

    impl Outages {
        pub const fn none() -> Self {
            Self
        }

        #[allow(clippy::unused_self)]
        pub const fn begin(&self, _launch: LaunchId) {}

        #[allow(clippy::unused_self)]
        pub fn end(&self, _launch: LaunchId) -> std::future::Ready<Vec<Interruption>> {
            std::future::ready(Vec::new())
        }
    }
}

#[cfg(feature = "connectivity")]
mod enabled {
    use std::{collections::HashMap, pin::Pin};

    use aither_connectivity::{Change, ChangeKind, Connectivity};
    use aither_core::llm::{Interruption, InterruptionCause};
    use async_channel::{Receiver, Sender};
    use executor_core::{Executor, Task};
    use futures::Stream;
    use futures_lite::{StreamExt, future};
    use jiff::{Timestamp, tz::TimeZone};

    use crate::LaunchId;

    enum Request {
        Begin {
            launch: LaunchId,
            at: Timestamp,
        },
        End {
            launch: LaunchId,
            at: Timestamp,
            reply: Sender<Vec<Interruption>>,
        },
    }

    /// A handle on the task recording outages; records nothing until a
    /// connectivity source is given.
    #[derive(Clone)]
    pub struct Outages {
        requests: Option<Sender<Request>>,
    }

    impl Outages {
        pub const fn none() -> Self {
            Self { requests: None }
        }

        /// Subscribes to `connectivity` once and records for every command
        /// that begins afterwards. The task ends with the last handle.
        pub fn watching<E: Executor>(connectivity: &Connectivity, executor: &E) -> Self {
            let (requests, incoming) = async_channel::unbounded();
            let changes = Box::pin(connectivity.subscribe());
            let recorder = Recorder::new(connectivity);
            executor.spawn(recorder.run(changes, incoming)).detach();
            Self {
                requests: Some(requests),
            }
        }

        /// Starts recording for a command that is starting now.
        pub fn begin(&self, launch: LaunchId) {
            if let Some(requests) = &self.requests {
                let _ = requests.try_send(Request::Begin {
                    launch,
                    at: Timestamp::now(),
                });
            }
        }

        /// Stops recording for a command that ended now and returns the
        /// intervals it overlapped, oldest first.
        pub async fn end(&self, launch: LaunchId) -> Vec<Interruption> {
            let Some(requests) = &self.requests else {
                return Vec::new();
            };
            let (reply, answer) = async_channel::bounded(1);
            let request = Request::End {
                launch,
                at: Timestamp::now(),
                reply,
            };
            // The task is gone only when the runtime is shutting down.
            if requests.send(request).await.is_err() {
                return Vec::new();
            }
            answer.recv().await.unwrap_or_default()
        }
    }

    struct Span {
        cause: InterruptionCause,
        from: Timestamp,
        to: Timestamp,
    }

    struct Running {
        started: Timestamp,
        spans: Vec<Span>,
    }

    impl Running {
        fn record(&mut self, cause: InterruptionCause, since: Timestamp, to: Timestamp) {
            let from = since.max(self.started);
            if from <= to {
                self.spans.push(Span { cause, from, to });
            }
        }
    }

    struct Recorder {
        sleeping_since: Option<Timestamp>,
        offline_since: Option<Timestamp>,
        running: HashMap<LaunchId, Running>,
    }

    impl Recorder {
        fn new(connectivity: &Connectivity) -> Self {
            Self {
                sleeping_since: None,
                offline_since: (!connectivity.is_online()).then(Timestamp::now),
                running: HashMap::new(),
            }
        }

        async fn run(
            mut self,
            mut changes: Pin<Box<impl Stream<Item = Change> + Send>>,
            incoming: Receiver<Request>,
        ) {
            enum Step {
                Change(Option<Change>),
                Request(Option<Request>),
            }
            let mut watching = true;
            loop {
                // Changes first, so a request never overtakes one reported
                // before it.
                let step = future::or(
                    async {
                        if watching {
                            Step::Change(changes.next().await)
                        } else {
                            future::pending().await
                        }
                    },
                    async { Step::Request(incoming.recv().await.ok()) },
                )
                .await;
                match step {
                    Step::Change(Some(change)) => self.apply(change),
                    Step::Change(None) => watching = false,
                    Step::Request(None) => return,
                    Step::Request(Some(request)) => {
                        while watching {
                            match future::poll_once(changes.next()).await {
                                Some(Some(change)) => self.apply(change),
                                Some(None) => watching = false,
                                None => break,
                            }
                        }
                        self.handle(request).await;
                    }
                }
            }
        }

        async fn handle(&mut self, request: Request) {
            match request {
                Request::Begin { launch, at } => {
                    self.running.insert(
                        launch,
                        Running {
                            started: at,
                            spans: Vec::new(),
                        },
                    );
                }
                Request::End { launch, at, reply } => {
                    let interruptions = self.finish(launch, at);
                    let _ = reply.send(interruptions).await;
                }
            }
        }

        fn apply(&mut self, Change { kind, at }: Change) {
            match kind {
                ChangeKind::Sleeping => {
                    self.sleeping_since.get_or_insert(at);
                }
                ChangeKind::Offline => {
                    self.offline_since.get_or_insert(at);
                }
                ChangeKind::Woke => {
                    if let Some(since) = self.sleeping_since.take() {
                        self.close(InterruptionCause::Asleep, since, at);
                    }
                }
                ChangeKind::Online => {
                    if let Some(since) = self.offline_since.take() {
                        self.close(InterruptionCause::Offline, since, at);
                    }
                }
            }
        }

        fn close(&mut self, cause: InterruptionCause, since: Timestamp, to: Timestamp) {
            for running in self.running.values_mut() {
                running.record(cause, since, to);
            }
        }

        /// Closes the intervals still open at `at` and hands over the
        /// command's record.
        fn finish(&mut self, launch: LaunchId, at: Timestamp) -> Vec<Interruption> {
            let Some(mut running) = self.running.remove(&launch) else {
                return Vec::new();
            };
            if let Some(since) = self.sleeping_since {
                running.record(InterruptionCause::Asleep, since, at);
            }
            if let Some(since) = self.offline_since {
                running.record(InterruptionCause::Offline, since, at);
            }
            running.spans.sort_by_key(|span| span.from);
            let zone = TimeZone::system();
            running
                .spans
                .iter()
                .map(|span| Interruption {
                    cause: span.cause,
                    from: local_time(span.from, &zone),
                    to: local_time(span.to, &zone),
                })
                .collect()
        }
    }

    fn local_time(at: Timestamp, zone: &TimeZone) -> String {
        at.to_zoned(zone.clone()).strftime("%H:%M:%S").to_string()
    }
}
