//! State: keyed conditions the model is told about only when they change.
//!
//! Memory pressure, machine load, the free space of a volume: each is a
//! condition with one current value, and only the latest value matters. A
//! [`StateSource`] reports those values; the context keeps, per key, the
//! value the model last saw, and an alert is written only as a request is
//! assembled and only when the current value differs from it. Nothing is
//! written while the agent is idle, so a condition that rises and settles
//! back between two requests reaches the model as nothing at all.
//!
//! A state alert is restated rather than kept: every reassembly, a cold
//! cache's or a compaction's, removes the earlier alerts and what the model
//! saw with them, and the next request writes the current value again only
//! when it is not the normal one, which the model assumes until told
//! otherwise.

use std::collections::BTreeMap;
use std::future::Future;
use std::hash::Hash;
use std::pin::Pin;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::source::Block;

/// A state whose value differs from the one the model last saw, as the
/// model is to be told it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateChange<'a, K, V> {
    key: &'a K,
    seen: Option<&'a V>,
    now: &'a V,
}

impl<'a, K, V> StateChange<'a, K, V> {
    /// The state that changed.
    #[must_use]
    pub const fn key(&self) -> &'a K {
        self.key
    }

    /// The value the model last saw; `None` when it was never told, or
    /// nothing it was told is still in the context, so it assumes the
    /// normal value.
    #[must_use]
    pub const fn seen(&self) -> Option<&'a V> {
        self.seen
    }

    /// The value now.
    #[must_use]
    pub const fn now(&self) -> &'a V {
        self.now
    }
}

/// Supplies state: keyed conditions with a latest value.
///
/// The agent reads the source only as a request is assembled. The context
/// keeps the value the model last saw of every key, and the agent appends
/// the alert [`announce`](Self::announce) writes only when the latest value
/// and that one differ. A return to the value the model already saw writes
/// nothing. After a reassembly the model has seen nothing, so a key is
/// restated only when its value is not the normal one.
///
/// State never starts a run of the agent.
///
/// # Example
///
/// ```
/// use aither_agent::{Block, StateChange, StateSource};
///
/// struct SleepMode(bool);
///
/// impl StateSource for SleepMode {
///     const ID: &'static str = "sleep_mode";
///     type Key = ();
///     type Value = bool;
///
///     async fn poll(&mut self) -> Vec<((), bool)> {
///         vec![((), self.0)]
///     }
///
///     fn is_normal(&self, (): &(), asleep: &bool) -> bool {
///         !asleep
///     }
///
///     fn announce(&self, change: &StateChange<'_, (), bool>) -> Block {
///         let text = if *change.now() { "The host sleeps." } else { "The host is awake." };
///         Block::new("sleep_mode", text)
///     }
/// }
/// ```
pub trait StateSource: Send + Sync + 'static {
    /// Names this source in the context, which keeps what the model saw of
    /// its states under it. It is unique among the agent's state sources
    /// and does not change between versions of the embedder, or a
    /// restored context would attach what the model saw to another source.
    const ID: &'static str;

    /// The identity of one state, such as a volume. It serializes with the
    /// context.
    type Key: Eq + Hash + Clone + Serialize + DeserializeOwned + Send + Sync + 'static;

    /// A state's value. Two values are the same state when they are equal.
    /// It serializes with the context.
    type Value: PartialEq + Clone + Serialize + DeserializeOwned + Send + Sync + 'static;

    /// The latest value of every state that may have changed since the
    /// last call; a key left out keeps the value it had.
    ///
    /// Called only as a request is assembled, never while the agent is
    /// idle, so a source may drain whatever it collected meanwhile. It must
    /// not block.
    fn poll(&mut self) -> impl Future<Output = Vec<(Self::Key, Self::Value)>> + Send;

    /// Whether `value` is the normal value of the state `key`: the one the
    /// model assumes while it has been told nothing.
    fn is_normal(&self, key: &Self::Key, value: &Self::Value) -> bool;

    /// The alert telling the model of `change`, appended to the
    /// conversation once and removed at the next reassembly.
    fn announce(&self, change: &StateChange<'_, Self::Key, Self::Value>) -> Block;
}

/// One state: its latest value.
struct Entry<K, V> {
    key: K,
    now: V,
}

/// A [`StateSource`] with the latest value of each of its states.
struct Tracked<S: StateSource> {
    source: S,
    /// In the order the keys first appeared, so alerts come in a stable
    /// order.
    entries: Vec<Entry<S::Key, S::Value>>,
}

impl<S: StateSource> Tracked<S> {
    fn record(&mut self, key: S::Key, value: S::Value) {
        match self.entries.iter_mut().find(|entry| entry.key == key) {
            Some(entry) => entry.now = value,
            None => self.entries.push(Entry { key, now: value }),
        }
    }

    /// The alerts for every state the model does not know as it stands,
    /// and what it has seen of the source's states after them.
    ///
    /// `stored` is what the context held of this source: the keys and the
    /// values the model last saw for them.
    fn alerts(&self, stored: Option<serde_json::Value>) -> (Vec<Block>, serde_json::Value) {
        let mut seen: Vec<(S::Key, S::Value)> = stored.map_or_else(Vec::new, |stored| {
            serde_json::from_value(stored).unwrap_or_else(|error| {
                panic!(
                    "the context holds what the model saw of `{}` in another form: {error}",
                    S::ID
                )
            })
        });
        let mut alerts = Vec::new();
        for entry in &self.entries {
            let position = seen.iter().position(|(key, _)| *key == entry.key);
            let known = position.map_or_else(
                || self.source.is_normal(&entry.key, &entry.now),
                |position| seen[position].1 == entry.now,
            );
            if known {
                continue;
            }
            alerts.push(self.source.announce(&StateChange {
                key: &entry.key,
                seen: position.map(|position| &seen[position].1),
                now: &entry.now,
            }));
            match position {
                Some(position) => seen[position].1 = entry.now.clone(),
                None => seen.push((entry.key.clone(), entry.now.clone())),
            }
        }
        let seen = serde_json::to_value(&seen).expect("the keys and values of a state serialize");
        (alerts, seen)
    }
}

/// What a state source answers a request with: its alerts, and what the
/// model has seen of its states after them.
type Answer = (Vec<Block>, serde_json::Value);

/// Object-safe twin of [`StateSource`], with the agent's bookkeeping.
trait StateSourceImpl: Send + Sync {
    fn id(&self) -> &'static str;

    fn on_request(
        &mut self,
        stored: Option<serde_json::Value>,
    ) -> Pin<Box<dyn Future<Output = Answer> + Send + '_>>;
}

impl<S: StateSource> StateSourceImpl for Tracked<S> {
    fn id(&self) -> &'static str {
        S::ID
    }

    fn on_request(
        &mut self,
        stored: Option<serde_json::Value>,
    ) -> Pin<Box<dyn Future<Output = Answer> + Send + '_>> {
        Box::pin(async move {
            for (key, value) in self.source.poll().await {
                self.record(key, value);
            }
            self.alerts(stored)
        })
    }
}

/// What the model has seen of every state source's states, under the
/// source's [`ID`](StateSource::ID).
///
/// It is part of the [`Context`](super::Context), so it serializes, forks
/// and is replaced with it. A source's part is its keys and the values the
/// model last saw for them, in the serde form of the source's own types.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeenStates(BTreeMap<String, serde_json::Value>);

impl SeenStates {
    /// Whether the model has seen nothing of any state.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// What the model saw of the source `id`.
    pub(crate) fn get(&self, id: &str) -> Option<&serde_json::Value> {
        self.0.get(id)
    }

    /// Records what the model has now seen of the source `id`.
    pub(crate) fn set(&mut self, id: &str, seen: serde_json::Value) {
        self.0.insert(id.to_owned(), seen);
    }

    /// Forgets everything the model saw: the alerts that told it are gone.
    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }
}

/// A type-erased [`StateSource`], so an agent can hold a heterogeneous
/// set. What the model has seen of its states lives in the context.
pub struct AnyStateSource {
    inner: Box<dyn StateSourceImpl>,
}

impl std::fmt::Debug for AnyStateSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnyStateSource")
            .finish_non_exhaustive()
    }
}

impl AnyStateSource {
    /// Erases the concrete type of `source`.
    #[must_use]
    pub fn new<S: StateSource>(source: S) -> Self {
        Self {
            inner: Box::new(Tracked {
                source,
                entries: Vec::new(),
            }),
        }
    }

    /// The source's [`ID`](StateSource::ID).
    pub(crate) fn id(&self) -> &'static str {
        self.inner.id()
    }

    /// Reads the source and returns the alerts for the states whose value
    /// differs from what the model last saw, and what it has seen of the
    /// source's states then.
    ///
    /// `stored` is what the context held of this source: `None` when the
    /// model has seen nothing of it, as after a reassembly.
    pub(crate) fn on_request(
        &mut self,
        stored: Option<serde_json::Value>,
    ) -> impl Future<Output = Answer> + Send + '_ {
        self.inner.on_request(stored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A disk whose free-space level a test sets; `0` is normal.
    struct Disk {
        levels: async_channel::Receiver<u8>,
    }

    impl StateSource for Disk {
        const ID: &'static str = "disk";
        type Key = String;
        type Value = u8;

        fn poll(&mut self) -> impl Future<Output = Vec<(String, u8)>> + Send {
            std::future::ready(
                std::iter::from_fn(|| self.levels.try_recv().ok())
                    .map(|level| ("/work".to_owned(), level))
                    .collect(),
            )
        }

        fn is_normal(&self, _: &String, level: &u8) -> bool {
            *level == 0
        }

        fn announce(&self, change: &StateChange<'_, String, u8>) -> Block {
            Block::new(
                "disk",
                format!("{} {:?} -> {}", change.key(), change.seen(), change.now()),
            )
        }
    }

    /// A disk source and the seen states of the context it is read for.
    struct Harness {
        levels: async_channel::Sender<u8>,
        source: AnyStateSource,
        seen: SeenStates,
    }

    impl Harness {
        fn new() -> Self {
            let (levels, receiver) = async_channel::unbounded();
            Self {
                levels,
                source: AnyStateSource::new(Disk { levels: receiver }),
                seen: SeenStates::default(),
            }
        }

        async fn send(&self, level: u8) {
            self.levels.send(level).await.expect("open");
        }

        /// A request: the alerts' texts. A reassembly before it empties
        /// what the model saw.
        async fn request(&mut self, reassembled: bool) -> Vec<String> {
            if reassembled {
                self.seen.clear();
            }
            let id = self.source.id();
            let (alerts, seen) = self.source.on_request(self.seen.get(id).cloned()).await;
            self.seen.set(id, seen);
            alerts
                .iter()
                .map(|block| block.content().to_owned())
                .collect()
        }
    }

    const NONE: [&str; 0] = [];

    /// The normal value is what the model assumes: nothing is said of it,
    /// a change from it is said once, and only the latest value counts.
    #[tokio::test]
    async fn only_a_value_the_model_does_not_know_is_announced() {
        let mut disk = Harness::new();
        disk.send(0).await;
        assert_eq!(disk.request(false).await, NONE);

        disk.send(1).await;
        disk.send(2).await;
        assert_eq!(disk.request(false).await, ["/work None -> 2"]);
        assert_eq!(disk.request(false).await, NONE, "said once");

        disk.send(0).await;
        assert_eq!(disk.request(false).await, ["/work Some(2) -> 0"]);
    }

    /// A value that changes and comes back between two requests is no
    /// news.
    #[tokio::test]
    async fn a_return_to_the_value_the_model_saw_says_nothing() {
        let mut disk = Harness::new();
        disk.send(2).await;
        assert_eq!(disk.request(false).await.len(), 1);
        disk.send(0).await;
        disk.send(2).await;
        assert_eq!(disk.request(false).await, NONE);
    }

    /// After a reassembly the model knows nothing: a value that is not
    /// normal is restated, a normal one is not.
    #[tokio::test]
    async fn a_reassembly_restates_only_a_value_that_is_not_normal() {
        let mut disk = Harness::new();
        disk.send(2).await;
        assert_eq!(disk.request(false).await.len(), 1);
        assert_eq!(disk.request(true).await, ["/work None -> 2"]);

        disk.send(0).await;
        assert_eq!(disk.request(false).await.len(), 1);
        assert_eq!(disk.request(true).await, NONE);
    }
}
