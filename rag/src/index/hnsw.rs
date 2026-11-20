//! HNSW-based vector index using instant-distance.

use arc_swap::{ArcSwap, Guard};
use async_lock::Mutex;
use instant_distance::{Builder, HnswMap, Point, Search};
use ordered_float::OrderedFloat;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::error::{RagError, Result};
use crate::types::{Chunk, IndexEntry, SearchResult};

use super::VectorIndex;

/// A point wrapper for instant-distance that stores an embedding vector.
#[derive(Clone, Debug)]
struct EmbeddingPoint {
    embedding: Vec<f32>,
}

impl Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance = 1 - cosine_similarity
        // This gives smaller values for more similar vectors
        1.0 - cosine_similarity(&self.embedding, &other.embedding)
    }
}

/// Computes cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut norm_a, mut norm_b) = (0.0f32, 0.0f32, 0.0f32);
    for (lhs, rhs) in a.iter().zip(b) {
        dot += lhs * rhs;
        norm_a += lhs * lhs;
        norm_b += rhs * rhs;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Internal state for the HNSW index.
struct IndexState {
    /// All stored entries (chunks + embeddings).
    entries: Vec<IndexEntry>,
    /// Map from chunk ID to index in entries.
    id_to_index: HashMap<String, usize>,
    /// Reference-counted content hashes for deduplication.
    content_hashes: HashMap<u64, usize>,
    /// The HNSW index (rebuilt after modifications).
    hnsw: Option<HnswMap<EmbeddingPoint, usize>>,
    /// Whether the index needs rebuilding.
    dirty: bool,
}

impl IndexState {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            id_to_index: HashMap::new(),
            content_hashes: HashMap::new(),
            hnsw: None,
            dirty: false,
        }
    }

    fn increment_hash(&mut self, hash: u64) {
        *self.content_hashes.entry(hash).or_insert(0) += 1;
    }

    fn decrement_hash(&mut self, hash: u64) {
        if let Some(count) = self.content_hashes.get_mut(&hash) {
            *count -= 1;
            if *count == 0 {
                self.content_hashes.remove(&hash);
            }
        }
    }

    fn upsert_entry(&mut self, chunk: Chunk, embedding: Vec<f32>) {
        if let Some(&idx) = self.id_to_index.get(&chunk.id) {
            let old_hash = self.entries[idx].chunk.content_hash;
            self.decrement_hash(old_hash);
            self.increment_hash(chunk.content_hash);
            self.entries[idx] = IndexEntry::new(chunk, embedding);
        } else {
            let idx = self.entries.len();
            self.id_to_index.insert(chunk.id.clone(), idx);
            self.increment_hash(chunk.content_hash);
            self.entries.push(IndexEntry::new(chunk, embedding));
        }
    }

    fn build_hnsw(entries: &[IndexEntry]) -> Option<HnswMap<EmbeddingPoint, usize>> {
        if entries.is_empty() {
            return None;
        }

        let points: Vec<EmbeddingPoint> = entries
            .iter()
            .map(|e| EmbeddingPoint {
                embedding: e.embedding.clone(),
            })
            .collect();

        let indices: Vec<usize> = (0..entries.len()).collect();
        Some(Builder::default().build(points, indices))
    }

    fn rebuilt(&self) -> Self {
        let mut next = self.clone_without_index();
        next.hnsw = Self::build_hnsw(&next.entries);
        next.dirty = false;
        next
    }

    /// A state holding `entries`, with the lookups derived from them.
    fn from_entries(entries: Vec<IndexEntry>) -> Self {
        let mut state = Self::new();
        for (idx, entry) in entries.into_iter().enumerate() {
            state.id_to_index.insert(entry.chunk.id.clone(), idx);
            state.increment_hash(entry.chunk.content_hash);
            state.entries.push(entry);
        }
        state.dirty = !state.entries.is_empty();
        state
    }

    fn clone_without_index(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            id_to_index: self.id_to_index.clone(),
            content_hashes: self.content_hashes.clone(),
            hnsw: None,
            dirty: true,
        }
    }
}

fn search_on_state(
    state: &IndexState,
    query: &[f32],
    top_k: usize,
    threshold: f32,
) -> Vec<SearchResult> {
    let Some(ref hnsw) = state.hnsw else {
        return Vec::new();
    };
    let query_point = EmbeddingPoint {
        embedding: query.to_vec(),
    };

    let mut search = Search::default();
    let mut results = Vec::new();

    for candidate in hnsw.search(&query_point, &mut search).take(top_k) {
        let idx = *candidate.value;
        let entry = &state.entries[idx];

        let similarity = 1.0 - candidate.distance;
        if similarity >= threshold {
            results.push(SearchResult {
                chunk: entry.chunk.clone(),
                score: similarity,
            });
        }
    }

    results.sort_by_key(|r| std::cmp::Reverse(OrderedFloat(r.score)));
    results
}

/// HNSW-based vector index for approximate nearest neighbor search.
///
/// This index uses the `instant-distance` crate for efficient similarity search.
/// It maintains an internal list of entries and rebuilds the HNSW graph as needed.
///
/// # Example
///
/// ```rust
/// use aither_rag::index::{HnswIndex, VectorIndex};
/// use aither_rag::Chunk;
///
/// let index = HnswIndex::new(384); // 384-dimensional embeddings
/// // index.insert(chunk, embedding).unwrap();
/// // let results = index.search(&query_embedding, 5, 0.0).unwrap();
/// ```
pub struct HnswIndex {
    dimension: usize,
    state: ArcSwap<IndexState>,
    /// Coordinates single-flight HNSW graph rebuilds across concurrent searchers.
    rebuild_lock: Mutex<()>,
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.load();
        f.debug_struct("HnswIndex")
            .field("dimension", &self.dimension)
            .field("len", &state.entries.len())
            .finish_non_exhaustive()
    }
}

impl HnswIndex {
    /// Creates a new HNSW index with the specified embedding dimension.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            state: ArcSwap::from_pointee(IndexState::new()),
            rebuild_lock: Mutex::new(()),
        }
    }

    const fn validate_dimension(&self, actual: usize) -> Result<()> {
        if actual != self.dimension {
            return Err(RagError::DimensionMismatch {
                expected: self.dimension,
                actual,
            });
        }
        Ok(())
    }

    /// Checks whether `hash` is held by at least one chunk outside `excluded_sources`.
    pub(crate) fn contains_hash_outside_sources(
        &self,
        hash: u64,
        excluded_sources: &HashSet<String>,
    ) -> bool {
        let state = self.state.load();
        if !state.content_hashes.contains_key(&hash) {
            return false;
        }
        if excluded_sources.is_empty() {
            return true;
        }
        state.entries.iter().any(|entry| {
            entry.chunk.content_hash == hash
                && !excluded_sources.contains(&entry.chunk.source_id)
                && !excluded_sources.contains(&entry.chunk.id)
        })
    }

    /// Atomically removes all prior chunks belonging to `sources` and inserts
    /// `entries` in a single copy-on-write transition.
    pub(crate) fn replace_sources_and_insert(
        &self,
        sources: &HashSet<String>,
        entries: &[(Chunk, Vec<f32>)],
    ) -> Result<()> {
        for (_, embedding) in entries {
            self.validate_dimension(embedding.len())?;
        }

        if sources.is_empty() && entries.is_empty() {
            return Ok(());
        }

        self.state.rcu(|state| {
            let mut kept: Vec<IndexEntry> = if sources.is_empty() {
                state.entries.clone()
            } else {
                state
                    .entries
                    .iter()
                    .filter(|entry| {
                        !sources.contains(&entry.chunk.source_id)
                            && !sources.contains(&entry.chunk.id)
                    })
                    .cloned()
                    .collect()
            };

            let removed_any = kept.len() != state.entries.len();
            if !removed_any && entries.is_empty() {
                return Arc::clone(state);
            }

            let mut next = if removed_any {
                IndexState::from_entries(std::mem::take(&mut kept))
            } else {
                state.clone_without_index()
            };

            for (chunk, embedding) in entries {
                next.upsert_entry(chunk.clone(), embedding.clone());
            }
            next.dirty = !next.entries.is_empty();
            Arc::new(next)
        });

        Ok(())
    }

    fn ensure_ready_sync(&self) -> Arc<IndexState> {
        let fast = self.state.load_full();
        if fast.entries.is_empty() || (!fast.dirty && fast.hnsw.is_some()) {
            return fast;
        }

        let rebuild_guard = self.rebuild_lock.lock_blocking();
        let ready = loop {
            let current = self.state.load_full();
            if current.entries.is_empty() || (!current.dirty && current.hnsw.is_some()) {
                break current;
            }

            let rebuilt = Arc::new(current.rebuilt());
            let previous =
                Guard::into_inner(self.state.compare_and_swap(&current, Arc::clone(&rebuilt)));
            if Arc::ptr_eq(&current, &previous) {
                break rebuilt;
            }
        };
        drop(rebuild_guard);
        ready
    }

    async fn ensure_ready_async(&self) -> Arc<IndexState> {
        let fast = self.state.load_full();
        if fast.entries.is_empty() || (!fast.dirty && fast.hnsw.is_some()) {
            return fast;
        }

        let rebuild_guard = self.rebuild_lock.lock().await;
        let ready = loop {
            let current = self.state.load_full();
            if current.entries.is_empty() || (!current.dirty && current.hnsw.is_some()) {
                break current;
            }

            let snapshot = Arc::clone(&current);
            let rebuilt = blocking::unblock(move || Arc::new(snapshot.rebuilt())).await;
            let previous =
                Guard::into_inner(self.state.compare_and_swap(&current, Arc::clone(&rebuilt)));
            if Arc::ptr_eq(&current, &previous) {
                break rebuilt;
            }
        };
        drop(rebuild_guard);
        ready
    }

    /// Searches asynchronously, offloading HNSW graph construction (single-flight)
    /// and nearest-neighbor traversal onto the blocking pool.
    pub(crate) async fn search_async(
        &self,
        query: &[f32],
        top_k: usize,
        threshold: f32,
    ) -> Result<Vec<SearchResult>> {
        self.validate_dimension(query.len())?;
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let state = self.ensure_ready_async().await;
        if state.entries.is_empty() || state.hnsw.is_none() {
            return Ok(Vec::new());
        }

        let query_vec = query.to_vec();
        Ok(blocking::unblock(move || search_on_state(&state, &query_vec, top_k, threshold)).await)
    }
}

impl VectorIndex for HnswIndex {
    fn insert(&self, chunk: Chunk, embedding: Vec<f32>) -> Result<()> {
        self.insert_batch(vec![(chunk, embedding)])
    }

    fn insert_batch(&self, entries: Vec<(Chunk, Vec<f32>)>) -> Result<()> {
        for (_, embedding) in &entries {
            self.validate_dimension(embedding.len())?;
        }
        if entries.is_empty() {
            return Ok(());
        }

        self.state.rcu(|state| {
            let mut next = state.clone_without_index();
            for (chunk, embedding) in &entries {
                next.upsert_entry(chunk.clone(), embedding.clone());
            }
            next
        });
        Ok(())
    }

    fn remove(&self, chunk_id: &str) -> bool {
        let mut found = false;
        // The closure runs again on a newer state when another update wins,
        // so the entry is looked up in the state each attempt sees.
        self.state.rcu(|state| {
            let Some(&idx) = state.id_to_index.get(chunk_id) else {
                found = false;
                return Arc::clone(state);
            };
            found = true;
            let mut next = state.clone_without_index();
            let hash = next.entries[idx].chunk.content_hash;
            next.decrement_hash(hash);

            let removed = next.entries.swap_remove(idx);
            next.id_to_index.remove(&removed.chunk.id);

            if idx < next.entries.len() {
                let swapped_id = next.entries[idx].chunk.id.clone();
                next.id_to_index.insert(swapped_id, idx);
            }
            next.dirty = !next.entries.is_empty();

            Arc::new(next)
        });
        found
    }

    fn remove_source(&self, source_id: &str) -> usize {
        let mut count = 0;
        self.state.rcu(|state| {
            let kept: Vec<IndexEntry> = state
                .entries
                .iter()
                .filter(|entry| entry.chunk.source_id != source_id)
                .cloned()
                .collect();
            count = state.entries.len() - kept.len();
            if count == 0 {
                return Arc::clone(state);
            }
            Arc::new(IndexState::from_entries(kept))
        });
        count
    }

    fn search(&self, query: &[f32], top_k: usize, threshold: f32) -> Result<Vec<SearchResult>> {
        self.validate_dimension(query.len())?;
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let state = self.ensure_ready_sync();
        Ok(search_on_state(&state, query, top_k, threshold))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn len(&self) -> usize {
        self.state.load().entries.len()
    }

    fn clear(&self) {
        self.state.store(Arc::new(IndexState::new()));
    }

    fn entries(&self) -> Vec<IndexEntry> {
        self.state.load().entries.clone()
    }

    fn load(&self, entries: Vec<IndexEntry>) -> Result<()> {
        for entry in &entries {
            self.validate_dimension(entry.embedding.len())?;
        }

        self.state
            .store(Arc::new(IndexState::from_entries(entries)));
        Ok(())
    }

    fn contains_hash(&self, hash: u64) -> bool {
        self.state.load().content_hashes.contains_key(&hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(id: &str, text: &str) -> Chunk {
        Chunk::new(id, text, "doc1", 0, crate::dedup::content_hash(text))
    }

    #[test]
    fn insert_and_search() {
        let index = HnswIndex::new(4);

        let chunk1 = make_chunk("c1", "hello");
        let chunk2 = make_chunk("c2", "world");

        index.insert(chunk1, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(chunk2, vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 2);

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 1, 0.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "c1");
    }

    #[test]
    fn remove_entry() {
        let index = HnswIndex::new(4);

        let chunk = make_chunk("c1", "hello");
        index.insert(chunk, vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 1);
        assert!(index.remove("c1"));
        assert_eq!(index.len(), 0);
        assert!(!index.remove("c1"));
    }

    /// Two removals of one entry race on a shared index: exactly one removes
    /// it, and no other entry is lost. Each round starts both threads on a
    /// barrier so their lookups and updates overlap.
    #[test]
    fn concurrent_removals_of_one_entry_remove_it_once() {
        for round in 0..2000 {
            let index = HnswIndex::new(2);
            for (n, id) in ["c0", "c1", "c2"].into_iter().enumerate() {
                let hash = u64::try_from(n).unwrap();
                index
                    .insert(
                        Chunk::new(id, id, "doc", n, hash),
                        vec![1.0, f32::from(u8::try_from(n).unwrap())],
                    )
                    .unwrap();
            }

            let barrier = std::sync::Barrier::new(2);
            let removed = std::thread::scope(|scope| {
                let racers: Vec<_> = (0..2)
                    .map(|_| {
                        scope.spawn(|| {
                            barrier.wait();
                            index.remove("c0")
                        })
                    })
                    .collect();
                racers
                    .into_iter()
                    .map(|racer| racer.join().expect("racer"))
                    .filter(|removed| *removed)
                    .count()
            });
            assert_eq!(removed, 1, "round {round}: one removal wins");

            let mut left: Vec<_> = index.entries().into_iter().map(|e| e.chunk.id).collect();
            left.sort();
            assert_eq!(left, ["c1", "c2"], "round {round}");
        }
    }

    #[test]
    fn content_hashes_are_reference_counted_across_duplicates_and_updates() {
        let index = HnswIndex::new(2);
        let shared_text = "shared chunk";
        let shared_hash = crate::dedup::content_hash(shared_text);
        let other_text = "other chunk";
        let other_hash = crate::dedup::content_hash(other_text);

        index
            .insert_batch(vec![
                (
                    Chunk::new("doc1#0", shared_text, "doc1", 0, shared_hash),
                    vec![1.0, 0.0],
                ),
                (
                    Chunk::new("doc2#0", shared_text, "doc2", 0, shared_hash),
                    vec![0.8, 0.2],
                ),
            ])
            .unwrap();

        assert!(index.contains_hash(shared_hash));

        // Removing the first duplicate must keep the shared hash alive for doc2#0.
        assert!(index.remove("doc1#0"));
        assert!(
            index.contains_hash(shared_hash),
            "hash must remain while doc2#0 still holds it"
        );

        // Updating doc2#0 to new content drops shared_hash's refcount to zero.
        index
            .insert(
                Chunk::new("doc2#0", other_text, "doc2", 0, other_hash),
                vec![0.0, 1.0],
            )
            .unwrap();
        assert!(!index.contains_hash(shared_hash));
        assert!(index.contains_hash(other_hash));

        assert!(index.remove("doc2#0"));
        assert!(!index.contains_hash(other_hash));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn batch_insert_and_concurrent_async_search_rebuild_cleanly() {
        let index = Arc::new(HnswIndex::new(4));
        let batch: Vec<(Chunk, Vec<f32>)> = (0..64usize)
            .map(|i| {
                let id = format!("chunk_{i}");
                let chunk = make_chunk(&id, &id);
                #[allow(clippy::cast_precision_loss)]
                let emb = vec![1.0, (i as f32) * 0.01, 0.25, 0.5];
                (chunk, emb)
            })
            .collect();
        index.insert_batch(batch).unwrap();
        assert_eq!(index.len(), 64);

        let mut tasks = Vec::new();
        for _ in 0..8 {
            let idx = Arc::clone(&index);
            tasks.push(tokio::spawn(async move {
                idx.search_async(&[1.0, 0.0, 0.25, 0.5], 5, 0.0)
                    .await
                    .unwrap()
            }));
        }
        for task in tasks {
            let results = task.await.unwrap();
            assert_eq!(results.len(), 5);
        }
    }

    #[test]
    fn update_existing() {
        let index = HnswIndex::new(4);

        let chunk1 = make_chunk("c1", "hello");
        index.insert(chunk1, vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        let chunk2 = make_chunk("c1", "world");
        index.insert(chunk2, vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 1);

        let results = index.search(&[0.0, 1.0, 0.0, 0.0], 1, 0.0).unwrap();
        assert_eq!(results[0].chunk.text, "world");
    }

    #[test]
    fn dimension_mismatch() {
        let index = HnswIndex::new(4);
        let chunk = make_chunk("c1", "hello");

        let result = index.insert(chunk, vec![1.0, 0.0]);
        assert!(matches!(result, Err(RagError::DimensionMismatch { .. })));
    }

    #[test]
    fn threshold_filtering() {
        let index = HnswIndex::new(4);

        let chunk1 = make_chunk("c1", "hello");
        let chunk2 = make_chunk("c2", "world");

        index.insert(chunk1, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(chunk2, vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 10, 0.9).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "c1");
    }

    #[test]
    fn contains_hash() {
        let index = HnswIndex::new(4);

        let text = "hello world";
        let hash = crate::dedup::content_hash(text);
        let chunk = Chunk::new("c1", text, "doc1", 0, hash);

        assert!(!index.contains_hash(hash));
        index.insert(chunk, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(index.contains_hash(hash));
    }

    #[test]
    fn load_entries() {
        let index = HnswIndex::new(4);

        let entries = vec![
            IndexEntry::new(make_chunk("c1", "hello"), vec![1.0, 0.0, 0.0, 0.0]),
            IndexEntry::new(make_chunk("c2", "world"), vec![0.0, 1.0, 0.0, 0.0]),
        ];

        index.load(entries).unwrap();
        assert_eq!(index.len(), 2);

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 1, 0.0).unwrap();
        assert_eq!(results[0].chunk.id, "c1");
    }

    #[test]
    fn clear_index() {
        let index = HnswIndex::new(4);

        let chunk = make_chunk("c1", "hello");
        index.insert(chunk, vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 1);
        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }
}
