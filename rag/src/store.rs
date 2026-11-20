//! Core RAG store implementation.

use std::collections::HashSet;
use std::sync::Arc;

use aither_core::embedding::EmbeddingModel;
use futures_util::stream::{self, StreamExt, TryStreamExt};

use crate::chunking::{Chunker, FixedSizeChunker};
use crate::cleaning::{BasicCleaner, Cleaner};
use crate::config::RagConfig;
use crate::error::{RagError, Result};
use crate::index::{HnswIndex, VectorIndex};
use crate::types::{Chunk, Document, SearchResult};

/// Maximum number of chunk embedding requests kept in flight concurrently
/// during batch document ingestion, bounding provider/model saturation.
const MAX_CONCURRENT_EMBEDDINGS: usize = 16;

/// Embeds a batch of texts concurrently in input order using bounded concurrency.
///
/// # Errors
///
/// Returns the first embedding error encountered.
pub async fn embed_batch<M: EmbeddingModel>(
    embedder: &M,
    texts: Vec<String>,
) -> Result<Vec<Vec<f32>>> {
    // `embed` ties its returned future to the shared lifetime of `&self` and
    // `&str`, so texts are owned — an `&str` parameter would leak its region
    // into this future's opaque type and break higher-ranked callers.
    stream::iter(texts)
        .map(|text| async move { embedder.embed(&text).await.map_err(RagError::Embedding) })
        .buffered(MAX_CONCURRENT_EMBEDDINGS)
        .try_collect()
        .await
}

/// The core RAG store that manages documents, cleaning, chunking, and indexing.
///
/// `RagStore` combines an embedding model with a vector index to provide
/// efficient semantic search over documents.
pub struct RagStore<M: EmbeddingModel, C: Chunker = FixedSizeChunker, L: Cleaner = BasicCleaner> {
    embedder: M,
    index: Arc<HnswIndex>,
    chunker: C,
    cleaner: L,
    config: RagConfig,
}

impl<M: EmbeddingModel, C: Chunker, L: Cleaner> std::fmt::Debug for RagStore<M, C, L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RagStore")
            .field("index", &self.index)
            .field("chunker", &self.chunker.name())
            .field("cleaner", &self.cleaner.name())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<M> RagStore<M, FixedSizeChunker, BasicCleaner>
where
    M: EmbeddingModel + Send + Sync + 'static,
{
    /// Creates a new RAG store with the given embedding model.
    ///
    /// Uses default configuration and fixed-size chunking.
    #[must_use]
    pub fn new(embedder: M) -> Self {
        let dimension = embedder.dim();
        Self {
            embedder,
            index: Arc::new(HnswIndex::new(dimension)),
            chunker: FixedSizeChunker::default(),
            cleaner: BasicCleaner,
            config: RagConfig::default(),
        }
    }

    /// Creates a new RAG store with custom configuration.
    #[must_use]
    pub fn with_config(embedder: M, config: RagConfig) -> Self {
        let dimension = embedder.dim();
        Self {
            embedder,
            index: Arc::new(HnswIndex::new(dimension)),
            chunker: FixedSizeChunker::default(),
            cleaner: BasicCleaner,
            config,
        }
    }
}

impl<M, C, L> RagStore<M, C, L>
where
    M: EmbeddingModel + Send + Sync + 'static,
    C: Chunker,
    L: Cleaner,
{
    /// Creates a new RAG store with explicit components.
    #[must_use]
    pub fn with_components(embedder: M, config: RagConfig, chunker: C, cleaner: L) -> Self {
        let dimension = embedder.dim();
        Self {
            embedder,
            index: Arc::new(HnswIndex::new(dimension)),
            chunker,
            cleaner,
            config,
        }
    }

    /// Sets a custom chunker for this store.
    #[must_use]
    pub fn with_chunker<C2: Chunker>(self, chunker: C2) -> RagStore<M, C2, L> {
        RagStore {
            embedder: self.embedder,
            index: self.index,
            chunker,
            cleaner: self.cleaner,
            config: self.config,
        }
    }

    /// Sets a custom cleaner for this store.
    #[must_use]
    pub fn with_cleaner<L2: Cleaner>(self, cleaner: L2) -> RagStore<M, C, L2> {
        RagStore {
            embedder: self.embedder,
            index: self.index,
            chunker: self.chunker,
            cleaner,
            config: self.config,
        }
    }

    /// Inserts a document into the store, replacing any prior chunks stored for
    /// the same document ID.
    ///
    /// The document is cleaned, chunked, deduplicated (if enabled), embedded as
    /// a batch, and indexed in a single state update.
    ///
    /// # Returns
    /// The number of chunks actually inserted (may be less than total chunks
    /// if deduplication is enabled and duplicates are found).
    /// # Errors
    ///
    /// Returns an error if the document cannot be embedded or stored.
    pub async fn insert(&self, document: Document) -> Result<usize> {
        self.insert_batch(vec![document]).await
    }

    /// Inserts multiple documents into the store, clearing prior chunks of any
    /// re-inserted document IDs and batching chunk embeddings and index updates.
    ///
    /// # Returns
    /// The total number of chunks inserted across all documents.
    /// # Errors
    ///
    /// Returns an error if any document cannot be embedded or stored.
    pub async fn insert_batch(&self, documents: Vec<Document>) -> Result<usize> {
        if documents.is_empty() {
            return Ok(0);
        }

        let replaced_sources: HashSet<String> = documents.iter().map(|d| d.id.clone()).collect();
        let mut seen_batch_hashes = HashSet::new();
        let mut to_embed = Vec::new();

        for document in &documents {
            let cleaned = self.cleaner.clean(document);
            let chunks = self.chunker.chunk(&cleaned)?;
            for chunk in chunks {
                if self.config.deduplication
                    && (!seen_batch_hashes.insert(chunk.content_hash)
                        || self
                            .index
                            .contains_hash_outside_sources(chunk.content_hash, &replaced_sources))
                {
                    continue;
                }
                to_embed.push(chunk);
            }
        }

        let texts: Vec<String> = to_embed.iter().map(|c| c.text.clone()).collect();
        let embeddings = embed_batch(&self.embedder, texts).await?;
        let inserted = to_embed.len();

        let entries: Vec<(Chunk, Vec<f32>)> = to_embed.into_iter().zip(embeddings).collect();
        self.index
            .replace_sources_and_insert(&replaced_sources, &entries)?;

        Ok(inserted)
    }

    /// Inserts a pre-chunked chunk with a precomputed embedding.
    ///
    /// Useful for loading from persistence or custom chunking workflows.
    /// # Errors
    ///
    /// Returns an error if the embedding's dimension does not match the index.
    pub fn insert_with_embedding(&self, chunk: Chunk, embedding: Vec<f32>) -> Result<()> {
        self.index.insert(chunk, embedding)
    }

    /// Deletes a document and all its chunks from the store.
    ///
    /// # Returns
    /// `true` if any chunks were removed.
    pub fn delete(&self, doc_id: &str) -> bool {
        let removed = self.index.remove(doc_id);
        self.index.remove_source(doc_id) > 0 || removed
    }

    /// Searches for chunks similar to the query.
    /// # Errors
    ///
    /// Returns an error if the query cannot be embedded, or its dimension does
    /// not match the index.
    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        self.search_with_k(query, self.config.default_top_k).await
    }

    /// Searches for chunks similar to the query with a custom result count.
    /// # Errors
    ///
    /// Returns an error if the query cannot be embedded, or its dimension does
    /// not match the index.
    pub async fn search_with_k(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let embedding = self
            .embedder
            .embed(query)
            .await
            .map_err(RagError::Embedding)?;

        self.index
            .search_async(&embedding, top_k, self.config.similarity_threshold)
            .await
    }

    /// Returns the number of indexed chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Returns `true` if the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Clears all data from the store.
    pub fn clear(&self) {
        self.index.clear();
    }

    /// Returns a reference to the underlying index.
    pub fn index(&self) -> &HnswIndex {
        &self.index
    }

    /// Returns a reference to the embedder.
    pub const fn embedder(&self) -> &M {
        &self.embedder
    }

    /// Returns a reference to the chunker.
    pub const fn chunker(&self) -> &C {
        &self.chunker
    }

    /// Returns a reference to the cleaner.
    pub const fn cleaner(&self) -> &L {
        &self.cleaner
    }

    /// Returns the configuration.
    pub const fn config(&self) -> &RagConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::EmbeddingModel;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    struct MockEmbedder {
        dimension: usize,
        calls: Arc<AtomicUsize>,
    }

    impl MockEmbedder {
        fn new(dimension: usize) -> Self {
            Self {
                dimension,
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl EmbeddingModel for MockEmbedder {
        fn dim(&self) -> usize {
            self.dimension
        }

        fn embed(
            &self,
            text: &str,
        ) -> impl std::future::Future<Output = aither_core::Result<Vec<f32>>> + Send {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut vec = vec![0.0; self.dimension];
            for (idx, value) in vec.iter_mut().enumerate() {
                let digit = u8::try_from((text.len() + idx) % 10).expect("modulo result fits u8");
                *value = f32::from(digit) / 10.0;
            }
            std::future::ready(Ok(vec))
        }
    }

    #[tokio::test]
    async fn insert_and_search() {
        let embedder = MockEmbedder::new(4);
        let store = RagStore::new(embedder);

        let doc = Document::new("doc1", "Hello world");
        store.insert(doc).await.unwrap();

        assert_eq!(store.len(), 1);

        let results = store.search("hello").await.unwrap();
        assert_ne!(results.len(), 0);
    }

    #[tokio::test]
    async fn delete_document() {
        let embedder = MockEmbedder::new(4);
        let store = RagStore::new(embedder);

        let doc = Document::new("doc1", "Hello world");
        store.insert(doc).await.unwrap();

        assert!(store.delete("doc1"));
        assert_eq!(store.len(), 0);
        assert!(!store.delete("doc1"));
    }

    #[tokio::test]
    async fn deduplication() {
        let embedder = MockEmbedder::new(4);
        let config = RagConfig::builder().deduplication(true).build();
        let store = RagStore::with_config(embedder, config);

        let doc1 = Document::new("doc1", "Same content");
        let doc2 = Document::new("doc2", "Same content");

        let inserted1 = store.insert(doc1).await.unwrap();
        let inserted2 = store.insert(doc2).await.unwrap();

        assert_eq!(inserted1, 1);
        assert_eq!(inserted2, 0);
        assert_eq!(store.len(), 1);
    }

    /// Splits a document's text at every `|`.
    struct BarChunker;

    impl Chunker for BarChunker {
        fn chunk(&self, doc: &Document) -> Result<Vec<Chunk>> {
            Ok(doc
                .text
                .split('|')
                .enumerate()
                .map(|(index, text)| {
                    Chunk::new(
                        format!("{}#chunk_{index}", doc.id),
                        text,
                        doc.id.clone(),
                        index,
                        crate::dedup::content_hash(text),
                    )
                })
                .collect())
        }

        fn name(&self) -> &'static str {
            "bar"
        }
    }

    #[tokio::test]
    async fn reinserting_shorter_document_removes_trailing_chunks_even_with_dedup() {
        let config = RagConfig::builder().deduplication(true).build();
        let store = RagStore::with_config(MockEmbedder::new(4), config).with_chunker(BarChunker);

        assert_eq!(
            store
                .insert(Document::new("doc1", "alpha|beta|gamma|delta"))
                .await
                .unwrap(),
            4
        );
        assert_eq!(store.len(), 4);

        // Re-insert shorter version keeping `alpha` and replacing `beta..delta` with `omega`.
        assert_eq!(
            store
                .insert(Document::new("doc1", "alpha|omega"))
                .await
                .unwrap(),
            2
        );
        assert_eq!(store.len(), 2);

        let mut texts: Vec<String> = store
            .index()
            .entries()
            .into_iter()
            .map(|e| e.chunk.text)
            .collect();
        texts.sort();
        assert_eq!(texts, ["alpha", "omega"]);
    }

    #[tokio::test]
    async fn deleting_a_document_removes_chunks_after_one_dedup_skipped() {
        let config = RagConfig::builder().deduplication(true).build();
        let store = RagStore::with_config(MockEmbedder::new(4), config).with_chunker(BarChunker);

        store
            .insert(Document::new("a", "shared|first"))
            .await
            .unwrap();
        assert_eq!(
            store
                .insert(Document::new("b", "shared|second"))
                .await
                .unwrap(),
            1
        );
        assert_eq!(store.len(), 3);

        assert!(store.delete("b"));

        assert_eq!(store.len(), 2);
        assert!(
            store
                .index()
                .entries()
                .iter()
                .all(|entry| entry.chunk.source_id == "a"),
            "no chunk of the deleted document is left"
        );
    }

    #[tokio::test]
    async fn no_deduplication() {
        let embedder = MockEmbedder::new(4);
        let config = RagConfig::builder().deduplication(false).build();
        let store = RagStore::with_config(embedder, config);

        let doc1 = Document::new("doc1", "Same content");
        let doc2 = Document::new("doc2", "Same content");

        store.insert(doc1).await.unwrap();
        store.insert(doc2).await.unwrap();

        assert_eq!(store.len(), 2);
    }
}
