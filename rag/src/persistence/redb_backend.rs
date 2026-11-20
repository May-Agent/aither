//! redb-based embedded database persistence.

use redb::{Database, ReadableTable, TableDefinition};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{RagError, Result};
use crate::types::IndexEntry;

use super::Persistence;

const ENTRIES_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("entries");

/// Embedded database persistence using redb.
///
/// This backend provides durable storage using redb, a pure-Rust embedded
/// key-value database. It supports atomic full-snapshot replacement and crash recovery.
///
/// # Example
///
/// ```rust,no_run
/// use aither_rag::persistence::{Persistence, RedbPersistence};
///
/// let persistence = RedbPersistence::new("./index.redb").unwrap();
/// // persistence.save(&entries).await?;
/// // let loaded = persistence.load().await?;
/// ```
pub struct RedbPersistence {
    path: PathBuf,
    db: Arc<Database>,
}

impl std::fmt::Debug for RedbPersistence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedbPersistence")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl RedbPersistence {
    /// Creates or opens a redb persistence backend.
    ///
    /// # Arguments
    /// * `path` - Path to the database file
    ///
    /// # Errors
    /// Returns an error if the database cannot be opened or created.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let db = Database::create(&path).map_err(|e| RagError::Database(e.to_string()))?;

        Ok(Self {
            path,
            db: Arc::new(db),
        })
    }

    fn save_entries_sync(db: &Database, entries: &[IndexEntry]) -> Result<()> {
        let write_txn = db
            .begin_write()
            .map_err(|e| RagError::Database(e.to_string()))?;

        match write_txn.delete_table(ENTRIES_TABLE) {
            Ok(_) | Err(redb::TableError::TableDoesNotExist(_)) => {}
            Err(e) => return Err(RagError::Database(e.to_string())),
        }

        if !entries.is_empty() {
            let mut table = write_txn
                .open_table(ENTRIES_TABLE)
                .map_err(|e| RagError::Database(e.to_string()))?;

            for entry in entries {
                let serialized = serde_json::to_vec(entry)
                    .map_err(|e| RagError::Serialization(e.to_string()))?;

                table
                    .insert(entry.chunk.id.as_str(), serialized.as_slice())
                    .map_err(|e| RagError::Database(e.to_string()))?;
            }
        }

        write_txn
            .commit()
            .map_err(|e| RagError::Database(e.to_string()))?;

        Ok(())
    }

    fn load_entries_sync(db: &Database) -> Result<Vec<IndexEntry>> {
        let read_txn = db
            .begin_read()
            .map_err(|e| RagError::Database(e.to_string()))?;

        let table = match read_txn.open_table(ENTRIES_TABLE) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(RagError::Database(e.to_string())),
        };

        let mut entries = Vec::new();

        for result in table
            .iter()
            .map_err(|e| RagError::Database(e.to_string()))?
        {
            let (_, value) = result.map_err(|e| RagError::Database(e.to_string()))?;
            let entry: IndexEntry = serde_json::from_slice(value.value())
                .map_err(|e| RagError::Serialization(e.to_string()))?;
            entries.push(entry);
        }

        Ok(entries)
    }
}

impl Persistence for RedbPersistence {
    fn save<'a>(
        &'a self,
        entries: &'a [IndexEntry],
    ) -> impl std::future::Future<Output = Result<()>> + Send + 'a {
        let db = Arc::clone(&self.db);
        let owned = entries.to_vec();
        blocking::unblock(move || Self::save_entries_sync(&db, &owned))
    }

    fn load(&self) -> impl std::future::Future<Output = Result<Vec<IndexEntry>>> + Send + '_ {
        let db = Arc::clone(&self.db);
        blocking::unblock(move || Self::load_entries_sync(&db))
    }

    fn extension(&self) -> &'static str {
        "redb"
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Chunk;
    use tempfile::tempdir;

    fn make_entry(id: &str, text: &str) -> IndexEntry {
        let chunk = Chunk::new(id, text, "doc1", 0, crate::dedup::content_hash(text));
        IndexEntry::new(chunk, vec![1.0, 2.0, 3.0, 4.0])
    }

    #[tokio::test]
    async fn save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.redb");
        let persistence = RedbPersistence::new(&path).unwrap();

        let entries = vec![make_entry("c1", "hello"), make_entry("c2", "world")];

        persistence.save(&entries).await.unwrap();

        let loaded = persistence.load().await.unwrap();
        assert_eq!(loaded.len(), 2);

        let ids: Vec<_> = loaded.iter().map(|e| e.chunk.id.as_str()).collect();
        assert!(ids.contains(&"c1"));
        assert!(ids.contains(&"c2"));
    }

    #[tokio::test]
    async fn load_empty_db() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.redb");
        let persistence = RedbPersistence::new(&path).unwrap();

        let loaded = persistence.load().await.unwrap();
        assert!(loaded.is_empty());
    }

    #[tokio::test]
    async fn save_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.redb");
        let persistence = RedbPersistence::new(&path).unwrap();

        persistence.save(&[]).await.unwrap();
        let loaded = persistence.load().await.unwrap();
        assert!(loaded.is_empty());
    }

    #[tokio::test]
    async fn overwrite_entries() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.redb");
        let persistence = RedbPersistence::new(&path).unwrap();

        let entries1 = vec![make_entry("c1", "hello")];
        persistence.save(&entries1).await.unwrap();

        let entries2 = vec![make_entry("c1", "world")];
        persistence.save(&entries2).await.unwrap();

        let loaded = persistence.load().await.unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].chunk.text, "world");
    }

    #[tokio::test]
    async fn save_removes_deleted_keys_and_clears_when_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("delete_stale.redb");
        let persistence = RedbPersistence::new(&path).unwrap();

        persistence
            .save(&[
                make_entry("c1", "first"),
                make_entry("c2", "second"),
                make_entry("c3", "third"),
            ])
            .await
            .unwrap();
        assert_eq!(persistence.load().await.unwrap().len(), 3);

        // Save with c1 and c3 removed: only c2 must remain.
        persistence
            .save(&[make_entry("c2", "second")])
            .await
            .unwrap();
        let remaining = persistence.load().await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].chunk.id, "c2");

        // Save empty slice: database must load empty.
        persistence.save(&[]).await.unwrap();
        assert!(persistence.load().await.unwrap().is_empty());
    }
}
