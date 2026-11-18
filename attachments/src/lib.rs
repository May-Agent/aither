//! Attachment cache for provider file uploads.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use async_fs::File;
use futures_lite::io::AsyncReadExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CACHE_FILE_NAME: &str = "file_cache.json";

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Returns the default cache directory for attachments.
#[must_use]
pub fn default_cache_dir() -> PathBuf {
    std::env::var_os("AITHER_CACHE_DIR").map_or_else(
        || std::env::temp_dir().join("aither").join("cache"),
        PathBuf::from,
    )
}

/// A cache entry for an uploaded file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// SHA-256 hash of file content (for detecting changes).
    pub content_hash: String,
    /// Provider name (e.g., "gemini", "openai").
    pub provider: String,
    /// Provider-specific reference (file URI or `file_id`).
    pub reference: String,
    /// When this reference expires (None if it never expires).
    #[serde(with = "system_time_option")]
    pub expires_at: Option<SystemTime>,
}

impl CacheEntry {
    /// Create a new cache entry.
    #[must_use]
    pub const fn new(
        content_hash: String,
        provider: String,
        reference: String,
        expires_at: Option<SystemTime>,
    ) -> Self {
        Self {
            content_hash,
            provider,
            reference,
            expires_at,
        }
    }

    /// Check if this cache entry is still valid (not expired and hash matches).
    #[must_use]
    pub fn is_valid(&self, current_hash: &str) -> bool {
        if self.content_hash != current_hash {
            return false;
        }
        self.expires_at
            .is_none_or(|expires| SystemTime::now() < expires)
    }
}

/// Structured key identifying an uploaded file for a specific provider.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct CacheKey {
    path: PathBuf,
    provider: String,
}

/// File cache for tracking uploaded files across providers.
///
/// The cache stores file references keyed by `(canonical_path, provider)` pairs.
/// Each entry includes the content hash to detect file changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCache {
    /// Cache directory for persistence.
    #[serde(skip)]
    cache_dir: PathBuf,
    /// Map of (`canonical_path`, provider) -> `CacheEntry`.
    #[serde(with = "entries_serde")]
    entries: HashMap<CacheKey, CacheEntry>,
}

impl FileCache {
    /// Open a file cache with the given cache directory.
    ///
    /// Loads existing cache data if the cache file exists.
    ///
    /// # Errors
    ///
    /// Returns an error when the cache file cannot be read or parsed.
    pub async fn open(cache_dir: PathBuf) -> std::io::Result<Self> {
        let cache_file = cache_dir.join(CACHE_FILE_NAME);
        match async_fs::read_to_string(&cache_file).await {
            Ok(contents) => {
                let cache: Self = serde_json::from_str(&contents)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                Ok(Self {
                    cache_dir,
                    entries: cache.entries,
                })
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Self {
                cache_dir,
                entries: HashMap::new(),
            }),
            Err(err) => Err(err),
        }
    }

    /// Save cache to disk atomically via a sibling temporary file and rename.
    ///
    /// # Errors
    ///
    /// Returns an error when the cache directory cannot be created or file cannot be written.
    pub async fn save(&self) -> std::io::Result<()> {
        async_fs::create_dir_all(&self.cache_dir).await?;
        let cache_file = self.cache_dir.join(CACHE_FILE_NAME);
        let contents = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp_file = self.cache_dir.join(format!(
            ".{CACHE_FILE_NAME}.tmp.{}.{seq}",
            std::process::id()
        ));

        if let Err(err) = async_fs::write(&tmp_file, contents).await {
            let _ = async_fs::remove_file(&tmp_file).await;
            return Err(err);
        }
        if let Err(err) = async_fs::rename(&tmp_file, &cache_file).await {
            let _ = async_fs::remove_file(&tmp_file).await;
            return Err(err);
        }
        Ok(())
    }

    /// Generate a canonical cache key from file path and provider.
    async fn cache_key(file_path: &Path, provider: &str) -> std::io::Result<CacheKey> {
        let path = async_fs::canonicalize(file_path).await?;
        Ok(CacheKey {
            path,
            provider: provider.to_string(),
        })
    }

    /// Get a cached entry for a file if it exists and is valid.
    ///
    /// Returns None if:
    /// - No cache entry exists
    /// - The file has changed (hash mismatch)
    /// - The cache entry has expired
    ///
    /// # Errors
    ///
    /// Returns an error when path canonicalization or file hashing fails.
    pub async fn get(
        &self,
        file_path: &Path,
        provider: &str,
    ) -> std::io::Result<Option<CacheEntry>> {
        let key = Self::cache_key(file_path, provider).await?;
        let Some(entry) = self.entries.get(&key) else {
            return Ok(None);
        };

        let current_hash = hash_file(&key.path).await?;
        if entry.is_valid(&current_hash) {
            Ok(Some(entry.clone()))
        } else {
            Ok(None)
        }
    }

    /// Insert a new cache entry.
    ///
    /// The content hash is computed automatically from the file.
    ///
    /// # Errors
    ///
    /// Returns an error when path canonicalization or file hashing fails.
    pub async fn insert(
        &mut self,
        file_path: &Path,
        provider: &str,
        reference: String,
        expires_at: Option<SystemTime>,
    ) -> std::io::Result<()> {
        let key = Self::cache_key(file_path, provider).await?;
        let content_hash = hash_file(&key.path).await?;
        let entry = CacheEntry::new(content_hash, provider.to_string(), reference, expires_at);
        self.entries.insert(key, entry);
        Ok(())
    }

    /// Remove a cache entry.
    pub async fn remove(&mut self, file_path: &Path, provider: &str) -> Option<CacheEntry> {
        if let Ok(key) = Self::cache_key(file_path, provider).await
            && let Some(removed) = self.entries.remove(&key)
        {
            return Some(removed);
        }
        let fallback_key = CacheKey {
            path: file_path.to_path_buf(),
            provider: provider.to_string(),
        };
        self.entries.remove(&fallback_key)
    }

    /// Remove all expired entries from the cache.
    ///
    /// Returns true if any entries were removed.
    pub fn prune_expired(&mut self) -> bool {
        let now = SystemTime::now();
        let before = self.entries.len();
        self.entries
            .retain(|_, entry| entry.expires_at.is_none_or(|expires| now < expires));
        before != self.entries.len()
    }

    /// Calculate expiration time from a duration.
    #[must_use]
    pub fn expires_in(duration: Duration) -> Option<SystemTime> {
        SystemTime::now().checked_add(duration)
    }
}

async fn hash_file(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = file.read(&mut buffer).await?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    let result = hasher.finalize();
    Ok(const_hex::encode(result))
}

mod entries_serde {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{CacheEntry, CacheKey};

    #[derive(Serialize, Deserialize)]
    struct StoredEntry {
        path: PathBuf,
        #[serde(flatten)]
        entry: CacheEntry,
    }

    #[allow(clippy::implicit_hasher)]
    pub fn serialize<S>(
        entries: &HashMap<CacheKey, CacheEntry>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut items: Vec<StoredEntry> = entries
            .iter()
            .map(|(key, entry)| StoredEntry {
                path: key.path.clone(),
                entry: entry.clone(),
            })
            .collect();
        items.sort_by(|a, b| {
            a.path
                .cmp(&b.path)
                .then_with(|| a.entry.provider.cmp(&b.entry.provider))
        });
        items.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<CacheKey, CacheEntry>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let items = Vec::<StoredEntry>::deserialize(deserializer)?;
        Ok(items
            .into_iter()
            .map(|item| {
                let key = CacheKey {
                    path: item.path,
                    provider: item.entry.provider.clone(),
                };
                (key, item.entry)
            })
            .collect())
    }
}

/// Serde support for `Option<SystemTime>`.
mod system_time_option {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[allow(clippy::ref_option)]
    pub fn serialize<S>(time: &Option<SystemTime>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match time {
            Some(t) => {
                let duration = t.duration_since(UNIX_EPOCH).unwrap_or_default();
                Some(duration.as_secs()).serialize(serializer)
            }
            None => None::<u64>.serialize(serializer),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<SystemTime>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs: Option<u64> = Option::deserialize(deserializer)?;
        Ok(secs.map(|s| UNIX_EPOCH + Duration::from_secs(s)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_file_cache_insert_get() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let cache_dir = dir.path().join("cache");
        let file_path = dir.path().join("file.txt");
        async_fs::write(&file_path, b"hello")
            .await
            .expect("write file");

        let mut cache = FileCache::open(cache_dir).await.expect("open cache");
        cache
            .insert(&file_path, "openai", "file-1".to_string(), None)
            .await
            .expect("insert");
        cache.save().await.expect("save");

        let cache = FileCache::open(dir.path().join("cache"))
            .await
            .expect("reload");
        let entry = cache
            .get(&file_path, "openai")
            .await
            .expect("get")
            .expect("entry");
        assert_eq!(entry.reference, "file-1");
    }

    #[tokio::test]
    async fn test_file_cache_canonicalizes_dotdot_and_symlink_paths() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let cache_dir = dir.path().join("cache");
        let sub_dir = dir.path().join("sub");
        async_fs::create_dir_all(&sub_dir)
            .await
            .expect("create subdir");
        let real_file = sub_dir.join("payload.bin");
        async_fs::write(&real_file, b"content")
            .await
            .expect("write file");

        let uncanonical = dir
            .path()
            .join("sub")
            .join("..")
            .join("sub")
            .join("payload.bin");

        let mut cache = FileCache::open(cache_dir.clone())
            .await
            .expect("open cache");
        cache
            .insert(&uncanonical, "gemini", "uri-1".to_string(), None)
            .await
            .expect("insert");

        let hit = cache
            .get(&real_file, "gemini")
            .await
            .expect("get real")
            .expect("canonical hit");
        assert_eq!(hit.reference, "uri-1");

        #[cfg(unix)]
        {
            let link_path = dir.path().join("link.bin");
            std::os::unix::fs::symlink(&real_file, &link_path).expect("symlink");
            let link_hit = cache
                .get(&link_path, "gemini")
                .await
                .expect("get link")
                .expect("symlink hit");
            assert_eq!(link_hit.reference, "uri-1");
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_file_cache_structured_keys_avoid_delimiter_collisions() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let cache_dir = dir.path().join("cache");
        let file_a = dir.path().join("a::b");
        let file_b = dir.path().join("a");
        async_fs::write(&file_a, b"same").await.expect("write a");
        async_fs::write(&file_b, b"same").await.expect("write b");

        let mut cache = FileCache::open(cache_dir.clone())
            .await
            .expect("open cache");
        cache
            .insert(&file_a, "c", "ref-a".to_string(), None)
            .await
            .expect("insert a");
        cache
            .insert(&file_b, "b::c", "ref-b".to_string(), None)
            .await
            .expect("insert b");
        cache.save().await.expect("save");

        let reloaded = FileCache::open(cache_dir).await.expect("reload");
        assert_eq!(
            reloaded
                .get(&file_a, "c")
                .await
                .expect("get a")
                .expect("entry a")
                .reference,
            "ref-a"
        );
        assert_eq!(
            reloaded
                .get(&file_b, "b::c")
                .await
                .expect("get b")
                .expect("entry b")
                .reference,
            "ref-b"
        );
    }

    #[tokio::test]
    async fn test_file_cache_save_is_atomic_without_leftover_temp_files() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let cache_dir = dir.path().join("cache");
        let file_path = dir.path().join("file.txt");
        async_fs::write(&file_path, b"hello")
            .await
            .expect("write file");

        let mut cache = FileCache::open(cache_dir.clone())
            .await
            .expect("open cache");
        cache
            .insert(&file_path, "openai", "file-1".to_string(), None)
            .await
            .expect("insert 1");
        cache.save().await.expect("save 1");

        cache
            .insert(&file_path, "openai", "file-2".to_string(), None)
            .await
            .expect("insert 2");
        cache.save().await.expect("save 2");

        let files: Vec<_> = std::fs::read_dir(&cache_dir)
            .expect("read cache dir")
            .map(|e| e.expect("entry").file_name())
            .collect();
        assert_eq!(files, vec![std::ffi::OsString::from(CACHE_FILE_NAME)]);

        let reloaded = FileCache::open(cache_dir).await.expect("reload");
        assert_eq!(
            reloaded
                .get(&file_path, "openai")
                .await
                .expect("get")
                .expect("entry")
                .reference,
            "file-2"
        );
    }
}
