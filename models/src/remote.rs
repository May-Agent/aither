//! Feature-gated async fetch for `LiteLLM` model data.
//!
//! Enable with `features = ["remote"]`.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::convert;
use crate::registry::ModelRegistry;

const LITELLM_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

const CACHE_MAX_AGE_SECS: u64 = 24 * 60 * 60;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Fetch the latest `LiteLLM` model data and build a registry.
///
/// If `cache_path` is provided and the file is less than 24 hours old,
/// the cached version is used instead of fetching.
///
/// # Errors
///
/// Returns an error if the network request fails, returns a non-success HTTP
/// status, produces invalid JSON, or fails cache I/O.
pub async fn fetch_litellm(cache_path: Option<&Path>) -> Result<ModelRegistry, FetchError> {
    fetch_litellm_from_url(LITELLM_URL, cache_path).await
}

async fn fetch_litellm_from_url(
    url: &str,
    cache_path: Option<&Path>,
) -> Result<ModelRegistry, FetchError> {
    let bytes = fetch_with_cache(url, cache_path).await?;
    Ok(ModelRegistry::try_from_litellm_json(&bytes)?)
}

/// Error type for remote fetch operations.
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    /// Network request failed.
    #[error("network error: {0}")]
    Network(#[source] Box<zenwave::Error>),
    /// Remote server returned a non-success HTTP status.
    #[error("unexpected HTTP status: {status}")]
    HttpStatus {
        /// HTTP status code returned by the remote server.
        status: u16,
    },
    /// Body read error.
    #[error("body read error: {0}")]
    Body(#[from] zenwave::BodyError),
    /// Response or cached payload was not valid `LiteLLM` JSON.
    #[error("invalid LiteLLM JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// I/O error reading/writing cache.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<zenwave::Error> for FetchError {
    fn from(e: zenwave::Error) -> Self {
        if let zenwave::Error::Http { status, .. } = e {
            return Self::HttpStatus {
                status: status.as_u16(),
            };
        }
        Self::Network(Box::new(e))
    }
}

async fn fetch_with_cache(url: &str, cache_path: Option<&Path>) -> Result<Vec<u8>, FetchError> {
    // Check cache freshness
    if let Some(path) = cache_path
        && let Ok(meta) = async_fs::metadata(path).await
        && let Ok(modified) = meta.modified()
    {
        let age = modified.elapsed().unwrap_or_default();
        if age.as_secs() < CACHE_MAX_AGE_SECS {
            return Ok(async_fs::read(path).await?);
        }
    }

    // Fetch from network
    let response = zenwave::get(url).await?;
    let status = response.status();
    if !status.is_success() {
        return Err(FetchError::HttpStatus {
            status: status.as_u16(),
        });
    }
    let bytes = response.into_body().into_bytes().await?.to_vec();

    // Validate JSON before persisting so error pages or partial payloads never
    // poison the on-disk cache.
    let _ = convert::try_parse_litellm_json(&bytes)?;

    // Write cache atomically
    if let Some(path) = cache_path {
        write_file_atomic(path, &bytes).await?;
    }

    Ok(bytes)
}

async fn write_file_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        async_fs::create_dir_all(parent).await?;
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("litellm_cache");
    let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_path = parent.join(format!(".{file_name}.tmp.{}.{seq}", std::process::id()));

    if let Err(err) = async_fs::write(&tmp_path, bytes).await {
        let _ = async_fs::remove_file(&tmp_path).await;
        return Err(err);
    }
    if let Err(err) = async_fs::rename(&tmp_path, path).await {
        let _ = async_fs::remove_file(&tmp_path).await;
        return Err(err);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    fn spawn_single_response_server(
        status_line: &'static str,
        body: &'static str,
    ) -> (String, Arc<AtomicUsize>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().expect("local addr");
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_clone = Arc::clone(&hits);

        let handle = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                hits_clone.fetch_add(1, Ordering::SeqCst);
                let mut req_buf = [0u8; 1024];
                let _ = stream.read(&mut req_buf);
                let response = format!(
                    "HTTP/1.1 {status_line}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });

        (format!("http://{addr}/models.json"), hits, handle)
    }

    #[tokio::test]
    async fn http_error_is_rejected_and_not_cached() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_path = dir.path().join("models.json");
        let (url, _hits, handle) =
            spawn_single_response_server("500 Internal Server Error", "upstream error");

        let err = fetch_litellm_from_url(&url, Some(&cache_path))
            .await
            .expect_err("500 status must fail");
        handle.join().expect("server thread");

        assert!(
            matches!(err, FetchError::HttpStatus { status: 500 }),
            "unexpected error: {err:?}"
        );
        assert!(!cache_path.exists(), "error page must not be cached");
    }

    #[tokio::test]
    async fn invalid_json_is_rejected_and_not_cached() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_path = dir.path().join("models.json");
        let (url, _hits, handle) = spawn_single_response_server("200 OK", "<html>not json</html>");

        let err = fetch_litellm_from_url(&url, Some(&cache_path))
            .await
            .expect_err("malformed JSON must fail");
        handle.join().expect("server thread");

        assert!(matches!(err, FetchError::Json(_)));
        assert!(!cache_path.exists(), "malformed payload must not be cached");
    }

    #[tokio::test]
    async fn valid_payload_writes_cache_atomically_and_reuses_fresh_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_path = dir.path().join("nested/models.json");
        let payload = r#"{"gpt-test":{"litellm_provider":"openai","mode":"chat","input_cost_per_token":1e-6,"output_cost_per_token":2e-6}}"#;
        let (url, hits, handle) = spawn_single_response_server("200 OK", payload);

        let registry = fetch_litellm_from_url(&url, Some(&cache_path))
            .await
            .expect("valid fetch");
        handle.join().expect("server thread");
        assert!(registry.lookup("gpt-test").is_some());
        assert!(cache_path.exists());

        // No temporary files should remain next to the cache file.
        let entries: Vec<_> = std::fs::read_dir(cache_path.parent().expect("parent"))
            .expect("read cache dir")
            .map(|e| e.expect("dir entry").file_name())
            .collect();
        assert_eq!(entries, vec![std::ffi::OsString::from("models.json")]);

        // Subsequent fetch should read from fresh cache without contacting server.
        let cached = fetch_litellm_from_url(&url, Some(&cache_path))
            .await
            .expect("cached fetch");
        assert!(cached.lookup("gpt-test").is_some());
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn corrupt_fresh_cache_returns_json_error_without_panicking() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_path = dir.path().join("models.json");
        std::fs::write(&cache_path, b"not json").expect("write corrupt cache");

        let err = fetch_litellm_from_url("http://127.0.0.1:1/unused", Some(&cache_path))
            .await
            .expect_err("corrupt cache must return error");
        assert!(matches!(err, FetchError::Json(_)));
    }
}
