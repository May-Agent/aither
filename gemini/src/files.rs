//! Gemini Files API client for file uploads.
//!
//! This module provides a client for the Gemini Files API, which allows uploading
//! files for use in multimodal requests. Files are stored for 48 hours before expiration.
//!
//! See: <https://ai.google.dev/api/files>

use std::future::IntoFuture as _;
use std::time::SystemTime;

#[cfg(not(target_arch = "wasm32"))]
use async_fs;
use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use url::Url;
use zenwave::{Client, client, header};

use crate::client::within_idle;
use crate::config::{AuthMode, GeminiConfig, USER_AGENT};
use crate::error::GeminiError;

/// Gemini file state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FileState {
    /// File is being processed.
    Processing,
    /// File is ready for use.
    Active,
    /// File processing failed.
    Failed,
    /// Unknown state.
    #[serde(other)]
    Unknown,
}

/// A file uploaded to the Gemini Files API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFile {
    /// File resource name (e.g., "files/abc123").
    pub name: String,
    /// Display name.
    #[serde(default)]
    pub display_name: String,
    /// MIME type.
    pub mime_type: String,
    /// File size in bytes.
    #[serde(default)]
    pub size_bytes: String,
    /// File creation time.
    #[serde(default)]
    pub create_time: String,
    /// File update time.
    #[serde(default)]
    pub update_time: String,
    /// Expiration time (RFC 3339 format).
    #[serde(default)]
    pub expiration_time: String,
    /// URI to use in requests.
    #[serde(default)]
    pub uri: String,
    /// Current file state.
    pub state: FileState,
    /// SHA-256 hash of the file.
    #[serde(default)]
    pub sha256_hash: String,
}

impl GeminiFile {
    /// Parse the expiration time to `SystemTime`.
    ///
    /// Returns `None` if parsing fails.
    #[must_use]
    pub fn expiration(&self) -> Option<SystemTime> {
        parse_rfc3339(&self.expiration_time)
    }

    /// Check if the file is ready for use.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.state == FileState::Active
    }
}

/// Response wrapper for file upload.
#[derive(Debug, Deserialize)]
pub struct UploadFileResponse {
    /// The uploaded file.
    pub file: GeminiFile,
}

/// Response wrapper for file list.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListFilesResponse {
    /// The files.
    #[serde(default)]
    pub files: Vec<GeminiFile>,
    /// Next page token.
    #[serde(default)]
    pub next_page_token: Option<String>,
}

fn csprng_boundary() -> Result<String, GeminiError> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).map_err(|err| {
        GeminiError::Api(format!(
            "failed to generate CSPRNG multipart boundary: {err}"
        ))
    })?;
    let rand_u128 = u128::from_be_bytes(bytes);
    Ok(format!("----aither-{rand_u128:032x}"))
}

fn build_related_multipart(
    file_name: &str,
    mime_type: &str,
    data: &[u8],
) -> Result<(String, Vec<u8>), GeminiError> {
    if file_name.is_empty() || file_name.bytes().any(|b| b.is_ascii_control()) {
        return Err(GeminiError::Api(
            "invalid upload file_name: must be non-empty without control characters".to_string(),
        ));
    }
    if mime_type.is_empty()
        || mime_type
            .bytes()
            .any(|b| b.is_ascii_control() || matches!(b, b'"' | b'\\'))
    {
        return Err(GeminiError::Api(
            "invalid upload mime_type: must be non-empty without control characters, quotes, or backslashes".to_string(),
        ));
    }

    let boundary = csprng_boundary()?;
    let metadata = serde_json::json!({
        "file": {
            "displayName": file_name
        }
    });
    let metadata_json = serde_json::to_string(&metadata).map_err(GeminiError::Json)?;

    let mut body = Vec::with_capacity(metadata_json.len() + data.len() + 256);
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Type: application/json; charset=UTF-8\r\n\r\n");
    body.extend_from_slice(metadata_json.as_bytes());
    body.extend_from_slice(b"\r\n");

    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(format!("Content-Type: {mime_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(data);
    body.extend_from_slice(b"\r\n");

    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    Ok((boundary, body))
}

/// Upload raw bytes to the Gemini Files API.
///
/// This is the cross-platform upload entry point (including wasm).
///
/// # Errors
///
/// Returns an error if the upload request fails or the provider rejects
/// the file.
pub async fn upload_bytes(
    cfg: &GeminiConfig,
    file_name: &str,
    mime_type: &str,
    data: Vec<u8>,
) -> Result<GeminiFile, GeminiError> {
    let upload_url = build_upload_url(cfg)?;
    let (boundary, body) = build_related_multipart(file_name, mime_type, &data)?;

    let mut backend = client();
    let mut builder = backend
        .post(&upload_url)
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?
        .header("X-Goog-Upload-Protocol", "multipart")
        .map_err(GeminiError::from_http)?;

    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }

    let content_type = format!("multipart/related; boundary={boundary}");
    let builder = builder
        .header(header::CONTENT_TYPE.as_str(), &content_type)
        .map_err(GeminiError::from_http)?
        .bytes_body(body);

    let response: UploadFileResponse = within_idle(cfg.idle_timeout, builder.json())
        .await?
        .map_err(GeminiError::from_http)?;

    Ok(response.file)
}

/// Upload a local file path to the Gemini Files API.
///
/// This is a native-only convenience API. For wasm, use [`upload_bytes`].
///
/// # Errors
///
/// Returns an error when the file cannot be read or upload fails.
#[cfg(not(target_arch = "wasm32"))]
///
/// # Errors
///
/// Returns an error if the file cannot be read, the upload fails, or the
/// provider rejects the file.
pub async fn upload_file(cfg: &GeminiConfig, path: &Path) -> Result<GeminiFile, GeminiError> {
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let mime_type = crate::mime::mime_from_path(path).unwrap_or("application/octet-stream");
    let data = async_fs::read(path).await?;

    upload_bytes(cfg, file_name, mime_type, data).await
}

/// Delete a file from the Gemini Files API.
///
/// # Arguments
/// * `cfg` - Gemini configuration
/// * `name` - File resource name (e.g., "files/abc123")
///
/// # Errors
///
/// Returns an error if the request fails, or the provider rejects it.
pub async fn delete_file(cfg: &GeminiConfig, name: &str) -> Result<(), GeminiError> {
    let url = cfg.endpoint(name);

    let mut backend = client();
    let mut builder = backend
        .delete(&url)
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?;

    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }

    let _response = within_idle(cfg.idle_timeout, builder.into_future())
        .await?
        .map_err(GeminiError::from_http)?;
    Ok(())
}

/// Get file metadata.
///
/// # Arguments
/// * `cfg` - Gemini configuration
/// * `name` - File resource name (e.g., "files/abc123")
///
/// # Errors
///
/// Returns an error if the request fails, or the provider rejects it.
pub async fn get_file(cfg: &GeminiConfig, name: &str) -> Result<GeminiFile, GeminiError> {
    let url = cfg.endpoint(name);

    let mut backend = client();
    let mut builder = backend
        .get(&url)
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?;

    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }

    within_idle(cfg.idle_timeout, builder.json())
        .await?
        .map_err(GeminiError::from_http)
}

fn build_list_files_url(
    cfg: &GeminiConfig,
    page_size: Option<u32>,
    page_token: Option<&str>,
) -> Result<String, GeminiError> {
    let mut url = Url::parse(&cfg.endpoint("files"))
        .map_err(|err| GeminiError::Api(format!("invalid Gemini files endpoint URL: {err}")))?;
    if page_size.is_some() || page_token.is_some() {
        let mut pairs = url.query_pairs_mut();
        if let Some(size) = page_size {
            pairs.append_pair("pageSize", &size.to_string());
        }
        if let Some(token) = page_token {
            pairs.append_pair("pageToken", token);
        }
    }
    Ok(url.to_string())
}

/// List uploaded files.
///
/// # Arguments
/// * `cfg` - Gemini configuration
/// * `page_size` - Maximum number of files to return
/// * `page_token` - Page token from previous response
///
/// # Errors
///
/// Returns an error if the request fails, or the provider rejects it.
pub async fn list_files(
    cfg: &GeminiConfig,
    page_size: Option<u32>,
    page_token: Option<&str>,
) -> Result<ListFilesResponse, GeminiError> {
    let url = build_list_files_url(cfg, page_size, page_token)?;

    let mut backend = client();
    let mut builder = backend
        .get(&url)
        .map_err(GeminiError::from_http)?
        .header(header::USER_AGENT.as_str(), USER_AGENT)
        .map_err(GeminiError::from_http)?;

    if cfg.auth == AuthMode::Header {
        builder = builder
            .header("x-goog-api-key", cfg.api_key.clone())
            .map_err(GeminiError::from_http)?;
    }

    within_idle(cfg.idle_timeout, builder.json())
        .await?
        .map_err(GeminiError::from_http)
}

/// Build the upload URL from `cfg.base_url` and `cfg.auth`.
fn build_upload_url(cfg: &GeminiConfig) -> Result<String, GeminiError> {
    let mut url = Url::parse(&cfg.base_url)
        .map_err(|err| GeminiError::Api(format!("invalid Gemini base_url: {err}")))?;
    let trimmed = url.path().trim_end_matches('/').to_string();
    let upload_path = if trimmed.is_empty() {
        "/upload/v1beta/files".to_string()
    } else if trimmed.starts_with("/upload/") {
        format!("{trimmed}/files")
    } else if let Some(pos) = trimmed.rfind("/v1") {
        let (prefix, version) = trimmed.split_at(pos);
        format!("{prefix}/upload{version}/files")
    } else {
        format!("/upload{trimmed}/files")
    };
    url.set_path(&upload_path);
    if cfg.auth == AuthMode::Query {
        url.query_pairs_mut().append_pair("key", &cfg.api_key);
    }
    Ok(url.to_string())
}

/// Parse an RFC 3339 timestamp to `SystemTime`.
///
/// Returns `None` for an empty or malformed timestamp — including the empty
/// string the API sends for a file that never expires.
fn parse_rfc3339(s: &str) -> Option<SystemTime> {
    let timestamp: jiff::Timestamp = s.trim().parse().ok()?;
    Some(SystemTime::from(timestamp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rfc3339_expiration() {
        // 2024-01-15T10:30:00Z is 1_705_314_600 seconds after the epoch.
        let parsed = parse_rfc3339("2024-01-15T10:30:00Z").expect("should parse");
        let secs = parsed
            .duration_since(std::time::UNIX_EPOCH)
            .expect("after the epoch")
            .as_secs();
        assert_eq!(secs, 1_705_314_600);

        // Fractional seconds and a numeric offset are both valid RFC 3339.
        assert!(parse_rfc3339("2024-01-15T10:30:00.123456Z").is_some());
        assert!(parse_rfc3339("  2024-01-15T11:30:00+01:00  ").is_some());

        // What the API sends when a file never expires, and outright garbage.
        assert!(parse_rfc3339("").is_none());
        assert!(parse_rfc3339("not a timestamp").is_none());
    }

    #[test]
    fn test_mime_from_path() {
        assert_eq!(
            crate::mime::mime_from_path(Path::new("/path/to/image.png")),
            Some("image/png")
        );
        assert_eq!(
            crate::mime::mime_from_path(Path::new("/path/to/video.mp4")),
            Some("video/mp4")
        );
        assert_eq!(
            crate::mime::mime_from_path(Path::new("/path/to/audio.mp3")),
            Some("audio/mpeg")
        );
        assert_eq!(
            crate::mime::mime_from_path(Path::new("/path/to/doc.pdf")),
            Some("application/pdf")
        );
        assert_eq!(
            crate::mime::mime_from_path(Path::new("/path/to/unknown.aither_unknown")),
            None
        );
    }

    #[test]
    fn upload_and_list_urls_respect_base_url_and_auth_modes() {
        let header_client = crate::Gemini::new("sec+key");
        assert_eq!(
            build_upload_url(header_client.config()).expect("upload url"),
            "https://generativelanguage.googleapis.com/upload/v1beta/files"
        );
        assert_eq!(
            build_list_files_url(header_client.config(), Some(25), Some("next+tok"))
                .expect("list url"),
            "https://generativelanguage.googleapis.com/v1beta/files?pageSize=25&pageToken=next%2Btok"
        );
        assert_eq!(
            build_list_files_url(header_client.config(), None, None).expect("bare list url"),
            "https://generativelanguage.googleapis.com/v1beta/files"
        );

        let proxy_query = crate::Gemini::new("sec+key")
            .with_base_url("http://127.0.0.1:9090/gemini/v1beta")
            .with_auth_mode(AuthMode::Query);
        assert_eq!(
            build_upload_url(proxy_query.config()).expect("proxy upload url"),
            "http://127.0.0.1:9090/gemini/upload/v1beta/files?key=sec%2Bkey"
        );
        assert_eq!(
            build_list_files_url(proxy_query.config(), Some(10), None).expect("proxy list url"),
            "http://127.0.0.1:9090/gemini/v1beta/files?key=sec%2Bkey&pageSize=10"
        );
    }

    #[test]
    fn related_multipart_rejects_header_injection() {
        assert!(build_related_multipart("bad\r\nname.pdf", "application/pdf", b"x").is_err());
        assert!(build_related_multipart("ok.pdf", "application/pdf\r\nX-Inject: 1", b"x").is_err());
        assert!(build_related_multipart("ok.pdf", "application/\"pdf\"", b"x").is_err());

        let (b1, body) =
            build_related_multipart("doc.pdf", "application/pdf", b"hello").expect("valid");
        let b2 = csprng_boundary().expect("second boundary");
        assert_ne!(b1, b2);
        let rendered = String::from_utf8_lossy(&body);
        assert!(rendered.contains("\"displayName\":\"doc.pdf\""));
        assert!(rendered.contains("Content-Type: application/pdf"));
    }
}
