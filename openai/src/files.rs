//! `OpenAI` Files API client for file uploads.
//!
//! This module provides a client for the `OpenAI` Files API, which allows uploading
//! files for use in various endpoints (vision, assistants, fine-tuning).
//!
//! See: <https://platform.openai.com/docs/api-reference/files>

use std::fmt::Write as _;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use url::Url;
use zenwave::{
    Client, client, header,
    multipart::{Multipart, MultipartPart},
};

use crate::error::OpenAIError;
#[cfg(not(target_arch = "wasm32"))]
use crate::mime::mime_from_path;

/// The purpose of the uploaded file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FilePurpose {
    /// For use with vision endpoints.
    Vision,
    /// For use with Assistants API.
    Assistants,
    /// For use with fine-tuning.
    FineTune,
    /// For use with batch requests.
    Batch,
}

impl FilePurpose {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Vision => "vision",
            Self::Assistants => "assistants",
            Self::FineTune => "fine-tune",
            Self::Batch => "batch",
        }
    }
}

/// Status of a file in the `OpenAI` system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileStatus {
    /// File is being processed.
    Uploaded,
    /// File is ready for use.
    Processed,
    /// File processing resulted in an error.
    Error,
    /// File is being deleted.
    Deleting,
    /// Unknown status.
    #[serde(other)]
    Unknown,
}

/// A file object from the `OpenAI` Files API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFile {
    /// File ID (e.g., "file-abc123").
    pub id: String,
    /// Object type (always "file").
    pub object: String,
    /// Size of the file in bytes.
    pub bytes: u64,
    /// Unix timestamp when the file was created.
    pub created_at: u64,
    /// Name of the file.
    pub filename: String,
    /// Purpose of the file.
    pub purpose: String,
    /// Current status.
    #[serde(default)]
    pub status: Option<FileStatus>,
    /// Error message if processing failed.
    #[serde(default)]
    pub status_details: Option<String>,
}

impl OpenAIFile {
    /// Get the creation time as `SystemTime`.
    #[must_use]
    pub fn created(&self) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(self.created_at)
    }

    /// Check if the file is ready for use.
    #[must_use]
    pub const fn is_ready(&self) -> bool {
        matches!(self.status, Some(FileStatus::Processed) | None)
    }
}

/// Response from deleting a file.
#[derive(Debug, Deserialize)]
pub struct DeleteFileResponse {
    /// File ID.
    pub id: String,
    /// Object type.
    pub object: String,
    /// Whether the deletion was successful.
    pub deleted: bool,
}

/// Response from listing files.
#[derive(Debug, Deserialize)]
pub struct ListFilesResponse {
    /// Object type (always "list").
    pub object: String,
    /// The files.
    pub data: Vec<OpenAIFile>,
}

/// Configuration for `OpenAI` Files API.
#[derive(Clone)]
pub struct FilesConfig {
    /// API key for authentication.
    pub api_key: String,
    /// Base URL for the API.
    pub base_url: String,
    /// Organization ID (optional).
    pub organization: Option<String>,
}

impl std::fmt::Debug for FilesConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilesConfig")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &self.base_url)
            .field("organization", &self.organization)
            .finish()
    }
}

impl FilesConfig {
    /// Create a new files configuration.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            organization: None,
        }
    }

    /// Set a custom base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the organization ID.
    #[must_use]
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    fn files_endpoint(&self) -> String {
        format!("{}/files", self.base_url.trim_end_matches('/'))
    }
}

pub(crate) fn csprng_multipart_boundary() -> Result<String, OpenAIError> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).map_err(|err| {
        OpenAIError::Api(format!(
            "failed to generate CSPRNG multipart boundary: {err}"
        ))
    })?;
    let mut hex = String::with_capacity(32);
    for byte in bytes {
        let _ = write!(&mut hex, "{byte:02x}");
    }
    Ok(format!("----aither-{hex}"))
}

pub(crate) fn encode_multipart_parts(
    parts: Vec<MultipartPart>,
) -> Result<(String, Vec<u8>), OpenAIError> {
    let mut multipart = Multipart::new().boundary(csprng_multipart_boundary()?);
    for part in parts {
        multipart.push(part);
    }
    Ok(multipart.encode())
}

fn validate_multipart_upload_metadata(file_name: &str, mime_type: &str) -> Result<(), OpenAIError> {
    if file_name.is_empty()
        || file_name
            .bytes()
            .any(|b| b.is_ascii_control() || matches!(b, b'"' | b'\\'))
    {
        return Err(OpenAIError::Api(
            "invalid upload file_name: must be non-empty without control characters, quotes, or backslashes".to_string(),
        ));
    }
    if mime_type.is_empty()
        || mime_type
            .bytes()
            .any(|b| b.is_ascii_control() || matches!(b, b'"' | b'\\'))
    {
        return Err(OpenAIError::Api(
            "invalid upload mime_type: must be non-empty without control characters, quotes, or backslashes".to_string(),
        ));
    }
    Ok(())
}

fn build_upload_multipart(
    file_name: &str,
    mime_type: &str,
    data: Vec<u8>,
    purpose: FilePurpose,
) -> Result<(String, Vec<u8>), OpenAIError> {
    validate_multipart_upload_metadata(file_name, mime_type)?;
    encode_multipart_parts(vec![
        MultipartPart::text("purpose", purpose.as_str()),
        MultipartPart::binary("file", file_name.to_owned(), mime_type.to_owned(), data),
    ])
}

/// Upload raw bytes to `OpenAI` Files API.
///
/// This is the cross-platform upload entry point (including wasm).
///
/// # Arguments
/// * `cfg` - Files configuration
/// * `file_name` - File name reported to `OpenAI`
/// * `mime_type` - MIME type for multipart metadata
/// * `data` - File content bytes
/// * `purpose` - Purpose of the file
///
/// # Errors
///
/// Returns an error if the upload request fails or the provider rejects
/// the file.
pub async fn upload_bytes(
    cfg: &FilesConfig,
    file_name: &str,
    mime_type: &str,
    data: Vec<u8>,
    purpose: FilePurpose,
) -> Result<OpenAIFile, OpenAIError> {
    let endpoint = cfg.files_endpoint();
    let (boundary, body) = build_upload_multipart(file_name, mime_type, data, purpose)?;
    let content_type = format!("multipart/form-data; boundary={boundary}");

    let mut backend = client();
    let mut builder = backend
        .post(&endpoint)
        .map_err(OpenAIError::from_http)?
        .header(header::CONTENT_TYPE.as_str(), &content_type)
        .map_err(OpenAIError::from_http)?
        .header(
            header::AUTHORIZATION.as_str(),
            format!("Bearer {}", cfg.api_key),
        )
        .map_err(OpenAIError::from_http)?;

    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org)
            .map_err(OpenAIError::from_http)?;
    }

    let builder = builder.bytes_body(body);

    builder.json().await.map_err(OpenAIError::from_http)
}

/// Upload a local file path to `OpenAI`.
///
/// This is a native-only convenience API. For wasm, use [`upload_bytes`].
///
/// # Errors
///
/// Returns an error when the file cannot be read or the upload request fails.
#[cfg(not(target_arch = "wasm32"))]
///
/// # Errors
///
/// Returns an error if the file cannot be read, the upload fails, or the
/// provider rejects the file.
pub async fn upload_file(
    cfg: &FilesConfig,
    path: &Path,
    purpose: FilePurpose,
) -> Result<OpenAIFile, OpenAIError> {
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let mime_type = mime_from_path(path).unwrap_or("application/octet-stream");
    let data = async_fs::read(path).await?;

    upload_bytes(cfg, file_name, mime_type, data, purpose).await
}

/// Delete a file from `OpenAI`.
///
/// # Arguments
/// * `cfg` - Files configuration
/// * `file_id` - ID of the file to delete
///
/// # Errors
///
/// Returns an error if the request fails, or the provider rejects it.
pub async fn delete_file(
    cfg: &FilesConfig,
    file_id: &str,
) -> Result<DeleteFileResponse, OpenAIError> {
    let endpoint = format!("{}/{}", cfg.files_endpoint(), file_id);

    let mut backend = client();
    let mut builder = backend
        .delete(&endpoint)
        .map_err(OpenAIError::from_http)?
        .header(
            header::AUTHORIZATION.as_str(),
            format!("Bearer {}", cfg.api_key),
        )
        .map_err(OpenAIError::from_http)?;

    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org)
            .map_err(OpenAIError::from_http)?;
    }

    builder.json().await.map_err(OpenAIError::from_http)
}

/// Get file metadata.
///
/// # Arguments
/// * `cfg` - Files configuration
/// * `file_id` - ID of the file
///
/// # Errors
///
/// Returns an error if the request fails, or the provider rejects it.
pub async fn get_file(cfg: &FilesConfig, file_id: &str) -> Result<OpenAIFile, OpenAIError> {
    let endpoint = format!("{}/{}", cfg.files_endpoint(), file_id);

    let mut backend = client();
    let mut builder = backend
        .get(&endpoint)
        .map_err(OpenAIError::from_http)?
        .header(
            header::AUTHORIZATION.as_str(),
            format!("Bearer {}", cfg.api_key),
        )
        .map_err(OpenAIError::from_http)?;

    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org)
            .map_err(OpenAIError::from_http)?;
    }

    builder.json().await.map_err(OpenAIError::from_http)
}

/// List uploaded files.
///
/// # Arguments
/// * `cfg` - Files configuration
/// * `purpose` - Optional filter by purpose
///
/// # Errors
///
/// Returns an error if the request fails, or the provider rejects it.
pub async fn list_files(
    cfg: &FilesConfig,
    purpose: Option<FilePurpose>,
) -> Result<ListFilesResponse, OpenAIError> {
    let mut url = Url::parse(&cfg.files_endpoint())
        .map_err(|err| OpenAIError::Api(format!("invalid files endpoint URL: {err}")))?;
    if let Some(p) = purpose {
        url.query_pairs_mut().append_pair("purpose", p.as_str());
    }
    let endpoint = url.to_string();

    let mut backend = client();
    let mut builder = backend
        .get(&endpoint)
        .map_err(OpenAIError::from_http)?
        .header(
            header::AUTHORIZATION.as_str(),
            format!("Bearer {}", cfg.api_key),
        )
        .map_err(OpenAIError::from_http)?;

    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org)
            .map_err(OpenAIError::from_http)?;
    }

    builder.json().await.map_err(OpenAIError::from_http)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mime_from_path() {
        assert_eq!(
            mime_from_path(Path::new("/path/to/image.png")),
            Some("image/png")
        );
        assert_eq!(
            mime_from_path(Path::new("/path/to/video.mp4")),
            Some("video/mp4")
        );
        assert_eq!(
            mime_from_path(Path::new("/path/to/doc.pdf")),
            Some("application/pdf")
        );
    }

    #[test]
    fn test_file_purpose_as_str() {
        assert_eq!(FilePurpose::Vision.as_str(), "vision");
        assert_eq!(FilePurpose::Assistants.as_str(), "assistants");
        assert_eq!(FilePurpose::FineTune.as_str(), "fine-tune");
        assert_eq!(FilePurpose::Batch.as_str(), "batch");
    }

    #[test]
    fn upload_multipart_rejects_header_injection_and_encodes_clean_parts() {
        assert!(
            build_upload_multipart(
                "bad\r\nX-Inject: 1.png",
                "image/png",
                vec![1],
                FilePurpose::Vision
            )
            .is_err()
        );
        assert!(
            build_upload_multipart("bad\"quote.png", "image/png", vec![1], FilePurpose::Vision)
                .is_err()
        );
        assert!(
            build_upload_multipart(
                "ok.png",
                "image/png\r\nX-Inject: 1",
                vec![1],
                FilePurpose::Vision
            )
            .is_err()
        );

        let (b1, body) = build_upload_multipart(
            "report.pdf",
            "application/pdf",
            b"hello".to_vec(),
            FilePurpose::Assistants,
        )
        .expect("valid multipart upload");
        let b2 = csprng_multipart_boundary().expect("second boundary");
        assert_ne!(b1, b2);
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("name=\"purpose\""));
        assert!(text.contains("assistants"));
        assert!(text.contains("filename=\"report.pdf\""));
        assert!(text.contains("Content-Type: application/pdf"));
    }

    #[test]
    fn files_config_debug_redacts_api_key() {
        let cfg = FilesConfig::new("sk-secret-files-key");
        let rendered = format!("{cfg:?}");
        assert!(!rendered.contains("sk-secret-files-key"));
        assert!(rendered.contains("[REDACTED]"));
    }
}
