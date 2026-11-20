//! Local ONNX Runtime embedding models for aither.
//!
//! This crate provides [`OrtEmbedding`], an implementation of [`aither_core::EmbeddingModel`]
//! that runs ONNX embedding models locally using ONNX Runtime.
//!
//! # Features
//!
//! - **No auto-download**: You provide the model and tokenizer paths
//! - **Auto-detect dimension**: Embedding dimension is detected from model outputs
//! - **Multiple pooling strategies**: `LastToken`, `Mean`, `Cls`
//! - **GPU acceleration**: CUDA and `CoreML` enabled by default
//!
//! # Example
//!
//! ```rust,no_run
//! use aither_ort::{OrtEmbedding, PoolingStrategy};
//! use aither_core::EmbeddingModel;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load from model directory (auto-finds model.onnx and tokenizer.json)
//! let embedder = OrtEmbedding::from_directory("./models/Qwen3-Embedding-0.6B-ONNX")?;
//! println!("Embedding dimension: {}", embedder.dim());
//!
//! // Generate embeddings
//! let embedding = embedder.embed("Hello, world!").await?;
//! # Ok(())
//! # }
//! ```

mod error;
mod pooling;

pub use error::OrtError;
pub use pooling::PoolingStrategy;

use async_channel::Sender;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use aither_core::EmbeddingModel;
use ndarray::{ArrayD, Ix2, Ix3};
use ort::session::{Session, builder::GraphOptimizationLevel};
use tokenizers::Tokenizer;

/// Bounded queue depth for pending embedding requests sent to the dedicated ONNX worker thread.
const INFERENCE_QUEUE_CAPACITY: usize = 64;

/// An embedding model backed by ONNX Runtime.
///
/// This struct wraps an ONNX model session and tokenizer to provide
/// text embedding functionality. It implements [`EmbeddingModel`] for
/// seamless integration with other aither components.
///
/// # Example
///
/// ```rust,no_run
/// use aither_ort::OrtEmbedding;
///
/// // Simple: load from directory
/// let embedder = OrtEmbedding::from_directory("./model")?;
///
/// // Custom: use builder
/// let embedder = OrtEmbedding::builder()
///     .model_path("./model/model.onnx")
///     .tokenizer_path("./model/tokenizer.json")
///     .build()?;
/// # Ok::<(), aither_ort::OrtError>(())
/// ```
pub struct OrtEmbedding {
    inference_tx: Sender<InferenceRequest>,
    tokenizer: Arc<Tokenizer>,
    dimension: usize,
    pooling: PoolingStrategy,
    normalize: bool,
}

struct InferenceRequest {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    response_tx: async_channel::Sender<Result<ArrayD<f32>, OrtError>>,
}

impl std::fmt::Debug for OrtEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtEmbedding")
            .field("dimension", &self.dimension)
            .field("pooling", &self.pooling)
            .field("normalize", &self.normalize)
            .finish_non_exhaustive()
    }
}

impl OrtEmbedding {
    /// Load an embedding model from a directory.
    ///
    /// This method automatically locates `model.onnx` (or files in `onnx/` subdirectory)
    /// and `tokenizer.json` within the specified directory.
    ///
    /// # Arguments
    /// * `path` - Path to the model directory
    ///
    /// # Errors
    /// Returns an error if the model or tokenizer cannot be found or loaded.
    pub fn from_directory(path: impl AsRef<Path>) -> Result<Self, OrtError> {
        let dir = path.as_ref();

        // Find model file
        let model_path = find_model_file(dir)?;

        // Find tokenizer file
        let tokenizer_path = find_tokenizer_file(dir)?;

        Self::builder()
            .model_path(model_path)
            .tokenizer_path(tokenizer_path)
            .build()
    }

    /// Create a builder for custom configuration.
    #[must_use]
    pub fn builder() -> OrtEmbeddingBuilder {
        OrtEmbeddingBuilder::default()
    }

    /// Returns the embedding dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the pooling strategy.
    #[must_use]
    pub const fn pooling(&self) -> PoolingStrategy {
        self.pooling
    }

    /// Returns whether L2 normalization is enabled.
    #[must_use]
    pub const fn normalize(&self) -> bool {
        self.normalize
    }
}

struct EncodedTokens {
    input_ids: Vec<i64>,
    onnx_mask: Vec<i64>,
    pooling_mask: Vec<u32>,
}

impl EmbeddingModel for OrtEmbedding {
    type Error = OrtError;

    fn dim(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, Self::Error> {
        let tokenizer = Arc::clone(&self.tokenizer);
        let owned_text = text.to_owned();
        let encoded = blocking::unblock(move || -> Result<EncodedTokens, OrtError> {
            let encoding = tokenizer
                .encode(owned_text.as_str(), true)
                .map_err(|e| OrtError::Tokenization(e.to_string()))?;
            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
            let pooling_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
            let onnx_mask: Vec<i64> = pooling_mask.iter().map(|&m| i64::from(m)).collect();
            Ok(EncodedTokens {
                input_ids,
                onnx_mask,
                pooling_mask,
            })
        })
        .await?;

        if encoded.input_ids.is_empty() {
            return Err(OrtError::Tokenization(
                "tokenizer produced empty token sequence".to_string(),
            ));
        }

        let (response_tx, response_rx) = async_channel::bounded(1);
        self.inference_tx
            .send(InferenceRequest {
                input_ids: encoded.input_ids,
                attention_mask: encoded.onnx_mask,
                response_tx,
            })
            .await
            .map_err(|_| OrtError::Worker("inference worker is unavailable".to_string()))?;
        let hidden_states_owned = response_rx
            .recv()
            .await
            .map_err(|_| OrtError::Worker("inference worker dropped response".to_string()))??;

        let embedding = extract_embedding(
            hidden_states_owned,
            &encoded.pooling_mask,
            self.pooling,
            self.normalize,
        )?;
        Ok(embedding)
    }
}

fn extract_embedding(
    hidden_states_owned: ArrayD<f32>,
    attention_mask: &[u32],
    pooling: PoolingStrategy,
    normalize: bool,
) -> Result<Vec<f32>, OrtError> {
    let rank = hidden_states_owned.shape().len();
    let mut embedding = match rank {
        // Rank-2 `[batch, hidden_dim]` outputs are already sentence-pooled by the ONNX graph.
        2 => {
            let pooled = hidden_states_owned
                .into_dimensionality::<Ix2>()
                .map_err(|e| OrtError::Shape(e.to_string()))?;
            if pooled.shape()[0] == 0 {
                return Err(OrtError::Shape(
                    "empty batch dimension in rank-2 output".to_string(),
                ));
            }
            pooled.row(0).to_vec()
        }
        3 => {
            let hidden_states_3d = hidden_states_owned
                .into_dimensionality::<Ix3>()
                .map_err(|e| OrtError::Shape(e.to_string()))?;
            if hidden_states_3d.shape()[0] == 0 || hidden_states_3d.shape()[1] == 0 {
                return Err(OrtError::Shape(
                    "empty batch or sequence dimension in rank-3 output".to_string(),
                ));
            }
            pooling.apply(&hidden_states_3d.view(), attention_mask)
        }
        _ => return Err(OrtError::InvalidOutputShape(rank)),
    };

    if normalize {
        l2_normalize(&mut embedding);
    }

    Ok(embedding)
}

/// Builder for [`OrtEmbedding`].
#[derive(Debug)]
pub struct OrtEmbeddingBuilder {
    model_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    pooling: PoolingStrategy,
    normalize: bool,
}

impl Default for OrtEmbeddingBuilder {
    fn default() -> Self {
        Self {
            model_path: None,
            tokenizer_path: None,
            pooling: PoolingStrategy::default(),
            normalize: true,
        }
    }
}

impl OrtEmbeddingBuilder {
    /// Set the path to the ONNX model file.
    #[must_use]
    pub fn model_path(mut self, path: impl AsRef<Path>) -> Self {
        self.model_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the path to the tokenizer.json file.
    #[must_use]
    pub fn tokenizer_path(mut self, path: impl AsRef<Path>) -> Self {
        self.tokenizer_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the pooling strategy.
    ///
    /// Default: [`PoolingStrategy::LastToken`] (for Qwen3-style models)
    #[must_use]
    pub const fn pooling(mut self, strategy: PoolingStrategy) -> Self {
        self.pooling = strategy;
        self
    }

    /// Enable or disable L2 normalization.
    ///
    /// Default: `true`
    #[must_use]
    pub const fn normalize(mut self, enabled: bool) -> Self {
        self.normalize = enabled;
        self
    }

    /// Build the [`OrtEmbedding`] instance.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Model path is not specified
    /// - Model file cannot be loaded
    /// - Tokenizer file cannot be loaded
    /// - Model output dimension cannot be determined
    pub fn build(self) -> Result<OrtEmbedding, OrtError> {
        let model_path = self.model_path.ok_or(OrtError::MissingModelPath)?;

        if !model_path.exists() {
            return Err(OrtError::ModelNotFound(model_path));
        }

        // Load tokenizer
        let tokenizer_path = self.tokenizer_path.ok_or_else(|| {
            // Try to find tokenizer relative to model
            OrtError::TokenizerNotFound(model_path.parent().unwrap_or(&model_path).to_path_buf())
        })?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| OrtError::tokenizer(&tokenizer_path, e.to_string()))?;

        // Load ONNX session with optimizations
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus())?
            .commit_from_file(&model_path)?;

        // Auto-detect dimension from model outputs
        let dimension = detect_embedding_dimension(&session)?;
        let inference_tx = spawn_inference_worker(session)?;

        Ok(OrtEmbedding {
            inference_tx,
            tokenizer: Arc::new(tokenizer),
            dimension,
            pooling: self.pooling,
            normalize: self.normalize,
        })
    }
}

/// L2 normalize a vector in place.
fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
}

/// Find the ONNX model file in a directory.
fn find_model_file(dir: &Path) -> Result<PathBuf, OrtError> {
    // Check common locations
    let candidates = [
        dir.join("model.onnx"),
        dir.join("onnx/model.onnx"),
        dir.join("onnx/model_fp32.onnx"),
        dir.join("onnx/model_fp16.onnx"),
        dir.join("onnx/model_q8.onnx"),
    ];

    for candidate in &candidates {
        if candidate.exists() {
            return Ok(candidate.clone());
        }
    }

    // Look for any .onnx file
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "onnx") {
                return Ok(path);
            }
        }
    }

    // Check onnx subdirectory
    let onnx_dir = dir.join("onnx");
    if onnx_dir.exists()
        && let Ok(entries) = std::fs::read_dir(&onnx_dir)
    {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "onnx") {
                return Ok(path);
            }
        }
    }

    Err(OrtError::ModelNotFound(dir.to_path_buf()))
}

/// Find the tokenizer.json file in a directory.
fn find_tokenizer_file(dir: &Path) -> Result<PathBuf, OrtError> {
    let candidates = [dir.join("tokenizer.json"), dir.join("onnx/tokenizer.json")];

    for candidate in &candidates {
        if candidate.exists() {
            return Ok(candidate.clone());
        }
    }

    Err(OrtError::TokenizerNotFound(dir.to_path_buf()))
}

/// Detect the embedding dimension from model output metadata.
fn detect_embedding_dimension(session: &Session) -> Result<usize, OrtError> {
    // Look for the output that contains hidden states
    for output in session.outputs() {
        // Get tensor type info if available
        if let ort::value::ValueType::Tensor { shape, .. } = output.dtype() {
            // Expect shape [batch, seq_len, hidden_dim] or [batch, hidden_dim]
            if shape.len() >= 2 {
                // Last dimension is typically the hidden dimension
                if let Some(&dim) = shape.last()
                    && dim > 0
                {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    return Ok(dim as usize);
                }
            }
        }
    }

    // Fallback: common dimensions
    // This shouldn't happen with well-formed models
    Err(OrtError::InvalidOutputShape(0))
}

/// Get number of CPU cores for parallelism.
fn num_cpus() -> usize {
    std::thread::available_parallelism().map_or(4, std::num::NonZero::get)
}

fn spawn_inference_worker(session: Session) -> Result<Sender<InferenceRequest>, OrtError> {
    let (tx, rx) = async_channel::bounded::<InferenceRequest>(INFERENCE_QUEUE_CAPACITY);
    std::thread::Builder::new()
        .name("ort-embedding-worker".to_string())
        .spawn(move || run_inference_worker(session, &rx))
        .map_err(|error| OrtError::Worker(error.to_string()))?;
    Ok(tx)
}

fn run_inference_worker(session: Session, rx: &async_channel::Receiver<InferenceRequest>) {
    let mut session = session;
    while let Ok(request) = rx.recv_blocking() {
        let result = run_inference(&mut session, request.input_ids, request.attention_mask);
        let _ = request.response_tx.send_blocking(result);
    }
}

fn run_inference(
    session: &mut Session,
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
) -> Result<ArrayD<f32>, OrtError> {
    let seq_len = input_ids.len();
    let input_ids_tensor =
        ort::value::Tensor::from_array(([1, seq_len], input_ids.into_boxed_slice()))
            .map_err(OrtError::from)?;
    let attention_mask_tensor =
        ort::value::Tensor::from_array(([1, seq_len], attention_mask.into_boxed_slice()))
            .map_err(OrtError::from)?;

    let outputs = session
        .run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ])
        .map_err(OrtError::from)?;

    let hidden_states = outputs
        .get("last_hidden_state")
        .or_else(|| outputs.get("hidden_states"))
        .or_else(|| outputs.get("sentence_embedding"))
        .or_else(|| outputs.get("output"))
        .ok_or(OrtError::InvalidOutputShape(0))?;

    hidden_states
        .try_extract_array::<f32>()
        .map(|view| view.to_owned())
        .map_err(OrtError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_model_path() {
        let result = OrtEmbeddingBuilder::default().build();
        assert!(matches!(result, Err(OrtError::MissingModelPath)));
    }

    #[test]
    fn builder_validates_model_exists() {
        let result = OrtEmbeddingBuilder::default()
            .model_path("/nonexistent/model.onnx")
            .tokenizer_path("/nonexistent/tokenizer.json")
            .build();
        assert!(matches!(result, Err(OrtError::ModelNotFound(_))));
    }

    #[test]
    fn l2_normalize_works() {
        let mut vec = vec![3.0, 4.0];
        l2_normalize(&mut vec);
        assert!((vec[0] - 0.6).abs() < 1e-6);
        assert!((vec[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let mut vec = vec![0.0, 0.0];
        l2_normalize(&mut vec);
        assert_eq!(vec, vec![0.0, 0.0]);
    }

    #[test]
    fn builder_defaults_normalize_to_true() {
        let builder = OrtEmbedding::builder();
        assert!(builder.normalize);
        let unnormalized = builder.normalize(false);
        assert!(!unnormalized.normalize);
    }

    #[test]
    fn rank_two_outputs_bypass_pooling_with_multi_token_masks() {
        let rank2 = ndarray::Array2::from_shape_vec((1, 2), vec![3.0_f32, 4.0_f32])
            .unwrap()
            .into_dyn();
        // Six-token attention mask would panic if rank-2 were reshaped to [1, 1, 2] and pooled.
        let mask = [1, 1, 1, 1, 1, 1];
        let normalized =
            extract_embedding(rank2.clone(), &mask, PoolingStrategy::LastToken, true).unwrap();
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);

        let raw = extract_embedding(rank2, &mask, PoolingStrategy::Mean, false).unwrap();
        assert_eq!(raw, vec![3.0, 4.0]);
    }

    #[test]
    fn extract_embedding_rejects_empty_and_invalid_ranks() {
        let rank1 = ndarray::Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn();
        assert!(matches!(
            extract_embedding(rank1, &[1], PoolingStrategy::Cls, true),
            Err(OrtError::InvalidOutputShape(1))
        ));

        let empty_rank2 = ndarray::Array2::<f32>::from_shape_vec((0, 4), Vec::new())
            .unwrap()
            .into_dyn();
        assert!(matches!(
            extract_embedding(empty_rank2, &[1], PoolingStrategy::Cls, true),
            Err(OrtError::Shape(_))
        ));
    }
}
