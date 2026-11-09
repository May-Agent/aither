//! Identifying the language of a text with a small ONNX model.
//!
//! The model is `onnx-community/language_detection-ONNX` (int8, 201 labels),
//! fetched and verified by `build.rs` and embedded in the binary. It runs on
//! ONNX Runtime with its default intra-op threading, which follows the host's
//! available parallelism.

use std::fmt;
use std::sync::{Arc, Mutex};

use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use super::Language;

const MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/language_model.onnx"));
const TOKENIZER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/language_tokenizer.json"));
/// The model's labels in output order, one per line.
pub(super) const LABELS: &str = include_str!(concat!(env!("OUT_DIR"), "/language_labels.txt"));

/// The length every input is padded or truncated to, in tokens.
///
/// The Neural Engine takes static shapes only, so one length serves every
/// text. 128 tokens hold a sentence or a short paragraph of any script (a
/// FLORES-200 sentence is 20 to 80 tokens); the language of longer text is
/// judged on its first 128 tokens, and a long text is judged paragraph by
/// paragraph besides.
const SEQUENCE_LENGTH: usize = 128;
const PAD_TOKEN: &str = "[PAD]";

/// Why the language detector could not be built or run.
#[derive(Debug, thiserror::Error)]
pub enum DetectorError {
    /// The embedded tokenizer is unusable.
    #[error("language detector tokenizer: {0}")]
    Tokenizer(String),
    /// ONNX Runtime failed to load the model or run it.
    #[error("language detector runtime: {0}")]
    Runtime(#[from] ort::Error),
    /// The model has an output label that is not a language this crate
    /// knows.
    #[error("language detector label `{0}` is not a known language")]
    UnknownLabel(String),
    /// The model takes an input this crate does not know how to fill.
    #[error("language detector input `{0}` is not understood")]
    UnknownInput(String),
}

/// The loaded model. Clones share the session.
#[derive(Clone)]
pub(super) struct Detector {
    inner: Arc<Inner>,
}

struct Inner {
    tokenizer: Tokenizer,
    /// Running a session needs exclusive access; ONNX Runtime already spreads
    /// one run over every core, so runs gain nothing from overlapping.
    session: Mutex<Session>,
    inputs: Vec<Input>,
    /// The language of each output label.
    labels: Vec<Language>,
}

#[derive(Clone, Copy)]
enum Input {
    Ids,
    AttentionMask,
    TokenTypeIds,
}

impl Input {
    const fn name(self) -> &'static str {
        match self {
            Self::Ids => "input_ids",
            Self::AttentionMask => "attention_mask",
            Self::TokenTypeIds => "token_type_ids",
        }
    }
}

impl fmt::Debug for Detector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Detector")
            .field("labels", &self.inner.labels.len())
            .finish_non_exhaustive()
    }
}

impl Detector {
    /// Loads the embedded model. This is CPU work and takes tens of
    /// milliseconds.
    ///
    /// # Errors
    ///
    /// [`DetectorError`] when the model, its tokenizer or its labels are not
    /// what this crate expects.
    pub(super) fn load() -> Result<Self, DetectorError> {
        let mut tokenizer = Tokenizer::from_bytes(TOKENIZER)
            .map_err(|error| DetectorError::Tokenizer(error.to_string()))?;
        let pad_id = tokenizer
            .token_to_id(PAD_TOKEN)
            .ok_or_else(|| DetectorError::Tokenizer(format!("no {PAD_TOKEN} token")))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: SEQUENCE_LENGTH,
                ..TruncationParams::default()
            }))
            .map_err(|error| DetectorError::Tokenizer(error.to_string()))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(SEQUENCE_LENGTH),
            pad_id,
            pad_token: PAD_TOKEN.to_string(),
            ..PaddingParams::default()
        }));
        let session = Session::builder()?;
        // The Neural Engine runs fp16 `MLProgram` graphs of static shape.
        #[cfg(target_vendor = "apple")]
        let session = session.with_execution_providers([ort::ep::CoreML::default()
            .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
            .with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndNeuralEngine)
            .with_static_input_shapes(true)
            .build()])?;
        let session = session.commit_from_memory(MODEL)?;
        let inputs = session
            .inputs()
            .iter()
            .map(|input| match input.name() {
                "input_ids" => Ok(Input::Ids),
                "attention_mask" => Ok(Input::AttentionMask),
                "token_type_ids" => Ok(Input::TokenTypeIds),
                other => Err(DetectorError::UnknownInput(other.to_string())),
            })
            .collect::<Result<_, _>>()?;
        let labels = LABELS
            .lines()
            .map(|label| {
                Language::from_label(label)
                    .ok_or_else(|| DetectorError::UnknownLabel(label.to_string()))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            inner: Arc::new(Inner {
                tokenizer,
                session: Mutex::new(session),
                inputs,
                labels,
            }),
        })
    }

    /// The language `text` is in. This is CPU work.
    ///
    /// # Errors
    ///
    /// [`DetectorError`] when tokenizing or running the model fails.
    pub(super) fn detect(&self, text: &str) -> Result<Language, DetectorError> {
        let inner = &*self.inner;
        let encoding = inner
            .tokenizer
            .encode(text, true)
            .map_err(|error| DetectorError::Tokenizer(error.to_string()))?;
        let length = encoding.get_ids().len();
        let widen = |values: &[u32]| values.iter().map(|&value| i64::from(value)).collect();
        let feed = inner
            .inputs
            .iter()
            .map(|input| {
                let values: Vec<i64> = match input {
                    Input::Ids => widen(encoding.get_ids()),
                    Input::AttentionMask => widen(encoding.get_attention_mask()),
                    Input::TokenTypeIds => vec![0; length],
                };
                Ok((input.name(), Tensor::from_array(([1, length], values))?))
            })
            .collect::<Result<Vec<_>, ort::Error>>()?;
        let mut session = inner.session.lock().expect("language session lock");
        let outputs = session.run(feed)?;
        let (_, logits) = outputs[0].try_extract_tensor::<f32>()?;
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .expect("the model outputs one logit per label");
        Ok(inner.labels[best])
    }
}
