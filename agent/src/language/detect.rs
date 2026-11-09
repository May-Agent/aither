//! Identifying the language of a text with a small ONNX model.
//!
//! The model is `onnx-community/language_detection-ONNX` (int8, 201 labels, 24.7
//! MB; a 4-layer BERT exported from `alexneakameni/language_detection`, MIT).
//! Nothing of it is committed: the files are taken from a pinned revision of the
//! Hugging Face repository, checked against the sha256 hashes below, and kept in
//! the user's cache directory, so the first use downloads them and later uses
//! only read them. The model runs on ONNX Runtime with its default intra-op
//! threading, which follows the host's available parallelism.

use std::fmt;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ort::session::Session;
use ort::value::Tensor;
use sha2::{Digest as _, Sha256};
use tokenizers::{Tokenizer, TruncationParams};

use super::Language;

/// The commit of the Hugging Face repository the files are taken from.
const REVISION: &str = "859bc93be10a6f8ad4a070911a4caaa092cdcd2e";
const REPOSITORY: &str = "onnx-community/language_detection-ONNX";

/// A file of the repository: its path there, its name in the cache and its
/// sha256.
struct File {
    remote: &'static str,
    local: &'static str,
    sha256: &'static str,
}

const MODEL: File = File {
    remote: "onnx/model_int8.onnx",
    local: "model_int8.onnx",
    sha256: "ee8599c721d24f678ef10cee403f4deda3a40e7e237e9983b3df11bf3f34ec2e",
};
const TOKENIZER: File = File {
    remote: "tokenizer.json",
    local: "tokenizer.json",
    sha256: "ad313cb966420a8b21180d04071e84e49ab4c48005866131b7fec0672dfac935",
};

/// The model's labels in output order, one per line; the `id2label` of the
/// repository's `config.json`.
pub(super) const LABELS: &str = include_str!("labels.txt");

/// The longest input in tokens. 128 tokens hold a sentence or a short
/// paragraph of any script (a FLORES-200 sentence is 20 to 80 tokens); the
/// language of a longer paragraph is judged on its first 128 tokens.
const SEQUENCE_LENGTH: usize = 128;

/// Why the language detector could not be built or run.
#[derive(Debug, thiserror::Error)]
pub enum DetectorError {
    /// The host has no cache directory to keep the model in.
    #[error("the language detector has no cache directory for its model")]
    NoCacheDirectory,
    /// Reading or writing a model file failed.
    #[error("language detector file {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// What failed.
        source: std::io::Error,
    },
    /// Downloading a model file failed.
    #[error("language detector download of {url}: {message}")]
    Download {
        /// Where the file comes from.
        url: String,
        /// What failed.
        message: String,
    },
    /// A downloaded file is not the pinned one.
    #[error("language detector file {url} has sha256 {actual}, expected {expected}")]
    Hash {
        /// Where the file comes from.
        url: String,
        /// The hash of what was received.
        actual: String,
        /// The pinned hash.
        expected: &'static str,
    },
    /// The tokenizer is unusable.
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

fn sha256(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    Sha256::digest(bytes)
        .iter()
        .fold(String::new(), |mut hex, byte| {
            write!(hex, "{byte:02x}").expect("writing to a String cannot fail");
            hex
        })
}

/// The verified content of `file` in `directory`, downloaded first when it is
/// not there or is not the pinned one.
fn fetch(directory: &Path, file: &File) -> Result<Vec<u8>, DetectorError> {
    let path = directory.join(file.local);
    if let Ok(bytes) = fs::read(&path)
        && sha256(&bytes) == file.sha256
    {
        return Ok(bytes);
    }
    let url = format!(
        "https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{}",
        file.remote
    );
    let download = |message: String| DetectorError::Download {
        url: url.clone(),
        message,
    };
    tracing::info!(%url, "downloading the language detection model");
    let mut bytes = Vec::new();
    ureq::get(&url)
        .call()
        .map_err(|error| download(error.to_string()))?
        .into_body()
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|error| download(error.to_string()))?;
    let actual = sha256(&bytes);
    if actual != file.sha256 {
        return Err(DetectorError::Hash {
            url,
            actual,
            expected: file.sha256,
        });
    }
    let io = |source| DetectorError::Io {
        path: path.clone(),
        source,
    };
    fs::create_dir_all(directory).map_err(io)?;
    // Written whole and renamed, so a concurrent reader never sees half a file.
    let partial = path.with_extension(format!(
        "partial{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    fs::write(&partial, &bytes).map_err(io)?;
    fs::rename(&partial, &path).map_err(io)?;
    Ok(bytes)
}

impl Detector {
    /// Loads the model, downloading it into the user's cache directory when it
    /// is not there. This blocks on the network and the disk, then on CPU work
    /// of tens of milliseconds.
    ///
    /// # Errors
    ///
    /// [`DetectorError`] when the files cannot be obtained or are not what
    /// this crate expects.
    pub(super) fn load() -> Result<Self, DetectorError> {
        let directory = dirs::cache_dir()
            .ok_or(DetectorError::NoCacheDirectory)?
            .join("aither")
            .join("language-detection")
            .join(REVISION);
        let tokenizer = fetch(&directory, &TOKENIZER)?;
        let model = fetch(&directory, &MODEL)?;
        Self::from_bytes(&model, &tokenizer)
    }

    fn from_bytes(model: &[u8], tokenizer: &[u8]) -> Result<Self, DetectorError> {
        let mut tokenizer = Tokenizer::from_bytes(tokenizer)
            .map_err(|error| DetectorError::Tokenizer(error.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: SEQUENCE_LENGTH,
                ..TruncationParams::default()
            }))
            .map_err(|error| DetectorError::Tokenizer(error.to_string()))?;
        let session = Session::builder()?.commit_from_memory(model)?;
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
    #[allow(
        clippy::significant_drop_tightening,
        reason = "the outputs borrow the locked session"
    )]
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
        let best = {
            let mut session = inner.session.lock().expect("language session lock");
            let outputs = session.run(feed)?;
            let (_, logits) = outputs[0].try_extract_tensor::<f32>()?;
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .expect("the model outputs one logit per label")
        };
        Ok(inner.labels[best])
    }
}
