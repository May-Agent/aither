//! Identifying the language of a text with a small neural model.
//!
//! The model is `alexneakameni/language_detection` (201 labels; a 4-layer BERT,
//! MIT), run as the int8 ONNX export of `onnx-community/language_detection-ONNX`
//! on ONNX Runtime where there is no Core ML, and as a float16 Core ML
//! conversion of the same weights on Apple platforms, where the Neural Engine
//! runs it. Nothing of it is committed: each platform's model and the `WordPiece`
//! tokenizer are [`Pin`]s, downloaded on first use into the user's cache
//! directory by an [`ondevice::Store`] and verified against their sha256, and
//! later uses only read them.

use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ondevice::{File, Pin, Sha256, Store, Transport, Zenwave};
use tokenizers::{Tokenizer, TruncationParams};

use super::Language;

// The platform's runtime: its model pin, what that installs as, and the model
// run on it.
#[cfg(target_vendor = "apple")]
#[path = "detect/coreml.rs"]
mod runtime;
#[cfg(not(target_vendor = "apple"))]
#[path = "detect/onnx.rs"]
mod runtime;

use runtime::Runtime;
pub(super) use runtime::{Kind as ModelKind, MODEL};

/// The tokenizer, at the ONNX repository's pinned revision; the Core ML
/// conversion shares it.
pub(super) const TOKENIZER: Pin<'static, File> = Pin {
    url: "https://huggingface.co/onnx-community/language_detection-ONNX/resolve/859bc93be10a6f8ad4a070911a4caaa092cdcd2e/tokenizer.json",
    sha256: Sha256::from_hex("ad313cb966420a8b21180d04071e84e49ab4c48005866131b7fec0672dfac935"),
    size: 1_109_464,
    kind: File,
};

/// The model's labels in output order, one per line; the `id2label` of the
/// repository's `config.json`.
pub(super) const LABELS: &str = include_str!("labels.txt");

/// The longest input in tokens. 128 tokens hold a sentence or a short
/// paragraph of any script (a FLORES-200 sentence is 20 to 80 tokens); the
/// language of a longer paragraph is judged on its first 128 tokens. It is the
/// largest bucket of the Core ML model.
const SEQUENCE_LENGTH: usize = 128;

/// Why the language detector could not be built or run.
#[derive(Debug, thiserror::Error)]
pub enum DetectorError {
    /// The host has no cache directory to keep the model in.
    #[error("the language detector has no cache directory for its model")]
    NoCacheDirectory,
    /// Downloading, verifying or installing a model file failed.
    #[error("language detector model file: {0}")]
    Store(String),
    /// The tokenizer is unusable.
    #[error("language detector tokenizer: {0}")]
    Tokenizer(String),
    /// The model runtime failed to load the model or run it.
    #[error("language detector runtime: {0}")]
    Runtime(String),
    /// The model has an output label that is not a language this crate
    /// knows.
    #[error("language detector label `{0}` is not a known language")]
    UnknownLabel(String),
    /// The model takes an input this crate does not know how to fill.
    #[error("language detector input `{0}` is not understood")]
    UnknownInput(String),
}

/// The loaded model. Clones share the runtime.
#[derive(Clone)]
pub(super) struct Detector {
    inner: Arc<Inner>,
}

struct Inner {
    tokenizer: Tokenizer,
    /// Running the model needs exclusive access, and one run already spreads
    /// over every core (ONNX Runtime) or the Neural Engine (Core ML), so runs
    /// gain nothing from overlapping.
    runtime: Mutex<Runtime>,
    /// The language of each output label.
    labels: Vec<Language>,
}

impl fmt::Debug for Detector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Detector")
            .field("labels", &self.inner.labels.len())
            .finish_non_exhaustive()
    }
}

/// The store the detector's files are kept in: the user's cache directory.
fn default_store() -> Result<Store<Zenwave>, DetectorError> {
    let root = dirs::cache_dir()
        .ok_or(DetectorError::NoCacheDirectory)?
        .join("aither")
        .join("language-detection");
    Ok(Store::new(root, Zenwave))
}

impl Detector {
    /// Loads the model through the user's cache directory's store,
    /// downloading it on the first use on this machine.
    ///
    /// # Errors
    ///
    /// [`DetectorError`] when the files cannot be obtained or are not what
    /// this crate expects.
    pub(super) async fn load_async() -> Result<Self, DetectorError> {
        Self::load_from(&default_store()?, &MODEL).await
    }

    /// Loads the model `model` pins and the tokenizer, from `store`.
    ///
    /// # Errors
    ///
    /// As [`load_async`](Self::load_async).
    pub(super) async fn load_from<T: Transport>(
        store: &Store<T>,
        model: &Pin<'_, ModelKind>,
    ) -> Result<Self, DetectorError> {
        let (tokenizer, model) =
            futures_lite::future::try_zip(install(store, &TOKENIZER), install(store, model))
                .await?;
        blocking::unblock(move || Self::open(&tokenizer, &model)).await
    }

    fn open(tokenizer: &std::path::Path, model: &std::path::Path) -> Result<Self, DetectorError> {
        let mut tokenizer = Tokenizer::from_file(tokenizer)
            .map_err(|error| DetectorError::Tokenizer(error.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: SEQUENCE_LENGTH,
                ..TruncationParams::default()
            }))
            .map_err(|error| DetectorError::Tokenizer(error.to_string()))?;
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
                runtime: Mutex::new(Runtime::open(model)?),
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
        let best = inner
            .runtime
            .lock()
            .expect("language runtime lock")
            .best(encoding.get_ids())?;
        Ok(inner.labels[best])
    }
}

/// The installed `pin`, downloaded on the first use.
async fn install<T: Transport, K: ondevice::Content>(
    store: &Store<T>,
    pin: &Pin<'_, K>,
) -> Result<PathBuf, DetectorError> {
    store
        .ensure(pin, |_| {})
        .await
        .map_err(|error| DetectorError::Store(error.to_string()))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use futures_lite::StreamExt as _;
    use ondevice::{Response, Store, Transport, Zenwave};

    use super::{Detector, MODEL, default_store};

    /// A [`Transport`] that serves the files of a local directory by file name
    /// and downloads everything else: how the tests reach a model that is
    /// built locally and not yet published.
    #[derive(Debug)]
    pub(in crate::language) struct Mirror {
        directory: PathBuf,
    }

    /// A failure of the [`Mirror`].
    #[derive(Debug, thiserror::Error)]
    #[error("{0}")]
    pub(in crate::language) struct MirrorError(String);

    impl Transport for Mirror {
        type Error = MirrorError;
        type Chunk = Vec<u8>;
        type Body = futures_lite::stream::Boxed<Result<Vec<u8>, MirrorError>>;

        async fn get(&self, url: &str) -> Result<Response<Self::Body>, MirrorError> {
            let name = url.rsplit('/').next().unwrap_or(url);
            let local = self.directory.join(name);
            if blocking::unblock({
                let local = local.clone();
                move || local.is_file()
            })
            .await
            {
                let bytes = blocking::unblock(move || std::fs::read(local))
                    .await
                    .map_err(|error| MirrorError(error.to_string()))?;
                return Ok(Response {
                    length: Some(bytes.len() as u64),
                    body: futures_lite::stream::once(Ok(bytes)).boxed(),
                });
            }
            let response = Zenwave
                .get(url)
                .await
                .map_err(|error| MirrorError(error.to_string()))?;
            Ok(Response {
                length: response.length,
                body: response
                    .body
                    .map(|chunk| {
                        chunk
                            .map(|bytes| bytes.as_ref().to_vec())
                            .map_err(|error| MirrorError(error.to_string()))
                    })
                    .boxed(),
            })
        }
    }

    /// The store the tests load from: the user's cache directory's, fed from
    /// the directory `AITHER_LANGUAGE_MIRROR` names when it is set.
    pub(in crate::language) fn mirror_store() -> Store<Mirror> {
        let directory = std::env::var_os("AITHER_LANGUAGE_MIRROR")
            .map_or_else(|| PathBuf::from("/nonexistent"), PathBuf::from);
        let root = default_store()
            .expect("the tests need a cache directory")
            .root()
            .to_path_buf();
        Store::new(root, Mirror { directory })
    }

    impl Detector {
        /// The detector the tests judge with: loaded through
        /// [`mirror_store`], so a model built locally and not yet published
        /// can be served.
        pub(in crate::language) async fn for_tests() -> Self {
            Self::load_from(&mirror_store(), &MODEL)
                .await
                .unwrap_or_else(|error| panic!("{error}"))
        }
    }
}
