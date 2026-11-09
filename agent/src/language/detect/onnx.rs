//! The detector on ONNX Runtime, for the platforms without Core ML.
//!
//! The model is `onnx/model_int8.onnx` of `onnx-community/language_detection-ONNX`
//! at the pinned revision. ONNX Runtime runs it with its default intra-op
//! threading, which follows the host's available parallelism.

use std::path::Path;

use ondevice::{File, Pin, Sha256};
use ort::session::Session;
use ort::value::Tensor;

use super::DetectorError;

/// What the model installs as.
pub(in crate::language) use ondevice::File as Kind;

/// The int8 ONNX model, 201 labels, 24.7 MB.
pub(in crate::language) const MODEL: Pin<'static, File> = Pin {
    url: "https://huggingface.co/onnx-community/language_detection-ONNX/resolve/859bc93be10a6f8ad4a070911a4caaa092cdcd2e/onnx/model_int8.onnx",
    sha256: Sha256::from_hex("ee8599c721d24f678ef10cee403f4deda3a40e7e237e9983b3df11bf3f34ec2e"),
    size: 24_714_523,
    kind: File,
};

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

/// An ONNX Runtime session of the model.
pub(super) struct Runtime {
    session: Session,
    inputs: Vec<Input>,
}

impl Runtime {
    /// Loads the model file at `path`.
    pub(super) fn open(path: &Path) -> Result<Self, DetectorError> {
        let runtime = |error: ort::Error| DetectorError::Runtime(error.to_string());
        let session = Session::builder()
            .map_err(runtime)?
            .commit_from_file(path)
            .map_err(runtime)?;
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
        Ok(Self { session, inputs })
    }

    /// The index of the label the model gives the highest logit for the
    /// unpadded token sequence `ids`.
    pub(super) fn best(&mut self, ids: &[u32]) -> Result<usize, DetectorError> {
        let runtime = |error: ort::Error| DetectorError::Runtime(error.to_string());
        let length = ids.len();
        let feed = self
            .inputs
            .iter()
            .map(|input| {
                let values: Vec<i64> = match input {
                    Input::Ids => ids.iter().map(|&id| i64::from(id)).collect(),
                    Input::AttentionMask => vec![1; length],
                    Input::TokenTypeIds => vec![0; length],
                };
                Ok((input.name(), Tensor::from_array(([1, length], values))?))
            })
            .collect::<Result<Vec<_>, ort::Error>>()
            .map_err(runtime)?;
        let outputs = self.session.run(feed).map_err(runtime)?;
        let (_, logits) = outputs[0].try_extract_tensor::<f32>().map_err(runtime)?;
        Ok(logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .expect("the model outputs one logit per label"))
    }
}
