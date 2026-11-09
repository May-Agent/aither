//! The detector on Core ML, for Apple platforms.
//!
//! The model is the float16 Core ML conversion of the same weights the ONNX
//! model was exported from (`agent/convert`): one compiled multifunction
//! model with a function per sequence-length bucket, run on the CPU and the
//! Neural Engine. The conversion's verification checks that it picks the same
//! language as the ONNX model on FLORES-200 sentences.

use std::path::Path;

use ondevice::{Archive, Pin, Sha256};
use ondevice_coreml::{ComputeUnits, Model, Sequence, f16};

use super::DetectorError;

/// What the model installs as: a directory holding `manifest.json` and the
/// compiled model.
pub(in crate::language) use ondevice::Archive as Kind;

/// The converted model.
///
/// The archive is what `agent/convert` packages. The digest and size are those
/// of the archive a local run of that tool produced, which the tests serve; the
/// URL names where the publishing run puts it, and its revision and the digest
/// and size of the archive that run produces replace these once it has.
pub(in crate::language) const MODEL: Pin<'static, Archive> = Pin {
    url: "https://huggingface.co/lexoliu/aither-language-detection-coreml/resolve/main/language-coreml-41e39797.tar.zst",
    sha256: Sha256::from_hex("810065ee25a70c05d301be9c277a283dc1005e9ee92845088d9fd8b95a2a09f0"),
    size: 44_698_915,
    kind: Archive,
};

/// A Core ML model of the detector.
pub(super) struct Runtime {
    model: Model,
}

impl Runtime {
    /// Loads the model directory `path`.
    pub(super) fn open(path: &Path) -> Result<Self, DetectorError> {
        let model = Model::open(path, ComputeUnits::CpuAndNeuralEngine)
            .map_err(|error| DetectorError::Runtime(error.to_string()))?;
        Ok(Self { model })
    }

    /// The index of the label the model gives the highest logit for the
    /// unpadded token sequence `ids`; the model pads it to its smallest
    /// bucket.
    pub(super) fn best(&mut self, ids: &[u32]) -> Result<usize, DetectorError> {
        let runtime = |error: ondevice_coreml::Error| DetectorError::Runtime(error.to_string());
        let ids: Vec<i32> = ids
            .iter()
            .map(|&id| i32::try_from(id).expect("a vocabulary id fits in 32 bits"))
            .collect();
        let mask = vec![f16::ONE; ids.len()];
        let prediction = self
            .model
            .predict(&[
                &Sequence {
                    name: "input_ids",
                    values: &ids,
                    pad: 0,
                },
                &Sequence {
                    name: "attention_mask",
                    values: &mask,
                    pad: f16::ZERO,
                },
            ])
            .map_err(runtime)?;
        let logits = prediction.output::<f16>("logits").map_err(runtime)?;
        Ok(logits
            .data()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .expect("the model outputs one logit per label"))
    }
}
