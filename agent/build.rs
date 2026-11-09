//! Fetches the language-detection model for the `languages` feature.
//!
//! The model is `onnx-community/language_detection-ONNX` (an fp16 export of
//! `alexneakameni/language_detection`, MIT), taken from a pinned Hugging Face
//! revision and checked against the sha256 hashes below. The files land in
//! `OUT_DIR`, where the crate embeds them, so nothing large is committed and
//! nothing downloads at run time. A file already in `OUT_DIR` with the right
//! hash is not fetched again.

#[cfg(feature = "languages")]
mod model {
    use std::fs;
    use std::io::Read as _;
    use std::path::Path;

    use sha2::{Digest as _, Sha256};

    /// The commit of the Hugging Face repository the files are taken from.
    const REVISION: &str = "859bc93be10a6f8ad4a070911a4caaa092cdcd2e";
    const REPOSITORY: &str = "onnx-community/language_detection-ONNX";

    /// A file of the repository: its path there, its name in `OUT_DIR` and its
    /// sha256.
    struct File {
        remote: &'static str,
        local: &'static str,
        sha256: &'static str,
    }

    const MODEL: File = File {
        remote: "onnx/model_fp16.onnx",
        local: "language_model.onnx",
        sha256: "db571eee8fd2d18ab2815e8ef1632b924cca6f683bb6dd2de53f042b8bde3df5",
    };
    const TOKENIZER: File = File {
        remote: "tokenizer.json",
        local: "language_tokenizer.json",
        sha256: "ad313cb966420a8b21180d04071e84e49ab4c48005866131b7fec0672dfac935",
    };
    const CONFIG: File = File {
        remote: "config.json",
        local: "language_config.json",
        sha256: "eb8da40fd3e28f487ff3cfa65b180f1b78aadbc09b488d1ba580097b62b886a2",
    };

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn sha256(bytes: &[u8]) -> String {
        hex(&Sha256::digest(bytes))
    }

    /// The verified content of `file`, from `out_dir` or, failing that, the
    /// network.
    fn fetch(out_dir: &Path, file: &File) -> Vec<u8> {
        let path = out_dir.join(file.local);
        if let Ok(bytes) = fs::read(&path)
            && sha256(&bytes) == file.sha256
        {
            return bytes;
        }
        let url = format!(
            "https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{}",
            file.remote
        );
        let mut bytes = Vec::new();
        ureq::get(&url)
            .call()
            .unwrap_or_else(|error| panic!("failed to fetch {url}: {error}"))
            .into_body()
            .into_reader()
            .read_to_end(&mut bytes)
            .unwrap_or_else(|error| panic!("failed to read {url}: {error}"));
        let actual = sha256(&bytes);
        assert!(
            actual == file.sha256,
            "{url} has sha256 {actual}, expected {}",
            file.sha256
        );
        fs::write(&path, &bytes).unwrap_or_else(|error| panic!("failed to write {path:?}: {error}"));
        bytes
    }

    pub fn run() {
        let out_dir = std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR");
        let out_dir = Path::new(&out_dir);
        fetch(out_dir, &MODEL);
        fetch(out_dir, &TOKENIZER);

        // The labels in the order of the model's outputs, one per line.
        let config: serde_json::Value =
            serde_json::from_slice(&fetch(out_dir, &CONFIG)).expect("config.json is JSON");
        let id2label = config["id2label"]
            .as_object()
            .expect("config.json has id2label");
        let mut labels = String::new();
        for id in 0..id2label.len() {
            let label = id2label[&id.to_string()]
                .as_str()
                .expect("id2label values are strings");
            labels.push_str(label);
            labels.push('\n');
        }
        fs::write(out_dir.join("language_labels.txt"), labels)
            .unwrap_or_else(|error| panic!("failed to write the labels: {error}"));
    }
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    #[cfg(feature = "languages")]
    model::run();
}
