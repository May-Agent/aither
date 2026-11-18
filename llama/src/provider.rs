use crate::{Llama, LlamaError};
use aither_core::llm::{
    LanguageModelProvider, model::Profile as ModelProfile, provider::Profile as ProviderProfile,
};
use futures_lite::StreamExt;
use std::{
    future::Future,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

/// Provider for local llama.cpp GGUF models from disk.
#[derive(Clone, Debug)]
pub struct LlamaProvider {
    inner: Arc<ProviderConfig>,
}

impl LlamaProvider {
    /// Create a provider rooted at a directory containing GGUF models.
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            inner: Arc::new(ProviderConfig {
                models_dir: models_dir.into(),
            }),
        }
    }
}

impl LanguageModelProvider for LlamaProvider {
    type Model = Llama;
    type Error = LlamaError;

    fn list_models(&self) -> impl Future<Output = Result<Vec<ModelProfile>, Self::Error>> + Send {
        let cfg = self.inner.clone();
        async move {
            let mut profiles = Vec::new();
            let mut entries = async_fs::read_dir(&cfg.models_dir)
                .await
                .map_err(|err| LlamaError::Model(err.to_string()))?;

            while let Some(entry) = entries.next().await {
                let entry = entry.map_err(|err| LlamaError::Model(err.to_string()))?;
                let path = entry.path();
                let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
                    continue;
                };
                if ext != "gguf" {
                    continue;
                }
                let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                profiles.push(ModelProfile::new(
                    name.to_string(),
                    "llama.cpp",
                    name.to_string(),
                    "Local GGUF model",
                    8192,
                ));
            }

            Ok(profiles)
        }
    }

    fn get_model(
        &self,
        name: &str,
    ) -> impl Future<Output = Result<Self::Model, Self::Error>> + Send {
        let cfg = self.inner.clone();
        let name = name.to_string();
        async move {
            let path = resolve_gguf_model_path(&cfg.models_dir, &name)?;
            blocking::unblock(move || Llama::from_file(path)).await
        }
    }

    fn profile() -> ProviderProfile {
        ProviderProfile::new("llama", "Local llama.cpp GGUF provider")
    }
}

fn resolve_gguf_model_path(models_dir: &Path, name: &str) -> Result<PathBuf, LlamaError> {
    if name.is_empty() || name.contains('/') || name.contains('\\') {
        return Err(LlamaError::Model(format!(
            "invalid GGUF model file name '{name}'"
        )));
    }
    let candidate = Path::new(name);
    let mut components = candidate.components();
    let Some(Component::Normal(file_name)) = components.next() else {
        return Err(LlamaError::Model(format!(
            "invalid GGUF model file name '{name}'"
        )));
    };
    if components.next().is_some() {
        return Err(LlamaError::Model(format!(
            "invalid GGUF model file name '{name}'"
        )));
    }
    let has_gguf_ext = Path::new(file_name)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
    let has_stem = Path::new(file_name)
        .file_stem()
        .is_some_and(|stem| !stem.is_empty());
    if !has_gguf_ext || !has_stem {
        return Err(LlamaError::Model(format!(
            "model file '{name}' must have a .gguf extension"
        )));
    }
    Ok(models_dir.join(file_name))
}

#[derive(Debug, Clone)]
struct ProviderConfig {
    models_dir: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_gguf_model_path_accepts_single_gguf_name() {
        let dir = Path::new("/models");
        assert_eq!(
            resolve_gguf_model_path(dir, "qwen3-4b.gguf").unwrap(),
            PathBuf::from("/models/qwen3-4b.gguf")
        );
    }

    #[test]
    fn resolve_gguf_model_path_rejects_traversal_and_non_gguf() {
        let dir = Path::new("/models");
        for bad in [
            "",
            ".",
            "..",
            ".gguf",
            "../secret.gguf",
            "nested/model.gguf",
            "nested\\model.gguf",
            "/etc/passwd.gguf",
            "weights.safetensors",
        ] {
            assert!(
                resolve_gguf_model_path(dir, bad).is_err(),
                "expected '{bad}' to be rejected"
            );
        }
    }
}
