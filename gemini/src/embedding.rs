use aither_core::EmbeddingModel;

use crate::{
    client::embed_content,
    config::Gemini,
    error::GeminiError,
    types::{EmbedContentRequest, GeminiContent},
};

impl EmbeddingModel for Gemini {
    type Error = GeminiError;

    fn dim(&self) -> usize {
        self.config().embedding_dimensions
    }

    fn embed(
        &self,
        text: &str,
    ) -> impl core::future::Future<Output = Result<Vec<f32>, Self::Error>> + Send {
        let cfg = self.config();
        let prompt = text.to_owned();
        async move {
            let request =
                EmbedContentRequest::new(&cfg.embedding_model, GeminiContent::text("user", prompt));
            let response = embed_content(cfg, request).await?;
            Ok(response.embedding.values)
        }
    }
}
