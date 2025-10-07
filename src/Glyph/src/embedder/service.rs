use std::sync::Arc;
use tonic::{Request, Response, Status};
use crate::embedder::model::EmbeddingModel;
use crate::embedder::proto::{EmbedRequest, EmbedResponse, Embedder, Embedding};


pub struct EmbedderService {
    pub model: Arc<EmbeddingModel>,
}

#[tonic::async_trait]
impl Embedder for EmbedderService {
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let texts = request.into_inner().texts;

        if texts.is_empty() {
            return Ok(Response::new(EmbedResponse::default()));
        }

        let model = self.model.clone();
        let embeddings_result = tokio::task::spawn_blocking(move || model.embed_batch(&texts))
            .await
            .map_err(|e| Status::internal(format!("Task join error: {}", e)))?;

        match embeddings_result {
            Ok(embeddings_vec) => {
                let reply = EmbedResponse {
                    embeddings: embeddings_vec
                        .into_iter()
                        .map(|v| Embedding { values: v })
                        .collect(),
                };
                Ok(Response::new(reply))
            }
            Err(_e) => {
                eprintln!("Failed to generate embeddings: {:?}", _e);
                Err(Status::internal("Failed to generate embeddings."))
            }
        }
    }
}
