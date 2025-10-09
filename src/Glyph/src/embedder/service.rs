use std::error::Error;
use crate::embedder::model::EmbeddingModel;
use crate::embedder::proto::{
    embedder_server::Embedder, EmbedSingleRequest, EmbedSingleResponse, Embedding, IndexRequest,
    IndexResponse,
};
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

pub struct EmbedderService {
    pub model: Arc<EmbeddingModel>,
}

type IndexTextsStream = Pin<Box<dyn Stream<Item = Result<IndexResponse, Status>> + Send>>;

#[tonic::async_trait]
impl Embedder for EmbedderService {
    // ... (The embed_single function remains the same) ...
    async fn embed_single(
        &self,
        request: Request<EmbedSingleRequest>,
    ) -> Result<Response<EmbedSingleResponse>, Status> {
        let text = request.into_inner().text;
        if text.is_empty() {
            return Err(Status::invalid_argument("Text cannot be empty"));
        }

        let model = self.model.clone();

        let embedding_result = tokio::task::spawn_blocking(move || model.embed_batch(&[text]))
            .await
            .map_err(|e| Status::internal(format!("Task join error: {}", e)))?;

        match embedding_result {
            Ok(mut embeddings_vec) => {
                let embedding = embeddings_vec
                    .pop()
                    .ok_or_else(|| Status::internal("Model returned no embedding"))?;
                let reply = EmbedSingleResponse {
                    embedding: Some(Embedding { values: embedding }),
                };
                Ok(Response::new(reply))
            }
            Err(e) => {
                eprintln!("Failed to generate embedding: {:?}", e);
                Err(Status::internal("Failed to generate embedding."))
            }
        }
    }

    type IndexTextsStream = IndexTextsStream;

    async fn index_texts(
        &self,
        request: Request<Streaming<IndexRequest>>,
    ) -> Result<Response<Self::IndexTextsStream>, Status> {
        let mut request_stream = request.into_inner();
        let model = self.model.clone();
        let (tx, rx) = mpsc::channel(32); // Increased buffer size slightly

        // This main task will only be responsible for receiving requests
        // and spawning new tasks to process them.
        tokio::spawn(async move {
            while let Some(result) = request_stream.next().await {
                match result {
                    Ok(req) => {
                        let model_clone = model.clone();
                        let tx_clone = tx.clone();

                        // **THE FIX:** Spawn a new async task for each message.
                        // This allows the server to immediately process the next message
                        // from the client without waiting for the previous one to finish.
                        tokio::spawn(async move {
                            let response = tokio::task::spawn_blocking(move || {
                                let doc_id = req.document_id;
                                let text = req.text;
                                let result = model_clone.embed_batch(&[text]);

                                match result {
                                    Ok(mut embeddings) => IndexResponse {
                                        document_id: doc_id,
                                        embedding: embeddings.pop().map(|v| Embedding { values: v }),
                                        success: true,
                                    },
                                    Err(_) => IndexResponse {
                                        document_id: doc_id,
                                        embedding: None,
                                        success: false,
                                    },
                                }
                            })
                                .await
                                .expect("Blocking task failed");

                            if tx_clone.send(Ok(response)).await.is_err() {
                                eprintln!("Response channel closed by client.");
                            }
                        });
                    }
                    Err(e) => {
                        if let Some(io_err) = e.source() {
                            eprintln!("IO error from client stream: {}", io_err);
                        }
                    }
                }
            }
        });

        let output_stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(output_stream) as Self::IndexTextsStream))
    }
}