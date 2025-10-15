use crate::clipembedder::model::ClipEmbeddingModel;
use crate::clipembedder::proto::{
    ClipEmbedder, EmbedImageRequest, EmbedResponse, EmbedTextRequest, Embedding, IndexImageRequest,
    IndexResponse,
};
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

pub struct ClipEmbedderService {
    pub model: Arc<Mutex<ClipEmbeddingModel>>,
}

struct ImageBatch {
    document_ids: Vec<String>,
    images: Vec<Vec<u8>>,
}

#[tonic::async_trait]
impl ClipEmbedder for ClipEmbedderService {
    async fn embed_text(
        &self,
        request: Request<EmbedTextRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let text = request.into_inner().text;
        if text.is_empty() {
            return Err(Status::invalid_argument("Text cannot be empty"));
        }

        let model = self.model.clone();
        let embedding = tokio::task::spawn_blocking(move || model.lock().unwrap().embed_texts(&[text]))
            .await
            .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
            .map_err(|e| Status::internal(format!("Embedding generation failed: {}", e)))?
            .pop()
            .ok_or_else(|| Status::internal("Model returned no embedding"))?;

        Ok(Response::new(EmbedResponse {
            embedding: Some(Embedding { values: embedding }),
        }))
    }

    async fn embed_image(
        &self,
        request: Request<EmbedImageRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let image_bytes = request.into_inner().image;
        if image_bytes.is_empty() {
            return Err(Status::invalid_argument("Image bytes cannot be empty"));
        }

        let model = self.model.clone();
        let embedding = tokio::task::spawn_blocking(move || {
            model.lock().unwrap().embed_images(&[image_bytes.clone()])
        })
            .await
            .map_err(|e| Status::internal(format!("Task join error: {}", e)))?
            .map_err(|e| Status::internal(format!("Embedding generation failed: {}", e)))?
            .pop()
            .ok_or_else(|| Status::internal("Model returned no embedding"))?;

        Ok(Response::new(EmbedResponse {
            embedding: Some(Embedding { values: embedding }),
        }))
    }

    type IndexImagesStream = Pin<Box<dyn Stream<Item = Result<IndexResponse, Status>> + Send>>;

    async fn index_images(
        &self,
        request: Request<Streaming<IndexImageRequest>>,
    ) -> Result<Response<Self::IndexImagesStream>, Status> {
        let mut request_stream = request.into_inner();
        let model = self.model.clone();
        let (batch_tx, mut batch_rx) = mpsc::channel::<ImageBatch>(4);
        let (response_tx, response_rx) = mpsc::channel(32);

        // Worker task to process image batches
        tokio::spawn(async move {
            while let Some(batch) = batch_rx.recv().await {
                let model = model.clone();
                let response_tx = response_tx.clone();
                tokio::task::spawn_blocking(move || {
                    let embeddings_result = model.lock().unwrap().embed_images(&batch.images);
                    match embeddings_result {
                        Ok(embeddings) => {
                            for (i, doc_id) in batch.document_ids.iter().enumerate() {
                                let response = IndexResponse {
                                    document_id: doc_id.clone(),
                                    embedding: embeddings
                                        .get(i)
                                        .map(|v| Embedding { values: v.clone() }),
                                    success: true,
                                };
                                if response_tx.blocking_send(Ok(response)).is_err() {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Batch embedding failed: {:?}", e);
                            for doc_id in batch.document_ids {
                                let response = IndexResponse {
                                    document_id: doc_id,
                                    embedding: None,
                                    success: false,
                                };
                                if response_tx.blocking_send(Ok(response)).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                });
            }
        });

        // Task to create batches from the client stream
        tokio::spawn(async move {
            const BATCH_SIZE: usize = 16;
            const BATCH_TIMEOUT: Duration = Duration::from_millis(500);
            let mut batch_ids = Vec::with_capacity(BATCH_SIZE);
            let mut batch_images = Vec::with_capacity(BATCH_SIZE);

            loop {
                match tokio::time::timeout(BATCH_TIMEOUT, request_stream.next()).await {
                    Ok(Some(Ok(req))) => {
                        batch_ids.push(req.document_id);
                        batch_images.push(req.image);

                        if batch_ids.len() >= BATCH_SIZE {
                            let batch = ImageBatch {
                                document_ids: batch_ids,
                                images: batch_images,
                            };
                            if batch_tx.send(batch).await.is_err() {
                                break;
                            }
                            batch_ids = Vec::with_capacity(BATCH_SIZE);
                            batch_images = Vec::with_capacity(BATCH_SIZE);
                        }
                    }
                    Ok(None) | Err(_) => {
                        if !batch_ids.is_empty() {
                            let batch = ImageBatch {
                                document_ids: batch_ids,
                                images: batch_images,
                            };
                            let _ = batch_tx.send(batch).await;
                        }
                        break;
                    }
                    Ok(Some(Err(e))) => {
                        eprintln!("Client stream error: {}", e);
                        break;
                    }
                }
            }
        });

        let output_stream = ReceiverStream::new(response_rx);
        Ok(Response::new(Box::pin(output_stream)))
    }
}