use crate::embedder::model::EmbeddingModel;
use crate::embedder::proto::{
    embedder_server::Embedder, EmbedSingleRequest, EmbedSingleResponse, Embedding, IndexRequest,
    IndexResponse,
};
use futures::Stream;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

pub struct EmbedderService {
    pub model: Arc<Mutex<EmbeddingModel>>,
}

type IndexTextsStream = Pin<Box<dyn Stream<Item = Result<IndexResponse, Status>> + Send>>;

struct Batch {
    document_ids: Vec<String>,
    texts: Vec<String>,
}

#[tonic::async_trait]
impl Embedder for EmbedderService {
    
    async fn embed_single(
        &self,
        request: Request<EmbedSingleRequest>,
    ) -> Result<Response<EmbedSingleResponse>, Status> {
        let text = request.into_inner().text;
        if text.is_empty() {
            return Err(Status::invalid_argument("Text cannot be empty"));
        }

        let model = self.model.clone();

        let embedding_result = tokio::task::spawn_blocking(move || {
            let model_guard = model.lock().expect("Mutex lock failed");
            model_guard.embed_batch(&[text])
        })
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

        // The batch_tx channel has a small buffer. If the model worker can't keep up,
        // this channel will fill up, and the `send` call will wait, creating backpressure.
        let (batch_tx, mut batch_rx) = mpsc::channel::<Batch>(4); // Small buffer for backpressure
        let (response_tx, response_rx) = mpsc::channel(32);

        // Spawn a dedicated worker task to process batches.
        // This task receives batches, runs the model, and sends results back.
        tokio::spawn(async move {
            while let Some(batch) = batch_rx.recv().await {
                let model_clone = model.clone();
                let response_tx_clone = response_tx.clone();

                tokio::task::spawn_blocking(move || {
                    let model_guard = model_clone.lock().expect("Mutex lock failed");
                    let embeddings = model_guard.embed_batch(&batch.texts);

                    match embeddings {
                        Ok(embeddings) => {
                            for (i, doc_id) in batch.document_ids.iter().enumerate() {
                                let response = IndexResponse {
                                    document_id: doc_id.clone(),
                                    embedding: embeddings.get(i).map(|v| Embedding { values: v.clone() }),
                                    success: true,
                                };
                                if response_tx_clone.blocking_send(Ok(response)).is_err() {
                                    break; // Client disconnected
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
                                if response_tx_clone.blocking_send(Ok(response)).is_err() {
                                    break; // Client disconnected
                                }
                            }
                        }
                    }
                });
            }
        });

        // Spawn a task to read from the client stream and create batches.
        tokio::spawn(async move {
            const BATCH_SIZE: usize = 32;
            const BATCH_TIMEOUT: Duration = Duration::from_millis(500);

            let mut batch_ids = Vec::with_capacity(BATCH_SIZE);
            let mut batch_texts = Vec::with_capacity(BATCH_SIZE);

            loop {
                match tokio::time::timeout(BATCH_TIMEOUT, request_stream.next()).await {
                    // Message received from stream
                    Ok(Some(Ok(req))) => {
                        batch_ids.push(req.document_id);
                        batch_texts.push(req.text);

                        if batch_ids.len() >= BATCH_SIZE {
                            let batch = Batch { document_ids: batch_ids, texts: batch_texts };
                            if batch_tx.send(batch).await.is_err() {
                                break; // Worker task died
                            }
                            batch_ids = Vec::with_capacity(BATCH_SIZE);
                            batch_texts = Vec::with_capacity(BATCH_SIZE);
                        }
                    }
                    // Stream ended or timed out
                    Ok(None) | Err(_) => {
                        if !batch_ids.is_empty() {
                            let batch = Batch { document_ids: batch_ids, texts: batch_texts };
                            let _ = batch_tx.send(batch).await; // Send final batch
                        }
                        break; // End of stream
                    }
                    // Client stream error
                    Ok(Some(Err(e))) => {
                        eprintln!("Client stream error: {}", e);
                        break;
                    }
                }
            }
        });

        // Return the response stream to the client.
        let output_stream = ReceiverStream::new(response_rx);
        Ok(Response::new(Box::pin(output_stream) as Self::IndexTextsStream))
    }
}