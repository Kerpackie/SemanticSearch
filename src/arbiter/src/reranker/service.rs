use crate::reranker::model::RerankerModel;
use crate::reranker::proto::{
    reranker_server::Reranker, RankedDocument, RerankRequest, RerankResponse,
    ScorePairRequest, ScorePairResponse,
};
use std::sync::{Arc, Mutex};
use tonic::{Request, Response, Status};
use tracing::{info, warn, debug};

pub struct RerankerService {
    pub model: Arc<Mutex<RerankerModel>>,
}

#[tonic::async_trait]
impl Reranker for RerankerService {
    /// Reranks a list of documents based on relevance to a query.
    async fn rerank(
        &self,
        request: Request<RerankRequest>,
    ) -> Result<Response<RerankResponse>, Status> {
        let inner = request.into_inner();
        let query = inner.query;
        let documents = inner.documents;
        let top_k = inner.top_k as usize;

        info!(
            query = %query,
            num_documents = documents.len(),
            top_k = top_k,
            "Received rerank request"
        );

        if query.is_empty() {
            warn!("Query is empty");
            return Err(Status::invalid_argument("Query cannot be empty"));
        }

        if documents.is_empty() {
            info!("No documents to rerank, returning empty results");
            return Ok(Response::new(RerankResponse { results: vec![] }));
        }

        let model = self.model.clone();
        let texts: Vec<String> = documents.iter().map(|d| d.text.clone()).collect();
        let doc_ids: Vec<String> = documents.iter().map(|d| d.id.clone()).collect();
        let doc_texts: Vec<String> = documents.iter().map(|d| d.text.clone()).collect();

        // Log each document being scored (with preview)
        for (idx, doc) in documents.iter().enumerate() {
            let preview: String = doc.text.chars().take(100).collect();
            debug!(
                index = idx,
                doc_id = %doc.id,
                text_preview = %preview,
                "Document to rerank"
            );
        }

        info!("Starting scoring task");
        let scores_result = tokio::task::spawn_blocking(move || {
            let model_guard = model.lock().expect("Mutex lock failed");
            model_guard.score_pairs(&query, &texts)
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?;

        match scores_result {
            Ok(scores) => {
                info!("Scoring completed, {} scores computed", scores.len());
                
                // Log each individual score
                for (idx, score) in scores.iter().enumerate() {
                    let doc_preview: String = doc_texts[idx].chars().take(80).collect();
                    info!(
                        index = idx,
                        doc_id = %doc_ids[idx],
                        score = %score,
                        text_preview = %doc_preview,
                        "Document scored"
                    );
                }

                // Combine scores with documents and sort by score descending
                let mut ranked: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                info!("Documents sorted by relevance score (descending)");
                
                // Log the ranking order
                for (rank, (original_idx, score)) in ranked.iter().enumerate().take(10) {
                    let doc_preview: String = doc_texts[*original_idx].chars().take(60).collect();
                    info!(
                        rank = rank + 1,
                        original_index = original_idx,
                        doc_id = %doc_ids[*original_idx],
                        score = %score,
                        text_preview = %doc_preview,
                        "Ranked result"
                    );
                }

                // Apply top_k if specified
                let results: Vec<RankedDocument> = ranked
                    .into_iter()
                    .take(if top_k > 0 { top_k } else { doc_ids.len() })
                    .enumerate()
                    .map(|(rank, (original_idx, score))| RankedDocument {
                        id: doc_ids[original_idx].clone(),
                        text: doc_texts[original_idx].clone(),
                        score,
                        rank: (rank + 1) as i32,
                    })
                    .collect();

                info!(
                    num_results = results.len(),
                    "Reranking completed successfully"
                );

                Ok(Response::new(RerankResponse { results }))
            }
            Err(e) => {
                warn!(error = %e, "Failed to score documents");
                eprintln!("Failed to score documents: {:?}", e);
                Err(Status::internal(format!("Failed to score documents: {}", e)))
            }
        }
    }

    /// Scores a single query-document pair.
    async fn score_pair(
        &self,
        request: Request<ScorePairRequest>,
    ) -> Result<Response<ScorePairResponse>, Status> {
        let inner = request.into_inner();
        let query = inner.query;
        let document = inner.document;

        info!(
            query = %query,
            document_preview = %document.chars().take(100).collect::<String>(),
            "Received score_pair request"
        );

        if query.is_empty() {
            warn!("Query is empty");
            return Err(Status::invalid_argument("Query cannot be empty"));
        }

        if document.is_empty() {
            warn!("Document is empty");
            return Err(Status::invalid_argument("Document cannot be empty"));
        }

        let model = self.model.clone();

        let score_result = tokio::task::spawn_blocking(move || {
            let model_guard = model.lock().expect("Mutex lock failed");
            model_guard.score_single(&query, &document)
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?;

        match score_result {
            Ok(score) => {
                info!(score = %score, "Score computed successfully");
                Ok(Response::new(ScorePairResponse { score }))
            }
            Err(e) => {
                warn!(error = %e, "Failed to score pair");
                eprintln!("Failed to score pair: {:?}", e);
                Err(Status::internal("Failed to score pair."))
            }
        }
    }
}
