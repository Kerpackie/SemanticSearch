use crate::reranker::model::RerankerModel;
use crate::reranker::proto::{
    reranker_server::Reranker, RankedDocument, RerankRequest, RerankResponse,
    ScorePairRequest, ScorePairResponse,
};
use std::sync::{Arc, Mutex};
use tonic::{Request, Response, Status};

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

        if query.is_empty() {
            return Err(Status::invalid_argument("Query cannot be empty"));
        }

        if documents.is_empty() {
            return Ok(Response::new(RerankResponse { results: vec![] }));
        }

        let model = self.model.clone();
        let texts: Vec<String> = documents.iter().map(|d| d.text.clone()).collect();
        let doc_ids: Vec<String> = documents.iter().map(|d| d.id.clone()).collect();
        let doc_texts: Vec<String> = documents.iter().map(|d| d.text.clone()).collect();

        let scores_result = tokio::task::spawn_blocking(move || {
            let model_guard = model.lock().expect("Mutex lock failed");
            model_guard.score_pairs(&query, &texts)
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?;

        match scores_result {
            Ok(scores) => {
                // Combine scores with documents and sort by score descending
                let mut ranked: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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

                Ok(Response::new(RerankResponse { results }))
            }
            Err(e) => {
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

        if query.is_empty() {
            return Err(Status::invalid_argument("Query cannot be empty"));
        }

        if document.is_empty() {
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
            Ok(score) => Ok(Response::new(ScorePairResponse { score })),
            Err(e) => {
                eprintln!("Failed to score pair: {:?}", e);
                Err(Status::internal("Failed to score pair."))
            }
        }
    }
}
