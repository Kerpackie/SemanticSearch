use candle_core::{Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use anyhow::{Error as E, Result};
use tracing::{info, debug};

/// A reranker model for scoring query-document relevance.
/// Uses a cross-encoder architecture (BERT-based) that takes query-document pairs
/// and outputs relevance scores.
pub struct RerankerModel {
    pub model: BertModel,
    pub classifier: Linear,
    pub tokenizer: Tokenizer,
    pub device: Device,
}

impl RerankerModel {
    /// Creates a new RerankerModel with the specified model from HuggingFace.
    /// Recommended models:
    /// - "cross-encoder/ms-marco-MiniLM-L-6-v2" (22M params, fast, good quality)
    /// - "cross-encoder/ms-marco-MiniLM-L-12-v2" (33M params, better quality)
    /// - "cross-encoder/ms-marco-TinyBERT-L-2-v2" (4.4M params, very fast)
    pub fn new(model_id: &str) -> Result<Self> {
        let device = if candle_core::utils::metal_is_available() {
            Device::new_metal(0)?
        } else {
            Device::Cpu
        };
        
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        
        // Cross-encoder models have weights prefixed with "bert."
        let bert_vb = vb.pp("bert");
        let model = BertModel::load(bert_vb, &config)?;
        
        // Load the classifier head for reranking (outputs a single score)
        // The classifier weights are at classifier.weight and classifier.bias
        let classifier_vb = vb.pp("classifier");
        let num_labels = 1; // Rerankers output a single relevance score
        let classifier = candle_nn::linear(config.hidden_size, num_labels, classifier_vb)?;

        Ok(Self {
            model,
            classifier,
            tokenizer,
            device,
        })
    }

    /// Scores a batch of query-document pairs.
    /// Returns relevance scores where higher = more relevant.
    pub fn score_pairs(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            debug!("No documents to score");
            return Ok(vec![]);
        }

        debug!(
            num_documents = documents.len(),
            query_len = query.len(),
            "Starting batch scoring"
        );

        let mut tokenizer = self.tokenizer.clone();

        // Configure padding for batch processing
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        // Configure truncation to handle long texts
        tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: 512,
            ..Default::default()
        })).map_err(E::msg)?;

        // Encode query-document pairs
        // For rerankers, we encode (query, document) pairs together
        let pairs: Vec<(String, String)> = documents
            .iter()
            .map(|doc| (query.to_string(), doc.clone()))
            .collect();

        debug!("Tokenizing {} query-document pairs", pairs.len());
        let tokens = tokenizer
            .encode_batch(pairs, true)
            .map_err(E::msg)?;

        debug!("Building input tensors");
        let token_ids: Vec<Tensor> = tokens
            .iter()
            .map(|tokens| Tensor::new(tokens.get_ids(), &self.device))
            .collect::<candle_core::Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        
        // For cross-encoders, token_type_ids distinguish query (0) from document (1)
        let token_type_ids: Vec<Tensor> = tokens
            .iter()
            .map(|tokens| Tensor::new(tokens.get_type_ids(), &self.device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let token_type_ids = Tensor::stack(&token_type_ids, 0)?;

        let attention_mask: Vec<Tensor> = tokens
            .iter()
            .map(|tokens| Tensor::new(tokens.get_attention_mask(), &self.device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;

        // Run the model
        debug!("Running BERT model forward pass");
        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Debug: print tensor shapes
        debug!("embeddings shape: {:?}", embeddings.shape());

        // For rerankers, we use the [CLS] token's output (position 0 in sequence)
        // embeddings shape: [batch_size, seq_len, hidden_size]
        // We want: [batch_size, hidden_size]
        let (batch_size, _seq_len, hidden_size) = embeddings.dims3()?;
        let cls_output = embeddings.narrow(1, 0, 1)?.reshape((batch_size, hidden_size))?;
        
        debug!("cls_output shape: {:?}", cls_output.shape());

        // Pass through the classifier head to get relevance logits
        let logits = self.classifier.forward(&cls_output)?; // Shape: [batch_size, 1]
        
        debug!("logits shape: {:?}", logits.shape());

        let logits = logits.squeeze(1)?; // Shape: [batch_size]
        
        // Get the raw scores
        let scores = logits.to_vec1::<f32>()?;

        debug!("Raw logits computed: {:?}", scores);

        // Apply sigmoid to convert logits to probabilities (0-1 range)
        let sigmoid_scores: Vec<f32> = scores.iter().map(|&x| sigmoid(x)).collect();

        // Apply min-max normalization to spread scores across 0.0-1.0 range
        // This makes scores more interpretable for ranking purposes
        let scores = normalize_scores(&sigmoid_scores);

        info!(
            num_scores = scores.len(),
            min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min),
            max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            avg_score = scores.iter().sum::<f32>() / scores.len() as f32,
            "Batch scoring completed (normalized)"
        );

        Ok(scores)
    }

    /// Scores a single query-document pair.
    pub fn score_single(&self, query: &str, document: &str) -> Result<f32> {
        debug!("Scoring single query-document pair");
        let scores = self.score_pairs(query, &[document.to_string()])?;
        let score = scores.into_iter().next().ok_or_else(|| E::msg("No score returned"))?;
        debug!(score = %score, "Single pair score computed");
        Ok(score)
    }
}

/// Sigmoid activation function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Min-max normalization to spread scores across 0.0-1.0 range.
/// This makes relative ranking scores more interpretable.
/// If all scores are identical, returns 0.5 for all.
fn normalize_scores(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return vec![];
    }
    
    let min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_score - min_score;
    
    if range < 1e-9 {
        // All scores are effectively identical
        return vec![0.5; scores.len()];
    }
    
    scores.iter().map(|&s| (s - min_score) / range).collect()
}

