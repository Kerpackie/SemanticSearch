use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::sync::Arc;
use tokenizers::Tokenizer;
use anyhow::{Error as E, Result};
use crate::utils::normalize_l2;

pub struct EmbeddingModel {
    pub model: BertModel,
    pub tokenizer: Tokenizer,
    pub device: Device,
}

impl EmbeddingModel {
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
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    // In src/main.rs -> impl EmbeddingModel

    pub fn embed_batch(&self, sentences: &[String]) -> Result<Vec<Vec<f32>>> {
        // It's better to clone the tokenizer once and configure it.
        let mut tokenizer = self.tokenizer.clone();

        // ⚙️ 1. CONFIGURE PADDING
        // This is the crucial step. We tell the tokenizer to find the longest
        // sentence in the batch and pad all other sentences to that length.
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest;
        } else {
            let pp = tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }

        // This will now produce tokens where each sequence has the same length.
        let tokens = tokenizer.encode_batch(sentences.to_vec(), true).map_err(E::msg)?;

        let token_ids: Vec<Tensor> = tokens
            .iter()
            .map(|tokens| Tensor::new(tokens.get_ids(), &self.device))
            .collect::<candle_core::Result<Vec<_>>>()?;

        // This `stack` operation will now succeed because all tensors have the same shape.
        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        // ✅ 2. GENERATE THE ATTENTION MASK
        // The attention mask tells the model to ignore the padding tokens.
        let attention_mask: Vec<Tensor> = tokens
            .iter()
            .map(|tokens| Tensor::new(tokens.get_attention_mask(), &self.device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;

        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // 3. POOLING WITH THE MASK
        // We use the attention mask to ensure padding tokens don't affect the final average embedding.
        let attention_mask = attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;
        let embeddings = embeddings.broadcast_mul(&attention_mask)?.sum(1)?;
        let sum_mask = attention_mask.sum(1)?;
        let embeddings = embeddings.broadcast_div(&sum_mask)?;

        let normalized_embeddings = normalize_l2(&embeddings)?;

        Ok(normalized_embeddings.to_vec2()?)
    }
}
