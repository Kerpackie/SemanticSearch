use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use tokenizers::Tokenizer;

pub struct ClipEmbeddingModel {
    model: clip::ClipModel,
    tokenizer: Tokenizer,
    pub device: Device,
    image_size: usize,
}

impl ClipEmbeddingModel {
    /// Creates a new model from the HuggingFace hub.
    pub fn new(model_id: &str) -> Result<Self> {
        let device = if candle_core::utils::metal_is_available() {
            Device::new_metal(0)?
        } else {
            Device::Cpu
        };

        // Use the exact repository details from the working example
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "refs/pr/15".to_string(), // Specific revision from the example
        ));

        let model_filename = repo.get("model.safetensors")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;

        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Use the hardcoded config from the example, which is simpler and more reliable
        let config = clip::ClipConfig::vit_base_patch32();
        let image_size = config.image_size;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_filename], DType::F32, &device)? };
        // Use the simpler constructor from the example
        let model = clip::ClipModel::new(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            image_size,
        })
    }

    /// Generates embeddings for a batch of text.
    pub fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let pad_id = *self
            .tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or(E::msg("No pad token"))?;

        let mut tokens = vec![];
        for text in texts {
            let encoding = self.tokenizer.encode(text.clone(), true).map_err(E::msg)?;
            tokens.push(encoding.get_ids().to_vec());
        }

        let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);
        for token_vec in tokens.iter_mut() {
            let len_diff = max_len - token_vec.len();
            if len_diff > 0 {
                token_vec.extend(vec![pad_id; len_diff]);
            }
        }

        let token_ids = Tensor::new(tokens, &self.device)?;
        let embeddings = self.model.get_text_features(&token_ids)?;

        // Normalize embeddings, which is crucial for similarity search
        let embeddings = normalize_l2(&embeddings)?;
        Ok(embeddings.to_vec2()?)
    }

    /// Generates embeddings for a batch of images provided as raw bytes.
    pub fn embed_images(&self, image_bytes_batch: &[Vec<u8>]) -> Result<Vec<Vec<f32>>> {
        let mut image_tensors = vec![];
        for image_bytes in image_bytes_batch {
            let tensor = self.preprocess_image(image_bytes)?;
            image_tensors.push(tensor);
        }
        let image_tensors = Tensor::stack(&image_tensors, 0)?.to_device(&self.device)?;

        let embeddings = self.model.get_image_features(&image_tensors)?;
        let embeddings = normalize_l2(&embeddings)?;
        Ok(embeddings.to_vec2()?)
    }

    /// Preprocesses a single image from bytes into a tensor.
    /// This logic is taken directly from the working example you provided.
    fn preprocess_image(&self, image_bytes: &[u8]) -> Result<Tensor> {
        let img = image::load_from_memory(image_bytes)?;
        let (height, width) = (self.image_size, self.image_size);

        let img = img.resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );
        let img = img.to_rgb8();
        let img_data = img.into_raw();

        let tensor = Tensor::from_vec(img_data, (height, width, 3), &Device::Cpu)?
            .permute((2, 0, 1))?
            .to_dtype(DType::F32)?
            .affine(2. / 255., -1.)?; // Normalization
        Ok(tensor)
    }
}

/// Helper function to normalize the embeddings.
fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}