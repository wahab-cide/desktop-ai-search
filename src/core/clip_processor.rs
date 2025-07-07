use crate::error::{AppError, IndexingError, Result};
use candle_core::{Device, Tensor, DType, Result as CandleResult};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{ClipModel, ClipConfig as CandleClipConfig};
use hf_hub::api::tokio::Api;
use image::{DynamicImage, ImageBuffer, Rgb};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipProcessorConfig {
    pub model_id: String,
    pub revision: String,
    pub embedding_dim: usize,
    pub image_size: u32,
    pub normalize_embeddings: bool,
    pub batch_size: usize,
}

impl Default for ClipProcessorConfig {
    fn default() -> Self {
        Self {
            model_id: "openai/clip-vit-base-patch32".to_string(),
            revision: "main".to_string(),
            embedding_dim: 512,
            image_size: 224,
            normalize_embeddings: true,
            batch_size: 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImageEmbedding {
    pub image_path: String,
    pub embedding: Vec<f32>,
    pub model_id: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct TextImageSimilarity {
    pub image_path: String,
    pub similarity_score: f32,
    pub text_query: String,
}

pub struct ClipProcessor {
    model: Option<ClipModel>,
    device: Device,
    config: ClipProcessorConfig,
    models_dir: PathBuf,
    current_model_info: Option<ClipModelInfo>,
    embedding_cache: Arc<Mutex<lru::LruCache<String, Vec<f32>>>>,
}

#[derive(Debug, Clone)]
pub struct ClipModelInfo {
    pub model_id: String,
    pub embedding_dim: usize,
    pub loaded_at: std::time::Instant,
}

impl ClipProcessor {
    pub fn new() -> Result<Self> {
        let device = Device::new_metal(0).unwrap_or_else(|_| Device::Cpu);
        let models_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".cache")
            .join("aisearch")
            .join("clip_models");

        std::fs::create_dir_all(&models_dir)?;

        Ok(Self {
            model: None,
            device,
            config: ClipProcessorConfig::default(),
            models_dir,
            current_model_info: None,
            embedding_cache: Arc::new(Mutex::new(
                lru::LruCache::new(std::num::NonZeroUsize::new(256).unwrap())
            )),
        })
    }

    pub fn with_config(mut self, config: ClipProcessorConfig) -> Self {
        self.config = config;
        self
    }

    /// Download and load CLIP model
    pub async fn load_model(&mut self, model_id: Option<&str>) -> Result<()> {
        let model_id = model_id.unwrap_or(&self.config.model_id);
        
        println!("Loading CLIP model: {}", model_id);
        
        // Download model files if not cached
        let model_path = self.download_model(model_id).await?;
        
        // Load the model - try both .safetensors and .bin formats
        let config_path = model_path.join("config.json");
        let safetensors_path = model_path.join("model.safetensors");
        let pytorch_path = model_path.join("pytorch_model.bin");
        
        let model_weights_path = if safetensors_path.exists() {
            safetensors_path
        } else if pytorch_path.exists() {
            pytorch_path
        } else {
            return Err(AppError::Indexing(IndexingError::Processing(
                format!("Missing model weight files for {} (tried both .safetensors and .bin)", model_id)
            )));
        };
        
        if !config_path.exists() {
            return Err(AppError::Indexing(IndexingError::Processing(
                format!("Missing config.json for {}", model_id)
            )));
        }

        // Load the actual CLIP model using candle-transformers
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_weights_path], DType::F32, &self.device)
                .map_err(|e| AppError::Indexing(IndexingError::Processing(
                    format!("Failed to load model weights: {}", e)
                )))?
        };
        
        let clip_config = CandleClipConfig::vit_base_patch32();
        let model = ClipModel::new(vb, &clip_config)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to create CLIP model: {}", e)
            )))?;
            
        self.model = Some(model);
        self.current_model_info = Some(ClipModelInfo {
            model_id: model_id.to_string(),
            embedding_dim: self.config.embedding_dim,
            loaded_at: std::time::Instant::now(),
        });

        println!("✅ CLIP model '{}' loaded successfully", model_id);
        Ok(())
    }

    /// Download CLIP model from HuggingFace
    async fn download_model(&self, model_id: &str) -> Result<PathBuf> {
        let model_path = self.models_dir.join(model_id.replace("/", "--"));
        
        if model_path.exists() {
            println!("Using cached model at: {}", model_path.display());
            return Ok(model_path);
        }

        println!("Downloading CLIP model: {}", model_id);
        std::fs::create_dir_all(&model_path)?;

        let api = Api::new().map_err(|e| AppError::Indexing(IndexingError::Processing(
            format!("Failed to create HF Hub API: {}", e)
        )))?;
        let repo = api.model(model_id.to_string());

        // Download required files for CLIP models
        let files_to_download = vec![
            "config.json",
            "pytorch_model.bin",  // CLIP models often use .bin instead of .safetensors
            "tokenizer.json",
            "preprocessor_config.json",
            "vocab.json",         // Often needed for tokenizer
            "merges.txt",         // BPE merges file
        ];

        for file in files_to_download {
            match repo.get(file).await {
                Ok(file_path) => {
                    let dest_path = model_path.join(file);
                    std::fs::copy(&file_path, &dest_path)?;
                    println!("Downloaded: {}", file);
                }
                Err(e) => {
                    println!("Warning: Could not download {}: {}", file, e);
                    // Some files are optional, continue with others
                }
            }
        }

        Ok(model_path)
    }

    /// Generate embeddings for images
    pub async fn generate_image_embeddings(&self, image_paths: &[String]) -> Result<Vec<Vec<f32>>> {
        let Some(ref model) = self.model else {
            return Err(AppError::Indexing(IndexingError::Processing(
                "CLIP model not loaded".to_string()
            )));
        };

        let mut embeddings = Vec::new();
        let mut cache = self.embedding_cache.lock().await;

        for image_path in image_paths {
            // Check cache first
            if let Some(cached_embedding) = cache.get(image_path) {
                embeddings.push(cached_embedding.clone());
                continue;
            }

            // Load and preprocess image
            let image = self.load_and_preprocess_image(image_path).await?;
            
            // Convert to tensor
            let image_tensor = self.image_to_tensor(&image)?;
            
            // Generate embedding using CLIP model
            let embedding_vec = self.encode_image(&image_tensor).await?;

            // Normalize if configured
            let embedding_vec = if self.config.normalize_embeddings {
                self.normalize_embedding(&embedding_vec)
            } else {
                embedding_vec
            };

            // Cache the embedding
            cache.put(image_path.clone(), embedding_vec.clone());
            embeddings.push(embedding_vec);
        }

        Ok(embeddings)
    }

    /// Generate embeddings for text queries
    pub async fn generate_text_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let Some(ref model) = self.model else {
            return Err(AppError::Indexing(IndexingError::Processing(
                "CLIP model not loaded".to_string()
            )));
        };

        let mut embeddings = Vec::new();
        let mut cache = self.embedding_cache.lock().await;

        for text in texts {
            // Check cache first
            let cache_key = format!("text:{}", text);
            if let Some(cached_embedding) = cache.get(&cache_key) {
                embeddings.push(cached_embedding.clone());
                continue;
            }

            // Tokenize text
            let tokens = self.tokenize_text(text)?;
            
            // Generate embedding using CLIP model
            let embedding_vec = self.encode_text(&tokens).await?;

            // Normalize if configured
            let embedding_vec = if self.config.normalize_embeddings {
                self.normalize_embedding(&embedding_vec)
            } else {
                embedding_vec
            };

            // Cache the embedding
            cache.put(cache_key, embedding_vec.clone());
            embeddings.push(embedding_vec);
        }

        Ok(embeddings)
    }

    /// Calculate similarity between text and images
    pub async fn find_similar_images(&self, text_query: &str, image_embeddings: &[(String, Vec<f32>)], top_k: usize) -> Result<Vec<TextImageSimilarity>> {
        // Generate text embedding
        let text_embeddings = self.generate_text_embeddings(&[text_query.to_string()]).await?;
        let text_embedding = &text_embeddings[0];

        // Calculate similarities
        let mut similarities: Vec<TextImageSimilarity> = image_embeddings
            .iter()
            .map(|(image_path, image_embedding)| {
                let similarity = self.cosine_similarity(text_embedding, image_embedding);
                TextImageSimilarity {
                    image_path: image_path.clone(),
                    similarity_score: similarity,
                    text_query: text_query.to_string(),
                }
            })
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top k results
        similarities.truncate(top_k);
        Ok(similarities)
    }

    /// Encode image tensor using CLIP vision encoder
    async fn encode_image(&self, image_tensor: &Tensor) -> Result<Vec<f32>> {
        let model = self.model.as_ref().ok_or_else(|| {
            AppError::Indexing(IndexingError::Processing("CLIP model not loaded".to_string()))
        })?;

        // Use the actual CLIP model to encode the image
        let image_features = model.get_image_features(image_tensor)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to encode image: {}", e)
            )))?;

        // Convert tensor to Vec<f32>
        let embedding_vec = image_features.to_vec1::<f32>()
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to convert image features to vector: {}", e)
            )))?;
            
        Ok(embedding_vec)
    }

    /// Encode text tokens using CLIP text encoder  
    async fn encode_text(&self, tokens: &Tensor) -> Result<Vec<f32>> {
        let model = self.model.as_ref().ok_or_else(|| {
            AppError::Indexing(IndexingError::Processing("CLIP model not loaded".to_string()))
        })?;

        // Use the actual CLIP model to encode the text
        let text_features = model.get_text_features(tokens)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to encode text: {}", e)
            )))?;

        // Convert tensor to Vec<f32>
        let embedding_vec = text_features.to_vec1::<f32>()
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to convert text features to vector: {}", e)
            )))?;
            
        Ok(embedding_vec)
    }

    /// Load and preprocess image for CLIP
    async fn load_and_preprocess_image(&self, image_path: &str) -> Result<DynamicImage> {
        let image = image::open(image_path)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to load image {}: {}", image_path, e)
            )))?;

        // Resize to model's expected input size
        let resized = image.resize_exact(
            self.config.image_size,
            self.config.image_size,
            image::imageops::FilterType::Lanczos3,
        );

        Ok(resized)
    }

    /// Convert image to tensor
    fn image_to_tensor(&self, image: &DynamicImage) -> Result<Tensor> {
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();
        
        // Convert to f32 and normalize to [0, 1]
        let image_data: Vec<f32> = rgb_image
            .pixels()
            .flat_map(|pixel| {
                let r = pixel[0] as f32 / 255.0;
                let g = pixel[1] as f32 / 255.0;
                let b = pixel[2] as f32 / 255.0;
                [r, g, b]
            })
            .collect();

        // Reshape to [3, height, width] (CHW format)
        let tensor = Tensor::from_vec(image_data, (height as usize, width as usize, 3), &self.device)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to create tensor: {}", e)
            )))?;

        // Transpose from HWC to CHW
        let tensor = tensor.permute((2, 0, 1))
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to permute tensor: {}", e)
            )))?;

        // Normalize with CLIP's standard normalization
        // mean = [0.48145466, 0.4578275, 0.40821073]
        // std = [0.26862954, 0.26130258, 0.27577711]
        let mean = Tensor::new(&[0.48145466f32, 0.4578275f32, 0.40821073f32], &self.device)?
            .reshape((3, 1, 1))?;
        let std = Tensor::new(&[0.26862954f32, 0.26130258f32, 0.27577711f32], &self.device)?
            .reshape((3, 1, 1))?;

        let normalized = tensor.sub(&mean)?.div(&std)?;

        // Add batch dimension
        let batched = normalized.unsqueeze(0)?;

        Ok(batched)
    }

    /// Tokenize text for CLIP
    fn tokenize_text(&self, text: &str) -> Result<Tensor> {
        // CLIP tokenizer implementation based on BPE
        // This is a simplified version - in production you'd use the actual CLIP tokenizer
        
        // Start of text token
        let mut tokens: Vec<i64> = vec![49406]; // <|startoftext|>
        
        // Simple word-level tokenization (placeholder for actual BPE)
        // In practice, this would use the actual CLIP tokenizer from HuggingFace
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words.iter().take(75) { // Leave room for start/end tokens
            // Convert word to token ID (simplified mapping)
            // In practice, this would use the actual vocabulary
            let word_hash = word.chars().map(|c| c as u32).sum::<u32>() % 49408;
            tokens.push(word_hash as i64);
        }
        
        // End of text token
        tokens.push(49407); // <|endoftext|>
        
        // Pad to CLIP's max sequence length (77 tokens)
        tokens.resize(77, 0);

        // Create tensor with shape [1, 77] for batch dimension (CLIP expects i64 input_ids)
        Tensor::new(tokens.as_slice(), &self.device)?
            .reshape(&[1, 77])
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to create token tensor: {}", e)
            )))
    }

    /// Normalize embedding vector
    fn normalize_embedding(&self, embedding: &[f32]) -> Vec<f32> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding.to_vec()
        }
    }

    /// Calculate cosine similarity between two embeddings
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Get current model information
    pub fn get_current_model_info(&self) -> Option<&ClipModelInfo> {
        self.current_model_info.as_ref()
    }

    /// Check if model is loaded
    pub fn is_model_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Clear embedding cache
    pub async fn clear_cache(&self) {
        let mut cache = self.embedding_cache.lock().await;
        cache.clear();
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let cache = self.embedding_cache.lock().await;
        (cache.len(), cache.cap().get())
    }
}

impl Default for ClipProcessor {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_clip_processor_creation() {
        let processor = ClipProcessor::new().unwrap();
        assert!(!processor.is_model_loaded());
    }

    #[tokio::test]
    async fn test_cosine_similarity() {
        let processor = ClipProcessor::new().unwrap();
        
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = processor.cosine_similarity(&a, &b);
        
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_embedding() {
        let processor = ClipProcessor::new().unwrap();
        let embedding = vec![3.0, 4.0, 0.0];
        let normalized = processor.normalize_embedding(&embedding);
        
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}