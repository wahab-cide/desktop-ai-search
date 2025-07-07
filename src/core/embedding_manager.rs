use crate::error::{AppError, Result};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use tokio::fs;
use reqwest;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use tokenizers::Tokenizer;

/// Information about an available embedding model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelInfo {
    pub id: String,
    pub name: String,
    pub dimensions: usize,
    pub max_sequence_length: usize,
    pub model_size_mb: u64,
    pub description: String,
    pub huggingface_id: String,
}

/// Configuration for embedding generation
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub normalize: bool,
    pub max_length: Option<usize>,
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            max_length: Some(512),
            batch_size: 32,
        }
    }
}

/// Manages embedding model downloading, loading, and inference
pub struct EmbeddingManager {
    models_dir: PathBuf,
    available_models: HashMap<String, EmbeddingModelInfo>,
    current_model: Option<LoadedEmbeddingModel>,
    client: reqwest::Client,
    device: Device,
}

/// A loaded embedding model ready for inference
struct LoadedEmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    config: EmbeddingConfig,
    info: EmbeddingModelInfo,
}

impl EmbeddingManager {
    /// Create a new embedding manager
    pub fn new() -> Result<Self> {
        let models_dir = Self::get_models_directory()?;
        std::fs::create_dir_all(&models_dir)?;

        let client = reqwest::Client::builder()
            .user_agent("aisearch/0.1.0 (Desktop AI Search Engine)")
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| AppError::Configuration(format!("Failed to create HTTP client: {}", e)))?;

        let device = Device::Cpu; // TODO: Add GPU detection
        
        Ok(Self {
            models_dir,
            available_models: Self::create_default_models(),
            current_model: None,
            client,
            device,
        })
    }

    /// Get the models directory path
    fn get_models_directory() -> Result<PathBuf> {
        let home = std::env::var("HOME").map_err(|_| {
            AppError::Configuration("Could not determine home directory".to_string())
        })?;
        
        Ok(PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("aisearch")
            .join("embedding-models"))
    }

    /// Create the default set of available embedding models
    fn create_default_models() -> HashMap<String, EmbeddingModelInfo> {
        let mut models = HashMap::new();

        // all-MiniLM-L6-v2 - Our primary model
        models.insert("all-minilm-l6-v2".to_string(), EmbeddingModelInfo {
            id: "all-minilm-l6-v2".to_string(),
            name: "All MiniLM L6 v2".to_string(),
            dimensions: 384,
            max_sequence_length: 512,
            model_size_mb: 22,
            description: "Fast, lightweight sentence transformer model".to_string(),
            huggingface_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        });

        // all-MiniLM-L12-v2 - Larger, more accurate
        models.insert("all-minilm-l12-v2".to_string(), EmbeddingModelInfo {
            id: "all-minilm-l12-v2".to_string(),
            name: "All MiniLM L12 v2".to_string(),
            dimensions: 384,
            max_sequence_length: 512,
            model_size_mb: 45,
            description: "Higher accuracy sentence transformer model".to_string(),
            huggingface_id: "sentence-transformers/all-MiniLM-L12-v2".to_string(),
        });

        // all-mpnet-base-v2 - High quality embeddings
        models.insert("all-mpnet-base-v2".to_string(), EmbeddingModelInfo {
            id: "all-mpnet-base-v2".to_string(),
            name: "All MPNet Base v2".to_string(),
            dimensions: 768,
            max_sequence_length: 512,
            model_size_mb: 420,
            description: "High-quality embeddings, larger model".to_string(),
            huggingface_id: "sentence-transformers/all-mpnet-base-v2".to_string(),
        });

        models
    }

    /// Get list of available embedding models
    pub fn get_available_models(&self) -> Vec<&EmbeddingModelInfo> {
        self.available_models.values().collect()
    }

    /// Check if a model is already downloaded
    pub async fn is_model_downloaded(&self, model_id: &str) -> Result<bool> {
        let model_info = self.available_models.get(model_id).ok_or_else(|| {
            AppError::Configuration(format!("Unknown model: {}", model_id))
        })?;

        let model_dir = self.models_dir.join(&model_info.id);
        let config_path = model_dir.join("config.json");
        let model_path = model_dir.join("model.safetensors");
        let tokenizer_path = model_dir.join("tokenizer.json");

        Ok(config_path.exists() && model_path.exists() && tokenizer_path.exists())
    }

    /// Download an embedding model from HuggingFace
    pub async fn download_model(&self, model_id: &str) -> Result<()> {
        let model_info = self.available_models.get(model_id).ok_or_else(|| {
            AppError::Configuration(format!("Unknown model: {}", model_id))
        })?;

        if self.is_model_downloaded(model_id).await? {
            println!("Model '{}' is already downloaded", model_info.name);
            return Ok(());
        }

        let model_dir = self.models_dir.join(&model_info.id);
        fs::create_dir_all(&model_dir).await?;

        println!("Downloading embedding model: {}", model_info.name);

        // Download required files - try safetensors format first
        let files = vec![
            ("config.json", "config.json"),
            ("model.safetensors", "model.safetensors"),
            ("tokenizer.json", "tokenizer.json"),
            ("tokenizer_config.json", "tokenizer_config.json"),
        ];

        for (local_name, remote_name) in files {
            let url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                model_info.huggingface_id, remote_name
            );

            let file_path = model_dir.join(local_name);
            
            match self.download_file(&url, &file_path).await {
                Ok(_) => println!("  ✓ Downloaded {}", local_name),
                Err(e) => {
                    // Try alternative names for model files
                    if local_name == "model.safetensors" {
                        // Try pytorch_model.bin as fallback
                        let alt_url = format!(
                            "https://huggingface.co/{}/resolve/main/pytorch_model.bin",
                            model_info.huggingface_id
                        );
                        
                        match self.download_file(&alt_url, &file_path).await {
                            Ok(_) => println!("  ✓ Downloaded {} (pytorch fallback)", local_name),
                            Err(_) => return Err(e),
                        }
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        println!("✅ Model '{}' downloaded successfully", model_info.name);
        Ok(())
    }

    /// Download a single file with progress
    async fn download_file(&self, url: &str, path: &Path) -> Result<()> {
        let response = self.client.get(url).send().await
            .map_err(|e| AppError::Configuration(format!("Download request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(AppError::Configuration(format!(
                "Download failed with status: {}", 
                response.status()
            )));
        }

        let mut stream = response.bytes_stream();
        let mut file = fs::File::create(path).await?;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                AppError::Configuration(format!("Error reading download stream: {}", e))
            })?;
            
            tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;
        }

        tokio::io::AsyncWriteExt::flush(&mut file).await?;
        Ok(())
    }

    /// Load a model for inference
    pub async fn load_model(&mut self, model_id: &str, config: Option<EmbeddingConfig>) -> Result<()> {
        let model_info = self.available_models.get(model_id).ok_or_else(|| {
            AppError::Configuration(format!("Unknown model: {}", model_id))
        })?;

        if !self.is_model_downloaded(model_id).await? {
            return Err(AppError::Configuration(format!(
                "Model '{}' is not downloaded. Use download_model() first.", model_id
            )));
        }

        let model_dir = self.models_dir.join(&model_info.id);
        
        println!("Loading embedding model: {}", model_info.name);

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| AppError::Configuration(format!("Failed to load tokenizer: {}", e)))?;

        // Load model config
        let config_path = model_dir.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)?;
        let bert_config: BertConfig = serde_json::from_str(&config_content)
            .map_err(|e| AppError::Configuration(format!("Failed to parse model config: {}", e)))?;

        // Load model weights
        let model_path = model_dir.join("model.safetensors");
        let vb = unsafe { 
            VarBuilder::from_mmaped_safetensors(&[model_path], candle_core::DType::F32, &self.device)?
        };
        
        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| AppError::Configuration(format!("Failed to load BERT model: {}", e)))?;

        self.current_model = Some(LoadedEmbeddingModel {
            model,
            tokenizer,
            config: config.unwrap_or_default(),
            info: model_info.clone(),
        });

        println!("✅ Model '{}' loaded successfully", model_info.name);
        Ok(())
    }

    /// Generate embeddings for a list of texts
    pub async fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let loaded_model = self.current_model.as_ref().ok_or_else(|| {
            AppError::Configuration("No model loaded. Use load_model() first.".to_string())
        })?;

        let mut all_embeddings = Vec::new();

        // Process texts in batches
        for chunk in texts.chunks(loaded_model.config.batch_size) {
            let batch_embeddings = self.process_batch(chunk, loaded_model).await?;
            all_embeddings.extend(batch_embeddings);
        }

        Ok(all_embeddings)
    }

    /// Process a batch of texts and return embeddings
    async fn process_batch(&self, texts: &[String], loaded_model: &LoadedEmbeddingModel) -> Result<Vec<Vec<f32>>> {
        // Tokenize texts
        let encoded = loaded_model.tokenizer.encode_batch(texts.to_vec(), true)
            .map_err(|e| AppError::Configuration(format!("Tokenization failed: {}", e)))?;

        // Convert to tensors
        let mut token_ids = Vec::new();
        let mut attention_masks = Vec::new();

        for encoding in &encoded {
            let ids = encoding.get_ids();
            let attention_mask = encoding.get_attention_mask();

            // Convert u32 token IDs to i64 for BERT model
            let ids_i64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
            let attention_mask_i64: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();

            token_ids.push(Tensor::new(ids_i64, &self.device)?);
            attention_masks.push(Tensor::new(attention_mask_i64, &self.device)?);
        }

        // Stack tensors
        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_masks = Tensor::stack(&attention_masks, 0)?;

        // Run inference
        let embeddings = loaded_model.model.forward(&token_ids, &attention_masks, None)?;
        
        // Mean pooling (average over sequence length)
        let pooled = self.mean_pooling(&embeddings, &attention_masks)?;

        // Convert to Vec<Vec<f32>>
        let embeddings_data = pooled.to_vec2::<f32>()?;
        
        // Normalize if requested
        let mut result = embeddings_data;
        if loaded_model.config.normalize {
            for embedding in &mut result {
                Self::normalize_embedding(embedding);
            }
        }

        Ok(result)
    }

    /// Perform mean pooling on embeddings
    fn mean_pooling(&self, embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Convert attention mask to f32 and add dimension for broadcasting
        let attention_mask_f32 = attention_mask.to_dtype(candle_core::DType::F32)?;
        let attention_mask = attention_mask_f32.unsqueeze(2)?;
        let masked_embeddings = embeddings.broadcast_mul(&attention_mask)?;
        
        // Sum over sequence length
        let sum_embeddings = masked_embeddings.sum(1)?;
        
        // Calculate attention mask sum for normalization
        let sum_mask = attention_mask.sum(1)?;
        
        // Avoid division by zero
        let sum_mask = sum_mask.clamp(1e-9, f32::INFINITY)?;
        
        // Divide to get mean
        let mean_pooled = sum_embeddings.broadcast_div(&sum_mask)?;
        
        Ok(mean_pooled)
    }

    /// Normalize an embedding vector (L2 normalization)
    fn normalize_embedding(embedding: &mut [f32]) {
        let length: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if length > 0.0 {
            embedding.iter_mut().for_each(|x| *x /= length);
        }
    }

    /// Get information about the currently loaded model
    pub fn get_current_model_info(&self) -> Option<&EmbeddingModelInfo> {
        self.current_model.as_ref().map(|m| &m.info)
    }

    /// Check if a model is currently loaded
    pub fn is_model_loaded(&self) -> bool {
        self.current_model.is_some()
    }

    /// Unload the current model to free memory
    pub fn unload_model(&mut self) {
        self.current_model = None;
    }

    /// Compute cosine similarity between two embeddings
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embedding_manager_creation() {
        let manager = EmbeddingManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(EmbeddingManager::cosine_similarity(&a, &b), 1.0);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(EmbeddingManager::cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_normalize_embedding() {
        let mut embedding = vec![3.0, 4.0, 0.0];
        EmbeddingManager::normalize_embedding(&mut embedding);
        let expected = vec![0.6, 0.8, 0.0];
        for (a, b) in embedding.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}