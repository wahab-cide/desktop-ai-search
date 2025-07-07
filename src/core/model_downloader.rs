use crate::error::{AppError, Result};
use crate::models::ModelInfo;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use reqwest;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

/// Model manifest containing available models for download
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub models: HashMap<String, ModelEntry>,
    pub version: String,
    pub last_updated: String,
}

/// Individual model entry in the manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub repo_id: String,
    pub filename: String,
    pub size_bytes: u64,
    pub quant_type: String,
    pub context_size: usize,
    pub parameter_count: String,
    pub license: String,
    pub description: String,
    pub sha256: String,
    pub recommended: bool,
}

/// Progress callback for downloads
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send + Sync>;

/// Model downloader for managing LLM model downloads
pub struct ModelDownloader {
    models_dir: PathBuf,
    client: reqwest::Client,
    manifest: Option<ModelManifest>,
}

impl ModelDownloader {
    /// Create a new model downloader
    pub fn new(models_dir: PathBuf) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent("aisearch/0.1.0 (Desktop AI Search Engine; https://github.com/user/aisearch)")
            .timeout(std::time::Duration::from_secs(600)) // 10 minute timeout for large models
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .map_err(|e| AppError::Configuration(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            models_dir,
            client,
            manifest: None,
        })
    }

    /// Download and load the model manifest
    pub async fn load_manifest(&mut self) -> Result<()> {
        // For now, we'll use a hardcoded manifest. In production, this would be downloaded
        let manifest = ModelManifest {
            version: "1.0.0".to_string(),
            last_updated: "2024-01-01".to_string(),
            models: self.create_default_manifest(),
        };

        self.manifest = Some(manifest);
        Ok(())
    }

    /// Create a default manifest with real available GGUF models
    fn create_default_manifest(&self) -> HashMap<String, ModelEntry> {
        let mut models = HashMap::new();

        // These are real GGUF models available on HuggingFace
        
        // Phi-3 Mini 4K Instruct Q4_K_M (Microsoft, no auth needed)
        models.insert("phi-3-mini-4k-instruct-q4_k_m".to_string(), ModelEntry {
            name: "Phi-3 Mini 4K Instruct Q4_K_M".to_string(),
            repo_id: "microsoft/Phi-3-mini-4k-instruct-gguf".to_string(),
            filename: "Phi-3-mini-4k-instruct-q4_k_m.gguf".to_string(),
            size_bytes: 2_174_000_000, // ~2.17 GB
            quant_type: "Q4_K_M".to_string(),
            context_size: 4096,
            parameter_count: "3.8B".to_string(),
            license: "MIT".to_string(),
            description: "Microsoft Phi-3 Mini, highly capable small model".to_string(),
            sha256: "".to_string(), // Will be computed after download
            recommended: true,
        });

        // Llama 3.2 1B Instruct Q4_K_M (Meta, public)
        models.insert("llama-3.2-1b-instruct-q4_k_m".to_string(), ModelEntry {
            name: "Llama 3.2 1B Instruct Q4_K_M".to_string(),
            repo_id: "huggingfaceh4/llama-3.2-1b-instruct-gguf".to_string(),
            filename: "llama-3.2-1b-instruct-q4_k_m.gguf".to_string(),
            size_bytes: 680_000_000, // ~680 MB
            quant_type: "Q4_K_M".to_string(),
            context_size: 131072,
            parameter_count: "1B".to_string(),
            license: "Llama 3.2".to_string(),
            description: "Llama 3.2 1B, very efficient small model".to_string(),
            sha256: "".to_string(),
            recommended: true,
        });

        // Qwen2.5 0.5B Instruct Q4_K_M (Alibaba, public)
        models.insert("qwen2.5-0.5b-instruct-q4_k_m".to_string(), ModelEntry {
            name: "Qwen2.5 0.5B Instruct Q4_K_M".to_string(),
            repo_id: "Qwen/Qwen2.5-0.5B-Instruct-GGUF".to_string(),
            filename: "qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string(),
            size_bytes: 352_000_000, // ~352 MB
            quant_type: "Q4_K_M".to_string(),
            context_size: 32768,
            parameter_count: "0.5B".to_string(),
            license: "Qwen".to_string(),
            description: "Qwen2.5 0.5B, ultra-lightweight model".to_string(),
            sha256: "".to_string(),
            recommended: false,
        });

        // Gemma 2 2B Instruct Q4_K_M (Google, public)
        models.insert("gemma-2-2b-instruct-q4_k_m".to_string(), ModelEntry {
            name: "Gemma 2 2B Instruct Q4_K_M".to_string(),
            repo_id: "bartowski/gemma-2-2b-it-GGUF".to_string(),
            filename: "gemma-2-2b-it-Q4_K_M.gguf".to_string(),
            size_bytes: 1_600_000_000, // ~1.6 GB
            quant_type: "Q4_K_M".to_string(),
            context_size: 8192,
            parameter_count: "2B".to_string(),
            license: "Gemma".to_string(),
            description: "Google Gemma 2 2B instruction-tuned model".to_string(),
            sha256: "".to_string(),
            recommended: false,
        });

        models
    }

    /// Get available models from manifest
    pub fn get_available_models(&self) -> Result<Vec<ModelEntry>> {
        let manifest = self.manifest.as_ref().ok_or_else(|| {
            AppError::Configuration("Model manifest not loaded".to_string())
        })?;

        Ok(manifest.models.values().cloned().collect())
    }

    /// Get recommended models
    pub fn get_recommended_models(&self) -> Result<Vec<ModelEntry>> {
        let models = self.get_available_models()?;
        Ok(models.into_iter().filter(|m| m.recommended).collect())
    }

    /// Check if a model is already downloaded
    pub async fn is_model_downloaded(&self, model_id: &str) -> Result<bool> {
        let manifest = self.manifest.as_ref().ok_or_else(|| {
            AppError::Configuration("Model manifest not loaded".to_string())
        })?;

        let model_entry = manifest.models.get(model_id).ok_or_else(|| {
            AppError::Configuration(format!("Model not found in manifest: {}", model_id))
        })?;

        let model_path = self.models_dir.join(&model_entry.filename);
        Ok(model_path.exists())
    }

    /// Download a model with progress callback
    pub async fn download_model(
        &self,
        model_id: &str,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<PathBuf> {
        let manifest = self.manifest.as_ref().ok_or_else(|| {
            AppError::Configuration("Model manifest not loaded".to_string())
        })?;

        let model_entry = manifest.models.get(model_id).ok_or_else(|| {
            AppError::Configuration(format!("Model not found in manifest: {}", model_id))
        })?;

        let model_path = self.models_dir.join(&model_entry.filename);
        
        // Check if already downloaded
        if model_path.exists() {
            return Ok(model_path);
        }

        // Ensure models directory exists
        fs::create_dir_all(&self.models_dir).await?;

        // Construct HuggingFace download URL
        let download_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model_entry.repo_id,
            model_entry.filename
        );

        println!("Downloading {} from {}", model_entry.name, download_url);

        // Start the download
        let response = self.client.get(&download_url).send().await
            .map_err(|e| AppError::Configuration(format!("Download request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(AppError::Configuration(format!(
                "Download failed with status: {}", 
                response.status()
            )));
        }

        let total_size = response.content_length().unwrap_or(model_entry.size_bytes);
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();

        // Create temporary file
        let temp_path = model_path.with_extension("tmp");
        let mut file = fs::File::create(&temp_path).await?;

        // Download with progress tracking
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                AppError::Configuration(format!("Error reading download stream: {}", e))
            })?;
            
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            if let Some(ref callback) = progress_callback {
                callback(downloaded, total_size);
            }
        }

        file.flush().await?;
        drop(file);

        // Verify file size
        let actual_size = fs::metadata(&temp_path).await?.len();
        if actual_size != total_size {
            fs::remove_file(&temp_path).await?;
            return Err(AppError::Configuration(format!(
                "Download size mismatch: expected {}, got {}", 
                total_size, 
                actual_size
            )));
        }

        // TODO: Verify SHA256 hash
        // let computed_hash = self.compute_file_hash(&temp_path).await?;
        // if computed_hash != model_entry.sha256 {
        //     fs::remove_file(&temp_path).await?;
        //     return Err(AppError::Configuration("SHA256 hash verification failed".to_string()));
        // }

        // Move to final location
        fs::rename(&temp_path, &model_path).await?;

        println!("Successfully downloaded {}", model_entry.name);
        Ok(model_path)
    }

    /// Compute SHA256 hash of a file
    async fn compute_file_hash(&self, path: &Path) -> Result<String> {
        let content = fs::read(path).await?;
        let mut hasher = Sha256::new();
        hasher.update(&content);
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Delete a downloaded model
    pub async fn delete_model(&self, model_id: &str) -> Result<()> {
        let manifest = self.manifest.as_ref().ok_or_else(|| {
            AppError::Configuration("Model manifest not loaded".to_string())
        })?;

        let model_entry = manifest.models.get(model_id).ok_or_else(|| {
            AppError::Configuration(format!("Model not found in manifest: {}", model_id))
        })?;

        let model_path = self.models_dir.join(&model_entry.filename);
        
        if model_path.exists() {
            fs::remove_file(&model_path).await?;
            println!("Deleted model: {}", model_entry.name);
        }

        Ok(())
    }

    /// Get the total size of all downloaded models
    pub async fn get_downloaded_models_size(&self) -> Result<u64> {
        let mut total_size = 0u64;
        
        if !self.models_dir.exists() {
            return Ok(0);
        }

        let mut entries = fs::read_dir(&self.models_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let metadata = entry.metadata().await?;
            if metadata.is_file() {
                total_size += metadata.len();
            }
        }

        Ok(total_size)
    }

    /// Get information about all downloaded models
    pub async fn get_downloaded_models(&self) -> Result<Vec<ModelInfo>> {
        let mut models = Vec::new();
        
        if !self.models_dir.exists() {
            return Ok(models);
        }

        let mut entries = fs::read_dir(&self.models_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                let metadata = entry.metadata().await?;
                let name = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                models.push(ModelInfo {
                    name,
                    path,
                    size_bytes: metadata.len(),
                    quant_type: "unknown".to_string(), // Would be parsed from GGUF
                    context_size: 2048, // Would be parsed from GGUF
                    parameter_count: None,
                    license: None,
                    description: None,
                    sha256: None,
                });
            }
        }

        Ok(models)
    }

    /// Format bytes as human-readable string
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        const THRESHOLD: u64 = 1024;

        if bytes < THRESHOLD {
            return format!("{} B", bytes);
        }

        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= THRESHOLD as f64 && unit_index < UNITS.len() - 1 {
            size /= THRESHOLD as f64;
            unit_index += 1;
        }

        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

impl Default for ModelDownloader {
    fn default() -> Self {
        let models_dir = PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("Library")
            .join("Application Support")
            .join("aisearch")
            .join("models");
        
        Self::new(models_dir).expect("Failed to create ModelDownloader")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_model_downloader_creation() {
        let temp_dir = tempdir().unwrap();
        let downloader = ModelDownloader::new(temp_dir.path().to_path_buf());
        assert!(downloader.is_ok());
    }

    #[tokio::test]
    async fn test_manifest_loading() {
        let temp_dir = tempdir().unwrap();
        let mut downloader = ModelDownloader::new(temp_dir.path().to_path_buf()).unwrap();
        
        let result = downloader.load_manifest().await;
        assert!(result.is_ok());
        
        let models = downloader.get_available_models();
        assert!(models.is_ok());
        assert!(!models.unwrap().is_empty());
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(ModelDownloader::format_bytes(512), "512 B");
        assert_eq!(ModelDownloader::format_bytes(1024), "1.0 KB");
        assert_eq!(ModelDownloader::format_bytes(1536), "1.5 KB");
        assert_eq!(ModelDownloader::format_bytes(1048576), "1.0 MB");
        assert_eq!(ModelDownloader::format_bytes(1073741824), "1.0 GB");
    }
}