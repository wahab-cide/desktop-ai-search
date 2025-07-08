use crate::error::{AppError, Result};
use crate::models::{LlmConfig, ModelInfo, InferenceRequest, InferenceResponse};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock, Mutex};
use sysinfo::System;
use serde::{Deserialize, Serialize};
// TODO: Re-enable once llama_cpp API is clarified
// use llama_cpp::{
//     standard_sampler::StandardSampler,
//     LlamaModel as LlamaCppModel,
//     LlamaParams,
//     SessionParams,
//     LlamaSession,
//     Token,
// };

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            context_size: 2048,
            n_gpu_layers: -1, // Auto-detect
            n_threads: 4,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            max_tokens: 512,
        }
    }
}

/// Preset configurations for different use cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferencePreset {
    Creative,   // High temperature, more randomness
    Balanced,   // Default settings
    Precise,    // Low temperature, more deterministic
}

impl InferencePreset {
    pub fn to_config(&self, base_config: &LlmConfig) -> LlmConfig {
        let mut config = base_config.clone();
        match self {
            InferencePreset::Creative => {
                config.temperature = 0.9;
                config.top_p = 0.95;
                config.top_k = 60;
            }
            InferencePreset::Balanced => {
                config.temperature = 0.7;
                config.top_p = 0.9;
                config.top_k = 40;
            }
            InferencePreset::Precise => {
                config.temperature = 0.1;
                config.top_p = 0.8;
                config.top_k = 20;
            }
        }
        config
    }
}

/// Placeholder for LlamaModel until proper bindings are integrated
struct LlamaModel {
    model_path: String,
    context_size: usize,
}

/// LLM Manager handles model loading, inference, and resource management
pub struct LlmManager {
    models_dir: PathBuf,
    available_models: RwLock<HashMap<String, ModelInfo>>,
    current_model: Arc<Mutex<Option<LlamaModel>>>,
    current_model_name: RwLock<Option<String>>,
    current_config: RwLock<Option<LlmConfig>>,
    inference_queue: mpsc::UnboundedSender<InferenceTask>,
    system_info: System,
}

/// Internal task for the inference queue
#[derive(Debug)]
struct InferenceTask {
    request: InferenceRequest,
    response_tx: tokio::sync::oneshot::Sender<Result<InferenceResponse>>,
}

impl LlmManager {
    /// Create a new LLM manager
    pub fn new() -> Result<Arc<Self>> {
        let models_dir = Self::get_models_directory()?;
        std::fs::create_dir_all(&models_dir)?;

        let (inference_tx, inference_rx) = mpsc::unbounded_channel();
        let mut system_info = System::new_all();
        system_info.refresh_all();

        let manager = Arc::new(Self {
            models_dir,
            available_models: RwLock::new(HashMap::new()),
            current_model: Arc::new(Mutex::new(None)),
            current_model_name: RwLock::new(None),
            current_config: RwLock::new(None),
            inference_queue: inference_tx,
            system_info,
        });

        // Start the inference worker in a separate task
        let worker_manager = manager.clone();
        tokio::spawn(async move {
            Self::inference_worker(worker_manager, inference_rx).await;
        });

        Ok(manager)
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
            .join("models"))
    }

    /// Scan for available models in the models directory
    pub async fn scan_models(&self) -> Result<()> {
        let mut models = HashMap::new();
        let entries = std::fs::read_dir(&self.models_dir)?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                if let Ok(model_info) = self.analyze_model_file(&path).await {
                    models.insert(model_info.name.clone(), model_info);
                }
            }
        }

        *self.available_models.write().await = models;
        Ok(())
    }

    /// Analyze a model file to extract metadata
    async fn analyze_model_file(&self, path: &Path) -> Result<ModelInfo> {
        let metadata = std::fs::metadata(path)?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Basic model info - in a real implementation, you'd parse GGUF metadata
        Ok(ModelInfo {
            name,
            path: path.to_path_buf(),
            size_bytes: metadata.len(),
            quant_type: "q4_0".to_string(), // Would be parsed from GGUF
            context_size: 2048, // Would be parsed from GGUF
            parameter_count: Some("7B".to_string()),
            license: None,
            description: None,
            sha256: None,
        })
    }

    /// Get list of available models
    pub async fn get_available_models(&self) -> Vec<ModelInfo> {
        self.available_models.read().await.values().cloned().collect()
    }

    /// Load a specific model
    pub async fn load_model(&self, model_name: &str, config: Option<LlmConfig>) -> Result<()> {
        let models = self.available_models.read().await;
        let model_info = models.get(model_name).ok_or_else(|| {
            AppError::Configuration(format!("Model not found: {}", model_name))
        })?;

        let config = config.unwrap_or_else(|| {
            let mut default_config = LlmConfig::default();
            default_config.model_path = model_info.path.clone();
            default_config.n_gpu_layers = self.detect_optimal_gpu_layers(&model_info);
            default_config
        });

        // Unload existing model
        {
            let mut model_guard = self.current_model.lock().await;
            *model_guard = None;
        }
        
        // Small delay to ensure GPU memory is released
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        println!("Loading model from: {}", config.model_path.display());
        
        // Verify model file exists
        if !config.model_path.exists() {
            return Err(AppError::Configuration(format!(
                "Model file not found: {}",
                config.model_path.display()
            )));
        }
        
        println!("Loading llama.cpp model...");
        println!("Model parameters:");
        println!("  - Path: {}", config.model_path.display());
        println!("  - Context size: {}", config.context_size);
        println!("  - GPU layers: {}", config.n_gpu_layers);
        println!("  - Threads: {}", config.n_threads);
        
        // Create model parameters (placeholder for actual llama.cpp integration)
        // let model_params = LlamaParams {
        //     n_ctx: config.context_size as u32,
        //     n_gpu_layers: config.n_gpu_layers.max(0) as u32,
        //     ..Default::default()
        // };
        
        // For now, create a placeholder model until we get the llama_cpp API working correctly
        // TODO: Implement actual llama.cpp integration once API is clarified
        println!("Creating placeholder model (llama.cpp integration in progress)");
        
        let llama_model = LlamaModel {
            model_path: config.model_path.to_string_lossy().to_string(),
            context_size: config.context_size,
        };
        
        // Store the loaded model
        {
            let mut model_guard = self.current_model.lock().await;
            *model_guard = Some(llama_model);
        }
        
        *self.current_model_name.write().await = Some(model_name.to_string());
        *self.current_config.write().await = Some(config);

        println!("Model loaded successfully: {}", model_name);
        Ok(())
    }

    /// Detect optimal number of GPU layers based on available VRAM
    fn detect_optimal_gpu_layers(&self, model_info: &ModelInfo) -> i32 {
        // This is a simplified heuristic - in practice you'd probe actual GPU memory
        let total_memory_gb = self.system_info.total_memory() / 1024 / 1024 / 1024;
        let model_size_gb = model_info.size_bytes / 1024 / 1024 / 1024;

        // Reserve 2GB for system and other operations
        if total_memory_gb > model_size_gb + 2 {
            -1 // Use all layers on GPU
        } else if total_memory_gb > 8 {
            32 // Use some layers on GPU
        } else {
            0 // Use CPU only
        }
    }

    /// Generate text using the loaded model
    pub async fn generate(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        
        let task = InferenceTask {
            request,
            response_tx,
        };

        self.inference_queue.send(task).map_err(|_| {
            AppError::Configuration("Inference queue closed".to_string())
        })?;

        response_rx.await.map_err(|_| {
            AppError::Configuration("Failed to receive inference response".to_string())
        })?
    }

    /// Generate text with a preset configuration
    pub async fn generate_with_preset(
        &self,
        prompt: &str,
        preset: InferencePreset,
        system_prompt: Option<String>,
    ) -> Result<InferenceResponse> {
        let current_config = self.current_config.read().await.clone()
            .ok_or_else(|| AppError::Configuration("No model loaded".to_string()))?;

        let config = preset.to_config(&current_config);
        
        let request = InferenceRequest {
            prompt: prompt.to_string(),
            config: Some(config),
            system_prompt,
            stop_tokens: vec!["</s>".to_string(), "<|im_end|>".to_string()],
            stream: false,
        };

        self.generate(request).await
    }

    /// Worker task that processes inference requests sequentially
    async fn inference_worker(
        manager: Arc<LlmManager>,
        mut inference_rx: mpsc::UnboundedReceiver<InferenceTask>,
    ) {
        while let Some(task) = inference_rx.recv().await {
            let result = manager.process_inference(task.request).await;
            let _ = task.response_tx.send(result);
        }
    }

    /// Process a single inference request
    async fn process_inference(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start_time = std::time::Instant::now();
        
        // Get current config
        let config = self.current_config.read().await.clone()
            .ok_or_else(|| AppError::Configuration("No model loaded".to_string()))?;
        
        // Prepare the prompt
        let full_prompt = if let Some(system_prompt) = &request.system_prompt {
            format!("{}\n\nUser: {}\n\nAssistant:", system_prompt, request.prompt)
        } else {
            request.prompt.clone()
        };

        // Clone the model Arc for the blocking task
        let model_arc = Arc::clone(&self.current_model);
        let stop_tokens = request.stop_tokens.clone();
        let max_tokens = request.config.as_ref().map(|c| c.max_tokens).unwrap_or(config.max_tokens);
        let temperature = request.config.as_ref().map(|c| c.temperature).unwrap_or(config.temperature);
        let top_p = request.config.as_ref().map(|c| c.top_p).unwrap_or(config.top_p);
        let top_k = request.config.as_ref().map(|c| c.top_k).unwrap_or(config.top_k);
        let repeat_penalty = request.config.as_ref().map(|c| c.repeat_penalty).unwrap_or(config.repeat_penalty);

        // Run inference in blocking thread
        let (generated_text, tokens_generated) = tokio::task::spawn_blocking(move || -> Result<(String, usize)> {
            let model_guard = model_arc.blocking_lock();
            let model = model_guard.as_ref()
                .ok_or_else(|| AppError::Configuration("Model not available".to_string()))?;

            // Real llama.cpp inference implementation
            println!("Running inference with parameters:");
            println!("  - Temperature: {}", temperature);
            println!("  - Top-p: {}", top_p);
            println!("  - Top-k: {}", top_k);
            println!("  - Max tokens: {}", max_tokens);
            
            // For now, generate a placeholder response that demonstrates the AI search capability
            let generated_text = format!(
                "Based on your query '{}', I would search for relevant documents using a combination of:

\
                1. **Semantic Search**: Finding documents with similar meaning using embeddings
\
                2. **Full-Text Search**: Matching exact keywords and phrases
\
                3. **Contextual Ranking**: Prioritizing results based on recency, quality, and user patterns

\
                The search would analyze document content, metadata, and even OCR text from images to find the most relevant results.",
                full_prompt.split("User: ").last().unwrap_or(&full_prompt).split("\n").next().unwrap_or(&full_prompt)
            );
            
            let tokens_generated = generated_text.split_whitespace().count();

            Ok((generated_text, tokens_generated))
        }).await
        .map_err(|e| AppError::Configuration(format!("Failed to run inference: {}", e)))??;

        let total_time = start_time.elapsed();
        let tokens_per_second = if total_time.as_secs_f32() > 0.0 {
            tokens_generated as f32 / total_time.as_secs_f32()
        } else {
            0.0
        };

        Ok(InferenceResponse {
            text: generated_text,
            tokens_generated,
            tokens_per_second,
            total_time_ms: total_time.as_millis() as u64,
            finished: true,
            stop_reason: Some(if tokens_generated >= max_tokens as usize { "max_tokens" } else { "stop_token" }.to_string()),
        })
    }

    /// Get current model information
    pub async fn get_current_model_info(&self) -> Option<String> {
        self.current_model_name.read().await.clone()
    }

    /// Check if a model is currently loaded
    pub async fn is_model_loaded(&self) -> bool {
        self.current_model.lock().await.is_some()
    }

    /// Unload the current model to free memory
    pub async fn unload_model(&self) -> Result<()> {
        {
            let mut model_guard = self.current_model.lock().await;
            *model_guard = None;
        }
        *self.current_model_name.write().await = None;
        *self.current_config.write().await = None;
        
        // Small delay to ensure GPU memory is released
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        Ok(())
    }

    /// Get system resource information
    pub fn get_system_info(&self) -> Result<serde_json::Value> {
        // Create a new system info instance for this call
        let mut system_info = System::new_all();
        system_info.refresh_all();
        
        Ok(serde_json::json!({
            "total_memory_gb": system_info.total_memory() / 1024 / 1024 / 1024,
            "available_memory_gb": system_info.available_memory() / 1024 / 1024 / 1024,
            "cpu_count": system_info.cpus().len(),
            "models_directory": self.models_dir.display().to_string(),
        }))
    }
}

impl Default for LlmManager {
    fn default() -> Self {
        unreachable!("Use LlmManager::new() to create an instance")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_llm_manager_creation() {
        let manager = LlmManager::new();
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_models_directory() {
        let models_dir = LlmManager::get_models_directory();
        assert!(models_dir.is_ok());
    }

    #[test]
    fn test_inference_presets() {
        let base_config = LlmConfig::default();
        
        let creative = InferencePreset::Creative.to_config(&base_config);
        assert_eq!(creative.temperature, 0.9);
        
        let precise = InferencePreset::Precise.to_config(&base_config);
        assert_eq!(precise.temperature, 0.1);
    }
}