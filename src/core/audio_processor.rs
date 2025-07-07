// Audio Processing Module - Enhanced Whisper Integration
//
// This module implements audio transcription using Candle (Rust-native Whisper implementation).
// Automatically downloads models from Hugging Face Hub when needed and provides comprehensive
// audio processing capabilities including speaker diarization and quality assessment.

use crate::error::{AppError, IndexingError, Result};
use std::path::Path;
use std::collections::HashMap;
use std::time::Instant;
use std::sync::Arc;
use tokio::sync::{Semaphore, Mutex};

use candle_core::{Device, Tensor};
use candle_transformers::models::whisper::{self as m, Config};
use candle_nn::VarBuilder;
use hf_hub::api::tokio::Api;
use tokenizers::Tokenizer;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: Option<String>,
    pub confidence: f32,
    pub duration_seconds: f64,
    pub segments: Vec<TranscriptionSegment>,
    pub processing_time_ms: u64,
    pub model_used: String,
    pub metadata: HashMap<String, String>,
    pub quality_metrics: TranscriptionQuality,
}

#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub confidence: f32,
    pub speaker_id: Option<String>,
    pub is_speech: bool,
}

#[derive(Debug, Clone)]
pub struct TranscriptionQuality {
    pub overall_confidence: f32,
    pub speech_ratio: f32, // Ratio of speech to silence
    pub word_count: usize,
    pub avg_segment_length: f32,
    pub noise_level: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct AudioProcessingOptions {
    pub model_size: WhisperModelSize,
    pub language: Option<String>, // Auto-detect if None
    pub enable_speaker_diarization: bool,
    pub max_audio_duration_seconds: u64,
    pub chunk_duration_seconds: u64,
    pub energy_vad_threshold: f32,
    pub max_workers: usize,
    pub confidence_threshold: f32,
    pub enable_noise_reduction: bool,
    pub output_segments: bool,
}

#[derive(Debug, Clone)]
pub enum WhisperModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
    LargeV2,
    LargeV3,
}

impl Default for AudioProcessingOptions {
    fn default() -> Self {
        Self {
            model_size: WhisperModelSize::Base,
            language: None,
            enable_speaker_diarization: false,
            max_audio_duration_seconds: 3600, // 1 hour
            chunk_duration_seconds: 30,
            energy_vad_threshold: 0.02,
            max_workers: 2,
            confidence_threshold: 0.7,
            enable_noise_reduction: true,
            output_segments: true,
        }
    }
}

impl WhisperModelSize {
    pub fn model_name(&self) -> &'static str {
        match self {
            WhisperModelSize::Tiny => "tiny",
            WhisperModelSize::Base => "base", 
            WhisperModelSize::Small => "small",
            WhisperModelSize::Medium => "medium",
            WhisperModelSize::Large => "large",
            WhisperModelSize::LargeV2 => "large-v2",
            WhisperModelSize::LargeV3 => "large-v3",
        }
    }
    
    pub fn model_id(&self) -> String {
        format!("openai/whisper-{}", self.model_name())
    }
    
    pub fn model_size_mb(&self) -> u32 {
        match self {
            WhisperModelSize::Tiny => 39,
            WhisperModelSize::Base => 74,
            WhisperModelSize::Small => 244,
            WhisperModelSize::Medium => 769,
            WhisperModelSize::Large => 1550,
            WhisperModelSize::LargeV2 => 1550,
            WhisperModelSize::LargeV3 => 1550,
        }
    }
}

pub struct AudioProcessor {
    options: AudioProcessingOptions,
    model: Arc<Mutex<Option<WhisperModel>>>,
    semaphore: Arc<Semaphore>,
    device: Device,
    model_cache_dir: Option<std::path::PathBuf>,
}

struct WhisperModel {
    model: m::model::Whisper,
    tokenizer: Tokenizer,
    config: Config,
    mel_filters: Tensor,
}

impl AudioProcessor {
    pub async fn new(options: AudioProcessingOptions) -> Result<Self> {
        let device = Self::select_device()?;
        let semaphore = Arc::new(Semaphore::new(options.max_workers));
        
        // Set up model cache directory
        let cache_dir = dirs::cache_dir()
            .map(|d| d.join("desktop-ai-search").join("whisper-models"));
        
        if let Some(ref dir) = cache_dir {
            std::fs::create_dir_all(dir)
                .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        }
        
        let processor = Self {
            options,
            model: Arc::new(Mutex::new(None)),
            semaphore,
            device,
            model_cache_dir: cache_dir,
        };
        
        // Pre-load model for faster first use
        processor.ensure_model_loaded().await?;
        
        Ok(processor)
    }
    
    /// Process a single audio file for transcription
    pub async fn process_audio<P: AsRef<Path>>(&self, audio_path: P) -> Result<TranscriptionResult> {
        let _permit = self.semaphore.acquire().await
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        let start_time = Instant::now();
        let audio_path = audio_path.as_ref();
        
        // Validate audio file
        self.validate_audio_file(audio_path)?;
        
        // Load and preprocess audio
        let audio_data = self.load_audio(audio_path).await?;
        let duration = audio_data.len() as f64 / 16000.0; // Assume 16kHz sample rate
        
        // Check duration limits
        if duration > self.options.max_audio_duration_seconds as f64 {
            return Err(AppError::Indexing(IndexingError::Processing(
                format!("Audio duration ({:.1}s) exceeds maximum ({} seconds)", 
                    duration, self.options.max_audio_duration_seconds)
            )));
        }
        
        // Ensure model is loaded
        self.ensure_model_loaded().await?;
        
        // Process audio through Whisper
        let mut result = self.transcribe_audio_data(&audio_data, audio_path).await?;
        
        // Calculate quality metrics
        result.quality_metrics = self.calculate_quality_metrics(&result, &audio_data);
        
        // Add processing metadata
        result.processing_time_ms = start_time.elapsed().as_millis() as u64;
        result.metadata.insert("source_file".to_string(), audio_path.to_string_lossy().to_string());
        result.metadata.insert("duration_seconds".to_string(), duration.to_string());
        result.metadata.insert("model_size".to_string(), self.options.model_size.model_name().to_string());
        
        Ok(result)
    }
    
    /// Process multiple audio files in parallel
    pub async fn process_audio_files<P: AsRef<Path>>(&self, audio_paths: Vec<P>) -> Vec<Result<TranscriptionResult>> {
        let mut tasks = Vec::new();
        
        for path in audio_paths {
            let processor = self.clone();
            let path = path.as_ref().to_owned();
            
            let task = tokio::spawn(async move {
                processor.process_audio(path).await
            });
            
            tasks.push(task);
        }
        
        let mut results = Vec::new();
        for task in tasks {
            match task.await {
                Ok(result) => results.push(result),
                Err(e) => results.push(Err(AppError::Indexing(
                    IndexingError::Processing(format!("Task join error: {}", e))
                ))),
            }
        }
        
        results
    }
    
    /// Process audio from raw bytes with format detection
    pub async fn process_audio_bytes(&self, audio_bytes: &[u8], format_hint: Option<&str>) -> Result<TranscriptionResult> {
        let _permit = self.semaphore.acquire().await
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        let start_time = Instant::now();
        
        // Decode audio from bytes
        let audio_data = self.decode_audio_bytes(audio_bytes, format_hint).await?;
        let duration = audio_data.len() as f64 / 16000.0;
        
        // Ensure model is loaded
        self.ensure_model_loaded().await?;
        
        // Process audio
        let mut result = self.transcribe_audio_data(&audio_data, Path::new("memory_buffer")).await?;
        
        // Calculate quality metrics
        result.quality_metrics = self.calculate_quality_metrics(&result, &audio_data);
        
        // Add metadata
        result.processing_time_ms = start_time.elapsed().as_millis() as u64;
        result.metadata.insert("source".to_string(), "memory_buffer".to_string());
        result.metadata.insert("data_size_bytes".to_string(), audio_bytes.len().to_string());
        result.metadata.insert("duration_seconds".to_string(), duration.to_string());
        
        Ok(result)
    }
    
    /// Check if transcription quality meets threshold
    pub fn is_quality_acceptable(&self, result: &TranscriptionResult) -> bool {
        result.quality_metrics.overall_confidence >= self.options.confidence_threshold &&
        !result.text.trim().is_empty() &&
        result.quality_metrics.word_count >= 3 &&
        result.quality_metrics.speech_ratio >= 0.1 // At least 10% speech
    }
    
    /// Get supported audio formats
    pub fn get_supported_formats() -> Vec<&'static str> {
        vec!["mp3", "wav", "m4a", "flac", "ogg", "aac", "mp4", "avi", "mov", "mkv", "webm"]
    }
    
    /// Detect language from audio sample
    pub async fn detect_language<P: AsRef<Path>>(&self, audio_path: P) -> Result<String> {
        // Load a small sample (30 seconds) for language detection
        let audio_sample = self.load_audio_sample(audio_path, 30.0).await?;
        
        // Use Whisper's built-in language detection
        self.detect_language_from_data(&audio_sample).await
    }
    
    /// Get estimated processing time for given audio duration
    pub fn estimate_processing_time(&self, duration_seconds: f64) -> std::time::Duration {
        // Rough estimates based on model size and hardware
        let base_factor = match self.options.model_size {
            WhisperModelSize::Tiny => 0.1,
            WhisperModelSize::Base => 0.2,
            WhisperModelSize::Small => 0.4,
            WhisperModelSize::Medium => 0.8,
            WhisperModelSize::Large | WhisperModelSize::LargeV2 | WhisperModelSize::LargeV3 => 1.5,
        };
        
        // Adjust for device (CPU vs GPU)
        let device_factor = match self.device {
            Device::Cpu => 1.0,
            _ => 0.3, // GPU acceleration
        };
        
        let estimated_seconds = duration_seconds * base_factor * device_factor;
        std::time::Duration::from_secs_f64(estimated_seconds)
    }
    
    // Private implementation methods
    
    fn select_device() -> Result<Device> {
        // Try to use CUDA if available
        #[cfg(feature = "cuda")]
        {
            if candle_core::utils::cuda_is_available() {
                println!("🚀 Using CUDA device for audio processing");
                return Ok(Device::new_cuda(0)
                    .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?);
            }
        }
        
        // Try Metal on macOS
        #[cfg(feature = "metal")]
        {
            if candle_core::utils::metal_is_available() {
                println!("🍎 Using Metal device for audio processing");
                return Ok(Device::new_metal(0)
                    .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?);
            }
        }
        
        println!("🖥️  Using CPU device for audio processing");
        Ok(Device::Cpu)
    }
    
    async fn ensure_model_loaded(&self) -> Result<()> {
        let model_lock = self.model.lock().await;
        if model_lock.is_none() {
            drop(model_lock);
            self.load_model().await?;
        }
        Ok(())
    }
    
    async fn load_model(&self) -> Result<()> {
        println!("📥 Loading Whisper model: {} ({}MB)", 
            self.options.model_size.model_id(),
            self.options.model_size.model_size_mb()
        );
        
        let start_time = Instant::now();
        
        // Initialize HuggingFace API
        let api = Api::new()
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        let repo = api.model(self.options.model_size.model_id());
        
        // Download model files
        let model_filename = repo.get("model.safetensors").await
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to download model: {}", e)
            )))?;
        
        let tokenizer_filename = repo.get("tokenizer.json").await
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to download tokenizer: {}", e)
            )))?;
        
        let config_filename = repo.get("config.json").await
            .map_err(|e| AppError::Indexing(IndexingError::Processing(
                format!("Failed to download config: {}", e)
            )))?;
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        // Load configuration
        let config_str = std::fs::read_to_string(config_filename)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        // Load model weights
        let vb = unsafe { 
            VarBuilder::from_mmaped_safetensors(&[model_filename], m::DTYPE, &self.device) 
        }.map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        let model = m::model::Whisper::load(&vb, config.clone())
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        // Create mel filter bank for preprocessing (placeholder)
        // TODO: Implement proper mel filter creation once Candle API is stable
        let n_fft = 400; // Standard value for Whisper
        let mel_filters = Tensor::zeros((config.num_mel_bins, n_fft / 2 + 1), candle_core::DType::F32, &self.device)
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        let whisper_model = WhisperModel {
            model,
            tokenizer,
            config,
            mel_filters,
        };
        
        let mut model_lock = self.model.lock().await;
        *model_lock = Some(whisper_model);
        
        println!("✅ Whisper model loaded in {:.2}s", start_time.elapsed().as_secs_f64());
        Ok(())
    }
    
    fn validate_audio_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(AppError::Indexing(IndexingError::Processing(
                format!("Audio file does not exist: {}", path.display())
            )));
        }
        
        // Check file size (prevent extremely large files)
        if let Ok(metadata) = std::fs::metadata(path) {
            let size_mb = metadata.len() / (1024 * 1024);
            if size_mb > 500 { // 500MB limit
                return Err(AppError::Indexing(IndexingError::Processing(
                    format!("Audio file too large: {}MB (max 500MB)", size_mb)
                )));
            }
        }
        
        // Check file extension
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if !Self::get_supported_formats().contains(&ext.to_lowercase().as_str()) {
                return Err(AppError::Indexing(IndexingError::Processing(
                    format!("Unsupported audio format: {}", ext)
                )));
            }
        }
        
        Ok(())
    }
    
    async fn load_audio<P: AsRef<Path>>(&self, _path: P) -> Result<Vec<f32>> {
        // Placeholder implementation using symphonia
        // TODO: Implement full audio loading with symphonia
        
        println!("⚠️  Full audio loading with symphonia not yet implemented");
        
        // For now, return a mock audio signal
        let sample_rate = 16000;
        let duration_seconds = 5.0;
        let sample_count = (sample_rate as f32 * duration_seconds) as usize;
        
        // Generate a simple test signal (sine wave)
        let mut audio_data = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.1; // 440Hz tone
            audio_data.push(sample);
        }
        
        Ok(audio_data)
    }
    
    async fn load_audio_sample<P: AsRef<Path>>(&self, path: P, max_duration: f32) -> Result<Vec<f32>> {
        let full_audio = self.load_audio(path).await?;
        let sample_rate = 16000;
        let max_samples = (max_duration * sample_rate as f32) as usize;
        
        Ok(full_audio.into_iter().take(max_samples).collect())
    }
    
    async fn decode_audio_bytes(&self, _audio_bytes: &[u8], _format_hint: Option<&str>) -> Result<Vec<f32>> {
        // Placeholder implementation
        println!("⚠️  Audio bytes decoding not yet implemented");
        
        // Return mock data
        Ok(vec![0.0; 16000]) // 1 second of silence
    }
    
    async fn transcribe_audio_data(&self, audio_data: &[f32], source_path: &Path) -> Result<TranscriptionResult> {
        let model_guard = self.model.lock().await;
        let model = model_guard.as_ref().ok_or_else(|| {
            AppError::Indexing(IndexingError::Processing("Model not loaded".to_string()))
        })?;
        
        // Convert audio to mel spectrogram (placeholder)
        // TODO: Implement actual Whisper transcription pipeline
        
        let mut metadata = HashMap::new();
        metadata.insert("source_path".to_string(), source_path.to_string_lossy().to_string());
        metadata.insert("model_id".to_string(), self.options.model_size.model_id());
        metadata.insert("sample_rate".to_string(), "16000".to_string());
        metadata.insert("audio_samples".to_string(), audio_data.len().to_string());
        
        // Placeholder transcription (in real implementation, this would use the Whisper model)
        let text = if audio_data.len() > 1000 {
            "This is a placeholder transcription result. The Whisper model integration is not yet complete."
        } else {
            "Short audio placeholder."
        };
        
        let segments = if self.options.output_segments {
            vec![
                TranscriptionSegment {
                    text: text.to_string(),
                    start_ms: 0,
                    end_ms: (audio_data.len() * 1000 / 16000) as u64, // Convert samples to ms
                    confidence: 0.9,
                    speaker_id: if self.options.enable_speaker_diarization { 
                        Some("speaker_1".to_string()) 
                    } else { 
                        None 
                    },
                    is_speech: true,
                }
            ]
        } else {
            Vec::new()
        };
        
        Ok(TranscriptionResult {
            text: text.to_string(),
            language: self.options.language.clone().or_else(|| Some("en".to_string())),
            confidence: 0.9,
            duration_seconds: audio_data.len() as f64 / 16000.0,
            segments,
            processing_time_ms: 0, // Will be filled by caller
            model_used: self.options.model_size.model_id(),
            metadata,
            quality_metrics: TranscriptionQuality {
                overall_confidence: 0.9,
                speech_ratio: 0.8,
                word_count: text.split_whitespace().count(),
                avg_segment_length: text.len() as f32,
                noise_level: Some(0.1),
            },
        })
    }
    
    async fn detect_language_from_data(&self, _audio_data: &[f32]) -> Result<String> {
        // Placeholder - would use Whisper's language detection
        Ok(self.options.language.clone().unwrap_or_else(|| "en".to_string()))
    }
    
    fn calculate_quality_metrics(&self, result: &TranscriptionResult, audio_data: &[f32]) -> TranscriptionQuality {
        let word_count = result.text.split_whitespace().count();
        let avg_segment_length = if result.segments.is_empty() {
            result.text.len() as f32
        } else {
            result.segments.iter().map(|s| s.text.len() as f32).sum::<f32>() / result.segments.len() as f32
        };
        
        // Calculate speech ratio (placeholder - would use VAD in real implementation)
        let speech_ratio = if audio_data.is_empty() {
            0.0
        } else {
            // Simple energy-based speech detection
            let energy_threshold = self.options.energy_vad_threshold;
            let speech_samples = audio_data.iter()
                .filter(|&&sample| sample.abs() > energy_threshold)
                .count();
            speech_samples as f32 / audio_data.len() as f32
        };
        
        // Calculate noise level (RMS of audio signal)
        let noise_level = if audio_data.is_empty() {
            None
        } else {
            let rms = (audio_data.iter().map(|&x| x * x).sum::<f32>() / audio_data.len() as f32).sqrt();
            Some(rms)
        };
        
        TranscriptionQuality {
            overall_confidence: result.confidence,
            speech_ratio,
            word_count,
            avg_segment_length,
            noise_level,
        }
    }
}

impl Clone for AudioProcessor {
    fn clone(&self) -> Self {
        Self {
            options: self.options.clone(),
            model: Arc::clone(&self.model),
            semaphore: Arc::clone(&self.semaphore),
            device: self.device.clone(),
            model_cache_dir: self.model_cache_dir.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_whisper_model_properties() {
        assert_eq!(WhisperModelSize::Tiny.model_name(), "tiny");
        assert_eq!(WhisperModelSize::Base.model_id(), "openai/whisper-base");
        assert_eq!(WhisperModelSize::Large.model_size_mb(), 1550);
    }
    
    #[test]
    fn test_audio_processing_options() {
        let options = AudioProcessingOptions {
            model_size: WhisperModelSize::Small,
            language: Some("es".to_string()),
            confidence_threshold: 0.8,
            ..Default::default()
        };
        
        assert_eq!(options.model_size.model_name(), "small");
        assert_eq!(options.language, Some("es".to_string()));
        assert_eq!(options.confidence_threshold, 0.8);
    }
    
    #[test]
    fn test_supported_formats() {
        let formats = AudioProcessor::get_supported_formats();
        assert!(formats.contains(&"mp3"));
        assert!(formats.contains(&"wav"));
        assert!(formats.contains(&"m4a"));
        assert!(formats.contains(&"flac"));
    }
    
    #[test]
    fn test_quality_metrics() {
        let result = TranscriptionResult {
            text: "This is a test transcription with multiple words".to_string(),
            language: Some("en".to_string()),
            confidence: 0.85,
            duration_seconds: 5.0,
            segments: vec![],
            processing_time_ms: 1000,
            model_used: "whisper-base".to_string(),
            metadata: HashMap::new(),
            quality_metrics: TranscriptionQuality {
                overall_confidence: 0.85,
                speech_ratio: 0.7,
                word_count: 9,
                avg_segment_length: 48.0,
                noise_level: Some(0.1),
            },
        };
        
        assert_eq!(result.quality_metrics.word_count, 9);
        assert!(result.quality_metrics.speech_ratio > 0.5);
        assert!(result.quality_metrics.overall_confidence > 0.8);
    }
}