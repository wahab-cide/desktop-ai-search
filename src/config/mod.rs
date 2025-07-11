use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use dirs;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub database: DatabaseConfig,
    pub models: ModelConfig,
    pub search: SearchConfig,
    pub indexing: IndexingConfig,
    pub ui: UIConfig,
    pub performance: PerformanceConfig,
    pub cache: CacheConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub path: PathBuf,
    pub connection_pool_size: usize,
    pub timeout_seconds: u64,
    pub backup_interval_hours: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub models_dir: PathBuf,
    pub default_embedding_model: String,
    pub default_llm_model: String,
    pub download_timeout_seconds: u64,
    pub model_cache_size_mb: usize,
    pub auto_download_models: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub timeout_seconds: u64,
    pub max_results: usize,
    pub default_search_mode: String,
    pub enable_spell_check: bool,
    pub confidence_threshold: f32,
    pub similarity_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    pub max_concurrent_files: usize,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub enable_ocr: bool,
    pub enable_audio_transcription: bool,
    pub auto_reindex_interval_hours: u64,
    pub excluded_extensions: Vec<String>,
    pub excluded_directories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIConfig {
    pub theme: String,
    pub default_window_width: u32,
    pub default_window_height: u32,
    pub auto_save_window_state: bool,
    pub show_advanced_filters: bool,
    pub results_per_page: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enable_caching: bool,
    pub cache_size_mb: usize,
    pub enable_gpu_acceleration: bool,
    pub max_memory_usage_mb: usize,
    pub enable_background_optimization: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub max_entries: usize,
    pub max_memory_mb: usize,
    pub default_ttl_hours: u64,
    pub cleanup_interval_minutes: u64,
    pub enable_search_cache: bool,
    pub enable_embedding_cache: bool,
    pub enable_model_cache: bool,
    pub enable_query_cache: bool,
    pub compression_enabled: bool,
    pub persistence_enabled: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        let data_dir = dirs::data_dir()
            .unwrap_or_else(|| std::env::current_dir().unwrap())
            .join("desktop-ai-search");
        
        Self {
            database: DatabaseConfig {
                path: data_dir.join("search.db"),
                connection_pool_size: 15,
                timeout_seconds: 30,
                backup_interval_hours: 24,
            },
            models: ModelConfig {
                models_dir: data_dir.join("models"),
                default_embedding_model: "all-minilm-l6-v2".to_string(),
                default_llm_model: "llama-7b".to_string(),
                download_timeout_seconds: 1800, // 30 minutes
                model_cache_size_mb: 2048, // 2GB
                auto_download_models: true,
            },
            search: SearchConfig {
                timeout_seconds: 30,
                max_results: 100,
                default_search_mode: "hybrid".to_string(),
                enable_spell_check: true,
                confidence_threshold: 0.7,
                similarity_threshold: 0.75,
            },
            indexing: IndexingConfig {
                max_concurrent_files: 12,
                chunk_size: 1000,
                chunk_overlap: 100,
                enable_ocr: true,
                enable_audio_transcription: true,
                auto_reindex_interval_hours: 168, // 1 week
                excluded_extensions: vec![
                    "exe".to_string(), "dll".to_string(), "so".to_string(),
                    "dylib".to_string(), "bin".to_string(), "tmp".to_string(),
                ],
                excluded_directories: vec![
                    "node_modules".to_string(), "target".to_string(), 
                    ".git".to_string(), ".svn".to_string(), "__pycache__".to_string(),
                ],
            },
            ui: UIConfig {
                theme: "system".to_string(),
                default_window_width: 1200,
                default_window_height: 800,
                auto_save_window_state: true,
                show_advanced_filters: false,
                results_per_page: 20,
            },
            performance: PerformanceConfig {
                enable_caching: true,
                cache_size_mb: 512,
                enable_gpu_acceleration: true,
                max_memory_usage_mb: 4096,
                enable_background_optimization: true,
            },
            cache: CacheConfig {
                max_entries: 10000,
                max_memory_mb: 512,
                default_ttl_hours: 24,
                cleanup_interval_minutes: 5,
                enable_search_cache: true,
                enable_embedding_cache: true,
                enable_model_cache: true,
                enable_query_cache: true,
                compression_enabled: false,
                persistence_enabled: false,
            },
        }
    }
}

impl AppConfig {
    /// Load configuration from file or create default
    pub fn load() -> Result<Self> {
        let config_path = Self::config_file_path();
        
        if config_path.exists() {
            let config_content = fs::read_to_string(&config_path)?;
            let config: AppConfig = toml::from_str(&config_content)
                .map_err(|e| crate::error::AppError::Configuration(format!("Failed to parse config: {}", e)))?;
            Ok(config)
        } else {
            // Create default configuration
            let default_config = AppConfig::default();
            default_config.save()?;
            Ok(default_config)
        }
    }
    
    /// Save configuration to file
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_file_path();
        
        // Ensure config directory exists
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        let config_content = toml::to_string_pretty(self)
            .map_err(|e| crate::error::AppError::Configuration(format!("Failed to serialize config: {}", e)))?;
        
        fs::write(&config_path, config_content)?;
        Ok(())
    }
    
    /// Get path to configuration file
    pub fn config_file_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| dirs::home_dir().unwrap().join(".config"))
            .join("desktop-ai-search")
            .join("config.toml")
    }
    
    /// Validate configuration and create necessary directories
    pub fn validate_and_setup(&self) -> Result<()> {
        // Ensure database directory exists
        if let Some(parent) = self.database.path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        // Ensure models directory exists
        fs::create_dir_all(&self.models.models_dir)?;
        
        // Validate configuration values
        if self.database.connection_pool_size == 0 {
            return Err(crate::error::AppError::Configuration("Database connection pool size must be greater than 0".to_string()));
        }
        
        if self.indexing.max_concurrent_files == 0 {
            return Err(crate::error::AppError::Configuration("Max concurrent files must be greater than 0".to_string()));
        }
        
        if self.search.max_results == 0 {
            return Err(crate::error::AppError::Configuration("Max results must be greater than 0".to_string()));
        }
        
        Ok(())
    }
    
    /// Get database URL with connection parameters
    pub fn get_database_url(&self) -> String {
        format!("sqlite://{}?pool_size={}&timeout={}", 
                self.database.path.display(), 
                self.database.connection_pool_size,
                self.database.timeout_seconds)
    }
    
    /// Get models directory path
    pub fn get_models_dir(&self) -> &PathBuf {
        &self.models.models_dir
    }
    
    /// Get search timeout as Duration
    pub fn get_search_timeout(&self) -> Duration {
        Duration::from_secs(self.search.timeout_seconds)
    }
    
    /// Get indexing timeout as Duration
    pub fn get_download_timeout(&self) -> Duration {
        Duration::from_secs(self.models.download_timeout_seconds)
    }
}

/// Configuration manager for runtime config updates
pub struct ConfigManager {
    config: AppConfig,
}

impl ConfigManager {
    pub fn new() -> Result<Self> {
        let config = AppConfig::load()?;
        config.validate_and_setup()?;
        Ok(Self { config })
    }
    
    pub fn get_config(&self) -> &AppConfig {
        &self.config
    }
    
    pub fn update_config(&mut self, new_config: AppConfig) -> Result<()> {
        new_config.validate_and_setup()?;
        new_config.save()?;
        self.config = new_config;
        Ok(())
    }
    
    pub fn reset_to_defaults(&mut self) -> Result<()> {
        let default_config = AppConfig::default();
        self.update_config(default_config)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_default_config_creation() {
        let config = AppConfig::default();
        assert_eq!(config.database.connection_pool_size, 15);
        assert_eq!(config.models.default_embedding_model, "all-minilm-l6-v2");
        assert_eq!(config.search.max_results, 100);
    }
    
    #[test]
    fn test_config_serialization() {
        let config = AppConfig::default();
        let serialized = toml::to_string_pretty(&config).unwrap();
        let deserialized: AppConfig = toml::from_str(&serialized).unwrap();
        
        assert_eq!(config.database.connection_pool_size, deserialized.database.connection_pool_size);
        assert_eq!(config.models.default_embedding_model, deserialized.models.default_embedding_model);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = AppConfig::default();
        
        // Test invalid pool size
        config.database.connection_pool_size = 0;
        assert!(config.validate_and_setup().is_err());
        
        // Test invalid concurrent files
        config.database.connection_pool_size = 15;
        config.indexing.max_concurrent_files = 0;
        assert!(config.validate_and_setup().is_err());
    }
}