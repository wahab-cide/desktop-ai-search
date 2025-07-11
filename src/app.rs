use crate::core::llm_manager::LlmManager;
use crate::core::embedding_manager::EmbeddingManager;
use crate::database::Database;
use crate::config::{AppConfig, ConfigManager};
use crate::recovery::{RecoveryManager, get_recovery_manager};
use crate::cache::{init_cache_manager, CacheConfig};
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use tauri::{Manager, State};

/// Application state containing all shared resources
pub struct AppState {
    pub database: Arc<Database>,
    pub llm_manager: Arc<LlmManager>,
    pub embedding_manager: Arc<Mutex<EmbeddingManager>>,
    pub config: Arc<Mutex<ConfigManager>>,
    pub recovery_manager: Arc<Mutex<RecoveryManager>>,
}

impl AppState {
    /// Create new application state
    pub async fn new() -> Result<Self> {
        // Initialize configuration manager
        let config_manager = ConfigManager::new()?;
        let config = config_manager.get_config().clone();
        
        // Initialize database with configured path
        let database = Arc::new(Database::new(&config.database.path)?);
        
        // Initialize LLM manager
        let llm_manager = LlmManager::new()?;
        
        // Initialize embedding manager
        let mut embedding_manager = EmbeddingManager::new()?;
        
        // Auto-load the default embedding model from config
        if config.models.auto_download_models {
            embedding_manager.download_model(&config.models.default_embedding_model).await?;
            embedding_manager.load_model(&config.models.default_embedding_model, None).await?;
        }
        
        let embedding_manager = Arc::new(Mutex::new(embedding_manager));
        let config_manager = Arc::new(Mutex::new(config_manager));
        let recovery_manager = get_recovery_manager().await;
        
        Ok(Self {
            database,
            llm_manager,
            embedding_manager,
            config: config_manager,
            recovery_manager,
        })
    }
}

/// Setup function for Tauri app
pub async fn setup_app_async(app: &mut tauri::App) -> Result<()> {
    // Initialize configuration manager
    let config_manager = ConfigManager::new()?;
    let config = config_manager.get_config().clone();
    
    println!("✅ Configuration loaded from: {}", AppConfig::config_file_path().display());
    
    // Initialize database with configured path
    let database = Arc::new(Database::new(&config.database.path)?);
    app.manage(database);
    println!("✅ Database initialized successfully at: {}", config.database.path.display());
    
    // Initialize LLM Manager
    match LlmManager::new() {
        Ok(llm_manager) => {
            app.manage(Arc::new(llm_manager));
            println!("✅ LLM Manager initialized successfully");
        }
        Err(e) => {
            println!("⚠️  LLM Manager initialization failed: {}", e);
            println!("    AI features will be limited");
        }
    }
    
    // Initialize Embedding Manager with pre-loaded model
    match EmbeddingManager::new() {
        Ok(mut embedding_manager) => {
            // Pre-load the default embedding model for better performance
            match async {
                // Auto-download if needed and enabled in config
                if config.models.auto_download_models {
                    if let Err(_) = embedding_manager.load_model(&config.models.default_embedding_model, None).await {
                        embedding_manager.download_model(&config.models.default_embedding_model).await?;
                        embedding_manager.load_model(&config.models.default_embedding_model, None).await?;
                    }
                }
                Ok::<(), crate::error::AppError>(())
            }.await {
                Ok(_) => {
                    println!("✅ Embedding model '{}' pre-loaded for fast search", config.models.default_embedding_model);
                }
                Err(e) => {
                    println!("⚠️  Failed to pre-load embedding model: {}", e);
                    println!("    First search may be slower");
                }
            }
            
            app.manage(Arc::new(Mutex::new(embedding_manager)));
            println!("✅ Embedding Manager initialized successfully");
        }
        Err(e) => {
            println!("⚠️  Embedding Manager initialization failed: {}", e);
            println!("    Semantic search features will be limited");
        }
    }
    
    // Store config manager for runtime access
    app.manage(Arc::new(Mutex::new(config_manager)));
    println!("✅ Configuration manager initialized successfully");
    
    // Initialize cache manager
    let cache_config = CacheConfig {
        max_entries: config.cache.max_entries,
        max_memory_mb: config.cache.max_memory_mb,
        default_ttl: std::time::Duration::from_secs(config.cache.default_ttl_hours * 3600),
        cleanup_interval: std::time::Duration::from_secs(config.cache.cleanup_interval_minutes * 60),
        eviction_policy: crate::cache::EvictionPolicy::Lru,
        compression_enabled: config.cache.compression_enabled,
        persistence_enabled: config.cache.persistence_enabled,
    };
    
    match init_cache_manager(cache_config).await {
        Ok(_) => println!("✅ Cache manager initialized successfully"),
        Err(e) => {
            println!("⚠️  Cache manager initialization failed: {}", e);
            println!("    Continuing without advanced caching");
        }
    }

    // Initialize recovery manager
    let recovery_manager = get_recovery_manager().await;
    app.manage(recovery_manager.clone());
    println!("✅ Recovery manager initialized successfully");
    
    // Perform initial health check
    let health_status = recovery_manager.lock().await.health_check().await;
    println!("🔍 Initial health check:");
    for (component, is_healthy) in health_status {
        let status = if is_healthy { "✅ Healthy" } else { "❌ Unhealthy" };
        println!("  {} - {}", component, status);
    }
    
    Ok(())
}

/// Synchronous wrapper for the async setup function
pub fn setup_app(app: &mut tauri::App) -> Result<()> {
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(setup_app_async(app))
    })
}