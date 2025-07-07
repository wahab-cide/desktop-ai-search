use crate::core::llm_manager::LlmManager;
use crate::core::embedding_manager::EmbeddingManager;
use crate::database::Database;
use crate::error::Result;
use std::sync::Arc;
use tokio::sync::Mutex;
use tauri::{Manager, State};

/// Application state containing all shared resources
pub struct AppState {
    pub database: Arc<Database>,
    pub llm_manager: Arc<LlmManager>,
    pub embedding_manager: Arc<Mutex<EmbeddingManager>>,
}

impl AppState {
    /// Create new application state
    pub async fn new() -> Result<Self> {
        // Get database path
        let db_path = dirs::data_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("desktop-ai-search")
            .join("search.db");
        
        // Ensure directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Initialize database
        let database = Arc::new(Database::new(&db_path)?);
        
        // Initialize LLM manager
        let llm_manager = LlmManager::new()?;
        
        // Initialize embedding manager
        let mut embedding_manager = EmbeddingManager::new()?;
        
        // Auto-load the default embedding model
        embedding_manager.download_model("all-minilm-l6-v2").await?;
        embedding_manager.load_model("all-minilm-l6-v2", None).await?;
        
        let embedding_manager = Arc::new(Mutex::new(embedding_manager));
        
        Ok(Self {
            database,
            llm_manager,
            embedding_manager,
        })
    }
}

/// Setup function for Tauri app
pub fn setup_app(app: &mut tauri::App) -> Result<()> {
    // Create runtime for async initialization
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| crate::error::AppError::Configuration(format!("Failed to create runtime: {}", e)))?;
    
    // Initialize app state
    let app_state = runtime.block_on(AppState::new())?;
    
    // Store state in Tauri
    app.manage(app_state.database.clone());
    app.manage(app_state.llm_manager.clone());
    app.manage(app_state.embedding_manager.clone());
    
    println!("Application initialized successfully");
    Ok(())
}