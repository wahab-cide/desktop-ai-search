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
    // Initialize database
    let db_path = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("desktop-ai-search")
        .join("search.db");
    
    // Ensure directory exists
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    let database = Arc::new(Database::new(&db_path)?);
    app.manage(database);
    println!("✅ Database initialized successfully at: {}", db_path.display());
    
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
    
    // Initialize Embedding Manager  
    match EmbeddingManager::new() {
        Ok(embedding_manager) => {
            app.manage(Arc::new(Mutex::new(embedding_manager)));
            println!("✅ Embedding Manager initialized successfully");
        }
        Err(e) => {
            println!("⚠️  Embedding Manager initialization failed: {}", e);
            println!("    Semantic search features will be limited");
        }
    }
    
    Ok(())
}