use crate::database::Database;
use crate::models::{Document, FileType};
use crate::core::{
    document_processor::{DocumentProcessor, ProcessingOptions, ProcessingProgress},
    text_extractor::TextExtractor,
    embedding_manager::EmbeddingManager,
};
use crate::utils::file_types::FileTypeDetector;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use uuid::Uuid;
use chrono::Utc;
use std::collections::HashMap;
use std::time::Instant;

/// Advanced indexing state with detailed tracking
#[derive(Debug, Clone, Default)]
pub struct IndexingState {
    pub total_files: usize,
    pub processed_files: usize,
    pub failed_files: usize,
    pub total_chunks: usize,
    pub current_file: String,
    pub processing_rate: f64, // files per second
    pub errors: Vec<String>,
    pub start_time: Option<Instant>,
    pub estimated_completion: Option<std::time::Duration>,
}

/// Global indexing state with thread safety
static INDEXING_STATE: once_cell::sync::Lazy<Arc<Mutex<IndexingState>>> = 
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(IndexingState::default())));

/// Enhanced single file indexing with full pipeline
#[tauri::command]
pub async fn index_file(path: String) -> Result<(), String> {
    println!("🔍 Indexing file: {}", path);
    
    let file_path = Path::new(&path);
    if !file_path.exists() {
        return Err(format!("File does not exist: {}", path));
    }
    
    // Initialize components
    let database = Arc::new(
        Database::new("search.db")
            .map_err(|e| format!("Failed to initialize database: {}", e))?
    );
    
    let processing_options = ProcessingOptions {
        max_concurrent_documents: 1,
        extract_metadata: true,
        skip_empty_documents: true,
        min_content_length: 50,
        ..Default::default()
    };
    
    let mut processor = DocumentProcessor::new(processing_options);
    processor.set_database(database.clone());
    
    // Optional: Set up embedding manager if available
    if let Ok(embedding_manager) = EmbeddingManager::new() {
        processor.set_embedding_manager(Arc::new(Mutex::new(embedding_manager))).await;
    }
    
    // Process single document
    match processor.process_document(file_path, None).await {
        Ok(result) => {
            // Update global state
            let mut state = INDEXING_STATE.lock().await;
            state.processed_files += 1;
            state.total_chunks += result.chunks.len();
            drop(state);
            
            println!("✅ Successfully indexed: {} ({} chunks)", path, result.chunks.len());
            Ok(())
        },
        Err(e) => {
            // Update error state
            let mut state = INDEXING_STATE.lock().await;
            state.failed_files += 1;
            state.errors.push(format!("{}: {}", path, e));
            drop(state);
            
            Err(format!("Failed to index {}: {}", path, e))
        }
    }
}

/// Enhanced directory indexing with parallel processing
#[tauri::command]
pub async fn index_directory(directory_path: String) -> Result<(), String> {
    println!("📁 Indexing directory: {}", directory_path);
    
    let dir_path = Path::new(&directory_path);
    if !dir_path.exists() || !dir_path.is_dir() {
        return Err(format!("Directory does not exist: {}", directory_path));
    }
    
    // Initialize components
    let database = Arc::new(
        Database::new("search.db")
            .map_err(|e| format!("Failed to initialize database: {}", e))?
    );
    
    let processing_options = ProcessingOptions {
        max_concurrent_documents: 4,
        extract_metadata: true,
        skip_empty_documents: true,
        min_content_length: 50,
        ..Default::default()
    };
    
    let mut processor = DocumentProcessor::new(processing_options);
    processor.set_database(database.clone());
    
    // Optional: Set up embedding manager if available
    if let Ok(embedding_manager) = EmbeddingManager::new() {
        processor.set_embedding_manager(Arc::new(Mutex::new(embedding_manager))).await;
    }
    
    // Discover indexable files
    let indexable_files = discover_indexable_files(dir_path).await?;
    
    // Initialize state
    {
        let mut state = INDEXING_STATE.lock().await;
        *state = IndexingState {
            total_files: indexable_files.len(),
            start_time: Some(Instant::now()),
            ..Default::default()
        };
    }
    
    println!("📊 Found {} indexable files", indexable_files.len());
    
    // Set up progress channel
    let (progress_tx, mut progress_rx) = mpsc::channel::<ProcessingProgress>(100);
    
    // Start progress monitoring task
    let progress_task = tokio::spawn(async move {
        while let Some(progress) = progress_rx.recv().await {
            let mut state = INDEXING_STATE.lock().await;
            state.current_file = progress.current_file.unwrap_or_default();
            
            // Calculate processing rate
            if let Some(start_time) = state.start_time {
                let elapsed = start_time.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    state.processing_rate = state.processed_files as f64 / elapsed;
                    
                    // Estimate completion time
                    let remaining_files = state.total_files.saturating_sub(state.processed_files);
                    if state.processing_rate > 0.0 {
                        let remaining_seconds = remaining_files as f64 / state.processing_rate;
                        state.estimated_completion = Some(std::time::Duration::from_secs_f64(remaining_seconds));
                    }
                }
            }
        }
    });
    
    // Process documents with progress tracking
    let results = processor.process_documents(indexable_files, Some(progress_tx)).await;
    
    // Stop progress monitoring
    progress_task.abort();
    
    // Update final state
    {
        let mut state = INDEXING_STATE.lock().await;
        state.processed_files = results.iter().filter(|r| r.is_ok()).count();
        state.failed_files = results.iter().filter(|r| r.is_err()).count();
        state.total_chunks = results.iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|result| result.chunks.len())
            .sum();
            
        // Collect errors
        for result in &results {
            if let Err(e) = result {
                state.errors.push(e.to_string());
            }
        }
    }
    
    let state = INDEXING_STATE.lock().await;
    println!("✅ Directory indexing completed: {}/{} files processed, {} chunks created", 
             state.processed_files, state.total_files, state.total_chunks);
    
    if state.failed_files > 0 {
        println!("⚠️  {} files failed to process", state.failed_files);
    }
    
    Ok(())
}

/// Incremental directory indexing (only process changed files)
#[tauri::command]
pub async fn index_directory_incremental(directory_path: String) -> Result<(), String> {
    println!("🔄 Incremental indexing directory: {}", directory_path);
    
    let dir_path = Path::new(&directory_path);
    if !dir_path.exists() || !dir_path.is_dir() {
        return Err(format!("Directory does not exist: {}", directory_path));
    }
    
    // Initialize database
    let database = Arc::new(
        Database::new("search.db")
            .map_err(|e| format!("Failed to initialize database: {}", e))?
    );
    
    // Discover all files
    let all_files = discover_indexable_files(dir_path).await?;
    
    // Filter files that need reprocessing
    let mut files_to_process = Vec::new();
    for file_path in all_files {
        if let Ok(should_reindex) = should_reindex_file(&database, &file_path).await {
            if should_reindex {
                files_to_process.push(file_path);
            }
        }
    }
    
    if files_to_process.is_empty() {
        println!("✅ All files are up to date, no indexing needed");
        return Ok(());
    }
    
    println!("📊 Found {} files that need reprocessing", files_to_process.len());
    
    // Process only changed files
    let processing_options = ProcessingOptions {
        max_concurrent_documents: 4,
        extract_metadata: true,
        skip_empty_documents: true,
        min_content_length: 50,
        ..Default::default()
    };
    
    let mut processor = DocumentProcessor::new(processing_options);
    processor.set_database(database.clone());
    
    // Optional: Set up embedding manager if available
    if let Ok(embedding_manager) = EmbeddingManager::new() {
        processor.set_embedding_manager(Arc::new(Mutex::new(embedding_manager))).await;
    }
    
    // Initialize state for incremental processing
    {
        let mut state = INDEXING_STATE.lock().await;
        *state = IndexingState {
            total_files: files_to_process.len(),
            start_time: Some(Instant::now()),
            ..Default::default()
        };
    }
    
    let results = processor.process_documents(files_to_process, None).await;
    
    let processed = results.iter().filter(|r| r.is_ok()).count();
    let failed = results.iter().filter(|r| r.is_err()).count();
    
    println!("✅ Incremental indexing completed: {}/{} files processed", 
             processed, processed + failed);
    
    Ok(())
}

/// Get detailed indexing status
#[tauri::command]
pub async fn get_indexing_status() -> Result<IndexingState, String> {
    let state = INDEXING_STATE.lock().await;
    Ok(state.clone())
}

/// Get indexing statistics
#[tauri::command]
pub async fn get_indexing_statistics() -> Result<HashMap<String, serde_json::Value>, String> {
    let database = Database::new("search.db")
        .map_err(|e| format!("Failed to initialize database: {}", e))?;
    
    let mut stats = HashMap::new();
    
    // Document count by type
    if let Ok(doc_stats) = database.get_document_statistics().await {
        stats.insert("document_statistics".to_string(), serde_json::to_value(doc_stats).unwrap());
    }
    
    // Total chunks and embeddings
    if let Ok(chunk_count) = database.get_total_chunk_count().await {
        stats.insert("total_chunks".to_string(), serde_json::Value::Number(chunk_count.into()));
    }
    
    if let Ok(embedding_count) = database.get_total_embedding_count().await {
        stats.insert("total_embeddings".to_string(), serde_json::Value::Number(embedding_count.into()));
    }
    
    // Index health
    if let Ok(health) = database.check_index_health().await {
        stats.insert("index_health".to_string(), serde_json::to_value(health).unwrap());
    }
    
    Ok(stats)
}

/// Reset indexing state
#[tauri::command]
pub async fn reset_indexing_state() -> Result<(), String> {
    let mut state = INDEXING_STATE.lock().await;
    *state = IndexingState::default();
    println!("🔄 Indexing state reset");
    Ok(())
}

/// Discover all indexable files in a directory recursively
async fn discover_indexable_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let file_detector = FileTypeDetector::new();
    
    fn collect_files_recursive(
        dir: &Path, 
        files: &mut Vec<PathBuf>, 
        detector: &FileTypeDetector
    ) -> Result<(), String> {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    // Skip hidden directories and common non-indexable directories
                    if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                        if !dir_name.starts_with('.') && 
                           !matches!(dir_name, "node_modules" | "target" | ".git" | ".svn" | "__pycache__") {
                            collect_files_recursive(&path, files, detector)?;
                        }
                    }
                } else if path.is_file() {
                    // Check if file is indexable
                    if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                        if !file_name.starts_with('.') && detector.is_indexable(&path) {
                            files.push(path);
                        }
                    }
                }
            }
        }
        Ok(())
    }
    
    collect_files_recursive(dir, &mut files, &file_detector)?;
    Ok(files)
}

/// Check if a file should be reindexed based on modification time and content hash
async fn should_reindex_file(database: &Database, file_path: &PathBuf) -> Result<bool, crate::error::AppError> {
    let path_str = file_path.to_string_lossy().to_string();
    
    // Check if document exists in database
    if let Some(existing_doc) = database.get_document_by_path(&path_str)? {
        // Check modification time
        if let Ok(metadata) = std::fs::metadata(file_path) {
            if let Ok(modified) = metadata.modified() {
                let modified_chrono = chrono::DateTime::<Utc>::from(modified);
                
                // If file hasn't been modified since last index, skip
                if modified_chrono <= existing_doc.last_indexed {
                    return Ok(false);
                }
            }
        }
        
        // Check content hash if modification time is newer
        if let Ok(content) = std::fs::read_to_string(file_path) {
            let content_hash = format!("{:x}", md5::compute(&content));
            if content_hash == existing_doc.content_hash {
                return Ok(false); // Content unchanged
            }
        }
    }
    
    // File is new or has changed
    Ok(true)
}

/// Background indexing daemon (runs continuously)
#[tauri::command]
pub async fn start_background_indexing(
    directory_path: String, 
    interval_seconds: u64
) -> Result<(), String> {
    println!("🔄 Starting background indexing for: {}", directory_path);
    
    let dir_path = PathBuf::from(directory_path);
    
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_seconds));
        
        loop {
            interval.tick().await;
            
            if let Err(e) = index_directory_incremental(dir_path.to_string_lossy().to_string()).await {
                eprintln!("Background indexing error: {}", e);
            }
        }
    });
    
    Ok(())
}