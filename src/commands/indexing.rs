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

/// Simple file type determination based on extension
fn determine_file_type(file_path: &str) -> FileType {
    let path = std::path::Path::new(file_path);
    if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
        match extension.to_lowercase().as_str() {
            "pdf" => FileType::Pdf,
            "docx" | "doc" => FileType::Docx,
            "txt" => FileType::Text,
            "md" | "markdown" => FileType::Markdown,
            "html" | "htm" => FileType::Html,
            "jpg" | "jpeg" | "png" | "gif" | "bmp" => FileType::Image,
            "mp3" | "wav" | "flac" | "ogg" => FileType::Audio,
            "mp4" | "avi" | "mkv" | "mov" => FileType::Video,
            "eml" | "msg" => FileType::Email,
            _ => FileType::Unknown,
        }
    } else {
        FileType::Unknown
    }
}

/// Advanced indexing state with detailed tracking
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct IndexingState {
    pub total_files: usize,
    pub processed_files: usize,
    pub failed_files: usize,
    pub total_chunks: usize,
    pub current_file: String,
    pub processing_rate: f64, // files per second
    pub errors: Vec<String>,
    #[serde(skip)]
    pub start_time: Option<Instant>,
    #[serde(skip)]
    pub estimated_completion: Option<std::time::Duration>,
}

/// Global indexing state with thread safety
static INDEXING_STATE: once_cell::sync::Lazy<Arc<Mutex<IndexingState>>> = 
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(IndexingState::default())));

/// Enhanced single file indexing with full pipeline
#[tauri::command]
pub async fn index_file(
    database: tauri::State<'_, Arc<Database>>,
    path: String
) -> Result<(), String> {
    println!("🔍 Indexing file: {}", path);
    
    let file_path = Path::new(&path);
    if !file_path.exists() {
        return Err(format!("File does not exist: {}", path));
    }
    
    // Use the database from Tauri state
    let database = database.inner().clone();
    
    let processing_options = ProcessingOptions {
        max_concurrent_documents: 1,
        extract_metadata: true,
        skip_empty_documents: true,
        min_content_length: 50,
        ..Default::default()
    };
    
    let mut processor = DocumentProcessor::new(processing_options);
    
    // Optional: Set up embedding manager if available
    if let Ok(_embedding_manager) = EmbeddingManager::new() {
        // Use a default model ID - this should be configurable
        let _ = processor.set_embedding_manager("sentence-transformers/all-MiniLM-L6-v2").await;
    }
    
    // Create a Document instance from the file path
    let document = Document {
        id: uuid::Uuid::new_v4(),
        file_path: path.clone(),
        file_type: determine_file_type(&path),
        file_size: std::fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0),
        content_hash: String::new(), // Will be populated by processor
        creation_date: chrono::Utc::now(),
        modification_date: chrono::Utc::now(),
        last_indexed: chrono::Utc::now(),
        metadata: std::collections::HashMap::new(),
    };
    
    // Process single document
    match processor.process_document(&document, &file_path).await {
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
pub async fn index_directory(
    database: tauri::State<'_, Arc<Database>>,
    directoryPath: String
) -> Result<(), String> {
    println!("📁 Indexing directory: {}", directoryPath);
    
    let dir_path = Path::new(&directoryPath);
    if !dir_path.exists() {
        let error_msg = format!("Directory does not exist: {}", directoryPath);
        eprintln!("❌ {}", error_msg);
        return Err(error_msg);
    }
    if !dir_path.is_dir() {
        let error_msg = format!("Path is not a directory: {}", directoryPath);
        eprintln!("❌ {}", error_msg);
        return Err(error_msg);
    }
    
    println!("✅ Directory exists: {}", directoryPath);
    
    // Use the database from Tauri state
    let database = database.inner().clone();
    
    let processing_options = ProcessingOptions {
        max_concurrent_documents: 12, // Increased from 4 for better performance
        extract_metadata: true,
        skip_empty_documents: true,
        min_content_length: 50,
        ..Default::default()
    };
    
    let mut processor = DocumentProcessor::new(processing_options);
    
    // Optional: Set up embedding manager if available
    if let Ok(_embedding_manager) = EmbeddingManager::new() {
        // Use a default model ID - this should be configurable
        let _ = processor.set_embedding_manager("sentence-transformers/all-MiniLM-L6-v2").await;
    }
    
    // Discover indexable files
    println!("🔍 Discovering indexable files in: {}", directoryPath);
    let indexable_paths = discover_indexable_files(dir_path).await?;
    println!("📊 Found {} indexable files", indexable_paths.len());
    
    // Convert paths to Document objects
    let mut indexable_files = Vec::new();
    for path in indexable_paths {
        let document = Document {
            id: uuid::Uuid::new_v4(),
            file_path: path.to_string_lossy().to_string(),
            file_type: determine_file_type(&path.to_string_lossy()),
            file_size: std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
            content_hash: String::new(), // Will be populated by processor
            creation_date: chrono::Utc::now(),
            modification_date: chrono::Utc::now(),
            last_indexed: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        };
        indexable_files.push(document);
    }
    
    // Initialize state
    {
        let mut state = INDEXING_STATE.lock().await;
        *state = IndexingState {
            total_files: indexable_files.len(),
            start_time: Some(Instant::now()),
            ..Default::default()
        };
    }
    
    println!("📊 Created {} document objects", indexable_files.len());
    
    if indexable_files.is_empty() {
        println!("⚠️ No indexable files found in directory");
        return Ok(());
    }
    
    // Set up progress channel
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<ProcessingProgress>();
    
    // Start progress monitoring task
    let progress_task = tokio::spawn(async move {
        while let Some(progress) = progress_rx.recv().await {
            let mut state = INDEXING_STATE.lock().await;
            state.current_file = progress.current_document.unwrap_or_default();
            
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
    println!("🔄 Starting document processing...");
    let results = processor.process_documents(indexable_files, Some(progress_tx)).await;
    println!("🔄 Document processing completed");
    
    // Debug: Log what happened during processing
    match &results {
        Ok(processing_results) => {
            let total_chunks: usize = processing_results.iter().map(|r| r.chunks.len()).sum();
            println!("📊 Processing results: {} documents, {} total chunks", processing_results.len(), total_chunks);
            
            // Log details for first few documents
            for (i, result) in processing_results.iter().take(3).enumerate() {
                println!("📄 Document {}: {} ({} chunks, content length: {})", 
                         i + 1, 
                         result.document.file_path,
                         result.chunks.len(),
                         result.extraction_result.content.len());
            }
        }
        Err(e) => {
            println!("❌ Document processing failed: {}", e);
        }
    }
    
    // Stop progress monitoring
    progress_task.abort();
    
    // Save processing results to database and update final state
    {
        let mut state = INDEXING_STATE.lock().await;
        match results {
            Ok(processing_results) => {
                println!("💾 Saving {} processed documents to database...", processing_results.len());
                
                let mut saved_count = 0;
                let mut total_chunks = 0;
                let total_results = processing_results.len();
                
                for result in processing_results {
                    // Save document to database (upsert to handle duplicates)
                    let actual_document_id = match database.upsert_document(&result.document) {
                        Ok(id) => id,
                        Err(e) => {
                            eprintln!("❌ Failed to save document {}: {}", result.document.file_path, e);
                            state.errors.push(format!("Failed to save document: {}", e));
                            continue;
                        }
                    };
                    
                    // Save document content for FTS
                    let content = result.chunks.iter()
                        .map(|chunk| chunk.content.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    
                    if let Err(e) = database.insert_document_content(&actual_document_id, &content) {
                        eprintln!("❌ Failed to save document content {}: {}", result.document.file_path, e);
                        state.errors.push(format!("Failed to save document content: {}", e));
                    }
                    
                    // Save chunks to database using the actual document ID
                    for chunk in &result.chunks {
                        if let Err(e) = database.insert_document_chunk(
                            &actual_document_id,
                            chunk.chunk_index,
                            chunk.start_char,
                            chunk.end_char,
                            &chunk.content
                        ) {
                            eprintln!("❌ Failed to save chunk for {}: {}", result.document.file_path, e);
                            state.errors.push(format!("Failed to save chunk: {}", e));
                        }
                    }
                    
                    saved_count += 1;
                    total_chunks += result.chunks.len();
                }
                
                state.processed_files = saved_count;
                state.failed_files = total_results - saved_count;
                state.total_chunks = total_chunks;
                
                println!("✅ Saved {} documents with {} chunks to database", saved_count, total_chunks);
            }
            Err(e) => {
                eprintln!("❌ Document processing failed: {}", e);
                state.errors.push(format!("Processing failed: {}", e));
                state.failed_files = state.total_files;
            }
        }
    }
    
    let state = INDEXING_STATE.lock().await;
    println!("✅ Directory indexing completed: {}/{} files processed, {} chunks created", 
             state.processed_files, state.total_files, state.total_chunks);
    
    if state.failed_files > 0 {
        println!("⚠️  {} files failed to process", state.failed_files);
    }
    drop(state);
    
    // Rebuild FTS index after indexing
    println!("🔄 Rebuilding FTS search index...");
    match database.get_connection() {
        Ok(conn) => {
            // Clear and rebuild FTS index
            if let Err(e) = conn.execute("DELETE FROM chunks_fts", []) {
                eprintln!("⚠️ Failed to clear FTS index: {}", e);
            } else if let Err(e) = conn.execute(
                "INSERT INTO chunks_fts(rowid, content, document_id, id)
                 SELECT rowid, content, document_id, id FROM document_chunks",
                []
            ) {
                eprintln!("⚠️ Failed to rebuild FTS index: {}", e);
            } else {
                println!("✅ FTS search index rebuilt successfully");
            }
        }
        Err(e) => eprintln!("⚠️ Failed to get database connection for FTS rebuild: {}", e),
    }
    
    Ok(())
}

/// Incremental directory indexing (only process changed files)
#[tauri::command]
pub async fn index_directory_incremental(
    database: tauri::State<'_, Arc<Database>>,
    directoryPath: String
) -> Result<(), String> {
    println!("🔄 Incremental indexing directory: {}", directoryPath);
    
    let dir_path = Path::new(&directoryPath);
    if !dir_path.exists() || !dir_path.is_dir() {
        return Err(format!("Directory does not exist: {}", directoryPath));
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
        max_concurrent_documents: 12, // Increased from 4 for better performance
        extract_metadata: true,
        skip_empty_documents: true,
        min_content_length: 50,
        ..Default::default()
    };
    
    let mut processor = DocumentProcessor::new(processing_options);
    
    // Optional: Set up embedding manager if available
    if let Ok(embedding_manager) = EmbeddingManager::new() {
        // Use a default model ID - this should be configurable
        let _ = processor.set_embedding_manager("sentence-transformers/all-MiniLM-L6-v2").await;
    }
    
    // Convert paths to Document objects
    let mut documents_to_process = Vec::new();
    for path in &files_to_process {
        let document = Document {
            id: uuid::Uuid::new_v4(),
            file_path: path.to_string_lossy().to_string(),
            file_type: determine_file_type(&path.to_string_lossy()),
            file_size: std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
            content_hash: String::new(),
            creation_date: chrono::Utc::now(),
            modification_date: chrono::Utc::now(),
            last_indexed: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        };
        documents_to_process.push(document);
    }
    
    // Initialize state for incremental processing
    {
        let mut state = INDEXING_STATE.lock().await;
        *state = IndexingState {
            total_files: documents_to_process.len(),
            start_time: Some(Instant::now()),
            ..Default::default()
        };
    }
    
    let results = processor.process_documents(documents_to_process, None).await;
    
    match results {
        Ok(processing_results) => {
            let processed = processing_results.len();
            println!("✅ Incremental indexing completed: {} files processed", processed);
        }
        Err(e) => {
            println!("❌ Incremental indexing failed: {}", e);
            return Err(format!("Incremental indexing failed: {}", e));
        }
    }
    
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
pub async fn get_indexing_statistics(
    database: tauri::State<'_, Arc<Database>>
) -> Result<HashMap<String, serde_json::Value>, String> {
    let database = database.inner();
    
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

/// Test command to verify parameter passing (no state)
#[tauri::command]
pub async fn test_indexing_params(directoryPath: String) -> Result<String, String> {
    println!("🧪 Test command received directoryPath: {}", directoryPath);
    Ok(format!("Received path: {}", directoryPath))
}

/// Test command with camelCase parameter name
#[tauri::command]
pub async fn test_camel_case(directoryPath: String) -> Result<String, String> {
    println!("🧪 Test camelCase received directoryPath: {}", directoryPath);
    Ok(format!("Received camelCase path: {}", directoryPath))
}

/// Simple index directory without database state (for testing)
#[tauri::command]
pub async fn index_directory_simple(directoryPath: String) -> Result<String, String> {
    println!("🧪 Simple index received directoryPath: {}", directoryPath);
    
    let dir_path = Path::new(&directoryPath);
    if !dir_path.exists() {
        return Err(format!("Directory does not exist: {}", directoryPath));
    }
    
    // Just count files without processing
    let indexable_paths = discover_indexable_files(dir_path).await?;
    let message = format!("Found {} indexable files in {}", indexable_paths.len(), directoryPath);
    println!("✅ {}", message);
    Ok(message)
}

/// Background indexing daemon (runs continuously)
#[tauri::command]
pub async fn start_background_indexing() -> Result<(), String> {
    println!("🔄 Starting background indexing...");
    
    // For now, just return success
    // TODO: Implement proper background indexing
    Ok(())
}

/// Clean up database by removing entries for files that no longer exist
#[tauri::command]
pub async fn cleanup_missing_files(
    database: tauri::State<'_, Arc<Database>>
) -> Result<HashMap<String, u32>, String> {
    println!("🧹 Starting database cleanup for missing files...");
    
    let database = database.inner();
    let mut cleanup_stats = HashMap::new();
    let mut removed_documents = 0u32;
    let mut removed_chunks = 0u32;
    let mut checked_files = 0u32;
    
    // Get all documents from database
    let documents = database.get_all_documents()
        .map_err(|e| format!("Failed to get documents: {}", e))?;
    
    let mut missing_file_ids = Vec::new();
    
    // Check each document's file path
    for doc in documents {
        checked_files += 1;
        let path = Path::new(&doc.file_path);
        
        if !path.exists() {
            println!("🗑️  Missing file: {}", doc.file_path);
            missing_file_ids.push(doc.id);
            removed_documents += 1;
        }
        
        // Progress indicator for large databases
        if checked_files % 100 == 0 {
            println!("📊 Checked {} files...", checked_files);
        }
    }
    
    // Remove documents and their chunks
    for doc_id in missing_file_ids {
        // Remove chunks first (due to foreign key constraints)
        let chunks_removed = database.delete_chunks_by_document_id(&doc_id)
            .map_err(|e| format!("Failed to delete chunks: {}", e))?;
        removed_chunks += chunks_removed;
        
        // Remove document
        database.delete_document(&doc_id)
            .map_err(|e| format!("Failed to delete document: {}", e))?;
    }
    
    // Rebuild FTS index after cleanup
    if removed_documents > 0 {
        println!("🔄 Rebuilding FTS search index after cleanup...");
        match database.get_connection() {
            Ok(conn) => {
                if let Err(e) = conn.execute("DELETE FROM chunks_fts", []) {
                    eprintln!("⚠️ Failed to clear FTS index: {}", e);
                } else if let Err(e) = conn.execute(
                    "INSERT INTO chunks_fts(rowid, content, document_id, id)
                     SELECT rowid, content, document_id, id FROM document_chunks",
                    []
                ) {
                    eprintln!("⚠️ Failed to rebuild FTS index: {}", e);
                } else {
                    println!("✅ FTS search index rebuilt successfully");
                }
            }
            Err(e) => eprintln!("⚠️ Failed to get database connection for FTS rebuild: {}", e),
        }
    }
    
    cleanup_stats.insert("checked_files".to_string(), checked_files);
    cleanup_stats.insert("removed_documents".to_string(), removed_documents);
    cleanup_stats.insert("removed_chunks".to_string(), removed_chunks);
    
    println!("✅ Cleanup completed: {} files checked, {} documents removed, {} chunks removed", 
             checked_files, removed_documents, removed_chunks);
    
    Ok(cleanup_stats)
}

/// Reset entire database (delete all indexed content)
#[tauri::command]
pub async fn reset_database(
    database: tauri::State<'_, Arc<Database>>
) -> Result<HashMap<String, u32>, String> {
    println!("🔥 Resetting entire database...");
    
    let database = database.inner();
    let mut reset_stats = HashMap::new();
    
    // Get counts before reset
    let documents_count = database.get_total_document_count()
        .map_err(|e| format!("Failed to get document count: {}", e))?;
    let chunks_count = database.get_total_chunk_count().await
        .map_err(|e| format!("Failed to get chunk count: {}", e))?;
    
    // Clear all tables
    match database.get_connection() {
        Ok(conn) => {
            // Delete in order due to foreign key constraints
            conn.execute("DELETE FROM chunks_fts", [])
                .map_err(|e| format!("Failed to clear FTS index: {}", e))?;
            
            conn.execute("DELETE FROM document_chunks", [])
                .map_err(|e| format!("Failed to clear chunks: {}", e))?;
            
            conn.execute("DELETE FROM documents", [])
                .map_err(|e| format!("Failed to clear documents: {}", e))?;
            
            // Reset auto-increment sequences if needed
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('documents', 'document_chunks')", [])
                .map_err(|e| format!("Failed to reset sequences: {}", e))?;
        }
        Err(e) => return Err(format!("Failed to get database connection: {}", e)),
    }
    
    reset_stats.insert("removed_documents".to_string(), documents_count as u32);
    reset_stats.insert("removed_chunks".to_string(), chunks_count as u32);
    
    println!("✅ Database reset completed: {} documents and {} chunks removed", 
             documents_count, chunks_count);
    
    Ok(reset_stats)
}