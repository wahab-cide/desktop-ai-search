use crate::error::{Result, AppError, IndexingError};
use crate::core::embedding_manager::EmbeddingManager;
use crate::database::Database;
use serde::{Deserialize, Serialize};
use tauri::State;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingModelInfo {
    pub id: String,
    pub name: String,
    pub dimensions: usize,
    pub size_mb: u64,
    pub downloaded: bool,
    pub loaded: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateEmbeddingsRequest {
    pub texts: Vec<String>,
    pub normalize: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model_id: String,
    pub dimensions: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimilarityRequest {
    pub text1: String,
    pub text2: String,
}

/// List available embedding models
#[tauri::command]
pub async fn list_embedding_models(
    embedding_manager: State<'_, Arc<Mutex<EmbeddingManager>>>,
) -> std::result::Result<Vec<EmbeddingModelInfo>, String> {
    let manager = embedding_manager.lock().await;
    let available_models = manager.get_available_models();
    let current_model = manager.get_current_model_info();
    
    let mut models = Vec::new();
    for model in available_models {
        let downloaded = manager.is_model_downloaded(&model.id).await.map_err(|e| e.to_string())?;
        let loaded = current_model.map(|m| m.id == model.id).unwrap_or(false);
        
        models.push(EmbeddingModelInfo {
            id: model.id.clone(),
            name: model.name.clone(),
            dimensions: model.dimensions,
            size_mb: model.model_size_mb,
            downloaded,
            loaded,
        });
    }
    
    Ok(models)
}

/// Download an embedding model
#[tauri::command]
pub async fn download_embedding_model(
    model_id: String,
    embedding_manager: State<'_, Arc<Mutex<EmbeddingManager>>>,
) -> std::result::Result<bool, String> {
    let manager = embedding_manager.lock().await;
    manager.download_model(&model_id).await.map_err(|e| e.to_string())?;
    Ok(true)
}

/// Load an embedding model for use
#[tauri::command]
pub async fn load_embedding_model(
    model_id: String,
    embedding_manager: State<'_, Arc<Mutex<EmbeddingManager>>>,
) -> std::result::Result<bool, String> {
    let mut manager = embedding_manager.lock().await;
    
    // Download if not already downloaded
    if !manager.is_model_downloaded(&model_id).await.map_err(|e| e.to_string())? {
        manager.download_model(&model_id).await.map_err(|e| e.to_string())?;
    }
    
    manager.load_model(&model_id, None).await.map_err(|e| e.to_string())?;
    Ok(true)
}

/// Generate embeddings for given texts
#[tauri::command]
pub async fn generate_embeddings(
    request: GenerateEmbeddingsRequest,
    embedding_manager: State<'_, Arc<Mutex<EmbeddingManager>>>,
) -> std::result::Result<EmbeddingResponse, String> {
    let manager = embedding_manager.lock().await;
    
    // Check if model is loaded
    let model_info = manager.get_current_model_info()
        .ok_or_else(|| "No embedding model loaded. Please load a model first.".to_string())?;
    
    // Generate embeddings
    let embeddings = manager.generate_embeddings(&request.texts).await.map_err(|e| e.to_string())?;
    
    Ok(EmbeddingResponse {
        embeddings,
        model_id: model_info.id.clone(),
        dimensions: model_info.dimensions,
    })
}

/// Calculate cosine similarity between two texts
#[tauri::command]
pub async fn calculate_text_similarity(
    request: SimilarityRequest,
    embedding_manager: State<'_, Arc<Mutex<EmbeddingManager>>>,
) -> std::result::Result<f32, String> {
    let manager = embedding_manager.lock().await;
    
    // Generate embeddings for both texts
    let embeddings = manager.generate_embeddings(&[request.text1, request.text2]).await.map_err(|e| e.to_string())?;
    
    if embeddings.len() != 2 {
        return Err("Failed to generate embeddings for similarity calculation".to_string());
    }
    
    // Calculate cosine similarity
    let similarity = EmbeddingManager::cosine_similarity(&embeddings[0], &embeddings[1]);
    
    Ok(similarity)
}

/// Get embedding model status
#[tauri::command]
pub async fn get_embedding_status(
    embedding_manager: State<'_, Arc<Mutex<EmbeddingManager>>>,
) -> std::result::Result<serde_json::Value, String> {
    let manager = embedding_manager.lock().await;
    
    let current_model = manager.get_current_model_info();
    let model_loaded = manager.is_model_loaded();
    
    Ok(serde_json::json!({
        "model_loaded": model_loaded,
        "current_model": current_model.map(|m| {
            serde_json::json!({
                "id": m.id,
                "name": m.name,
                "dimensions": m.dimensions,
            })
        }),
    }))
}

/// Regenerate embeddings for all documents in the database
#[tauri::command]
pub async fn regenerate_all_embeddings(
    database: State<'_, Arc<Database>>,
    embedding_manager: State<'_, Arc<Mutex<EmbeddingManager>>>,
) -> std::result::Result<serde_json::Value, String> {
    // Check if model is loaded
    {
        let manager = embedding_manager.lock().await;
        if !manager.is_model_loaded() {
            return Err("No embedding model loaded. Please load a model first.".to_string());
        }
    }
    
    // Get all chunks without embeddings
    let chunks_without_embeddings = database.get_chunks_without_embeddings(100).await.map_err(|e| e.to_string())?;
    let total_chunks = chunks_without_embeddings.len();
    
    if total_chunks == 0 {
        return Ok(serde_json::json!({
            "status": "complete",
            "message": "All chunks already have embeddings",
            "processed": 0,
            "total": 0,
        }));
    }
    
    println!("Regenerating embeddings for {} chunks", total_chunks);
    
    // Process in batches
    let batch_size = 32;
    let mut processed = 0;
    
    for batch in chunks_without_embeddings.chunks(batch_size) {
        // Extract chunk texts
        let texts: Vec<String> = batch.iter().map(|c| c.content.clone()).collect();
        
        // Generate embeddings
        let embeddings = {
            let manager = embedding_manager.lock().await;
            manager.generate_embeddings(&texts).await.map_err(|e| e.to_string())?
        };
        
        // Store embeddings in database
        for (chunk, embedding) in batch.iter().zip(embeddings.iter()) {
            let document_id = chunk.document_id.as_ref()
                .ok_or_else(|| "Chunk missing document_id".to_string())?;
            
            database.store_chunk_embedding(
                &chunk.id,
                document_id,
                embedding,
                "all-minilm-l6-v2", // TODO: Get from current model
            ).await.map_err(|e| e.to_string())?;
        }
        
        processed += batch.len();
        println!("Processed {}/{} chunks", processed, total_chunks);
    }
    
    Ok(serde_json::json!({
        "status": "complete",
        "message": format!("Generated embeddings for {} chunks", processed),
        "processed": processed,
        "total": total_chunks,
    }))
}