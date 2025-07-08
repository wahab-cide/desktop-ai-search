use crate::error::Result;
use crate::core::llm_manager::{LlmManager, InferencePreset};
use crate::models::{InferenceRequest, InferenceResponse, LlmConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::State;

#[derive(Debug, Serialize, Deserialize)]
pub struct AiQueryRequest {
    pub query: String,
    pub search_context: Option<String>,
    pub preset: Option<String>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AiQueryResponse {
    pub response: String,
    pub tokens_generated: usize,
    pub tokens_per_second: f32,
    pub time_ms: u64,
    pub model_info: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub models: Vec<ModelInfo>,
    pub current_model: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub size_mb: u64,
    pub parameters: Option<String>,
    pub loaded: bool,
}

/// Initialize the LLM manager
#[tauri::command]
pub async fn init_ai_system(
    llm_manager: State<'_, Arc<LlmManager>>,
) -> std::result::Result<bool, String> {
    // Scan for available models
    llm_manager.scan_models().await.map_err(|e| e.to_string())?;
    
    // Get available models
    let models = llm_manager.get_available_models().await;
    
    // Auto-load the first available model if any
    if let Some(first_model) = models.first() {
        llm_manager.load_model(&first_model.name, None).await.map_err(|e| e.to_string())?;
        println!("Auto-loaded model: {}", first_model.name);
    }
    
    Ok(true)
}

/// List available AI models
#[tauri::command]
pub async fn list_ai_models(
    llm_manager: State<'_, Arc<LlmManager>>,
) -> std::result::Result<ModelListResponse, String> {
    // Scan for models first
    llm_manager.scan_models().await.map_err(|e| e.to_string())?;
    
    let available_models = llm_manager.get_available_models().await;
    let current_model = llm_manager.get_current_model_info().await;
    
    let models = available_models.into_iter().map(|m| {
        let loaded = current_model.as_ref() == Some(&m.name);
        ModelInfo {
            name: m.name.clone(),
            size_mb: m.size_bytes / (1024 * 1024),
            parameters: m.parameter_count,
            loaded,
        }
    }).collect();
    
    Ok(ModelListResponse {
        models,
        current_model,
    })
}

/// Load a specific AI model
#[tauri::command]
pub async fn load_ai_model(
    model_name: String,
    llm_manager: State<'_, Arc<LlmManager>>,
) -> std::result::Result<bool, String> {
    llm_manager.load_model(&model_name, None).await.map_err(|e| e.to_string())?;
    Ok(true)
}

/// Process an AI query with search context
#[tauri::command]
pub async fn process_ai_query(
    request: AiQueryRequest,
    llm_manager: State<'_, Arc<LlmManager>>,
) -> std::result::Result<AiQueryResponse, String> {
    // Check if model is loaded
    if !llm_manager.is_model_loaded().await {
        return Err("No AI model loaded. Please load a model first.".to_string());
    }
    
    // Prepare system prompt with search context
    let system_prompt = if let Some(context) = request.search_context {
        Some(format!(
            "You are an AI assistant helping with document search and analysis. \
            The user is searching in the context of: {}. \
            Provide helpful, concise responses that focus on search strategies and relevant information.",
            context
        ))
    } else {
        Some(
            "You are an AI assistant helping with document search and analysis. \
            Provide helpful, concise responses that focus on search strategies and relevant information."
            .to_string()
        )
    };
    
    // Determine inference preset
    let preset = match request.preset.as_deref() {
        Some("creative") => InferencePreset::Creative,
        Some("precise") => InferencePreset::Precise,
        _ => InferencePreset::Balanced,
    };
    
    // Generate response
    let response = llm_manager.generate_with_preset(
        &request.query,
        preset,
        system_prompt,
    ).await.map_err(|e| e.to_string())?;
    
    // Get model info
    let model_info = llm_manager.get_current_model_info().await;
    
    Ok(AiQueryResponse {
        response: response.text,
        tokens_generated: response.tokens_generated,
        tokens_per_second: response.tokens_per_second,
        time_ms: response.total_time_ms,
        model_info,
    })
}

/// Generate search query suggestions using AI
#[tauri::command]
pub async fn generate_query_suggestions(
    partial_query: String,
    context: Option<String>,
    llm_manager: State<'_, Arc<LlmManager>>,
) -> std::result::Result<Vec<String>, String> {
    if !llm_manager.is_model_loaded().await {
        // Return empty suggestions if no model loaded
        return Ok(vec![]);
    }
    
    let prompt = format!(
        "Generate 5 search query suggestions that complete or expand on: '{}'\n\
        Context: {}\n\
        Format: Return only the query suggestions, one per line, no numbering or explanation.",
        partial_query,
        context.unwrap_or_else(|| "General document search".to_string())
    );
    
    let response = llm_manager.generate_with_preset(
        &prompt,
        InferencePreset::Precise,
        Some("You are a search query suggestion engine. Generate only relevant search queries.".to_string()),
    ).await.map_err(|e| e.to_string())?;
    
    // Parse suggestions from response
    let suggestions: Vec<String> = response.text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(5)
        .map(|s| s.trim().to_string())
        .collect();
    
    Ok(suggestions)
}

/// Analyze search results using AI
#[tauri::command]
pub async fn analyze_search_results(
    query: String,
    results_summary: String,
    llm_manager: State<'_, Arc<LlmManager>>,
) -> std::result::Result<String, String> {
    if !llm_manager.is_model_loaded().await {
        return Err("No AI model loaded. Please load a model first.".to_string());
    }
    
    let prompt = format!(
        "Analyze these search results for the query '{}' and provide insights:\n\n{}\n\n\
        Provide a brief analysis highlighting patterns, key findings, and suggestions for refining the search.",
        query,
        results_summary
    );
    
    let response = llm_manager.generate_with_preset(
        &prompt,
        InferencePreset::Balanced,
        Some("You are analyzing search results to help users find information more effectively.".to_string()),
    ).await.map_err(|e| e.to_string())?;
    
    Ok(response.text)
}

/// Get system information for AI features
#[tauri::command]
pub async fn get_ai_system_info(
    llm_manager: State<'_, Arc<LlmManager>>,
) -> std::result::Result<serde_json::Value, String> {
    let mut info = llm_manager.get_system_info().map_err(|e| e.to_string())?;
    
    // Add AI-specific information
    info["model_loaded"] = serde_json::json!(llm_manager.is_model_loaded().await);
    info["current_model"] = serde_json::json!(llm_manager.get_current_model_info().await);
    
    Ok(info)
}