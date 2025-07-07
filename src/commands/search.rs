use crate::models::{SearchResult, ResultType, RankingFactors, ContentSnippet};
use crate::core::hybrid_search::HybridSearchEngine;
use crate::database::Database;
use crate::error::Result as AppResult;
use std::sync::Arc;
use uuid::Uuid;

/// Global search service state
static mut SEARCH_ENGINE: Option<Arc<std::sync::Mutex<HybridSearchEngine>>> = None;
static INIT_SEARCH: std::sync::Once = std::sync::Once::new();

/// Initialize the search engine (called once)
fn get_search_engine() -> Arc<std::sync::Mutex<HybridSearchEngine>> {
    unsafe {
        INIT_SEARCH.call_once(|| {
            // Initialize database and search engine
            let database = match Database::new("search.db") {
                Ok(db) => Arc::new(db),
                Err(_) => {
                    // Fallback to in-memory database for development
                    Arc::new(Database::new(":memory:").expect("Failed to create in-memory database"))
                }
            };
            
            let search_engine = HybridSearchEngine::new(database);
            SEARCH_ENGINE = Some(Arc::new(std::sync::Mutex::new(search_engine)));
        });
        
        SEARCH_ENGINE.as_ref().unwrap().clone()
    }
}

#[tauri::command]
pub async fn search_documents(query: String) -> Result<Vec<SearchResult>, String> {
    if query.trim().is_empty() {
        return Ok(vec![]);
    }
    
    let search_engine = get_search_engine();
    let mut engine = search_engine.lock().map_err(|e| format!("Failed to acquire search engine lock: {}", e))?;
    
    // Perform the actual hybrid search
    let search_results = engine.search(&query).await
        .map_err(|e| format!("Search failed: {}", e))?;
    
    // Convert from hybrid search results to API search results
    let mut api_results = Vec::new();
    
    for (index, result) in search_results.iter().enumerate() {
        // Extract content snippet from the result
        let snippet_text = if result.content.len() > 200 {
            format!("{}...", &result.content[..200])
        } else {
            result.content.clone()
        };
        
        let content_snippet = ContentSnippet {
            text: snippet_text.clone(),
            start_position: 0,
            end_position: snippet_text.len(),
            context: result.content.clone(),
        };
        
        // Determine result type based on search result metadata
        let result_type = match result.source {
            crate::core::hybrid_search::SearchResultSource::FullTextSearch => ResultType::Exact,
            crate::core::hybrid_search::SearchResultSource::SemanticSearch => ResultType::Semantic,
            crate::core::hybrid_search::SearchResultSource::HybridFusion => ResultType::Hybrid,
        };
        
        // Create ranking factors from available scores
        let ranking_factors = RankingFactors {
            text_score: result.relevance_score as f64,
            semantic_score: result.relevance_score as f64, // Use same score for now
            recency_boost: 0.1,
            access_frequency: 0.5, // TODO: Get from user interaction tracking
            file_importance: 0.7,  // TODO: Calculate based on document metadata
        };
        
        // Get file path from document ID (simplified for now)
        let file_path = format!("/documents/{}", result.document_id);
        
        let api_result = SearchResult {
            document_id: result.document_id,
            relevance_score: result.relevance_score as f64,
            matched_content: vec![content_snippet],
            result_type,
            ranking_factors,
            
            // UI display fields
            id: (index + 1).to_string(),
            content: result.content.clone(),
            file_path,
            score: result.relevance_score as f64,
        };
        
        api_results.push(api_result);
    }
    
    // Limit results for UI performance
    api_results.truncate(50);
    
    Ok(api_results)
}

#[tauri::command]
pub async fn search_with_filters(
    query: String,
    file_types: Option<Vec<String>>,
    limit: Option<usize>,
) -> Result<Vec<SearchResult>, String> {
    // For now, just delegate to search_documents
    // TODO: Implement actual filtering based on file_types
    search_documents(query).await
}

#[tauri::command]
pub async fn get_search_suggestions(
    partial_query: String,
) -> Result<Vec<String>, String> {
    // Simple prefix-based suggestions for now
    if partial_query.is_empty() {
        return Ok(vec![]);
    }
    
    let suggestions = vec![
        format!("{} in documents", partial_query),
        format!("{} recent files", partial_query),
        format!("{} pdf", partial_query),
        format!("{} this week", partial_query),
        format!("{} code", partial_query),
    ];
    
    Ok(suggestions)
}