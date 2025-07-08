use crate::core::hybrid_search::{HybridSearchEngine, SearchMode, SearchResultSource};
use crate::core::search_suggestions::{SearchSuggestionSystem, SuggestionConfig};
use crate::core::user_intelligence::{UserIntelligenceSystem, UserIntelligenceConfig};
use crate::core::query_intent::{QueryIntentClassifier, QueryClassifierConfig};
use crate::core::ranking::SearchContext;
use crate::database::Database;
use crate::error::Result;
use crate::app::AppState;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::Mutex;
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResultV2 {
    pub id: String,
    pub file_path: String,
    pub content: String,
    pub highlighted_content: Option<String>,
    pub score: f64,
    pub match_type: String,
    pub file_type: Option<String>,
    pub file_size: Option<i64>,
    pub modified_date: Option<String>,
    pub created_date: Option<String>,
    pub author: Option<String>,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponseV2 {
    pub results: Vec<SearchResultV2>,
    pub total: usize,
    pub query_intent: Option<String>,
    pub suggested_query: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchFiltersV2 {
    pub file_types: Vec<String>,
    pub date_range: String,
    pub search_mode: String,
    pub include_content: bool,
    pub include_ocr: bool,
    pub search_images: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchSuggestionV2 {
    pub query: String,
    #[serde(rename = "type")]
    pub suggestion_type: String,
    pub category: Option<String>,
    pub count: Option<usize>,
    pub confidence: Option<f32>,
}

// Create search engine directly without global state
fn create_search_engine(database: Arc<Database>) -> HybridSearchEngine {
    HybridSearchEngine::new(database)
}

async fn create_suggestion_engine() -> Result<SearchSuggestionSystem> {
    let config = SuggestionConfig::default();
    let user_config = UserIntelligenceConfig::default();
    let user_intelligence = UserIntelligenceSystem::new(user_config);
    let classifier_config = QueryClassifierConfig::default();
    let intent_classifier = QueryIntentClassifier::new(classifier_config)?;
    let engine = SearchSuggestionSystem::new(config, user_intelligence, intent_classifier).await?;
    Ok(engine)
}

#[tauri::command]
pub async fn search(
    database: tauri::State<'_, Arc<Database>>,
    query: String,
    filters: SearchFiltersV2
) -> std::result::Result<SearchResponseV2, String> {
    // Get database from state
    let database = database.inner().clone();
    
    // Create search engine directly
    let mut search_engine = create_search_engine(database.clone());
    
    // Parse date range filter
    let date_filter = match filters.date_range.as_str() {
        "today" => Some((Utc::now() - Duration::days(1), Utc::now())),
        "week" => Some((Utc::now() - Duration::days(7), Utc::now())),
        "month" => Some((Utc::now() - Duration::days(30), Utc::now())),
        "year" => Some((Utc::now() - Duration::days(365), Utc::now())),
        _ => None,
    };
    
    // Determine search mode
    let search_mode = match filters.search_mode.as_str() {
        "exact" => SearchMode::Precise,
        "semantic" => SearchMode::Exploratory,
        _ => SearchMode::Balanced,
    };
    
    // Perform search with mode
    search_engine.set_mode(search_mode);
    let results = search_engine.search(&query).await
        .map_err(|e| format!("Search failed: {}", e))?;
    
    // Convert results to frontend format
    let mut frontend_results = Vec::new();
    
    for (idx, result) in results.iter().enumerate() {
        // Get document from database
        let document_opt = database.get_document_by_id(&result.document_id)
            .map_err(|e| format!("Failed to get document: {}", e))?;
        
        let document = match document_opt {
            Some(doc) => doc,
            None => continue, // Skip if document not found
        };
        
        // Apply filters
        if !filters.file_types.is_empty() {
            let file_type_str = match document.file_type {
                crate::models::FileType::Pdf => "pdf",
                crate::models::FileType::Docx => "docx",
                crate::models::FileType::Text => "txt",
                crate::models::FileType::Markdown => "md",
                crate::models::FileType::Email => "email",
                crate::models::FileType::Image => "image",
                crate::models::FileType::Audio => "audio",
                crate::models::FileType::Video => "video",
                _ => "unknown",
            };
            if !filters.file_types.contains(&file_type_str.to_string()) {
                continue;
            }
        }
        
        if let Some((start, end)) = date_filter {
            if document.modification_date < start || document.modification_date > end {
                continue;
            }
        }
        
        // Create highlighted content
        let highlighted_content = if query.len() > 2 {
            let query_lower = query.to_lowercase();
            let content_lower = result.content.to_lowercase();
            
            if let Some(pos) = content_lower.find(&query_lower) {
                let start = pos.saturating_sub(50);
                let end = (pos + query_lower.len() + 150).min(result.content.len());
                let snippet = &result.content[start..end];
                
                Some(snippet.replace(&query, &format!("<mark>{}</mark>", query)))
            } else {
                None
            }
        } else {
            None
        };
        
        // Determine match type
        let match_type = match result.source {
            SearchResultSource::FullTextSearch => "exact",
            SearchResultSource::SemanticSearch => "semantic",
            SearchResultSource::HybridFusion => "hybrid",
        };
        
        frontend_results.push(SearchResultV2 {
            id: format!("result_{}", idx),
            file_path: document.file_path.clone(),
            content: result.content.clone(),
            highlighted_content,
            score: result.relevance_score as f64,
            match_type: match_type.to_string(),
            file_type: Some(match document.file_type {
                crate::models::FileType::Pdf => "pdf".to_string(),
                crate::models::FileType::Docx => "docx".to_string(),
                crate::models::FileType::Text => "txt".to_string(),
                crate::models::FileType::Markdown => "md".to_string(),
                crate::models::FileType::Email => "email".to_string(),
                crate::models::FileType::Image => "image".to_string(),
                crate::models::FileType::Audio => "audio".to_string(),
                crate::models::FileType::Video => "video".to_string(),
                _ => "unknown".to_string(),
            }),
            file_size: Some(document.file_size as i64),
            modified_date: Some(document.modification_date.to_rfc3339()),
            created_date: Some(document.creation_date.to_rfc3339()),
            author: document.metadata.get("author").cloned(),
            tags: document.metadata.get("tags").and_then(|t| serde_json::from_str(t).ok()),
        });
    }
    
    // TODO: Get query intent from advanced query processor
    let query_intent = if query.contains(" AND ") || query.contains(" OR ") {
        Some("Boolean search query detected".to_string())
    } else if query.contains("type:") || query.contains("author:") {
        Some("Field-specific search query".to_string())
    } else {
        None
    };
    
    Ok(SearchResponseV2 {
        total: frontend_results.len(),
        results: frontend_results,
        query_intent,
        suggested_query: None,
    })
}

#[tauri::command]
pub async fn get_search_suggestions_v2(
    database: tauri::State<'_, Arc<Database>>,
    partial_query: String,
    context: Option<String>
) -> std::result::Result<Vec<SearchSuggestionV2>, String> {
    let _database = database.inner().clone();
    
    let suggestion_engine = create_suggestion_engine().await
        .map_err(|e| format!("Failed to initialize suggestion engine: {}", e))?;
    
    // Create a dummy user_id for now - in a real app this would come from authentication
    let user_id = Uuid::new_v4();
    
    // Create a search context
    let context = SearchContext {
        session_id: Uuid::new_v4(),
        current_project: None,
        recent_documents: Vec::new(),
        active_applications: Vec::new(),
        search_history: vec![partial_query.clone()],
        timestamp: Utc::now(),
    };
    
    let suggestions = suggestion_engine.get_suggestions(user_id, &partial_query, &context).await
        .map_err(|e| format!("Failed to get suggestions: {}", e))?;
    
    // Convert to frontend format
    let frontend_suggestions: Vec<SearchSuggestionV2> = suggestions.into_iter()
        .map(|s| SearchSuggestionV2 {
            query: s.base.query,
            suggestion_type: match s.base.source {
                crate::core::user_intelligence::SuggestionSource::Personal => "history".to_string(),
                crate::core::user_intelligence::SuggestionSource::Popular => "completion".to_string(),
                crate::core::user_intelligence::SuggestionSource::Contextual => "contextual".to_string(),
                crate::core::user_intelligence::SuggestionSource::Trending => "trending".to_string(),
                crate::core::user_intelligence::SuggestionSource::Semantic => "semantic".to_string(),
            },
            category: s.category,
            count: Some(s.recent_usage_count),
            confidence: Some(s.base.confidence),
        })
        .collect();
    
    Ok(frontend_suggestions)
}

#[tauri::command]
pub async fn get_file_type_counts(
    database: tauri::State<'_, Arc<Database>>
) -> std::result::Result<HashMap<String, usize>, String> {
    // For now, return empty counts since the method doesn't exist
    // TODO: Implement proper file type statistics in database
    let mut counts = HashMap::new();
    counts.insert("pdf".to_string(), 0);
    counts.insert("docx".to_string(), 0);
    counts.insert("txt".to_string(), 0);
    counts.insert("md".to_string(), 0);
    counts.insert("email".to_string(), 0);
    counts.insert("image".to_string(), 0);
    counts.insert("audio".to_string(), 0);
    counts.insert("video".to_string(), 0);
    
    Ok(counts)
}