use crate::models::{SearchResult, ResultType, RankingFactors, ContentSnippet};
use uuid::Uuid;

#[tauri::command]
pub async fn search_documents(query: String) -> Result<Vec<SearchResult>, String> {
    // TODO: Implement actual search logic
    Ok(vec![SearchResult {
        document_id: Uuid::new_v4(),
        relevance_score: 0.95,
        matched_content: vec![ContentSnippet {
            text: format!("Mock result for: {}", query),
            start_position: 0,
            end_position: query.len(),
            context: "Mock context".to_string(),
        }],
        result_type: ResultType::Exact,
        ranking_factors: RankingFactors {
            text_score: 0.9,
            semantic_score: 0.8,
            recency_boost: 0.1,
            access_frequency: 0.5,
            file_importance: 0.7,
        },
        // UI display fields for compatibility
        id: "1".to_string(),
        content: format!("Mock result for: {}", query),
        file_path: "/mock/path".to_string(),
        score: 0.95,
    }])
}