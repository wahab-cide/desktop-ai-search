use crate::models::SearchResult;

#[tauri::command]
pub async fn search_documents(query: String) -> Result<Vec<SearchResult>, String> {
    // TODO: Implement actual search logic
    Ok(vec![SearchResult {
        id: "1".to_string(),
        content: format!("Mock result for: {}", query),
        file_path: "/mock/path".to_string(),
        score: 0.95,
    }])
}