use crate::database::{Database, operations::SearchResult as DbSearchResult};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Validate database schema for debugging
fn validate_database_schema(database: &Database) -> Result<(), String> {
    let conn = database.get_connection()
        .map_err(|e| format!("Failed to get database connection: {}", e))?;
    
    // Check if chunks_fts table exists
    let table_exists: bool = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='chunks_fts'",
        [],
        |row| row.get::<_, i32>(0)
    ).map(|count| count > 0)
    .map_err(|e| format!("Failed to check chunks_fts table: {}", e))?;
    
    println!("🔍 chunks_fts table exists: {}", table_exists);
    
    if table_exists {
        // Check table schema
        let schema_result = conn.prepare("PRAGMA table_info(chunks_fts)")
            .and_then(|mut stmt| {
                let rows = stmt.query_map([], |row| {
                    Ok((
                        row.get::<_, String>(1)?, // column name
                        row.get::<_, String>(2)?, // column type
                    ))
                })?;
                let mut columns = Vec::new();
                for row in rows {
                    columns.push(row?);
                }
                Ok(columns)
            });
            
        match schema_result {
            Ok(columns) => {
                println!("🔍 chunks_fts columns:");
                for (name, type_) in columns {
                    println!("  - {} ({})", name, type_);
                }
            }
            Err(e) => println!("⚠️ Failed to get chunks_fts schema: {}", e),
        }
        
        // For external content FTS5 tables, just check if we can query it
        // Don't use '*' as it's not a valid FTS5 query
        let fts_check: Result<i32, _> = conn.query_row(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'test OR the'",
            [],
            |row| row.get(0)
        );
        
        match fts_check {
            Ok(_) => {
                println!("🔍 chunks_fts is queryable");
                // Now check how many document_chunks rows we have
                let dc_count: Result<i32, _> = conn.query_row(
                    "SELECT COUNT(*) FROM document_chunks",
                    [],
                    |row| row.get(0)
                );
                if let Ok(count) = dc_count {
                    println!("🔍 document_chunks has {} rows available for FTS", count);
                }
            },
            Err(e) => {
                println!("⚠️ FTS5 table check failed: {}", e);
                println!("⚠️ This suggests the FTS5 table may not be properly initialized");
            }
        }
    }
    
    // Also check document_chunks table
    let chunks_count: Result<i32, _> = conn.query_row(
        "SELECT COUNT(*) FROM document_chunks",
        [],
        |row| row.get(0)
    );
    
    match chunks_count {
        Ok(count) => println!("🔍 document_chunks has {} rows", count),
        Err(e) => println!("⚠️ Failed to count document_chunks: {}", e),
    }
    
    // Test a simple FTS query to isolate the issue
    if table_exists {
        println!("🔍 Testing simple FTS query...");
        let simple_test = conn.prepare("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'test'")
            .and_then(|mut stmt| {
                stmt.query_row([], |row| row.get::<_, i32>(0))
            });
            
        match simple_test {
            Ok(count) => println!("🔍 Simple FTS test passed, found {} matches for 'test'", count),
            Err(e) => println!("❌ Simple FTS test failed: {}", e),
        }
    }
    
    Ok(())
}

/// Frontend-compatible SearchResult
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub file_path: String,
    pub content: String,
    pub highlighted_content: Option<String>,
    pub score: f64,
    pub match_type: Option<String>,
    pub file_type: Option<String>,
    pub file_size: Option<u64>,
    pub modified_date: Option<String>,
    pub created_date: Option<String>,
    pub author: Option<String>,
    pub tags: Option<Vec<String>>,
}


#[tauri::command]
pub async fn search_documents(
    database: tauri::State<'_, Arc<Database>>,
    query: String
) -> Result<Vec<SearchResult>, String> {
    println!("🔍 Search called with query: '{}'", query);
    
    if query.trim().is_empty() {
        println!("🔍 Empty query, returning empty results");
        return Ok(vec![]);
    }
    
    // Use database from Tauri state (same as indexing)
    let database = database.inner().clone();
    
    // First, let's validate the database schema
    println!("🔍 Validating database schema...");
    validate_database_schema(&database)?;
    
    println!("🔍 Calling database.search_documents with query: '{}'", query);
    
    // Get database search results
    let db_results = database.search_documents(&query, 50)
        .map_err(|e| {
            println!("❌ Database search failed: {}", e);
            format!("Database search failed: {}", e)
        })?;
    
    println!("🔍 Database returned {} results", db_results.len());
    
    // Convert from database search results to API search results
    let mut api_results = Vec::new();
    
    for (index, db_result) in db_results.iter().enumerate() {
        println!("🔍 Processing result {}: document_id={}, score={}", index, db_result.document_id, db_result.relevance_score);
        
        // Get document details from database
        let document = database.get_document_by_id(&db_result.document_id)
            .map_err(|e| {
                println!("❌ Failed to get document {}: {}", db_result.document_id, e);
                format!("Failed to get document: {}", e)
            })?;
        
        let document = match document {
            Some(doc) => {
                println!("🔍 Found document: {}", doc.file_path);
                doc
            },
            None => {
                println!("⚠️ Document {} not found, skipping", db_result.document_id);
                continue; // Skip if document not found
            }
        };
        
        // Determine file type string
        let file_type = match document.file_type {
            crate::models::FileType::Pdf => "pdf",
            crate::models::FileType::Docx => "docx",
            crate::models::FileType::Text => "txt",
            crate::models::FileType::Markdown => "md",
            crate::models::FileType::Html => "html",
            crate::models::FileType::Image => "image",
            crate::models::FileType::Audio => "audio",
            crate::models::FileType::Video => "video",
            crate::models::FileType::Email => "email",
            _ => "unknown",
        };

        let api_result = SearchResult {
            id: (index + 1).to_string(),
            file_path: document.file_path,
            content: db_result.highlighted_content.as_ref().unwrap_or(&db_result.content).clone(),
            highlighted_content: db_result.highlighted_content.clone(),
            score: (db_result.relevance_score as f64).max(0.0).min(1.0), // Normalize to 0-1
            match_type: Some("exact".to_string()), // FTS results are exact matches
            file_type: Some(file_type.to_string()),
            file_size: Some(document.file_size),
            modified_date: Some(document.modification_date.to_rfc3339()),
            created_date: Some(document.creation_date.to_rfc3339()),
            author: document.metadata.get("author").cloned(),
            tags: document.metadata.get("tags").and_then(|t| serde_json::from_str(t).ok()),
        };
        
        api_results.push(api_result);
    }
    
    // Limit results for UI performance
    api_results.truncate(50);
    
    println!("🔍 Search completed. Returning {} results to frontend", api_results.len());
    for (i, result) in api_results.iter().take(3).enumerate() {
        println!("🔍 Result {}: {} (score: {})", i + 1, result.file_path, result.score);
    }
    
    Ok(api_results)
}

#[tauri::command]
pub async fn search_with_filters(
    database: tauri::State<'_, Arc<Database>>,
    query: String,
    file_types: Option<Vec<String>>,
    limit: Option<usize>,
) -> Result<Vec<SearchResult>, String> {
    // For now, just delegate to search_documents
    // TODO: Implement actual filtering based on file_types
    let mut results = search_documents(database, query).await?;
    
    // Apply file type filtering if specified
    if let Some(types) = file_types {
        if !types.is_empty() {
            results.retain(|result| {
                if let Some(ref file_type) = result.file_type {
                    types.contains(file_type)
                } else {
                    false
                }
            });
        }
    }
    
    // Apply limit if specified
    if let Some(limit) = limit {
        results.truncate(limit);
    }
    
    Ok(results)
}

#[tauri::command]
pub async fn rebuild_search_index(
    database: tauri::State<'_, Arc<Database>>,
) -> Result<String, String> {
    println!("🔧 Rebuilding search index...");
    
    let database = database.inner();
    
    // Use the new rebuild_fts_index method
    let rows_affected = database.rebuild_fts_index()
        .map_err(|e| format!("Failed to rebuild FTS index: {}", e))?;
    
    let message = format!("✅ Search index rebuilt successfully with {} entries.", rows_affected);
    println!("{}", message);
    
    Ok(message)
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