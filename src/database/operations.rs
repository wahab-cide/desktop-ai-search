use crate::error::{AppError, DatabaseError, Result};
use crate::models::{Document, FileType, IndexMetadata, IndexingStatus, PerformanceMetrics};
use crate::database::Database;
use rusqlite::{params, Transaction};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::collections::HashMap;
use regex::Regex;

// Helper to convert rusqlite errors
trait SqliteResultExt<T> {
    fn map_db_err(self) -> Result<T>;
}

impl<T> SqliteResultExt<T> for rusqlite::Result<T> {
    fn map_db_err(self) -> Result<T> {
        self.map_err(|e| AppError::Database(DatabaseError::Sqlite(e)))
    }
}

/// Prepare FTS5 query by escaping special characters and handling phrases
fn prepare_fts5_query(query: &str) -> String {
    let query = query.trim();
    
    // If already contains quotes or special operators, assume user knows FTS5 syntax
    if query.contains('"') || query.contains(" AND ") || query.contains(" OR ") {
        return query.to_string();
    }
    
    // For simple queries, use prefix matching on each word
    let words: Vec<&str> = query.split_whitespace().collect();
    if words.is_empty() {
        return query.to_string();
    }
    
    // Create a query that matches any of the words with prefix matching
    let terms: Vec<String> = words.iter()
        .map(|word| {
            let escaped = word.replace('"', "\"\"")
                             .replace('*', "")
                             .replace(':', "")
                             .replace('(', "")
                             .replace(')', "");
            // Use prefix matching for better results
            format!("{}*", escaped)
        })
        .collect();
    
    // Join with OR to match any of the terms
    terms.join(" OR ")
}

use crate::core::chunker::TextChunk;

impl Database {
    pub fn insert_document(&self, document: &Document) -> Result<()> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "INSERT INTO documents (
                id, file_path, content_hash, file_type, creation_date, 
                modification_date, last_indexed, file_size, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ).map_db_err()?;
        
        let metadata_json = serde_json::to_string(&document.metadata)?;
        let file_type_json = serde_json::to_string(&document.file_type)?;
        
        stmt.execute(params![
            document.id.to_string(),
            document.file_path,
            document.content_hash,
            file_type_json,
            document.creation_date.timestamp(),
            document.modification_date.timestamp(),
            document.last_indexed.timestamp(),
            document.file_size,
            metadata_json
        ]).map_db_err()?;
        
        Ok(())
    }
    
    pub fn update_document(&self, document: &Document) -> Result<()> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "UPDATE documents SET 
                content_hash = ?, file_type = ?, modification_date = ?, 
                last_indexed = ?, file_size = ?, metadata = ?
            WHERE id = ?"
        ).map_db_err()?;
        
        let metadata_json = serde_json::to_string(&document.metadata)?;
        let file_type_json = serde_json::to_string(&document.file_type)?;
        
        stmt.execute(params![
            document.content_hash,
            file_type_json,
            document.modification_date.timestamp(),
            document.last_indexed.timestamp(),
            document.file_size,
            metadata_json,
            document.id.to_string()
        ]).map_db_err()?;
        
        Ok(())
    }
    
    pub fn get_document_by_id(&self, id: &Uuid) -> Result<Option<Document>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT * FROM documents WHERE id = ?"
        ).map_db_err()?;
        
        let result = stmt.query_row(params![id.to_string()], |row| {
            Document::from_row(row)
        });
        
        match result {
            Ok(document) => Ok(Some(document)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(AppError::Database(DatabaseError::Sqlite(e))),
        }
    }
    
    pub fn get_document_by_path(&self, path: &str) -> Result<Option<Document>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT * FROM documents WHERE file_path = ?"
        ).map_db_err()?;
        
        let result = stmt.query_row(params![path], |row| {
            Document::from_row(row)
        });
        
        match result {
            Ok(document) => Ok(Some(document)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(AppError::Database(DatabaseError::Sqlite(e))),
        }
    }
    
    pub fn get_document_by_hash(&self, hash: &str) -> Result<Option<Document>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT * FROM documents WHERE content_hash = ?"
        ).map_db_err()?;
        
        let result = stmt.query_row(params![hash], |row| {
            Document::from_row(row)
        });
        
        match result {
            Ok(document) => Ok(Some(document)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(AppError::Database(DatabaseError::Sqlite(e))),
        }
    }
    
    /// Insert or update document (upsert) - handles duplicate file paths gracefully
    /// Returns the actual document ID that was used in the database
    pub fn upsert_document(&self, document: &Document) -> Result<Uuid> {
        // Check if document already exists by file path
        if let Some(existing_doc) = self.get_document_by_path(&document.file_path)? {
            // Clear existing chunks and content for this document
            self.delete_document_chunks(&existing_doc.id)?;
            
            // Update existing document with new data but keep existing ID
            let updated_doc = Document {
                id: existing_doc.id, // Keep the existing ID
                ..document.clone()
            };
            self.update_document(&updated_doc)?;
            Ok(existing_doc.id) // Return the existing ID
        } else {
            // Insert new document
            self.insert_document(document)?;
            Ok(document.id) // Return the new ID
        }
    }
    
    pub fn delete_document(&self, id: &Uuid) -> Result<()> {
        let conn = self.get_connection()?;
        let tx = conn.unchecked_transaction().map_db_err()?;
        
        // Delete from FTS index first
        tx.execute(
            "DELETE FROM search_index WHERE rowid IN (
                SELECT rowid FROM documents WHERE id = ?
            )",
            params![id.to_string()]
        ).map_db_err()?;
        
        // Delete document chunks
        tx.execute(
            "DELETE FROM document_chunks WHERE document_id = ?",
            params![id.to_string()]
        ).map_db_err()?;
        
        // Delete embeddings
        tx.execute(
            "DELETE FROM embeddings WHERE document_id = ?",
            params![id.to_string()]
        ).map_db_err()?;
        
        // Delete document
        tx.execute(
            "DELETE FROM documents WHERE id = ?",
            params![id.to_string()]
        ).map_db_err()?;
        
        tx.commit().map_db_err()?;
        Ok(())
    }
    
    pub fn insert_document_content(&self, document_id: &Uuid, content: &str) -> Result<()> {
        let conn = self.get_connection()?;
        
        // Insert into FTS index
        conn.execute(
            "INSERT INTO search_index(rowid, content) 
            SELECT rowid, ? FROM documents WHERE id = ?",
            params![content, document_id.to_string()]
        ).map_db_err()?;
        
        Ok(())
    }
    
    pub fn insert_document_chunk(
        &self, 
        document_id: &Uuid, 
        chunk_index: usize, 
        start_char: usize, 
        end_char: usize, 
        content: &str
    ) -> Result<()> {
        let conn = self.get_connection()?;
        let chunk_id = Uuid::new_v4();
        
        conn.execute(
            "INSERT INTO document_chunks (
                id, document_id, chunk_index, start_char, end_char, content, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                chunk_id.to_string(),
                document_id.to_string(),
                chunk_index,
                start_char,
                end_char,
                content,
                Utc::now().timestamp()
            ]
        ).map_db_err()?;
        
        Ok(())
    }
    
    pub fn full_text_search(&self, query: &str, limit: usize) -> Result<Vec<Document>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT d.* FROM documents d 
            JOIN search_index si ON d.rowid = si.rowid 
            WHERE search_index MATCH ? 
            ORDER BY rank 
            LIMIT ?"
        ).map_db_err()?;
        
        let rows = stmt.query_map(params![query, limit], |row| {
            Document::from_row(row)
        }).map_db_err()?;
        
        let mut documents = Vec::new();
        for row in rows {
            documents.push(row.map_db_err()?);
        }
        
        Ok(documents)
    }
    
    pub fn get_indexing_status(&self) -> Result<IndexMetadata> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT * FROM indexing_status WHERE id = 1"
        ).map_db_err()?;
        
        let result = stmt.query_row([], |row| {
            let errors_json: String = row.get("errors")?;
            let metrics_json: String = row.get("performance_metrics")?;
            let status_str: String = row.get("status")?;
            
            let error_counts: HashMap<String, usize> = serde_json::from_str(&errors_json)
                .unwrap_or_default();
            let performance_metrics: PerformanceMetrics = serde_json::from_str(&metrics_json)
                .unwrap_or(PerformanceMetrics {
                    documents_per_second: 0.0,
                    bytes_per_second: 0.0,
                    average_processing_time: 0.0,
                    memory_usage_mb: 0.0,
                });
            
            let indexing_status = match status_str.as_str() {
                "idle" => IndexingStatus::Idle,
                "running" => IndexingStatus::Running,
                "paused" => IndexingStatus::Paused,
                s if s.starts_with("error:") => IndexingStatus::Error(s[6..].to_string()),
                _ => IndexingStatus::Idle,
            };
            
            Ok(IndexMetadata {
                indexing_status,
                total_documents: row.get::<_, usize>("total_files")?,
                last_full_index: DateTime::from_timestamp(row.get::<_, i64>("last_update")?, 0)
                    .unwrap_or(DateTime::default()),
                error_counts,
                performance_metrics,
            })
        }).map_db_err()?;
        
        Ok(result)
    }
    
    pub fn update_indexing_status(&self, status: &IndexingStatus) -> Result<()> {
        let conn = self.get_connection()?;
        let status_str = match status {
            IndexingStatus::Idle => "idle".to_string(),
            IndexingStatus::Running => "running".to_string(),
            IndexingStatus::Paused => "paused".to_string(),
            IndexingStatus::Error(msg) => format!("error:{}", msg),
        };
        
        conn.execute(
            "UPDATE indexing_status SET status = ?, last_update = ? WHERE id = 1",
            params![status_str, Utc::now().timestamp()]
        ).map_db_err()?;
        
        Ok(())
    }
    
    pub fn get_document_count(&self) -> Result<usize> {
        let conn = self.get_connection()?;
        let count: usize = conn.query_row(
            "SELECT COUNT(*) FROM documents",
            [],
            |row| row.get(0)
        ).map_db_err()?;
        
        Ok(count)
    }
    
    pub fn get_documents_modified_since(&self, since: DateTime<Utc>) -> Result<Vec<Document>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT * FROM documents WHERE modification_date > ? ORDER BY modification_date DESC"
        ).map_db_err()?;
        
        let rows = stmt.query_map(params![since.timestamp()], |row| {
            Document::from_row(row)
        }).map_db_err()?;
        
        let mut documents = Vec::new();
        for row in rows {
            documents.push(row.map_db_err()?);
        }
        
        Ok(documents)
    }
    
    pub fn batch_insert_documents(&self, documents: &[Document]) -> Result<()> {
        let conn = self.get_connection()?;
        let tx = conn.unchecked_transaction().map_db_err()?;
        
        {
            let mut stmt = tx.prepare(
                "INSERT INTO documents (
                    id, file_path, content_hash, file_type, creation_date, 
                    modification_date, last_indexed, file_size, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ).map_db_err()?;
            
            for document in documents {
                let metadata_json = serde_json::to_string(&document.metadata)?;
                let file_type_json = serde_json::to_string(&document.file_type)?;
                
                stmt.execute(params![
                    document.id.to_string(),
                    document.file_path,
                    document.content_hash,
                    file_type_json,
                    document.creation_date.timestamp(),
                    document.modification_date.timestamp(),
                    document.last_indexed.timestamp(),
                    document.file_size,
                    metadata_json
                ]).map_db_err()?;
            }
        }
        
        tx.commit().map_db_err()?;
        Ok(())
    }
    
    // === Chunk and Embedding Operations ===
    
    /// Insert a document chunk with its embedding
    pub fn insert_chunk_with_embedding(
        &self, 
        chunk_id: &str,
        document_id: &Uuid,
        chunk_index: usize,
        start_char: usize,
        end_char: usize,
        content: &str,
        word_count: usize,
        sentence_count: usize,
        overlap_start: bool,
        overlap_end: bool,
        embedding: Option<&[f32]>,
        model_id: Option<&str>,
    ) -> Result<()> {
        let conn = self.get_connection()?;
        let tx = conn.unchecked_transaction().map_db_err()?;
        
        // Insert chunk
        {
            let mut stmt = tx.prepare(
                "INSERT INTO document_chunks (
                    id, document_id, chunk_index, start_char, end_char, content,
                    word_count, sentence_count, overlap_start, overlap_end, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ).map_db_err()?;
            
            stmt.execute(params![
                chunk_id,
                document_id.to_string(),
                chunk_index as i64,
                start_char as i64,
                end_char as i64,
                content,
                word_count as i64,
                sentence_count as i64,
                overlap_start,
                overlap_end,
                Utc::now().timestamp()
            ]).map_db_err()?;
        }
        
        // Insert embedding if provided
        if let Some(embedding_data) = embedding {
            let mut stmt = tx.prepare(
                "INSERT INTO embeddings (
                    id, chunk_id, document_id, embedding, model_id, 
                    embedding_dimensions, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)"
            ).map_db_err()?;
            
            // Convert f32 slice to bytes
            let embedding_bytes: Vec<u8> = embedding_data.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();
            
            let embedding_id = format!("emb_{}", Uuid::new_v4());
            let model = model_id.unwrap_or("all-minilm-l6-v2");
            
            stmt.execute(params![
                embedding_id,
                chunk_id,
                document_id.to_string(),
                embedding_bytes,
                model,
                embedding_data.len() as i64,
                Utc::now().timestamp()
            ]).map_db_err()?;
        }
        
        tx.commit().map_db_err()?;
        Ok(())
    }
    
    /// Batch insert chunks with embeddings for better performance
    pub fn batch_insert_chunks_with_embeddings(
        &self,
        chunks_data: &[(String, Uuid, usize, usize, usize, String, usize, usize, bool, bool, Option<Vec<f32>>)],
        model_id: &str,
    ) -> Result<()> {
        let conn = self.get_connection()?;
        let tx = conn.unchecked_transaction().map_db_err()?;
        
        {
            // Prepare statements in a scope so they get dropped before commit
            let mut chunk_stmt = tx.prepare(
                "INSERT INTO document_chunks (
                    id, document_id, chunk_index, start_char, end_char, content,
                    word_count, sentence_count, overlap_start, overlap_end, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ).map_db_err()?;
            
            let mut embedding_stmt = tx.prepare(
                "INSERT INTO embeddings (
                    id, chunk_id, document_id, embedding, model_id, 
                    embedding_dimensions, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)"
            ).map_db_err()?;
            
            for (chunk_id, document_id, chunk_index, start_char, end_char, content, 
                 word_count, sentence_count, overlap_start, overlap_end, embedding) in chunks_data {
                
                // Insert chunk
                chunk_stmt.execute(params![
                    chunk_id,
                    document_id.to_string(),
                    *chunk_index as i64,
                    *start_char as i64,
                    *end_char as i64,
                    content,
                    *word_count as i64,
                    *sentence_count as i64,
                    overlap_start,
                    overlap_end,
                    Utc::now().timestamp()
                ]).map_db_err()?;
                
                // Insert embedding if available
                if let Some(embedding_data) = embedding {
                    let embedding_bytes: Vec<u8> = embedding_data.iter()
                        .flat_map(|&f| f.to_le_bytes())
                        .collect();
                    
                    let embedding_id = format!("emb_{}", Uuid::new_v4());
                    
                    embedding_stmt.execute(params![
                        embedding_id,
                        chunk_id,
                        document_id.to_string(),
                        embedding_bytes,
                        model_id,
                        embedding_data.len() as i64,
                        Utc::now().timestamp()
                    ]).map_db_err()?;
                }
            }
        } // Statements are dropped here
        
        tx.commit().map_db_err()?;
        Ok(())
    }
    
    /// Retrieve chunks with embeddings for a document
    pub fn get_chunks_with_embeddings(&self, document_id: &Uuid) -> Result<Vec<ChunkWithEmbedding>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT 
                chunk_id, document_id, chunk_index, start_char, end_char, content,
                word_count, sentence_count, overlap_start, overlap_end, c.created_at,
                embedding_id, embedding, model_id, embedding_dimensions
            FROM chunks_with_embeddings c
            WHERE document_id = ?
            ORDER BY chunk_index"
        ).map_db_err()?;
        
        let rows = stmt.query_map(params![document_id.to_string()], |row| {
            let embedding_data: Option<Vec<u8>> = row.get("embedding")?;
            let embedding_vec = embedding_data.map(|bytes| {
                bytes.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            });
            
            Ok(ChunkWithEmbedding {
                chunk_id: row.get("chunk_id")?,
                document_id: Uuid::parse_str(&row.get::<_, String>("document_id")?).unwrap(),
                chunk_index: row.get::<_, i64>("chunk_index")? as usize,
                start_char: row.get::<_, i64>("start_char")? as usize,
                end_char: row.get::<_, i64>("end_char")? as usize,
                content: row.get("content")?,
                word_count: row.get::<_, i64>("word_count")? as usize,
                sentence_count: row.get::<_, i64>("sentence_count")? as usize,
                overlap_start: row.get("overlap_start")?,
                overlap_end: row.get("overlap_end")?,
                embedding: embedding_vec,
                model_id: row.get("model_id").ok(),
                embedding_dimensions: row.get::<_, Option<i64>>("embedding_dimensions")?.map(|d| d as usize),
            })
        }).map_db_err()?;
        
        let mut chunks = Vec::new();
        for row in rows {
            chunks.push(row.map_db_err()?);
        }
        
        Ok(chunks)
    }
    
    /// Find similar chunks using cosine similarity (basic implementation)
    /// For production, you'd want to use a proper vector database
    pub fn find_similar_chunks(&self, query_embedding: &[f32], limit: usize, threshold: f32) -> Result<Vec<SimilarChunk>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT 
                chunk_id, document_id, chunk_index, content, embedding, model_id
            FROM chunks_with_embeddings 
            WHERE embedding IS NOT NULL
            ORDER BY chunk_index"
        ).map_db_err()?;
        
        let rows = stmt.query_map([], |row| {
            let embedding_data: Vec<u8> = row.get("embedding")?;
            let embedding_vec: Vec<f32> = embedding_data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            
            Ok((
                row.get::<_, String>("chunk_id")?,
                Uuid::parse_str(&row.get::<_, String>("document_id")?).unwrap(),
                row.get::<_, i64>("chunk_index")? as usize,
                row.get::<_, String>("content")?,
                embedding_vec,
                row.get::<_, String>("model_id")?,
            ))
        }).map_db_err()?;
        
        let mut similarities = Vec::new();
        
        for row in rows {
            let (chunk_id, document_id, chunk_index, content, embedding, model_id) = row.map_db_err()?;
            
            // Calculate cosine similarity
            let similarity = cosine_similarity(query_embedding, &embedding);
            
            if similarity >= threshold {
                similarities.push(SimilarChunk {
                    chunk_id,
                    document_id,
                    chunk_index,
                    content,
                    similarity_score: similarity,
                    model_id,
                });
            }
        }
        
        // Sort by similarity score (descending) and limit results
        similarities.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
        similarities.truncate(limit);
        
        Ok(similarities)
    }
    
    /// Full-text search using FTS5 index
    pub fn search_documents(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let conn = self.get_connection()?;
        
        // Prepare FTS5 query - handle different query formats
        let fts_query = prepare_fts5_query(query);
        println!("🔍 FTS query prepared: '{}'", fts_query);
        
        // First, let's try a simpler query to debug
        let test_query = "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?";
        match conn.query_row(test_query, params![&fts_query], |row| row.get::<_, i32>(0)) {
            Ok(count) => println!("🔍 FTS match count for '{}': {}", fts_query, count),
            Err(e) => println!("❌ FTS test query failed: {}", e),
        }
        
        // Simplified query for external content FTS5 - use subquery instead of CTE
        let mut stmt = conn.prepare(
            "SELECT 
                dc.id as chunk_id,
                dc.document_id,
                dc.content,
                1.0 as relevance_score,
                SUBSTR(dc.content, 1, 100) as highlighted_content,
                d.file_path,
                d.file_type
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.rowid IN (
                SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?
            )
            LIMIT ?"
        ).map_db_err()?;
        
        let mut results = Vec::new();
        
        let rows = stmt.query_map(params![&fts_query, limit], |row| {
            let relevance_score: f64 = row.get("relevance_score")?;
            
            Ok(SearchResult {
                chunk_id: row.get("chunk_id")?,
                document_id: Uuid::parse_str(&row.get::<_, String>("document_id")?).unwrap(),
                content: row.get("content")?,
                relevance_score: relevance_score as f32,
                highlighted_content: Some(row.get("highlighted_content")?),
                path: std::path::PathBuf::from(row.get::<_, String>("file_path")?),
                file_type: row.get("file_type")?,
            })
        }).map_db_err()?;
        
        for row in rows {
            results.push(row.map_db_err()?);
        }
        
        Ok(results)
    }
    
    
    /// Delete all chunks and embeddings for a document
    pub fn delete_document_chunks(&self, document_id: &Uuid) -> Result<()> {
        let conn = self.get_connection()?;
        let tx = conn.unchecked_transaction().map_db_err()?;
        
        // Delete embeddings first (due to foreign key constraints)
        tx.execute(
            "DELETE FROM embeddings WHERE document_id = ?",
            params![document_id.to_string()]
        ).map_db_err()?;
        
        // Delete chunks
        tx.execute(
            "DELETE FROM document_chunks WHERE document_id = ?",
            params![document_id.to_string()]
        ).map_db_err()?;
        
        tx.commit().map_db_err()?;
        Ok(())
    }

    /// Insert image metadata
    pub fn insert_image(&self, image: &ImageMetadata) -> Result<()> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "INSERT INTO images (
                id, file_path, file_name, file_size, width, height, format,
                content_hash, creation_date, modification_date, last_indexed, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ).map_db_err()?;
        
        let metadata_json = serde_json::to_string(&image.metadata)?;
        
        stmt.execute(params![
            image.id,
            image.file_path,
            image.file_name,
            image.file_size,
            image.width,
            image.height,
            image.format,
            image.content_hash,
            image.creation_date.timestamp(),
            image.modification_date.timestamp(),
            image.last_indexed.timestamp(),
            metadata_json
        ]).map_db_err()?;
        
        Ok(())
    }

    /// Insert image embedding
    pub fn insert_image_embedding(
        &self,
        image_id: &str,
        model_id: &str,
        embedding: &[f32],
    ) -> Result<()> {
        let conn = self.get_connection()?;
        
        // Convert f32 slice to bytes
        let embedding_bytes: Vec<u8> = embedding.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        
        let mut stmt = conn.prepare(
            "INSERT INTO image_embeddings (
                id, image_id, model_id, embedding, embedding_dimensions
            ) VALUES (?, ?, ?, ?, ?)"
        ).map_db_err()?;
        
        let embedding_id = format!("img_emb_{}", Uuid::new_v4());
        
        stmt.execute(params![
            embedding_id,
            image_id,
            model_id,
            embedding_bytes,
            embedding.len() as i64
        ]).map_db_err()?;
        
        Ok(())
    }

    /// Get images with embeddings
    pub fn get_images_with_embeddings(&self, model_id: Option<&str>) -> Result<Vec<ImageWithEmbedding>> {
        let conn = self.get_connection()?;
        
        let query = if let Some(_model) = model_id {
            "SELECT * FROM images_with_embeddings WHERE embedding_model_id = ?"
        } else {
            "SELECT * FROM images_with_embeddings WHERE embedding IS NOT NULL"
        };
        
        let mut stmt = conn.prepare(query).map_db_err()?;
        
        let mut images = Vec::new();
        
        if let Some(model) = model_id {
            let rows = stmt.query_map(params![model], |row| {
                ImageWithEmbedding::from_row(row)
            }).map_db_err()?;
            
            for row in rows {
                images.push(row.map_db_err()?);
            }
        } else {
            let rows = stmt.query_map([], |row| {
                ImageWithEmbedding::from_row(row)
            }).map_db_err()?;
            
            for row in rows {
                images.push(row.map_db_err()?);
            }
        };
        
        Ok(images)
    }

    /// Get document statistics by file type
    pub async fn get_document_statistics(&self) -> Result<HashMap<String, usize>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT file_type, COUNT(*) as count FROM documents GROUP BY file_type"
        ).map_db_err()?;
        
        let mut stats = HashMap::new();
        let rows = stmt.query_map([], |row| {
            let file_type: String = row.get("file_type")?;
            let count: usize = row.get("count")?;
            Ok((file_type, count))
        }).map_db_err()?;
        
        for row in rows {
            let (file_type, count) = row.map_db_err()?;
            stats.insert(file_type, count);
        }
        
        Ok(stats)
    }

    /// Get total number of chunks
    pub async fn get_total_chunk_count(&self) -> Result<usize> {
        let conn = self.get_connection()?;
        let count: usize = conn.query_row(
            "SELECT COUNT(*) FROM document_chunks",
            [],
            |row| row.get(0)
        ).map_db_err()?;
        
        Ok(count)
    }

    /// Get total number of embeddings
    pub async fn get_total_embedding_count(&self) -> Result<usize> {
        let conn = self.get_connection()?;
        let count: usize = conn.query_row(
            "SELECT COUNT(*) FROM embeddings",
            [],
            |row| row.get(0)
        ).map_db_err()?;
        
        Ok(count)
    }

    /// Check index health and integrity
    pub async fn check_index_health(&self) -> Result<HashMap<String, serde_json::Value>> {
        let conn = self.get_connection()?;
        let mut health = HashMap::new();
        
        // Check database integrity
        let integrity_ok: String = conn.query_row(
            "PRAGMA integrity_check",
            [],
            |row| row.get(0)
        ).map_db_err()?;
        
        health.insert("integrity".to_string(), serde_json::Value::String(integrity_ok));
        
        // Check FTS5 table status
        let fts_count: usize = conn.query_row(
            "SELECT COUNT(*) FROM chunks_fts",
            [],
            |row| row.get(0)
        ).unwrap_or(0);
        
        health.insert("fts5_entries".to_string(), serde_json::Value::Number(fts_count.into()));
        
        // Check for orphaned records
        let orphaned_chunks: usize = conn.query_row(
            "SELECT COUNT(*) FROM document_chunks dc 
             LEFT JOIN documents d ON dc.document_id = d.id 
             WHERE d.id IS NULL",
            [],
            |row| row.get(0)
        ).unwrap_or(0);
        
        health.insert("orphaned_chunks".to_string(), serde_json::Value::Number(orphaned_chunks.into()));
        
        let orphaned_embeddings: usize = conn.query_row(
            "SELECT COUNT(*) FROM embeddings e 
             LEFT JOIN document_chunks dc ON e.chunk_id = dc.id 
             WHERE dc.id IS NULL",
            [],
            |row| row.get(0)
        ).unwrap_or(0);
        
        health.insert("orphaned_embeddings".to_string(), serde_json::Value::Number(orphaned_embeddings.into()));
        
        // Database size information
        let page_count: usize = conn.query_row(
            "PRAGMA page_count",
            [],
            |row| row.get(0)
        ).unwrap_or(0);
        
        let page_size: usize = conn.query_row(
            "PRAGMA page_size",
            [],
            |row| row.get(0)
        ).unwrap_or(0);
        
        let db_size_mb = (page_count * page_size) as f64 / (1024.0 * 1024.0);
        health.insert("database_size_mb".to_string(), serde_json::Value::Number(
            serde_json::Number::from_f64(db_size_mb).unwrap_or(serde_json::Number::from(0))
        ));
        
        Ok(health)
    }

    /// Find similar images based on text query
    pub fn find_similar_images_by_text(
        &self,
        text_query: &str,
        limit: usize,
        threshold: f32,
        model_id: &str,
    ) -> Result<Vec<SimilarImage>> {
        let conn = self.get_connection()?;
        
        let mut stmt = conn.prepare(
            "SELECT 
                i.id as image_id,
                i.file_path,
                i.file_name,
                i.format,
                i.width,
                i.height,
                its.similarity_score,
                its.query_text
            FROM images i
            JOIN image_text_similarities its ON i.id = its.image_id
            WHERE its.query_text = ? 
            AND its.model_id = ?
            AND its.similarity_score >= ?
            ORDER BY its.similarity_score DESC
            LIMIT ?"
        ).map_db_err()?;
        
        let rows = stmt.query_map(params![text_query, model_id, threshold, limit], |row| {
            Ok(SimilarImage {
                id: row.get("image_id")?,
                file_path: row.get("file_path")?,
                file_name: row.get("file_name")?,
                format: row.get("format")?,
                width: row.get::<_, Option<i64>>("width")?.map(|w| w as i32),
                height: row.get::<_, Option<i64>>("height")?.map(|h| h as i32),
                similarity_score: row.get::<_, f64>("similarity_score")? as f32,
                query_text: row.get("query_text")?,
                model_id: row.get("model_id")?,
            })
        }).map_db_err()?;
        
        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_db_err()?);
        }
        
        Ok(results)
    }

    /// Cache image-text similarity result
    pub fn cache_image_text_similarity(
        &self,
        image_id: &str,
        text_hash: &str,
        query_text: &str,
        similarity_score: f32,
        model_id: &str,
    ) -> Result<()> {
        let conn = self.get_connection()?;
        
        let mut stmt = conn.prepare(
            "INSERT OR REPLACE INTO image_text_similarities (
                id, image_id, text_hash, query_text, similarity_score, model_id
            ) VALUES (?, ?, ?, ?, ?, ?)"
        ).map_db_err()?;
        
        let similarity_id = format!("sim_{}", Uuid::new_v4());
        
        stmt.execute(params![
            similarity_id,
            image_id,
            text_hash,
            query_text,
            similarity_score,
            model_id
        ]).map_db_err()?;
        
        Ok(())
    }

    /// Get cached text embedding
    pub fn get_cached_text_embedding(&self, text_hash: &str, model_id: &str) -> Result<Option<Vec<f32>>> {
        let conn = self.get_connection()?;
        
        let mut stmt = conn.prepare(
            "SELECT embedding FROM text_embeddings 
             WHERE text_hash = ? AND model_id = ?"
        ).map_db_err()?;
        
        let result = stmt.query_row(params![text_hash, model_id], |row| {
            let embedding_data: Vec<u8> = row.get("embedding")?;
            let embedding_vec: Vec<f32> = embedding_data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok(embedding_vec)
        });
        
        match result {
            Ok(embedding) => Ok(Some(embedding)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(AppError::Database(DatabaseError::Sqlite(e))),
        }
    }

    /// Cache text embedding
    pub fn cache_text_embedding(
        &self,
        text_hash: &str,
        text_content: &str,
        embedding: &[f32],
        model_id: &str,
    ) -> Result<()> {
        let conn = self.get_connection()?;
        
        // Convert f32 slice to bytes
        let embedding_bytes: Vec<u8> = embedding.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        
        let mut stmt = conn.prepare(
            "INSERT OR REPLACE INTO text_embeddings (
                id, text_hash, text_content, embedding, embedding_dimensions, model_id
            ) VALUES (?, ?, ?, ?, ?, ?)"
        ).map_db_err()?;
        
        let embedding_id = format!("txt_emb_{}", Uuid::new_v4());
        
        stmt.execute(params![
            embedding_id,
            text_hash,
            text_content,
            embedding_bytes,
            embedding.len() as i64,
            model_id
        ]).map_db_err()?;
        
        Ok(())
    }
}

// Helper struct for returning chunk data with embeddings
#[derive(Debug, Clone)]
pub struct ChunkWithEmbedding {
    pub chunk_id: String,
    pub document_id: Uuid,
    pub chunk_index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub content: String,
    pub word_count: usize,
    pub sentence_count: usize,
    pub overlap_start: bool,
    pub overlap_end: bool,
    pub embedding: Option<Vec<f32>>,
    pub model_id: Option<String>,
    pub embedding_dimensions: Option<usize>,
}

// Helper struct for similarity search results
#[derive(Debug, Clone)]
pub struct SimilarChunk {
    pub chunk_id: String,
    pub document_id: Uuid,
    pub chunk_index: usize,
    pub content: String,
    pub similarity_score: f32,
    pub model_id: String,
}

// Helper struct for FTS search results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub document_id: Uuid,
    pub content: String,
    pub relevance_score: f32,
    pub highlighted_content: Option<String>,
    pub path: std::path::PathBuf,
    pub file_type: String,
}

// Helper function to calculate cosine similarity
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

// CLIP-related helper structs
#[derive(Debug, Clone)]
pub struct ImageMetadata {
    pub id: String,
    pub file_path: String,
    pub file_name: String,
    pub file_size: i64,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub format: Option<String>,
    pub content_hash: String,
    pub creation_date: DateTime<Utc>,
    pub modification_date: DateTime<Utc>,
    pub last_indexed: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ImageWithEmbedding {
    pub id: String,
    pub file_path: String,
    pub file_name: String,
    pub file_size: i64,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub format: Option<String>,
    pub content_hash: String,
    pub creation_date: DateTime<Utc>,
    pub modification_date: DateTime<Utc>,
    pub last_indexed: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
    pub embedding_model_id: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub embedding_dimensions: Option<usize>,
    pub embedding_created_at: Option<DateTime<Utc>>,
}

impl ImageWithEmbedding {
    pub fn from_row(row: &rusqlite::Row) -> rusqlite::Result<Self> {
        // Convert binary embedding back to f32 vector
        let embedding: Option<Vec<f32>> = match row.get::<_, Option<Vec<u8>>>("embedding")? {
            Some(bytes) => {
                let floats: Vec<f32> = bytes.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Some(floats)
            }
            None => None,
        };

        // Parse metadata JSON
        let metadata_json: String = row.get("metadata")?;
        let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)
            .unwrap_or_default();

        Ok(ImageWithEmbedding {
            id: row.get("id")?,
            file_path: row.get("file_path")?,
            file_name: row.get("file_name")?,
            file_size: row.get("file_size")?,
            width: row.get("width")?,
            height: row.get("height")?,
            format: row.get("format")?,
            content_hash: row.get("content_hash")?,
            creation_date: DateTime::from_timestamp(row.get::<_, i64>("creation_date")?, 0)
                .unwrap_or_else(Utc::now),
            modification_date: DateTime::from_timestamp(row.get::<_, i64>("modification_date")?, 0)
                .unwrap_or_else(Utc::now),
            last_indexed: DateTime::from_timestamp(row.get::<_, i64>("last_indexed")?, 0)
                .unwrap_or_else(Utc::now),
            metadata,
            embedding_model_id: row.get("embedding_model_id")?,
            embedding,
            embedding_dimensions: row.get::<_, Option<i64>>("embedding_dimensions")?.map(|d| d as usize),
            embedding_created_at: row.get::<_, Option<i64>>("embedding_created_at")?
                .and_then(|ts| DateTime::from_timestamp(ts, 0)),
        })
    }
}

#[derive(Debug, Clone)]
pub struct SimilarImage {
    pub id: String,
    pub file_path: String,
    pub file_name: String,
    pub format: Option<String>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub similarity_score: f32,
    pub query_text: String,
    pub model_id: String,
}

impl Database {
    /// Get chunks that don't have embeddings yet
    pub async fn get_chunks_without_embeddings(&self, limit: usize) -> Result<Vec<TextChunk>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT c.id, c.document_id, c.content, c.start_position, c.end_position,
                    c.chunk_index, c.total_chunks, c.metadata, c.created_at
             FROM document_chunks c
             LEFT JOIN embeddings e ON c.id = e.chunk_id
             WHERE e.id IS NULL
             LIMIT ?"
        ).map_db_err()?;
        
        let chunks = stmt.query_map(params![limit as i64], |row| {
            Ok(TextChunk {
                id: row.get("id")?,
                content: row.get("content")?,
                start_char: row.get::<_, i64>("start_position")? as usize,
                end_char: row.get::<_, i64>("end_position")? as usize,
                chunk_index: row.get::<_, i64>("chunk_index")? as usize,
                word_count: 0, // Default value
                sentence_count: 0, // Default value
                overlap_start: false, // Default value
                overlap_end: false, // Default value
                embedding: None,
                
                // Database fields
                document_id: Some(row.get("document_id")?),
                start_position: Some(row.get::<_, i64>("start_position")? as usize),
                end_position: Some(row.get::<_, i64>("end_position")? as usize),
                total_chunks: Some(row.get::<_, i64>("total_chunks")? as usize),
                metadata: row.get::<_, Option<String>>("metadata")?
                    .and_then(|s| serde_json::from_str(&s).ok()),
            })
        }).map_db_err()?.collect::<rusqlite::Result<Vec<_>>>().map_db_err()?;
        
        Ok(chunks)
    }
    
    /// Store embedding for a chunk
    pub async fn store_chunk_embedding(
        &self,
        chunk_id: &str,
        document_id: &str,
        embedding: &[f32],
        model_id: &str,
    ) -> Result<()> {
        let conn = self.get_connection()?;
        
        // Convert embedding to bytes
        let embedding_bytes: Vec<u8> = embedding.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        
        let embedding_id = format!("emb_{}", Uuid::new_v4());
        
        conn.execute(
            "INSERT INTO embeddings (
                id, chunk_id, document_id, embedding, model_id,
                embedding_dimensions, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                embedding_id,
                chunk_id,
                document_id,
                embedding_bytes,
                model_id,
                embedding.len() as i64,
                Utc::now().timestamp(),
            ],
        ).map_db_err()?;
        
        Ok(())
    }
    
    /// Get file type counts for all indexed documents
    pub fn get_file_type_counts(&self) -> Result<HashMap<String, usize>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT file_type, COUNT(*) as count FROM documents GROUP BY file_type"
        ).map_db_err()?;
        
        let mut counts = HashMap::new();
        let rows = stmt.query_map([], |row| {
            let file_type_json: String = row.get("file_type")?;
            let count: i64 = row.get("count")?;
            Ok((file_type_json, count as usize))
        }).map_db_err()?;
        
        for row in rows {
            let (file_type_json, count) = row.map_db_err()?;
            // Parse the FileType enum to get the string representation
            if let Ok(file_type) = serde_json::from_str::<crate::models::FileType>(&file_type_json) {
                let type_str = match file_type {
                    crate::models::FileType::Pdf => "pdf",
                    crate::models::FileType::Docx => "docx",
                    crate::models::FileType::Text => "txt",
                    crate::models::FileType::Markdown => "md",
                    crate::models::FileType::Email => "email",
                    crate::models::FileType::Image => {
                        // For images, we need to check the actual file extension
                        // This is a simplified approach - we'll count all images as different types
                        "jpg" // Default to jpg for now
                    },
                    crate::models::FileType::Audio => "mp3",
                    crate::models::FileType::Video => "mp4",
                    _ => "unknown",
                };
                *counts.entry(type_str.to_string()).or_insert(0) += count;
            }
        }
        
        // Get more detailed image counts by extension
        let image_counts = self.get_image_file_counts()?;
        for (ext, count) in image_counts {
            counts.insert(ext, count);
        }
        
        Ok(counts)
    }
    
    /// Get detailed image file counts by extension
    fn get_image_file_counts(&self) -> Result<HashMap<String, usize>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT file_path FROM documents WHERE file_type = ?"
        ).map_db_err()?;
        
        let file_type_json = serde_json::to_string(&crate::models::FileType::Image)?;
        let rows = stmt.query_map(params![file_type_json], |row| {
            let file_path: String = row.get("file_path")?;
            Ok(file_path)
        }).map_db_err()?;
        
        let mut counts = HashMap::new();
        for row in rows {
            let file_path = row.map_db_err()?;
            if let Some(ext) = file_path.split('.').last() {
                let ext_lower = ext.to_lowercase();
                *counts.entry(ext_lower).or_insert(0) += 1;
            }
        }
        
        Ok(counts)
    }
    
    /// Get all documents of a specific file type
    pub fn get_documents_by_file_type(&self, file_type: &str, limit: Option<usize>) -> Result<Vec<Document>> {
        let conn = self.get_connection()?;
        
        // Map frontend file type strings to our FileType enum
        let db_file_type = match file_type {
            "pdf" => crate::models::FileType::Pdf,
            "docx" | "doc" => crate::models::FileType::Docx, // Map both docx and doc to Docx
            "txt" => crate::models::FileType::Text,
            "md" => crate::models::FileType::Markdown,
            "email" => crate::models::FileType::Email,
            "jpg" | "jpeg" | "png" | "gif" | "webp" | "svg" => crate::models::FileType::Image,
            "mp3" | "wav" | "flac" => crate::models::FileType::Audio,
            "mp4" | "avi" | "mov" | "mkv" => crate::models::FileType::Video,
            _ => return Ok(Vec::new()), // Unknown file type
        };
        
        let file_type_json = serde_json::to_string(&db_file_type)?;
        
        let query = if let Some(limit) = limit {
            format!(
                "SELECT * FROM documents WHERE file_type = ? ORDER BY modification_date DESC LIMIT {}",
                limit
            )
        } else {
            "SELECT * FROM documents WHERE file_type = ? ORDER BY modification_date DESC".to_string()
        };
        
        let mut stmt = conn.prepare(&query).map_db_err()?;
        let rows = stmt.query_map(params![file_type_json], |row| {
            Document::from_row(row)
        }).map_db_err()?;
        
        let mut documents = Vec::new();
        for row in rows {
            let document = row.map_db_err()?;
            
            // For image types, filter by actual file extension
            if matches!(db_file_type, crate::models::FileType::Image) {
                if let Some(ext) = document.file_path.split('.').last() {
                    let ext_lower = ext.to_lowercase();
                    if file_type == ext_lower || (file_type == "jpeg" && ext_lower == "jpg") {
                        documents.push(document);
                    }
                }
            } else {
                documents.push(document);
            }
        }
        
        Ok(documents)
    }
    
    /// Get all documents with pagination
    pub fn get_all_documents_paginated(&self, offset: usize, limit: usize) -> Result<Vec<Document>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT * FROM documents ORDER BY modification_date DESC LIMIT ? OFFSET ?"
        ).map_db_err()?;
        
        let rows = stmt.query_map(params![limit, offset], |row| {
            Document::from_row(row)
        }).map_db_err()?;
        
        let mut documents = Vec::new();
        for row in rows {
            documents.push(row.map_db_err()?);
        }
        
        Ok(documents)
    }
    
    /// Get total document count
    pub fn get_total_document_count(&self) -> Result<usize> {
        let conn = self.get_connection()?;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM documents",
            [],
            |row| row.get(0)
        ).map_db_err()?;
        
        Ok(count as usize)
    }
    
    /// Rebuild FTS5 index from scratch
    pub fn rebuild_fts_index(&self) -> Result<usize> {
        let conn = self.get_connection()?;
        let tx = conn.unchecked_transaction().map_db_err()?;
        
        println!("🔧 Rebuilding FTS5 index...");
        
        // First, clear the existing FTS index
        tx.execute("DELETE FROM chunks_fts", [])
            .map_db_err()?;
        
        // Count chunks to process
        let chunk_count: i64 = tx.query_row(
            "SELECT COUNT(*) FROM document_chunks",
            [],
            |row| row.get(0)
        ).map_db_err()?;
        
        println!("🔧 Found {} chunks to index", chunk_count);
        
        // Rebuild the FTS index from document_chunks
        let rows_affected = tx.execute(
            "INSERT INTO chunks_fts(rowid, content, document_id, id)
             SELECT rowid, content, document_id, id FROM document_chunks",
            []
        ).map_db_err()?;
        
        println!("🔧 Inserted {} rows into FTS index", rows_affected);
        
        // Optimize the FTS index
        tx.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')", [])
            .map_db_err()?;
        
        tx.commit().map_db_err()?;
        
        // Verify the rebuild
        let fts_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'the OR a OR of'",
            [],
            |row| row.get(0)
        ).unwrap_or(0);
        
        println!("✅ FTS index rebuilt. Test query found {} matches", fts_count);
        
        Ok(rows_affected)
    }
    
    /// Get all documents without pagination (for cleanup operations)
    pub fn get_all_documents(&self) -> Result<Vec<Document>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT * FROM documents ORDER BY modification_date DESC"
        ).map_db_err()?;
        
        let rows = stmt.query_map([], |row| {
            Document::from_row(row)
        }).map_db_err()?;
        
        let mut documents = Vec::new();
        for row in rows {
            documents.push(row.map_db_err()?);
        }
        
        Ok(documents)
    }
    
    
    /// Delete all chunks for a document and return count of deleted chunks
    pub fn delete_chunks_by_document_id(&self, document_id: &Uuid) -> Result<u32> {
        let conn = self.get_connection()?;
        let document_id_str = document_id.to_string();
        
        // First delete embeddings for these chunks
        conn.execute(
            "DELETE FROM embeddings WHERE document_id = ?",
            params![document_id_str]
        ).map_db_err()?;
        
        // Then delete the chunks
        let rows_affected = conn.execute(
            "DELETE FROM document_chunks WHERE document_id = ?",
            params![document_id_str]
        ).map_db_err()?;
        
        Ok(rows_affected as u32)
    }
}