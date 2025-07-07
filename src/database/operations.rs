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
    
    // If already contains quotes, assume user knows FTS5 syntax
    if query.contains('"') {
        return query.to_string();
    }
    
    // Handle phrase queries (multiple words)
    if query.contains(' ') {
        let words: Vec<&str> = query.split_whitespace().collect();
        if words.len() > 1 {
            // For multi-word queries, create phrase query (exact match gets priority)
            return format!("\"{}\"", query);
        }
    }
    
    // Single word - escape special characters and add prefix matching
    let escaped = query.replace('"', "\"\"")
                      .replace('*', "")
                      .replace(':', "")
                      .replace('(', "")
                      .replace(')', "");
    
    // For single terms, use both exact and prefix matching
    if escaped.len() > 2 {
        format!("({} OR {}*)", escaped, escaped)
    } else {
        escaped
    }
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
        
        // Use FTS5 for full-text search with BM25 ranking
        let mut stmt = conn.prepare(
            "SELECT 
                chunks_fts.chunk_id,
                chunks_fts.document_id,
                chunks_fts.content,
                bm25(chunks_fts) as relevance_score,
                snippet(chunks_fts, 0, '<mark>', '</mark>', '...', 32) as highlighted_content
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts)
            LIMIT ?"
        ).map_db_err()?;
        
        let mut results = Vec::new();
        
        let rows = stmt.query_map(params![&fts_query, limit], |row| {
            let bm25_score: f64 = row.get("relevance_score")?;
            // Convert BM25 score (negative) to positive relevance score
            let relevance_score = (-bm25_score).max(0.0) as f32;
            
            Ok(SearchResult {
                chunk_id: row.get("chunk_id")?,
                document_id: Uuid::parse_str(&row.get::<_, String>("document_id")?).unwrap(),
                content: row.get("content")?,
                relevance_score,
                highlighted_content: Some(row.get("highlighted_content")?),
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
        
        let query = if let Some(model) = model_id {
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
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk_id: String,
    pub document_id: Uuid,
    pub content: String,
    pub relevance_score: f32,
    pub highlighted_content: Option<String>,
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
}