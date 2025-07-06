use crate::error::{AppError, DatabaseError, Result};
use crate::models::{Document, FileType, IndexMetadata, IndexingStatus, PerformanceMetrics};
use crate::database::Database;
use rusqlite::{params, Transaction};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::collections::HashMap;

// Helper to convert rusqlite errors
trait SqliteResultExt<T> {
    fn map_db_err(self) -> Result<T>;
}

impl<T> SqliteResultExt<T> for rusqlite::Result<T> {
    fn map_db_err(self) -> Result<T> {
        self.map_err(|e| AppError::Database(DatabaseError::Sqlite(e)))
    }
}

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
        )?;
        
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
        ])?;
        
        Ok(())
    }
    
    pub fn get_document_by_id(&self, id: &Uuid) -> Result<Option<Document>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT * FROM documents WHERE id = ?"
        )?;
        
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
        )?;
        
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
        )?;
        
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
        let tx = conn.unchecked_transaction()?;
        
        // Delete from FTS index first
        tx.execute(
            "DELETE FROM search_index WHERE rowid IN (
                SELECT rowid FROM documents WHERE id = ?
            )",
            params![id.to_string()]
        )?;
        
        // Delete document chunks
        tx.execute(
            "DELETE FROM document_chunks WHERE document_id = ?",
            params![id.to_string()]
        )?;
        
        // Delete embeddings
        tx.execute(
            "DELETE FROM embeddings WHERE document_id = ?",
            params![id.to_string()]
        )?;
        
        // Delete document
        tx.execute(
            "DELETE FROM documents WHERE id = ?",
            params![id.to_string()]
        )?;
        
        tx.commit()?;
        Ok(())
    }
    
    pub fn insert_document_content(&self, document_id: &Uuid, content: &str) -> Result<()> {
        let conn = self.get_connection()?;
        
        // Insert into FTS index
        conn.execute(
            "INSERT INTO search_index(rowid, content) 
            SELECT rowid, ? FROM documents WHERE id = ?",
            params![content, document_id.to_string()]
        )?;
        
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
        )?;
        
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
        )?;
        
        let rows = stmt.query_map(params![query, limit], |row| {
            Document::from_row(row)
        })?;
        
        let mut documents = Vec::new();
        for row in rows {
            documents.push(row?);
        }
        
        Ok(documents)
    }
    
    pub fn get_indexing_status(&self) -> Result<IndexMetadata> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT * FROM indexing_status WHERE id = 1"
        )?;
        
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
        })?;
        
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
        )?;
        
        Ok(())
    }
    
    pub fn get_document_count(&self) -> Result<usize> {
        let conn = self.get_connection()?;
        let count: usize = conn.query_row(
            "SELECT COUNT(*) FROM documents",
            [],
            |row| row.get(0)
        )?;
        
        Ok(count)
    }
    
    pub fn get_documents_modified_since(&self, since: DateTime<Utc>) -> Result<Vec<Document>> {
        let conn = self.get_connection()?;
        let mut stmt = conn.prepare(
            "SELECT * FROM documents WHERE modification_date > ? ORDER BY modification_date DESC"
        )?;
        
        let rows = stmt.query_map(params![since.timestamp()], |row| {
            Document::from_row(row)
        })?;
        
        let mut documents = Vec::new();
        for row in rows {
            documents.push(row?);
        }
        
        Ok(documents)
    }
    
    pub fn batch_insert_documents(&self, documents: &[Document]) -> Result<()> {
        let conn = self.get_connection()?;
        let tx = conn.unchecked_transaction()?;
        
        {
            let mut stmt = tx.prepare(
                "INSERT INTO documents (
                    id, file_path, content_hash, file_type, creation_date, 
                    modification_date, last_indexed, file_size, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )?;
            
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
                ])?;
            }
        }
        
        tx.commit()?;
        Ok(())
    }
}