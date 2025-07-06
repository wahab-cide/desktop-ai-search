use crate::error::{AppError, DatabaseError, Result};
use crate::models::{Document, IndexMetadata, PerformanceMetrics, IndexingStatus};
use rusqlite::{Connection, params, Row};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use uuid::Uuid;
use std::path::Path;

// Helper to convert rusqlite errors
trait SqliteResultExt<T> {
    fn map_db_err(self) -> Result<T>;
}

impl<T> SqliteResultExt<T> for rusqlite::Result<T> {
    fn map_db_err(self) -> Result<T> {
        self.map_err(|e| AppError::Database(DatabaseError::Sqlite(e)))
    }
}

pub mod migrations;
pub mod operations;

pub type DbPool = Pool<SqliteConnectionManager>;

pub struct Database {
    pool: DbPool,
}

impl Database {
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let manager = SqliteConnectionManager::file(db_path)
            .with_init(|conn| {
                // Enable WAL mode for better performance
                conn.execute("PRAGMA journal_mode = WAL", [])?;
                conn.execute("PRAGMA synchronous = NORMAL", [])?;
                conn.execute("PRAGMA cache_size = 64000", [])?;
                conn.execute("PRAGMA temp_store = MEMORY", [])?;
                conn.execute("PRAGMA mmap_size = 134217728", [])?; // 128MB
                Ok(())
            });
        
        let pool = Pool::builder()
            .max_size(15)
            .build(manager)
            .map_err(DatabaseError::Pool)?;
        
        let db = Self { pool };
        
        // Initialize schema
        db.initialize_schema()?;
        
        Ok(db)
    }
    
    pub fn get_connection(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        self.pool.get().map_err(DatabaseError::Pool).map_err(AppError::Database)
    }
    
    fn initialize_schema(&self) -> Result<()> {
        let conn = self.get_connection()?;
        
        // Create documents table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL,
                file_type TEXT NOT NULL,
                creation_date INTEGER NOT NULL,
                modification_date INTEGER NOT NULL,
                last_indexed INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )",
            [],
        ).map_err(DatabaseError::Sqlite)?;
        
        // Create FTS5 external content table
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
                content,
                content='documents',
                content_rowid='rowid'
            )",
            [],
        ).map_err(DatabaseError::Sqlite)?;
        
        // Create indexing status table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS indexing_status (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_files INTEGER NOT NULL DEFAULT 0,
                indexed_files INTEGER NOT NULL DEFAULT 0,
                last_update INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'idle',
                errors TEXT NOT NULL DEFAULT '{}',
                performance_metrics TEXT NOT NULL DEFAULT '{}'
            )",
            [],
        ).map_err(DatabaseError::Sqlite)?;
        
        // Create vector embeddings table (for future use)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                model_version TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )",
            [],
        ).map_err(DatabaseError::Sqlite)?;
        
        // Create document chunks table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )",
            [],
        ).map_err(DatabaseError::Sqlite)?;
        
        // Create indices for performance
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents(file_path)",
            [],
        ).map_err(DatabaseError::Sqlite)?;
        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)",
            [],
        ).map_err(DatabaseError::Sqlite)?;
        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_last_indexed ON documents(last_indexed)",
            [],
        ).map_err(DatabaseError::Sqlite)?;
        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id)",
            [],
        ).map_err(DatabaseError::Sqlite)?;
        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON embeddings(document_id)",
            [],
        ).map_err(DatabaseError::Sqlite)?;
        
        // Initialize indexing status if not exists
        conn.execute(
            "INSERT OR IGNORE INTO indexing_status (id) VALUES (1)",
            [],
        ).map_err(DatabaseError::Sqlite)?;
        
        Ok(())
    }
    
    pub fn health_check(&self) -> Result<bool> {
        let conn = self.get_connection()?;
        
        // Check database integrity
        let mut stmt = conn.prepare("PRAGMA integrity_check").map_db_err()?;
        let integrity_result: String = stmt.query_row([], |row| row.get(0)).map_db_err()?;
        
        if integrity_result != "ok" {
            return Err(AppError::Database(DatabaseError::Corruption(integrity_result)));
        }
        
        // Check FTS5 index integrity
        let mut stmt = conn.prepare("INSERT INTO search_index(search_index) VALUES('integrity-check')").map_db_err()?;
        stmt.execute([]).map_db_err()?;
        
        Ok(true)
    }
    
    pub fn optimize(&self) -> Result<()> {
        let conn = self.get_connection()?;
        
        // Optimize database
        conn.execute("PRAGMA optimize", []).map_db_err()?;
        
        // Rebuild FTS index if needed
        conn.execute("INSERT INTO search_index(search_index) VALUES('rebuild')", []).map_db_err()?;
        
        Ok(())
    }
    
    pub fn backup<P: AsRef<Path>>(&self, backup_path: P) -> Result<()> {
        let conn = self.get_connection()?;
        let backup_path = backup_path.as_ref().to_string_lossy();
        
        conn.execute(
            &format!("VACUUM INTO '{}'", backup_path),
            [],
        ).map_db_err()?;
        
        Ok(())
    }
}

impl Document {
    pub fn from_row(row: &Row) -> rusqlite::Result<Self> {
        let metadata_json: String = row.get("metadata")?;
        let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)
            .unwrap_or_default();
        
        Ok(Document {
            id: Uuid::parse_str(&row.get::<_, String>("id")?)
                .map_err(|_e| rusqlite::Error::InvalidColumnType(0, "id".to_string(), rusqlite::types::Type::Text))?,
            file_path: row.get("file_path")?,
            content_hash: row.get("content_hash")?,
            file_type: serde_json::from_str(&row.get::<_, String>("file_type")?)
                .unwrap_or(crate::models::FileType::Unknown),
            creation_date: DateTime::from_timestamp(row.get::<_, i64>("creation_date")?, 0)
                .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap()),
            modification_date: DateTime::from_timestamp(row.get::<_, i64>("modification_date")?, 0)
                .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap()),
            last_indexed: DateTime::from_timestamp(row.get::<_, i64>("last_indexed")?, 0)
                .unwrap_or_else(|| DateTime::from_timestamp(0, 0).unwrap()),
            file_size: row.get::<_, u64>("file_size")?,
            metadata,
        })
    }
}