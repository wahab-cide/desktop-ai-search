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
                conn.execute_batch("
                    PRAGMA journal_mode = WAL;
                    PRAGMA synchronous = NORMAL;
                    PRAGMA cache_size = 64000;
                    PRAGMA temp_store = MEMORY;
                    PRAGMA mmap_size = 134217728;
                ")?;
                Ok(())
            });
        
        let pool = Pool::builder()
            .max_size(15)
            .build(manager)
            .map_err(DatabaseError::Pool)?;
        
        let db = Self { pool };
        
        // Run migrations
        let conn = db.get_connection()?;
        migrations::run_migrations(&conn)?;
        drop(conn);
        
        Ok(db)
    }
    
    pub fn get_connection(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        self.pool.get().map_err(DatabaseError::Pool).map_err(AppError::Database)
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