use crate::error::{AppError, DatabaseError, Result};
use rusqlite::Connection;

// Helper to convert rusqlite errors
trait SqliteResultExt<T> {
    fn map_db_err(self) -> Result<T>;
}

impl<T> SqliteResultExt<T> for rusqlite::Result<T> {
    fn map_db_err(self) -> Result<T> {
        self.map_err(|e| AppError::Database(DatabaseError::Sqlite(e)))
    }
}

pub struct Migration {
    pub version: u32,
    pub description: String,
    pub sql: String,
}

impl Migration {
    pub fn new(version: u32, description: &str, sql: &str) -> Self {
        Self {
            version,
            description: description.to_string(),
            sql: sql.to_string(),
        }
    }
}

pub fn get_migrations() -> Vec<Migration> {
    vec![
        Migration::new(
            1,
            "Initial schema",
            r#"
            -- Documents table
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL,
                file_type TEXT NOT NULL,
                creation_date INTEGER NOT NULL,
                modification_date INTEGER NOT NULL,
                last_indexed INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            -- FTS5 external content table
            CREATE VIRTUAL TABLE search_index USING fts5(
                content,
                content='documents',
                content_rowid='rowid'
            );

            -- Indexing status table
            CREATE TABLE indexing_status (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_files INTEGER NOT NULL DEFAULT 0,
                indexed_files INTEGER NOT NULL DEFAULT 0,
                last_update INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'idle',
                errors TEXT NOT NULL DEFAULT '{}',
                performance_metrics TEXT NOT NULL DEFAULT '{}'
            );

            -- Initialize indexing status
            INSERT INTO indexing_status (id) VALUES (1);
            "#,
        ),
        Migration::new(
            2,
            "Add vector embeddings support",
            r#"
            CREATE TABLE embeddings (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                model_version TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            "#,
        ),
        Migration::new(
            3,
            "Add document chunking",
            r#"
            CREATE TABLE document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            "#,
        ),
        Migration::new(
            4,
            "Add performance indices",
            r#"
            CREATE INDEX idx_documents_file_path ON documents(file_path);
            CREATE INDEX idx_documents_content_hash ON documents(content_hash);
            CREATE INDEX idx_documents_last_indexed ON documents(last_indexed);
            CREATE INDEX idx_documents_file_type ON documents(file_type);
            CREATE INDEX idx_documents_modification_date ON documents(modification_date);

            CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);
            CREATE INDEX idx_chunks_chunk_index ON document_chunks(document_id, chunk_index);

            CREATE INDEX idx_embeddings_document_id ON embeddings(document_id);
            CREATE INDEX idx_embeddings_model_version ON embeddings(model_version);
            "#,
        ),
        Migration::new(
            5,
            "Enhanced chunks and embeddings schema",
            r#"
            DROP TABLE IF EXISTS embeddings;

            ALTER TABLE document_chunks ADD COLUMN word_count INTEGER DEFAULT 0;
            ALTER TABLE document_chunks ADD COLUMN sentence_count INTEGER DEFAULT 0;
            ALTER TABLE document_chunks ADD COLUMN overlap_start BOOLEAN DEFAULT FALSE;
            ALTER TABLE document_chunks ADD COLUMN overlap_end BOOLEAN DEFAULT FALSE;

            CREATE TABLE embeddings (
                id TEXT PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                model_id TEXT NOT NULL DEFAULT 'all-minilm-l6-v2',
                embedding_dimensions INTEGER NOT NULL DEFAULT 384,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE INDEX idx_embeddings_chunk_id ON embeddings(chunk_id);
            CREATE INDEX idx_embeddings_document_id_new ON embeddings(document_id);
            CREATE INDEX idx_embeddings_model_id ON embeddings(model_id);
            CREATE INDEX idx_chunks_word_count ON document_chunks(word_count);

            CREATE TABLE semantic_search_cache (
                id TEXT PRIMARY KEY,
                query_hash TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE
            );

            CREATE INDEX idx_semantic_cache_query ON semantic_search_cache(query_hash);
            CREATE INDEX idx_semantic_cache_score ON semantic_search_cache(similarity_score DESC);
            CREATE INDEX idx_semantic_cache_expires ON semantic_search_cache(expires_at);

            CREATE VIEW chunks_with_embeddings AS
            SELECT 
                c.id as chunk_id,
                c.document_id,
                c.chunk_index,
                c.start_char,
                c.end_char,
                c.content,
                c.word_count,
                c.sentence_count,
                c.overlap_start,
                c.overlap_end,
                c.created_at,
                e.id as embedding_id,
                e.embedding,
                e.model_id,
                e.embedding_dimensions
            FROM document_chunks c
            LEFT JOIN embeddings e ON c.id = e.chunk_id;
            "#,
        ),
        Migration::new(
            6,
            "Add CLIP embeddings support",
            include_str!("../sql/006_add_clip_embeddings.sql"),
        ),
        Migration::new(
            7,
            "Add FTS5 full-text search for chunks",
            include_str!("../sql/007_add_fts5_chunks.sql"),
        ),
    ]
}

pub fn run_migrations(conn: &Connection) -> Result<()> {
    // Create migrations table if it doesn't exist
    conn.execute(
        "CREATE TABLE IF NOT EXISTS migrations (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at INTEGER NOT NULL
        )",
        [],
    ).map_db_err()?;
    
    // Get current migration version
    let current_version: u32 = conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM migrations",
        [],
        |row| row.get(0)
    ).unwrap_or(0);
    
    let migrations = get_migrations();
    
    for migration in migrations {
        if migration.version > current_version {
            println!("Running migration {}: {}", migration.version, migration.description);
            
            // Execute migration in transaction
            let tx = conn.unchecked_transaction().map_db_err()?;
            
            // Execute the migration SQL
            tx.execute_batch(&migration.sql)
                .map_err(|e| DatabaseError::Migration(format!("Migration {} failed: {}", migration.version, e)))?;
            
            // Record migration as applied
            tx.execute(
                "INSERT INTO migrations (version, description, applied_at) VALUES (?, ?, ?)",
                rusqlite::params![
                    migration.version,
                    migration.description,
                    chrono::Utc::now().timestamp()
                ]
            ).map_db_err()?;
            
            tx.commit().map_db_err()?;
            
            println!("Migration {} completed successfully", migration.version);
        }
    }
    
    Ok(())
}

pub fn rollback_migration(conn: &Connection, target_version: u32) -> Result<()> {
    // This is a simplified rollback - in production you'd want proper down migrations
    let current_version: u32 = conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM migrations",
        [],
        |row| row.get(0)
    ).unwrap_or(0);
    
    if target_version >= current_version {
        return Err(AppError::Database(DatabaseError::Migration(
            format!("Cannot rollback to version {} from {}", target_version, current_version)
        )));
    }
    
    // Remove migration records
    conn.execute(
        "DELETE FROM migrations WHERE version > ?",
        rusqlite::params![target_version]
    ).map_db_err()?;
    
    println!("Rolled back to migration version {}", target_version);
    
    Ok(())
}