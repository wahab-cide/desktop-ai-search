use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),
    
    #[error("File system error: {0}")]
    FileSystem(#[from] FileSystemError),
    
    #[error("Indexing error: {0}")]
    Indexing(#[from] IndexingError),
    
    #[error("Search error: {0}")]
    Search(#[from] SearchError),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("UUID parse error: {0}")]
    Uuid(#[from] uuid::Error),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    
    #[error("Connection pool error: {0}")]
    Pool(#[from] r2d2::Error),
    
    #[error("Migration error: {0}")]
    Migration(String),
    
    #[error("Schema validation error: {0}")]
    Schema(String),
    
    #[error("Database corruption detected: {0}")]
    Corruption(String),
    
    #[error("Transaction error: {0}")]
    Transaction(String),
}

#[derive(Error, Debug)]
pub enum FileSystemError {
    #[error("File not found: {0}")]
    NotFound(String),
    
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    
    #[error("File watcher error: {0}")]
    Watcher(#[from] notify::Error),
    
    #[error("Hash computation error: {0}")]
    Hash(String),
    
    #[error("File type detection error: {0}")]
    FileType(String),
    
    #[error("Metadata extraction error: {0}")]
    Metadata(String),
}

#[derive(Error, Debug)]
pub enum IndexingError {
    #[error("Document processing error: {0}")]
    Processing(String),
    
    #[error("Content extraction error: {0}")]
    Extraction(String),
    
    #[error("OCR error: {0}")]
    Ocr(String),
    
    #[error("Audio transcription error: {0}")]
    Transcription(String),
    
    #[error("Embedding generation error: {0}")]
    Embedding(String),
    
    #[error("Chunking error: {0}")]
    Chunking(String),
    
    #[error("Index update error: {0}")]
    IndexUpdate(String),
}

#[derive(Error, Debug)]
pub enum SearchError {
    #[error("Query parsing error: {0}")]
    QueryParsing(String),
    
    #[error("Vector search error: {0}")]
    VectorSearch(String),
    
    #[error("Full-text search error: {0}")]
    FullTextSearch(String),
    
    #[error("Result ranking error: {0}")]
    Ranking(String),
    
    #[error("Result fusion error: {0}")]
    Fusion(String),
    
    #[error("Timeout error: query took too long")]
    Timeout,
    
    #[error("No results found")]
    NoResults,
}

pub type Result<T> = std::result::Result<T, AppError>;