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
    
    #[error("Network error: {0}")]
    Network(#[from] NetworkError),
    
    #[error("Model error: {0}")]
    Model(#[from] ModelError),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("UUID parse error: {0}")]
    Uuid(#[from] uuid::Error),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("ML/Tensor error: {0}")]
    Tensor(#[from] candle_core::Error),
    
    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),
    
    #[error("System resource error: {0}")]
    SystemResource(#[from] SystemResourceError),
    
    #[error("Recovery error: {0}")]
    Recovery(String),
    
    #[error("Critical system error: {0}")]
    Critical(String),
    
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
    
    #[error("OCR initialization error: {0}")]
    OcrInitialization(String),
    
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
    
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
    
    #[error("No results found")]
    NoResults,
}

#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    
    #[error("Download failed: {0}")]
    Download(String),
    
    #[error("Network timeout: {0}")]
    Timeout(String),
    
    #[error("Connection failed: {0}")]
    Connection(String),
    
    #[error("DNS resolution failed: {0}")]
    DnsResolution(String),
    
    #[error("SSL/TLS error: {0}")]
    Ssl(String),
}

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Model not found: {0}")]
    NotFound(String),
    
    #[error("Model loading failed: {0}")]
    LoadingFailed(String),
    
    #[error("Model inference failed: {0}")]
    InferenceFailed(String),
    
    #[error("Model corrupted: {0}")]
    Corrupted(String),
    
    #[error("Model version mismatch: {0}")]
    VersionMismatch(String),
    
    #[error("Model download failed: {0}")]
    DownloadFailed(String),
    
    #[error("Insufficient VRAM: {0}")]
    InsufficientVram(String),
    
    #[error("Model initialization failed: {0}")]
    InitializationFailed(String),
}

#[derive(Error, Debug)]
pub enum SystemResourceError {
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
    
    #[error("Out of disk space: {0}")]
    OutOfDiskSpace(String),
    
    #[error("CPU overload: {0}")]
    CpuOverload(String),
    
    #[error("GPU error: {0}")]
    Gpu(String),
    
    #[error("System overload: {0}")]
    SystemOverload(String),
    
    #[error("Resource timeout: {0}")]
    ResourceTimeout(String),
    
    #[error("Resource unavailable: {0}")]
    ResourceUnavailable(String),
}

impl AppError {
    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            AppError::Database(db_err) => db_err.is_recoverable(),
            AppError::FileSystem(fs_err) => fs_err.is_recoverable(),
            AppError::Search(search_err) => search_err.is_recoverable(),
            AppError::Network(net_err) => net_err.is_recoverable(),
            AppError::Model(model_err) => model_err.is_recoverable(),
            AppError::SystemResource(sys_err) => sys_err.is_recoverable(),
            AppError::Io(_) => true,
            AppError::Configuration(_) => false,
            AppError::Critical(_) => false,
            _ => true,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            AppError::Critical(_) => ErrorSeverity::Critical,
            AppError::Database(DatabaseError::Corruption(_)) => ErrorSeverity::Critical,
            AppError::SystemResource(SystemResourceError::OutOfMemory(_)) => ErrorSeverity::High,
            AppError::SystemResource(SystemResourceError::OutOfDiskSpace(_)) => ErrorSeverity::High,
            AppError::Configuration(_) => ErrorSeverity::High,
            AppError::Model(ModelError::Corrupted(_)) => ErrorSeverity::Medium,
            AppError::Network(_) => ErrorSeverity::Medium,
            AppError::Search(SearchError::Timeout) => ErrorSeverity::Medium,
            _ => ErrorSeverity::Low,
        }
    }
    
    /// Get suggested recovery action
    pub fn recovery_action(&self) -> Option<RecoveryAction> {
        match self {
            AppError::Network(_) => Some(RecoveryAction::Retry),
            AppError::Model(ModelError::LoadingFailed(_)) => Some(RecoveryAction::ReloadModel),
            AppError::Database(DatabaseError::Pool(_)) => Some(RecoveryAction::RestartDatabase),
            AppError::SystemResource(SystemResourceError::OutOfMemory(_)) => Some(RecoveryAction::ClearCache),
            AppError::Search(SearchError::Timeout) => Some(RecoveryAction::SimplifyQuery),
            _ => None,
        }
    }
}

impl DatabaseError {
    pub fn is_recoverable(&self) -> bool {
        match self {
            DatabaseError::Corruption(_) => false,
            DatabaseError::Schema(_) => false,
            _ => true,
        }
    }
}

impl FileSystemError {
    pub fn is_recoverable(&self) -> bool {
        match self {
            FileSystemError::PermissionDenied(_) => false,
            FileSystemError::InvalidPath(_) => false,
            _ => true,
        }
    }
}

impl SearchError {
    pub fn is_recoverable(&self) -> bool {
        match self {
            SearchError::InvalidQuery(_) => false,
            SearchError::NoResults => false,
            _ => true,
        }
    }
}

impl NetworkError {
    pub fn is_recoverable(&self) -> bool {
        match self {
            NetworkError::DnsResolution(_) => false,
            NetworkError::Ssl(_) => false,
            _ => true,
        }
    }
}

impl ModelError {
    pub fn is_recoverable(&self) -> bool {
        match self {
            ModelError::Corrupted(_) => false,
            ModelError::VersionMismatch(_) => false,
            ModelError::InsufficientVram(_) => false,
            _ => true,
        }
    }
}

impl SystemResourceError {
    pub fn is_recoverable(&self) -> bool {
        match self {
            SystemResourceError::OutOfDiskSpace(_) => false,
            SystemResourceError::ResourceUnavailable(_) => false,
            _ => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryAction {
    Retry,
    ReloadModel,
    RestartDatabase,
    ClearCache,
    SimplifyQuery,
    RestartApplication,
    ResetConfiguration,
}

pub type Result<T> = std::result::Result<T, AppError>;