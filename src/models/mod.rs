use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: Uuid,
    pub file_path: String,
    pub content_hash: String, // SHA256
    pub file_type: FileType,
    pub creation_date: DateTime<Utc>,
    pub modification_date: DateTime<Utc>,
    pub last_indexed: DateTime<Utc>,
    pub file_size: u64,
    pub metadata: HashMap<String, String>, // JSON-serializable metadata
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FileType {
    Text,
    Pdf,
    Docx,
    Image,
    Audio,
    Video,
    Email,
    Markdown,
    Html,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub document_id: Uuid,
    pub relevance_score: f64,
    pub matched_content: Vec<ContentSnippet>,
    pub result_type: ResultType,
    pub ranking_factors: RankingFactors,
    
    // UI display fields
    pub id: String,
    pub content: String,
    pub file_path: String,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSnippet {
    pub text: String,
    pub start_position: usize,
    pub end_position: usize,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResultType {
    Exact,
    Semantic,
    Fuzzy,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingFactors {
    pub text_score: f64,
    pub semantic_score: f64,
    pub recency_boost: f64,
    pub access_frequency: f64,
    pub file_importance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub indexing_status: IndexingStatus,
    pub total_documents: usize,
    pub last_full_index: DateTime<Utc>,
    pub error_counts: HashMap<String, usize>,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexingStatus {
    Idle,
    Running,
    Paused,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub documents_per_second: f64,
    pub bytes_per_second: f64,
    pub average_processing_time: f64,
    pub memory_usage_mb: f64,
}

// LLM-related models
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub model_path: PathBuf,
    pub context_size: usize,
    pub n_gpu_layers: i32,
    pub n_threads: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repeat_penalty: f32,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub quant_type: String,
    pub context_size: usize,
    pub parameter_count: Option<String>,
    pub license: Option<String>,
    pub description: Option<String>,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub config: Option<LlmConfig>,
    pub system_prompt: Option<String>,
    pub stop_tokens: Vec<String>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub tokens_per_second: f32,
    pub total_time_ms: u64,
    pub finished: bool,
    pub stop_reason: Option<String>,
}