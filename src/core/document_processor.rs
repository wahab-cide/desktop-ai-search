use crate::error::{AppError, IndexingError, Result};
use crate::models::{Document, FileType};
use crate::core::{
    text_extractor::{TextExtractor, ExtractionResult, ExtractionMode},
    chunker::{TextChunker, ChunkingOptions, TextChunk},
    embedding_manager::{EmbeddingManager, EmbeddingConfig},
};
use crate::database::Database;
use std::path::Path;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Semaphore, mpsc};
use uuid::Uuid;
use chrono::Utc;

#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub document: Document,
    pub chunks: Vec<TextChunk>,
    pub extraction_result: ExtractionResult,
    pub processing_time_ms: u64,
    pub requires_ocr: bool,
    pub requires_transcription: bool,
}

#[derive(Debug, Clone)]
pub struct ProcessingOptions {
    pub chunking_options: ChunkingOptions,
    pub max_concurrent_documents: usize,
    pub skip_empty_documents: bool,
    pub min_content_length: usize,
    pub extract_metadata: bool,
    pub preserve_original_structure: bool,
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self {
            chunking_options: ChunkingOptions::default(),
            max_concurrent_documents: 4,
            skip_empty_documents: true,
            min_content_length: 50,
            extract_metadata: true,
            preserve_original_structure: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProcessingProgress {
    pub total_documents: usize,
    pub processed_documents: usize,
    pub current_document: Option<String>,
    pub chunks_created: usize,
    pub documents_requiring_ocr: usize,
    pub documents_requiring_transcription: usize,
    pub processing_errors: usize,
    pub elapsed_time_ms: u64,
}

pub struct DocumentProcessor {
    text_extractor: TextExtractor,
    chunker: TextChunker,
    embedding_manager: Option<Arc<tokio::sync::Mutex<EmbeddingManager>>>,
    options: ProcessingOptions,
    processing_semaphore: Arc<Semaphore>,
}

impl DocumentProcessor {
    pub fn new(options: ProcessingOptions) -> Self {
        let semaphore = Arc::new(Semaphore::new(options.max_concurrent_documents));
        
        Self {
            text_extractor: TextExtractor::new(),
            chunker: TextChunker::new(options.chunking_options.clone()),
            embedding_manager: None,
            options,
            processing_semaphore: semaphore,
        }
    }
    
    /// Set the embedding manager for generating semantic embeddings
    pub async fn set_embedding_manager(&mut self, model_id: &str) -> Result<()> {
        let mut embedding_manager = EmbeddingManager::new()?;
        
        // Download model if not already downloaded
        if !embedding_manager.is_model_downloaded(model_id).await? {
            println!("Downloading embedding model: {}", model_id);
            embedding_manager.download_model(model_id).await?;
        }
        
        // Load the model
        embedding_manager.load_model(model_id, Some(EmbeddingConfig::default())).await?;
        
        self.embedding_manager = Some(Arc::new(tokio::sync::Mutex::new(embedding_manager)));
        Ok(())
    }
    
    /// Check if embedding generation is enabled
    pub fn has_embedding_manager(&self) -> bool {
        self.embedding_manager.is_some()
    }
    
    pub async fn process_document<P: AsRef<Path>>(
        &self,
        document: &Document,
        path: P,
    ) -> Result<ProcessingResult> {
        let _permit = self.processing_semaphore.acquire().await
            .map_err(|e| AppError::Indexing(IndexingError::Processing(e.to_string())))?;
        
        let start_time = std::time::Instant::now();
        
        // Extract text from document
        let extraction_result = self.text_extractor
            .extract_text(path, &document.file_type)
            .await?;
        
        let mut requires_ocr = false;
        let mut requires_transcription = false;
        
        // Check if special processing is needed
        match extraction_result.extraction_mode {
            ExtractionMode::OcrRequired => {
                requires_ocr = true;
            }
            ExtractionMode::TranscriptionRequired => {
                requires_transcription = true;
            }
            _ => {}
        }
        
        // Skip if content is too short and we're configured to skip empty documents
        if self.options.skip_empty_documents && 
           extraction_result.content.len() < self.options.min_content_length {
            if !requires_ocr && !requires_transcription {
                return Err(AppError::Indexing(IndexingError::Processing(
                    "Document content too short and no special processing required".to_string()
                )));
            }
        }
        
        // Create chunks from extracted text
        let mut chunks = if !extraction_result.content.is_empty() {
            self.chunker.chunk_text(&extraction_result.content)?
        } else {
            Vec::new()
        };
        
        // Generate embeddings for chunks if embedding manager is available
        if let Some(embedding_manager) = &self.embedding_manager {
            if !chunks.is_empty() {
                chunks = self.generate_chunk_embeddings(chunks, embedding_manager.clone()).await?;
            }
        }
        
        // Validate chunks
        self.chunker.validate_chunks(&chunks)?;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        Ok(ProcessingResult {
            document: document.clone(),
            chunks,
            extraction_result,
            processing_time_ms: processing_time,
            requires_ocr,
            requires_transcription,
        })
    }
    
    pub async fn process_documents(
        &self,
        documents: Vec<Document>,
        progress_sender: Option<mpsc::UnboundedSender<ProcessingProgress>>,
    ) -> Result<Vec<ProcessingResult>> {
        let total_documents = documents.len();
        let mut results = Vec::new();
        let mut processed = 0;
        let mut chunks_created = 0;
        let mut documents_requiring_ocr = 0;
        let mut documents_requiring_transcription = 0;
        let mut processing_errors = 0;
        let start_time = std::time::Instant::now();
        
        for document in documents {
            // Send progress update
            if let Some(ref sender) = progress_sender {
                let progress = ProcessingProgress {
                    total_documents,
                    processed_documents: processed,
                    current_document: Some(document.file_path.clone()),
                    chunks_created,
                    documents_requiring_ocr,
                    documents_requiring_transcription,
                    processing_errors,
                    elapsed_time_ms: start_time.elapsed().as_millis() as u64,
                };
                let _ = sender.send(progress);
            }
            
            match self.process_document(&document, &document.file_path).await {
                Ok(result) => {
                    chunks_created += result.chunks.len();
                    if result.requires_ocr {
                        documents_requiring_ocr += 1;
                    }
                    if result.requires_transcription {
                        documents_requiring_transcription += 1;
                    }
                    results.push(result);
                }
                Err(e) => {
                    eprintln!("Error processing document {}: {}", document.file_path, e);
                    processing_errors += 1;
                }
            }
            
            processed += 1;
        }
        
        // Send final progress update
        if let Some(ref sender) = progress_sender {
            let final_progress = ProcessingProgress {
                total_documents,
                processed_documents: processed,
                current_document: None,
                chunks_created,
                documents_requiring_ocr,
                documents_requiring_transcription,
                processing_errors,
                elapsed_time_ms: start_time.elapsed().as_millis() as u64,
            };
            let _ = sender.send(final_progress);
        }
        
        Ok(results)
    }
    
    pub async fn store_processing_results(
        &self,
        database: &Database,
        results: Vec<ProcessingResult>,
    ) -> Result<usize> {
        let mut stored_chunks = 0;
        
        for result in results {
            // First insert the document
            database.insert_document(&result.document)?;
            
            // Store document content in FTS index
            if !result.extraction_result.content.is_empty() {
                database.insert_document_content(&result.document.id, &result.extraction_result.content)?;
            }
            
            // Store individual chunks with embeddings
            for chunk in &result.chunks {
                let embedding_data = chunk.embedding.as_ref().map(|e| e.as_slice());
                
                database.insert_chunk_with_embedding(
                    &chunk.id,
                    &result.document.id,
                    chunk.chunk_index,
                    chunk.start_char,
                    chunk.end_char,
                    &chunk.content,
                    chunk.word_count,
                    chunk.sentence_count,
                    chunk.overlap_start,
                    chunk.overlap_end,
                    embedding_data,
                    Some("all-minilm-l6-v2"),
                )?;
                stored_chunks += 1;
            }
        }
        
        Ok(stored_chunks)
    }
    
    pub async fn reprocess_for_ocr<P: AsRef<Path>>(
        &self,
        document: &Document,
        path: P,
        ocr_text: String,
    ) -> Result<ProcessingResult> {
        let start_time = std::time::Instant::now();
        
        // Create extraction result from OCR text
        let extraction_result = ExtractionResult {
            content: ocr_text.clone(),
            extraction_mode: ExtractionMode::PdfOcr,
            page_count: None,
            word_count: ocr_text.split_whitespace().count(),
            character_count: ocr_text.len(),
            language: self.text_extractor.detect_language(&ocr_text),
            metadata: HashMap::new(),
        };
        
        // Create chunks from OCR text
        let chunks = if !ocr_text.is_empty() {
            self.chunker.chunk_text(&ocr_text)?
        } else {
            Vec::new()
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        Ok(ProcessingResult {
            document: document.clone(),
            chunks,
            extraction_result,
            processing_time_ms: processing_time,
            requires_ocr: false,
            requires_transcription: false,
        })
    }
    
    pub async fn reprocess_for_transcription<P: AsRef<Path>>(
        &self,
        document: &Document,
        path: P,
        transcript_text: String,
        timestamps: Vec<(u64, u64, String)>, // (start_ms, end_ms, text)
    ) -> Result<ProcessingResult> {
        let start_time = std::time::Instant::now();
        
        // Create extraction result from transcript
        let mut metadata = HashMap::new();
        metadata.insert("has_timestamps".to_string(), "true".to_string());
        metadata.insert("segment_count".to_string(), timestamps.len().to_string());
        
        let extraction_result = ExtractionResult {
            content: transcript_text.clone(),
            extraction_mode: ExtractionMode::TranscriptionRequired,
            page_count: None,
            word_count: transcript_text.split_whitespace().count(),
            character_count: transcript_text.len(),
            language: self.text_extractor.detect_language(&transcript_text),
            metadata,
        };
        
        // Create chunks from transcript - potentially with timestamp boundaries
        let chunks = if !transcript_text.is_empty() {
            self.create_timestamp_aware_chunks(&transcript_text, &timestamps)?
        } else {
            Vec::new()
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        Ok(ProcessingResult {
            document: document.clone(),
            chunks,
            extraction_result,
            processing_time_ms: processing_time,
            requires_ocr: false,
            requires_transcription: false,
        })
    }
    
    fn create_timestamp_aware_chunks(
        &self,
        text: &str,
        timestamps: &[(u64, u64, String)],
    ) -> Result<Vec<TextChunk>> {
        if timestamps.is_empty() {
            return self.chunker.chunk_text(text);
        }
        
        let mut chunks = Vec::new();
        let mut current_text = String::new();
        let mut current_start = 0;
        let mut chunk_index = 0;
        
        for (start_ms, end_ms, segment_text) in timestamps {
            // Add segment to current chunk
            if current_text.is_empty() {
                current_text = segment_text.clone();
                current_start = *start_ms as usize;
            } else {
                current_text.push(' ');
                current_text.push_str(segment_text);
            }
            
            // Check if we should create a chunk
            if current_text.len() >= self.options.chunking_options.target_chunk_size {
                let chunk = TextChunk {
                    id: format!("chunk_{}", chunk_index),
                    content: current_text.clone(),
                    start_char: current_start,
                    end_char: *end_ms as usize,
                    chunk_index,
                    word_count: current_text.split_whitespace().count(),
                    sentence_count: 1, // Approximate for audio segments
                    overlap_start: chunk_index > 0,
                    overlap_end: true,
                    embedding: None, // Will be populated later
                    document_id: None, start_position: None, end_position: None,
                    total_chunks: None, metadata: None,
                };
                
                chunks.push(chunk);
                chunk_index += 1;
                
                // Start new chunk with some overlap
                let overlap_words: Vec<&str> = current_text
                    .split_whitespace()
                    .rev()
                    .take(5) // Take last 5 words as overlap
                    .collect();
                current_text = overlap_words.into_iter().rev().collect::<Vec<_>>().join(" ");
                current_start = *start_ms as usize;
            }
        }
        
        // Add final chunk if there's remaining content
        if !current_text.is_empty() {
            let chunk = TextChunk {
                id: format!("chunk_{}", chunk_index),
                content: current_text.clone(),
                start_char: current_start,
                end_char: current_start + current_text.len(),
                chunk_index,
                word_count: current_text.split_whitespace().count(),
                sentence_count: 1,
                overlap_start: chunk_index > 0,
                overlap_end: false,
                embedding: None, // Will be populated later
                document_id: None, start_position: None, end_position: None,
                total_chunks: None, metadata: None,
            };
            chunks.push(chunk);
        }
        
        Ok(chunks)
    }
    
    pub fn get_processing_statistics(&self, results: &[ProcessingResult]) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        stats.insert("total_documents".to_string(), serde_json::Value::Number(results.len().into()));
        
        let total_chunks: usize = results.iter().map(|r| r.chunks.len()).sum();
        stats.insert("total_chunks".to_string(), serde_json::Value::Number(total_chunks.into()));
        
        let total_words: usize = results.iter()
            .map(|r| r.extraction_result.word_count)
            .sum();
        stats.insert("total_words".to_string(), serde_json::Value::Number(total_words.into()));
        
        let total_characters: usize = results.iter()
            .map(|r| r.extraction_result.character_count)
            .sum();
        stats.insert("total_characters".to_string(), serde_json::Value::Number(total_characters.into()));
        
        let avg_processing_time: f64 = results.iter()
            .map(|r| r.processing_time_ms as f64)
            .sum::<f64>() / results.len() as f64;
        stats.insert("avg_processing_time_ms".to_string(), serde_json::Value::Number(
            serde_json::Number::from_f64(avg_processing_time).unwrap_or(serde_json::Number::from(0))
        ));
        
        let ocr_required = results.iter().filter(|r| r.requires_ocr).count();
        stats.insert("documents_requiring_ocr".to_string(), serde_json::Value::Number(ocr_required.into()));
        
        let transcription_required = results.iter().filter(|r| r.requires_transcription).count();
        stats.insert("documents_requiring_transcription".to_string(), serde_json::Value::Number(transcription_required.into()));
        
        // Group by file type
        let mut file_type_counts = HashMap::new();
        for result in results {
            let file_type = format!("{:?}", result.document.file_type);
            *file_type_counts.entry(file_type).or_insert(0) += 1;
        }
        stats.insert("file_type_distribution".to_string(), serde_json::Value::Object(
            file_type_counts.into_iter()
                .map(|(k, v)| (k, serde_json::Value::Number(v.into())))
                .collect()
        ));
        
        stats
    }
    
    /// Generate embeddings for text chunks
    async fn generate_chunk_embeddings(
        &self,
        mut chunks: Vec<TextChunk>,
        embedding_manager: Arc<tokio::sync::Mutex<EmbeddingManager>>,
    ) -> Result<Vec<TextChunk>> {
        if chunks.is_empty() {
            return Ok(chunks);
        }
        
        // Extract text content from all chunks
        let chunk_texts: Vec<String> = chunks.iter()
            .map(|chunk| chunk.content.clone())
            .collect();
        
        // Generate embeddings for all chunks at once (batch processing)
        let manager = embedding_manager.lock().await;
        let embeddings = manager.generate_embeddings(&chunk_texts).await
            .map_err(|e| AppError::Indexing(IndexingError::Embedding(
                format!("Failed to generate embeddings: {}", e)
            )))?;
        drop(manager); // Release the lock early
        
        // Assign embeddings to chunks
        for (chunk, embedding) in chunks.iter_mut().zip(embeddings.into_iter()) {
            chunk.embedding = Some(embedding);
        }
        
        Ok(chunks)
    }
}

impl Default for DocumentProcessor {
    fn default() -> Self {
        Self::new(ProcessingOptions::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;
    
    #[tokio::test]
    async fn test_document_processing() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = "This is a test document with multiple sentences. It should be processed correctly and split into appropriate chunks. The processor should handle this text efficiently.";
        writeln!(temp_file, "{}", content).unwrap();
        temp_file.flush().unwrap();
        
        let document = Document {
            id: Uuid::new_v4(),
            file_path: temp_file.path().to_string_lossy().to_string(),
            content_hash: "test_hash".to_string(),
            file_type: FileType::Text,
            creation_date: Utc::now(),
            modification_date: Utc::now(),
            last_indexed: Utc::now(),
            file_size: content.len() as u64,
            metadata: HashMap::new(),
        };
        
        let processor = DocumentProcessor::default();
        let result = processor.process_document(&document, temp_file.path()).await.unwrap();
        
        assert!(!result.chunks.is_empty());
        assert!(!result.extraction_result.content.is_empty());
        assert!(result.processing_time_ms > 0);
        assert!(!result.requires_ocr);
        assert!(!result.requires_transcription);
    }
    
    #[tokio::test]
    async fn test_batch_processing() {
        let documents = vec![
            Document {
                id: Uuid::new_v4(),
                file_path: "/test/doc1.txt".to_string(),
                content_hash: "hash1".to_string(),
                file_type: FileType::Text,
                creation_date: Utc::now(),
                modification_date: Utc::now(),
                last_indexed: Utc::now(),
                file_size: 100,
                metadata: HashMap::new(),
            },
            Document {
                id: Uuid::new_v4(),
                file_path: "/test/doc2.txt".to_string(),
                content_hash: "hash2".to_string(),
                file_type: FileType::Markdown,
                creation_date: Utc::now(),
                modification_date: Utc::now(),
                last_indexed: Utc::now(),
                file_size: 200,
                metadata: HashMap::new(),
            },
        ];
        
        let processor = DocumentProcessor::default();
        
        // Note: This will fail because the files don't exist, but we can test the structure
        let result = processor.process_documents(documents, None).await;
        assert!(result.is_ok()); // Should return Ok even with processing errors
    }
    
    #[test]
    fn test_processing_statistics() {
        let processor = DocumentProcessor::default();
        let results = vec![
            ProcessingResult {
                document: Document {
                    id: Uuid::new_v4(),
                    file_path: "/test/doc1.txt".to_string(),
                    content_hash: "hash1".to_string(),
                    file_type: FileType::Text,
                    creation_date: Utc::now(),
                    modification_date: Utc::now(),
                    last_indexed: Utc::now(),
                    file_size: 100,
                    metadata: HashMap::new(),
                },
                chunks: vec![],
                extraction_result: ExtractionResult {
                    content: "Test content".to_string(),
                    extraction_mode: ExtractionMode::Text,
                    page_count: None,
                    word_count: 2,
                    character_count: 12,
                    language: Some("en".to_string()),
                    metadata: HashMap::new(),
                },
                processing_time_ms: 100,
                requires_ocr: false,
                requires_transcription: false,
            },
        ];
        
        let stats = processor.get_processing_statistics(&results);
        
        assert_eq!(stats.get("total_documents"), Some(&serde_json::Value::Number(1.into())));
        assert_eq!(stats.get("total_words"), Some(&serde_json::Value::Number(2.into())));
        assert_eq!(stats.get("total_characters"), Some(&serde_json::Value::Number(12.into())));
    }
}