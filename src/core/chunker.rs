use crate::error::{AppError, IndexingError, Result};
use regex::Regex;
use std::collections::VecDeque;
use uuid;

#[derive(Debug, Clone)]
pub struct TextChunk {
    pub id: String,
    pub content: String,
    pub start_char: usize,
    pub end_char: usize,
    pub chunk_index: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub overlap_start: bool,
    pub overlap_end: bool,
    pub embedding: Option<Vec<f32>>, // Semantic embedding vector
    
    // Database fields
    pub document_id: Option<String>,
    pub start_position: Option<usize>,
    pub end_position: Option<usize>,
    pub total_chunks: Option<usize>,
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Clone)]
pub struct ChunkingOptions {
    pub max_chunk_size: usize,      // Maximum characters per chunk
    pub target_chunk_size: usize,   // Target characters per chunk
    pub overlap_percentage: f32,    // Percentage overlap between chunks
    pub min_chunk_size: usize,      // Minimum characters per chunk
    pub sentence_boundary_split: bool, // Split on sentence boundaries
    pub preserve_paragraphs: bool,  // Try to keep paragraphs together
}

impl Default for ChunkingOptions {
    fn default() -> Self {
        Self {
            max_chunk_size: 1500,
            target_chunk_size: 1000,
            overlap_percentage: 0.15, // 15% overlap
            min_chunk_size: 100,
            sentence_boundary_split: true,
            preserve_paragraphs: true,
        }
    }
}

pub struct TextChunker {
    options: ChunkingOptions,
    sentence_splitter: Regex,
    paragraph_splitter: Regex,
    whitespace_normalizer: Regex,
}

impl TextChunker {
    pub fn new(options: ChunkingOptions) -> Self {
        Self {
            options,
            sentence_splitter: Regex::new(r"[.!?]+\s+").unwrap(),
            paragraph_splitter: Regex::new(r"\n\s*\n").unwrap(),
            whitespace_normalizer: Regex::new(r"\s+").unwrap(),
        }
    }
    
    pub fn chunk_text(&self, text: &str) -> Result<Vec<TextChunk>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        
        // Normalize whitespace first
        let normalized_text = self.whitespace_normalizer.replace_all(text, " ");
        let text = normalized_text.trim();
        
        if text.len() <= self.options.max_chunk_size {
            // Text is small enough to be a single chunk
            return Ok(vec![self.create_chunk(text.to_string(), 0, text.len(), 0, false, false)]);
        }
        
        if self.options.sentence_boundary_split {
            self.chunk_by_sentences(text)
        } else {
            self.chunk_by_characters(text)
        }
    }
    
    fn chunk_by_sentences(&self, text: &str) -> Result<Vec<TextChunk>> {
        let sentences = self.split_into_sentences(text);
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_start = 0;
        let mut chunk_index = 0;
        let mut sentence_positions = Vec::new();
        
        // Track sentence positions for accurate character positions
        let mut pos = 0;
        for sentence in &sentences {
            sentence_positions.push(pos);
            pos += sentence.len();
        }
        
        for (i, sentence) in sentences.iter().enumerate() {
            let potential_chunk = if current_chunk.is_empty() {
                sentence.clone()
            } else {
                format!("{} {}", current_chunk, sentence)
            };
            
            if potential_chunk.len() <= self.options.max_chunk_size {
                // Can add this sentence to current chunk
                current_chunk = potential_chunk;
            } else {
                // Current chunk would be too large, finalize current chunk
                if !current_chunk.is_empty() {
                    let chunk_end = current_start + current_chunk.len();
                    let overlap_start = chunk_index > 0;
                    
                    chunks.push(self.create_chunk(
                        current_chunk.clone(),
                        current_start,
                        chunk_end,
                        chunk_index,
                        overlap_start,
                        false, // We'll set overlap_end later
                    ));
                    
                    chunk_index += 1;
                }
                
                // Start new chunk with overlap from previous chunk
                let overlap_text = self.calculate_overlap(&current_chunk);
                current_chunk = if overlap_text.is_empty() {
                    sentence.clone()
                } else {
                    format!("{} {}", overlap_text, sentence)
                };
                
                current_start = if overlap_text.is_empty() {
                    sentence_positions[i]
                } else {
                    // Calculate start position accounting for overlap
                    sentence_positions[i].saturating_sub(overlap_text.len())
                };
            }
        }
        
        // Add final chunk
        if !current_chunk.is_empty() {
            let chunk_end = current_start + current_chunk.len();
            let overlap_start = chunk_index > 0;
            
            chunks.push(self.create_chunk(
                current_chunk,
                current_start,
                chunk_end,
                chunk_index,
                overlap_start,
                false,
            ));
        }
        
        // Set overlap_end flags
        if chunks.len() > 1 {
            for i in 0..chunks.len() - 1 {
                chunks[i].overlap_end = true;
            }
        }
        
        Ok(chunks)
    }
    
    fn chunk_by_characters(&self, text: &str) -> Result<Vec<TextChunk>> {
        let mut chunks = Vec::new();
        let mut chunk_index = 0;
        let mut start = 0;
        
        while start < text.len() {
            let mut end = (start + self.options.target_chunk_size).min(text.len());
            
            // Try to find a good break point (space, punctuation)
            if end < text.len() {
                let break_search_start = (end - 100).max(start);
                if let Some(break_pos) = self.find_break_point(&text[break_search_start..end]) {
                    end = break_search_start + break_pos;
                }
            }
            
            let chunk_text = text[start..end].to_string();
            let overlap_start = chunk_index > 0;
            let overlap_end = end < text.len();
            
            chunks.push(self.create_chunk(
                chunk_text,
                start,
                end,
                chunk_index,
                overlap_start,
                overlap_end,
            ));
            
            // Calculate next start with overlap
            let overlap_size = ((end - start) as f32 * self.options.overlap_percentage) as usize;
            start = end.saturating_sub(overlap_size);
            chunk_index += 1;
        }
        
        Ok(chunks)
    }
    
    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut last_end = 0;
        
        for mat in self.sentence_splitter.find_iter(text) {
            let sentence = text[last_end..mat.end()].trim().to_string();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            last_end = mat.end();
        }
        
        // Add remaining text as final sentence
        if last_end < text.len() {
            let final_sentence = text[last_end..].trim().to_string();
            if !final_sentence.is_empty() {
                sentences.push(final_sentence);
            }
        }
        
        sentences
    }
    
    fn calculate_overlap(&self, chunk: &str) -> String {
        let overlap_chars = (chunk.len() as f32 * self.options.overlap_percentage) as usize;
        if overlap_chars == 0 || chunk.len() <= overlap_chars {
            return String::new();
        }
        
        let start_pos = chunk.len() - overlap_chars;
        
        // Try to start overlap at a word boundary
        if let Some(word_start) = chunk[start_pos..].find(' ') {
            chunk[start_pos + word_start..].trim().to_string()
        } else {
            chunk[start_pos..].to_string()
        }
    }
    
    fn find_break_point(&self, text: &str) -> Option<usize> {
        // Look for sentence endings first
        if let Some(sentence_end) = text.rfind(|c: char| ".!?".contains(c)) {
            return Some(sentence_end + 1);
        }
        
        // Look for paragraph breaks
        if let Some(para_break) = text.rfind("\n\n") {
            return Some(para_break + 2);
        }
        
        // Look for line breaks
        if let Some(line_break) = text.rfind('\n') {
            return Some(line_break + 1);
        }
        
        // Look for word boundaries
        if let Some(space) = text.rfind(' ') {
            return Some(space + 1);
        }
        
        None
    }
    
    fn create_chunk(
        &self,
        content: String,
        start_char: usize,
        end_char: usize,
        chunk_index: usize,
        overlap_start: bool,
        overlap_end: bool,
    ) -> TextChunk {
        let word_count = content.split_whitespace().count();
        let sentence_count = self.sentence_splitter.find_iter(&content).count().max(1);
        
        TextChunk {
            id: uuid::Uuid::new_v4().to_string(),
            content,
            start_char,
            end_char,
            chunk_index,
            word_count,
            sentence_count,
            overlap_start,
            overlap_end,
            embedding: None, // Will be populated later by DocumentProcessor
            
            // Database fields (not used during chunking)
            document_id: None,
            start_position: None,
            end_position: None,
            total_chunks: None,
            metadata: None,
        }
    }
    
    pub fn merge_chunks(&self, chunks: Vec<TextChunk>) -> String {
        chunks.into_iter()
            .map(|chunk| chunk.content)
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    pub fn get_chunk_context(&self, chunks: &[TextChunk], target_index: usize, context_size: usize) -> String {
        let start_index = target_index.saturating_sub(context_size);
        let end_index = (target_index + context_size + 1).min(chunks.len());
        
        chunks[start_index..end_index]
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    pub fn validate_chunks(&self, chunks: &[TextChunk]) -> Result<()> {
        for (i, chunk) in chunks.iter().enumerate() {
            if chunk.content.len() > self.options.max_chunk_size {
                return Err(AppError::Indexing(IndexingError::Chunking(
                    format!("Chunk {} exceeds maximum size: {} > {}", 
                           i, chunk.content.len(), self.options.max_chunk_size)
                )));
            }
            
            if chunk.content.len() < self.options.min_chunk_size && chunks.len() > 1 {
                return Err(AppError::Indexing(IndexingError::Chunking(
                    format!("Chunk {} below minimum size: {} < {}", 
                           i, chunk.content.len(), self.options.min_chunk_size)
                )));
            }
            
            if chunk.start_char >= chunk.end_char {
                return Err(AppError::Indexing(IndexingError::Chunking(
                    format!("Invalid chunk positions: start {} >= end {}", 
                           chunk.start_char, chunk.end_char)
                )));
            }
        }
        
        Ok(())
    }
}

impl Default for TextChunker {
    fn default() -> Self {
        Self::new(ChunkingOptions::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_small_text_chunking() {
        let chunker = TextChunker::default();
        let text = "This is a short text that should be in one chunk.";
        
        let chunks = chunker.chunk_text(text).unwrap();
        
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, text);
        assert_eq!(chunks[0].start_char, 0);
        assert_eq!(chunks[0].end_char, text.len());
        assert!(!chunks[0].overlap_start);
        assert!(!chunks[0].overlap_end);
    }
    
    #[test]
    fn test_sentence_based_chunking() {
        let chunker = TextChunker::new(ChunkingOptions {
            max_chunk_size: 100,
            target_chunk_size: 80,
            overlap_percentage: 0.2,
            ..Default::default()
        });
        
        let text = "First sentence is here. Second sentence follows. Third sentence continues. Fourth sentence ends. Fifth sentence concludes.";
        
        let chunks = chunker.chunk_text(text).unwrap();
        
        assert!(chunks.len() > 1);
        
        // Validate chunks
        chunker.validate_chunks(&chunks).unwrap();
        
        // Check that all chunks are within size limits
        for chunk in &chunks {
            assert!(chunk.content.len() <= 100);
            assert!(chunk.word_count > 0);
            assert!(chunk.sentence_count > 0);
        }
        
        // Check overlap flags
        if chunks.len() > 1 {
            assert!(!chunks[0].overlap_start);
            assert!(chunks[0].overlap_end);
            assert!(chunks[chunks.len() - 1].overlap_start);
            assert!(!chunks[chunks.len() - 1].overlap_end);
        }
    }
    
    #[test]
    fn test_character_based_chunking() {
        let chunker = TextChunker::new(ChunkingOptions {
            max_chunk_size: 50,
            target_chunk_size: 40,
            overlap_percentage: 0.1,
            sentence_boundary_split: false,
            ..Default::default()
        });
        
        let text = "This is a longer text that needs to be split into multiple chunks based on character count rather than sentence boundaries to test the character-based chunking algorithm.";
        
        let chunks = chunker.chunk_text(text).unwrap();
        
        assert!(chunks.len() > 1);
        
        // Validate chunks
        chunker.validate_chunks(&chunks).unwrap();
        
        // Check size constraints
        for chunk in &chunks {
            assert!(chunk.content.len() <= 50);
        }
    }
    
    #[test]
    fn test_overlap_calculation() {
        let chunker = TextChunker::new(ChunkingOptions {
            overlap_percentage: 0.2,
            ..Default::default()
        });
        
        let chunk_text = "This is a test chunk with multiple words for overlap testing.";
        let overlap = chunker.calculate_overlap(chunk_text);
        
        assert!(!overlap.is_empty());
        assert!(overlap.len() < chunk_text.len());
        
        // Overlap should be roughly 20% of original
        let expected_overlap_size = (chunk_text.len() as f32 * 0.2) as usize;
        assert!(overlap.len() >= expected_overlap_size - 10); // Allow some variance for word boundaries
    }
    
    #[test]
    fn test_sentence_splitting() {
        let chunker = TextChunker::default();
        let text = "First sentence. Second sentence! Third sentence? Fourth statement.";
        
        let sentences = chunker.split_into_sentences(text);
        
        assert_eq!(sentences.len(), 4);
        assert!(sentences[0].contains("First sentence"));
        assert!(sentences[1].contains("Second sentence"));
        assert!(sentences[2].contains("Third sentence"));
        assert!(sentences[3].contains("Fourth statement"));
    }
    
    #[test]
    fn test_chunk_merging() {
        let chunker = TextChunker::default();
        let chunks = vec![
            TextChunk {
                id: "chunk_0".to_string(),
                content: "First chunk content.".to_string(),
                start_char: 0,
                end_char: 20,
                chunk_index: 0,
                word_count: 3,
                sentence_count: 1,
                overlap_start: false,
                overlap_end: true,
                embedding: None,
                document_id: None,
                start_position: None,
                end_position: None,
                total_chunks: None,
                metadata: None,
            },
            TextChunk {
                id: "chunk_1".to_string(),
                content: "Second chunk content.".to_string(),
                start_char: 15,
                end_char: 36,
                chunk_index: 1,
                word_count: 3,
                sentence_count: 1,
                overlap_start: true,
                overlap_end: false,
                embedding: None,
                document_id: None,
                start_position: None,
                end_position: None,
                total_chunks: None,
                metadata: None,
            },
        ];
        
        let merged = chunker.merge_chunks(chunks);
        assert_eq!(merged, "First chunk content. Second chunk content.");
    }
    
    #[test]
    fn test_chunk_context() {
        let chunker = TextChunker::default();
        let chunks = vec![
            TextChunk {
                id: "chunk_0".to_string(),
                content: "First chunk.".to_string(),
                start_char: 0, end_char: 12, chunk_index: 0,
                word_count: 2, sentence_count: 1,
                overlap_start: false, overlap_end: true,
                embedding: None,
                document_id: None, start_position: None, end_position: None,
                total_chunks: None, metadata: None,
            },
            TextChunk {
                id: "chunk_1".to_string(),
                content: "Second chunk.".to_string(),
                start_char: 10, end_char: 23, chunk_index: 1,
                word_count: 2, sentence_count: 1,
                overlap_start: true, overlap_end: true,
                embedding: None,
                document_id: None, start_position: None, end_position: None,
                total_chunks: None, metadata: None,
            },
            TextChunk {
                id: "chunk_2".to_string(),
                content: "Third chunk.".to_string(),
                start_char: 20, end_char: 32, chunk_index: 2,
                word_count: 2, sentence_count: 1,
                overlap_start: true, overlap_end: false,
                embedding: None,
                document_id: None, start_position: None, end_position: None,
                total_chunks: None, metadata: None,
            },
        ];
        
        let context = chunker.get_chunk_context(&chunks, 1, 1);
        assert_eq!(context, "First chunk. Second chunk. Third chunk.");
    }
}