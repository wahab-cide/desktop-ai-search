use crate::error::{AppError, IndexingError, Result};
use crate::models::FileType;
use std::path::Path;
use std::fs::File;
use std::io::{Read, BufReader};
use pulldown_cmark::{Parser, html::push_html, Options, Event};
use html2text;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct ExtractionResult {
    pub content: String,
    pub extraction_mode: ExtractionMode,
    pub page_count: Option<usize>,
    pub word_count: usize,
    pub character_count: usize,
    pub language: Option<String>,
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum ExtractionMode {
    Text,
    PdfText,
    PdfOcr,
    DocxStructured,
    HtmlConverted,
    MarkdownParsed,
    OcrRequired,
    TranscriptionRequired,
}

pub struct TextExtractor {
    html_cleaner: Regex,
    whitespace_normalizer: Regex,
    sentence_splitter: Regex,
}

impl TextExtractor {
    pub fn new() -> Self {
        Self {
            html_cleaner: Regex::new(r"<[^>]*>").unwrap(),
            whitespace_normalizer: Regex::new(r"\s+").unwrap(),
            sentence_splitter: Regex::new(r"[.!?]+\s+").unwrap(),
        }
    }
    
    pub async fn extract_text<P: AsRef<Path>>(
        &self,
        path: P,
        file_type: &FileType,
    ) -> Result<ExtractionResult> {
        let path = path.as_ref();
        
        match file_type {
            FileType::Text => self.extract_plain_text(path).await,
            FileType::Pdf => self.extract_pdf_text(path).await,
            FileType::Docx => self.extract_docx_text(path).await,
            FileType::Html => self.extract_html_text(path).await,
            FileType::Markdown => self.extract_markdown_text(path).await,
            FileType::Email => self.extract_email_text(path).await,
            FileType::Image => self.prepare_for_ocr(path).await,
            FileType::Audio => self.prepare_for_transcription(path).await,
            _ => Err(AppError::Indexing(IndexingError::Extraction(
                format!("Unsupported file type: {:?}", file_type)
            ))),
        }
    }
    
    async fn extract_plain_text<P: AsRef<Path>>(&self, path: P) -> Result<ExtractionResult> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| AppError::Indexing(IndexingError::Extraction(e.to_string())))?;
        
        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| AppError::Indexing(IndexingError::Extraction(e.to_string())))?;
        
        // Normalize whitespace
        let normalized_content = self.whitespace_normalizer.replace_all(&content, " ").to_string();
        
        Ok(ExtractionResult {
            word_count: normalized_content.split_whitespace().count(),
            character_count: normalized_content.len(),
            content: normalized_content,
            extraction_mode: ExtractionMode::Text,
            page_count: None,
            language: self.detect_language(&content),
            metadata: std::collections::HashMap::new(),
        })
    }
    
    async fn extract_pdf_text<P: AsRef<Path>>(&self, path: P) -> Result<ExtractionResult> {
        let path = path.as_ref();
        
        // Try text extraction first
        match pdf_extract::extract_text(path) {
            Ok(content) if !content.trim().is_empty() => {
                let normalized_content = self.whitespace_normalizer.replace_all(&content, " ").to_string();
                
                Ok(ExtractionResult {
                    content: normalized_content.clone(),
                    extraction_mode: ExtractionMode::PdfText,
                    page_count: self.estimate_pdf_pages(&normalized_content),
                    word_count: normalized_content.split_whitespace().count(),
                    character_count: normalized_content.len(),
                    language: self.detect_language(&normalized_content),
                    metadata: std::collections::HashMap::new(),
                })
            }
            _ => {
                // PDF text extraction failed, will need OCR
                Ok(ExtractionResult {
                    content: String::new(),
                    extraction_mode: ExtractionMode::OcrRequired,
                    page_count: None,
                    word_count: 0,
                    character_count: 0,
                    language: None,
                    metadata: std::collections::HashMap::new(),
                })
            }
        }
    }
    
    async fn extract_docx_text<P: AsRef<Path>>(&self, path: P) -> Result<ExtractionResult> {
        let path = path.as_ref();
        
        // Basic DOCX extraction - in a full implementation you'd use a proper DOCX library
        // For now, we'll try to extract as ZIP and parse the XML
        match self.extract_docx_content(path) {
            Ok(content) => {
                let normalized_content = self.whitespace_normalizer.replace_all(&content, " ").to_string();
                
                Ok(ExtractionResult {
                    content: normalized_content.clone(),
                    extraction_mode: ExtractionMode::DocxStructured,
                    page_count: None,
                    word_count: normalized_content.split_whitespace().count(),
                    character_count: normalized_content.len(),
                    language: self.detect_language(&normalized_content),
                    metadata: std::collections::HashMap::new(),
                })
            }
            Err(_) => {
                // DOCX extraction failed
                Err(AppError::Indexing(IndexingError::Extraction(
                    "Failed to extract DOCX content".to_string()
                )))
            }
        }
    }
    
    async fn extract_html_text<P: AsRef<Path>>(&self, path: P) -> Result<ExtractionResult> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| AppError::Indexing(IndexingError::Extraction(e.to_string())))?;
        
        let mut html_content = String::new();
        file.read_to_string(&mut html_content)
            .map_err(|e| AppError::Indexing(IndexingError::Extraction(e.to_string())))?;
        
        // Convert HTML to plain text
        let text_content = html2text::from_read(html_content.as_bytes(), 80);
        let normalized_content = self.whitespace_normalizer.replace_all(&text_content, " ").to_string();
        
        Ok(ExtractionResult {
            content: normalized_content.clone(),
            extraction_mode: ExtractionMode::HtmlConverted,
            page_count: None,
            word_count: normalized_content.split_whitespace().count(),
            character_count: normalized_content.len(),
            language: self.detect_language(&normalized_content),
            metadata: std::collections::HashMap::new(),
        })
    }
    
    async fn extract_markdown_text<P: AsRef<Path>>(&self, path: P) -> Result<ExtractionResult> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| AppError::Indexing(IndexingError::Extraction(e.to_string())))?;
        
        let mut markdown_content = String::new();
        file.read_to_string(&mut markdown_content)
            .map_err(|e| AppError::Indexing(IndexingError::Extraction(e.to_string())))?;
        
        // Parse markdown and convert to plain text
        let parser = Parser::new_ext(&markdown_content, Options::all());
        let mut plain_text = String::new();
        
        for event in parser {
            match event {
                Event::Text(text) | Event::Code(text) => {
                    plain_text.push_str(&text);
                    plain_text.push(' ');
                }
                Event::SoftBreak | Event::HardBreak => {
                    plain_text.push(' ');
                }
                _ => {}
            }
        }
        
        let normalized_content = self.whitespace_normalizer.replace_all(&plain_text, " ").to_string();
        
        Ok(ExtractionResult {
            content: normalized_content.clone(),
            extraction_mode: ExtractionMode::MarkdownParsed,
            page_count: None,
            word_count: normalized_content.split_whitespace().count(),
            character_count: normalized_content.len(),
            language: self.detect_language(&normalized_content),
            metadata: std::collections::HashMap::new(),
        })
    }
    
    async fn extract_email_text<P: AsRef<Path>>(&self, path: P) -> Result<ExtractionResult> {
        // For now, treat email as plain text - in a full implementation you'd parse MIME
        self.extract_plain_text(path).await
    }
    
    async fn prepare_for_ocr<P: AsRef<Path>>(&self, path: P) -> Result<ExtractionResult> {
        // Try to perform OCR on the image
        use crate::core::ocr_processor::{OcrProcessor, OcrOptions};
        
        let ocr_options = OcrOptions::default();
        match OcrProcessor::new(ocr_options) {
            Ok(processor) => {
                match processor.process_image(path.as_ref()).await {
                    Ok(ocr_result) => {
                        let normalized_content = self.whitespace_normalizer.replace_all(&ocr_result.text, " ").to_string();
                        
                        Ok(ExtractionResult {
                            content: normalized_content.clone(),
                            extraction_mode: ExtractionMode::OcrRequired,
                            page_count: Some(1),
                            word_count: normalized_content.split_whitespace().count(),
                            character_count: normalized_content.len(),
                            language: ocr_result.language,
                            metadata: ocr_result.metadata,
                        })
                    }
                    Err(_) => {
                        // If OCR fails, return placeholder indicating OCR was attempted but failed
                        Ok(ExtractionResult {
                            content: String::new(),
                            extraction_mode: ExtractionMode::OcrRequired,
                            page_count: None,
                            word_count: 0,
                            character_count: 0,
                            language: None,
                            metadata: std::collections::HashMap::new(),
                        })
                    }
                }
            }
            Err(_) => {
                // OCR processor initialization failed (likely Tesseract not installed)
                Ok(ExtractionResult {
                    content: String::new(),
                    extraction_mode: ExtractionMode::OcrRequired,
                    page_count: None,
                    word_count: 0,
                    character_count: 0,
                    language: None,
                    metadata: std::collections::HashMap::new(),
                })
            }
        }
    }
    
    async fn prepare_for_transcription<P: AsRef<Path>>(&self, path: P) -> Result<ExtractionResult> {
        // Return placeholder for audio transcription
        Ok(ExtractionResult {
            content: String::new(),
            extraction_mode: ExtractionMode::TranscriptionRequired,
            page_count: None,
            word_count: 0,
            character_count: 0,
            language: None,
            metadata: std::collections::HashMap::new(),
        })
    }
    
    fn extract_docx_content<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        let file = File::open(path.as_ref())
            .map_err(|e| AppError::Indexing(IndexingError::Extraction(e.to_string())))?;
        
        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| AppError::Indexing(IndexingError::Extraction(e.to_string())))?;
        
        // Try to extract document.xml
        let mut document_xml = archive.by_name("word/document.xml")
            .map_err(|e| AppError::Indexing(IndexingError::Extraction(e.to_string())))?;
        
        let mut xml_content = String::new();
        document_xml.read_to_string(&mut xml_content)
            .map_err(|e| AppError::Indexing(IndexingError::Extraction(e.to_string())))?;
        
        // Simple XML text extraction (in production, you'd use a proper XML parser)
        let text = self.html_cleaner.replace_all(&xml_content, " ").to_string();
        Ok(text)
    }
    
    pub fn detect_language(&self, content: &str) -> Option<String> {
        // Simple language detection heuristic - in production you'd use a proper library
        if content.len() < 50 {
            return None;
        }
        
        // Very basic English detection
        let english_indicators = ["the", "and", "or", "is", "are", "was", "were"];
        let words: Vec<&str> = content.split_whitespace().take(100).collect();
        let english_count = words.iter()
            .filter(|word| english_indicators.contains(&word.to_lowercase().as_str()))
            .count();
        
        if english_count > words.len() / 10 {
            Some("en".to_string())
        } else {
            Some("unknown".to_string())
        }
    }
    
    fn estimate_pdf_pages(&self, content: &str) -> Option<usize> {
        // Rough estimation: ~500 words per page
        let word_count = content.split_whitespace().count();
        if word_count > 0 {
            Some((word_count / 500).max(1))
        } else {
            None
        }
    }
    
    pub fn clean_extracted_text(&self, text: &str) -> String {
        // Remove excessive whitespace and normalize
        let cleaned = self.whitespace_normalizer.replace_all(text, " ");
        cleaned.trim().to_string()
    }
    
    pub fn is_meaningful_content(&self, text: &str) -> bool {
        let words = text.split_whitespace().count();
        let chars = text.trim().len();
        
        // Heuristics for meaningful content
        words >= 3 && chars >= 20 && (chars as f64 / words as f64) > 2.0
    }
}

impl Default for TextExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;
    
    #[tokio::test]
    async fn test_plain_text_extraction() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = "This is a test document with multiple sentences. It should be processed correctly.";
        writeln!(temp_file, "{}", content).unwrap();
        temp_file.flush().unwrap();
        
        let extractor = TextExtractor::new();
        let result = extractor.extract_text(temp_file.path(), &FileType::Text).await.unwrap();
        
        assert!(matches!(result.extraction_mode, ExtractionMode::Text));
        assert!(result.content.contains("test document"));
        assert!(result.word_count > 0);
        assert!(result.character_count > 0);
    }
    
    #[tokio::test]
    async fn test_markdown_extraction() {
        let mut temp_file = NamedTempFile::with_suffix(".md").unwrap();
        let markdown_content = r#"
# Test Document

This is a **bold** text with [links](http://example.com).

## Section 2

- List item 1
- List item 2

Some `code` here.
        "#;
        write!(temp_file, "{}", markdown_content).unwrap();
        temp_file.flush().unwrap();
        
        let extractor = TextExtractor::new();
        let result = extractor.extract_text(temp_file.path(), &FileType::Markdown).await.unwrap();
        
        assert!(matches!(result.extraction_mode, ExtractionMode::MarkdownParsed));
        assert!(result.content.contains("Test Document"));
        assert!(result.content.contains("bold"));
        assert!(!result.content.contains("#")); // Markdown syntax should be removed
        assert!(!result.content.contains("**")); // Bold syntax should be removed
    }
    
    #[tokio::test]
    async fn test_html_extraction() {
        let mut temp_file = NamedTempFile::with_suffix(".html").unwrap();
        let html_content = r##"
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Title</h1>
            <p>This is a <strong>paragraph</strong> with <a href="#">links</a>.</p>
            <div>Another section with content.</div>
        </body>
        </html>
        "##;
        write!(temp_file, "{}", html_content).unwrap();
        temp_file.flush().unwrap();
        
        let extractor = TextExtractor::new();
        let result = extractor.extract_text(temp_file.path(), &FileType::Html).await.unwrap();
        
        assert!(matches!(result.extraction_mode, ExtractionMode::HtmlConverted));
        assert!(result.content.contains("Main Title"));
        assert!(result.content.contains("paragraph"));
        assert!(!result.content.contains("<")); // HTML tags should be removed
    }
    
    #[test]
    fn test_language_detection() {
        let extractor = TextExtractor::new();
        
        let english_text = "The quick brown fox jumps over the lazy dog. This is clearly English text.";
        assert_eq!(extractor.detect_language(english_text), Some("en".to_string()));
        
        let short_text = "Hi";
        assert_eq!(extractor.detect_language(short_text), None);
        
        let random_text = "Lorem ipsum dolor sit amet consectetur adipiscing elit";
        assert_eq!(extractor.detect_language(random_text), Some("unknown".to_string()));
    }
    
    #[test]
    fn test_content_meaningfulness() {
        let extractor = TextExtractor::new();
        
        assert!(extractor.is_meaningful_content("This is a meaningful sentence with enough content."));
        assert!(!extractor.is_meaningful_content("Hi"));
        assert!(!extractor.is_meaningful_content("   "));
        assert!(!extractor.is_meaningful_content("a b c"));
    }
}