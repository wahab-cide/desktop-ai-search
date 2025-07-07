use crate::error::{AppError, FileSystemError, Result};
use crate::models::FileType;
use std::path::Path;
use std::fs::File;
use std::io::{Read, BufReader};
use std::collections::HashMap;
use infer;
use mime_guess;

#[derive(Debug, Clone)]
pub struct FileClassificationResult {
    pub file_type: FileType,
    pub is_text: bool,
    pub should_embed: bool,
    pub confidence: f32,
    pub mime_type: Option<String>,
}

pub struct FileTypeDetector {
    mime_overrides: HashMap<String, FileType>,
}

impl FileTypeDetector {
    pub fn new() -> Self {
        let mut mime_overrides = HashMap::new();
        
        // Add specific MIME type mappings
        mime_overrides.insert("application/pdf".to_string(), FileType::Pdf);
        mime_overrides.insert("application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string(), FileType::Docx);
        mime_overrides.insert("text/markdown".to_string(), FileType::Markdown);
        mime_overrides.insert("text/html".to_string(), FileType::Html);
        mime_overrides.insert("message/rfc822".to_string(), FileType::Email);
        
        Self { mime_overrides }
    }
    
    pub async fn detect_file_type<P: AsRef<Path>>(&self, path: P) -> Result<FileClassificationResult> {
        let path = path.as_ref();
        
        // First, try magic byte detection using infer crate
        let magic_result = self.detect_by_magic_bytes(path).await?;
        
        // Fallback to extension-based detection
        let extension_result = self.detect_by_extension(path)?;
        
        // Combine results with confidence scoring
        let final_result = self.combine_detection_results(magic_result, extension_result)?;
        
        Ok(final_result)
    }
    
    async fn detect_by_magic_bytes<P: AsRef<Path>>(&self, path: P) -> Result<Option<FileClassificationResult>> {
        let path = path.as_ref();
        
        // Read first 8KB for magic byte detection
        let mut file = File::open(path)
            .map_err(|e| AppError::FileSystem(FileSystemError::FileType(e.to_string())))?;
        
        let mut buffer = vec![0; 8192];
        let bytes_read = file.read(&mut buffer)
            .map_err(|e| AppError::FileSystem(FileSystemError::FileType(e.to_string())))?;
        
        buffer.truncate(bytes_read);
        
        // Use infer crate for magic byte detection
        if let Some(kind) = infer::get(&buffer) {
            let file_type = self.infer_type_to_file_type(&kind);
            let is_text = self.is_text_type(&file_type);
            let should_embed = self.should_embed_type(&file_type);
            
            return Ok(Some(FileClassificationResult {
                file_type,
                is_text,
                should_embed,
                confidence: 0.9, // High confidence for magic byte detection
                mime_type: Some(kind.mime_type().to_string()),
            }));
        }
        
        // Check for office document containers (ZIP-based)
        if self.is_zip_container(&buffer) {
            if let Some(office_type) = self.detect_office_format(path).await? {
                return Ok(Some(office_type));
            }
        }
        
        // Check if it's a text file by analyzing content
        if self.is_likely_text(&buffer) {
            return Ok(Some(FileClassificationResult {
                file_type: FileType::Text,
                is_text: true,
                should_embed: true,
                confidence: 0.7,
                mime_type: Some("text/plain".to_string()),
            }));
        }
        
        Ok(None)
    }
    
    fn detect_by_extension<P: AsRef<Path>>(&self, path: P) -> Result<FileClassificationResult> {
        let path = path.as_ref();
        
        let mime_guess = mime_guess::from_path(path);
        let mime_type = mime_guess.first();
        
        let (file_type, confidence) = if let Some(ref mime) = mime_type {
            let mime_str = mime.as_ref();
            
            if let Some(mapped_type) = self.mime_overrides.get(mime_str) {
                (mapped_type.clone(), 0.8)
            } else if mime_str.starts_with("text/") {
                (FileType::Text, 0.7)
            } else if mime_str.starts_with("image/") {
                (FileType::Image, 0.7)
            } else if mime_str.starts_with("audio/") {
                (FileType::Audio, 0.7)
            } else if mime_str.starts_with("video/") {
                (FileType::Video, 0.7)
            } else {
                (FileType::Unknown, 0.3)
            }
        } else {
            // Fallback to extension analysis
            match path.extension().and_then(|ext| ext.to_str()) {
                Some("pdf") => (FileType::Pdf, 0.6),
                Some("docx") | Some("doc") => (FileType::Docx, 0.6),
                Some("md") | Some("markdown") => (FileType::Markdown, 0.6),
                Some("html") | Some("htm") => (FileType::Html, 0.6),
                Some("txt") | Some("text") => (FileType::Text, 0.6),
                Some("jpg") | Some("jpeg") | Some("png") | Some("gif") | Some("bmp") | Some("webp") => (FileType::Image, 0.6),
                Some("mp3") | Some("wav") | Some("flac") | Some("ogg") => (FileType::Audio, 0.6),
                Some("mp4") | Some("avi") | Some("mkv") | Some("mov") => (FileType::Video, 0.6),
                Some("eml") | Some("msg") => (FileType::Email, 0.6),
                _ => (FileType::Unknown, 0.1),
            }
        };
        
        let is_text = self.is_text_type(&file_type);
        let should_embed = self.should_embed_type(&file_type);
        
        Ok(FileClassificationResult {
            file_type,
            is_text,
            should_embed,
            confidence,
            mime_type: mime_type.map(|m| m.to_string()),
        })
    }
    
    fn combine_detection_results(
        &self,
        magic_result: Option<FileClassificationResult>,
        extension_result: FileClassificationResult,
    ) -> Result<FileClassificationResult> {
        match magic_result {
            Some(magic) if magic.confidence > extension_result.confidence => Ok(magic),
            Some(magic) if magic.file_type != FileType::Unknown => Ok(magic),
            _ => Ok(extension_result),
        }
    }
    
    async fn detect_office_format<P: AsRef<Path>>(&self, path: P) -> Result<Option<FileClassificationResult>> {
        let path = path.as_ref();
        
        // For ZIP containers, we need to peek inside to check for Office document structures
        // This is a simplified version - in production you'd want to use a proper ZIP reader
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("docx") | Some("docm") => {
                Ok(Some(FileClassificationResult {
                    file_type: FileType::Docx,
                    is_text: true,
                    should_embed: true,
                    confidence: 0.85,
                    mime_type: Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string()),
                }))
            }
            Some("pages") => {
                Ok(Some(FileClassificationResult {
                    file_type: FileType::Docx, // Treat Pages as document type
                    is_text: true,
                    should_embed: true,
                    confidence: 0.8,
                    mime_type: Some("application/x-iwork-pages-sffpages".to_string()),
                }))
            }
            _ => Ok(None),
        }
    }
    
    fn infer_type_to_file_type(&self, kind: &infer::Type) -> FileType {
        match kind.mime_type() {
            "application/pdf" => FileType::Pdf,
            "text/html" => FileType::Html,
            mime if mime.starts_with("text/") => FileType::Text,
            mime if mime.starts_with("image/") => FileType::Image,
            mime if mime.starts_with("audio/") => FileType::Audio,
            mime if mime.starts_with("video/") => FileType::Video,
            _ => FileType::Unknown,
        }
    }
    
    fn is_zip_container(&self, buffer: &[u8]) -> bool {
        // ZIP magic bytes: PK (0x50 0x4B)
        buffer.len() >= 4 && buffer[0] == 0x50 && buffer[1] == 0x4B
    }
    
    fn is_likely_text(&self, buffer: &[u8]) -> bool {
        if buffer.is_empty() {
            return false;
        }
        
        // Check for UTF-8 BOM
        if buffer.len() >= 3 && buffer[0] == 0xEF && buffer[1] == 0xBB && buffer[2] == 0xBF {
            return true;
        }
        
        // Sample analysis - check if most bytes are printable ASCII or valid UTF-8
        let sample_size = std::cmp::min(buffer.len(), 1024);
        let sample = &buffer[..sample_size];
        
        let mut printable_count = 0;
        let mut total_count = 0;
        
        for &byte in sample {
            total_count += 1;
            if byte >= 32 && byte <= 126 || byte == b'\n' || byte == b'\r' || byte == b'\t' {
                printable_count += 1;
            }
        }
        
        // Consider it text if > 80% of characters are printable
        let printable_ratio = printable_count as f32 / total_count as f32;
        printable_ratio > 0.8
    }
    
    fn is_text_type(&self, file_type: &FileType) -> bool {
        matches!(
            file_type,
            FileType::Text | FileType::Markdown | FileType::Html | FileType::Email | FileType::Docx
        )
    }
    
    fn should_embed_type(&self, file_type: &FileType) -> bool {
        matches!(
            file_type,
            FileType::Text 
            | FileType::Pdf 
            | FileType::Docx 
            | FileType::Markdown 
            | FileType::Html 
            | FileType::Email
            | FileType::Image   // Now supported with real OCR
            | FileType::Audio   // Now supported with real Whisper
        )
    }
    
    pub fn get_indexable_types() -> Vec<FileType> {
        vec![
            FileType::Text,
            FileType::Pdf,
            FileType::Docx,
            FileType::Markdown,
            FileType::Html,
            FileType::Email,
            FileType::Image, // For OCR
            FileType::Audio, // For transcription
        ]
    }
    
    pub fn should_process_for_indexing(&self, classification: &FileClassificationResult) -> bool {
        classification.should_embed && classification.confidence > 0.5
    }
    
    /// Check if a file should be indexed based on its path and type
    pub fn is_indexable<P: AsRef<Path>>(&self, path: P) -> bool {
        let path = path.as_ref();
        
        // Skip hidden files and directories
        if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
            if file_name.starts_with('.') {
                return false;
            }
        }
        
        // Skip common non-indexable directories
        if path.is_dir() {
            if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                if matches!(dir_name, "node_modules" | "target" | ".git" | ".svn" | "__pycache__" | ".vscode" | ".idea") {
                    return false;
                }
            }
        }
        
        // Check file extension for quick filtering
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            let ext_lower = ext.to_lowercase();
            return matches!(ext_lower.as_str(), 
                "txt" | "md" | "markdown" | "html" | "htm" | "pdf" | "docx" | "doc" |
                "png" | "jpg" | "jpeg" | "gif" | "bmp" | "webp" |
                "mp3" | "wav" | "flac" | "ogg" | "m4a" | "aac" |
                "mp4" | "avi" | "mkv" | "mov" | "webm" |
                "eml" | "msg"
            );
        }
        
        // If no extension, try to detect if it's a text file
        if path.is_file() {
            // For files without extensions, do a basic check
            if let Ok(mut file) = std::fs::File::open(path) {
                let mut buffer = [0; 512];
                if let Ok(bytes_read) = std::io::Read::read(&mut file, &mut buffer) {
                    return self.is_likely_text(&buffer[..bytes_read]);
                }
            }
        }
        
        false
    }
}

impl Default for FileTypeDetector {
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
    async fn test_text_file_detection() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "This is a test text file.").unwrap();
        temp_file.flush().unwrap();
        
        let detector = FileTypeDetector::new();
        let result = detector.detect_file_type(temp_file.path()).await.unwrap();
        
        assert!(matches!(result.file_type, FileType::Text));
        assert!(result.is_text);
        assert!(result.should_embed);
        assert!(result.confidence > 0.5);
    }
    
    #[tokio::test]
    async fn test_pdf_extension_detection() {
        let temp_file = NamedTempFile::with_suffix(".pdf").unwrap();
        
        let detector = FileTypeDetector::new();
        let result = detector.detect_file_type(temp_file.path()).await.unwrap();
        
        // Should detect as PDF based on extension even if content doesn't match
        assert!(matches!(result.file_type, FileType::Pdf));
        assert!(!result.is_text); // PDF is not directly text
        assert!(result.should_embed); // But should be processed for indexing
    }
    
    #[test]
    fn test_indexable_types() {
        let indexable = FileTypeDetector::get_indexable_types();
        assert!(indexable.contains(&FileType::Text));
        assert!(indexable.contains(&FileType::Pdf));
        assert!(indexable.contains(&FileType::Image));
        assert!(!indexable.contains(&FileType::Unknown));
    }
    
    #[test]
    fn test_text_content_analysis() {
        let detector = FileTypeDetector::new();
        
        // Test printable text
        let text_content = b"Hello, world!\nThis is a text file.";
        assert!(detector.is_likely_text(text_content));
        
        // Test binary content
        let binary_content = &[0x00, 0x01, 0x02, 0x03, 0xFF, 0xFE];
        assert!(!detector.is_likely_text(binary_content));
        
        // Test UTF-8 BOM
        let utf8_bom = &[0xEF, 0xBB, 0xBF, b'H', b'e', b'l', b'l', b'o'];
        assert!(detector.is_likely_text(utf8_bom));
    }
}