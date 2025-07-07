use crate::error::Result;
use crate::models::{Document, FileType};
use crate::utils::{
    file_watcher::{FileSystemWatcher, WatcherEvent, DebouncedEventProcessor},
    file_types::{FileTypeDetector, FileClassificationResult},
    hashing::TieredHasher,
    metadata::{MetadataExtractor, TraversalOptions, ExtractedMetadata},
};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::mpsc;
use uuid::Uuid;
use chrono::Utc;

pub struct FileProcessor {
    file_watcher: FileSystemWatcher,
    event_receiver: mpsc::UnboundedReceiver<WatcherEvent>,
    debounced_processor: DebouncedEventProcessor,
    file_type_detector: FileTypeDetector,
    hasher: TieredHasher,
    metadata_extractor: MetadataExtractor,
}

impl FileProcessor {
    pub fn new() -> Result<Self> {
        let (file_watcher, event_receiver) = FileSystemWatcher::new()?;
        let debounced_processor = DebouncedEventProcessor::new(Duration::from_millis(500));
        let file_type_detector = FileTypeDetector::new();
        let hasher = TieredHasher::new();
        
        let traversal_options = TraversalOptions {
            max_depth: Some(10),
            follow_symlinks: false,
            include_hidden: false,
            exclude_patterns: vec![
                ".git".to_string(),
                ".svn".to_string(),
                "node_modules".to_string(),
                ".DS_Store".to_string(),
                "Thumbs.db".to_string(),
                ".cache".to_string(),
                ".tmp".to_string(),
            ],
            min_file_size: Some(1), // Skip empty files
            max_file_size: Some(100 * 1024 * 1024), // Skip files > 100MB
        };
        
        let metadata_extractor = MetadataExtractor::new(traversal_options);
        
        Ok(Self {
            file_watcher,
            event_receiver,
            debounced_processor,
            file_type_detector,
            hasher,
            metadata_extractor,
        })
    }
    
    pub async fn process_file<P: AsRef<Path>>(&mut self, path: P) -> Result<Option<Document>> {
        let path = path.as_ref();
        
        // Skip if not a regular file
        if !path.is_file() {
            return Ok(None);
        }
        
        // Extract metadata
        let metadata = self.metadata_extractor.extract_metadata(path)?;
        
        // Skip if file is too small or too large (based on our criteria)
        if metadata.basic.size == 0 || metadata.basic.size > 100 * 1024 * 1024 {
            return Ok(None);
        }
        
        // Detect file type
        let classification = self.file_type_detector.detect_file_type(path).await?;
        
        // Skip if not indexable
        if !self.file_type_detector.should_process_for_indexing(&classification) {
            return Ok(None);
        }
        
        // Compute content hash
        let content_hash = self.hasher.compute_hash(path).await?;
        
        // Create Document
        let document = Document {
            id: Uuid::new_v4(),
            file_path: path.to_string_lossy().to_string(),
            content_hash,
            file_type: classification.file_type.clone(),
            creation_date: metadata.basic.created,
            modification_date: metadata.basic.modified,
            last_indexed: Utc::now(),
            file_size: metadata.basic.size,
            metadata: self.build_metadata_map(&metadata, &classification),
        };
        
        Ok(Some(document))
    }
    
    pub async fn scan_directory<P: AsRef<Path>>(&mut self, root_path: P) -> Result<Vec<Document>> {
        let root_path = root_path.as_ref();
        
        // Get all files in the directory
        let file_paths = self.metadata_extractor.traverse_directory(root_path)?;
        
        println!("Found {} files to process", file_paths.len());
        
        let mut documents = Vec::new();
        let mut processed = 0;
        
        for file_path in &file_paths {
            if let Some(document) = self.process_file(&file_path).await? {
                documents.push(document);
            }
            
            processed += 1;
            if processed % 100 == 0 {
                println!("Processed {}/{} files", processed, file_paths.len());
            }
        }
        
        println!("Successfully processed {} documents", documents.len());
        Ok(documents)
    }
    
    pub fn start_watching<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.file_watcher.watch_directory(path)
    }
    
    pub fn stop_watching<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.file_watcher.unwatch_directory(path)
    }
    
    pub async fn process_events<F>(&mut self, mut event_handler: F) -> Result<()>
    where
        F: FnMut(WatcherEvent, Option<Document>) -> Result<()>,
    {
        while let Some(event) = self.event_receiver.recv().await {
            // Process the event to get a document if applicable
            let document = match &event {
                WatcherEvent::Created(path) | WatcherEvent::Modified(path) => {
                    self.process_file(path).await.unwrap_or(None)
                }
                WatcherEvent::Renamed { new_path, .. } => {
                    self.process_file(new_path).await.unwrap_or(None)
                }
                WatcherEvent::Deleted(_) => None,
            };
            
            // Call the event handler
            event_handler(event, document)?;
        }
        
        Ok(())
    }
    
    pub async fn find_duplicates<P: AsRef<Path>>(&mut self, root_path: P) -> Result<Vec<Vec<PathBuf>>> {
        let file_paths = self.metadata_extractor.traverse_directory(root_path)?;
        self.hasher.find_duplicates(file_paths).await
    }
    
    pub fn get_file_count_estimate<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        self.metadata_extractor.get_file_count_estimate(path)
    }
    
    fn build_metadata_map(
        &self,
        metadata: &ExtractedMetadata,
        classification: &FileClassificationResult,
    ) -> HashMap<String, String> {
        let mut map = HashMap::new();
        
        // Basic metadata
        map.insert("permissions".to_string(), metadata.basic.permissions.clone());
        map.insert("is_symlink".to_string(), metadata.basic.is_symlink.to_string());
        map.insert("is_hidden".to_string(), metadata.basic.is_hidden.to_string());
        
        if let Some(device_id) = metadata.basic.device_id {
            map.insert("device_id".to_string(), device_id.to_string());
        }
        
        if let Some(inode) = metadata.basic.inode {
            map.insert("inode".to_string(), inode.to_string());
        }
        
        // Extended metadata
        for (key, value) in &metadata.extended {
            map.insert(key.clone(), value.clone());
        }
        
        // File classification
        map.insert("is_text".to_string(), classification.is_text.to_string());
        map.insert("should_embed".to_string(), classification.should_embed.to_string());
        map.insert("detection_confidence".to_string(), classification.confidence.to_string());
        
        if let Some(ref mime_type) = classification.mime_type {
            map.insert("mime_type".to_string(), mime_type.clone());
        }
        
        // EXIF data (if available)
        if let Some(ref exif) = metadata.exif {
            for (key, value) in exif {
                map.insert(format!("exif_{}", key), value.clone());
            }
        }
        
        // ID3 data (if available)
        if let Some(ref id3) = metadata.id3 {
            for (key, value) in id3 {
                map.insert(format!("id3_{}", key), value.clone());
            }
        }
        
        map
    }
}

impl Default for FileProcessor {
    fn default() -> Self {
        Self::new().expect("Failed to create FileProcessor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{tempdir, NamedTempFile};
    use std::fs;
    use std::io::Write;
    
    #[tokio::test]
    async fn test_file_processing() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "This is a test document for processing.").unwrap();
        temp_file.flush().unwrap();
        
        let mut processor = FileProcessor::new().unwrap();
        let document = processor.process_file(temp_file.path()).await.unwrap();
        
        assert!(document.is_some());
        let doc = document.unwrap();
        assert!(!doc.content_hash.is_empty());
        assert!(matches!(doc.file_type, FileType::Text));
        assert!(doc.file_size > 0);
        assert!(!doc.metadata.is_empty());
    }
    
    #[tokio::test]
    async fn test_directory_scanning() {
        let temp_dir = tempdir().unwrap();
        
        // Create test files
        let file1 = temp_dir.path().join("document1.txt");
        let file2 = temp_dir.path().join("document2.md");
        let file3 = temp_dir.path().join("image.jpg");
        
        fs::write(&file1, "Content of document 1").unwrap();
        fs::write(&file2, "# Markdown Document\n\nSome content.").unwrap();
        fs::write(&file3, "fake jpeg content").unwrap(); // Not a real JPEG, but will be detected by extension
        
        let mut processor = FileProcessor::new().unwrap();
        let documents = processor.scan_directory(temp_dir.path()).await.unwrap();
        
        // Should find the text files
        assert!(documents.len() >= 2); // At least the text and markdown files
        
        // Verify we have the expected file types
        let file_types: Vec<_> = documents.iter().map(|d| &d.file_type).collect();
        assert!(file_types.contains(&&FileType::Text));
        assert!(file_types.contains(&&FileType::Markdown));
    }
    
    #[tokio::test]
    async fn test_duplicate_detection() {
        let temp_dir = tempdir().unwrap();
        
        // Create identical files
        let content = "Identical content for duplicate detection";
        let file1 = temp_dir.path().join("file1.txt");
        let file2 = temp_dir.path().join("file2.txt");
        let file3 = temp_dir.path().join("different.txt");
        
        fs::write(&file1, content).unwrap();
        fs::write(&file2, content).unwrap();
        fs::write(&file3, "Different content").unwrap();
        
        let mut processor = FileProcessor::new().unwrap();
        let duplicates = processor.find_duplicates(temp_dir.path()).await.unwrap();
        
        // Should find one group of duplicates
        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0].len(), 2);
        assert!(duplicates[0].contains(&file1));
        assert!(duplicates[0].contains(&file2));
    }
}