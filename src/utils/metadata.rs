use crate::error::{AppError, FileSystemError, Result};
use crate::models::FileType;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::fs::MetadataExt; // For Unix-specific metadata
use walkdir::{WalkDir, DirEntry};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedMetadata {
    pub basic: BasicMetadata,
    pub extended: HashMap<String, String>,
    pub exif: Option<HashMap<String, String>>,
    pub id3: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicMetadata {
    pub size: u64,
    pub created: DateTime<Utc>,
    pub modified: DateTime<Utc>,
    pub accessed: DateTime<Utc>,
    pub permissions: String,
    pub is_symlink: bool,
    pub is_hidden: bool,
    pub device_id: Option<u64>,
    pub inode: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TraversalOptions {
    pub max_depth: Option<usize>,
    pub follow_symlinks: bool,
    pub include_hidden: bool,
    pub exclude_patterns: Vec<String>,
    pub min_file_size: Option<u64>,
    pub max_file_size: Option<u64>,
}

impl Default for TraversalOptions {
    fn default() -> Self {
        Self {
            max_depth: None,
            follow_symlinks: false,
            include_hidden: false,
            exclude_patterns: vec![
                ".git".to_string(),
                ".svn".to_string(),
                "node_modules".to_string(),
                ".DS_Store".to_string(),
                "Thumbs.db".to_string(),
            ],
            min_file_size: None,
            max_file_size: None,
        }
    }
}

pub struct MetadataExtractor {
    visited_inodes: HashSet<(u64, u64)>, // (device_id, inode) for symlink loop detection
    options: TraversalOptions,
}

impl MetadataExtractor {
    pub fn new(options: TraversalOptions) -> Self {
        Self {
            visited_inodes: HashSet::new(),
            options,
        }
    }
    
    pub fn extract_metadata<P: AsRef<Path>>(&self, path: P) -> Result<ExtractedMetadata> {
        let path = path.as_ref();
        let metadata = fs::metadata(path)
            .map_err(|e| AppError::FileSystem(FileSystemError::Metadata(e.to_string())))?;
        
        let basic = self.extract_basic_metadata(path, &metadata)?;
        let extended = self.extract_extended_metadata(path, &metadata)?;
        
        // Extract format-specific metadata based on file type
        let exif = self.extract_exif_metadata(path)?;
        let id3 = self.extract_id3_metadata(path)?;
        
        Ok(ExtractedMetadata {
            basic,
            extended,
            exif,
            id3,
        })
    }
    
    fn extract_basic_metadata(&self, path: &Path, metadata: &fs::Metadata) -> Result<BasicMetadata> {
        let size = metadata.len();
        
        let created = metadata.created()
            .unwrap_or_else(|_| metadata.modified().unwrap_or(std::time::UNIX_EPOCH))
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let modified = metadata.modified()
            .unwrap_or(std::time::UNIX_EPOCH)
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let accessed = metadata.accessed()
            .unwrap_or_else(|_| metadata.modified().unwrap_or(std::time::UNIX_EPOCH))
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let permissions = format!("{:o}", metadata.permissions().mode() & 0o777);
        let is_symlink = metadata.file_type().is_symlink();
        let is_hidden = path.file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with('.'))
            .unwrap_or(false);
        
        // Unix-specific metadata
        let device_id = Some(metadata.dev());
        let inode = Some(metadata.ino());
        
        Ok(BasicMetadata {
            size,
            created: DateTime::from_timestamp(created as i64, 0).unwrap_or_default(),
            modified: DateTime::from_timestamp(modified as i64, 0).unwrap_or_default(),
            accessed: DateTime::from_timestamp(accessed as i64, 0).unwrap_or_default(),
            permissions,
            is_symlink,
            is_hidden,
            device_id,
            inode,
        })
    }
    
    fn extract_extended_metadata(&self, path: &Path, metadata: &fs::Metadata) -> Result<HashMap<String, String>> {
        let mut extended = HashMap::new();
        
        // File extension
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            extended.insert("extension".to_string(), extension.to_lowercase());
        }
        
        // File name without extension
        if let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) {
            extended.insert("filename_stem".to_string(), stem.to_string());
        }
        
        // Full filename
        if let Some(filename) = path.file_name().and_then(|name| name.to_str()) {
            extended.insert("filename".to_string(), filename.to_string());
        }
        
        // Parent directory
        if let Some(parent) = path.parent().and_then(|p| p.to_str()) {
            extended.insert("parent_directory".to_string(), parent.to_string());
        }
        
        // File type classification
        extended.insert("is_regular_file".to_string(), metadata.is_file().to_string());
        extended.insert("is_directory".to_string(), metadata.is_dir().to_string());
        extended.insert("is_symlink".to_string(), metadata.file_type().is_symlink().to_string());
        
        // Size classification
        let size_category = match metadata.len() {
            0 => "empty",
            1..=1024 => "tiny",
            1025..=10_240 => "small",
            10_241..=1_048_576 => "medium",
            1_048_577..=104_857_600 => "large",
            _ => "huge",
        };
        extended.insert("size_category".to_string(), size_category.to_string());
        
        Ok(extended)
    }
    
    fn extract_exif_metadata(&self, path: &Path) -> Result<Option<HashMap<String, String>>> {
        // Check if this is an image file that might have EXIF data
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            match extension.to_lowercase().as_str() {
                "jpg" | "jpeg" | "tiff" | "tif" => {
                    // In a real implementation, you would use an EXIF library like `exif`
                    // For now, we'll return a placeholder
                    let mut exif_data = HashMap::new();
                    exif_data.insert("format".to_string(), "EXIF_PLACEHOLDER".to_string());
                    exif_data.insert("extracted_with".to_string(), "placeholder".to_string());
                    return Ok(Some(exif_data));
                }
                _ => {}
            }
        }
        Ok(None)
    }
    
    fn extract_id3_metadata(&self, path: &Path) -> Result<Option<HashMap<String, String>>> {
        // Check if this is an audio file that might have ID3 tags
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            match extension.to_lowercase().as_str() {
                "mp3" | "m4a" | "flac" | "ogg" => {
                    // In a real implementation, you would use an ID3 library like `id3`
                    // For now, we'll return a placeholder
                    let mut id3_data = HashMap::new();
                    id3_data.insert("format".to_string(), "ID3_PLACEHOLDER".to_string());
                    id3_data.insert("extracted_with".to_string(), "placeholder".to_string());
                    return Ok(Some(id3_data));
                }
                _ => {}
            }
        }
        Ok(None)
    }
    
    pub fn traverse_directory<P: AsRef<Path>>(
        &mut self,
        root_path: P,
    ) -> Result<Vec<PathBuf>> {
        let root_path = root_path.as_ref();
        let mut file_paths = Vec::new();
        
        let mut walker = WalkDir::new(root_path);
        
        if let Some(max_depth) = self.options.max_depth {
            walker = walker.max_depth(max_depth);
        }
        
        if self.options.follow_symlinks {
            walker = walker.follow_links(true);
        }
        
        for entry in walker.into_iter() {
            let entry = entry.map_err(|e| {
                AppError::FileSystem(FileSystemError::Metadata(e.to_string()))
            })?;
            
            if self.should_process_entry(&entry)? {
                if entry.file_type().is_file() {
                    file_paths.push(entry.path().to_path_buf());
                }
            }
        }
        
        Ok(file_paths)
    }
    
    fn should_process_entry(&mut self, entry: &DirEntry) -> Result<bool> {
        let path = entry.path();
        let metadata = entry.metadata().map_err(|e| {
            AppError::FileSystem(FileSystemError::Metadata(e.to_string()))
        })?;
        
        // Check for symbolic link loops
        if metadata.file_type().is_symlink() && !self.options.follow_symlinks {
            return Ok(false);
        }
        
        if metadata.file_type().is_symlink() || metadata.nlink() > 1 {
            let device_id = metadata.dev();
            let inode = metadata.ino();
            let inode_key = (device_id, inode);
            
            if self.visited_inodes.contains(&inode_key) {
                // Already visited this inode, skip to prevent loops
                return Ok(false);
            }
            
            self.visited_inodes.insert(inode_key);
        }
        
        // Check hidden files
        if !self.options.include_hidden {
            if let Some(filename) = path.file_name().and_then(|name| name.to_str()) {
                if filename.starts_with('.') {
                    return Ok(false);
                }
            }
        }
        
        // Check exclude patterns
        let path_str = path.to_string_lossy();
        for pattern in &self.options.exclude_patterns {
            if path_str.contains(pattern) {
                return Ok(false);
            }
        }
        
        // Check file size limits (only for files)
        if metadata.is_file() {
            let size = metadata.len();
            
            if let Some(min_size) = self.options.min_file_size {
                if size < min_size {
                    return Ok(false);
                }
            }
            
            if let Some(max_size) = self.options.max_file_size {
                if size > max_size {
                    return Ok(false);
                }
            }
        }
        
        Ok(true)
    }
    
    pub fn resolve_symlink<P: AsRef<Path>>(&self, path: P) -> Result<PathBuf> {
        let path = path.as_ref();
        
        if !path.is_symlink() {
            return Ok(path.to_path_buf());
        }
        
        fs::canonicalize(path)
            .map_err(|e| AppError::FileSystem(FileSystemError::Metadata(e.to_string())))
    }
    
    pub fn get_symlink_target<P: AsRef<Path>>(&self, path: P) -> Result<Option<PathBuf>> {
        let path = path.as_ref();
        
        if !path.is_symlink() {
            return Ok(None);
        }
        
        match fs::read_link(path) {
            Ok(target) => Ok(Some(target)),
            Err(_) => Ok(None), // Broken symlink
        }
    }
    
    pub fn is_broken_symlink<P: AsRef<Path>>(&self, path: P) -> Result<bool> {
        let path = path.as_ref();
        
        if !path.is_symlink() {
            return Ok(false);
        }
        
        // Try to get metadata of the target
        match fs::metadata(path) {
            Ok(_) => Ok(false), // Symlink is valid
            Err(_) => Ok(true), // Symlink is broken
        }
    }
    
    pub fn get_file_count_estimate<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        let path = path.as_ref();
        
        if !path.is_dir() {
            return Ok(if path.is_file() { 1 } else { 0 });
        }
        
        let mut count = 0;
        let walker = WalkDir::new(path)
            .max_depth(self.options.max_depth.unwrap_or(10))
            .follow_links(self.options.follow_symlinks);
        
        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                count += 1;
                
                // For performance, cap the count estimation
                if count > 100_000 {
                    break;
                }
            }
        }
        
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{tempdir, NamedTempFile};
    use std::fs;
    use std::io::Write;
    
    #[test]
    fn test_basic_metadata_extraction() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Test content").unwrap();
        temp_file.flush().unwrap();
        
        let extractor = MetadataExtractor::new(TraversalOptions::default());
        let metadata = extractor.extract_metadata(temp_file.path()).unwrap();
        
        assert!(metadata.basic.size > 0);
        assert!(!metadata.basic.is_symlink);
        assert!(!metadata.basic.is_hidden);
        assert!(metadata.basic.device_id.is_some());
        assert!(metadata.basic.inode.is_some());
    }
    
    #[test]
    fn test_directory_traversal() {
        let temp_dir = tempdir().unwrap();
        
        // Create some test files
        let file1 = temp_dir.path().join("file1.txt");
        let file2 = temp_dir.path().join("file2.txt");
        let subdir = temp_dir.path().join("subdir");
        fs::create_dir(&subdir).unwrap();
        let file3 = subdir.join("file3.txt");
        
        fs::write(&file1, "content1").unwrap();
        fs::write(&file2, "content2").unwrap();
        fs::write(&file3, "content3").unwrap();
        
        let mut extractor = MetadataExtractor::new(TraversalOptions::default());
        let files = extractor.traverse_directory(temp_dir.path()).unwrap();
        
        assert_eq!(files.len(), 3);
        assert!(files.contains(&file1));
        assert!(files.contains(&file2));
        assert!(files.contains(&file3));
    }
    
    #[test]
    fn test_exclude_patterns() {
        let temp_dir = tempdir().unwrap();
        
        // Create files, including some that should be excluded
        let normal_file = temp_dir.path().join("normal.txt");
        let git_dir = temp_dir.path().join(".git");
        fs::create_dir(&git_dir).unwrap();
        let git_file = git_dir.join("config");
        
        fs::write(&normal_file, "content").unwrap();
        fs::write(&git_file, "git config").unwrap();
        
        let options = TraversalOptions {
            exclude_patterns: vec![".git".to_string()],
            ..Default::default()
        };
        
        let mut extractor = MetadataExtractor::new(options);
        let files = extractor.traverse_directory(temp_dir.path()).unwrap();
        
        assert_eq!(files.len(), 1);
        assert!(files.contains(&normal_file));
        assert!(!files.contains(&git_file));
    }
    
    #[test]
    fn test_file_size_limits() {
        let temp_dir = tempdir().unwrap();
        
        // Create files of different sizes
        let small_file = temp_dir.path().join("small.txt");
        let large_file = temp_dir.path().join("large.txt");
        
        fs::write(&small_file, "tiny").unwrap(); // 4 bytes
        fs::write(&large_file, "a".repeat(2000)).unwrap(); // 2000 bytes
        
        let options = TraversalOptions {
            min_file_size: Some(10),
            max_file_size: Some(1500),
            ..Default::default()
        };
        
        let mut extractor = MetadataExtractor::new(options);
        let files = extractor.traverse_directory(temp_dir.path()).unwrap();
        
        // Only files between 10 and 1500 bytes should be included
        assert_eq!(files.len(), 0); // Neither file meets the criteria
        
        let options2 = TraversalOptions {
            min_file_size: Some(1),
            max_file_size: Some(10),
            ..Default::default()
        };
        
        let mut extractor2 = MetadataExtractor::new(options2);
        let files2 = extractor2.traverse_directory(temp_dir.path()).unwrap();
        
        assert_eq!(files2.len(), 1);
        assert!(files2.contains(&small_file));
    }
    
    #[test]
    fn test_hidden_file_handling() {
        let temp_dir = tempdir().unwrap();
        
        let normal_file = temp_dir.path().join("normal.txt");
        let hidden_file = temp_dir.path().join(".hidden.txt");
        
        fs::write(&normal_file, "content").unwrap();
        fs::write(&hidden_file, "hidden content").unwrap();
        
        // Test without including hidden files
        let options = TraversalOptions {
            include_hidden: false,
            ..Default::default()
        };
        
        let mut extractor = MetadataExtractor::new(options);
        let files = extractor.traverse_directory(temp_dir.path()).unwrap();
        
        assert_eq!(files.len(), 1);
        assert!(files.contains(&normal_file));
        
        // Test with including hidden files
        let options2 = TraversalOptions {
            include_hidden: true,
            ..Default::default()
        };
        
        let mut extractor2 = MetadataExtractor::new(options2);
        let files2 = extractor2.traverse_directory(temp_dir.path()).unwrap();
        
        assert_eq!(files2.len(), 2);
        assert!(files2.contains(&normal_file));
        assert!(files2.contains(&hidden_file));
    }
}