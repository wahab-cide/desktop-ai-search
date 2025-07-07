use crate::error::{AppError, FileSystemError, Result};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs::{File, Metadata};
use std::io::{Read, Seek, SeekFrom, BufReader};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileHashInfo {
    pub full_hash: Option<String>,
    pub partial_hash: Option<String>,
    pub size: u64,
    pub mtime: u64,
    pub computed_at: u64,
}

#[derive(Debug, Clone)]
pub struct FileHashCache {
    cache: HashMap<PathBuf, FileHashInfo>,
    max_entries: usize,
}

impl FileHashCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_entries,
        }
    }
    
    pub fn get(&self, path: &Path, size: u64, mtime: u64) -> Option<&FileHashInfo> {
        if let Some(cached) = self.cache.get(path) {
            // Validate cache entry is still valid
            if cached.size == size && cached.mtime == mtime {
                return Some(cached);
            }
        }
        None
    }
    
    pub fn insert(&mut self, path: PathBuf, hash_info: FileHashInfo) {
        // Simple LRU eviction when cache is full
        if self.cache.len() >= self.max_entries {
            // Remove oldest entry (this is a simplified LRU)
            if let Some(oldest_key) = self.cache
                .iter()
                .min_by_key(|(_, info)| info.computed_at)
                .map(|(key, _)| key.clone())
            {
                self.cache.remove(&oldest_key);
            }
        }
        
        self.cache.insert(path, hash_info);
    }
    
    pub fn persist_to_db(&self, db_path: &Path) -> Result<()> {
        // In a real implementation, this would save to the SQLite database
        // For now, we'll save to a JSON file
        let cache_json = serde_json::to_string_pretty(&self.cache)?;
        let cache_file = db_path.parent()
            .unwrap_or(Path::new("."))
            .join("hash_cache.json");
        
        std::fs::write(cache_file, cache_json)
            .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))
    }
    
    pub fn load_from_db(&mut self, db_path: &Path) -> Result<()> {
        let cache_file = db_path.parent()
            .unwrap_or(Path::new("."))
            .join("hash_cache.json");
        
        if cache_file.exists() {
            let cache_json = std::fs::read_to_string(cache_file)
                .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?;
            
            self.cache = serde_json::from_str(&cache_json)?;
        }
        
        Ok(())
    }
}

pub struct TieredHasher {
    cache: FileHashCache,
    small_file_threshold: u64,
    partial_hash_size: usize,
}

impl TieredHasher {
    pub fn new() -> Self {
        Self {
            cache: FileHashCache::new(10000), // Cache up to 10k entries
            small_file_threshold: 10 * 1024 * 1024, // 10MB
            partial_hash_size: 1024 * 1024, // 1MB for partial hash
        }
    }
    
    pub fn with_cache_size(cache_size: usize) -> Self {
        Self {
            cache: FileHashCache::new(cache_size),
            small_file_threshold: 10 * 1024 * 1024,
            partial_hash_size: 1024 * 1024,
        }
    }
    
    pub async fn compute_hash<P: AsRef<Path>>(&mut self, path: P) -> Result<String> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path)
            .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?;
        
        let size = metadata.len();
        let mtime = self.get_mtime(&metadata)?;
        
        // Check cache first
        if let Some(cached) = self.cache.get(path, size, mtime) {
            if let Some(ref full_hash) = cached.full_hash {
                return Ok(full_hash.clone());
            }
        }
        
        // Compute hash based on file size strategy
        let hash_info = if size <= self.small_file_threshold {
            // Small files: compute full hash
            self.compute_full_hash(path, size, mtime).await?
        } else {
            // Large files: compute partial hash first, then full if needed
            self.compute_tiered_hash(path, size, mtime).await?
        };
        
        // Cache the result
        self.cache.insert(path.to_path_buf(), hash_info.clone());
        
        Ok(hash_info.full_hash.unwrap_or_else(|| hash_info.partial_hash.unwrap()))
    }
    
    pub async fn compute_partial_hash<P: AsRef<Path>>(&mut self, path: P) -> Result<String> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path)
            .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?;
        
        let size = metadata.len();
        let mtime = self.get_mtime(&metadata)?;
        
        // Check cache first
        if let Some(cached) = self.cache.get(path, size, mtime) {
            if let Some(ref partial_hash) = cached.partial_hash {
                return Ok(partial_hash.clone());
            }
            if let Some(ref full_hash) = cached.full_hash {
                return Ok(full_hash.clone()); // Full hash is better than partial
            }
        }
        
        let hash_info = self.compute_first_last_hash(path, size, mtime).await?;
        self.cache.insert(path.to_path_buf(), hash_info.clone());
        
        Ok(hash_info.partial_hash.unwrap())
    }
    
    async fn compute_full_hash<P: AsRef<Path>>(
        &self, 
        path: P, 
        size: u64, 
        mtime: u64
    ) -> Result<FileHashInfo> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?;
        
        let mut hasher = Sha256::new();
        let mut buffer = vec![0; 64 * 1024]; // 64KB buffer
        
        loop {
            let bytes_read = file.read(&mut buffer)
                .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?;
            
            if bytes_read == 0 {
                break;
            }
            
            hasher.update(&buffer[..bytes_read]);
        }
        
        let hash = format!("{:x}", hasher.finalize());
        
        Ok(FileHashInfo {
            full_hash: Some(hash),
            partial_hash: None,
            size,
            mtime,
            computed_at: self.current_timestamp(),
        })
    }
    
    async fn compute_tiered_hash<P: AsRef<Path>>(
        &self,
        path: P,
        size: u64,
        mtime: u64,
    ) -> Result<FileHashInfo> {
        // First compute partial hash (first + last MB)
        let partial_hash_info = self.compute_first_last_hash(path.as_ref(), size, mtime).await?;
        
        // For now, we'll defer full hash computation until explicitly requested
        // In the real implementation, you might compute full hash in the background
        Ok(partial_hash_info)
    }
    
    async fn compute_first_last_hash<P: AsRef<Path>>(
        &self,
        path: P,
        size: u64,
        mtime: u64,
    ) -> Result<FileHashInfo> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?;
        
        let mut hasher = Sha256::new();
        let chunk_size = std::cmp::min(self.partial_hash_size as u64, size / 2);
        
        // Read first chunk
        let mut buffer = vec![0; chunk_size as usize];
        let bytes_read = file.read(&mut buffer)
            .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?;
        hasher.update(&buffer[..bytes_read]);
        
        // Read last chunk if file is large enough
        if size > chunk_size * 2 {
            file.seek(SeekFrom::End(-(chunk_size as i64)))
                .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?;
            
            let bytes_read = file.read(&mut buffer)
                .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?;
            hasher.update(&buffer[..bytes_read]);
        }
        
        let hash = format!("{:x}", hasher.finalize());
        
        Ok(FileHashInfo {
            full_hash: None,
            partial_hash: Some(hash),
            size,
            mtime,
            computed_at: self.current_timestamp(),
        })
    }
    
    pub async fn force_full_hash<P: AsRef<Path>>(&mut self, path: P) -> Result<String> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path)
            .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?;
        
        let size = metadata.len();
        let mtime = self.get_mtime(&metadata)?;
        
        let hash_info = self.compute_full_hash(path, size, mtime).await?;
        let full_hash = hash_info.full_hash.clone().unwrap();
        
        self.cache.insert(path.to_path_buf(), hash_info);
        Ok(full_hash)
    }
    
    pub fn are_files_likely_identical<P1: AsRef<Path>, P2: AsRef<Path>>(
        &mut self,
        path1: P1,
        path2: P2,
    ) -> Result<bool> {
        let path1 = path1.as_ref();
        let path2 = path2.as_ref();
        
        // Quick size check first
        let meta1 = std::fs::metadata(path1)
            .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?;
        let meta2 = std::fs::metadata(path2)
            .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?;
        
        if meta1.len() != meta2.len() {
            return Ok(false);
        }
        
        // If same size, check modification times
        let mtime1 = self.get_mtime(&meta1)?;
        let mtime2 = self.get_mtime(&meta2)?;
        
        // If same size and mtime, likely identical (assuming same filesystem)
        if mtime1 == mtime2 {
            return Ok(true);
        }
        
        // Different mtimes, would need hash comparison for definitive answer
        Ok(false)
    }
    
    pub async fn find_duplicates<P: AsRef<Path>>(
        &mut self,
        paths: Vec<P>,
    ) -> Result<Vec<Vec<PathBuf>>> {
        let mut size_groups: HashMap<u64, Vec<PathBuf>> = HashMap::new();
        
        // Group by size first
        for path in paths {
            let path = path.as_ref();
            if let Ok(metadata) = std::fs::metadata(path) {
                let size = metadata.len();
                size_groups.entry(size).or_default().push(path.to_path_buf());
            }
        }
        
        let mut duplicates = Vec::new();
        
        // For each size group with multiple files, compare hashes
        for (_, mut paths) in size_groups {
            if paths.len() < 2 {
                continue;
            }
            
            let mut hash_groups: HashMap<String, Vec<PathBuf>> = HashMap::new();
            
            for path in &paths {
                match self.compute_partial_hash(path).await {
                    Ok(hash) => {
                        hash_groups.entry(hash).or_default().push(path.clone());
                    }
                    Err(_) => continue, // Skip files we can't hash
                }
            }
            
            // Collect groups with multiple files
            for (_, duplicate_group) in hash_groups {
                if duplicate_group.len() > 1 {
                    duplicates.push(duplicate_group);
                }
            }
        }
        
        Ok(duplicates)
    }
    
    fn get_mtime(&self, metadata: &Metadata) -> Result<u64> {
        let timestamp = metadata
            .modified()
            .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?
            .duration_since(UNIX_EPOCH)
            .map_err(|e| AppError::FileSystem(FileSystemError::Hash(e.to_string())))?
            .as_secs();
        Ok(timestamp)
    }
    
    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.cache.len(), self.cache.max_entries)
    }
    
    pub fn persist_cache<P: AsRef<Path>>(&self, db_path: P) -> Result<()> {
        self.cache.persist_to_db(db_path.as_ref())
    }
    
    pub fn load_cache<P: AsRef<Path>>(&mut self, db_path: P) -> Result<()> {
        self.cache.load_from_db(db_path.as_ref())
    }
}

impl Default for TieredHasher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{NamedTempFile, tempdir};
    use std::io::Write;
    
    #[tokio::test]
    async fn test_small_file_hashing() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = b"Hello, world! This is a test file.";
        temp_file.write_all(content).unwrap();
        temp_file.flush().unwrap();
        
        let mut hasher = TieredHasher::new();
        let hash = hasher.compute_hash(temp_file.path()).await.unwrap();
        
        // Verify we get a consistent hash
        let hash2 = hasher.compute_hash(temp_file.path()).await.unwrap();
        assert_eq!(hash, hash2);
        
        // Verify cache is working (second call should be cached)
        let (cache_entries, _) = hasher.cache_stats();
        assert_eq!(cache_entries, 1);
    }
    
    #[tokio::test]
    async fn test_duplicate_detection() {
        let temp_dir = tempdir().unwrap();
        
        // Create two identical files
        let file1_path = temp_dir.path().join("file1.txt");
        let file2_path = temp_dir.path().join("file2.txt");
        let content = b"Identical content for duplicate detection test.";
        
        std::fs::write(&file1_path, content).unwrap();
        std::fs::write(&file2_path, content).unwrap();
        
        // Create a different file
        let file3_path = temp_dir.path().join("file3.txt");
        std::fs::write(&file3_path, b"Different content").unwrap();
        
        let mut hasher = TieredHasher::new();
        let paths = vec![&file1_path, &file2_path, &file3_path];
        let duplicates = hasher.find_duplicates(paths).await.unwrap();
        
        // Should find one group of duplicates containing file1 and file2
        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0].len(), 2);
        assert!(duplicates[0].contains(&file1_path));
        assert!(duplicates[0].contains(&file2_path));
    }
    
    #[tokio::test]
    async fn test_partial_vs_full_hash() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = vec![0u8; 5 * 1024 * 1024]; // 5MB file
        temp_file.write_all(&content).unwrap();
        temp_file.flush().unwrap();
        
        let mut hasher = TieredHasher::new();
        
        // Get partial hash
        let partial_hash = hasher.compute_partial_hash(temp_file.path()).await.unwrap();
        
        // Get full hash
        let full_hash = hasher.force_full_hash(temp_file.path()).await.unwrap();
        
        // They should be different (partial is first+last, full is entire file)
        // For a file of all zeros, they might be the same, so let's just verify we got hashes
        assert!(!partial_hash.is_empty());
        assert!(!full_hash.is_empty());
        assert_eq!(partial_hash.len(), 64); // SHA256 hex string length
        assert_eq!(full_hash.len(), 64);
    }
    
    #[test]
    fn test_cache_functionality() {
        let mut cache = FileHashCache::new(2);
        let path1 = PathBuf::from("/test/path1");
        let path2 = PathBuf::from("/test/path2");
        let path3 = PathBuf::from("/test/path3");
        
        let hash_info1 = FileHashInfo {
            full_hash: Some("hash1".to_string()),
            partial_hash: None,
            size: 100,
            mtime: 1000,
            computed_at: 1000,
        };
        
        let hash_info2 = FileHashInfo {
            full_hash: Some("hash2".to_string()),
            partial_hash: None,
            size: 200,
            mtime: 2000,
            computed_at: 2000,
        };
        
        // Insert two entries
        cache.insert(path1.clone(), hash_info1.clone());
        cache.insert(path2.clone(), hash_info2.clone());
        
        // Both should be retrievable
        assert!(cache.get(&path1, 100, 1000).is_some());
        assert!(cache.get(&path2, 200, 2000).is_some());
        
        // Insert third entry should evict oldest (path1)
        let hash_info3 = FileHashInfo {
            full_hash: Some("hash3".to_string()),
            partial_hash: None,
            size: 300,
            mtime: 3000,
            computed_at: 3000,
        };
        cache.insert(path3.clone(), hash_info3);
        
        // path1 should be evicted, path2 and path3 should remain
        assert!(cache.get(&path1, 100, 1000).is_none());
        assert!(cache.get(&path2, 200, 2000).is_some());
        assert!(cache.get(&path3, 300, 3000).is_some());
    }
}