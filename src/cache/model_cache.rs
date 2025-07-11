use super::{CacheConfig, CacheEntry, CacheStats};
use crate::error::{AppError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use lru::LruCache;
use std::num::NonZeroUsize;

/// Model cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCacheEntry {
    pub model_path: PathBuf,
    pub model_type: ModelType,
    pub model_size_bytes: u64,
    pub load_time_ms: u64,
    pub last_used: SystemTime,
    pub usage_count: u64,
    pub memory_footprint: usize,
    pub version: String,
    pub checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Embedding,
    Llm,
    Ocr,
    Audio,
    Vision,
}

/// Model file and metadata cache
pub struct ModelCache {
    cache: Arc<RwLock<LruCache<String, CacheEntry<ModelCacheEntry>>>>,
    model_references: Arc<RwLock<HashMap<String, u32>>>, // model_name -> reference_count
    memory_usage: Arc<RwLock<usize>>,
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
}

impl ModelCache {
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let cache_size = NonZeroUsize::new(config.max_entries / 8).unwrap_or(NonZeroUsize::new(100).unwrap());
        let cache = Arc::new(RwLock::new(LruCache::new(cache_size)));
        
        Ok(Self {
            cache,
            model_references: Arc::new(RwLock::new(HashMap::new())),
            memory_usage: Arc::new(RwLock::new(0)),
            config,
            stats: Arc::new(RwLock::new(CacheStats::new())),
        })
    }

    /// Cache model metadata
    pub async fn cache_model_metadata(
        &self,
        model_name: &str,
        model_path: PathBuf,
        model_type: ModelType,
        model_size_bytes: u64,
        load_time_ms: u64,
        memory_footprint: usize,
        version: String,
        checksum: String,
    ) -> Result<()> {
        let model_entry = ModelCacheEntry {
            model_path,
            model_type,
            model_size_bytes,
            load_time_ms,
            last_used: SystemTime::now(),
            usage_count: 1,
            memory_footprint,
            version,
            checksum,
        };

        let ttl = self.determine_ttl(&model_entry);
        let size_estimate = self.estimate_size(&model_entry);
        let cache_entry = CacheEntry::new(model_entry, ttl, size_estimate);

        // Store in cache
        self.cache.write().await.put(model_name.to_string(), cache_entry);

        // Update memory tracking
        *self.memory_usage.write().await += memory_footprint;

        // Initialize reference count
        self.model_references.write().await.insert(model_name.to_string(), 1);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.memory_usage += size_estimate;
        stats.entries = self.cache.read().await.len();

        Ok(())
    }

    /// Get cached model metadata
    pub async fn get_model_metadata(&self, model_name: &str) -> Option<ModelCacheEntry> {
        if let Some(entry) = self.cache.write().await.get(model_name) {
            if !entry.is_expired() {
                self.stats.write().await.record_hit();
                
                // Update usage statistics
                let mut updated_entry = entry.data.clone();
                updated_entry.last_used = SystemTime::now();
                updated_entry.usage_count += 1;
                
                // Update reference count
                if let Some(ref_count) = self.model_references.write().await.get_mut(model_name) {
                    *ref_count += 1;
                }
                
                return Some(updated_entry);
            } else {
                // Remove expired entry
                self.remove_model_internal(model_name).await;
                self.stats.write().await.record_eviction();
            }
        }

        self.stats.write().await.record_miss();
        None
    }

    /// Add reference to a model (for tracking active usage)
    pub async fn add_model_reference(&self, model_name: &str) -> bool {
        if let Some(ref_count) = self.model_references.write().await.get_mut(model_name) {
            *ref_count += 1;
            true
        } else {
            false
        }
    }

    /// Remove reference to a model
    pub async fn remove_model_reference(&self, model_name: &str) -> bool {
        if let Some(ref_count) = self.model_references.write().await.get_mut(model_name) {
            *ref_count = ref_count.saturating_sub(1);
            
            // If no references remain, consider for eviction
            if *ref_count == 0 {
                self.consider_eviction(model_name).await;
            }
            true
        } else {
            false
        }
    }

    /// Get models by type
    pub async fn get_models_by_type(&self, model_type: &ModelType) -> Vec<(String, ModelCacheEntry)> {
        let cache = self.cache.read().await;
        let mut results = Vec::new();
        
        for (model_name, entry) in cache.iter() {
            if !entry.is_expired() && std::mem::discriminant(&entry.data.model_type) == std::mem::discriminant(model_type) {
                results.push((model_name.clone(), entry.data.clone()));
            }
        }
        
        // Sort by usage frequency
        results.sort_by(|a, b| b.1.usage_count.cmp(&a.1.usage_count));
        results
    }

    /// Get memory usage by model type
    pub async fn get_memory_usage_by_type(&self) -> HashMap<String, usize> {
        let cache = self.cache.read().await;
        let mut usage = HashMap::new();
        
        for (_, entry) in cache.iter() {
            if !entry.is_expired() {
                let type_name = format!("{:?}", entry.data.model_type);
                *usage.entry(type_name).or_insert(0) += entry.data.memory_footprint;
            }
        }
        
        usage
    }

    /// Get least recently used models
    pub async fn get_lru_models(&self, count: usize) -> Vec<(String, ModelCacheEntry)> {
        let cache = self.cache.read().await;
        let mut models: Vec<_> = cache
            .iter()
            .filter(|(_, entry)| !entry.is_expired())
            .map(|(name, entry)| (name.clone(), entry.data.clone()))
            .collect();
        
        models.sort_by(|a, b| a.1.last_used.cmp(&b.1.last_used));
        models.truncate(count);
        models
    }

    /// Get most memory-intensive models
    pub async fn get_memory_intensive_models(&self, count: usize) -> Vec<(String, ModelCacheEntry)> {
        let cache = self.cache.read().await;
        let mut models: Vec<_> = cache
            .iter()
            .filter(|(_, entry)| !entry.is_expired())
            .map(|(name, entry)| (name.clone(), entry.data.clone()))
            .collect();
        
        models.sort_by(|a, b| b.1.memory_footprint.cmp(&a.1.memory_footprint));
        models.truncate(count);
        models
    }

    /// Clean unused models
    pub async fn clean_unused_models(&self) -> usize {
        let references = self.model_references.read().await;
        let unused_models: Vec<String> = references
            .iter()
            .filter(|(_, &count)| count == 0)
            .map(|(name, _)| name.clone())
            .collect();
        
        drop(references);
        
        let mut cleaned = 0;
        for model_name in unused_models {
            if self.remove_model(&model_name).await {
                cleaned += 1;
            }
        }
        
        cleaned
    }

    /// Remove model from cache
    pub async fn remove_model(&self, model_name: &str) -> bool {
        if self.remove_model_internal(model_name).await {
            self.model_references.write().await.remove(model_name);
            true
        } else {
            false
        }
    }

    /// Check if model is actively referenced
    pub async fn is_model_referenced(&self, model_name: &str) -> bool {
        self.model_references.read().await
            .get(model_name)
            .map(|&count| count > 0)
            .unwrap_or(false)
    }

    /// Get total memory usage
    pub async fn get_total_memory_usage(&self) -> usize {
        *self.memory_usage.read().await
    }

    /// Optimize cache by removing old or unused models
    pub async fn optimize_cache(&self, target_memory_mb: usize) -> usize {
        let target_bytes = target_memory_mb * 1024 * 1024;
        let current_usage = self.get_total_memory_usage().await;
        
        if current_usage <= target_bytes {
            return 0; // No optimization needed
        }
        
        let bytes_to_free = current_usage - target_bytes;
        let mut freed_bytes = 0;
        let mut removed_count = 0;
        
        // First, remove unused models
        let unused_models = self.get_unused_models().await;
        for (model_name, entry) in unused_models {
            if freed_bytes >= bytes_to_free {
                break;
            }
            
            if self.remove_model(&model_name).await {
                freed_bytes += entry.memory_footprint;
                removed_count += 1;
            }
        }
        
        // If still over limit, remove LRU models
        if freed_bytes < bytes_to_free {
            let lru_models = self.get_lru_models(50).await;
            for (model_name, entry) in lru_models {
                if freed_bytes >= bytes_to_free {
                    break;
                }
                
                if self.remove_model(&model_name).await {
                    freed_bytes += entry.memory_footprint;
                    removed_count += 1;
                }
            }
        }
        
        removed_count
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let mut stats = self.stats.read().await.clone();
        stats.entries = self.cache.read().await.len();
        stats.memory_usage = *self.memory_usage.read().await;
        
        // Calculate age statistics
        let cache = self.cache.read().await;
        let now = SystemTime::now();
        let ages: Vec<Duration> = cache
            .iter()
            .map(|(_, entry)| now.duration_since(entry.created_at).unwrap_or(Duration::ZERO))
            .collect();

        if !ages.is_empty() {
            stats.average_age = Duration::from_millis(
                ages.iter().map(|d| d.as_millis() as u64).sum::<u64>() / ages.len() as u64
            );
            stats.oldest_entry = ages.iter().max().cloned();
            stats.newest_entry = ages.iter().min().cloned();
        }

        stats
    }

    /// Clear all cached models
    pub async fn clear(&self) -> Result<()> {
        self.cache.write().await.clear();
        self.model_references.write().await.clear();
        *self.memory_usage.write().await = 0;
        *self.stats.write().await = CacheStats::new();
        Ok(())
    }

    // Private helper methods
    async fn remove_model_internal(&self, model_name: &str) -> bool {
        if let Some(entry) = self.cache.write().await.pop(model_name) {
            // Update memory usage
            let mut memory_usage = self.memory_usage.write().await;
            *memory_usage = memory_usage.saturating_sub(entry.data.memory_footprint);
            
            // Update stats
            let mut stats = self.stats.write().await;
            stats.evictions += 1;
            stats.entries = self.cache.read().await.len();
            stats.memory_usage = *memory_usage;
            
            true
        } else {
            false
        }
    }

    async fn consider_eviction(&self, model_name: &str) {
        // Only consider eviction if model hasn't been used recently
        if let Some(entry) = self.cache.read().await.peek(model_name) {
            let age = entry.last_accessed.elapsed().unwrap_or(Duration::ZERO);
            if age > Duration::from_secs(300) { // 5 minutes
                self.remove_model_internal(model_name).await;
            }
        }
    }

    async fn get_unused_models(&self) -> Vec<(String, ModelCacheEntry)> {
        let references = self.model_references.read().await;
        let cache = self.cache.read().await;
        let mut unused = Vec::new();
        
        for (model_name, &ref_count) in references.iter() {
            if ref_count == 0 {
                if let Some(entry) = cache.peek(model_name) {
                    if !entry.is_expired() {
                        unused.push((model_name.clone(), entry.data.clone()));
                    }
                }
            }
        }
        
        // Sort by last used time (oldest first)
        unused.sort_by(|a, b| a.1.last_used.cmp(&b.1.last_used));
        unused
    }

    fn determine_ttl(&self, entry: &ModelCacheEntry) -> Duration {
        let base_ttl = self.config.default_ttl;
        
        // Frequently used models get longer TTL
        if entry.usage_count > 10 {
            base_ttl.mul_f64(2.0)
        } else if entry.usage_count > 5 {
            base_ttl.mul_f64(1.5)
        } else {
            base_ttl
        }
    }

    fn estimate_size(&self, entry: &ModelCacheEntry) -> usize {
        let base_size = std::mem::size_of::<ModelCacheEntry>();
        let path_size = entry.model_path.to_string_lossy().len();
        let version_size = entry.version.len();
        let checksum_size = entry.checksum.len();
        
        base_size + path_size + version_size + checksum_size
    }
}