use super::{CacheConfig, CacheEntry, CacheStats};
use crate::error::{AppError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use lru::LruCache;
use std::num::NonZeroUsize;
use sha2::{Sha256, Digest};

/// Embedding cache entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCacheEntry {
    pub embedding: Vec<f32>,
    pub model_name: String,
    pub content_hash: String,
    pub dimension: usize,
    pub generation_time_ms: u64,
    pub confidence_score: f64,
}

/// Embedding cache with content-aware invalidation
pub struct EmbeddingCache {
    cache: Arc<RwLock<LruCache<String, CacheEntry<EmbeddingCacheEntry>>>>,
    content_hash_index: Arc<RwLock<HashMap<String, Vec<String>>>>, // content_hash -> cache_keys
    model_index: Arc<RwLock<HashMap<String, Vec<String>>>>, // model_name -> cache_keys
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
}

impl EmbeddingCache {
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let cache_size = NonZeroUsize::new(config.max_entries / 2).unwrap_or(NonZeroUsize::new(2000).unwrap());
        let cache = Arc::new(RwLock::new(LruCache::new(cache_size)));
        
        Ok(Self {
            cache,
            content_hash_index: Arc::new(RwLock::new(HashMap::new())),
            model_index: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats::new())),
        })
    }

    /// Cache an embedding with content and model awareness
    pub async fn cache_embedding(
        &self,
        content: &str,
        model_name: &str,
        embedding: Vec<f32>,
        generation_time_ms: u64,
        confidence_score: f64,
    ) -> Result<()> {
        let content_hash = self.calculate_content_hash(content);
        let cache_key = self.generate_cache_key(&content_hash, model_name);
        
        let embedding_entry = EmbeddingCacheEntry {
            embedding: embedding.clone(),
            model_name: model_name.to_string(),
            content_hash: content_hash.clone(),
            dimension: embedding.len(),
            generation_time_ms,
            confidence_score,
        };

        let ttl = self.determine_ttl(model_name, confidence_score);
        let size_estimate = self.estimate_size(&embedding_entry);
        let cache_entry = CacheEntry::new(embedding_entry, ttl, size_estimate);

        // Store in main cache
        self.cache.write().await.put(cache_key.clone(), cache_entry);

        // Update indexes
        self.content_hash_index.write().await
            .entry(content_hash)
            .or_insert_with(Vec::new)
            .push(cache_key.clone());

        self.model_index.write().await
            .entry(model_name.to_string())
            .or_insert_with(Vec::new)
            .push(cache_key);

        // Update stats
        let mut stats = self.stats.write().await;
        stats.memory_usage += size_estimate;
        stats.entries = self.cache.read().await.len();

        Ok(())
    }

    /// Get cached embedding
    pub async fn get_embedding(
        &self,
        content: &str,
        model_name: &str,
    ) -> Option<EmbeddingCacheEntry> {
        let content_hash = self.calculate_content_hash(content);
        let cache_key = self.generate_cache_key(&content_hash, model_name);
        
        if let Some(entry) = self.cache.write().await.get(&cache_key) {
            if !entry.is_expired() {
                self.stats.write().await.record_hit();
                return Some(entry.data.clone());
            } else {
                // Remove expired entry and update indexes
                self.remove_from_indexes(&cache_key, &entry.data.content_hash, &entry.data.model_name).await;
                self.cache.write().await.pop(&cache_key);
                self.stats.write().await.record_eviction();
            }
        }

        self.stats.write().await.record_miss();
        None
    }

    /// Get similar embeddings based on content similarity
    pub async fn get_similar_embeddings(
        &self,
        content: &str,
        model_name: &str,
        similarity_threshold: f64,
        max_results: usize,
    ) -> Vec<(EmbeddingCacheEntry, f64)> {
        let target_hash = self.calculate_content_hash(content);
        let mut results = Vec::new();
        
        let cache = self.cache.read().await;
        for (_, entry) in cache.iter() {
            if entry.data.model_name == model_name && !entry.is_expired() {
                let similarity = self.calculate_content_similarity(&target_hash, &entry.data.content_hash);
                if similarity >= similarity_threshold {
                    results.push((entry.data.clone(), similarity));
                }
            }
        }
        
        // Sort by similarity and limit results
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(max_results);
        
        results
    }

    /// Invalidate embeddings for a specific model
    pub async fn invalidate_model(&self, model_name: &str) -> usize {
        let mut invalidated = 0;
        
        if let Some(cache_keys) = self.model_index.write().await.remove(model_name) {
            let mut cache = self.cache.write().await;
            
            for cache_key in cache_keys {
                if let Some(entry) = cache.pop(&cache_key) {
                    // Remove from content hash index
                    if let Some(content_keys) = self.content_hash_index.write().await.get_mut(&entry.data.content_hash) {
                        content_keys.retain(|k| k != &cache_key);
                    }
                    invalidated += 1;
                }
            }
        }

        let mut stats = self.stats.write().await;
        stats.evictions += invalidated as u64;
        stats.entries = self.cache.read().await.len();

        invalidated
    }

    /// Invalidate embeddings by content hash
    pub async fn invalidate_content(&self, content: &str) -> usize {
        let content_hash = self.calculate_content_hash(content);
        let mut invalidated = 0;
        
        if let Some(cache_keys) = self.content_hash_index.write().await.remove(&content_hash) {
            let mut cache = self.cache.write().await;
            
            for cache_key in cache_keys {
                if let Some(entry) = cache.pop(&cache_key) {
                    // Remove from model index
                    if let Some(model_keys) = self.model_index.write().await.get_mut(&entry.data.model_name) {
                        model_keys.retain(|k| k != &cache_key);
                    }
                    invalidated += 1;
                }
            }
        }

        let mut stats = self.stats.write().await;
        stats.evictions += invalidated as u64;
        stats.entries = self.cache.read().await.len();

        invalidated
    }

    /// Get embeddings for a specific model
    pub async fn get_model_embeddings(&self, model_name: &str) -> Vec<EmbeddingCacheEntry> {
        let mut results = Vec::new();
        let cache = self.cache.read().await;
        
        if let Some(cache_keys) = self.model_index.read().await.get(model_name) {
            for cache_key in cache_keys {
                if let Some(entry) = cache.peek(cache_key) {
                    if !entry.is_expired() && entry.data.model_name == model_name {
                        results.push(entry.data.clone());
                    }
                }
            }
        }
        
        results
    }

    /// Batch cache multiple embeddings
    pub async fn cache_embeddings_batch(
        &self,
        embeddings: Vec<(String, String, Vec<f32>, u64, f64)>, // (content, model, embedding, time, confidence)
    ) -> Result<usize> {
        let mut cached_count = 0;
        
        for (content, model_name, embedding, generation_time_ms, confidence_score) in embeddings {
            if self.cache_embedding(&content, &model_name, embedding, generation_time_ms, confidence_score).await.is_ok() {
                cached_count += 1;
            }
        }
        
        Ok(cached_count)
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let mut stats = self.stats.read().await.clone();
        stats.entries = self.cache.read().await.len();
        
        // Calculate memory usage
        let cache = self.cache.read().await;
        stats.memory_usage = cache.iter().map(|(_, entry)| entry.size_estimate).sum();
        
        // Calculate age statistics
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

    /// Clear all cached embeddings
    pub async fn clear(&self) -> Result<()> {
        self.cache.write().await.clear();
        self.content_hash_index.write().await.clear();
        self.model_index.write().await.clear();
        *self.stats.write().await = CacheStats::new();
        Ok(())
    }

    /// Get memory usage by model
    pub async fn get_memory_usage_by_model(&self) -> HashMap<String, usize> {
        let mut usage = HashMap::new();
        let cache = self.cache.read().await;
        
        for (_, entry) in cache.iter() {
            let model = &entry.data.model_name;
            *usage.entry(model.clone()).or_insert(0) += entry.size_estimate;
        }
        
        usage
    }

    /// Optimize cache by removing low-confidence embeddings
    pub async fn optimize_by_confidence(&self, min_confidence: f64) -> usize {
        let mut cache = self.cache.write().await;
        let mut removed = 0;
        
        let keys_to_remove: Vec<String> = cache
            .iter()
            .filter(|(_, entry)| entry.data.confidence_score < min_confidence)
            .map(|(key, _)| key.clone())
            .collect();

        for key in keys_to_remove {
            if let Some(entry) = cache.pop(&key) {
                self.remove_from_indexes(&key, &entry.data.content_hash, &entry.data.model_name).await;
                removed += 1;
            }
        }

        let mut stats = self.stats.write().await;
        stats.evictions += removed as u64;
        stats.entries = cache.len();

        removed
    }

    // Private helper methods
    fn calculate_content_hash(&self, content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.trim().to_lowercase().as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn generate_cache_key(&self, content_hash: &str, model_name: &str) -> String {
        format!("{}:{}", model_name, content_hash)
    }

    fn calculate_content_similarity(&self, hash1: &str, hash2: &str) -> f64 {
        // Simple similarity based on hash prefix matching
        // In a real implementation, you might store actual content for semantic similarity
        let common_prefix = hash1.chars()
            .zip(hash2.chars())
            .take_while(|(a, b)| a == b)
            .count();
        
        common_prefix as f64 / hash1.len().max(hash2.len()) as f64
    }

    fn determine_ttl(&self, _model_name: &str, confidence_score: f64) -> Duration {
        let base_ttl = self.config.default_ttl;
        
        // High-confidence embeddings get longer TTL
        if confidence_score > 0.9 {
            base_ttl.mul_f64(2.0)
        } else if confidence_score > 0.7 {
            base_ttl.mul_f64(1.5)
        } else if confidence_score < 0.3 {
            base_ttl.mul_f64(0.5)
        } else {
            base_ttl
        }
    }

    fn estimate_size(&self, entry: &EmbeddingCacheEntry) -> usize {
        let base_size = std::mem::size_of::<EmbeddingCacheEntry>();
        let embedding_size = entry.embedding.len() * std::mem::size_of::<f32>();
        let string_sizes = entry.model_name.len() + entry.content_hash.len();
        
        base_size + embedding_size + string_sizes
    }

    async fn remove_from_indexes(&self, cache_key: &str, content_hash: &str, model_name: &str) {
        // Remove from content hash index
        if let Some(content_keys) = self.content_hash_index.write().await.get_mut(content_hash) {
            content_keys.retain(|k| k != cache_key);
            if content_keys.is_empty() {
                self.content_hash_index.write().await.remove(content_hash);
            }
        }

        // Remove from model index
        if let Some(model_keys) = self.model_index.write().await.get_mut(model_name) {
            model_keys.retain(|k| k != cache_key);
            if model_keys.is_empty() {
                self.model_index.write().await.remove(model_name);
            }
        }
    }
}