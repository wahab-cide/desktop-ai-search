use super::{CacheConfig, CacheEntry, CacheStats};
use crate::database::operations::SearchResult;
use crate::error::{AppError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use lru::LruCache;
use std::num::NonZeroUsize;
use sha2::{Sha256, Digest};

/// Search-specific cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchCacheEntry {
    pub results: Vec<SearchResult>,
    pub query_hash: String,
    pub filters: HashMap<String, String>,
    pub total_results: usize,
    pub search_time_ms: u64,
    pub result_quality_score: f64,
}

/// Search result cache with query normalization and invalidation
pub struct SearchCache {
    cache: Arc<RwLock<LruCache<String, CacheEntry<SearchCacheEntry>>>>,
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
    query_patterns: Arc<RwLock<HashMap<String, QueryPattern>>>,
}

#[derive(Debug, Clone)]
struct QueryPattern {
    frequency: u32,
    last_used: SystemTime,
    average_results: f64,
    cache_hit_rate: f64,
}

impl SearchCache {
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let cache_size = NonZeroUsize::new(config.max_entries / 4).unwrap_or(NonZeroUsize::new(1000).unwrap());
        let cache = Arc::new(RwLock::new(LruCache::new(cache_size)));
        
        Ok(Self {
            cache,
            config,
            stats: Arc::new(RwLock::new(CacheStats::new())),
            query_patterns: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Cache search results with intelligent key generation
    pub async fn cache_search_results(
        &self,
        query: &str,
        filters: &HashMap<String, String>,
        results: Vec<SearchResult>,
        search_time_ms: u64,
    ) -> Result<()> {
        let cache_key = self.generate_cache_key(query, filters);
        let normalized_query = self.normalize_query(query);
        
        // Calculate result quality score
        let quality_score = self.calculate_quality_score(&results, search_time_ms);
        
        let search_entry = SearchCacheEntry {
            results: results.clone(),
            query_hash: cache_key.clone(),
            filters: filters.clone(),
            total_results: results.len(),
            search_time_ms,
            result_quality_score: quality_score,
        };

        let ttl = self.determine_ttl(&normalized_query, &results).await;
        let size_estimate = self.estimate_size(&search_entry);
        let cache_entry = CacheEntry::new(search_entry, ttl, size_estimate);

        // Store in cache
        self.cache.write().await.put(cache_key.clone(), cache_entry);

        // Update query patterns
        self.update_query_pattern(&normalized_query, results.len()).await;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.memory_usage += size_estimate;
        stats.entries = self.cache.read().await.len();

        Ok(())
    }

    /// Get cached search results
    pub async fn get_cached_results(
        &self,
        query: &str,
        filters: &HashMap<String, String>,
    ) -> Option<SearchCacheEntry> {
        let cache_key = self.generate_cache_key(query, filters);
        
        if let Some(entry) = self.cache.write().await.get(&cache_key) {
            if !entry.is_expired() {
                // Update stats
                self.stats.write().await.record_hit();
                
                // Update query pattern
                let normalized_query = self.normalize_query(query);
                self.update_query_pattern_hit(&normalized_query).await;
                
                return Some(entry.data.clone());
            } else {
                // Remove expired entry
                self.cache.write().await.pop(&cache_key);
                self.stats.write().await.record_eviction();
            }
        }

        self.stats.write().await.record_miss();
        None
    }

    /// Invalidate cache entries based on content changes
    pub async fn invalidate_by_file_path(&self, file_path: &str) -> usize {
        let mut cache = self.cache.write().await;
        let mut invalidated = 0;
        
        // Collect keys that need invalidation
        let keys_to_remove: Vec<String> = cache
            .iter()
            .filter(|(_, entry)| {
                entry.data.results.iter().any(|result| 
                    result.path.to_string_lossy().contains(file_path)
                )
            })
            .map(|(key, _)| key.clone())
            .collect();

        // Remove the entries
        for key in keys_to_remove {
            cache.pop(&key);
            invalidated += 1;
        }

        // Update stats
        let mut stats = self.stats.write().await;
        stats.evictions += invalidated as u64;
        stats.entries = cache.len();

        invalidated
    }

    /// Invalidate cache entries by content type
    pub async fn invalidate_by_content_type(&self, content_type: &str) -> usize {
        let mut cache = self.cache.write().await;
        let mut invalidated = 0;
        
        let keys_to_remove: Vec<String> = cache
            .iter()
            .filter(|(_, entry)| {
                entry.data.results.iter().any(|result| 
                    result.file_type == content_type
                )
            })
            .map(|(key, _)| key.clone())
            .collect();

        for key in keys_to_remove {
            cache.pop(&key);
            invalidated += 1;
        }

        let mut stats = self.stats.write().await;
        stats.evictions += invalidated as u64;
        stats.entries = cache.len();

        invalidated
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

    /// Clear all cached entries
    pub async fn clear(&self) -> Result<()> {
        self.cache.write().await.clear();
        self.query_patterns.write().await.clear();
        *self.stats.write().await = CacheStats::new();
        Ok(())
    }

    /// Get popular query patterns for optimization
    pub async fn get_popular_patterns(&self, limit: usize) -> Vec<(String, QueryPattern)> {
        let patterns = self.query_patterns.read().await;
        let mut sorted_patterns: Vec<_> = patterns.iter().collect();
        
        sorted_patterns.sort_by(|(_, a), (_, b)| b.frequency.cmp(&a.frequency));
        
        sorted_patterns
            .into_iter()
            .take(limit)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Preload cache with popular queries
    pub async fn preload_popular_queries(&self, queries: Vec<String>) -> Result<()> {
        // This would be implemented to proactively cache popular queries
        // For now, just update the patterns to mark them as popular
        let mut patterns = self.query_patterns.write().await;
        
        for query in queries {
            let normalized = self.normalize_query(&query);
            patterns.entry(normalized).or_insert_with(|| QueryPattern {
                frequency: 1,
                last_used: SystemTime::now(),
                average_results: 0.0,
                cache_hit_rate: 0.0,
            }).frequency += 1;
        }
        
        Ok(())
    }

    // Private helper methods
    fn generate_cache_key(&self, query: &str, filters: &HashMap<String, String>) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.normalize_query(query).as_bytes());
        
        // Sort filters for consistent hashing
        let mut sorted_filters: Vec<_> = filters.iter().collect();
        sorted_filters.sort_by_key(|(k, _)| *k);
        
        for (key, value) in sorted_filters {
            hasher.update(key.as_bytes());
            hasher.update(b"=");
            hasher.update(value.as_bytes());
            hasher.update(b"&");
        }
        
        format!("{:x}", hasher.finalize())
    }

    fn normalize_query(&self, query: &str) -> String {
        // Normalize query for consistent caching
        query
            .trim()
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn calculate_quality_score(&self, results: &[SearchResult], search_time_ms: u64) -> f64 {
        // Calculate a quality score based on result relevance and search speed
        let relevance_score = if !results.is_empty() {
            results.iter().map(|r| r.relevance_score as f64).sum::<f64>() / results.len() as f64
        } else {
            0.0
        };
        
        let speed_score = if search_time_ms > 0 {
            1000.0 / search_time_ms as f64 // Inverse relationship with time
        } else {
            1.0
        };
        
        (relevance_score * 0.7) + (speed_score * 0.3)
    }

    async fn determine_ttl(&self, query: &str, results: &[SearchResult]) -> Duration {
        // Determine TTL based on query patterns and result stability
        let base_ttl = self.config.default_ttl;
        
        if let Some(pattern) = self.query_patterns.read().await.get(query) {
            if pattern.frequency > 10 {
                // Popular queries get longer TTL
                return base_ttl.mul_f64(1.5);
            }
        }
        
        // Results with high scores get longer TTL
        let avg_score = if !results.is_empty() {
            results.iter().map(|r| r.relevance_score as f64).sum::<f64>() / results.len() as f64
        } else {
            0.0
        };
        
        if avg_score > 0.8 {
            base_ttl.mul_f64(1.2)
        } else if avg_score < 0.3 {
            base_ttl.mul_f64(0.5)
        } else {
            base_ttl
        }
    }

    fn estimate_size(&self, entry: &SearchCacheEntry) -> usize {
        // Rough estimation of memory usage
        let base_size = std::mem::size_of::<SearchCacheEntry>();
        let results_size = entry.results.len() * std::mem::size_of::<SearchResult>();
        let filters_size = entry.filters.iter()
            .map(|(k, v)| k.len() + v.len())
            .sum::<usize>();
        
        base_size + results_size + filters_size
    }

    async fn update_query_pattern(&self, query: &str, result_count: usize) {
        let mut patterns = self.query_patterns.write().await;
        let pattern = patterns.entry(query.to_string()).or_insert_with(|| QueryPattern {
            frequency: 0,
            last_used: SystemTime::now(),
            average_results: 0.0,
            cache_hit_rate: 0.0,
        });

        pattern.frequency += 1;
        pattern.last_used = SystemTime::now();
        pattern.average_results = (pattern.average_results + result_count as f64) / 2.0;
    }

    async fn update_query_pattern_hit(&self, query: &str) {
        let mut patterns = self.query_patterns.write().await;
        if let Some(pattern) = patterns.get_mut(query) {
            pattern.last_used = SystemTime::now();
            pattern.cache_hit_rate = (pattern.cache_hit_rate + 1.0) / 2.0; // Moving average
        }
    }
}