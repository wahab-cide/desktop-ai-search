use crate::core::query_intent::{QueryIntent, QueryIntentClassifier, QueryClassifierConfig};
use crate::error::Result;
use moka::future::Cache;
use std::sync::Arc;
use std::time::Duration;
use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};

/// Cache statistics for monitoring performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hit_count: u64,
    pub miss_count: u64,
    pub hit_rate: f64,
    pub entry_count: u64,
    pub estimated_size: u64,
}

/// Configuration for the cached query classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedClassifierConfig {
    /// Base query classifier configuration
    pub classifier_config: QueryClassifierConfig,
    /// Maximum number of cached query classifications
    pub cache_size: u64,
    /// Time-to-live for cached entries (in seconds)
    pub cache_ttl_seconds: u64,
    /// Enable cache statistics collection
    pub enable_cache_stats: bool,
    /// Minimum query length to cache (avoid caching very short queries)
    pub min_query_length_to_cache: usize,
}

impl Default for CachedClassifierConfig {
    fn default() -> Self {
        Self {
            classifier_config: QueryClassifierConfig::default(),
            cache_size: 1000,
            cache_ttl_seconds: 300, // 5 minutes
            enable_cache_stats: true,
            min_query_length_to_cache: 3,
        }
    }
}

/// Cached query intent classifier with moka cache for performance optimization
pub struct CachedQueryClassifier {
    classifier: QueryIntentClassifier,
    cache: Cache<String, Arc<QueryIntent>>,
    config: CachedClassifierConfig,
    stats: Arc<std::sync::Mutex<CacheStats>>,
}

impl CachedQueryClassifier {
    /// Create a new cached query classifier
    pub async fn new(config: CachedClassifierConfig) -> Result<Self> {
        let classifier = QueryIntentClassifier::new(config.classifier_config.clone())?;
        
        let cache = Cache::builder()
            .max_capacity(config.cache_size)
            .time_to_live(Duration::from_secs(config.cache_ttl_seconds))
            .build();

        let stats = Arc::new(std::sync::Mutex::new(CacheStats {
            hit_count: 0,
            miss_count: 0,
            hit_rate: 0.0,
            entry_count: 0,
            estimated_size: 0,
        }));

        Ok(Self {
            classifier,
            cache,
            config,
            stats,
        })
    }

    /// Create a cached classifier with default configuration
    pub async fn default() -> Result<Self> {
        Self::new(CachedClassifierConfig::default()).await
    }

    /// Analyze query with caching optimization
    pub async fn analyze_query(&self, query: &str) -> Result<Arc<QueryIntent>> {
        let cache_key = self.generate_cache_key(query);
        
        // Check if query is too short to cache
        if query.len() < self.config.min_query_length_to_cache {
            let intent = self.classifier.analyze_query(query).await?;
            return Ok(Arc::new(intent));
        }

        // Try to get from cache first
        if let Some(cached_intent) = self.cache.get(&cache_key).await {
            if self.config.enable_cache_stats {
                self.update_cache_stats(true).await;
            }
            return Ok(cached_intent);
        }

        // Cache miss - compute and store
        if self.config.enable_cache_stats {
            self.update_cache_stats(false).await;
        }

        let intent = self.classifier.analyze_query(query).await?;
        let arc_intent = Arc::new(intent);
        
        // Store in cache
        self.cache.insert(cache_key, arc_intent.clone()).await;
        
        Ok(arc_intent)
    }

    /// Analyze multiple queries in batch for efficiency
    pub async fn analyze_queries_batch(&self, queries: &[String]) -> Result<Vec<Arc<QueryIntent>>> {
        let mut results = Vec::with_capacity(queries.len());
        
        for query in queries {
            let intent = self.analyze_query(query).await?;
            results.push(intent);
        }
        
        Ok(results)
    }

    /// Prefetch query classification for common patterns
    pub async fn prefetch_common_queries(&self, queries: &[String]) -> Result<()> {
        for query in queries {
            // Only prefetch if not already cached
            let cache_key = self.generate_cache_key(query);
            if self.cache.get(&cache_key).await.is_none() {
                let _ = self.analyze_query(query).await?;
            }
        }
        Ok(())
    }

    /// Warm up cache with frequently used query patterns
    pub async fn warm_cache(&self, common_patterns: &[String]) -> Result<()> {
        println!("🔥 Warming up query classification cache with {} patterns", common_patterns.len());
        
        for pattern in common_patterns {
            let _ = self.analyze_query(pattern).await?;
        }
        
        println!("✅ Cache warming completed");
        Ok(())
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> CacheStats {
        if !self.config.enable_cache_stats {
            return CacheStats {
                hit_count: 0,
                miss_count: 0,
                hit_rate: 0.0,
                entry_count: self.cache.entry_count(),
                estimated_size: self.cache.weighted_size(),
            };
        }

        let stats_guard = self.stats.lock().unwrap();
        let mut stats = stats_guard.clone();
        drop(stats_guard);

        // Update live stats
        stats.entry_count = self.cache.entry_count();
        stats.estimated_size = self.cache.weighted_size();
        
        // Calculate hit rate
        let total_requests = stats.hit_count + stats.miss_count;
        stats.hit_rate = if total_requests > 0 {
            stats.hit_count as f64 / total_requests as f64
        } else {
            0.0
        };

        stats
    }

    /// Clear all cached entries
    pub async fn clear_cache(&self) {
        self.cache.invalidate_all();
        
        if self.config.enable_cache_stats {
            let mut stats = self.stats.lock().unwrap();
            stats.hit_count = 0;
            stats.miss_count = 0;
            stats.hit_rate = 0.0;
        }
    }

    /// Invalidate cache entries for a specific query pattern
    pub async fn invalidate_pattern(&self, pattern: &str) {
        let cache_key = self.generate_cache_key(pattern);
        self.cache.invalidate(&cache_key).await;
    }

    /// Check if a query is likely cached
    pub async fn is_query_cached(&self, query: &str) -> bool {
        if query.len() < self.config.min_query_length_to_cache {
            return false;
        }
        
        let cache_key = self.generate_cache_key(query);
        self.cache.get(&cache_key).await.is_some()
    }

    /// Update cache configuration at runtime
    pub async fn update_config(&mut self, new_config: CachedClassifierConfig) -> Result<()> {
        // Create new classifier if base config changed
        if self.config.classifier_config.enable_classification != new_config.classifier_config.enable_classification
            || self.config.classifier_config.enable_ner != new_config.classifier_config.enable_ner
            || self.config.classifier_config.enable_temporal != new_config.classifier_config.enable_temporal {
            self.classifier = QueryIntentClassifier::new(new_config.classifier_config.clone())?;
        }

        // Clear cache if cache settings changed significantly
        if self.config.cache_size != new_config.cache_size || self.config.cache_ttl_seconds != new_config.cache_ttl_seconds {
            self.clear_cache().await;
        }

        self.config = new_config;
        Ok(())
    }

    /// Generate consistent cache key for queries
    fn generate_cache_key(&self, query: &str) -> String {
        // Normalize query for consistent caching
        let normalized = query.trim().to_lowercase();
        
        // Use SHA256 hash for consistent key generation
        let mut hasher = Sha256::new();
        hasher.update(normalized.as_bytes());
        let result = hasher.finalize();
        
        // Convert to hex string
        format!("query_{:x}", result)
    }

    /// Update cache statistics
    async fn update_cache_stats(&self, was_hit: bool) {
        if !self.config.enable_cache_stats {
            return;
        }

        let mut stats = self.stats.lock().unwrap();
        if was_hit {
            stats.hit_count += 1;
        } else {
            stats.miss_count += 1;
        }
    }

    /// Get default common query patterns for cache warming
    pub fn get_default_warm_patterns() -> Vec<String> {
        vec![
            // Document searches
            "find document".to_string(),
            "find file".to_string(),
            "search document".to_string(),
            "locate file".to_string(),
            
            // Content searches  
            "contains".to_string(),
            "mentions".to_string(),
            "about".to_string(),
            
            // Person searches
            "from john".to_string(),
            "by jane".to_string(),
            "sent by".to_string(),
            "created by".to_string(),
            
            // Temporal searches
            "recent".to_string(),
            "yesterday".to_string(),
            "last week".to_string(),
            "today".to_string(),
            "last month".to_string(),
            
            // File type searches
            "pdf files".to_string(),
            "image files".to_string(),
            "documents".to_string(),
            "presentations".to_string(),
            
            // Question answering
            "what is".to_string(),
            "how to".to_string(),
            "why does".to_string(),
            "where is".to_string(),
            
            // Similarity searches
            "similar to".to_string(),
            "like this".to_string(),
            "related to".to_string(),
        ]
    }

    /// Get classifier configuration
    pub fn get_config(&self) -> &CachedClassifierConfig {
        &self.config
    }

    /// Get underlying classifier reference (for advanced operations)
    pub fn get_classifier(&self) -> &QueryIntentClassifier {
        &self.classifier
    }
}

/// Query classification service that manages the cached classifier
pub struct QueryClassificationService {
    classifier: CachedQueryClassifier,
}

impl QueryClassificationService {
    /// Create a new query classification service
    pub async fn new(config: CachedClassifierConfig) -> Result<Self> {
        let mut classifier = CachedQueryClassifier::new(config).await?;
        
        // Warm up cache with common patterns
        let common_patterns = CachedQueryClassifier::get_default_warm_patterns();
        classifier.warm_cache(&common_patterns).await?;
        
        Ok(Self { classifier })
    }

    /// Create service with default configuration
    pub async fn default() -> Result<Self> {
        Self::new(CachedClassifierConfig::default()).await
    }

    /// Analyze a single query
    pub async fn classify_query(&self, query: &str) -> Result<Arc<QueryIntent>> {
        self.classifier.analyze_query(query).await
    }

    /// Analyze multiple queries efficiently
    pub async fn classify_queries_batch(&self, queries: &[String]) -> Result<Vec<Arc<QueryIntent>>> {
        self.classifier.analyze_queries_batch(queries).await
    }

    /// Get performance statistics
    pub async fn get_performance_stats(&self) -> CacheStats {
        self.classifier.get_cache_stats().await
    }

    /// Clear all cached classifications
    pub async fn clear_cache(&self) {
        self.classifier.clear_cache().await;
    }

    /// Check if the service is ready
    pub fn is_ready(&self) -> bool {
        true // Simple ready check - can be enhanced with health checks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cached_classifier_basic() {
        let classifier = CachedQueryClassifier::default().await.unwrap();
        
        let query = "find documents by John from last week";
        let result1 = classifier.analyze_query(query).await.unwrap();
        let result2 = classifier.analyze_query(query).await.unwrap();
        
        // Should be same instance due to caching
        assert!(Arc::ptr_eq(&result1, &result2));
        
        let stats = classifier.get_cache_stats().await;
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let classifier = CachedQueryClassifier::default().await.unwrap();
        
        // First query - cache miss
        let _ = classifier.analyze_query("test query 1").await.unwrap();
        
        // Second query - cache miss
        let _ = classifier.analyze_query("test query 2").await.unwrap();
        
        // Repeat first query - cache hit
        let _ = classifier.analyze_query("test query 1").await.unwrap();
        
        let stats = classifier.get_cache_stats().await;
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 2);
        assert!((stats.hit_rate - 0.333).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_short_query_no_cache() {
        let classifier = CachedQueryClassifier::default().await.unwrap();
        
        // Very short query should not be cached
        let result1 = classifier.analyze_query("hi").await.unwrap();
        let result2 = classifier.analyze_query("hi").await.unwrap();
        
        // Should be different instances (not cached)
        assert!(!Arc::ptr_eq(&result1, &result2));
        
        let stats = classifier.get_cache_stats().await;
        assert_eq!(stats.hit_count, 0);
        assert_eq!(stats.miss_count, 0); // Not counted as misses either
    }

    #[tokio::test]
    async fn test_service_initialization() {
        let service = QueryClassificationService::default().await.unwrap();
        assert!(service.is_ready());
        
        let result = service.classify_query("find my documents").await.unwrap();
        assert!(!result.intents.is_empty());
    }
}