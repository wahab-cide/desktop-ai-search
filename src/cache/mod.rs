use crate::error::{AppError, Result};
use crate::database::operations::SearchResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use lru::LruCache;
use moka::future::Cache as MokaCache;
use std::num::NonZeroUsize;

pub mod search_cache;
pub mod embedding_cache;
pub mod model_cache;
pub mod query_cache;

/// Cache entry with TTL and access tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    pub data: T,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub access_count: u64,
    pub ttl: Duration,
    pub size_estimate: usize,
}

impl<T> CacheEntry<T> {
    pub fn new(data: T, ttl: Duration, size_estimate: usize) -> Self {
        let now = SystemTime::now();
        Self {
            data,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            ttl,
            size_estimate,
        }
    }

    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed().unwrap_or(Duration::MAX) > self.ttl
    }

    pub fn access(&mut self) -> &T {
        self.last_accessed = SystemTime::now();
        self.access_count += 1;
        &self.data
    }

    pub fn age(&self) -> Duration {
        self.created_at.elapsed().unwrap_or(Duration::ZERO)
    }
}

/// Cache statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub entries: usize,
    pub memory_usage: usize,
    pub hit_rate: f64,
    pub average_age: Duration,
    pub oldest_entry: Option<Duration>,
    pub newest_entry: Option<Duration>,
}

impl CacheStats {
    pub fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
            entries: 0,
            memory_usage: 0,
            hit_rate: 0.0,
            average_age: Duration::ZERO,
            oldest_entry: None,
            newest_entry: None,
        }
    }

    pub fn record_hit(&mut self) {
        self.hits += 1;
        self.update_hit_rate();
    }

    pub fn record_miss(&mut self) {
        self.misses += 1;
        self.update_hit_rate();
    }

    pub fn record_eviction(&mut self) {
        self.evictions += 1;
    }

    fn update_hit_rate(&mut self) {
        let total = self.hits + self.misses;
        self.hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub max_entries: usize,
    pub max_memory_mb: usize,
    pub default_ttl: Duration,
    pub cleanup_interval: Duration,
    pub eviction_policy: EvictionPolicy,
    pub compression_enabled: bool,
    pub persistence_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    Lru,
    Lfu,
    Ttl,
    Size,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            max_memory_mb: 512,
            default_ttl: Duration::from_secs(3600), // 1 hour
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            eviction_policy: EvictionPolicy::Lru,
            compression_enabled: false,
            persistence_enabled: false,
        }
    }
}

/// Multi-layered cache manager
pub struct CacheManager {
    // L1: In-memory LRU cache for hot data
    l1_cache: Arc<RwLock<LruCache<String, CacheEntry<Vec<u8>>>>>,
    
    // L2: Persistent cache using moka for medium-term storage
    l2_cache: MokaCache<String, CacheEntry<Vec<u8>>>,
    
    // Cache-specific managers
    search_cache: search_cache::SearchCache,
    embedding_cache: embedding_cache::EmbeddingCache,
    model_cache: model_cache::ModelCache,
    query_cache: query_cache::QueryCache,
    
    // Configuration and statistics
    config: CacheConfig,
    stats: Arc<RwLock<HashMap<String, CacheStats>>>,
    
    // Cleanup task handle
    cleanup_handle: Option<tokio::task::JoinHandle<()>>,
}

impl CacheManager {
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let l1_size = NonZeroUsize::new(config.max_entries / 4).unwrap_or(NonZeroUsize::new(1000).unwrap());
        let l1_cache = Arc::new(RwLock::new(LruCache::new(l1_size)));
        
        let l2_cache = MokaCache::builder()
            .max_capacity(config.max_entries as u64)
            .time_to_live(config.default_ttl)
            .build();

        let search_cache = search_cache::SearchCache::new(config.clone()).await?;
        let embedding_cache = embedding_cache::EmbeddingCache::new(config.clone()).await?;
        let model_cache = model_cache::ModelCache::new(config.clone()).await?;
        let query_cache = query_cache::QueryCache::new(config.clone()).await?;

        let mut manager = Self {
            l1_cache,
            l2_cache,
            search_cache,
            embedding_cache,
            model_cache,
            query_cache,
            config,
            stats: Arc::new(RwLock::new(HashMap::new())),
            cleanup_handle: None,
        };

        manager.start_cleanup_task();
        Ok(manager)
    }

    /// Get data from cache with automatic tier promotion
    pub async fn get<T: for<'de> Deserialize<'de>>(&self, key: &str, cache_type: &str) -> Option<T> {
        // Try L1 cache first
        if let Some(entry) = self.get_from_l1(key).await {
            if !entry.is_expired() {
                self.record_hit(cache_type).await;
                if let Ok(data) = bincode::deserialize(&entry.data) {
                    return Some(data);
                }
            }
        }

        // Try L2 cache
        if let Some(entry) = self.l2_cache.get(key).await {
            if !entry.is_expired() {
                self.record_hit(cache_type).await;
                
                // Promote to L1 cache
                self.put_to_l1(key, entry.clone()).await;
                
                if let Ok(data) = bincode::deserialize(&entry.data) {
                    return Some(data);
                }
            }
        }

        self.record_miss(cache_type).await;
        None
    }

    /// Put data into cache with automatic tier management
    pub async fn put<T: Serialize>(&self, key: &str, value: &T, ttl: Option<Duration>, cache_type: &str) -> Result<()> {
        let serialized = bincode::serialize(value)
            .map_err(|e| AppError::Unknown(format!("Serialization error: {}", e)))?;
        
        let ttl = ttl.unwrap_or(self.config.default_ttl);
        let size_estimate = serialized.len();
        let entry = CacheEntry::new(serialized, ttl, size_estimate);

        // Always put in L2 cache
        self.l2_cache.insert(key.to_string(), entry.clone()).await;

        // Put in L1 cache if small enough
        if size_estimate < 1024 * 1024 { // 1MB threshold
            self.put_to_l1(key, entry).await;
        }

        self.update_memory_usage(cache_type, size_estimate as i64).await;
        Ok(())
    }

    /// Remove from all cache tiers
    pub async fn remove(&self, key: &str, cache_type: &str) -> bool {
        let l1_removed = self.remove_from_l1(key).await;
        let l2_removed = self.l2_cache.remove(key).await.is_some();
        
        if l1_removed || l2_removed {
            self.record_eviction(cache_type).await;
            true
        } else {
            false
        }
    }

    /// Clear all caches
    pub async fn clear(&self) -> Result<()> {
        self.l1_cache.write().await.clear();
        self.l2_cache.invalidate_all();
        self.l2_cache.run_pending_tasks().await;

        self.search_cache.clear().await?;
        self.embedding_cache.clear().await?;
        self.model_cache.clear().await?;
        self.query_cache.clear().await?;

        // Reset statistics
        self.stats.write().await.clear();

        Ok(())
    }

    /// Get cache statistics
    pub async fn get_stats(&self) -> HashMap<String, CacheStats> {
        let mut stats = self.stats.read().await.clone();
        
        // Update current entry counts
        let l1_count = self.l1_cache.read().await.len();
        let l2_count = self.l2_cache.entry_count();
        
        stats.entry("l1".to_string()).or_insert(CacheStats::new()).entries = l1_count;
        stats.entry("l2".to_string()).or_insert(CacheStats::new()).entries = l2_count as usize;

        // Add specialized cache stats
        stats.insert("search".to_string(), self.search_cache.get_stats().await);
        stats.insert("embedding".to_string(), self.embedding_cache.get_stats().await);
        stats.insert("model".to_string(), self.model_cache.get_stats().await);
        stats.insert("query".to_string(), self.query_cache.get_stats().await);

        stats
    }

    /// Get cache usage summary
    pub async fn get_usage_summary(&self) -> CacheUsageSummary {
        let stats = self.get_stats().await;
        let total_entries = stats.values().map(|s| s.entries).sum();
        let total_memory = stats.values().map(|s| s.memory_usage).sum();
        let average_hit_rate = if !stats.is_empty() {
            stats.values().map(|s| s.hit_rate).sum::<f64>() / stats.len() as f64
        } else {
            0.0
        };

        CacheUsageSummary {
            total_entries,
            total_memory_bytes: total_memory,
            total_memory_mb: total_memory / (1024 * 1024),
            average_hit_rate,
            cache_efficiency: self.calculate_efficiency(&stats).await,
            memory_pressure: self.calculate_memory_pressure().await,
        }
    }

    // L1 cache operations
    async fn get_from_l1(&self, key: &str) -> Option<CacheEntry<Vec<u8>>> {
        self.l1_cache.read().await.peek(key).cloned()
    }

    async fn put_to_l1(&self, key: &str, entry: CacheEntry<Vec<u8>>) {
        self.l1_cache.write().await.put(key.to_string(), entry);
    }

    async fn remove_from_l1(&self, key: &str) -> bool {
        self.l1_cache.write().await.pop(key).is_some()
    }

    // Statistics tracking
    async fn record_hit(&self, cache_type: &str) {
        let mut stats = self.stats.write().await;
        stats.entry(cache_type.to_string()).or_insert(CacheStats::new()).record_hit();
    }

    async fn record_miss(&self, cache_type: &str) {
        let mut stats = self.stats.write().await;
        stats.entry(cache_type.to_string()).or_insert(CacheStats::new()).record_miss();
    }

    async fn record_eviction(&self, cache_type: &str) {
        let mut stats = self.stats.write().await;
        stats.entry(cache_type.to_string()).or_insert(CacheStats::new()).record_eviction();
    }

    async fn update_memory_usage(&self, cache_type: &str, delta: i64) {
        let mut stats = self.stats.write().await;
        let cache_stats = stats.entry(cache_type.to_string()).or_insert(CacheStats::new());
        if delta > 0 {
            cache_stats.memory_usage += delta as usize;
        } else {
            cache_stats.memory_usage = cache_stats.memory_usage.saturating_sub((-delta) as usize);
        }
    }

    // Cleanup and maintenance
    fn start_cleanup_task(&mut self) {
        let l1_cache = self.l1_cache.clone();
        let l2_cache = self.l2_cache.clone();
        let cleanup_interval = self.config.cleanup_interval;

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            
            loop {
                interval.tick().await;
                
                // Clean expired entries from L1 cache
                let mut l1 = l1_cache.write().await;
                let keys_to_remove: Vec<String> = l1
                    .iter()
                    .filter(|(_, entry)| entry.is_expired())
                    .map(|(k, _)| k.clone())
                    .collect();
                
                for key in keys_to_remove {
                    l1.pop(&key);
                }
                drop(l1);

                // L2 cache cleanup is handled automatically by moka
                l2_cache.run_pending_tasks().await;
            }
        });

        self.cleanup_handle = Some(handle);
    }

    async fn calculate_efficiency(&self, stats: &HashMap<String, CacheStats>) -> f64 {
        // Calculate overall cache efficiency based on hit rates and memory usage
        let total_accesses: u64 = stats.values().map(|s| s.hits + s.misses).sum();
        let total_hits: u64 = stats.values().map(|s| s.hits).sum();
        
        if total_accesses > 0 {
            total_hits as f64 / total_accesses as f64
        } else {
            0.0
        }
    }

    async fn calculate_memory_pressure(&self) -> f64 {
        let current_usage = self.get_stats().await.values().map(|s| s.memory_usage).sum::<usize>();
        let max_usage = self.config.max_memory_mb * 1024 * 1024;
        
        if max_usage > 0 {
            current_usage as f64 / max_usage as f64
        } else {
            0.0
        }
    }

    // Public accessors for specialized caches
    pub fn search_cache(&self) -> &search_cache::SearchCache {
        &self.search_cache
    }

    pub fn embedding_cache(&self) -> &embedding_cache::EmbeddingCache {
        &self.embedding_cache
    }

    pub fn model_cache(&self) -> &model_cache::ModelCache {
        &self.model_cache
    }

    pub fn query_cache(&self) -> &query_cache::QueryCache {
        &self.query_cache
    }
}

impl Drop for CacheManager {
    fn drop(&mut self) {
        if let Some(handle) = self.cleanup_handle.take() {
            handle.abort();
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheUsageSummary {
    pub total_entries: usize,
    pub total_memory_bytes: usize,
    pub total_memory_mb: usize,
    pub average_hit_rate: f64,
    pub cache_efficiency: f64,
    pub memory_pressure: f64,
}

/// Global cache manager instance
pub static CACHE_MANAGER: once_cell::sync::Lazy<Arc<tokio::sync::RwLock<Option<CacheManager>>>> = 
    once_cell::sync::Lazy::new(|| Arc::new(tokio::sync::RwLock::new(None)));

/// Initialize the global cache manager
pub async fn init_cache_manager(config: CacheConfig) -> Result<()> {
    let manager = CacheManager::new(config).await?;
    *CACHE_MANAGER.write().await = Some(manager);
    Ok(())
}

/// Get the global cache manager
pub async fn get_cache_manager() -> Option<Arc<tokio::sync::RwLock<Option<CacheManager>>>> {
    if CACHE_MANAGER.read().await.is_some() {
        Some(CACHE_MANAGER.clone())
    } else {
        None
    }
}