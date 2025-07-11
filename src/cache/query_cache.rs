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

/// Query processing cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCacheEntry {
    pub original_query: String,
    pub processed_query: String,
    pub query_type: QueryType,
    pub intent: String,
    pub entities: Vec<String>,
    pub filters: HashMap<String, String>,
    pub processing_time_ms: u64,
    pub confidence_score: f64,
    pub expansions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryType {
    Simple,
    Boolean,
    Semantic,
    Hybrid,
    Contextual,
}

/// Query analysis and processing cache
pub struct QueryCache {
    cache: Arc<RwLock<LruCache<String, CacheEntry<QueryCacheEntry>>>>,
    query_patterns: Arc<RwLock<HashMap<String, QueryPatternAnalysis>>>,
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
}

#[derive(Debug, Clone)]
struct QueryPatternAnalysis {
    frequency: u32,
    last_seen: SystemTime,
    average_processing_time: f64,
    success_rate: f64,
    typical_results_count: f64,
}

impl QueryCache {
    pub async fn new(config: CacheConfig) -> Result<Self> {
        let cache_size = NonZeroUsize::new(config.max_entries / 4).unwrap_or(NonZeroUsize::new(1000).unwrap());
        let cache = Arc::new(RwLock::new(LruCache::new(cache_size)));
        
        Ok(Self {
            cache,
            query_patterns: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats::new())),
        })
    }

    /// Cache processed query
    pub async fn cache_processed_query(
        &self,
        original_query: &str,
        processed_query: String,
        query_type: QueryType,
        intent: String,
        entities: Vec<String>,
        filters: HashMap<String, String>,
        processing_time_ms: u64,
        confidence_score: f64,
        expansions: Vec<String>,
    ) -> Result<()> {
        let cache_key = self.generate_cache_key(original_query, &filters);
        
        let query_entry = QueryCacheEntry {
            original_query: original_query.to_string(),
            processed_query,
            query_type,
            intent,
            entities,
            filters,
            processing_time_ms,
            confidence_score,
            expansions,
        };

        let ttl = self.determine_ttl(confidence_score, processing_time_ms);
        let size_estimate = self.estimate_size(&query_entry);
        let cache_entry = CacheEntry::new(query_entry, ttl, size_estimate);

        // Store in cache
        self.cache.write().await.put(cache_key, cache_entry);

        // Update query patterns
        self.update_query_pattern(original_query, processing_time_ms, confidence_score).await;

        // Update stats
        let mut stats = self.stats.write().await;
        stats.memory_usage += size_estimate;
        stats.entries = self.cache.read().await.len();

        Ok(())
    }

    /// Get cached processed query
    pub async fn get_processed_query(
        &self,
        original_query: &str,
        filters: &HashMap<String, String>,
    ) -> Option<QueryCacheEntry> {
        let cache_key = self.generate_cache_key(original_query, filters);
        
        if let Some(entry) = self.cache.write().await.get(&cache_key) {
            if !entry.is_expired() {
                self.stats.write().await.record_hit();
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

    /// Get similar queries for suggestion
    pub async fn get_similar_queries(
        &self,
        query: &str,
        max_results: usize,
    ) -> Vec<(QueryCacheEntry, f64)> {
        let normalized_query = self.normalize_query(query);
        let query_tokens = self.tokenize_query(&normalized_query);
        let mut results = Vec::new();
        
        let cache = self.cache.read().await;
        for (_, entry) in cache.iter() {
            if !entry.is_expired() {
                let cached_tokens = self.tokenize_query(&self.normalize_query(&entry.data.original_query));
                let similarity = self.calculate_token_similarity(&query_tokens, &cached_tokens);
                
                if similarity > 0.3 { // Minimum similarity threshold
                    results.push((entry.data.clone(), similarity));
                }
            }
        }
        
        // Sort by similarity and limit results
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(max_results);
        
        results
    }

    /// Get query suggestions based on patterns
    pub async fn get_query_suggestions(
        &self,
        partial_query: &str,
        max_suggestions: usize,
    ) -> Vec<String> {
        let partial_normalized = self.normalize_query(partial_query);
        let mut suggestions = Vec::new();
        
        let cache = self.cache.read().await;
        for (_, entry) in cache.iter() {
            if !entry.is_expired() {
                let cached_normalized = self.normalize_query(&entry.data.original_query);
                
                if cached_normalized.starts_with(&partial_normalized) && 
                   cached_normalized != partial_normalized {
                    suggestions.push(entry.data.original_query.clone());
                }
            }
        }
        
        // Remove duplicates and sort by frequency
        suggestions.sort();
        suggestions.dedup();
        
        // Limit results
        suggestions.truncate(max_suggestions);
        suggestions
    }

    /// Get query expansions for a given query
    pub async fn get_query_expansions(&self, query: &str) -> Vec<String> {
        let normalized_query = self.normalize_query(query);
        let mut expansions = Vec::new();
        
        let cache = self.cache.read().await;
        for (_, entry) in cache.iter() {
            if !entry.is_expired() {
                let cached_normalized = self.normalize_query(&entry.data.original_query);
                
                if cached_normalized == normalized_query && !entry.data.expansions.is_empty() {
                    expansions.extend(entry.data.expansions.clone());
                }
            }
        }
        
        // Remove duplicates
        expansions.sort();
        expansions.dedup();
        expansions
    }

    /// Get queries by type
    pub async fn get_queries_by_type(&self, query_type: &QueryType) -> Vec<QueryCacheEntry> {
        let cache = self.cache.read().await;
        let mut results = Vec::new();
        
        for (_, entry) in cache.iter() {
            if !entry.is_expired() && std::mem::discriminant(&entry.data.query_type) == std::mem::discriminant(query_type) {
                results.push(entry.data.clone());
            }
        }
        
        // Sort by confidence score
        results.sort_by(|a, b| b.confidence_score.partial_cmp(&a.confidence_score).unwrap());
        results
    }

    /// Get popular query patterns
    pub async fn get_popular_patterns(&self, limit: usize) -> Vec<(String, QueryPatternAnalysis)> {
        let patterns = self.query_patterns.read().await;
        let mut sorted_patterns: Vec<_> = patterns.iter().collect();
        
        sorted_patterns.sort_by(|(_, a), (_, b)| b.frequency.cmp(&a.frequency));
        
        sorted_patterns
            .into_iter()
            .take(limit)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Analyze query performance
    pub async fn analyze_query_performance(&self) -> QueryPerformanceAnalysis {
        let cache = self.cache.read().await;
        let patterns = self.query_patterns.read().await;
        
        let mut total_processing_time = 0.0;
        let mut total_confidence = 0.0;
        let mut query_count = 0;
        let mut type_counts = HashMap::new();
        
        for (_, entry) in cache.iter() {
            if !entry.is_expired() {
                total_processing_time += entry.data.processing_time_ms as f64;
                total_confidence += entry.data.confidence_score;
                query_count += 1;
                
                let type_name = format!("{:?}", entry.data.query_type);
                *type_counts.entry(type_name).or_insert(0) += 1;
            }
        }
        
        let avg_processing_time = if query_count > 0 {
            total_processing_time / query_count as f64
        } else {
            0.0
        };
        
        let avg_confidence = if query_count > 0 {
            total_confidence / query_count as f64
        } else {
            0.0
        };
        
        QueryPerformanceAnalysis {
            total_queries: query_count,
            average_processing_time_ms: avg_processing_time,
            average_confidence_score: avg_confidence,
            query_type_distribution: type_counts,
            pattern_count: patterns.len(),
            cache_hit_rate: self.stats.read().await.hit_rate,
        }
    }

    /// Clean low-confidence queries
    pub async fn clean_low_confidence_queries(&self, min_confidence: f64) -> usize {
        let mut cache = self.cache.write().await;
        let mut removed = 0;
        
        let keys_to_remove: Vec<String> = cache
            .iter()
            .filter(|(_, entry)| entry.data.confidence_score < min_confidence)
            .map(|(key, _)| key.clone())
            .collect();

        for key in keys_to_remove {
            cache.pop(&key);
            removed += 1;
        }

        let mut stats = self.stats.write().await;
        stats.evictions += removed as u64;
        stats.entries = cache.len();

        removed
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

    /// Clear all cached queries
    pub async fn clear(&self) -> Result<()> {
        self.cache.write().await.clear();
        self.query_patterns.write().await.clear();
        *self.stats.write().await = CacheStats::new();
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
        query
            .trim()
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn tokenize_query(&self, query: &str) -> Vec<String> {
        query
            .split_whitespace()
            .map(|token| token.to_lowercase())
            .collect()
    }

    fn calculate_token_similarity(&self, tokens1: &[String], tokens2: &[String]) -> f64 {
        if tokens1.is_empty() || tokens2.is_empty() {
            return 0.0;
        }
        
        let set1: std::collections::HashSet<_> = tokens1.iter().collect();
        let set2: std::collections::HashSet<_> = tokens2.iter().collect();
        
        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        
        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    fn determine_ttl(&self, confidence_score: f64, processing_time_ms: u64) -> Duration {
        let base_ttl = self.config.default_ttl;
        
        // High-confidence, fast queries get longer TTL
        let confidence_factor = if confidence_score > 0.8 { 1.5 } else { 1.0 };
        let speed_factor = if processing_time_ms < 100 { 1.2 } else { 1.0 };
        
        base_ttl.mul_f64(confidence_factor * speed_factor)
    }

    fn estimate_size(&self, entry: &QueryCacheEntry) -> usize {
        let base_size = std::mem::size_of::<QueryCacheEntry>();
        let string_sizes = entry.original_query.len() + 
                          entry.processed_query.len() + 
                          entry.intent.len() +
                          entry.entities.iter().map(|s| s.len()).sum::<usize>() +
                          entry.expansions.iter().map(|s| s.len()).sum::<usize>();
        let filters_size = entry.filters.iter()
            .map(|(k, v)| k.len() + v.len())
            .sum::<usize>();
        
        base_size + string_sizes + filters_size
    }

    async fn update_query_pattern(&self, query: &str, processing_time_ms: u64, confidence_score: f64) {
        let normalized = self.normalize_query(query);
        let mut patterns = self.query_patterns.write().await;
        
        let pattern = patterns.entry(normalized).or_insert_with(|| QueryPatternAnalysis {
            frequency: 0,
            last_seen: SystemTime::now(),
            average_processing_time: 0.0,
            success_rate: 0.0,
            typical_results_count: 0.0,
        });

        pattern.frequency += 1;
        pattern.last_seen = SystemTime::now();
        pattern.average_processing_time = (pattern.average_processing_time + processing_time_ms as f64) / 2.0;
        pattern.success_rate = (pattern.success_rate + confidence_score) / 2.0;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPerformanceAnalysis {
    pub total_queries: usize,
    pub average_processing_time_ms: f64,
    pub average_confidence_score: f64,
    pub query_type_distribution: HashMap<String, usize>,
    pub pattern_count: usize,
    pub cache_hit_rate: f64,
}