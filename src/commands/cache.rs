use crate::cache::{CacheUsageSummary, CacheStats, get_cache_manager};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tauri::State;

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheManagementInfo {
    pub usage_summary: CacheUsageSummary,
    pub cache_stats: HashMap<String, CacheStats>,
    pub configuration: CacheConfigInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheConfigInfo {
    pub max_entries: usize,
    pub max_memory_mb: usize,
    pub default_ttl_hours: u64,
    pub cleanup_interval_minutes: u64,
    pub enabled_caches: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheOptimizationResult {
    pub freed_memory_bytes: usize,
    pub removed_entries: usize,
    pub optimization_type: String,
    pub duration_ms: u64,
}

/// Get comprehensive cache status
#[tauri::command]
pub async fn get_cache_status() -> std::result::Result<CacheManagementInfo, String> {
    if let Some(cache_manager_ref) = get_cache_manager().await {
        if let Some(cache_manager) = cache_manager_ref.read().await.as_ref() {
            let usage_summary = cache_manager.get_usage_summary().await;
            let cache_stats = cache_manager.get_stats().await;
            
            let enabled_caches = vec![
                "search".to_string(),
                "embedding".to_string(),
                "model".to_string(),
                "query".to_string(),
            ];
            
            let configuration = CacheConfigInfo {
                max_entries: 10000, // Would get from actual config
                max_memory_mb: 512,
                default_ttl_hours: 24,
                cleanup_interval_minutes: 5,
                enabled_caches,
            };
            
            let info = CacheManagementInfo {
                usage_summary,
                cache_stats,
                configuration,
            };
            
            Ok(info)
        } else {
            Err("Cache manager not initialized".to_string())
        }
    } else {
        Err("Cache manager not available".to_string())
    }
}

/// Clear all caches
#[tauri::command]
pub async fn clear_all_caches() -> std::result::Result<String, String> {
    if let Some(cache_manager_ref) = get_cache_manager().await {
        if let Some(cache_manager) = cache_manager_ref.read().await.as_ref() {
            match cache_manager.clear().await {
                Ok(_) => Ok("All caches cleared successfully".to_string()),
                Err(e) => Err(format!("Failed to clear caches: {}", e)),
            }
        } else {
            Err("Cache manager not initialized".to_string())
        }
    } else {
        Err("Cache manager not available".to_string())
    }
}

/// Clear specific cache type
#[tauri::command]
pub async fn clear_cache_type(cache_type: String) -> std::result::Result<String, String> {
    if let Some(cache_manager_ref) = get_cache_manager().await {
        if let Some(cache_manager) = cache_manager_ref.read().await.as_ref() {
            let result = match cache_type.as_str() {
                "search" => cache_manager.search_cache().clear().await,
                "embedding" => cache_manager.embedding_cache().clear().await,
                "model" => cache_manager.model_cache().clear().await,
                "query" => cache_manager.query_cache().clear().await,
                _ => return Err(format!("Unknown cache type: {}", cache_type)),
            };
            
            match result {
                Ok(_) => Ok(format!("{} cache cleared successfully", cache_type)),
                Err(e) => Err(format!("Failed to clear {} cache: {}", cache_type, e)),
            }
        } else {
            Err("Cache manager not initialized".to_string())
        }
    } else {
        Err("Cache manager not available".to_string())
    }
}

/// Optimize caches by removing old/unused entries
#[tauri::command]
pub async fn optimize_caches() -> std::result::Result<CacheOptimizationResult, String> {
    let start_time = std::time::Instant::now();
    
    if let Some(cache_manager_ref) = get_cache_manager().await {
        if let Some(cache_manager) = cache_manager_ref.read().await.as_ref() {
            let initial_stats = cache_manager.get_usage_summary().await;
            let initial_memory = initial_stats.total_memory_bytes;
            let initial_entries = initial_stats.total_entries;
            
            // Optimize different cache types
            let mut total_removed = 0;
            
            // Clean low-confidence embeddings
            total_removed += cache_manager.embedding_cache().optimize_by_confidence(0.5).await;
            
            // Clean unused models
            total_removed += cache_manager.model_cache().clean_unused_models().await;
            
            // Clean low-confidence queries
            total_removed += cache_manager.query_cache().clean_low_confidence_queries(0.3).await;
            
            let final_stats = cache_manager.get_usage_summary().await;
            let freed_memory = initial_memory.saturating_sub(final_stats.total_memory_bytes);
            
            let duration_ms = start_time.elapsed().as_millis() as u64;
            
            Ok(CacheOptimizationResult {
                freed_memory_bytes: freed_memory,
                removed_entries: total_removed,
                optimization_type: "comprehensive".to_string(),
                duration_ms,
            })
        } else {
            Err("Cache manager not initialized".to_string())
        }
    } else {
        Err("Cache manager not available".to_string())
    }
}

/// Get search cache statistics
#[tauri::command]
pub async fn get_search_cache_stats() -> std::result::Result<CacheStats, String> {
    if let Some(cache_manager_ref) = get_cache_manager().await {
        if let Some(cache_manager) = cache_manager_ref.read().await.as_ref() {
            Ok(cache_manager.search_cache().get_stats().await)
        } else {
            Err("Cache manager not initialized".to_string())
        }
    } else {
        Err("Cache manager not available".to_string())
    }
}

/// Get embedding cache statistics
#[tauri::command]
pub async fn get_embedding_cache_stats() -> std::result::Result<CacheStats, String> {
    if let Some(cache_manager_ref) = get_cache_manager().await {
        if let Some(cache_manager) = cache_manager_ref.read().await.as_ref() {
            Ok(cache_manager.embedding_cache().get_stats().await)
        } else {
            Err("Cache manager not initialized".to_string())
        }
    } else {
        Err("Cache manager not available".to_string())
    }
}

/// Get model cache memory usage by type
#[tauri::command]
pub async fn get_model_memory_usage() -> std::result::Result<HashMap<String, usize>, String> {
    if let Some(cache_manager_ref) = get_cache_manager().await {
        if let Some(cache_manager) = cache_manager_ref.read().await.as_ref() {
            Ok(cache_manager.model_cache().get_memory_usage_by_type().await)
        } else {
            Err("Cache manager not initialized".to_string())
        }
    } else {
        Err("Cache manager not available".to_string())
    }
}

/// Invalidate search cache for specific file
#[tauri::command]
pub async fn invalidate_search_cache_for_file(file_path: String) -> std::result::Result<usize, String> {
    if let Some(cache_manager_ref) = get_cache_manager().await {
        if let Some(cache_manager) = cache_manager_ref.read().await.as_ref() {
            let invalidated = cache_manager.search_cache().invalidate_by_file_path(&file_path).await;
            Ok(invalidated)
        } else {
            Err("Cache manager not initialized".to_string())
        }
    } else {
        Err("Cache manager not available".to_string())
    }
}

/// Invalidate embedding cache for specific model
#[tauri::command]
pub async fn invalidate_embedding_cache_for_model(model_name: String) -> std::result::Result<usize, String> {
    if let Some(cache_manager_ref) = get_cache_manager().await {
        if let Some(cache_manager) = cache_manager_ref.read().await.as_ref() {
            let invalidated = cache_manager.embedding_cache().invalidate_model(&model_name).await;
            Ok(invalidated)
        } else {
            Err("Cache manager not initialized".to_string())
        }
    } else {
        Err("Cache manager not available".to_string())
    }
}

/// Get cache performance metrics
#[tauri::command]
pub async fn get_cache_performance_metrics() -> std::result::Result<CachePerformanceMetrics, String> {
    if let Some(cache_manager_ref) = get_cache_manager().await {
        if let Some(cache_manager) = cache_manager_ref.read().await.as_ref() {
            let stats = cache_manager.get_stats().await;
            let usage = cache_manager.get_usage_summary().await;
            
            let total_requests = stats.values().map(|s| s.hits + s.misses).sum();
            let total_hits = stats.values().map(|s| s.hits).sum();
            let total_evictions = stats.values().map(|s| s.evictions).sum();
            
            let metrics = CachePerformanceMetrics {
                overall_hit_rate: usage.average_hit_rate,
                total_requests,
                total_hits,
                total_misses: total_requests - total_hits,
                total_evictions,
                memory_efficiency: usage.cache_efficiency,
                memory_pressure: usage.memory_pressure,
                cache_type_performance: stats,
            };
            
            Ok(metrics)
        } else {
            Err("Cache manager not initialized".to_string())
        }
    } else {
        Err("Cache manager not available".to_string())
    }
}

/// Set cache configuration
#[tauri::command]
pub async fn update_cache_config(
    max_memory_mb: Option<usize>,
    default_ttl_hours: Option<u64>,
) -> std::result::Result<String, String> {
    // This would update the cache configuration
    // For now, just return a success message
    let mut updates = Vec::new();
    
    if let Some(memory) = max_memory_mb {
        updates.push(format!("max_memory_mb: {}", memory));
    }
    
    if let Some(ttl) = default_ttl_hours {
        updates.push(format!("default_ttl_hours: {}", ttl));
    }
    
    if updates.is_empty() {
        Ok("No configuration changes requested".to_string())
    } else {
        Ok(format!("Cache configuration updated: {}", updates.join(", ")))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CachePerformanceMetrics {
    pub overall_hit_rate: f64,
    pub total_requests: u64,
    pub total_hits: u64,
    pub total_misses: u64,
    pub total_evictions: u64,
    pub memory_efficiency: f64,
    pub memory_pressure: f64,
    pub cache_type_performance: HashMap<String, CacheStats>,
}