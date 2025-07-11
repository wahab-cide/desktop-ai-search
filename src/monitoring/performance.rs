use crate::error::{AppError, Result};
use crate::monitoring::{MonitoringConfig, PerformanceMetrics, SearchPerformanceMetrics, IndexingPerformanceMetrics, CachePerformanceMetrics, ModelPerformanceMetrics};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

/// Performance event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceEvent {
    SearchQuery {
        query: String,
        start_time: SystemTime,
        duration: Duration,
        result_count: usize,
        success: bool,
    },
    IndexingOperation {
        file_path: String,
        start_time: SystemTime,
        duration: Duration,
        file_size: u64,
        success: bool,
    },
    CacheOperation {
        operation_type: CacheOperationType,
        cache_type: String,
        start_time: SystemTime,
        duration: Duration,
        success: bool,
    },
    ModelOperation {
        operation_type: ModelOperationType,
        model_name: String,
        start_time: SystemTime,
        duration: Duration,
        success: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheOperationType {
    Get,
    Put,
    Invalidate,
    Clear,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelOperationType {
    Load,
    Inference,
    EmbeddingGeneration,
    Unload,
}

/// Performance statistics over time windows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceWindow {
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub events: Vec<PerformanceEvent>,
    pub metrics: PerformanceMetrics,
}

/// Performance summary for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub time_range: Duration,
    pub total_events: usize,
    pub search_summary: SearchSummary,
    pub indexing_summary: IndexingSummary,
    pub cache_summary: CacheSummary,
    pub model_summary: ModelSummary,
    pub system_health_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSummary {
    pub total_queries: usize,
    pub avg_response_time: Duration,
    pub p95_response_time: Duration,
    pub p99_response_time: Duration,
    pub success_rate: f64,
    pub queries_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingSummary {
    pub total_files: usize,
    pub avg_processing_time: Duration,
    pub files_per_second: f64,
    pub success_rate: f64,
    pub total_size_processed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSummary {
    pub total_operations: usize,
    pub hit_rate: f64,
    pub avg_operation_time: Duration,
    pub operations_by_type: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    pub total_operations: usize,
    pub avg_inference_time: Duration,
    pub avg_loading_time: Duration,
    pub success_rate: f64,
    pub active_models: Vec<String>,
}

/// Performance monitor for tracking system performance
pub struct PerformanceMonitor {
    config: MonitoringConfig,
    events: Arc<RwLock<VecDeque<PerformanceEvent>>>,
    windows: Arc<RwLock<VecDeque<PerformanceWindow>>>,
    current_metrics: Arc<RwLock<PerformanceMetrics>>,
    is_running: Arc<RwLock<bool>>,
    processing_task: Option<tokio::task::JoinHandle<()>>,
}

impl PerformanceMonitor {
    pub async fn new(config: MonitoringConfig) -> Result<Self> {
        Ok(Self {
            config,
            events: Arc::new(RwLock::new(VecDeque::new())),
            windows: Arc::new(RwLock::new(VecDeque::new())),
            current_metrics: Arc::new(RwLock::new(Self::empty_metrics())),
            is_running: Arc::new(RwLock::new(false)),
            processing_task: None,
        })
    }

    /// Start performance monitoring
    pub async fn start_monitoring(&mut self) -> Result<()> {
        if *self.is_running.read().await {
            return Ok(());
        }

        *self.is_running.write().await = true;

        let events = self.events.clone();
        let windows = self.windows.clone();
        let current_metrics = self.current_metrics.clone();
        let is_running = self.is_running.clone();
        let sample_rate = self.config.performance_sample_rate;

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));

            while *is_running.read().await {
                interval.tick().await;

                // Process accumulated events
                if let Err(e) = Self::process_events(&events, &windows, &current_metrics, sample_rate).await {
                    eprintln!("Performance processing error: {}", e);
                }

                // Clean up old windows
                Self::cleanup_old_windows(&windows, Duration::from_secs(24 * 60 * 60)).await;
            }
        });

        self.processing_task = Some(handle);
        Ok(())
    }

    /// Stop performance monitoring
    pub async fn stop_monitoring(&mut self) -> Result<()> {
        *self.is_running.write().await = false;

        if let Some(handle) = self.processing_task.take() {
            handle.abort();
        }

        Ok(())
    }

    /// Record a performance event
    pub async fn record_event(&self, event: PerformanceEvent) -> Result<()> {
        // Apply sampling rate
        if rand::random::<f64>() > self.config.performance_sample_rate {
            return Ok(());
        }

        let mut events = self.events.write().await;
        events.push_back(event);

        // Limit queue size
        if events.len() > 10000 {
            events.pop_front();
        }

        Ok(())
    }

    /// Get current performance metrics
    pub async fn get_current_metrics(&self) -> Result<PerformanceMetrics> {
        Ok(self.current_metrics.read().await.clone())
    }

    /// Get performance summary for a duration
    pub async fn get_summary(&self, duration: Duration) -> Result<PerformanceSummary> {
        let cutoff = SystemTime::now() - duration;
        let windows = self.windows.read().await;

        // Collect events from the specified time range
        let relevant_events: Vec<PerformanceEvent> = windows
            .iter()
            .filter(|w| w.end_time >= cutoff)
            .flat_map(|w| w.events.iter())
            .cloned()
            .collect();

        let search_summary = self.calculate_search_summary(&relevant_events);
        let indexing_summary = self.calculate_indexing_summary(&relevant_events);
        let cache_summary = self.calculate_cache_summary(&relevant_events);
        let model_summary = self.calculate_model_summary(&relevant_events);

        let system_health_score = self.calculate_health_score(&search_summary, &indexing_summary).await;

        Ok(PerformanceSummary {
            time_range: duration,
            total_events: relevant_events.len(),
            search_summary,
            indexing_summary,
            cache_summary,
            model_summary,
            system_health_score,
        })
    }

    /// Get performance trends
    pub async fn get_trends(&self, duration: Duration) -> Result<Vec<PerformanceWindow>> {
        let cutoff = SystemTime::now() - duration;
        let windows = self.windows.read().await;

        Ok(windows
            .iter()
            .filter(|w| w.end_time >= cutoff)
            .cloned()
            .collect())
    }

    /// Get slow operations
    pub async fn get_slow_operations(&self, threshold: Duration) -> Result<Vec<PerformanceEvent>> {
        let events = self.events.read().await;

        Ok(events
            .iter()
            .filter(|event| {
                match event {
                    PerformanceEvent::SearchQuery { duration, .. } => *duration > threshold,
                    PerformanceEvent::IndexingOperation { duration, .. } => *duration > threshold,
                    PerformanceEvent::CacheOperation { duration, .. } => *duration > threshold,
                    PerformanceEvent::ModelOperation { duration, .. } => *duration > threshold,
                }
            })
            .cloned()
            .collect())
    }

    // Private helper methods
    async fn process_events(
        events: &Arc<RwLock<VecDeque<PerformanceEvent>>>,
        windows: &Arc<RwLock<VecDeque<PerformanceWindow>>>,
        current_metrics: &Arc<RwLock<PerformanceMetrics>>,
        _sample_rate: f64,
    ) -> Result<()> {
        let mut events_guard = events.write().await;
        if events_guard.is_empty() {
            return Ok(());
        }

        // Create a new performance window
        let window_events: Vec<PerformanceEvent> = events_guard.drain(..).collect();
        let window_start = SystemTime::now() - Duration::from_secs(30);
        let window_end = SystemTime::now();

        // Calculate metrics for this window
        let metrics = Self::calculate_window_metrics(&window_events);

        let window = PerformanceWindow {
            start_time: window_start,
            end_time: window_end,
            events: window_events,
            metrics: metrics.clone(),
        };

        // Store the window
        let mut windows_guard = windows.write().await;
        windows_guard.push_back(window);

        // Limit window history
        if windows_guard.len() > 1000 {
            windows_guard.pop_front();
        }

        // Update current metrics
        *current_metrics.write().await = metrics;

        Ok(())
    }

    fn calculate_window_metrics(events: &[PerformanceEvent]) -> PerformanceMetrics {
        let search_metrics = Self::calculate_search_metrics(events);
        let indexing_metrics = Self::calculate_indexing_metrics(events);
        let cache_metrics = Self::calculate_cache_metrics(events);
        let model_metrics = Self::calculate_model_metrics(events);

        PerformanceMetrics {
            search_performance: search_metrics,
            indexing_performance: indexing_metrics,
            cache_performance: cache_metrics,
            model_performance: model_metrics,
        }
    }

    fn calculate_search_metrics(events: &[PerformanceEvent]) -> SearchPerformanceMetrics {
        let search_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                PerformanceEvent::SearchQuery { duration, success, .. } => Some((duration, success)),
                _ => None,
            })
            .collect();

        if search_events.is_empty() {
            return SearchPerformanceMetrics {
                avg_response_time_ms: 0.0,
                p95_response_time_ms: 0.0,
                p99_response_time_ms: 0.0,
                queries_per_second: 0.0,
                success_rate: 1.0,
                error_rate: 0.0,
                timeout_rate: 0.0,
            };
        }

        let durations: Vec<f64> = search_events.iter().map(|(d, _)| d.as_millis() as f64).collect();
        let success_count = search_events.iter().filter(|(_, s)| **s).count();

        let avg_response_time = durations.iter().sum::<f64>() / durations.len() as f64;
        let p95_response_time = Self::percentile(&durations, 0.95);
        let p99_response_time = Self::percentile(&durations, 0.99);
        let success_rate = success_count as f64 / search_events.len() as f64;

        SearchPerformanceMetrics {
            avg_response_time_ms: avg_response_time,
            p95_response_time_ms: p95_response_time,
            p99_response_time_ms: p99_response_time,
            queries_per_second: search_events.len() as f64 / 30.0, // 30-second window
            success_rate,
            error_rate: 1.0 - success_rate,
            timeout_rate: 0.0, // Would need timeout tracking
        }
    }

    fn calculate_indexing_metrics(events: &[PerformanceEvent]) -> IndexingPerformanceMetrics {
        let indexing_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                PerformanceEvent::IndexingOperation { duration, file_size, success, .. } => {
                    Some((duration, file_size, success))
                }
                _ => None,
            })
            .collect();

        if indexing_events.is_empty() {
            return IndexingPerformanceMetrics {
                files_per_second: 0.0,
                avg_file_size_mb: 0.0,
                avg_processing_time_ms: 0.0,
                success_rate: 1.0,
                error_rate: 0.0,
                queue_size: 0,
            };
        }

        let durations: Vec<f64> = indexing_events.iter().map(|(d, _, _)| d.as_millis() as f64).collect();
        let total_size: u64 = indexing_events.iter().map(|(_, s, _)| **s).sum();
        let success_count = indexing_events.iter().filter(|(_, _, s)| **s).count();

        let avg_processing_time = durations.iter().sum::<f64>() / durations.len() as f64;
        let avg_file_size_mb = (total_size as f64) / (indexing_events.len() as f64 * 1024.0 * 1024.0);
        let success_rate = success_count as f64 / indexing_events.len() as f64;

        IndexingPerformanceMetrics {
            files_per_second: indexing_events.len() as f64 / 30.0,
            avg_file_size_mb,
            avg_processing_time_ms: avg_processing_time,
            success_rate,
            error_rate: 1.0 - success_rate,
            queue_size: 0, // Would need queue tracking
        }
    }

    fn calculate_cache_metrics(events: &[PerformanceEvent]) -> CachePerformanceMetrics {
        let cache_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                PerformanceEvent::CacheOperation { operation_type, success, .. } => {
                    Some((operation_type, success))
                }
                _ => None,
            })
            .collect();

        if cache_events.is_empty() {
            return CachePerformanceMetrics {
                hit_rate: 1.0,
                miss_rate: 0.0,
                eviction_rate: 0.0,
                memory_usage_mb: 0,
                entry_count: 0,
            };
        }

        let get_operations = cache_events.iter().filter(|(op, _)| matches!(op, CacheOperationType::Get)).count();
        let successful_gets = cache_events.iter().filter(|(op, s)| matches!(op, CacheOperationType::Get) && **s).count();

        let hit_rate = if get_operations > 0 {
            successful_gets as f64 / get_operations as f64
        } else {
            1.0
        };

        CachePerformanceMetrics {
            hit_rate,
            miss_rate: 1.0 - hit_rate,
            eviction_rate: 0.0, // Would need eviction tracking
            memory_usage_mb: 0, // Would need memory tracking
            entry_count: 0,     // Would need entry tracking
        }
    }

    fn calculate_model_metrics(events: &[PerformanceEvent]) -> ModelPerformanceMetrics {
        let model_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                PerformanceEvent::ModelOperation { operation_type, duration, .. } => {
                    Some((operation_type, duration))
                }
                _ => None,
            })
            .collect();

        if model_events.is_empty() {
            return ModelPerformanceMetrics {
                embedding_generation_time_ms: 0.0,
                model_load_time_ms: 0.0,
                inference_time_ms: 0.0,
                memory_usage_mb: 0,
                active_models: 0,
            };
        }

        let embedding_times: Vec<f64> = model_events
            .iter()
            .filter_map(|(op, d)| match op {
                ModelOperationType::EmbeddingGeneration => Some(d.as_millis() as f64),
                _ => None,
            })
            .collect();

        let load_times: Vec<f64> = model_events
            .iter()
            .filter_map(|(op, d)| match op {
                ModelOperationType::Load => Some(d.as_millis() as f64),
                _ => None,
            })
            .collect();

        let inference_times: Vec<f64> = model_events
            .iter()
            .filter_map(|(op, d)| match op {
                ModelOperationType::Inference => Some(d.as_millis() as f64),
                _ => None,
            })
            .collect();

        let avg_embedding_time = if !embedding_times.is_empty() {
            embedding_times.iter().sum::<f64>() / embedding_times.len() as f64
        } else {
            0.0
        };

        let avg_load_time = if !load_times.is_empty() {
            load_times.iter().sum::<f64>() / load_times.len() as f64
        } else {
            0.0
        };

        let avg_inference_time = if !inference_times.is_empty() {
            inference_times.iter().sum::<f64>() / inference_times.len() as f64
        } else {
            0.0
        };

        ModelPerformanceMetrics {
            embedding_generation_time_ms: avg_embedding_time,
            model_load_time_ms: avg_load_time,
            inference_time_ms: avg_inference_time,
            memory_usage_mb: 0, // Would need memory tracking
            active_models: 0,   // Would need model tracking
        }
    }

    fn calculate_search_summary(&self, events: &[PerformanceEvent]) -> SearchSummary {
        let search_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                PerformanceEvent::SearchQuery { duration, success, .. } => Some((duration, success)),
                _ => None,
            })
            .collect();

        if search_events.is_empty() {
            return SearchSummary {
                total_queries: 0,
                avg_response_time: Duration::ZERO,
                p95_response_time: Duration::ZERO,
                p99_response_time: Duration::ZERO,
                success_rate: 1.0,
                queries_per_second: 0.0,
            };
        }

        let durations: Vec<f64> = search_events.iter().map(|(d, _)| d.as_millis() as f64).collect();
        let success_count = search_events.iter().filter(|(_, s)| **s).count();

        let avg_response_time_ms = durations.iter().sum::<f64>() / durations.len() as f64;
        let p95_response_time_ms = Self::percentile(&durations, 0.95);
        let p99_response_time_ms = Self::percentile(&durations, 0.99);

        SearchSummary {
            total_queries: search_events.len(),
            avg_response_time: Duration::from_millis(avg_response_time_ms as u64),
            p95_response_time: Duration::from_millis(p95_response_time_ms as u64),
            p99_response_time: Duration::from_millis(p99_response_time_ms as u64),
            success_rate: success_count as f64 / search_events.len() as f64,
            queries_per_second: search_events.len() as f64 / 60.0, // Assume 1-minute window
        }
    }

    fn calculate_indexing_summary(&self, events: &[PerformanceEvent]) -> IndexingSummary {
        let indexing_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                PerformanceEvent::IndexingOperation { duration, file_size, success, .. } => {
                    Some((duration, file_size, success))
                }
                _ => None,
            })
            .collect();

        if indexing_events.is_empty() {
            return IndexingSummary {
                total_files: 0,
                avg_processing_time: Duration::ZERO,
                files_per_second: 0.0,
                success_rate: 1.0,
                total_size_processed: 0,
            };
        }

        let durations: Vec<f64> = indexing_events.iter().map(|(d, _, _)| d.as_millis() as f64).collect();
        let total_size: u64 = indexing_events.iter().map(|(_, s, _)| **s).sum();
        let success_count = indexing_events.iter().filter(|(_, _, s)| **s).count();

        let avg_processing_time_ms = durations.iter().sum::<f64>() / durations.len() as f64;

        IndexingSummary {
            total_files: indexing_events.len(),
            avg_processing_time: Duration::from_millis(avg_processing_time_ms as u64),
            files_per_second: indexing_events.len() as f64 / 60.0,
            success_rate: success_count as f64 / indexing_events.len() as f64,
            total_size_processed: total_size,
        }
    }

    fn calculate_cache_summary(&self, events: &[PerformanceEvent]) -> CacheSummary {
        let cache_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                PerformanceEvent::CacheOperation { operation_type, duration, success, .. } => {
                    Some((operation_type, duration, success))
                }
                _ => None,
            })
            .collect();

        if cache_events.is_empty() {
            return CacheSummary {
                total_operations: 0,
                hit_rate: 1.0,
                avg_operation_time: Duration::ZERO,
                operations_by_type: HashMap::new(),
            };
        }

        let get_operations = cache_events.iter().filter(|(op, _, _)| matches!(op, CacheOperationType::Get)).count();
        let successful_gets = cache_events.iter().filter(|(op, _, s)| matches!(op, CacheOperationType::Get) && **s).count();

        let hit_rate = if get_operations > 0 {
            successful_gets as f64 / get_operations as f64
        } else {
            1.0
        };

        let durations: Vec<f64> = cache_events.iter().map(|(_, d, _)| d.as_millis() as f64).collect();
        let avg_operation_time_ms = durations.iter().sum::<f64>() / durations.len() as f64;

        let mut operations_by_type = HashMap::new();
        for (op_type, _, _) in &cache_events {
            let key = format!("{:?}", op_type);
            *operations_by_type.entry(key).or_insert(0) += 1;
        }

        CacheSummary {
            total_operations: cache_events.len(),
            hit_rate,
            avg_operation_time: Duration::from_millis(avg_operation_time_ms as u64),
            operations_by_type,
        }
    }

    fn calculate_model_summary(&self, events: &[PerformanceEvent]) -> ModelSummary {
        let model_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                PerformanceEvent::ModelOperation { operation_type, duration, success, model_name, .. } => {
                    Some((operation_type, duration, success, model_name))
                }
                _ => None,
            })
            .collect();

        if model_events.is_empty() {
            return ModelSummary {
                total_operations: 0,
                avg_inference_time: Duration::ZERO,
                avg_loading_time: Duration::ZERO,
                success_rate: 1.0,
                active_models: Vec::new(),
            };
        }

        let inference_times: Vec<f64> = model_events
            .iter()
            .filter_map(|(op, d, _, _)| match op {
                ModelOperationType::Inference => Some(d.as_millis() as f64),
                _ => None,
            })
            .collect();

        let load_times: Vec<f64> = model_events
            .iter()
            .filter_map(|(op, d, _, _)| match op {
                ModelOperationType::Load => Some(d.as_millis() as f64),
                _ => None,
            })
            .collect();

        let success_count = model_events.iter().filter(|(_, _, s, _)| **s).count();

        let avg_inference_time_ms = if !inference_times.is_empty() {
            inference_times.iter().sum::<f64>() / inference_times.len() as f64
        } else {
            0.0
        };

        let avg_loading_time_ms = if !load_times.is_empty() {
            load_times.iter().sum::<f64>() / load_times.len() as f64
        } else {
            0.0
        };

        let active_models: Vec<String> = model_events
            .iter()
            .map(|(_, _, _, name)| name.to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        ModelSummary {
            total_operations: model_events.len(),
            avg_inference_time: Duration::from_millis(avg_inference_time_ms as u64),
            avg_loading_time: Duration::from_millis(avg_loading_time_ms as u64),
            success_rate: success_count as f64 / model_events.len() as f64,
            active_models,
        }
    }

    async fn calculate_health_score(&self, search_summary: &SearchSummary, indexing_summary: &IndexingSummary) -> f64 {
        let mut score = 100.0;

        // Penalize slow search response times
        if search_summary.avg_response_time > Duration::from_millis(1000) {
            score -= 20.0;
        } else if search_summary.avg_response_time > Duration::from_millis(500) {
            score -= 10.0;
        }

        // Penalize low success rates
        if search_summary.success_rate < 0.95 {
            score -= (1.0 - search_summary.success_rate) * 30.0;
        }

        if indexing_summary.success_rate < 0.95 {
            score -= (1.0 - indexing_summary.success_rate) * 20.0;
        }

        // Penalize slow indexing
        if indexing_summary.files_per_second < 1.0 {
            score -= 15.0;
        }

        score.max(0.0).min(100.0)
    }

    async fn cleanup_old_windows(windows: &Arc<RwLock<VecDeque<PerformanceWindow>>>, retention: Duration) {
        let mut windows_guard = windows.write().await;
        let cutoff = SystemTime::now() - retention;

        while let Some(front) = windows_guard.front() {
            if front.end_time < cutoff {
                windows_guard.pop_front();
            } else {
                break;
            }
        }
    }

    fn percentile(values: &[f64], percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((sorted.len() - 1) as f64 * percentile) as usize;
        sorted[index]
    }

    fn empty_metrics() -> PerformanceMetrics {
        PerformanceMetrics {
            search_performance: SearchPerformanceMetrics {
                avg_response_time_ms: 0.0,
                p95_response_time_ms: 0.0,
                p99_response_time_ms: 0.0,
                queries_per_second: 0.0,
                success_rate: 1.0,
                error_rate: 0.0,
                timeout_rate: 0.0,
            },
            indexing_performance: IndexingPerformanceMetrics {
                files_per_second: 0.0,
                avg_file_size_mb: 0.0,
                avg_processing_time_ms: 0.0,
                success_rate: 1.0,
                error_rate: 0.0,
                queue_size: 0,
            },
            cache_performance: CachePerformanceMetrics {
                hit_rate: 1.0,
                miss_rate: 0.0,
                eviction_rate: 0.0,
                memory_usage_mb: 0,
                entry_count: 0,
            },
            model_performance: ModelPerformanceMetrics {
                embedding_generation_time_ms: 0.0,
                model_load_time_ms: 0.0,
                inference_time_ms: 0.0,
                memory_usage_mb: 0,
                active_models: 0,
            },
        }
    }
}

impl Drop for PerformanceMonitor {
    fn drop(&mut self) {
        if let Some(handle) = self.processing_task.take() {
            handle.abort();
        }
    }
}