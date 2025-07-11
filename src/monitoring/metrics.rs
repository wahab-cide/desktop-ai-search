use crate::error::{AppError, Result};
use crate::monitoring::{MonitoringConfig, SystemMetrics, PerformanceMetrics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// A single metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub name: String,
    pub value: f64,
    pub timestamp: SystemTime,
    pub tags: HashMap<String, String>,
    pub unit: MetricUnit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricUnit {
    Count,
    Bytes,
    Milliseconds,
    Seconds,
    Percent,
    Rate,
    Custom(String),
}

/// Time-series metric storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSeries {
    pub name: String,
    pub points: Vec<MetricPoint>,
    pub aggregation_type: AggregationType,
    pub retention_period: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationType {
    Sum,
    Average,
    Maximum,
    Minimum,
    Count,
    P95,
    P99,
}

/// Metrics collector and aggregator
pub struct MetricsCollector {
    config: MonitoringConfig,
    metrics: Arc<RwLock<HashMap<String, MetricSeries>>>,
    collection_task: Option<tokio::task::JoinHandle<()>>,
    is_running: Arc<RwLock<bool>>,
}

impl MetricsCollector {
    pub async fn new(config: MonitoringConfig) -> Result<Self> {
        Ok(Self {
            config,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            collection_task: None,
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start metric collection
    pub async fn start_collection(&mut self) -> Result<()> {
        if *self.is_running.read().await {
            return Ok(());
        }

        *self.is_running.write().await = true;
        
        let metrics = self.metrics.clone();
        let is_running = self.is_running.clone();
        let interval = self.config.metrics_collection_interval;

        let handle = tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            while *is_running.read().await {
                interval_timer.tick().await;
                
                // Collect system metrics
                if let Ok(system_metrics) = Self::collect_system_metrics().await {
                    Self::store_system_metrics(&metrics, system_metrics).await;
                }

                // Clean up old metrics
                Self::cleanup_old_metrics(&metrics, Duration::from_secs(7 * 24 * 60 * 60)).await;
            }
        });

        self.collection_task = Some(handle);
        Ok(())
    }

    /// Stop metric collection
    pub async fn stop_collection(&mut self) -> Result<()> {
        *self.is_running.write().await = false;

        if let Some(handle) = self.collection_task.take() {
            handle.abort();
        }

        Ok(())
    }

    /// Record a custom metric
    pub async fn record_metric(&self, name: &str, value: f64, tags: HashMap<String, String>) -> Result<()> {
        let point = MetricPoint {
            name: name.to_string(),
            value,
            timestamp: SystemTime::now(),
            tags,
            unit: MetricUnit::Custom("custom".to_string()),
        };

        let mut metrics = self.metrics.write().await;
        let series = metrics.entry(name.to_string()).or_insert_with(|| MetricSeries {
            name: name.to_string(),
            points: Vec::new(),
            aggregation_type: AggregationType::Average,
            retention_period: Duration::from_secs(7 * 24 * 60 * 60),
        });

        series.points.push(point);
        Ok(())
    }

    /// Get metrics for a time range
    pub async fn get_metrics_range(&self, start: SystemTime, end: SystemTime) -> Result<Vec<MetricPoint>> {
        let metrics = self.metrics.read().await;
        let mut result = Vec::new();

        for series in metrics.values() {
            for point in &series.points {
                if point.timestamp >= start && point.timestamp <= end {
                    result.push(point.clone());
                }
            }
        }

        result.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(result)
    }

    /// Get aggregated metrics
    pub async fn get_aggregated_metrics(&self, name: &str, aggregation: AggregationType, duration: Duration) -> Result<f64> {
        let metrics = self.metrics.read().await;
        let cutoff = SystemTime::now() - duration;

        if let Some(series) = metrics.get(name) {
            let recent_points: Vec<f64> = series
                .points
                .iter()
                .filter(|p| p.timestamp >= cutoff)
                .map(|p| p.value)
                .collect();

            if recent_points.is_empty() {
                return Ok(0.0);
            }

            let result = match aggregation {
                AggregationType::Sum => recent_points.iter().sum(),
                AggregationType::Average => recent_points.iter().sum::<f64>() / recent_points.len() as f64,
                AggregationType::Maximum => recent_points.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                AggregationType::Minimum => recent_points.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                AggregationType::Count => recent_points.len() as f64,
                AggregationType::P95 => Self::percentile(&recent_points, 0.95),
                AggregationType::P99 => Self::percentile(&recent_points, 0.99),
            };

            Ok(result)
        } else {
            Ok(0.0)
        }
    }

    /// Get all metric names
    pub async fn get_metric_names(&self) -> Vec<String> {
        self.metrics.read().await.keys().cloned().collect()
    }

    /// Export metrics in a specific format
    pub async fn export_metrics(&self, format: ExportFormat) -> Result<Vec<u8>> {
        let metrics = self.metrics.read().await;
        
        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&*metrics)
                    .map_err(|e| AppError::Unknown(format!("JSON serialization error: {}", e)))?;
                Ok(json.into_bytes())
            }
            ExportFormat::Csv => {
                let mut csv = String::new();
                csv.push_str("name,value,timestamp,tags\n");
                
                for series in metrics.values() {
                    for point in &series.points {
                        let tags_json = serde_json::to_string(&point.tags).unwrap_or_default();
                        csv.push_str(&format!(
                            "{},{},{},{}\n",
                            point.name,
                            point.value,
                            point.timestamp.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                            tags_json
                        ));
                    }
                }
                
                Ok(csv.into_bytes())
            }
            ExportFormat::Prometheus => {
                let mut prom = String::new();
                
                for series in metrics.values() {
                    if let Some(latest) = series.points.last() {
                        prom.push_str(&format!(
                            "# TYPE {} gauge\n{} {}\n",
                            latest.name,
                            latest.name,
                            latest.value
                        ));
                    }
                }
                
                Ok(prom.into_bytes())
            }
        }
    }

    // Private helper methods
    async fn collect_system_metrics() -> Result<SystemMetrics> {
        // This would collect actual system metrics
        // For now, return placeholder data
        Ok(SystemMetrics {
            cpu_usage: 0.0,
            memory_usage: crate::monitoring::MemoryMetrics {
                total_mb: 8192,
                used_mb: 4096,
                available_mb: 4096,
                usage_percent: 50.0,
                swap_total_mb: 2048,
                swap_used_mb: 512,
            },
            disk_usage: crate::monitoring::DiskMetrics {
                total_gb: 512,
                used_gb: 256,
                available_gb: 256,
                usage_percent: 50.0,
                io_read_bytes: 0,
                io_write_bytes: 0,
            },
            network_metrics: crate::monitoring::NetworkMetrics {
                bytes_sent: 0,
                bytes_received: 0,
                packets_sent: 0,
                packets_received: 0,
                errors: 0,
            },
            process_metrics: crate::monitoring::ProcessMetrics {
                pid: std::process::id(),
                cpu_usage: 0.0,
                memory_usage_mb: 256,
                thread_count: 8,
                file_handles: 32,
                uptime: Duration::from_secs(3600),
            },
            timestamp: SystemTime::now(),
        })
    }

    async fn store_system_metrics(metrics: &Arc<RwLock<HashMap<String, MetricSeries>>>, system_metrics: SystemMetrics) {
        let mut metrics_guard = metrics.write().await;
        let timestamp = SystemTime::now();

        // Store CPU metrics
        Self::add_metric_point(&mut metrics_guard, "system.cpu.usage", system_metrics.cpu_usage, timestamp, HashMap::new());

        // Store memory metrics
        Self::add_metric_point(&mut metrics_guard, "system.memory.total_mb", system_metrics.memory_usage.total_mb as f64, timestamp, HashMap::new());
        Self::add_metric_point(&mut metrics_guard, "system.memory.used_mb", system_metrics.memory_usage.used_mb as f64, timestamp, HashMap::new());
        Self::add_metric_point(&mut metrics_guard, "system.memory.usage_percent", system_metrics.memory_usage.usage_percent, timestamp, HashMap::new());

        // Store disk metrics
        Self::add_metric_point(&mut metrics_guard, "system.disk.total_gb", system_metrics.disk_usage.total_gb as f64, timestamp, HashMap::new());
        Self::add_metric_point(&mut metrics_guard, "system.disk.used_gb", system_metrics.disk_usage.used_gb as f64, timestamp, HashMap::new());
        Self::add_metric_point(&mut metrics_guard, "system.disk.usage_percent", system_metrics.disk_usage.usage_percent, timestamp, HashMap::new());

        // Store process metrics
        Self::add_metric_point(&mut metrics_guard, "process.cpu.usage", system_metrics.process_metrics.cpu_usage, timestamp, HashMap::new());
        Self::add_metric_point(&mut metrics_guard, "process.memory.usage_mb", system_metrics.process_metrics.memory_usage_mb as f64, timestamp, HashMap::new());
    }

    fn add_metric_point(
        metrics: &mut HashMap<String, MetricSeries>,
        name: &str,
        value: f64,
        timestamp: SystemTime,
        tags: HashMap<String, String>,
    ) {
        let point = MetricPoint {
            name: name.to_string(),
            value,
            timestamp,
            tags,
            unit: MetricUnit::Custom("system".to_string()),
        };

        let series = metrics.entry(name.to_string()).or_insert_with(|| MetricSeries {
            name: name.to_string(),
            points: Vec::new(),
            aggregation_type: AggregationType::Average,
            retention_period: Duration::from_secs(7 * 24 * 60 * 60),
        });

        series.points.push(point);
    }

    async fn cleanup_old_metrics(metrics: &Arc<RwLock<HashMap<String, MetricSeries>>>, retention_period: Duration) {
        let mut metrics_guard = metrics.write().await;
        let cutoff = SystemTime::now() - retention_period;

        for series in metrics_guard.values_mut() {
            series.points.retain(|point| point.timestamp >= cutoff);
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
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Prometheus,
}

impl Drop for MetricsCollector {
    fn drop(&mut self) {
        if let Some(handle) = self.collection_task.take() {
            handle.abort();
        }
    }
}