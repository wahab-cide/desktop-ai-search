use crate::error::{AppError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use sysinfo::{System, Pid};

pub mod metrics;
pub mod alerts;
pub mod performance;
pub mod telemetry;

/// System monitoring manager
pub struct MonitoringManager {
    metrics_collector: Arc<RwLock<metrics::MetricsCollector>>,
    alert_manager: Arc<RwLock<alerts::AlertManager>>,
    performance_monitor: Arc<RwLock<performance::PerformanceMonitor>>,
    telemetry_system: Arc<RwLock<telemetry::TelemetrySystem>>,
    config: MonitoringConfig,
    system_info: Arc<RwLock<System>>,
    start_time: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub enable_alerts: bool,
    pub enable_performance_monitoring: bool,
    pub enable_telemetry: bool,
    pub metrics_collection_interval: Duration,
    pub alert_check_interval: Duration,
    pub performance_sample_rate: f64,
    pub telemetry_batch_size: usize,
    pub retention_days: u32,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_alerts: true,
            enable_performance_monitoring: true,
            enable_telemetry: false, // Disabled by default for privacy
            metrics_collection_interval: Duration::from_secs(30),
            alert_check_interval: Duration::from_secs(60),
            performance_sample_rate: 0.1, // 10% sampling
            telemetry_batch_size: 100,
            retention_days: 7,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_status: HealthStatus,
    pub components: HashMap<String, ComponentHealth>,
    pub system_metrics: SystemMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub alerts: Vec<Alert>,
    pub uptime: Duration,
    pub last_updated: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub last_check: SystemTime,
    pub response_time: Option<Duration>,
    pub error_count: u64,
    pub success_rate: f64,
    pub details: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: MemoryMetrics,
    pub disk_usage: DiskMetrics,
    pub network_metrics: NetworkMetrics,
    pub process_metrics: ProcessMetrics,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub total_mb: u64,
    pub used_mb: u64,
    pub available_mb: u64,
    pub usage_percent: f64,
    pub swap_total_mb: u64,
    pub swap_used_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskMetrics {
    pub total_gb: u64,
    pub used_gb: u64,
    pub available_gb: u64,
    pub usage_percent: f64,
    pub io_read_bytes: u64,
    pub io_write_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub errors: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessMetrics {
    pub pid: u32,
    pub cpu_usage: f64,
    pub memory_usage_mb: u64,
    pub thread_count: u64,
    pub file_handles: u64,
    pub uptime: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub search_performance: SearchPerformanceMetrics,
    pub indexing_performance: IndexingPerformanceMetrics,
    pub cache_performance: CachePerformanceMetrics,
    pub model_performance: ModelPerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPerformanceMetrics {
    pub avg_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub p99_response_time_ms: f64,
    pub queries_per_second: f64,
    pub success_rate: f64,
    pub error_rate: f64,
    pub timeout_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingPerformanceMetrics {
    pub files_per_second: f64,
    pub avg_file_size_mb: f64,
    pub avg_processing_time_ms: f64,
    pub success_rate: f64,
    pub error_rate: f64,
    pub queue_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePerformanceMetrics {
    pub hit_rate: f64,
    pub miss_rate: f64,
    pub eviction_rate: f64,
    pub memory_usage_mb: u64,
    pub entry_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    pub embedding_generation_time_ms: f64,
    pub model_load_time_ms: f64,
    pub inference_time_ms: f64,
    pub memory_usage_mb: u64,
    pub active_models: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub severity: AlertSeverity,
    pub title: String,
    pub description: String,
    pub component: String,
    pub metric: String,
    pub threshold: f64,
    pub current_value: f64,
    pub created_at: SystemTime,
    pub acknowledged: bool,
    pub resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

impl MonitoringManager {
    pub async fn new(config: MonitoringConfig) -> Result<Self> {
        let metrics_collector = Arc::new(RwLock::new(metrics::MetricsCollector::new(config.clone()).await?));
        let alert_manager = Arc::new(RwLock::new(alerts::AlertManager::new(config.clone()).await?));
        let performance_monitor = Arc::new(RwLock::new(performance::PerformanceMonitor::new(config.clone()).await?));
        let telemetry_system = Arc::new(RwLock::new(telemetry::TelemetrySystem::new(config.clone()).await?));
        let system_info = Arc::new(RwLock::new(System::new_all()));
        
        Ok(Self {
            metrics_collector,
            alert_manager,
            performance_monitor,
            telemetry_system,
            config,
            system_info,
            start_time: SystemTime::now(),
        })
    }

    /// Start monitoring services
    pub async fn start_monitoring(&self) -> Result<()> {
        if self.config.enable_metrics {
            self.metrics_collector.write().await.start_collection().await?;
        }

        if self.config.enable_alerts {
            self.alert_manager.write().await.start_monitoring().await?;
        }

        if self.config.enable_performance_monitoring {
            self.performance_monitor.write().await.start_monitoring().await?;
        }

        if self.config.enable_telemetry {
            self.telemetry_system.write().await.start_collection().await?;
        }

        Ok(())
    }

    /// Stop monitoring services
    pub async fn stop_monitoring(&self) -> Result<()> {
        self.metrics_collector.write().await.stop_collection().await?;
        self.alert_manager.write().await.stop_monitoring().await?;
        self.performance_monitor.write().await.stop_monitoring().await?;
        self.telemetry_system.write().await.stop_collection().await?;
        Ok(())
    }

    /// Get current system health
    pub async fn get_system_health(&self) -> Result<SystemHealth> {
        let overall_status = self.calculate_overall_health().await?;
        let components = self.get_component_health().await?;
        let system_metrics = self.collect_system_metrics().await?;
        let performance_metrics = self.performance_monitor.read().await.get_current_metrics().await?;
        let alerts = self.alert_manager.read().await.get_active_alerts().await?;
        let uptime = SystemTime::now().duration_since(self.start_time).unwrap_or(Duration::ZERO);

        Ok(SystemHealth {
            overall_status,
            components,
            system_metrics,
            performance_metrics,
            alerts,
            uptime,
            last_updated: SystemTime::now(),
        })
    }

    /// Record a custom metric
    pub async fn record_metric(&self, name: &str, value: f64, tags: HashMap<String, String>) -> Result<()> {
        self.metrics_collector.read().await.record_metric(name, value, tags).await
    }

    /// Record a performance event
    pub async fn record_performance_event(&self, event: performance::PerformanceEvent) -> Result<()> {
        self.performance_monitor.read().await.record_event(event).await
    }

    /// Create an alert
    pub async fn create_alert(&self, alert: Alert) -> Result<()> {
        self.alert_manager.read().await.create_alert(alert).await
    }

    /// Get metrics for a time range
    pub async fn get_metrics_range(&self, start: SystemTime, end: SystemTime) -> Result<Vec<metrics::MetricPoint>> {
        self.metrics_collector.read().await.get_metrics_range(start, end).await
    }

    /// Get performance summary
    pub async fn get_performance_summary(&self, duration: Duration) -> Result<performance::PerformanceSummary> {
        self.performance_monitor.read().await.get_summary(duration).await
    }

    /// Export telemetry data
    pub async fn export_telemetry(&self) -> Result<Vec<u8>> {
        self.telemetry_system.read().await.export_data().await
    }

    /// Get metrics collector
    pub fn metrics_collector(&self) -> &Arc<RwLock<metrics::MetricsCollector>> {
        &self.metrics_collector
    }

    /// Get alert manager
    pub fn alert_manager(&self) -> &Arc<RwLock<alerts::AlertManager>> {
        &self.alert_manager
    }

    /// Get performance monitor
    pub fn performance_monitor(&self) -> &Arc<RwLock<performance::PerformanceMonitor>> {
        &self.performance_monitor
    }

    /// Get telemetry system
    pub fn telemetry_system(&self) -> &Arc<RwLock<telemetry::TelemetrySystem>> {
        &self.telemetry_system
    }

    // Private helper methods
    async fn calculate_overall_health(&self) -> Result<HealthStatus> {
        let components = self.get_component_health().await?;
        let critical_count = components.values().filter(|c| matches!(c.status, HealthStatus::Critical)).count();
        let warning_count = components.values().filter(|c| matches!(c.status, HealthStatus::Warning)).count();

        if critical_count > 0 {
            Ok(HealthStatus::Critical)
        } else if warning_count > 0 {
            Ok(HealthStatus::Warning)
        } else {
            Ok(HealthStatus::Healthy)
        }
    }

    async fn get_component_health(&self) -> Result<HashMap<String, ComponentHealth>> {
        let mut components = HashMap::new();

        // Check database health
        components.insert("database".to_string(), self.check_database_health().await?);
        
        // Check cache health
        components.insert("cache".to_string(), self.check_cache_health().await?);
        
        // Check search engine health
        components.insert("search".to_string(), self.check_search_health().await?);
        
        // Check embedding system health
        components.insert("embedding".to_string(), self.check_embedding_health().await?);
        
        // Check model system health
        components.insert("models".to_string(), self.check_model_health().await?);

        Ok(components)
    }

    async fn check_database_health(&self) -> Result<ComponentHealth> {
        // This would perform actual database health checks
        Ok(ComponentHealth {
            name: "database".to_string(),
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            response_time: Some(Duration::from_millis(5)),
            error_count: 0,
            success_rate: 1.0,
            details: HashMap::new(),
        })
    }

    async fn check_cache_health(&self) -> Result<ComponentHealth> {
        // This would perform actual cache health checks
        Ok(ComponentHealth {
            name: "cache".to_string(),
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            response_time: Some(Duration::from_millis(1)),
            error_count: 0,
            success_rate: 1.0,
            details: HashMap::new(),
        })
    }

    async fn check_search_health(&self) -> Result<ComponentHealth> {
        // This would perform actual search health checks
        Ok(ComponentHealth {
            name: "search".to_string(),
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            response_time: Some(Duration::from_millis(50)),
            error_count: 0,
            success_rate: 1.0,
            details: HashMap::new(),
        })
    }

    async fn check_embedding_health(&self) -> Result<ComponentHealth> {
        // This would perform actual embedding health checks
        Ok(ComponentHealth {
            name: "embedding".to_string(),
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            response_time: Some(Duration::from_millis(100)),
            error_count: 0,
            success_rate: 1.0,
            details: HashMap::new(),
        })
    }

    async fn check_model_health(&self) -> Result<ComponentHealth> {
        // This would perform actual model health checks
        Ok(ComponentHealth {
            name: "models".to_string(),
            status: HealthStatus::Healthy,
            last_check: SystemTime::now(),
            response_time: Some(Duration::from_millis(200)),
            error_count: 0,
            success_rate: 1.0,
            details: HashMap::new(),
        })
    }

    async fn collect_system_metrics(&self) -> Result<SystemMetrics> {
        let mut system = self.system_info.write().await;
        system.refresh_all();

        // Memory metrics
        let memory_metrics = MemoryMetrics {
            total_mb: system.total_memory() / 1024 / 1024,
            used_mb: system.used_memory() / 1024 / 1024,
            available_mb: system.available_memory() / 1024 / 1024,
            usage_percent: (system.used_memory() as f64 / system.total_memory() as f64) * 100.0,
            swap_total_mb: system.total_swap() / 1024 / 1024,
            swap_used_mb: system.used_swap() / 1024 / 1024,
        };

        // Disk metrics (simplified)
        let disk_metrics = DiskMetrics {
            total_gb: 0,
            used_gb: 0,
            available_gb: 0,
            usage_percent: 0.0,
            io_read_bytes: 0,
            io_write_bytes: 0,
        };

        // Network metrics (simplified)
        let network_metrics = NetworkMetrics {
            bytes_sent: 0,
            bytes_received: 0,
            packets_sent: 0,
            packets_received: 0,
            errors: 0,
        };

        // Process metrics
        let current_pid = std::process::id();
        let process_metrics = if let Some(process) = system.process(Pid::from(current_pid as usize)) {
            ProcessMetrics {
                pid: current_pid,
                cpu_usage: process.cpu_usage() as f64,
                memory_usage_mb: process.memory() / 1024 / 1024,
                thread_count: 0, // Not available in current sysinfo version
                file_handles: 0, // Not available in current sysinfo version
                uptime: SystemTime::now().duration_since(self.start_time).unwrap_or(Duration::ZERO),
            }
        } else {
            ProcessMetrics {
                pid: current_pid,
                cpu_usage: 0.0,
                memory_usage_mb: 0,
                thread_count: 0,
                file_handles: 0,
                uptime: SystemTime::now().duration_since(self.start_time).unwrap_or(Duration::ZERO),
            }
        };

        Ok(SystemMetrics {
            cpu_usage: system.global_cpu_info().cpu_usage() as f64,
            memory_usage: memory_metrics,
            disk_usage: disk_metrics,
            network_metrics,
            process_metrics,
            timestamp: SystemTime::now(),
        })
    }
}

/// Global monitoring manager instance
pub static MONITORING_MANAGER: once_cell::sync::Lazy<Arc<tokio::sync::RwLock<Option<MonitoringManager>>>> = 
    once_cell::sync::Lazy::new(|| Arc::new(tokio::sync::RwLock::new(None)));

/// Initialize the global monitoring manager
pub async fn init_monitoring_manager(config: MonitoringConfig) -> Result<()> {
    let manager = MonitoringManager::new(config).await?;
    manager.start_monitoring().await?;
    *MONITORING_MANAGER.write().await = Some(manager);
    Ok(())
}

/// Get the global monitoring manager
pub async fn get_monitoring_manager() -> Option<Arc<tokio::sync::RwLock<Option<MonitoringManager>>>> {
    if MONITORING_MANAGER.read().await.is_some() {
        Some(MONITORING_MANAGER.clone())
    } else {
        None
    }
}