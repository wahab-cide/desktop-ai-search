use crate::error::Result;
use crate::monitoring::{
    get_monitoring_manager,
    Alert, AlertSeverity, SystemHealth, PerformanceMetrics,
};
use crate::monitoring::alerts::{AlertRule, AlertStats, NotificationChannel, AlertCondition};
use crate::monitoring::metrics::{MetricPoint, ExportFormat, AggregationType};
use crate::monitoring::performance::{PerformanceEvent, PerformanceSummary};
use crate::monitoring::telemetry::{TelemetryStats, PrivacySettings, ExportFormat as TelemetryExportFormat};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tauri::State;

/// Get current system health status
#[tauri::command]
pub async fn get_system_health() -> std::result::Result<SystemHealth, String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.get_system_health().await
                .map_err(|e| format!("Failed to get system health: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Get performance metrics
#[tauri::command]
pub async fn get_performance_metrics() -> std::result::Result<PerformanceMetrics, String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.performance_monitor().read().await.get_current_metrics().await
                .map_err(|e| format!("Failed to get performance metrics: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Get performance summary for a duration
#[tauri::command]
pub async fn get_performance_summary(duration_seconds: u64) -> std::result::Result<PerformanceSummary, String> {
    let duration = Duration::from_secs(duration_seconds);
    
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.get_performance_summary(duration).await
                .map_err(|e| format!("Failed to get performance summary: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Record a custom metric
#[tauri::command]
pub async fn record_metric(name: String, value: f64, tags: HashMap<String, String>) -> std::result::Result<(), String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.record_metric(&name, value, tags).await
                .map_err(|e| format!("Failed to record metric: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Get metrics for a time range
#[tauri::command]
pub async fn get_metrics_range(start_timestamp: u64, end_timestamp: u64) -> std::result::Result<Vec<MetricPoint>, String> {
    let start = SystemTime::UNIX_EPOCH + Duration::from_secs(start_timestamp);
    let end = SystemTime::UNIX_EPOCH + Duration::from_secs(end_timestamp);
    
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.get_metrics_range(start, end).await
                .map_err(|e| format!("Failed to get metrics range: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Get active alerts
#[tauri::command]
pub async fn get_active_alerts() -> std::result::Result<Vec<Alert>, String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.alert_manager().read().await.get_active_alerts().await
                .map_err(|e| format!("Failed to get active alerts: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Create a new alert
#[tauri::command]
pub async fn create_alert(alert: Alert) -> std::result::Result<(), String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.create_alert(alert).await
                .map_err(|e| format!("Failed to create alert: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Acknowledge an alert
#[tauri::command]
pub async fn acknowledge_alert(alert_id: String) -> std::result::Result<bool, String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.alert_manager().read().await.acknowledge_alert(&alert_id).await
                .map_err(|e| format!("Failed to acknowledge alert: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Resolve an alert
#[tauri::command]
pub async fn resolve_alert(alert_id: String) -> std::result::Result<bool, String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.alert_manager().read().await.resolve_alert(&alert_id).await
                .map_err(|e| format!("Failed to resolve alert: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Get alert statistics
#[tauri::command]
pub async fn get_alert_stats() -> std::result::Result<AlertStats, String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            Ok(manager.alert_manager().read().await.get_alert_stats().await)
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Add a new alert rule
#[tauri::command]
pub async fn add_alert_rule(rule: AlertRule) -> std::result::Result<(), String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.alert_manager().read().await.add_rule(rule).await
                .map_err(|e| format!("Failed to add alert rule: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Remove an alert rule
#[tauri::command]
pub async fn remove_alert_rule(rule_id: String) -> std::result::Result<bool, String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.alert_manager().read().await.remove_rule(&rule_id).await
                .map_err(|e| format!("Failed to remove alert rule: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Get all alert rules
#[tauri::command]
pub async fn get_alert_rules() -> std::result::Result<Vec<AlertRule>, String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            Ok(manager.alert_manager().read().await.get_rules().await)
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Record a performance event
#[tauri::command]
pub async fn record_performance_event(event: PerformanceEvent) -> std::result::Result<(), String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.record_performance_event(event).await
                .map_err(|e| format!("Failed to record performance event: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Export telemetry data
#[tauri::command]
pub async fn export_telemetry() -> std::result::Result<Vec<u8>, String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.export_telemetry().await
                .map_err(|e| format!("Failed to export telemetry: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Get telemetry statistics
#[tauri::command]
pub async fn get_telemetry_stats() -> std::result::Result<TelemetryStats, String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            Ok(manager.telemetry_system().read().await.get_statistics().await)
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Update telemetry privacy settings
#[tauri::command]
pub async fn update_telemetry_privacy_settings(settings: PrivacySettings) -> std::result::Result<(), String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.write().await.as_mut() {
            manager.telemetry_system().write().await.update_privacy_settings(settings).await
                .map_err(|e| format!("Failed to update privacy settings: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Get telemetry privacy settings
#[tauri::command]
pub async fn get_telemetry_privacy_settings() -> std::result::Result<PrivacySettings, String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            Ok(manager.telemetry_system().read().await.get_privacy_settings())
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Clear all telemetry data
#[tauri::command]
pub async fn clear_telemetry_data() -> std::result::Result<(), String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.telemetry_system().read().await.clear_data().await
                .map_err(|e| format!("Failed to clear telemetry data: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Export metrics in specified format
#[tauri::command]
pub async fn export_metrics(format: String) -> std::result::Result<Vec<u8>, String> {
    let export_format = match format.as_str() {
        "json" => ExportFormat::Json,
        "csv" => ExportFormat::Csv,
        "prometheus" => ExportFormat::Prometheus,
        _ => return Err(format!("Unsupported export format: {}", format)),
    };

    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.metrics_collector().read().await.export_metrics(export_format).await
                .map_err(|e| format!("Failed to export metrics: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Get aggregated metric
#[tauri::command]
pub async fn get_aggregated_metric(
    name: String,
    aggregation: String,
    duration_seconds: u64
) -> std::result::Result<f64, String> {
    let agg_type = match aggregation.as_str() {
        "sum" => AggregationType::Sum,
        "average" => AggregationType::Average,
        "max" => AggregationType::Maximum,
        "min" => AggregationType::Minimum,
        "count" => AggregationType::Count,
        "p95" => AggregationType::P95,
        "p99" => AggregationType::P99,
        _ => return Err(format!("Unsupported aggregation type: {}", aggregation)),
    };

    let duration = Duration::from_secs(duration_seconds);

    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            manager.metrics_collector().read().await.get_aggregated_metrics(&name, agg_type, duration).await
                .map_err(|e| format!("Failed to get aggregated metric: {}", e))
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}

/// Get metric names
#[tauri::command]
pub async fn get_metric_names() -> std::result::Result<Vec<String>, String> {
    if let Some(manager_arc) = get_monitoring_manager().await {
        if let Some(manager) = manager_arc.read().await.as_ref() {
            Ok(manager.metrics_collector().read().await.get_metric_names().await)
        } else {
            Err("Monitoring manager not initialized".to_string())
        }
    } else {
        Err("Monitoring manager not available".to_string())
    }
}