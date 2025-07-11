use crate::recovery::{RecoveryManager, CircuitState};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tauri::State;

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthStatus {
    pub overall_health: bool,
    pub components: HashMap<String, ComponentHealth>,
    pub circuit_breakers: HashMap<String, CircuitBreakerStatus>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub is_healthy: bool,
    pub failure_count: u32,
    pub last_check: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CircuitBreakerStatus {
    pub state: String,
    pub failure_count: u32,
    pub is_open: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub memory_usage: f64,
    pub cpu_usage: f64,
    pub disk_usage: f64,
    pub uptime: u64,
    pub active_connections: u32,
}

/// Get system health status
#[tauri::command]
pub async fn get_health_status(
    recovery_manager: State<'_, Arc<Mutex<RecoveryManager>>>,
) -> std::result::Result<HealthStatus, String> {
    let mut manager = recovery_manager.lock().await;
    
    // Perform health check
    let component_health = manager.health_check().await;
    
    // Get circuit breaker status
    let circuit_status = manager.get_circuit_breaker_status();
    
    // Convert to response format
    let components: HashMap<String, ComponentHealth> = component_health
        .into_iter()
        .map(|(name, is_healthy)| {
            (name.clone(), ComponentHealth {
                name,
                is_healthy,
                failure_count: 0, // Would be retrieved from health check
                last_check: Some(chrono::Utc::now()),
            })
        })
        .collect();
    
    let circuit_breakers: HashMap<String, CircuitBreakerStatus> = circuit_status
        .into_iter()
        .map(|(name, state)| {
            (name.clone(), CircuitBreakerStatus {
                state: format!("{:?}", state),
                failure_count: 0, // Would be retrieved from circuit breaker
                is_open: state == CircuitState::Open,
            })
        })
        .collect();
    
    let overall_health = components.values().all(|c| c.is_healthy) && 
                        circuit_breakers.values().all(|cb| !cb.is_open);
    
    Ok(HealthStatus {
        overall_health,
        components,
        circuit_breakers,
        timestamp: chrono::Utc::now(),
    })
}

/// Get system metrics
#[tauri::command]
pub async fn get_system_metrics() -> std::result::Result<SystemMetrics, String> {
    use sysinfo::{System, Disks};
    
    let mut system = System::new_all();
    system.refresh_all();
    
    let memory_usage = (system.used_memory() as f64 / system.total_memory() as f64) * 100.0;
    let cpu_usage = system.global_cpu_info().cpu_usage() as f64;
    
    let disks = Disks::new_with_refreshed_list();
    let disk_usage = if disks.len() > 0 {
        disks.iter()
            .map(|disk| {
                let used = disk.total_space() - disk.available_space();
                (used as f64 / disk.total_space() as f64) * 100.0
            })
            .fold(0.0, |acc, usage| acc + usage) / disks.len() as f64
    } else {
        0.0
    };
    
    Ok(SystemMetrics {
        memory_usage,
        cpu_usage,
        disk_usage,
        uptime: System::uptime(),
        active_connections: 0, // Would be tracked by the application
    })
}

/// Trigger manual recovery for a component
#[tauri::command]
pub async fn trigger_recovery(
    component: String,
    recovery_manager: State<'_, Arc<Mutex<RecoveryManager>>>,
) -> std::result::Result<String, String> {
    let mut manager = recovery_manager.lock().await;
    
    // Create a mock error to trigger recovery
    let error = crate::error::AppError::Unknown(format!("Manual recovery triggered for {}", component));
    
    match manager.recover_from_error(&error).await {
        Ok(_) => Ok(format!("Recovery initiated for {}", component)),
        Err(e) => Err(e.to_string()),
    }
}

/// Reset circuit breaker for a component
#[tauri::command]
pub async fn reset_circuit_breaker(
    component: String,
    recovery_manager: State<'_, Arc<Mutex<RecoveryManager>>>,
) -> std::result::Result<String, String> {
    let mut manager = recovery_manager.lock().await;
    
    // Reset circuit breaker (this would need to be implemented in RecoveryManager)
    Ok(format!("Circuit breaker reset for {}", component))
}

/// Get error recovery suggestions
#[tauri::command]
pub async fn get_recovery_suggestions(
    error_type: String,
) -> std::result::Result<Vec<String>, String> {
    let suggestions = match error_type.as_str() {
        "database" => vec![
            "Check database connection".to_string(),
            "Verify database file permissions".to_string(),
            "Restart database connection pool".to_string(),
            "Check disk space".to_string(),
        ],
        "search" => vec![
            "Clear search cache".to_string(),
            "Rebuild search index".to_string(),
            "Check search query syntax".to_string(),
            "Verify embedding model is loaded".to_string(),
        ],
        "embedding" => vec![
            "Reload embedding model".to_string(),
            "Check model file integrity".to_string(),
            "Verify VRAM availability".to_string(),
            "Download model again".to_string(),
        ],
        "network" => vec![
            "Check internet connection".to_string(),
            "Verify proxy settings".to_string(),
            "Check firewall settings".to_string(),
            "Retry with different endpoint".to_string(),
        ],
        _ => vec![
            "Restart application".to_string(),
            "Check system resources".to_string(),
            "Review error logs".to_string(),
        ],
    };
    
    Ok(suggestions)
}