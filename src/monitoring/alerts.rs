use crate::error::{AppError, Result};
use crate::monitoring::{MonitoringConfig, Alert, AlertSeverity, SystemHealth, HealthStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

/// Alert rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub metric_name: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub severity: AlertSeverity,
    pub enabled: bool,
    pub cooldown_duration: Duration,
    pub notification_channels: Vec<NotificationChannel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Log,
    Email(String),
    Webhook(String),
    Desktop,
}

/// Alert state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertState {
    pub rule_id: String,
    pub is_firing: bool,
    pub last_triggered: Option<SystemTime>,
    pub last_resolved: Option<SystemTime>,
    pub trigger_count: u64,
    pub current_value: f64,
}

/// Alert manager for monitoring system health
pub struct AlertManager {
    config: MonitoringConfig,
    rules: Arc<RwLock<HashMap<String, AlertRule>>>,
    states: Arc<RwLock<HashMap<String, AlertState>>>,
    active_alerts: Arc<RwLock<Vec<Alert>>>,
    monitoring_task: Option<tokio::task::JoinHandle<()>>,
    is_running: Arc<RwLock<bool>>,
}

impl AlertManager {
    pub async fn new(config: MonitoringConfig) -> Result<Self> {
        let mut manager = Self {
            config,
            rules: Arc::new(RwLock::new(HashMap::new())),
            states: Arc::new(RwLock::new(HashMap::new())),
            active_alerts: Arc::new(RwLock::new(Vec::new())),
            monitoring_task: None,
            is_running: Arc::new(RwLock::new(false)),
        };

        // Initialize default alert rules
        manager.setup_default_rules().await?;

        Ok(manager)
    }

    /// Start alert monitoring
    pub async fn start_monitoring(&mut self) -> Result<()> {
        if *self.is_running.read().await {
            return Ok(());
        }

        *self.is_running.write().await = true;

        let rules = self.rules.clone();
        let states = self.states.clone();
        let active_alerts = self.active_alerts.clone();
        let is_running = self.is_running.clone();
        let check_interval = self.config.alert_check_interval;

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(check_interval);

            while *is_running.read().await {
                interval.tick().await;

                if let Err(e) = Self::check_alerts(&rules, &states, &active_alerts).await {
                    eprintln!("Alert check failed: {}", e);
                }
            }
        });

        self.monitoring_task = Some(handle);
        Ok(())
    }

    /// Stop alert monitoring
    pub async fn stop_monitoring(&mut self) -> Result<()> {
        *self.is_running.write().await = false;

        if let Some(handle) = self.monitoring_task.take() {
            handle.abort();
        }

        Ok(())
    }

    /// Add a new alert rule
    pub async fn add_rule(&self, rule: AlertRule) -> Result<()> {
        let mut rules = self.rules.write().await;
        let mut states = self.states.write().await;

        rules.insert(rule.id.clone(), rule.clone());
        states.insert(rule.id.clone(), AlertState {
            rule_id: rule.id,
            is_firing: false,
            last_triggered: None,
            last_resolved: None,
            trigger_count: 0,
            current_value: 0.0,
        });

        Ok(())
    }

    /// Remove an alert rule
    pub async fn remove_rule(&self, rule_id: &str) -> Result<bool> {
        let mut rules = self.rules.write().await;
        let mut states = self.states.write().await;

        let removed = rules.remove(rule_id).is_some();
        states.remove(rule_id);

        Ok(removed)
    }

    /// Update an alert rule
    pub async fn update_rule(&self, rule: AlertRule) -> Result<()> {
        let mut rules = self.rules.write().await;
        rules.insert(rule.id.clone(), rule);
        Ok(())
    }

    /// Get all alert rules
    pub async fn get_rules(&self) -> Vec<AlertRule> {
        self.rules.read().await.values().cloned().collect()
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Result<Vec<Alert>> {
        Ok(self.active_alerts.read().await.clone())
    }

    /// Create a new alert
    pub async fn create_alert(&self, alert: Alert) -> Result<()> {
        let mut active_alerts = self.active_alerts.write().await;
        
        // Check if alert already exists
        if !active_alerts.iter().any(|a| a.id == alert.id) {
            active_alerts.push(alert.clone());
            self.send_notification(&alert).await?;
        }

        Ok(())
    }

    /// Acknowledge an alert
    pub async fn acknowledge_alert(&self, alert_id: &str) -> Result<bool> {
        let mut active_alerts = self.active_alerts.write().await;
        
        if let Some(alert) = active_alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.acknowledged = true;
            return Ok(true);
        }

        Ok(false)
    }

    /// Resolve an alert
    pub async fn resolve_alert(&self, alert_id: &str) -> Result<bool> {
        let mut active_alerts = self.active_alerts.write().await;
        let mut states = self.states.write().await;

        // Find and remove the alert
        let alert_index = active_alerts.iter().position(|a| a.id == alert_id);
        if let Some(index) = alert_index {
            let mut alert = active_alerts.remove(index);
            alert.resolved = true;

            // Update alert state
            if let Some(state) = states.get_mut(&alert.component) {
                state.is_firing = false;
                state.last_resolved = Some(SystemTime::now());
            }

            return Ok(true);
        }

        Ok(false)
    }

    /// Get alert statistics
    pub async fn get_alert_stats(&self) -> AlertStats {
        let active_alerts = self.active_alerts.read().await;
        let states = self.states.read().await;

        let total_alerts = active_alerts.len();
        let critical_alerts = active_alerts.iter().filter(|a| matches!(a.severity, AlertSeverity::Critical)).count();
        let warning_alerts = active_alerts.iter().filter(|a| matches!(a.severity, AlertSeverity::Warning)).count();
        let acknowledged_alerts = active_alerts.iter().filter(|a| a.acknowledged).count();

        let total_triggers = states.values().map(|s| s.trigger_count).sum();

        AlertStats {
            total_alerts,
            critical_alerts,
            warning_alerts,
            acknowledged_alerts,
            total_triggers,
            rules_count: states.len(),
        }
    }

    // Private helper methods
    async fn setup_default_rules(&mut self) -> Result<()> {
        let default_rules = vec![
            AlertRule {
                id: "high_cpu_usage".to_string(),
                name: "High CPU Usage".to_string(),
                description: "CPU usage is above 80%".to_string(),
                metric_name: "system.cpu.usage".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 80.0,
                severity: AlertSeverity::Warning,
                enabled: true,
                cooldown_duration: Duration::from_secs(300), // 5 minutes
                notification_channels: vec![NotificationChannel::Log, NotificationChannel::Desktop],
            },
            AlertRule {
                id: "high_memory_usage".to_string(),
                name: "High Memory Usage".to_string(),
                description: "Memory usage is above 85%".to_string(),
                metric_name: "system.memory.usage_percent".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 85.0,
                severity: AlertSeverity::Warning,
                enabled: true,
                cooldown_duration: Duration::from_secs(300),
                notification_channels: vec![NotificationChannel::Log, NotificationChannel::Desktop],
            },
            AlertRule {
                id: "low_disk_space".to_string(),
                name: "Low Disk Space".to_string(),
                description: "Disk usage is above 90%".to_string(),
                metric_name: "system.disk.usage_percent".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 90.0,
                severity: AlertSeverity::Critical,
                enabled: true,
                cooldown_duration: Duration::from_secs(600), // 10 minutes
                notification_channels: vec![NotificationChannel::Log, NotificationChannel::Desktop],
            },
            AlertRule {
                id: "search_response_time".to_string(),
                name: "Slow Search Response".to_string(),
                description: "Search response time is above 5 seconds".to_string(),
                metric_name: "search.avg_response_time_ms".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 5000.0,
                severity: AlertSeverity::Warning,
                enabled: true,
                cooldown_duration: Duration::from_secs(180), // 3 minutes
                notification_channels: vec![NotificationChannel::Log],
            },
        ];

        for rule in default_rules {
            self.add_rule(rule).await?;
        }

        Ok(())
    }

    async fn check_alerts(
        rules: &Arc<RwLock<HashMap<String, AlertRule>>>,
        states: &Arc<RwLock<HashMap<String, AlertState>>>,
        active_alerts: &Arc<RwLock<Vec<Alert>>>,
    ) -> Result<()> {
        let rules_guard = rules.read().await;
        let mut states_guard = states.write().await;

        for rule in rules_guard.values() {
            if !rule.enabled {
                continue;
            }

            // Get current metric value (simplified - would integrate with metrics collector)
            let current_value = Self::get_metric_value(&rule.metric_name).await?;

            // Check condition
            let should_fire = Self::evaluate_condition(&rule.condition, current_value, rule.threshold);

            if let Some(state) = states_guard.get_mut(&rule.id) {
                state.current_value = current_value;

                // Check cooldown
                let in_cooldown = if let Some(last_triggered) = state.last_triggered {
                    SystemTime::now().duration_since(last_triggered).unwrap_or(Duration::ZERO) < rule.cooldown_duration
                } else {
                    false
                };

                if should_fire && !state.is_firing && !in_cooldown {
                    // Fire alert
                    state.is_firing = true;
                    state.last_triggered = Some(SystemTime::now());
                    state.trigger_count += 1;

                    let alert = Alert {
                        id: format!("{}-{}", rule.id, uuid::Uuid::new_v4()),
                        severity: rule.severity.clone(),
                        title: rule.name.clone(),
                        description: format!("{}: {} (current: {}, threshold: {})", 
                            rule.description, rule.metric_name, current_value, rule.threshold),
                        component: rule.metric_name.clone(),
                        metric: rule.metric_name.clone(),
                        threshold: rule.threshold,
                        current_value,
                        created_at: SystemTime::now(),
                        acknowledged: false,
                        resolved: false,
                    };

                    // Add to active alerts
                    active_alerts.write().await.push(alert);

                } else if !should_fire && state.is_firing {
                    // Resolve alert
                    state.is_firing = false;
                    state.last_resolved = Some(SystemTime::now());

                    // Remove from active alerts
                    let mut active = active_alerts.write().await;
                    active.retain(|a| !a.metric.starts_with(&rule.metric_name) || a.resolved);
                }
            }
        }

        Ok(())
    }

    async fn get_metric_value(metric_name: &str) -> Result<f64> {
        // This would integrate with the metrics collector
        // For now, return simulated values
        match metric_name {
            "system.cpu.usage" => Ok(rand::random::<f64>() * 100.0),
            "system.memory.usage_percent" => Ok(rand::random::<f64>() * 100.0),
            "system.disk.usage_percent" => Ok(rand::random::<f64>() * 100.0),
            "search.avg_response_time_ms" => Ok(rand::random::<f64>() * 1000.0),
            _ => Ok(0.0),
        }
    }

    fn evaluate_condition(condition: &AlertCondition, current: f64, threshold: f64) -> bool {
        match condition {
            AlertCondition::GreaterThan => current > threshold,
            AlertCondition::LessThan => current < threshold,
            AlertCondition::Equal => (current - threshold).abs() < f64::EPSILON,
            AlertCondition::NotEqual => (current - threshold).abs() >= f64::EPSILON,
            AlertCondition::GreaterThanOrEqual => current >= threshold,
            AlertCondition::LessThanOrEqual => current <= threshold,
        }
    }

    async fn send_notification(&self, alert: &Alert) -> Result<()> {
        // For now, just log the alert
        println!("🚨 ALERT [{}]: {} - {} ({})", 
            match alert.severity {
                AlertSeverity::Critical => "CRITICAL",
                AlertSeverity::Warning => "WARNING", 
                AlertSeverity::Info => "INFO",
                AlertSeverity::Emergency => "EMERGENCY",
            },
            alert.title,
            alert.description,
            alert.current_value
        );

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertStats {
    pub total_alerts: usize,
    pub critical_alerts: usize,
    pub warning_alerts: usize,
    pub acknowledged_alerts: usize,
    pub total_triggers: u64,
    pub rules_count: usize,
}

impl Drop for AlertManager {
    fn drop(&mut self) {
        if let Some(handle) = self.monitoring_task.take() {
            handle.abort();
        }
    }
}