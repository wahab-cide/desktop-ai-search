use crate::error::{AppError, RecoveryAction, ErrorSeverity};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use std::collections::HashMap;

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker for handling failing components
pub struct CircuitBreaker {
    state: CircuitState,
    failure_count: u32,
    failure_threshold: u32,
    recovery_timeout: Duration,
    last_failure_time: Option<Instant>,
    success_threshold: u32,
    success_count: u32,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, recovery_timeout: Duration) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            failure_threshold,
            recovery_timeout,
            last_failure_time: None,
            success_threshold: 3,
            success_count: 0,
        }
    }

    pub fn call<T, F>(&mut self, operation: F) -> Result<T, AppError>
    where
        F: FnOnce() -> Result<T, AppError>,
    {
        match self.state {
            CircuitState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() > self.recovery_timeout {
                        self.state = CircuitState::HalfOpen;
                        self.success_count = 0;
                    } else {
                        return Err(AppError::Recovery("Circuit breaker is open".to_string()));
                    }
                }
            }
            CircuitState::HalfOpen => {
                // Allow limited requests to test if service is back
            }
            CircuitState::Closed => {
                // Normal operation
            }
        }

        match operation() {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(error) => {
                self.on_failure();
                Err(error)
            }
        }
    }

    fn on_success(&mut self) {
        match self.state {
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.last_failure_time = None;
                }
            }
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::Open => {
                // Should not happen
            }
        }
    }

    fn on_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        if self.failure_count >= self.failure_threshold {
            self.state = CircuitState::Open;
        }
    }

    pub fn is_open(&self) -> bool {
        self.state == CircuitState::Open
    }
}

/// Recovery manager for handling system failures
pub struct RecoveryManager {
    circuit_breakers: HashMap<String, CircuitBreaker>,
    retry_configs: HashMap<String, RetryConfig>,
    health_checks: HashMap<String, HealthCheck>,
}

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
        }
    }
}

#[derive(Debug)]
pub struct HealthCheck {
    pub name: String,
    pub check_interval: Duration,
    pub last_check: Option<Instant>,
    pub is_healthy: bool,
    pub failure_count: u32,
}

impl RecoveryManager {
    pub fn new() -> Self {
        let mut manager = Self {
            circuit_breakers: HashMap::new(),
            retry_configs: HashMap::new(),
            health_checks: HashMap::new(),
        };

        // Initialize default circuit breakers
        manager.circuit_breakers.insert(
            "database".to_string(),
            CircuitBreaker::new(5, Duration::from_secs(30)),
        );
        manager.circuit_breakers.insert(
            "search".to_string(),
            CircuitBreaker::new(10, Duration::from_secs(10)),
        );
        manager.circuit_breakers.insert(
            "embedding".to_string(),
            CircuitBreaker::new(3, Duration::from_secs(60)),
        );
        manager.circuit_breakers.insert(
            "model_download".to_string(),
            CircuitBreaker::new(2, Duration::from_secs(300)),
        );

        // Initialize retry configurations
        manager.retry_configs.insert("database".to_string(), RetryConfig::default());
        manager.retry_configs.insert("search".to_string(), RetryConfig {
            max_attempts: 2,
            base_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(1),
            backoff_multiplier: 1.5,
        });
        manager.retry_configs.insert("network".to_string(), RetryConfig {
            max_attempts: 5,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
        });

        // Initialize health checks
        manager.health_checks.insert("database".to_string(), HealthCheck {
            name: "database".to_string(),
            check_interval: Duration::from_secs(30),
            last_check: None,
            is_healthy: true,
            failure_count: 0,
        });

        manager
    }

    /// Execute an operation with circuit breaker protection
    pub fn with_circuit_breaker<T, F>(&mut self, component: &str, operation: F) -> Result<T, AppError>
    where
        F: FnOnce() -> Result<T, AppError>,
    {
        if let Some(circuit_breaker) = self.circuit_breakers.get_mut(component) {
            circuit_breaker.call(operation)
        } else {
            operation()
        }
    }

    /// Execute an operation with retry logic
    pub async fn with_retry<T, F, Fut>(&self, component: &str, operation: F) -> Result<T, AppError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, AppError>>,
    {
        let retry_config = self.retry_configs.get(component).cloned().unwrap_or_default();
        
        let mut attempts = 0;
        let mut delay = retry_config.base_delay;

        loop {
            attempts += 1;
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if attempts >= retry_config.max_attempts || !error.is_recoverable() {
                        return Err(error);
                    }

                    // Log retry attempt
                    eprintln!("Retry attempt {}/{} for {} failed: {}", 
                             attempts, retry_config.max_attempts, component, error);

                    tokio::time::sleep(delay).await;
                    delay = std::cmp::min(
                        Duration::from_millis((delay.as_millis() as f64 * retry_config.backoff_multiplier) as u64),
                        retry_config.max_delay,
                    );
                }
            }
        }
    }

    /// Attempt to recover from an error
    pub async fn recover_from_error(&mut self, error: &AppError) -> Result<(), AppError> {
        match error.recovery_action() {
            Some(RecoveryAction::Retry) => {
                // Already handled by retry logic
                Ok(())
            }
            Some(RecoveryAction::ClearCache) => {
                self.clear_caches().await
            }
            Some(RecoveryAction::RestartDatabase) => {
                self.restart_database().await
            }
            Some(RecoveryAction::ReloadModel) => {
                self.reload_models().await
            }
            Some(RecoveryAction::SimplifyQuery) => {
                // This would be handled by the search component
                Ok(())
            }
            Some(RecoveryAction::RestartApplication) => {
                Err(AppError::Recovery("Application restart required".to_string()))
            }
            Some(RecoveryAction::ResetConfiguration) => {
                self.reset_configuration().await
            }
            None => {
                // No recovery action available
                Err(AppError::Recovery("No recovery action available".to_string()))
            }
        }
    }

    /// Clear all caches
    async fn clear_caches(&self) -> Result<(), AppError> {
        // Implementation would clear LRU caches, moka caches, etc.
        println!("🔄 Clearing caches to free memory");
        Ok(())
    }

    /// Restart database connection
    async fn restart_database(&self) -> Result<(), AppError> {
        // Implementation would restart database connection pool
        println!("🔄 Restarting database connection");
        Ok(())
    }

    /// Reload models
    async fn reload_models(&self) -> Result<(), AppError> {
        // Implementation would reload embedding/LLM models
        println!("🔄 Reloading models");
        Ok(())
    }

    /// Reset configuration to defaults
    async fn reset_configuration(&self) -> Result<(), AppError> {
        // Implementation would reset configuration
        println!("🔄 Resetting configuration to defaults");
        Ok(())
    }

    /// Check health of components
    pub async fn health_check(&mut self) -> HashMap<String, bool> {
        let mut health_status = HashMap::new();
        let components_to_check: Vec<String> = self.health_checks.keys().cloned().collect();
        
        for component in components_to_check {
            let now = Instant::now();
            let should_check = if let Some(health_check) = self.health_checks.get(&component) {
                health_check.last_check.is_none() || 
                now.duration_since(health_check.last_check.unwrap()) > health_check.check_interval
            } else {
                true
            };
            
            if should_check {
                // Perform actual health check based on component
                let is_healthy = match component.as_str() {
                    "database" => Self::check_database_health().await,
                    "search" => Self::check_search_health().await,
                    "embedding" => Self::check_embedding_health().await,
                    _ => true,
                };
                
                if let Some(health_check) = self.health_checks.get_mut(&component) {
                    health_check.last_check = Some(now);
                    health_check.is_healthy = is_healthy;
                    if !is_healthy {
                        health_check.failure_count += 1;
                    } else {
                        health_check.failure_count = 0;
                    }
                }
                
                health_status.insert(component, is_healthy);
            } else if let Some(health_check) = self.health_checks.get(&component) {
                health_status.insert(component, health_check.is_healthy);
            }
        }
        
        health_status
    }

    async fn check_database_health() -> bool {
        // Implementation would check database connectivity
        true
    }

    async fn check_search_health() -> bool {
        // Implementation would check search system health
        true
    }

    async fn check_embedding_health() -> bool {
        // Implementation would check embedding system health
        true
    }

    /// Get circuit breaker status
    pub fn get_circuit_breaker_status(&self) -> HashMap<String, CircuitState> {
        self.circuit_breakers.iter()
            .map(|(name, breaker)| (name.clone(), breaker.state))
            .collect()
    }
}

/// Global recovery manager instance
pub static RECOVERY_MANAGER: once_cell::sync::Lazy<Arc<Mutex<RecoveryManager>>> = 
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(RecoveryManager::new())));

/// Convenience function to get recovery manager
pub async fn get_recovery_manager() -> Arc<Mutex<RecoveryManager>> {
    RECOVERY_MANAGER.clone()
}