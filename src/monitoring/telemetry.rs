use crate::error::{AppError, Result};
use crate::monitoring::{MonitoringConfig, SystemMetrics, PerformanceMetrics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Telemetry data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TelemetryData {
    Usage(UsageData),
    Performance(PerformanceData),
    Error(ErrorData),
    Feature(FeatureUsageData),
    System(SystemData),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageData {
    pub session_id: String,
    pub user_id: Option<String>,
    pub timestamp: SystemTime,
    pub action: String,
    pub metadata: HashMap<String, String>,
    pub duration: Option<Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceData {
    pub operation: String,
    pub duration_ms: u64,
    pub success: bool,
    pub timestamp: SystemTime,
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorData {
    pub error_type: String,
    pub error_message: String,
    pub stack_trace: Option<String>,
    pub timestamp: SystemTime,
    pub context: HashMap<String, String>,
    pub severity: ErrorSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureUsageData {
    pub feature_name: String,
    pub usage_count: u64,
    pub first_used: SystemTime,
    pub last_used: SystemTime,
    pub user_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemData {
    pub os_name: String,
    pub os_version: String,
    pub architecture: String,
    pub total_memory: u64,
    pub cpu_count: usize,
    pub app_version: String,
    pub timestamp: SystemTime,
}

/// Telemetry batch for efficient transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryBatch {
    pub batch_id: String,
    pub session_id: String,
    pub created_at: SystemTime,
    pub data: Vec<TelemetryData>,
    pub metadata: HashMap<String, String>,
}

/// Telemetry export format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
    Parquet,
    Custom(String),
}

/// Privacy settings for telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacySettings {
    pub collect_usage: bool,
    pub collect_performance: bool,
    pub collect_errors: bool,
    pub collect_system_info: bool,
    pub anonymize_data: bool,
    pub retention_days: u32,
}

impl Default for PrivacySettings {
    fn default() -> Self {
        Self {
            collect_usage: false,      // Disabled by default for privacy
            collect_performance: false,
            collect_errors: true,      // Errors can help improve the app
            collect_system_info: false,
            anonymize_data: true,
            retention_days: 30,
        }
    }
}

/// Telemetry system for collecting and exporting usage data
pub struct TelemetrySystem {
    config: MonitoringConfig,
    privacy_settings: PrivacySettings,
    data_buffer: Arc<RwLock<Vec<TelemetryData>>>,
    batches: Arc<RwLock<Vec<TelemetryBatch>>>,
    session_id: String,
    collection_task: Option<tokio::task::JoinHandle<()>>,
    is_running: Arc<RwLock<bool>>,
}

impl TelemetrySystem {
    pub async fn new(config: MonitoringConfig) -> Result<Self> {
        let session_id = uuid::Uuid::new_v4().to_string();
        
        Ok(Self {
            config,
            privacy_settings: PrivacySettings::default(),
            data_buffer: Arc::new(RwLock::new(Vec::new())),
            batches: Arc::new(RwLock::new(Vec::new())),
            session_id,
            collection_task: None,
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start telemetry collection
    pub async fn start_collection(&mut self) -> Result<()> {
        if *self.is_running.read().await {
            return Ok(());
        }

        *self.is_running.write().await = true;

        // Record system information if enabled
        if self.privacy_settings.collect_system_info {
            self.record_system_info().await?;
        }

        let data_buffer = self.data_buffer.clone();
        let batches = self.batches.clone();
        let is_running = self.is_running.clone();
        let session_id = self.session_id.clone();
        let batch_size = self.config.telemetry_batch_size;
        let privacy_settings = self.privacy_settings.clone();

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // Batch every minute

            while *is_running.read().await {
                interval.tick().await;

                if let Err(e) = Self::process_telemetry_data(&data_buffer, &batches, &session_id, batch_size, &privacy_settings).await {
                    eprintln!("Telemetry processing error: {}", e);
                }
            }
        });

        self.collection_task = Some(handle);
        Ok(())
    }

    /// Stop telemetry collection
    pub async fn stop_collection(&mut self) -> Result<()> {
        *self.is_running.write().await = false;

        if let Some(handle) = self.collection_task.take() {
            handle.abort();
        }

        // Process any remaining data
        self.flush_data().await?;

        Ok(())
    }

    /// Record usage data
    pub async fn record_usage(&self, action: &str, metadata: HashMap<String, String>, duration: Option<Duration>) -> Result<()> {
        if !self.privacy_settings.collect_usage {
            return Ok(());
        }

        let usage_data = UsageData {
            session_id: self.session_id.clone(),
            user_id: None, // Would be set if user identification is enabled
            timestamp: SystemTime::now(),
            action: action.to_string(),
            metadata: if self.privacy_settings.anonymize_data {
                self.anonymize_metadata(metadata)
            } else {
                metadata
            },
            duration,
        };

        self.data_buffer.write().await.push(TelemetryData::Usage(usage_data));
        Ok(())
    }

    /// Record performance data
    pub async fn record_performance(&self, operation: &str, duration_ms: u64, success: bool, context: HashMap<String, String>) -> Result<()> {
        if !self.privacy_settings.collect_performance {
            return Ok(());
        }

        let perf_data = PerformanceData {
            operation: operation.to_string(),
            duration_ms,
            success,
            timestamp: SystemTime::now(),
            context: if self.privacy_settings.anonymize_data {
                self.anonymize_metadata(context)
            } else {
                context
            },
        };

        self.data_buffer.write().await.push(TelemetryData::Performance(perf_data));
        Ok(())
    }

    /// Record error data
    pub async fn record_error(&self, error_type: &str, error_message: &str, stack_trace: Option<String>, context: HashMap<String, String>, severity: ErrorSeverity) -> Result<()> {
        if !self.privacy_settings.collect_errors {
            return Ok(());
        }

        let error_data = ErrorData {
            error_type: error_type.to_string(),
            error_message: if self.privacy_settings.anonymize_data {
                self.anonymize_error_message(error_message)
            } else {
                error_message.to_string()
            },
            stack_trace: if self.privacy_settings.anonymize_data {
                stack_trace.map(|st| self.anonymize_stack_trace(&st))
            } else {
                stack_trace
            },
            timestamp: SystemTime::now(),
            context: if self.privacy_settings.anonymize_data {
                self.anonymize_metadata(context)
            } else {
                context
            },
            severity,
        };

        self.data_buffer.write().await.push(TelemetryData::Error(error_data));
        Ok(())
    }

    /// Record feature usage
    pub async fn record_feature_usage(&self, feature_name: &str) -> Result<()> {
        if !self.privacy_settings.collect_usage {
            return Ok(());
        }

        let feature_data = FeatureUsageData {
            feature_name: feature_name.to_string(),
            usage_count: 1,
            first_used: SystemTime::now(),
            last_used: SystemTime::now(),
            user_id: None,
        };

        self.data_buffer.write().await.push(TelemetryData::Feature(feature_data));
        Ok(())
    }

    /// Update privacy settings
    pub async fn update_privacy_settings(&mut self, settings: PrivacySettings) -> Result<()> {
        self.privacy_settings = settings;
        Ok(())
    }

    /// Get current privacy settings
    pub fn get_privacy_settings(&self) -> PrivacySettings {
        self.privacy_settings.clone()
    }

    /// Export telemetry data
    pub async fn export_data(&self) -> Result<Vec<u8>> {
        let batches = self.batches.read().await;
        let json = serde_json::to_string_pretty(&*batches)
            .map_err(|e| AppError::Unknown(format!("JSON serialization error: {}", e)))?;
        Ok(json.into_bytes())
    }

    /// Export telemetry data in specific format
    pub async fn export_data_format(&self, format: ExportFormat) -> Result<Vec<u8>> {
        let batches = self.batches.read().await;

        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&*batches)
                    .map_err(|e| AppError::Unknown(format!("JSON serialization error: {}", e)))?;
                Ok(json.into_bytes())
            }
            ExportFormat::Csv => {
                let mut csv = String::new();
                csv.push_str("batch_id,session_id,timestamp,data_type,data\n");

                for batch in batches.iter() {
                    for data in &batch.data {
                        let data_type = match data {
                            TelemetryData::Usage(_) => "usage",
                            TelemetryData::Performance(_) => "performance",
                            TelemetryData::Error(_) => "error",
                            TelemetryData::Feature(_) => "feature",
                            TelemetryData::System(_) => "system",
                        };

                        let data_json = serde_json::to_string(data).unwrap_or_default();
                        csv.push_str(&format!(
                            "{},{},{},{},{}\n",
                            batch.batch_id,
                            batch.session_id,
                            batch.created_at.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                            data_type,
                            data_json.replace('"', "\"\"") // Escape quotes for CSV
                        ));
                    }
                }

                Ok(csv.into_bytes())
            }
            ExportFormat::Parquet => {
                // Would implement Parquet export
                Err(AppError::Unknown("Parquet export not implemented".to_string()))
            }
            ExportFormat::Custom(format_name) => {
                Err(AppError::Unknown(format!("Custom format '{}' not implemented", format_name)))
            }
        }
    }

    /// Get telemetry statistics
    pub async fn get_statistics(&self) -> TelemetryStats {
        let batches = self.batches.read().await;
        let buffer = self.data_buffer.read().await;

        let total_batches = batches.len();
        let total_data_points = batches.iter().map(|b| b.data.len()).sum::<usize>() + buffer.len();

        let mut data_type_counts = HashMap::new();
        for batch in batches.iter() {
            for data in &batch.data {
                let data_type = match data {
                    TelemetryData::Usage(_) => "usage",
                    TelemetryData::Performance(_) => "performance",
                    TelemetryData::Error(_) => "error",
                    TelemetryData::Feature(_) => "feature",
                    TelemetryData::System(_) => "system",
                };
                *data_type_counts.entry(data_type.to_string()).or_insert(0) += 1;
            }
        }

        TelemetryStats {
            total_batches,
            total_data_points,
            buffer_size: buffer.len(),
            data_type_counts,
            oldest_batch: batches.first().map(|b| b.created_at),
            newest_batch: batches.last().map(|b| b.created_at),
        }
    }

    /// Clear all telemetry data
    pub async fn clear_data(&self) -> Result<()> {
        self.data_buffer.write().await.clear();
        self.batches.write().await.clear();
        Ok(())
    }

    // Private helper methods
    async fn record_system_info(&self) -> Result<()> {
        let system_data = SystemData {
            os_name: std::env::consts::OS.to_string(),
            os_version: "unknown".to_string(), // Would collect actual OS version
            architecture: std::env::consts::ARCH.to_string(),
            total_memory: 0, // Would collect actual memory info
            cpu_count: num_cpus::get(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: SystemTime::now(),
        };

        self.data_buffer.write().await.push(TelemetryData::System(system_data));
        Ok(())
    }

    async fn flush_data(&self) -> Result<()> {
        let data_buffer = self.data_buffer.clone();
        let batches = self.batches.clone();
        let session_id = self.session_id.clone();
        let batch_size = self.config.telemetry_batch_size;
        let privacy_settings = self.privacy_settings.clone();

        Self::process_telemetry_data(&data_buffer, &batches, &session_id, batch_size, &privacy_settings).await
    }

    async fn process_telemetry_data(
        data_buffer: &Arc<RwLock<Vec<TelemetryData>>>,
        batches: &Arc<RwLock<Vec<TelemetryBatch>>>,
        session_id: &str,
        batch_size: usize,
        _privacy_settings: &PrivacySettings,
    ) -> Result<()> {
        let mut buffer = data_buffer.write().await;
        
        if buffer.is_empty() {
            return Ok(());
        }

        // Create batches from buffered data
        while buffer.len() >= batch_size {
            let batch_data: Vec<TelemetryData> = buffer.drain(..batch_size).collect();
            
            let batch = TelemetryBatch {
                batch_id: uuid::Uuid::new_v4().to_string(),
                session_id: session_id.to_string(),
                created_at: SystemTime::now(),
                data: batch_data,
                metadata: HashMap::new(),
            };

            batches.write().await.push(batch);
        }

        // If we have remaining data less than batch_size, create a partial batch
        if !buffer.is_empty() {
            let batch_data: Vec<TelemetryData> = buffer.drain(..).collect();
            
            let batch = TelemetryBatch {
                batch_id: uuid::Uuid::new_v4().to_string(),
                session_id: session_id.to_string(),
                created_at: SystemTime::now(),
                data: batch_data,
                metadata: HashMap::new(),
            };

            batches.write().await.push(batch);
        }

        // Clean up old batches based on retention policy
        let retention_duration = Duration::from_secs(30 * 24 * 60 * 60); // Default retention
        let cutoff = SystemTime::now() - retention_duration;
        
        let mut batches_guard = batches.write().await;
        batches_guard.retain(|batch| batch.created_at >= cutoff);

        Ok(())
    }

    fn anonymize_metadata(&self, metadata: HashMap<String, String>) -> HashMap<String, String> {
        let mut anonymized = HashMap::new();
        
        for (key, value) in metadata {
            let key_lower = key.to_lowercase();
            let anonymized_key = if key_lower.contains("path") || key_lower.contains("file") {
                // Keep the key but anonymize the value
                key.clone()
            } else {
                key.clone()
            };

            let anonymized_value = if key_lower.contains("path") || key_lower.contains("file") {
                // Anonymize file paths
                self.anonymize_path(&value)
            } else if key_lower.contains("user") || key_lower.contains("id") {
                // Hash user identifiers
                format!("hash_{}", self.hash_string(&value))
            } else {
                value
            };

            anonymized.insert(anonymized_key, anonymized_value);
        }

        anonymized
    }

    fn anonymize_error_message(&self, message: &str) -> String {
        // Remove potential file paths and user-specific information
        let patterns = vec![
            (regex::Regex::new(r"/[^\s]+").unwrap(), "[PATH]"),
            (regex::Regex::new(r"C:\\[^\s]+").unwrap(), "[PATH]"),
            (regex::Regex::new(r"user\s+\w+").unwrap(), "user [USER]"),
        ];

        let mut anonymized = message.to_string();
        for (pattern, replacement) in patterns {
            anonymized = pattern.replace_all(&anonymized, replacement).to_string();
        }

        anonymized
    }

    fn anonymize_stack_trace(&self, stack_trace: &str) -> String {
        // Similar to error message anonymization
        self.anonymize_error_message(stack_trace)
    }

    fn anonymize_path(&self, path: &str) -> String {
        // Replace actual path with anonymized version
        if path.is_empty() {
            return path.to_string();
        }

        let components: Vec<&str> = path.split('/').collect();
        if components.len() <= 2 {
            return "[PATH]".to_string();
        }

        // Keep first and last component, anonymize middle parts
        format!("{}/[...]/{}",
            components.first().unwrap_or(&""),
            components.last().unwrap_or(&"")
        )
    }

    fn hash_string(&self, input: &str) -> String {
        // Simple hash for anonymization (would use proper crypto hash in production)
        format!("{:x}", input.chars().map(|c| c as u32).sum::<u32>())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryStats {
    pub total_batches: usize,
    pub total_data_points: usize,
    pub buffer_size: usize,
    pub data_type_counts: HashMap<String, usize>,
    pub oldest_batch: Option<SystemTime>,
    pub newest_batch: Option<SystemTime>,
}

impl Drop for TelemetrySystem {
    fn drop(&mut self) {
        if let Some(handle) = self.collection_task.take() {
            handle.abort();
        }
    }
}