use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use super::{SecurityEvent, SecurityEventType, SecuritySeverity};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    pub id: String,
    pub timestamp: SystemTime,
    pub event_type: AuditEventType,
    pub severity: AuditSeverity,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub resource: Option<String>,
    pub action: Option<String>,
    pub outcome: AuditOutcome,
    pub details: HashMap<String, String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub request_id: Option<String>,
    pub correlation_id: Option<String>,
    pub source_component: String,
    pub data_classification: DataClassification,
    pub retention_period: Option<u32>, // Days
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    DataDeletion,
    SystemAccess,
    ConfigurationChange,
    PrivilegeEscalation,
    SecurityPolicyChange,
    FileAccess,
    SearchQuery,
    IndexOperation,
    BackupOperation,
    RestoreOperation,
    ExportOperation,
    ImportOperation,
    UserManagement,
    RoleManagement,
    PermissionChange,
    SessionManagement,
    SystemShutdown,
    SystemStartup,
    ErrorEvent,
    WarningEvent,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditOutcome {
    Success,
    Failure,
    Partial,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataClassification {
    Public,
    Internal,
    Confidential,
    Restricted,
    TopSecret,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    pub enabled: bool,
    pub log_file_path: PathBuf,
    pub max_file_size: u64,
    pub max_files: u32,
    pub buffer_size: usize,
    pub flush_interval: u64, // seconds
    pub encryption_enabled: bool,
    pub compression_enabled: bool,
    pub remote_logging_enabled: bool,
    pub remote_endpoint: Option<String>,
    pub retention_days: u32,
    pub anonymize_sensitive_data: bool,
    pub log_levels: Vec<AuditSeverity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub id: String,
    pub generated_at: SystemTime,
    pub period_start: SystemTime,
    pub period_end: SystemTime,
    pub total_events: usize,
    pub events_by_type: HashMap<String, usize>,
    pub events_by_severity: HashMap<String, usize>,
    pub events_by_outcome: HashMap<String, usize>,
    pub events_by_user: HashMap<String, usize>,
    pub security_incidents: Vec<SecurityIncident>,
    pub compliance_violations: Vec<ComplianceViolation>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIncident {
    pub id: String,
    pub timestamp: SystemTime,
    pub incident_type: String,
    pub severity: SecuritySeverity,
    pub description: String,
    pub affected_resources: Vec<String>,
    pub related_events: Vec<String>,
    pub status: IncidentStatus,
    pub assigned_to: Option<String>,
    pub resolution: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IncidentStatus {
    New,
    Investigating,
    Resolved,
    Closed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub id: String,
    pub timestamp: SystemTime,
    pub regulation: String,
    pub violation_type: String,
    pub description: String,
    pub severity: AuditSeverity,
    pub remediation: String,
}

pub struct AuditLogger {
    config: AuditConfig,
    file_writer: Mutex<Option<BufWriter<File>>>,
    buffer: Mutex<Vec<AuditLog>>,
    events: Mutex<Vec<SecurityEvent>>,
    current_file_size: Mutex<u64>,
    file_counter: Mutex<u32>,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_file_path: PathBuf::from("logs/audit.log"),
            max_file_size: 100 * 1024 * 1024, // 100MB
            max_files: 10,
            buffer_size: 1000,
            flush_interval: 60, // 1 minute
            encryption_enabled: false,
            compression_enabled: false,
            remote_logging_enabled: false,
            remote_endpoint: None,
            retention_days: 365,
            anonymize_sensitive_data: true,
            log_levels: vec![
                AuditSeverity::Low,
                AuditSeverity::Medium,
                AuditSeverity::High,
                AuditSeverity::Critical,
            ],
        }
    }
}

impl AuditLogger {
    pub fn new() -> Result<Self> {
        Self::with_config(AuditConfig::default())
    }

    pub fn with_config(config: AuditConfig) -> Result<Self> {
        // Create log directory if it doesn't exist
        if let Some(parent) = config.log_file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let logger = Self {
            config,
            file_writer: Mutex::new(None),
            buffer: Mutex::new(Vec::new()),
            events: Mutex::new(Vec::new()),
            current_file_size: Mutex::new(0),
            file_counter: Mutex::new(0),
        };

        logger.initialize_file_writer()?;

        Ok(logger)
    }

    pub fn log_event(&self, event: SecurityEvent) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let audit_log = self.convert_security_event_to_audit_log(event.clone())?;
        self.write_audit_log(audit_log)?;

        // Store event for reporting
        {
            let mut events = self.events.lock().unwrap();
            events.push(event);
        }

        Ok(())
    }

    pub fn log_audit_event(&self, event: AuditLog) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check if severity is enabled
        if !self.config.log_levels.contains(&event.severity) {
            return Ok(());
        }

        self.write_audit_log(event)?;
        Ok(())
    }

    pub fn log_authentication(&self, user_id: &str, outcome: AuditOutcome, details: HashMap<String, String>) -> Result<()> {
        let event = AuditLog {
            id: Uuid::new_v4().to_string(),
            timestamp: SystemTime::now(),
            event_type: AuditEventType::Authentication,
            severity: match outcome {
                AuditOutcome::Success => AuditSeverity::Low,
                AuditOutcome::Failure => AuditSeverity::Medium,
                _ => AuditSeverity::Low,
            },
            user_id: Some(user_id.to_string()),
            session_id: None,
            resource: None,
            action: Some("login".to_string()),
            outcome,
            details,
            ip_address: None,
            user_agent: None,
            request_id: None,
            correlation_id: None,
            source_component: "authentication".to_string(),
            data_classification: DataClassification::Internal,
            retention_period: Some(self.config.retention_days),
        };

        self.log_audit_event(event)
    }

    pub fn log_authorization(&self, user_id: &str, resource: &str, action: &str, outcome: AuditOutcome) -> Result<()> {
        let event = AuditLog {
            id: Uuid::new_v4().to_string(),
            timestamp: SystemTime::now(),
            event_type: AuditEventType::Authorization,
            severity: match outcome {
                AuditOutcome::Success => AuditSeverity::Low,
                AuditOutcome::Failure => AuditSeverity::Medium,
                _ => AuditSeverity::Low,
            },
            user_id: Some(user_id.to_string()),
            session_id: None,
            resource: Some(resource.to_string()),
            action: Some(action.to_string()),
            outcome,
            details: HashMap::new(),
            ip_address: None,
            user_agent: None,
            request_id: None,
            correlation_id: None,
            source_component: "authorization".to_string(),
            data_classification: DataClassification::Internal,
            retention_period: Some(self.config.retention_days),
        };

        self.log_audit_event(event)
    }

    pub fn log_data_access(&self, user_id: &str, resource: &str, data_classification: DataClassification) -> Result<()> {
        let event = AuditLog {
            id: Uuid::new_v4().to_string(),
            timestamp: SystemTime::now(),
            event_type: AuditEventType::DataAccess,
            severity: match data_classification {
                DataClassification::Public => AuditSeverity::Low,
                DataClassification::Internal => AuditSeverity::Low,
                DataClassification::Confidential => AuditSeverity::Medium,
                DataClassification::Restricted => AuditSeverity::High,
                DataClassification::TopSecret => AuditSeverity::Critical,
            },
            user_id: Some(user_id.to_string()),
            session_id: None,
            resource: Some(resource.to_string()),
            action: Some("read".to_string()),
            outcome: AuditOutcome::Success,
            details: HashMap::new(),
            ip_address: None,
            user_agent: None,
            request_id: None,
            correlation_id: None,
            source_component: "data_access".to_string(),
            data_classification,
            retention_period: Some(self.config.retention_days),
        };

        self.log_audit_event(event)
    }

    pub fn log_search_query(&self, user_id: &str, query: &str, result_count: usize) -> Result<()> {
        let sanitized_query = if self.config.anonymize_sensitive_data {
            self.sanitize_query(query)
        } else {
            query.to_string()
        };

        let mut details = HashMap::new();
        details.insert("query".to_string(), sanitized_query);
        details.insert("result_count".to_string(), result_count.to_string());

        let event = AuditLog {
            id: Uuid::new_v4().to_string(),
            timestamp: SystemTime::now(),
            event_type: AuditEventType::SearchQuery,
            severity: AuditSeverity::Low,
            user_id: Some(user_id.to_string()),
            session_id: None,
            resource: Some("search".to_string()),
            action: Some("query".to_string()),
            outcome: AuditOutcome::Success,
            details,
            ip_address: None,
            user_agent: None,
            request_id: None,
            correlation_id: None,
            source_component: "search".to_string(),
            data_classification: DataClassification::Internal,
            retention_period: Some(self.config.retention_days),
        };

        self.log_audit_event(event)
    }

    pub fn log_file_access(&self, user_id: &str, file_path: &str, action: &str, outcome: AuditOutcome) -> Result<()> {
        let sanitized_path = if self.config.anonymize_sensitive_data {
            self.sanitize_path(file_path)
        } else {
            file_path.to_string()
        };

        let mut details = HashMap::new();
        details.insert("file_path".to_string(), sanitized_path);

        let event = AuditLog {
            id: Uuid::new_v4().to_string(),
            timestamp: SystemTime::now(),
            event_type: AuditEventType::FileAccess,
            severity: match outcome {
                AuditOutcome::Success => AuditSeverity::Low,
                AuditOutcome::Failure => AuditSeverity::Medium,
                _ => AuditSeverity::Low,
            },
            user_id: Some(user_id.to_string()),
            session_id: None,
            resource: Some("file".to_string()),
            action: Some(action.to_string()),
            outcome,
            details,
            ip_address: None,
            user_agent: None,
            request_id: None,
            correlation_id: None,
            source_component: "file_system".to_string(),
            data_classification: DataClassification::Internal,
            retention_period: Some(self.config.retention_days),
        };

        self.log_audit_event(event)
    }

    pub fn get_events(&self, limit: Option<usize>) -> Result<Vec<SecurityEvent>> {
        let events = self.events.lock().unwrap();
        let events_to_return = if let Some(limit) = limit {
            events.iter().rev().take(limit).cloned().collect()
        } else {
            events.clone()
        };
        Ok(events_to_return)
    }

    pub fn generate_report(&self, start_time: SystemTime, end_time: SystemTime) -> Result<AuditReport> {
        let events = self.events.lock().unwrap();
        let filtered_events: Vec<&SecurityEvent> = events.iter()
            .filter(|event| event.timestamp >= start_time && event.timestamp <= end_time)
            .collect();

        let total_events = filtered_events.len();
        let mut events_by_type = HashMap::new();
        let mut events_by_severity = HashMap::new();
        let mut events_by_user = HashMap::new();

        for event in &filtered_events {
            // Count by type
            let type_name = format!("{:?}", event.event_type);
            *events_by_type.entry(type_name).or_insert(0) += 1;

            // Count by severity
            let severity_name = format!("{:?}", event.severity);
            *events_by_severity.entry(severity_name).or_insert(0) += 1;

            // Count by user
            if let Some(user_id) = &event.user_id {
                *events_by_user.entry(user_id.clone()).or_insert(0) += 1;
            }
        }

        let security_incidents = self.identify_security_incidents(&filtered_events)?;
        let compliance_violations = self.identify_compliance_violations(&filtered_events)?;
        let recommendations = self.generate_recommendations(&filtered_events)?;

        Ok(AuditReport {
            id: Uuid::new_v4().to_string(),
            generated_at: SystemTime::now(),
            period_start: start_time,
            period_end: end_time,
            total_events,
            events_by_type,
            events_by_severity,
            events_by_outcome: HashMap::new(), // Would need to track outcomes
            events_by_user,
            security_incidents,
            compliance_violations,
            recommendations,
        })
    }

    pub fn flush(&self) -> Result<()> {
        let mut buffer = self.buffer.lock().unwrap();
        if !buffer.is_empty() {
            for event in buffer.drain(..) {
                self.write_to_file(&event)?;
            }
        }

        if let Some(ref mut writer) = self.file_writer.lock().unwrap().as_mut() {
            writer.flush()?;
        }

        Ok(())
    }

    pub fn rotate_logs(&self) -> Result<()> {
        self.flush()?;
        
        // Close current file
        {
            let mut writer = self.file_writer.lock().unwrap();
            *writer = None;
        }

        // Rotate files
        let base_path = &self.config.log_file_path;
        let mut file_counter = self.file_counter.lock().unwrap();
        
        for i in (1..self.config.max_files).rev() {
            let old_path = if i == 1 {
                base_path.clone()
            } else {
                base_path.with_extension(format!("{}.{}", 
                    base_path.extension().unwrap_or_default().to_str().unwrap_or("log"),
                    i - 1
                ))
            };

            let new_path = base_path.with_extension(format!("{}.{}", 
                base_path.extension().unwrap_or_default().to_str().unwrap_or("log"),
                i
            ));

            if old_path.exists() {
                std::fs::rename(old_path, new_path)?;
            }
        }

        // Initialize new file
        self.initialize_file_writer()?;
        *file_counter += 1;

        Ok(())
    }

    fn write_audit_log(&self, event: AuditLog) -> Result<()> {
        // Check if we need to rotate logs
        {
            let current_size = *self.current_file_size.lock().unwrap();
            if current_size >= self.config.max_file_size {
                self.rotate_logs()?;
            }
        }

        // Buffer the event
        {
            let mut buffer = self.buffer.lock().unwrap();
            buffer.push(event);

            // Flush if buffer is full
            if buffer.len() >= self.config.buffer_size {
                drop(buffer);
                self.flush()?;
            }
        }

        Ok(())
    }

    fn write_to_file(&self, event: &AuditLog) -> Result<()> {
        let json_line = serde_json::to_string(event)?;
        let line = format!("{}\n", json_line);

        if let Some(ref mut writer) = self.file_writer.lock().unwrap().as_mut() {
            writer.write_all(line.as_bytes())?;
            
            let mut current_size = self.current_file_size.lock().unwrap();
            *current_size += line.len() as u64;
        }

        Ok(())
    }

    fn initialize_file_writer(&self) -> Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.config.log_file_path)?;

        let current_size = file.metadata()?.len();
        *self.current_file_size.lock().unwrap() = current_size;

        let writer = BufWriter::new(file);
        *self.file_writer.lock().unwrap() = Some(writer);

        Ok(())
    }

    fn convert_security_event_to_audit_log(&self, event: SecurityEvent) -> Result<AuditLog> {
        let audit_event_type = match event.event_type {
            SecurityEventType::AuthenticationAttempt => AuditEventType::Authentication,
            SecurityEventType::AuthenticationFailure => AuditEventType::Authentication,
            SecurityEventType::AuthorizationFailure => AuditEventType::Authorization,
            SecurityEventType::UnauthorizedAccess => AuditEventType::Authorization,
            SecurityEventType::InputValidationFailure => AuditEventType::Custom("input_validation".to_string()),
            SecurityEventType::PathTraversalAttempt => AuditEventType::Custom("path_traversal".to_string()),
            SecurityEventType::RateLimitExceeded => AuditEventType::Custom("rate_limit".to_string()),
            SecurityEventType::SuspiciousActivity => AuditEventType::Custom("suspicious_activity".to_string()),
            SecurityEventType::DataExfiltrationAttempt => AuditEventType::Custom("data_exfiltration".to_string()),
            SecurityEventType::SqlInjectionAttempt => AuditEventType::Custom("sql_injection".to_string()),
            SecurityEventType::XssAttempt => AuditEventType::Custom("xss_attempt".to_string()),
            SecurityEventType::MalwareDetection => AuditEventType::Custom("malware_detection".to_string()),
            SecurityEventType::ConfigurationChange => AuditEventType::ConfigurationChange,
            SecurityEventType::PrivilegeEscalation => AuditEventType::PrivilegeEscalation,
        };

        let audit_severity = match event.severity {
            super::SecuritySeverity::Low => AuditSeverity::Low,
            super::SecuritySeverity::Medium => AuditSeverity::Medium,
            super::SecuritySeverity::High => AuditSeverity::High,
            super::SecuritySeverity::Critical => AuditSeverity::Critical,
        };

        Ok(AuditLog {
            id: event.id,
            timestamp: event.timestamp,
            event_type: audit_event_type,
            severity: audit_severity,
            user_id: event.user_id,
            session_id: event.session_id,
            resource: event.resource,
            action: None,
            outcome: AuditOutcome::Unknown,
            details: event.metadata,
            ip_address: event.source_ip,
            user_agent: None,
            request_id: None,
            correlation_id: None,
            source_component: "security".to_string(),
            data_classification: DataClassification::Internal,
            retention_period: Some(self.config.retention_days),
        })
    }

    fn sanitize_query(&self, query: &str) -> String {
        // Remove potentially sensitive information from queries
        let sensitive_patterns = [
            r"password\s*[:=]\s*['\"]?[^'\"]*['\"]?",
            r"token\s*[:=]\s*['\"]?[^'\"]*['\"]?",
            r"key\s*[:=]\s*['\"]?[^'\"]*['\"]?",
            r"secret\s*[:=]\s*['\"]?[^'\"]*['\"]?",
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", // Credit card numbers
            r"\b\d{3}-\d{2}-\d{4}\b", // SSN
        ];

        let mut sanitized = query.to_string();
        for pattern in &sensitive_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                sanitized = regex.replace_all(&sanitized, "[REDACTED]").to_string();
            }
        }

        sanitized
    }

    fn sanitize_path(&self, path: &str) -> String {
        // Remove potentially sensitive parts of file paths
        if path.contains("password") || path.contains("secret") || path.contains("private") {
            return "[SENSITIVE_PATH]".to_string();
        }
        
        // Keep only the filename if it's a personal directory
        if path.contains("/home/") || path.contains("\\Users\\") {
            if let Some(filename) = std::path::Path::new(path).file_name() {
                return format!("[USER_DIR]/{}", filename.to_string_lossy());
            }
        }
        
        path.to_string()
    }

    fn identify_security_incidents(&self, events: &[&SecurityEvent]) -> Result<Vec<SecurityIncident>> {
        let mut incidents = Vec::new();
        
        // Look for patterns that might indicate security incidents
        let mut failed_auth_count = 0;
        let mut injection_attempts = 0;
        let mut path_traversal_attempts = 0;

        for event in events {
            match event.event_type {
                SecurityEventType::AuthenticationFailure => failed_auth_count += 1,
                SecurityEventType::SqlInjectionAttempt => injection_attempts += 1,
                SecurityEventType::PathTraversalAttempt => path_traversal_attempts += 1,
                _ => {}
            }
        }

        if failed_auth_count > 10 {
            incidents.push(SecurityIncident {
                id: Uuid::new_v4().to_string(),
                timestamp: SystemTime::now(),
                incident_type: "Brute Force Attack".to_string(),
                severity: super::SecuritySeverity::High,
                description: format!("Multiple authentication failures detected: {}", failed_auth_count),
                affected_resources: vec!["authentication_system".to_string()],
                related_events: Vec::new(),
                status: IncidentStatus::New,
                assigned_to: None,
                resolution: None,
            });
        }

        if injection_attempts > 0 {
            incidents.push(SecurityIncident {
                id: Uuid::new_v4().to_string(),
                timestamp: SystemTime::now(),
                incident_type: "SQL Injection Attempt".to_string(),
                severity: super::SecuritySeverity::Critical,
                description: format!("SQL injection attempts detected: {}", injection_attempts),
                affected_resources: vec!["database".to_string()],
                related_events: Vec::new(),
                status: IncidentStatus::New,
                assigned_to: None,
                resolution: None,
            });
        }

        if path_traversal_attempts > 0 {
            incidents.push(SecurityIncident {
                id: Uuid::new_v4().to_string(),
                timestamp: SystemTime::now(),
                incident_type: "Path Traversal Attempt".to_string(),
                severity: super::SecuritySeverity::High,
                description: format!("Path traversal attempts detected: {}", path_traversal_attempts),
                affected_resources: vec!["file_system".to_string()],
                related_events: Vec::new(),
                status: IncidentStatus::New,
                assigned_to: None,
                resolution: None,
            });
        }

        Ok(incidents)
    }

    fn identify_compliance_violations(&self, events: &[&SecurityEvent]) -> Result<Vec<ComplianceViolation>> {
        let mut violations = Vec::new();
        
        // Check for potential GDPR violations
        let mut data_access_without_consent = 0;
        
        for event in events {
            // This is a simplified check - in practice, you'd need more sophisticated analysis
            if matches!(event.event_type, SecurityEventType::DataExfiltrationAttempt) {
                data_access_without_consent += 1;
            }
        }

        if data_access_without_consent > 0 {
            violations.push(ComplianceViolation {
                id: Uuid::new_v4().to_string(),
                timestamp: SystemTime::now(),
                regulation: "GDPR".to_string(),
                violation_type: "Unauthorized Data Access".to_string(),
                description: "Potential unauthorized access to personal data".to_string(),
                severity: AuditSeverity::High,
                remediation: "Review access controls and data protection measures".to_string(),
            });
        }

        Ok(violations)
    }

    fn generate_recommendations(&self, events: &[&SecurityEvent]) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        
        let high_severity_count = events.iter()
            .filter(|e| matches!(e.severity, super::SecuritySeverity::High | super::SecuritySeverity::Critical))
            .count();

        if high_severity_count > 5 {
            recommendations.push("Consider implementing additional security monitoring and alerting".to_string());
        }

        let auth_failures = events.iter()
            .filter(|e| matches!(e.event_type, SecurityEventType::AuthenticationFailure))
            .count();

        if auth_failures > 20 {
            recommendations.push("Implement account lockout policies to prevent brute force attacks".to_string());
        }

        let injection_attempts = events.iter()
            .filter(|e| matches!(e.event_type, SecurityEventType::SqlInjectionAttempt))
            .count();

        if injection_attempts > 0 {
            recommendations.push("Review and strengthen input validation and parameterized queries".to_string());
        }

        Ok(recommendations)
    }
}