use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub mod validation;
pub mod encryption;
pub mod access_control;
pub mod audit;
pub mod sanitization;
pub mod rate_limiting;

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enable_input_validation: bool,
    pub enable_path_validation: bool,
    pub enable_rate_limiting: bool,
    pub enable_audit_logging: bool,
    pub enable_encryption: bool,
    pub max_request_size: usize,
    pub max_query_length: usize,
    pub max_path_length: usize,
    pub allowed_file_types: Vec<String>,
    pub blocked_file_types: Vec<String>,
    pub restricted_paths: Vec<PathBuf>,
    pub rate_limit_requests_per_minute: u32,
    pub session_timeout: Duration,
    pub max_concurrent_sessions: usize,
    pub enable_content_security: bool,
    pub enable_xss_protection: bool,
    pub enable_sql_injection_protection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    pub session_id: String,
    pub user_id: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub permissions: Vec<Permission>,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub access_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    pub resource: String,
    pub action: String,
    pub granted: bool,
    pub expires_at: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub id: String,
    pub event_type: SecurityEventType,
    pub severity: SecuritySeverity,
    pub description: String,
    pub source_ip: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub resource: Option<String>,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEventType {
    AuthenticationAttempt,
    AuthenticationFailure,
    AuthorizationFailure,
    InputValidationFailure,
    PathTraversalAttempt,
    RateLimitExceeded,
    SuspiciousActivity,
    UnauthorizedAccess,
    DataExfiltrationAttempt,
    SqlInjectionAttempt,
    XssAttempt,
    MalwareDetection,
    ConfigurationChange,
    PrivilegeEscalation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

pub struct SecurityManager {
    config: SecurityConfig,
    sessions: HashMap<String, SecurityContext>,
    rate_limiter: rate_limiting::RateLimiter,
    audit_logger: audit::AuditLogger,
    validator: validation::InputValidator,
    encryptor: encryption::Encryptor,
    access_controller: access_control::AccessController,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_input_validation: true,
            enable_path_validation: true,
            enable_rate_limiting: true,
            enable_audit_logging: true,
            enable_encryption: true,
            max_request_size: 10 * 1024 * 1024, // 10MB
            max_query_length: 1000,
            max_path_length: 4096,
            allowed_file_types: vec![
                "txt".to_string(),
                "pdf".to_string(),
                "doc".to_string(),
                "docx".to_string(),
                "md".to_string(),
                "html".to_string(),
                "json".to_string(),
                "csv".to_string(),
            ],
            blocked_file_types: vec![
                "exe".to_string(),
                "bat".to_string(),
                "cmd".to_string(),
                "com".to_string(),
                "pif".to_string(),
                "scr".to_string(),
                "vbs".to_string(),
                "js".to_string(),
                "jar".to_string(),
            ],
            restricted_paths: vec![
                PathBuf::from("/etc"),
                PathBuf::from("/proc"),
                PathBuf::from("/sys"),
                PathBuf::from("/dev"),
                PathBuf::from("C:\\Windows"),
                PathBuf::from("C:\\System32"),
            ],
            rate_limit_requests_per_minute: 1000,
            session_timeout: Duration::from_secs(3600), // 1 hour
            max_concurrent_sessions: 100,
            enable_content_security: true,
            enable_xss_protection: true,
            enable_sql_injection_protection: true,
        }
    }
}

impl SecurityManager {
    pub fn new(config: SecurityConfig) -> Result<Self> {
        let rate_limiter = rate_limiting::RateLimiter::new()?;
        let audit_logger = audit::AuditLogger::new()?;
        let validator = validation::InputValidator::new(&config)?;
        let encryptor = encryption::Encryptor::new()?;
        let access_controller = access_control::AccessController::new(&config)?;

        Ok(Self {
            config,
            sessions: HashMap::new(),
            rate_limiter,
            audit_logger,
            validator,
            encryptor,
            access_controller,
        })
    }

    pub fn create_session(&mut self, user_id: Option<String>, ip_address: Option<String>, user_agent: Option<String>) -> Result<String> {
        let session_id = Uuid::new_v4().to_string();
        let now = SystemTime::now();

        let context = SecurityContext {
            session_id: session_id.clone(),
            user_id: user_id.clone(),
            ip_address: ip_address.clone(),
            user_agent,
            permissions: Vec::new(),
            created_at: now,
            last_accessed: now,
            access_count: 0,
        };

        // Check session limits
        if self.sessions.len() >= self.config.max_concurrent_sessions {
            self.cleanup_expired_sessions();
            if self.sessions.len() >= self.config.max_concurrent_sessions {
                return Err(crate::error::AppError::SecurityViolation("Maximum concurrent sessions exceeded".to_string()));
            }
        }

        self.sessions.insert(session_id.clone(), context);

        // Log session creation
        self.log_security_event(SecurityEvent {
            id: Uuid::new_v4().to_string(),
            event_type: SecurityEventType::AuthenticationAttempt,
            severity: SecuritySeverity::Low,
            description: "Session created".to_string(),
            source_ip: ip_address,
            user_id,
            session_id: Some(session_id.clone()),
            resource: None,
            timestamp: now,
            metadata: HashMap::new(),
        })?;

        Ok(session_id)
    }

    pub fn validate_session(&mut self, session_id: &str) -> Result<bool> {
        if let Some(context) = self.sessions.get_mut(session_id) {
            let now = SystemTime::now();
            
            // Check if session has expired
            if now.duration_since(context.last_accessed).unwrap_or(Duration::from_secs(0)) > self.config.session_timeout {
                self.sessions.remove(session_id);
                return Ok(false);
            }

            // Update last accessed time
            context.last_accessed = now;
            context.access_count += 1;
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn validate_input(&self, input: &str, input_type: validation::InputType) -> Result<()> {
        if !self.config.enable_input_validation {
            return Ok(());
        }

        let result = self.validator.validate(input, input_type)?;
        if !result.is_valid {
            return Err(crate::error::AppError::ValidationError(
                result.errors.join(", ")
            ));
        }
        Ok(())
    }

    pub fn validate_path(&self, path: &Path) -> Result<()> {
        if !self.config.enable_path_validation {
            return Ok(());
        }

        // Check path length
        if path.as_os_str().len() > self.config.max_path_length {
            return Err(crate::error::AppError::SecurityViolation("Path too long".to_string()));
        }

        // Check for path traversal attempts
        if path.to_string_lossy().contains("..") {
            self.log_security_event(SecurityEvent {
                id: Uuid::new_v4().to_string(),
                event_type: SecurityEventType::PathTraversalAttempt,
                severity: SecuritySeverity::High,
                description: format!("Path traversal attempt: {}", path.display()),
                source_ip: None,
                user_id: None,
                session_id: None,
                resource: Some(path.to_string_lossy().to_string()),
                timestamp: SystemTime::now(),
                metadata: HashMap::new(),
            })?;
            return Err(crate::error::AppError::SecurityViolation("Path traversal not allowed".to_string()));
        }

        // Check restricted paths
        for restricted_path in &self.config.restricted_paths {
            if path.starts_with(restricted_path) {
                return Err(crate::error::AppError::SecurityViolation(format!("Access to restricted path: {}", restricted_path.display())));
            }
        }

        // Check file extension
        if let Some(extension) = path.extension() {
            let ext_str = extension.to_string_lossy().to_lowercase();
            
            if self.config.blocked_file_types.contains(&ext_str) {
                return Err(crate::error::AppError::SecurityViolation(format!("Blocked file type: {}", ext_str)));
            }
            
            if !self.config.allowed_file_types.is_empty() && !self.config.allowed_file_types.contains(&ext_str) {
                return Err(crate::error::AppError::SecurityViolation(format!("File type not allowed: {}", ext_str)));
            }
        }

        Ok(())
    }

    pub fn check_rate_limit(&mut self, identifier: &str) -> Result<bool> {
        if !self.config.enable_rate_limiting {
            return Ok(true);
        }

        let status = self.rate_limiter.check_rate_limit(identifier, "default")?;
        let allowed = status.allowed;
        
        if !allowed {
            self.log_security_event(SecurityEvent {
                id: Uuid::new_v4().to_string(),
                event_type: SecurityEventType::RateLimitExceeded,
                severity: SecuritySeverity::Medium,
                description: format!("Rate limit exceeded for identifier: {}", identifier),
                source_ip: None,
                user_id: None,
                session_id: None,
                resource: None,
                timestamp: SystemTime::now(),
                metadata: HashMap::new(),
            })?;
        }

        Ok(allowed)
    }

    pub fn sanitize_input(&self, input: &str) -> String {
        if !self.config.enable_content_security {
            return input.to_string();
        }

        sanitization::sanitize_input(input)
    }

    pub fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        if !self.config.enable_encryption {
            return Ok(data.to_vec());
        }

        self.encryptor.encrypt(data)
    }

    pub fn decrypt_data(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        if !self.config.enable_encryption {
            return Ok(encrypted_data.to_vec());
        }

        self.encryptor.decrypt(encrypted_data)
    }

    pub fn check_permission(&self, session_id: &str, resource: &str, action: &str) -> Result<bool> {
        if let Some(context) = self.sessions.get(session_id) {
            self.access_controller.check_permission(context, resource, action)
        } else {
            Ok(false)
        }
    }

    pub fn grant_permission(&mut self, session_id: &str, resource: &str, action: &str, expires_at: Option<SystemTime>) -> Result<()> {
        if let Some(context) = self.sessions.get_mut(session_id) {
            context.permissions.push(Permission {
                resource: resource.to_string(),
                action: action.to_string(),
                granted: true,
                expires_at,
            });
        }
        Ok(())
    }

    pub fn revoke_permission(&mut self, session_id: &str, resource: &str, action: &str) -> Result<()> {
        if let Some(context) = self.sessions.get_mut(session_id) {
            context.permissions.retain(|p| !(p.resource == resource && p.action == action));
        }
        Ok(())
    }

    pub fn log_security_event(&self, event: SecurityEvent) -> Result<()> {
        if !self.config.enable_audit_logging {
            return Ok(());
        }

        self.audit_logger.log_event(event)
    }

    pub fn get_security_events(&self, limit: Option<usize>) -> Result<Vec<SecurityEvent>> {
        self.audit_logger.get_events(limit)
    }

    pub fn generate_security_report(&self) -> Result<SecurityReport> {
        let events = self.get_security_events(Some(1000))?;
        let active_sessions = self.sessions.len();
        let total_events = events.len();
        let critical_events = events.iter().filter(|e| matches!(e.severity, SecuritySeverity::Critical)).count();
        let high_events = events.iter().filter(|e| matches!(e.severity, SecuritySeverity::High)).count();
        
        let threat_level = if critical_events > 0 {
            ThreatLevel::Critical
        } else if high_events > 5 {
            ThreatLevel::High
        } else if high_events > 0 {
            ThreatLevel::Medium
        } else {
            ThreatLevel::Low
        };

        Ok(SecurityReport {
            timestamp: SystemTime::now(),
            active_sessions,
            total_events,
            critical_events,
            high_events,
            threat_level,
            recommendations: self.generate_security_recommendations(&events),
        })
    }

    pub fn cleanup_expired_sessions(&mut self) {
        let now = SystemTime::now();
        let timeout = self.config.session_timeout;
        
        self.sessions.retain(|_, context| {
            now.duration_since(context.last_accessed).unwrap_or(Duration::from_secs(0)) <= timeout
        });
    }

    pub fn hash_password(&self, password: &str, salt: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(password.as_bytes());
        hasher.update(salt.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn generate_salt(&self) -> String {
        Uuid::new_v4().to_string()
    }

    pub fn verify_password(&self, password: &str, salt: &str, hash: &str) -> bool {
        let computed_hash = self.hash_password(password, salt);
        computed_hash == hash
    }

    fn generate_security_recommendations(&self, events: &[SecurityEvent]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        let failed_auth_count = events.iter()
            .filter(|e| matches!(e.event_type, SecurityEventType::AuthenticationFailure))
            .count();
            
        if failed_auth_count > 10 {
            recommendations.push("Consider implementing account lockout after failed authentication attempts".to_string());
        }
        
        let injection_attempts = events.iter()
            .filter(|e| matches!(e.event_type, SecurityEventType::SqlInjectionAttempt))
            .count();
            
        if injection_attempts > 0 {
            recommendations.push("Review and strengthen input validation for SQL injection prevention".to_string());
        }
        
        let path_traversal_attempts = events.iter()
            .filter(|e| matches!(e.event_type, SecurityEventType::PathTraversalAttempt))
            .count();
            
        if path_traversal_attempts > 0 {
            recommendations.push("Implement additional path validation and sandboxing".to_string());
        }
        
        if self.sessions.len() > self.config.max_concurrent_sessions * 8 / 10 {
            recommendations.push("Consider increasing session limits or implementing session cleanup".to_string());
        }
        
        recommendations
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityReport {
    pub timestamp: SystemTime,
    pub active_sessions: usize,
    pub total_events: usize,
    pub critical_events: usize,
    pub high_events: usize,
    pub threat_level: ThreatLevel,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatLevel {
    Low,
    Medium,
    High,
    Critical,
}

// Convenience functions for common security operations
pub fn secure_random_string(length: usize) -> String {
    use rand::Rng;
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::thread_rng();
    
    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

pub fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    
    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    
    result == 0
}

pub fn secure_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

// Security middleware for request validation
pub struct SecurityMiddleware {
    manager: SecurityManager,
}

impl SecurityMiddleware {
    pub fn new(config: SecurityConfig) -> Result<Self> {
        let manager = SecurityManager::new(config)?;
        Ok(Self { manager })
    }

    pub fn validate_request(&mut self, session_id: &str, path: &str, query: &str) -> Result<()> {
        // Validate session
        if !self.manager.validate_session(session_id)? {
            return Err(crate::error::AppError::SecurityViolation("Invalid session".to_string()));
        }

        // Check rate limit
        if !self.manager.check_rate_limit(session_id)? {
            return Err(crate::error::AppError::SecurityViolation("Rate limit exceeded".to_string()));
        }

        // Validate path
        self.manager.validate_path(Path::new(path))?;

        // Validate query
        self.manager.validate_input(query, validation::InputType::Query)?;

        Ok(())
    }
}