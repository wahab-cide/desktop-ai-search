use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub enabled: bool,
    pub window_size: Duration,
    pub max_requests: u32,
    pub burst_limit: u32,
    pub burst_window: Duration,
    pub cleanup_interval: Duration,
    pub block_duration: Duration,
    pub whitelist: Vec<String>,
    pub blacklist: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitRule {
    pub id: String,
    pub name: String,
    pub pattern: String,
    pub max_requests: u32,
    pub window_size: Duration,
    pub priority: u32,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
struct RequestRecord {
    timestamps: Vec<Instant>,
    blocked_until: Option<Instant>,
    total_requests: u64,
    violations: u32,
    first_request: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    pub allowed: bool,
    pub remaining: u32,
    pub reset_time: SystemTime,
    pub retry_after: Option<Duration>,
    pub rule_applied: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitViolation {
    pub id: String,
    pub timestamp: SystemTime,
    pub client_id: String,
    pub resource: String,
    pub requests_count: u32,
    pub limit: u32,
    pub window_size: Duration,
    pub severity: ViolationSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitMetrics {
    pub total_requests: u64,
    pub allowed_requests: u64,
    pub blocked_requests: u64,
    pub violations: u32,
    pub active_clients: u32,
    pub top_clients: Vec<(String, u64)>,
    pub violations_by_rule: HashMap<String, u32>,
}

pub struct RateLimiter {
    config: RateLimitConfig,
    rules: HashMap<String, RateLimitRule>,
    records: Mutex<HashMap<String, RequestRecord>>,
    violations: Mutex<Vec<RateLimitViolation>>,
    metrics: Mutex<RateLimitMetrics>,
    last_cleanup: Mutex<Instant>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_size: Duration::from_secs(60), // 1 minute
            max_requests: 100,
            burst_limit: 20,
            burst_window: Duration::from_secs(10),
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            block_duration: Duration::from_secs(300),  // 5 minutes
            whitelist: Vec::new(),
            blacklist: Vec::new(),
        }
    }
}

impl Default for RequestRecord {
    fn default() -> Self {
        Self {
            timestamps: Vec::new(),
            blocked_until: None,
            total_requests: 0,
            violations: 0,
            first_request: Instant::now(),
        }
    }
}

impl Default for RateLimitMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            allowed_requests: 0,
            blocked_requests: 0,
            violations: 0,
            active_clients: 0,
            top_clients: Vec::new(),
            violations_by_rule: HashMap::new(),
        }
    }
}

impl RateLimiter {
    pub fn new() -> Result<Self> {
        Self::with_config(RateLimitConfig::default())
    }

    pub fn with_config(config: RateLimitConfig) -> Result<Self> {
        let mut limiter = Self {
            config,
            rules: HashMap::new(),
            records: Mutex::new(HashMap::new()),
            violations: Mutex::new(Vec::new()),
            metrics: Mutex::new(RateLimitMetrics::default()),
            last_cleanup: Mutex::new(Instant::now()),
        };

        limiter.initialize_default_rules()?;
        Ok(limiter)
    }

    pub fn check_rate_limit(&self, client_id: &str, resource: &str) -> Result<RateLimitStatus> {
        if !self.config.enabled {
            return Ok(RateLimitStatus {
                allowed: true,
                remaining: self.config.max_requests,
                reset_time: SystemTime::now() + self.config.window_size,
                retry_after: None,
                rule_applied: None,
            });
        }

        // Check whitelist
        if self.config.whitelist.contains(&client_id.to_string()) {
            return Ok(RateLimitStatus {
                allowed: true,
                remaining: self.config.max_requests,
                reset_time: SystemTime::now() + self.config.window_size,
                retry_after: None,
                rule_applied: Some("whitelist".to_string()),
            });
        }

        // Check blacklist
        if self.config.blacklist.contains(&client_id.to_string()) {
            return Ok(RateLimitStatus {
                allowed: false,
                remaining: 0,
                reset_time: SystemTime::now() + self.config.window_size,
                retry_after: Some(self.config.block_duration),
                rule_applied: Some("blacklist".to_string()),
            });
        }

        // Clean up old records periodically
        self.cleanup_if_needed()?;

        let now = Instant::now();
        let mut records = self.records.lock().unwrap();
        let mut metrics = self.metrics.lock().unwrap();

        // Get or create record for client
        let record = records.entry(client_id.to_string()).or_insert_with(Default::default);

        // Check if client is currently blocked
        if let Some(blocked_until) = record.blocked_until {
            if now < blocked_until {
                metrics.blocked_requests += 1;
                let retry_after = blocked_until - now;
                return Ok(RateLimitStatus {
                    allowed: false,
                    remaining: 0,
                    reset_time: SystemTime::now() + retry_after,
                    retry_after: Some(retry_after),
                    rule_applied: Some("blocked".to_string()),
                });
            } else {
                // Block period has expired
                record.blocked_until = None;
            }
        }

        // Find applicable rule
        let applicable_rule = self.find_applicable_rule(resource);
        let (max_requests, window_size) = if let Some(rule) = &applicable_rule {
            (rule.max_requests, rule.window_size)
        } else {
            (self.config.max_requests, self.config.window_size)
        };

        // Clean up old timestamps
        let window_start = now - window_size;
        record.timestamps.retain(|&timestamp| timestamp > window_start);

        // Check rate limit
        let current_requests = record.timestamps.len() as u32;
        if current_requests >= max_requests {
            // Rate limit exceeded
            record.violations += 1;
            record.blocked_until = Some(now + self.config.block_duration);
            
            // Record violation
            let violation = RateLimitViolation {
                id: Uuid::new_v4().to_string(),
                timestamp: SystemTime::now(),
                client_id: client_id.to_string(),
                resource: resource.to_string(),
                requests_count: current_requests,
                limit: max_requests,
                window_size,
                severity: self.determine_violation_severity(record.violations),
            };

            let mut violations = self.violations.lock().unwrap();
            violations.push(violation);

            // Update metrics
            metrics.blocked_requests += 1;
            metrics.violations += 1;
            if let Some(rule) = &applicable_rule {
                *metrics.violations_by_rule.entry(rule.id.clone()).or_insert(0) += 1;
            }

            return Ok(RateLimitStatus {
                allowed: false,
                remaining: 0,
                reset_time: SystemTime::now() + window_size,
                retry_after: Some(self.config.block_duration),
                rule_applied: applicable_rule.map(|r| r.id),
            });
        }

        // Check burst limit
        if current_requests >= self.config.burst_limit {
            let burst_window_start = now - self.config.burst_window;
            let burst_requests = record.timestamps.iter()
                .filter(|&&timestamp| timestamp > burst_window_start)
                .count() as u32;

            if burst_requests >= self.config.burst_limit {
                // Burst limit exceeded
                record.violations += 1;
                record.blocked_until = Some(now + Duration::from_secs(60)); // Shorter block for burst

                metrics.blocked_requests += 1;
                metrics.violations += 1;

                return Ok(RateLimitStatus {
                    allowed: false,
                    remaining: 0,
                    reset_time: SystemTime::now() + self.config.burst_window,
                    retry_after: Some(Duration::from_secs(60)),
                    rule_applied: Some("burst_limit".to_string()),
                });
            }
        }

        // Allow request
        record.timestamps.push(now);
        record.total_requests += 1;
        
        // Update metrics
        metrics.total_requests += 1;
        metrics.allowed_requests += 1;

        let remaining = max_requests - current_requests - 1;
        Ok(RateLimitStatus {
            allowed: true,
            remaining,
            reset_time: SystemTime::now() + window_size,
            retry_after: None,
            rule_applied: applicable_rule.map(|r| r.id),
        })
    }

    pub fn add_rule(&mut self, rule: RateLimitRule) -> Result<()> {
        self.rules.insert(rule.id.clone(), rule);
        Ok(())
    }

    pub fn remove_rule(&mut self, rule_id: &str) -> Result<()> {
        self.rules.remove(rule_id);
        Ok(())
    }

    pub fn add_to_whitelist(&mut self, client_id: &str) -> Result<()> {
        if !self.config.whitelist.contains(&client_id.to_string()) {
            self.config.whitelist.push(client_id.to_string());
        }
        Ok(())
    }

    pub fn remove_from_whitelist(&mut self, client_id: &str) -> Result<()> {
        self.config.whitelist.retain(|id| id != client_id);
        Ok(())
    }

    pub fn add_to_blacklist(&mut self, client_id: &str) -> Result<()> {
        if !self.config.blacklist.contains(&client_id.to_string()) {
            self.config.blacklist.push(client_id.to_string());
        }
        Ok(())
    }

    pub fn remove_from_blacklist(&mut self, client_id: &str) -> Result<()> {
        self.config.blacklist.retain(|id| id != client_id);
        Ok(())
    }

    pub fn get_client_stats(&self, client_id: &str) -> Option<(u64, u32, Option<Instant>)> {
        let records = self.records.lock().unwrap();
        records.get(client_id).map(|record| {
            (record.total_requests, record.violations, record.blocked_until)
        })
    }

    pub fn get_violations(&self, limit: Option<usize>) -> Vec<RateLimitViolation> {
        let violations = self.violations.lock().unwrap();
        if let Some(limit) = limit {
            violations.iter().rev().take(limit).cloned().collect()
        } else {
            violations.clone()
        }
    }

    pub fn get_metrics(&self) -> RateLimitMetrics {
        let mut metrics = self.metrics.lock().unwrap();
        let records = self.records.lock().unwrap();
        
        // Update active clients count
        metrics.active_clients = records.len() as u32;
        
        // Update top clients
        let mut client_requests: Vec<(String, u64)> = records.iter()
            .map(|(client_id, record)| (client_id.clone(), record.total_requests))
            .collect();
        client_requests.sort_by(|a, b| b.1.cmp(&a.1));
        metrics.top_clients = client_requests.into_iter().take(10).collect();
        
        metrics.clone()
    }

    pub fn reset_client(&mut self, client_id: &str) -> Result<()> {
        let mut records = self.records.lock().unwrap();
        records.remove(client_id);
        Ok(())
    }

    pub fn clear_all(&mut self) -> Result<()> {
        let mut records = self.records.lock().unwrap();
        records.clear();
        
        let mut violations = self.violations.lock().unwrap();
        violations.clear();
        
        let mut metrics = self.metrics.lock().unwrap();
        *metrics = RateLimitMetrics::default();
        
        Ok(())
    }

    fn find_applicable_rule(&self, resource: &str) -> Option<&RateLimitRule> {
        let mut applicable_rules: Vec<&RateLimitRule> = self.rules.values()
            .filter(|rule| rule.enabled && self.matches_pattern(&rule.pattern, resource))
            .collect();

        // Sort by priority (higher priority first)
        applicable_rules.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        applicable_rules.first().copied()
    }

    fn matches_pattern(&self, pattern: &str, resource: &str) -> bool {
        if pattern == "*" {
            return true;
        }

        // Simple glob pattern matching
        if pattern.ends_with('*') {
            let prefix = &pattern[..pattern.len()-1];
            return resource.starts_with(prefix);
        }

        if pattern.starts_with('*') {
            let suffix = &pattern[1..];
            return resource.ends_with(suffix);
        }

        // Try regex matching
        if let Ok(regex) = regex::Regex::new(pattern) {
            return regex.is_match(resource);
        }

        // Exact match
        pattern == resource
    }

    fn determine_violation_severity(&self, violations: u32) -> ViolationSeverity {
        match violations {
            1..=2 => ViolationSeverity::Low,
            3..=5 => ViolationSeverity::Medium,
            6..=10 => ViolationSeverity::High,
            _ => ViolationSeverity::Critical,
        }
    }

    fn cleanup_if_needed(&self) -> Result<()> {
        let mut last_cleanup = self.last_cleanup.lock().unwrap();
        let now = Instant::now();
        
        if now.duration_since(*last_cleanup) > self.config.cleanup_interval {
            self.cleanup_old_records()?;
            *last_cleanup = now;
        }
        
        Ok(())
    }

    fn cleanup_old_records(&self) -> Result<()> {
        let now = Instant::now();
        let mut records = self.records.lock().unwrap();
        
        // Remove records that are older than window size and not blocked
        let cutoff = now - self.config.window_size * 2; // Keep records for 2x window size
        records.retain(|_, record| {
            if let Some(blocked_until) = record.blocked_until {
                // Keep blocked records
                now < blocked_until
            } else {
                // Keep records with recent activity
                record.first_request > cutoff || !record.timestamps.is_empty()
            }
        });
        
        // Clean up old violations (keep last 1000)
        let mut violations = self.violations.lock().unwrap();
        if violations.len() > 1000 {
            violations.drain(0..violations.len() - 1000);
        }
        
        Ok(())
    }

    fn initialize_default_rules(&mut self) -> Result<()> {
        // Search API rate limit
        let search_rule = RateLimitRule {
            id: "search_api".to_string(),
            name: "Search API Rate Limit".to_string(),
            pattern: "search/*".to_string(),
            max_requests: 60,
            window_size: Duration::from_secs(60),
            priority: 10,
            enabled: true,
        };
        self.add_rule(search_rule)?;

        // File operations rate limit
        let file_rule = RateLimitRule {
            id: "file_operations".to_string(),
            name: "File Operations Rate Limit".to_string(),
            pattern: "files/*".to_string(),
            max_requests: 30,
            window_size: Duration::from_secs(60),
            priority: 20,
            enabled: true,
        };
        self.add_rule(file_rule)?;

        // Authentication rate limit
        let auth_rule = RateLimitRule {
            id: "authentication".to_string(),
            name: "Authentication Rate Limit".to_string(),
            pattern: "auth/*".to_string(),
            max_requests: 5,
            window_size: Duration::from_secs(300), // 5 minutes
            priority: 100,
            enabled: true,
        };
        self.add_rule(auth_rule)?;

        // Admin operations rate limit
        let admin_rule = RateLimitRule {
            id: "admin_operations".to_string(),
            name: "Admin Operations Rate Limit".to_string(),
            pattern: "admin/*".to_string(),
            max_requests: 10,
            window_size: Duration::from_secs(60),
            priority: 50,
            enabled: true,
        };
        self.add_rule(admin_rule)?;

        Ok(())
    }
}

// IP-based rate limiting utilities
pub struct IpRateLimiter {
    limiter: RateLimiter,
}

impl IpRateLimiter {
    pub fn new() -> Result<Self> {
        Ok(Self {
            limiter: RateLimiter::new()?,
        })
    }

    pub fn check_ip(&self, ip_address: &str, resource: &str) -> Result<RateLimitStatus> {
        // Normalize IP address
        let normalized_ip = self.normalize_ip(ip_address);
        self.limiter.check_rate_limit(&normalized_ip, resource)
    }

    fn normalize_ip(&self, ip: &str) -> String {
        // Remove port if present
        if let Some(colon_pos) = ip.rfind(':') {
            if ip.chars().filter(|&c| c == ':').count() == 1 {
                // IPv4 with port
                return ip[..colon_pos].to_string();
            }
        }

        // For IPv6, we might want to normalize the representation
        // For now, just return as-is
        ip.to_string()
    }
}

// Session-based rate limiting
pub struct SessionRateLimiter {
    limiter: RateLimiter,
}

impl SessionRateLimiter {
    pub fn new() -> Result<Self> {
        Ok(Self {
            limiter: RateLimiter::new()?,
        })
    }

    pub fn check_session(&self, session_id: &str, resource: &str) -> Result<RateLimitStatus> {
        self.limiter.check_rate_limit(session_id, resource)
    }
}

// Rate limit middleware utilities
pub struct RateLimitMiddleware {
    ip_limiter: IpRateLimiter,
    session_limiter: SessionRateLimiter,
}

impl RateLimitMiddleware {
    pub fn new() -> Result<Self> {
        Ok(Self {
            ip_limiter: IpRateLimiter::new()?,
            session_limiter: SessionRateLimiter::new()?,
        })
    }

    pub fn check_request(&self, ip_address: &str, session_id: Option<&str>, resource: &str) -> Result<RateLimitStatus> {
        // Check IP-based rate limit first
        let ip_status = self.ip_limiter.check_ip(ip_address, resource)?;
        if !ip_status.allowed {
            return Ok(ip_status);
        }

        // Check session-based rate limit if session ID is provided
        if let Some(session_id) = session_id {
            let session_status = self.session_limiter.check_session(session_id, resource)?;
            if !session_status.allowed {
                return Ok(session_status);
            }
        }

        // Return the more restrictive of the two limits
        Ok(ip_status)
    }
}