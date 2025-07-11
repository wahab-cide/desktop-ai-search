use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use regex::Regex;
use tokio::sync::Mutex;

use super::{LogEntry, LogFilter, LogLevel};

pub struct LevelFilter {
    min_level: LogLevel,
}

pub struct TargetFilter {
    allowed_targets: Vec<String>,
    blocked_targets: Vec<String>,
}

pub struct RegexFilter {
    regex: Regex,
    include: bool, // true = include matches, false = exclude matches
}

pub struct RateLimitFilter {
    requests: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
    max_requests: usize,
    window: Duration,
}

pub struct TagFilter {
    required_tags: Vec<String>,
    excluded_tags: Vec<String>,
}

pub struct ThreadFilter {
    allowed_threads: Vec<String>,
    blocked_threads: Vec<String>,
}

pub struct MessageFilter {
    patterns: Vec<Regex>,
    exclude: bool,
}

pub struct CompositeFilter {
    filters: Vec<Box<dyn LogFilter>>,
    mode: FilterMode,
}

pub enum FilterMode {
    And, // All filters must pass
    Or,  // At least one filter must pass
}

impl LevelFilter {
    pub fn new(min_level: LogLevel) -> Self {
        Self { min_level }
    }
}

impl LogFilter for LevelFilter {
    fn should_log(&self, entry: &LogEntry) -> bool {
        entry.level.priority() <= self.min_level.priority()
    }
}

impl TargetFilter {
    pub fn new() -> Self {
        Self {
            allowed_targets: Vec::new(),
            blocked_targets: Vec::new(),
        }
    }

    pub fn allow_target(mut self, target: String) -> Self {
        self.allowed_targets.push(target);
        self
    }

    pub fn block_target(mut self, target: String) -> Self {
        self.blocked_targets.push(target);
        self
    }

    pub fn allow_targets(mut self, targets: Vec<String>) -> Self {
        self.allowed_targets.extend(targets);
        self
    }

    pub fn block_targets(mut self, targets: Vec<String>) -> Self {
        self.blocked_targets.extend(targets);
        self
    }
}

impl LogFilter for TargetFilter {
    fn should_log(&self, entry: &LogEntry) -> bool {
        // Check blocked targets first
        if self.blocked_targets.iter().any(|t| entry.target.contains(t)) {
            return false;
        }

        // If no allowed targets specified, allow all (except blocked)
        if self.allowed_targets.is_empty() {
            return true;
        }

        // Check if target matches any allowed targets
        self.allowed_targets.iter().any(|t| entry.target.contains(t))
    }
}

impl RegexFilter {
    pub fn new(pattern: &str, include: bool) -> Result<Self, regex::Error> {
        let regex = Regex::new(pattern)?;
        Ok(Self { regex, include })
    }

    pub fn include(pattern: &str) -> Result<Self, regex::Error> {
        Self::new(pattern, true)
    }

    pub fn exclude(pattern: &str) -> Result<Self, regex::Error> {
        Self::new(pattern, false)
    }
}

impl LogFilter for RegexFilter {
    fn should_log(&self, entry: &LogEntry) -> bool {
        let matches = self.regex.is_match(&entry.message);
        if self.include {
            matches
        } else {
            !matches
        }
    }
}

impl RateLimitFilter {
    pub fn new(max_requests: usize, window: Duration) -> Self {
        Self {
            requests: Arc::new(Mutex::new(HashMap::new())),
            max_requests,
            window,
        }
    }

    pub fn per_minute(max_requests: usize) -> Self {
        Self::new(max_requests, Duration::from_secs(60))
    }

    pub fn per_second(max_requests: usize) -> Self {
        Self::new(max_requests, Duration::from_secs(1))
    }

    fn get_rate_limit_key(&self, entry: &LogEntry) -> String {
        // Rate limit by combination of target and level
        format!("{}:{}", entry.target, entry.level.to_string())
    }

    async fn should_allow(&self, key: &str) -> bool {
        let mut requests = self.requests.lock().await;
        let now = Instant::now();
        
        // Get or create entry for this key
        let entry = requests.entry(key.to_string()).or_insert_with(Vec::new);
        
        // Remove old requests outside the window
        entry.retain(|&timestamp| now.duration_since(timestamp) <= self.window);
        
        // Check if we're under the rate limit
        if entry.len() < self.max_requests {
            entry.push(now);
            true
        } else {
            false
        }
    }
}

impl LogFilter for RateLimitFilter {
    fn should_log(&self, entry: &LogEntry) -> bool {
        let key = self.get_rate_limit_key(entry);
        
        // Use blocking call for simplicity - in real async context, this would be awaited
        tokio::runtime::Handle::current().block_on(self.should_allow(&key))
    }
}

impl TagFilter {
    pub fn new() -> Self {
        Self {
            required_tags: Vec::new(),
            excluded_tags: Vec::new(),
        }
    }

    pub fn require_tag(mut self, tag: String) -> Self {
        self.required_tags.push(tag);
        self
    }

    pub fn exclude_tag(mut self, tag: String) -> Self {
        self.excluded_tags.push(tag);
        self
    }

    pub fn require_tags(mut self, tags: Vec<String>) -> Self {
        self.required_tags.extend(tags);
        self
    }

    pub fn exclude_tags(mut self, tags: Vec<String>) -> Self {
        self.excluded_tags.extend(tags);
        self
    }
}

impl LogFilter for TagFilter {
    fn should_log(&self, entry: &LogEntry) -> bool {
        // Check excluded tags first
        if self.excluded_tags.iter().any(|tag| entry.tags.contains(tag)) {
            return false;
        }

        // Check required tags
        if !self.required_tags.is_empty() {
            return self.required_tags.iter().all(|tag| entry.tags.contains(tag));
        }

        true
    }
}

impl ThreadFilter {
    pub fn new() -> Self {
        Self {
            allowed_threads: Vec::new(),
            blocked_threads: Vec::new(),
        }
    }

    pub fn allow_thread(mut self, thread: String) -> Self {
        self.allowed_threads.push(thread);
        self
    }

    pub fn block_thread(mut self, thread: String) -> Self {
        self.blocked_threads.push(thread);
        self
    }
}

impl LogFilter for ThreadFilter {
    fn should_log(&self, entry: &LogEntry) -> bool {
        if let Some(ref thread) = entry.thread {
            // Check blocked threads first
            if self.blocked_threads.iter().any(|t| thread.contains(t)) {
                return false;
            }

            // If no allowed threads specified, allow all (except blocked)
            if self.allowed_threads.is_empty() {
                return true;
            }

            // Check if thread matches any allowed threads
            self.allowed_threads.iter().any(|t| thread.contains(t))
        } else {
            // No thread info - allow if no restrictions
            self.allowed_threads.is_empty()
        }
    }
}

impl MessageFilter {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            exclude: false,
        }
    }

    pub fn add_pattern(mut self, pattern: &str) -> Result<Self, regex::Error> {
        let regex = Regex::new(pattern)?;
        self.patterns.push(regex);
        Ok(self)
    }

    pub fn exclude_patterns(mut self, exclude: bool) -> Self {
        self.exclude = exclude;
        self
    }
}

impl LogFilter for MessageFilter {
    fn should_log(&self, entry: &LogEntry) -> bool {
        if self.patterns.is_empty() {
            return true;
        }

        let matches = self.patterns.iter().any(|pattern| pattern.is_match(&entry.message));
        
        if self.exclude {
            !matches
        } else {
            matches
        }
    }
}

impl CompositeFilter {
    pub fn new(mode: FilterMode) -> Self {
        Self {
            filters: Vec::new(),
            mode,
        }
    }

    pub fn and() -> Self {
        Self::new(FilterMode::And)
    }

    pub fn or() -> Self {
        Self::new(FilterMode::Or)
    }

    pub fn add_filter(mut self, filter: Box<dyn LogFilter>) -> Self {
        self.filters.push(filter);
        self
    }
}

impl LogFilter for CompositeFilter {
    fn should_log(&self, entry: &LogEntry) -> bool {
        if self.filters.is_empty() {
            return true;
        }

        match self.mode {
            FilterMode::And => self.filters.iter().all(|f| f.should_log(entry)),
            FilterMode::Or => self.filters.iter().any(|f| f.should_log(entry)),
        }
    }
}

// Specialized filters for common use cases
pub struct SensitiveDataFilter {
    patterns: Vec<Regex>,
}

impl SensitiveDataFilter {
    pub fn new() -> Self {
        let patterns = vec![
            // Credit card numbers
            Regex::new(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b").unwrap(),
            // SSN
            Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
            // Email addresses (basic)
            Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap(),
            // Phone numbers
            Regex::new(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b").unwrap(),
            // API keys (basic patterns)
            Regex::new(r"\b[A-Za-z0-9]{20,}\b").unwrap(),
            // Passwords in URLs
            Regex::new(r"://[^:/@]+:[^@/]+@").unwrap(),
        ];

        Self { patterns }
    }

    fn sanitize_message(&self, message: &str) -> String {
        let mut sanitized = message.to_string();
        
        for pattern in &self.patterns {
            sanitized = pattern.replace_all(&sanitized, "[REDACTED]").to_string();
        }
        
        sanitized
    }
}

impl LogFilter for SensitiveDataFilter {
    fn should_log(&self, _entry: &LogEntry) -> bool {
        // This filter doesn't block logs, it sanitizes them
        // The sanitization would happen in a custom formatter
        true
    }
}

pub struct PerformanceFilter {
    min_duration_ms: u64,
}

impl PerformanceFilter {
    pub fn new(min_duration_ms: u64) -> Self {
        Self { min_duration_ms }
    }
}

impl LogFilter for PerformanceFilter {
    fn should_log(&self, entry: &LogEntry) -> bool {
        if let Some(metadata) = entry.metadata.as_object() {
            if let Some(duration) = metadata.get("duration_ms") {
                if let Some(duration_val) = duration.as_u64() {
                    return duration_val >= self.min_duration_ms;
                }
            }
        }
        true // Allow logs without duration metadata
    }
}