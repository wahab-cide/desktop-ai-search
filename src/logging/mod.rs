use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

pub mod structured;
pub mod filters;
pub mod formatters;
pub mod appenders;

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    pub level: LogLevel,
    pub file_path: Option<PathBuf>,
    pub max_file_size: u64,
    pub max_files: u32,
    pub enable_console: bool,
    pub enable_file: bool,
    pub format: LogFormat,
    pub enable_structured: bool,
    pub buffer_size: usize,
    pub flush_interval: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogFormat {
    Json,
    Text,
    Compact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub target: String,
    pub message: String,
    pub module: Option<String>,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub thread: Option<String>,
    pub session_id: Option<String>,
    pub user_id: Option<String>,
    pub request_id: Option<String>,
    pub tags: Vec<String>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogContext {
    pub session_id: Option<String>,
    pub user_id: Option<String>,
    pub request_id: Option<String>,
    pub operation: Option<String>,
    pub component: Option<String>,
    pub tags: Vec<String>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

pub struct Logger {
    config: LogConfig,
    context: Arc<RwLock<LogContext>>,
    appenders: Vec<Box<dyn LogAppender>>,
    filters: Vec<Box<dyn LogFilter>>,
    formatter: Box<dyn LogFormatter>,
}

pub trait LogAppender: Send + Sync {
    fn append(&self, entry: &LogEntry) -> Result<()>;
    fn flush(&self) -> Result<()>;
}

pub trait LogFilter: Send + Sync {
    fn should_log(&self, entry: &LogEntry) -> bool;
}

pub trait LogFormatter: Send + Sync {
    fn format(&self, entry: &LogEntry) -> String;
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            file_path: Some(PathBuf::from("logs/app.log")),
            max_file_size: 10 * 1024 * 1024, // 10MB
            max_files: 5,
            enable_console: true,
            enable_file: true,
            format: LogFormat::Json,
            enable_structured: true,
            buffer_size: 1024,
            flush_interval: 1000, // 1 second
        }
    }
}

impl LogLevel {
    pub fn to_string(&self) -> &'static str {
        match self {
            LogLevel::Error => "ERROR",
            LogLevel::Warn => "WARN",
            LogLevel::Info => "INFO",
            LogLevel::Debug => "DEBUG",
            LogLevel::Trace => "TRACE",
        }
    }

    pub fn from_string(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "ERROR" => Some(LogLevel::Error),
            "WARN" => Some(LogLevel::Warn),
            "INFO" => Some(LogLevel::Info),
            "DEBUG" => Some(LogLevel::Debug),
            "TRACE" => Some(LogLevel::Trace),
            _ => None,
        }
    }

    pub fn priority(&self) -> u8 {
        match self {
            LogLevel::Error => 1,
            LogLevel::Warn => 2,
            LogLevel::Info => 3,
            LogLevel::Debug => 4,
            LogLevel::Trace => 5,
        }
    }
}

impl LogEntry {
    pub fn new(
        level: LogLevel,
        target: String,
        message: String,
        context: &LogContext,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            level,
            target,
            message,
            module: None,
            file: None,
            line: None,
            thread: std::thread::current().name().map(|s| s.to_string()),
            session_id: context.session_id.clone(),
            user_id: context.user_id.clone(),
            request_id: context.request_id.clone(),
            tags: context.tags.clone(),
            metadata: serde_json::Value::Object(context.metadata.clone()),
        }
    }

    pub fn with_location(mut self, module: &str, file: &str, line: u32) -> Self {
        self.module = Some(module.to_string());
        self.file = Some(file.to_string());
        self.line = Some(line);
        self
    }

    pub fn with_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        if let serde_json::Value::Object(ref mut map) = self.metadata {
            map.insert(key.to_string(), value);
        }
        self
    }

    pub fn with_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }
}

impl Default for LogContext {
    fn default() -> Self {
        Self {
            session_id: None,
            user_id: None,
            request_id: None,
            operation: None,
            component: None,
            tags: Vec::new(),
            metadata: serde_json::Map::new(),
        }
    }
}

impl Logger {
    pub fn new(config: LogConfig) -> Result<Self> {
        let context = Arc::new(RwLock::new(LogContext::default()));
        let mut appenders: Vec<Box<dyn LogAppender>> = Vec::new();
        let mut filters: Vec<Box<dyn LogFilter>> = Vec::new();
        
        // Add console appender
        if config.enable_console {
            appenders.push(Box::new(appenders::ConsoleAppender::new()));
        }
        
        // Add file appender
        if config.enable_file {
            if let Some(ref path) = config.file_path {
                // Create logs directory if it doesn't exist
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                
                appenders.push(Box::new(appenders::FileAppender::new(
                    path.clone(),
                    config.max_file_size,
                    config.max_files,
                )?));
            }
        }
        
        // Add level filter
        filters.push(Box::new(filters::LevelFilter::new(config.level.clone())));
        
        // Create formatter
        let formatter: Box<dyn LogFormatter> = match config.format {
            LogFormat::Json => Box::new(formatters::JsonFormatter::new()),
            LogFormat::Text => Box::new(formatters::TextFormatter::new()),
            LogFormat::Compact => Box::new(formatters::CompactFormatter::new()),
        };
        
        Ok(Self {
            config,
            context,
            appenders,
            filters,
            formatter,
        })
    }

    pub async fn log(&self, entry: LogEntry) -> Result<()> {
        // Apply filters
        for filter in &self.filters {
            if !filter.should_log(&entry) {
                return Ok(());
            }
        }

        // Format the entry
        let formatted = self.formatter.format(&entry);

        // Send to appenders
        for appender in &self.appenders {
            appender.append(&entry)?;
        }

        Ok(())
    }

    pub async fn error(&self, message: &str) -> Result<()> {
        let context = self.context.read().await;
        let entry = LogEntry::new(
            LogLevel::Error,
            "app".to_string(),
            message.to_string(),
            &context,
        );
        self.log(entry).await
    }

    pub async fn warn(&self, message: &str) -> Result<()> {
        let context = self.context.read().await;
        let entry = LogEntry::new(
            LogLevel::Warn,
            "app".to_string(),
            message.to_string(),
            &context,
        );
        self.log(entry).await
    }

    pub async fn info(&self, message: &str) -> Result<()> {
        let context = self.context.read().await;
        let entry = LogEntry::new(
            LogLevel::Info,
            "app".to_string(),
            message.to_string(),
            &context,
        );
        self.log(entry).await
    }

    pub async fn debug(&self, message: &str) -> Result<()> {
        let context = self.context.read().await;
        let entry = LogEntry::new(
            LogLevel::Debug,
            "app".to_string(),
            message.to_string(),
            &context,
        );
        self.log(entry).await
    }

    pub async fn trace(&self, message: &str) -> Result<()> {
        let context = self.context.read().await;
        let entry = LogEntry::new(
            LogLevel::Trace,
            "app".to_string(),
            message.to_string(),
            &context,
        );
        self.log(entry).await
    }

    pub async fn set_context(&self, context: LogContext) {
        let mut current_context = self.context.write().await;
        *current_context = context;
    }

    pub async fn update_context<F>(&self, f: F) 
    where 
        F: FnOnce(&mut LogContext),
    {
        let mut context = self.context.write().await;
        f(&mut context);
    }

    pub async fn with_context<F, R>(&self, context: LogContext, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let original_context = {
            let mut current_context = self.context.write().await;
            let original = current_context.clone();
            *current_context = context;
            original
        };

        let result = f();

        {
            let mut current_context = self.context.write().await;
            *current_context = original_context;
        }

        result
    }

    pub async fn flush(&self) -> Result<()> {
        for appender in &self.appenders {
            appender.flush()?;
        }
        Ok(())
    }
}

// Convenience macros for logging
#[macro_export]
macro_rules! log_error {
    ($logger:expr, $($arg:tt)*) => {
        $logger.error(&format!($($arg)*)).await.unwrap_or_else(|e| {
            eprintln!("Failed to log error: {}", e);
        });
    };
}

#[macro_export]
macro_rules! log_warn {
    ($logger:expr, $($arg:tt)*) => {
        $logger.warn(&format!($($arg)*)).await.unwrap_or_else(|e| {
            eprintln!("Failed to log warning: {}", e);
        });
    };
}

#[macro_export]
macro_rules! log_info {
    ($logger:expr, $($arg:tt)*) => {
        $logger.info(&format!($($arg)*)).await.unwrap_or_else(|e| {
            eprintln!("Failed to log info: {}", e);
        });
    };
}

#[macro_export]
macro_rules! log_debug {
    ($logger:expr, $($arg:tt)*) => {
        $logger.debug(&format!($($arg)*)).await.unwrap_or_else(|e| {
            eprintln!("Failed to log debug: {}", e);
        });
    };
}

#[macro_export]
macro_rules! log_trace {
    ($logger:expr, $($arg:tt)*) => {
        $logger.trace(&format!($($arg)*)).await.unwrap_or_else(|e| {
            eprintln!("Failed to log trace: {}", e);
        });
    };
}