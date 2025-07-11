use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::error::Result;
use super::{LogEntry, LogLevel, LogContext};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredLogEntry {
    pub id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: LogLevel,
    pub service: String,
    pub version: String,
    pub environment: String,
    pub message: String,
    pub operation: Option<String>,
    pub component: Option<String>,
    pub trace_id: Option<String>,
    pub span_id: Option<String>,
    pub parent_span_id: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub request_id: Option<String>,
    pub correlation_id: Option<String>,
    pub source: SourceInfo,
    pub duration_ms: Option<u64>,
    pub status: Option<String>,
    pub error: Option<ErrorInfo>,
    pub metrics: HashMap<String, Value>,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub module: Option<String>,
    pub function: Option<String>,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub thread: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    pub error_type: String,
    pub error_message: String,
    pub error_code: Option<String>,
    pub stack_trace: Option<String>,
    pub cause: Option<Box<ErrorInfo>>,
}

#[derive(Debug, Clone)]
pub struct StructuredLogger {
    service: String,
    version: String,
    environment: String,
    context: Arc<RwLock<LogContext>>,
    spans: Arc<RwLock<HashMap<String, SpanInfo>>>,
}

#[derive(Debug, Clone)]
pub struct SpanInfo {
    pub span_id: String,
    pub trace_id: String,
    pub parent_span_id: Option<String>,
    pub operation: String,
    pub start_time: Instant,
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct SpanGuard {
    logger: StructuredLogger,
    span_id: String,
    start_time: Instant,
}

impl StructuredLogger {
    pub fn new(service: String, version: String, environment: String) -> Self {
        Self {
            service,
            version,
            environment,
            context: Arc::new(RwLock::new(LogContext::default())),
            spans: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn log_structured(&self, entry: StructuredLogEntry) -> Result<()> {
        // Convert to regular log entry for processing
        let log_entry = LogEntry {
            id: entry.id,
            timestamp: entry.timestamp,
            level: entry.level,
            target: entry.service,
            message: entry.message,
            module: entry.source.module,
            file: entry.source.file,
            line: entry.source.line,
            thread: entry.source.thread,
            session_id: entry.session_id,
            user_id: entry.user_id,
            request_id: entry.request_id,
            tags: entry.tags,
            metadata: serde_json::to_value(&entry.metadata).unwrap_or_default(),
        };

        // Here you would send to your logging infrastructure
        // For now, we'll just serialize to JSON
        let json = serde_json::to_string(&entry)?;
        println!("{}", json);

        Ok(())
    }

    pub async fn info(&self, message: &str) -> Result<()> {
        let context = self.context.read().await;
        let entry = self.create_entry(LogLevel::Info, message.to_string(), &context).await;
        self.log_structured(entry).await
    }

    pub async fn warn(&self, message: &str) -> Result<()> {
        let context = self.context.read().await;
        let entry = self.create_entry(LogLevel::Warn, message.to_string(), &context).await;
        self.log_structured(entry).await
    }

    pub async fn error(&self, message: &str) -> Result<()> {
        let context = self.context.read().await;
        let entry = self.create_entry(LogLevel::Error, message.to_string(), &context).await;
        self.log_structured(entry).await
    }

    pub async fn debug(&self, message: &str) -> Result<()> {
        let context = self.context.read().await;
        let entry = self.create_entry(LogLevel::Debug, message.to_string(), &context).await;
        self.log_structured(entry).await
    }

    pub async fn trace(&self, message: &str) -> Result<()> {
        let context = self.context.read().await;
        let entry = self.create_entry(LogLevel::Trace, message.to_string(), &context).await;
        self.log_structured(entry).await
    }

    pub async fn error_with_details(&self, message: &str, error: &dyn std::error::Error) -> Result<()> {
        let context = self.context.read().await;
        let mut entry = self.create_entry(LogLevel::Error, message.to_string(), &context).await;
        
        entry.error = Some(ErrorInfo {
            error_type: error.to_string(),
            error_message: error.to_string(),
            error_code: None,
            stack_trace: Some(format!("{:?}", error)),
            cause: None,
        });

        self.log_structured(entry).await
    }

    pub async fn with_operation<F, R>(&self, operation: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _span = self.start_span(operation).await;
        f()
    }

    pub async fn start_span(&self, operation: &str) -> SpanGuard {
        let span_id = Uuid::new_v4().to_string();
        let trace_id = Uuid::new_v4().to_string();
        let start_time = Instant::now();

        let span_info = SpanInfo {
            span_id: span_id.clone(),
            trace_id,
            parent_span_id: None,
            operation: operation.to_string(),
            start_time,
            metadata: HashMap::new(),
        };

        {
            let mut spans = self.spans.write().await;
            spans.insert(span_id.clone(), span_info);
        }

        SpanGuard {
            logger: self.clone(),
            span_id,
            start_time,
        }
    }

    pub async fn record_metric(&self, name: &str, value: f64, unit: &str) -> Result<()> {
        let context = self.context.read().await;
        let mut entry = self.create_entry(LogLevel::Info, format!("Metric: {}", name), &context).await;
        
        entry.metrics.insert(name.to_string(), serde_json::json!({
            "value": value,
            "unit": unit,
            "timestamp": chrono::Utc::now(),
        }));

        self.log_structured(entry).await
    }

    pub async fn record_performance(&self, operation: &str, duration: Duration) -> Result<()> {
        let context = self.context.read().await;
        let mut entry = self.create_entry(
            LogLevel::Info,
            format!("Performance: {}", operation),
            &context,
        ).await;
        
        entry.duration_ms = Some(duration.as_millis() as u64);
        entry.operation = Some(operation.to_string());

        self.log_structured(entry).await
    }

    pub async fn audit_log(&self, action: &str, resource: &str, outcome: &str) -> Result<()> {
        let context = self.context.read().await;
        let mut entry = self.create_entry(
            LogLevel::Info,
            format!("Audit: {} on {}", action, resource),
            &context,
        ).await;
        
        entry.tags.push("audit".to_string());
        entry.metadata.insert("action".to_string(), serde_json::Value::String(action.to_string()));
        entry.metadata.insert("resource".to_string(), serde_json::Value::String(resource.to_string()));
        entry.metadata.insert("outcome".to_string(), serde_json::Value::String(outcome.to_string()));

        self.log_structured(entry).await
    }

    pub async fn security_log(&self, event: &str, severity: &str, details: HashMap<String, Value>) -> Result<()> {
        let context = self.context.read().await;
        let mut entry = self.create_entry(
            LogLevel::Warn,
            format!("Security: {}", event),
            &context,
        ).await;
        
        entry.tags.push("security".to_string());
        entry.metadata.insert("event".to_string(), serde_json::Value::String(event.to_string()));
        entry.metadata.insert("severity".to_string(), serde_json::Value::String(severity.to_string()));
        entry.metadata.extend(details);

        self.log_structured(entry).await
    }

    async fn create_entry(&self, level: LogLevel, message: String, context: &LogContext) -> StructuredLogEntry {
        let spans = self.spans.read().await;
        let current_span = spans.values().next(); // Get the most recent span

        StructuredLogEntry {
            id: Uuid::new_v4(),
            timestamp: chrono::Utc::now(),
            level,
            service: self.service.clone(),
            version: self.version.clone(),
            environment: self.environment.clone(),
            message,
            operation: context.operation.clone(),
            component: context.component.clone(),
            trace_id: current_span.map(|s| s.trace_id.clone()),
            span_id: current_span.map(|s| s.span_id.clone()),
            parent_span_id: current_span.and_then(|s| s.parent_span_id.clone()),
            user_id: context.user_id.clone(),
            session_id: context.session_id.clone(),
            request_id: context.request_id.clone(),
            correlation_id: None,
            source: SourceInfo {
                module: None,
                function: None,
                file: None,
                line: None,
                thread: std::thread::current().name().map(|s| s.to_string()),
            },
            duration_ms: None,
            status: None,
            error: None,
            metrics: HashMap::new(),
            tags: context.tags.clone(),
            metadata: context.metadata.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        }
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
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        let logger = self.logger.clone();
        let span_id = self.span_id.clone();

        // Log span completion
        tokio::spawn(async move {
            if let Err(e) = logger.record_performance(&format!("span_{}", span_id), duration).await {
                eprintln!("Failed to record span performance: {}", e);
            }

            // Remove span from active spans
            let mut spans = logger.spans.write().await;
            spans.remove(&span_id);
        });
    }
}

// Helper macros for structured logging
#[macro_export]
macro_rules! log_info_structured {
    ($logger:expr, $message:expr, $($key:expr => $value:expr),*) => {
        {
            let mut entry = $crate::logging::structured::StructuredLogEntry {
                id: uuid::Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                level: $crate::logging::LogLevel::Info,
                service: $logger.service.clone(),
                version: $logger.version.clone(),
                environment: $logger.environment.clone(),
                message: $message.to_string(),
                operation: None,
                component: None,
                trace_id: None,
                span_id: None,
                parent_span_id: None,
                user_id: None,
                session_id: None,
                request_id: None,
                correlation_id: None,
                source: $crate::logging::structured::SourceInfo {
                    module: Some(module_path!().to_string()),
                    function: None,
                    file: Some(file!().to_string()),
                    line: Some(line!()),
                    thread: std::thread::current().name().map(|s| s.to_string()),
                },
                duration_ms: None,
                status: None,
                error: None,
                metrics: std::collections::HashMap::new(),
                tags: Vec::new(),
                metadata: {
                    let mut metadata = std::collections::HashMap::new();
                    $(
                        metadata.insert($key.to_string(), serde_json::json!($value));
                    )*
                    metadata
                },
            };
            $logger.log_structured(entry).await
        }
    };
}

#[macro_export]
macro_rules! log_error_structured {
    ($logger:expr, $message:expr, $error:expr) => {
        {
            let mut entry = $crate::logging::structured::StructuredLogEntry {
                id: uuid::Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                level: $crate::logging::LogLevel::Error,
                service: $logger.service.clone(),
                version: $logger.version.clone(),
                environment: $logger.environment.clone(),
                message: $message.to_string(),
                operation: None,
                component: None,
                trace_id: None,
                span_id: None,
                parent_span_id: None,
                user_id: None,
                session_id: None,
                request_id: None,
                correlation_id: None,
                source: $crate::logging::structured::SourceInfo {
                    module: Some(module_path!().to_string()),
                    function: None,
                    file: Some(file!().to_string()),
                    line: Some(line!()),
                    thread: std::thread::current().name().map(|s| s.to_string()),
                },
                duration_ms: None,
                status: None,
                error: Some($crate::logging::structured::ErrorInfo {
                    error_type: std::any::type_name_of_val(&$error).to_string(),
                    error_message: $error.to_string(),
                    error_code: None,
                    stack_trace: Some(format!("{:?}", $error)),
                    cause: None,
                }),
                metrics: std::collections::HashMap::new(),
                tags: Vec::new(),
                metadata: std::collections::HashMap::new(),
            };
            $logger.log_structured(entry).await
        }
    };
}

// Performance logging utilities
pub struct PerformanceLogger {
    logger: StructuredLogger,
}

impl PerformanceLogger {
    pub fn new(logger: StructuredLogger) -> Self {
        Self { logger }
    }

    pub async fn time_operation<F, R>(&self, operation: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        if let Err(e) = self.logger.record_performance(operation, duration).await {
            eprintln!("Failed to record performance: {}", e);
        }

        result
    }

    pub async fn time_async_operation<F, R>(&self, operation: &str, f: F) -> R
    where
        F: std::future::Future<Output = R>,
    {
        let start = Instant::now();
        let result = f.await;
        let duration = start.elapsed();

        if let Err(e) = self.logger.record_performance(operation, duration).await {
            eprintln!("Failed to record performance: {}", e);
        }

        result
    }
}