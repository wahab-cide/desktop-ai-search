use chrono::{DateTime, Utc};
use serde_json;

use super::{LogEntry, LogFormatter, LogLevel};

pub struct JsonFormatter {
    include_metadata: bool,
    pretty: bool,
}

pub struct TextFormatter {
    include_thread: bool,
    include_module: bool,
    include_location: bool,
    timestamp_format: String,
}

pub struct CompactFormatter {
    include_timestamp: bool,
    include_level: bool,
}

pub struct StructuredFormatter {
    service_name: String,
    version: String,
    environment: String,
}

impl JsonFormatter {
    pub fn new() -> Self {
        Self {
            include_metadata: true,
            pretty: false,
        }
    }

    pub fn with_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    pub fn pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }
}

impl LogFormatter for JsonFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        let mut json = serde_json::json!({
            "id": entry.id,
            "timestamp": entry.timestamp,
            "level": entry.level.to_string(),
            "target": entry.target,
            "message": entry.message,
        });

        if let Some(ref module) = entry.module {
            json["module"] = serde_json::Value::String(module.clone());
        }

        if let Some(ref file) = entry.file {
            json["file"] = serde_json::Value::String(file.clone());
        }

        if let Some(line) = entry.line {
            json["line"] = serde_json::Value::Number(line.into());
        }

        if let Some(ref thread) = entry.thread {
            json["thread"] = serde_json::Value::String(thread.clone());
        }

        if let Some(ref session_id) = entry.session_id {
            json["session_id"] = serde_json::Value::String(session_id.clone());
        }

        if let Some(ref user_id) = entry.user_id {
            json["user_id"] = serde_json::Value::String(user_id.clone());
        }

        if let Some(ref request_id) = entry.request_id {
            json["request_id"] = serde_json::Value::String(request_id.clone());
        }

        if !entry.tags.is_empty() {
            json["tags"] = serde_json::Value::Array(
                entry.tags.iter().map(|t| serde_json::Value::String(t.clone())).collect()
            );
        }

        if self.include_metadata && !entry.metadata.is_null() {
            json["metadata"] = entry.metadata.clone();
        }

        if self.pretty {
            serde_json::to_string_pretty(&json).unwrap_or_else(|_| "{}".to_string())
        } else {
            serde_json::to_string(&json).unwrap_or_else(|_| "{}".to_string())
        }
    }
}

impl TextFormatter {
    pub fn new() -> Self {
        Self {
            include_thread: false,
            include_module: true,
            include_location: false,
            timestamp_format: "%Y-%m-%d %H:%M:%S%.3f".to_string(),
        }
    }

    pub fn with_thread(mut self, include: bool) -> Self {
        self.include_thread = include;
        self
    }

    pub fn with_module(mut self, include: bool) -> Self {
        self.include_module = include;
        self
    }

    pub fn with_location(mut self, include: bool) -> Self {
        self.include_location = include;
        self
    }

    pub fn with_timestamp_format(mut self, format: String) -> Self {
        self.timestamp_format = format;
        self
    }
}

impl LogFormatter for TextFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        let timestamp = entry.timestamp.format(&self.timestamp_format);
        let level = format!("{:5}", entry.level.to_string());
        
        let mut parts = Vec::new();
        parts.push(timestamp.to_string());
        parts.push(level);

        if self.include_thread {
            if let Some(ref thread) = entry.thread {
                parts.push(format!("[{}]", thread));
            }
        }

        if self.include_module {
            if let Some(ref module) = entry.module {
                parts.push(format!("{}:", module));
            }
        }

        if self.include_location {
            if let (Some(ref file), Some(line)) = (&entry.file, entry.line) {
                parts.push(format!("{}:{}:", file, line));
            }
        }

        parts.push(entry.message.clone());

        if !entry.tags.is_empty() {
            parts.push(format!("[{}]", entry.tags.join(", ")));
        }

        if let Some(ref request_id) = entry.request_id {
            parts.push(format!("req_id:{}", request_id));
        }

        parts.join(" ")
    }
}

impl CompactFormatter {
    pub fn new() -> Self {
        Self {
            include_timestamp: true,
            include_level: true,
        }
    }

    pub fn with_timestamp(mut self, include: bool) -> Self {
        self.include_timestamp = include;
        self
    }

    pub fn with_level(mut self, include: bool) -> Self {
        self.include_level = include;
        self
    }
}

impl LogFormatter for CompactFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        let mut parts = Vec::new();

        if self.include_timestamp {
            let timestamp = entry.timestamp.format("%H:%M:%S");
            parts.push(timestamp.to_string());
        }

        if self.include_level {
            let level = match entry.level {
                LogLevel::Error => "E",
                LogLevel::Warn => "W",
                LogLevel::Info => "I",
                LogLevel::Debug => "D",
                LogLevel::Trace => "T",
            };
            parts.push(level.to_string());
        }

        parts.push(entry.message.clone());

        parts.join(" ")
    }
}

impl StructuredFormatter {
    pub fn new(service_name: String, version: String, environment: String) -> Self {
        Self {
            service_name,
            version,
            environment,
        }
    }
}

impl LogFormatter for StructuredFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        let mut json = serde_json::json!({
            "@timestamp": entry.timestamp,
            "@version": "1",
            "service": self.service_name,
            "version": self.version,
            "environment": self.environment,
            "level": entry.level.to_string(),
            "logger": entry.target,
            "message": entry.message,
            "trace_id": entry.request_id,
            "span_id": entry.session_id,
        });

        if let Some(ref module) = entry.module {
            json["source"] = serde_json::json!({
                "module": module,
                "file": entry.file.as_ref().unwrap_or(&"unknown".to_string()),
                "line": entry.line.unwrap_or(0),
            });
        }

        if let Some(ref thread) = entry.thread {
            json["thread"] = serde_json::Value::String(thread.clone());
        }

        if let Some(ref user_id) = entry.user_id {
            json["user_id"] = serde_json::Value::String(user_id.clone());
        }

        if !entry.tags.is_empty() {
            json["tags"] = serde_json::Value::Array(
                entry.tags.iter().map(|t| serde_json::Value::String(t.clone())).collect()
            );
        }

        if !entry.metadata.is_null() {
            json["metadata"] = entry.metadata.clone();
        }

        // Add performance metrics if available
        if let Some(metadata) = entry.metadata.as_object() {
            if let Some(duration) = metadata.get("duration") {
                json["duration_ms"] = duration.clone();
            }
            if let Some(memory) = metadata.get("memory_usage") {
                json["memory_usage_mb"] = memory.clone();
            }
        }

        serde_json::to_string(&json).unwrap_or_else(|_| "{}".to_string())
    }
}

// Helper function to colorize log levels for terminal output
pub fn colorize_level(level: &LogLevel) -> String {
    match level {
        LogLevel::Error => format!("\x1b[31m{}\x1b[0m", level.to_string()), // Red
        LogLevel::Warn => format!("\x1b[33m{}\x1b[0m", level.to_string()),  // Yellow
        LogLevel::Info => format!("\x1b[32m{}\x1b[0m", level.to_string()),  // Green
        LogLevel::Debug => format!("\x1b[36m{}\x1b[0m", level.to_string()), // Cyan
        LogLevel::Trace => format!("\x1b[35m{}\x1b[0m", level.to_string()), // Magenta
    }
}

pub struct ColoredTextFormatter {
    inner: TextFormatter,
    enable_colors: bool,
}

impl ColoredTextFormatter {
    pub fn new() -> Self {
        Self {
            inner: TextFormatter::new(),
            enable_colors: atty::is(atty::Stream::Stdout),
        }
    }

    pub fn with_colors(mut self, enable: bool) -> Self {
        self.enable_colors = enable;
        self
    }

    pub fn with_thread(mut self, include: bool) -> Self {
        self.inner = self.inner.with_thread(include);
        self
    }

    pub fn with_module(mut self, include: bool) -> Self {
        self.inner = self.inner.with_module(include);
        self
    }

    pub fn with_location(mut self, include: bool) -> Self {
        self.inner = self.inner.with_location(include);
        self
    }
}

impl LogFormatter for ColoredTextFormatter {
    fn format(&self, entry: &LogEntry) -> String {
        let formatted = self.inner.format(entry);
        
        if self.enable_colors {
            let colored_level = colorize_level(&entry.level);
            formatted.replace(&entry.level.to_string(), &colored_level)
        } else {
            formatted
        }
    }
}