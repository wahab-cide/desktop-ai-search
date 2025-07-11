use std::collections::HashMap;

use regex::Regex;

/// Sanitizes input by removing or escaping potentially dangerous content
pub fn sanitize_input(input: &str) -> String {
    let mut sanitized = input.to_string();
    
    // Remove control characters except newlines and tabs
    sanitized = sanitized.chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\r' || *c == '\t')
        .collect();
    
    // Remove null bytes
    sanitized = sanitized.replace('\0', "");
    
    // Escape HTML characters
    sanitized = escape_html(&sanitized);
    
    // Remove potentially dangerous SQL patterns
    sanitized = sanitize_sql(&sanitized);
    
    // Remove script tags and javascript
    sanitized = sanitize_scripts(&sanitized);
    
    sanitized
}

/// Escapes HTML characters to prevent XSS attacks
pub fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
        .replace('/', "&#x2F;")
}

/// Sanitizes SQL-related patterns
pub fn sanitize_sql(input: &str) -> String {
    let dangerous_patterns = [
        (r"(?i)(union\s+select)", "UNION SELECT"),
        (r"(?i)(insert\s+into)", "INSERT INTO"),
        (r"(?i)(delete\s+from)", "DELETE FROM"),
        (r"(?i)(update\s+set)", "UPDATE SET"),
        (r"(?i)(drop\s+table)", "DROP TABLE"),
        (r"(?i)(alter\s+table)", "ALTER TABLE"),
        (r"(?i)(create\s+table)", "CREATE TABLE"),
        (r"(?i)(exec\s*\()", "EXEC("),
        (r"(?i)(execute\s*\()", "EXECUTE("),
        (r"(?i)(sp_executesql)", "sp_executesql"),
        (r"(?i)(xp_cmdshell)", "xp_cmdshell"),
        (r"(?i)(information_schema)", "information_schema"),
        (r"(?i)(sys\.)", "sys."),
    ];

    let mut sanitized = input.to_string();
    
    for (pattern, replacement) in &dangerous_patterns {
        if let Ok(regex) = Regex::new(pattern) {
            sanitized = regex.replace_all(&sanitized, format!("[BLOCKED:{}]", replacement)).to_string();
        }
    }
    
    sanitized
}

/// Removes script tags and javascript code
pub fn sanitize_scripts(input: &str) -> String {
    let script_patterns = [
        r"(?i)<script[^>]*>.*?</script>",
        r"(?i)<script[^>]*>",
        r"(?i)</script>",
        r"(?i)javascript:",
        r"(?i)vbscript:",
        r"(?i)data:text/html",
        r"(?i)data:text/javascript",
        r"(?i)onload\s*=",
        r"(?i)onerror\s*=",
        r"(?i)onclick\s*=",
        r"(?i)onmouseover\s*=",
        r"(?i)onmouseout\s*=",
        r"(?i)onfocus\s*=",
        r"(?i)onblur\s*=",
        r"(?i)onkeydown\s*=",
        r"(?i)onkeyup\s*=",
        r"(?i)onkeypress\s*=",
        r"(?i)onchange\s*=",
        r"(?i)onsubmit\s*=",
    ];

    let mut sanitized = input.to_string();
    
    for pattern in &script_patterns {
        if let Ok(regex) = Regex::new(pattern) {
            sanitized = regex.replace_all(&sanitized, "[BLOCKED:SCRIPT]").to_string();
        }
    }
    
    sanitized
}

/// Sanitizes file paths to prevent path traversal attacks
pub fn sanitize_path(path: &str) -> String {
    let mut sanitized = path.to_string();
    
    // Remove path traversal sequences
    sanitized = sanitized.replace("../", "");
    sanitized = sanitized.replace("..\\", "");
    sanitized = sanitized.replace("./", "");
    sanitized = sanitized.replace(".\\", "");
    
    // Remove null bytes
    sanitized = sanitized.replace('\0', "");
    
    // Remove control characters
    sanitized = sanitized.chars()
        .filter(|c| !c.is_control())
        .collect();
    
    // Remove leading/trailing whitespace
    sanitized = sanitized.trim().to_string();
    
    // Normalize path separators
    sanitized = sanitized.replace('\\', "/");
    
    // Remove duplicate slashes
    while sanitized.contains("//") {
        sanitized = sanitized.replace("//", "/");
    }
    
    sanitized
}

/// Sanitizes URLs to prevent malicious redirects
pub fn sanitize_url(url: &str) -> String {
    let mut sanitized = url.to_string();
    
    // Remove control characters
    sanitized = sanitized.chars()
        .filter(|c| !c.is_control())
        .collect();
    
    // Remove null bytes
    sanitized = sanitized.replace('\0', "");
    
    // Block dangerous protocols
    let dangerous_protocols = [
        "javascript:",
        "vbscript:",
        "data:text/html",
        "data:text/javascript",
        "file:",
        "ftp:",
    ];
    
    let sanitized_lower = sanitized.to_lowercase();
    for protocol in &dangerous_protocols {
        if sanitized_lower.starts_with(protocol) {
            return "[BLOCKED:DANGEROUS_PROTOCOL]".to_string();
        }
    }
    
    // Ensure URL starts with safe protocols
    if !sanitized_lower.starts_with("http://") && !sanitized_lower.starts_with("https://") && !sanitized_lower.starts_with("mailto:") {
        sanitized = format!("https://{}", sanitized);
    }
    
    sanitized
}

/// Sanitizes email addresses
pub fn sanitize_email(email: &str) -> String {
    let mut sanitized = email.to_string();
    
    // Remove control characters
    sanitized = sanitized.chars()
        .filter(|c| !c.is_control())
        .collect();
    
    // Remove null bytes
    sanitized = sanitized.replace('\0', "");
    
    // Remove leading/trailing whitespace
    sanitized = sanitized.trim().to_string();
    
    // Convert to lowercase
    sanitized = sanitized.to_lowercase();
    
    // Basic email validation
    if !sanitized.contains('@') || sanitized.chars().filter(|&c| c == '@').count() != 1 {
        return "[INVALID_EMAIL]".to_string();
    }
    
    sanitized
}

/// Sanitizes JSON input
pub fn sanitize_json(json: &str) -> String {
    let mut sanitized = json.to_string();
    
    // Remove control characters except newlines and tabs
    sanitized = sanitized.chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\r' || *c == '\t')
        .collect();
    
    // Remove null bytes
    sanitized = sanitized.replace('\0', "");
    
    // Check for prototype pollution patterns
    if sanitized.contains("__proto__") || sanitized.contains("constructor") || sanitized.contains("prototype") {
        return "[BLOCKED:PROTOTYPE_POLLUTION]".to_string();
    }
    
    sanitized
}

/// Sanitizes CSV data
pub fn sanitize_csv(csv: &str) -> String {
    let mut sanitized_lines = Vec::new();
    
    for line in csv.lines() {
        let sanitized_line = sanitize_csv_line(line);
        sanitized_lines.push(sanitized_line);
    }
    
    sanitized_lines.join("\n")
}

/// Sanitizes a single CSV line
pub fn sanitize_csv_line(line: &str) -> String {
    let fields: Vec<&str> = line.split(',').collect();
    let mut sanitized_fields = Vec::new();
    
    for field in fields {
        let mut sanitized_field = field.trim().to_string();
        
        // Remove dangerous formula prefixes
        if sanitized_field.starts_with('=') || 
           sanitized_field.starts_with('+') || 
           sanitized_field.starts_with('-') || 
           sanitized_field.starts_with('@') ||
           sanitized_field.starts_with('\t') ||
           sanitized_field.starts_with('\r') {
            sanitized_field = format!("'{}", sanitized_field);
        }
        
        sanitized_fields.push(sanitized_field);
    }
    
    sanitized_fields.join(",")
}

/// Sanitizes XML content
pub fn sanitize_xml(xml: &str) -> String {
    let mut sanitized = xml.to_string();
    
    // Remove control characters except newlines and tabs
    sanitized = sanitized.chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\r' || *c == '\t')
        .collect();
    
    // Remove null bytes
    sanitized = sanitized.replace('\0', "");
    
    // Block external entity references
    let dangerous_patterns = [
        r"(?i)<!ENTITY",
        r"(?i)<!DOCTYPE[^>]*\[",
        r"(?i)SYSTEM\s+['\"]",
        r"(?i)PUBLIC\s+['\"]",
    ];
    
    for pattern in &dangerous_patterns {
        if let Ok(regex) = Regex::new(pattern) {
            sanitized = regex.replace_all(&sanitized, "[BLOCKED:EXTERNAL_ENTITY]").to_string();
        }
    }
    
    sanitized
}

/// Sanitizes command line arguments
pub fn sanitize_command_args(args: &str) -> String {
    let mut sanitized = args.to_string();
    
    // Remove control characters
    sanitized = sanitized.chars()
        .filter(|c| !c.is_control())
        .collect();
    
    // Remove null bytes
    sanitized = sanitized.replace('\0', "");
    
    // Block dangerous command patterns
    let dangerous_patterns = [
        r"(?i)\|\s*rm\s",
        r"(?i)\|\s*del\s",
        r"(?i)\|\s*format\s",
        r"(?i)\|\s*shutdown\s",
        r"(?i)\|\s*reboot\s",
        r"(?i);\s*rm\s",
        r"(?i);\s*del\s",
        r"(?i)&&\s*rm\s",
        r"(?i)&&\s*del\s",
        r"(?i)`[^`]*`",
        r"(?i)\$\([^)]*\)",
    ];
    
    for pattern in &dangerous_patterns {
        if let Ok(regex) = Regex::new(pattern) {
            sanitized = regex.replace_all(&sanitized, "[BLOCKED:COMMAND_INJECTION]").to_string();
        }
    }
    
    sanitized
}

/// Sanitizes regular expressions to prevent ReDoS attacks
pub fn sanitize_regex(regex: &str) -> String {
    let mut sanitized = regex.to_string();
    
    // Remove control characters
    sanitized = sanitized.chars()
        .filter(|c| !c.is_control())
        .collect();
    
    // Remove null bytes
    sanitized = sanitized.replace('\0', "");
    
    // Check for potentially dangerous patterns that can cause ReDoS
    let dangerous_patterns = [
        r"\(\?\:",  // Non-capturing groups in loops
        r"\(\?\!",  // Negative lookahead
        r"\(\?\<",  // Lookbehind
        r"\(\?\=",  // Positive lookahead
        r"\*\*",    // Nested quantifiers
        r"\+\+",    // Nested quantifiers
        r"\{\d+,\}.*\{\d+,\}", // Multiple unbounded quantifiers
    ];
    
    for pattern in &dangerous_patterns {
        if let Ok(regex) = Regex::new(pattern) {
            if regex.is_match(&sanitized) {
                return "[BLOCKED:DANGEROUS_REGEX]".to_string();
            }
        }
    }
    
    // Limit regex length
    if sanitized.len() > 1000 {
        sanitized = sanitized[..1000].to_string();
    }
    
    sanitized
}

/// Removes sensitive information from logs
pub fn sanitize_log_message(message: &str) -> String {
    let sensitive_patterns = [
        (r"\bpassword\s*[:=]\s*['\"]?[^'\"]*['\"]?", "password=[REDACTED]"),
        (r"\btoken\s*[:=]\s*['\"]?[^'\"]*['\"]?", "token=[REDACTED]"),
        (r"\bkey\s*[:=]\s*['\"]?[^'\"]*['\"]?", "key=[REDACTED]"),
        (r"\bsecret\s*[:=]\s*['\"]?[^'\"]*['\"]?", "secret=[REDACTED]"),
        (r"\bapi_key\s*[:=]\s*['\"]?[^'\"]*['\"]?", "api_key=[REDACTED]"),
        (r"\bauthentication\s*[:=]\s*['\"]?[^'\"]*['\"]?", "authentication=[REDACTED]"),
        (r"\bauthorization\s*[:=]\s*['\"]?[^'\"]*['\"]?", "authorization=[REDACTED]"),
        (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[REDACTED-CARD]"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED-EMAIL]"),
        (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[REDACTED-IP]"),
    ];

    let mut sanitized = message.to_string();
    
    for (pattern, replacement) in &sensitive_patterns {
        if let Ok(regex) = Regex::new(pattern) {
            sanitized = regex.replace_all(&sanitized, *replacement).to_string();
        }
    }
    
    sanitized
}

/// Comprehensive sanitization function that applies multiple sanitization strategies
pub fn comprehensive_sanitize(input: &str, content_type: ContentType) -> String {
    match content_type {
        ContentType::Html => sanitize_html_content(input),
        ContentType::Json => sanitize_json(input),
        ContentType::Xml => sanitize_xml(input),
        ContentType::Csv => sanitize_csv(input),
        ContentType::Url => sanitize_url(input),
        ContentType::Email => sanitize_email(input),
        ContentType::Path => sanitize_path(input),
        ContentType::Command => sanitize_command_args(input),
        ContentType::Regex => sanitize_regex(input),
        ContentType::Log => sanitize_log_message(input),
        ContentType::General => sanitize_input(input),
    }
}

/// Content types for sanitization
#[derive(Debug, Clone, PartialEq)]
pub enum ContentType {
    Html,
    Json,
    Xml,
    Csv,
    Url,
    Email,
    Path,
    Command,
    Regex,
    Log,
    General,
}

/// Sanitizes HTML content more thoroughly
fn sanitize_html_content(html: &str) -> String {
    let mut sanitized = escape_html(html);
    
    // Remove iframe, object, embed, and other dangerous tags
    let dangerous_tags = [
        r"(?i)<iframe[^>]*>.*?</iframe>",
        r"(?i)<object[^>]*>.*?</object>",
        r"(?i)<embed[^>]*>.*?</embed>",
        r"(?i)<applet[^>]*>.*?</applet>",
        r"(?i)<form[^>]*>.*?</form>",
        r"(?i)<meta[^>]*>",
        r"(?i)<link[^>]*>",
        r"(?i)<base[^>]*>",
        r"(?i)<style[^>]*>.*?</style>",
    ];
    
    for pattern in &dangerous_tags {
        if let Ok(regex) = Regex::new(pattern) {
            sanitized = regex.replace_all(&sanitized, "[BLOCKED:HTML_TAG]").to_string();
        }
    }
    
    sanitized
}

/// Creates a sanitization report
pub fn create_sanitization_report(original: &str, sanitized: &str) -> SanitizationReport {
    let changes_made = original != sanitized;
    let mut blocked_patterns = Vec::new();
    
    if sanitized.contains("[BLOCKED:") {
        // Extract blocked patterns
        if let Ok(regex) = Regex::new(r"\[BLOCKED:([^\]]+)\]") {
            for cap in regex.captures_iter(sanitized) {
                if let Some(pattern) = cap.get(1) {
                    blocked_patterns.push(pattern.as_str().to_string());
                }
            }
        }
    }
    
    SanitizationReport {
        original_length: original.len(),
        sanitized_length: sanitized.len(),
        changes_made,
        blocked_patterns,
        safety_score: calculate_safety_score(original, sanitized),
    }
}

/// Sanitization report structure
#[derive(Debug, Clone)]
pub struct SanitizationReport {
    pub original_length: usize,
    pub sanitized_length: usize,
    pub changes_made: bool,
    pub blocked_patterns: Vec<String>,
    pub safety_score: f64,
}

/// Calculates a safety score based on the sanitization results
fn calculate_safety_score(original: &str, sanitized: &str) -> f64 {
    let mut score = 100.0;
    
    // Penalize for blocked patterns
    let blocked_count = sanitized.matches("[BLOCKED:").count();
    score -= blocked_count as f64 * 10.0;
    
    // Penalize for length reduction (might indicate removed content)
    if sanitized.len() < original.len() {
        let reduction_ratio = 1.0 - (sanitized.len() as f64 / original.len() as f64);
        score -= reduction_ratio * 20.0;
    }
    
    // Penalize for control characters in original
    let control_chars = original.chars().filter(|c| c.is_control()).count();
    score -= control_chars as f64 * 5.0;
    
    // Penalize for null bytes
    let null_bytes = original.matches('\0').count();
    score -= null_bytes as f64 * 15.0;
    
    score.max(0.0).min(100.0)
}