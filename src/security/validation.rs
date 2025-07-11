use std::collections::HashMap;
use std::path::Path;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use super::SecurityConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputType {
    Query,
    Path,
    Email,
    Username,
    Password,
    FileName,
    Url,
    Json,
    SqlParameter,
    HtmlContent,
    XmlContent,
    CsvData,
    CommandLine,
    RegexPattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub name: String,
    pub pattern: String,
    pub max_length: Option<usize>,
    pub min_length: Option<usize>,
    pub required: bool,
    pub allow_empty: bool,
    pub custom_validator: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub sanitized_value: Option<String>,
}

pub struct InputValidator {
    rules: HashMap<InputType, ValidationRule>,
    patterns: HashMap<String, Regex>,
    security_config: SecurityConfig,
}

impl InputValidator {
    pub fn new(config: &SecurityConfig) -> Result<Self> {
        let mut validator = Self {
            rules: HashMap::new(),
            patterns: HashMap::new(),
            security_config: config.clone(),
        };

        validator.initialize_default_rules()?;
        validator.compile_patterns()?;

        Ok(validator)
    }

    pub fn validate(&self, input: &str, input_type: InputType) -> Result<ValidationResult> {
        let rule = self.rules.get(&input_type).ok_or_else(|| {
            crate::error::AppError::ValidationError(format!("No validation rule found for input type: {:?}", input_type))
        })?;

        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            sanitized_value: None,
        };

        // Check if input is required
        if rule.required && input.is_empty() {
            result.is_valid = false;
            result.errors.push(format!("Input is required for type: {:?}", input_type));
            return Ok(result);
        }

        // Check if empty input is allowed
        if input.is_empty() && !rule.allow_empty {
            result.is_valid = false;
            result.errors.push("Empty input not allowed".to_string());
            return Ok(result);
        }

        // Skip further validation if input is empty and allowed
        if input.is_empty() && rule.allow_empty {
            return Ok(result);
        }

        // Check length constraints
        if let Some(max_length) = rule.max_length {
            if input.len() > max_length {
                result.is_valid = false;
                result.errors.push(format!("Input too long. Maximum length: {}", max_length));
            }
        }

        if let Some(min_length) = rule.min_length {
            if input.len() < min_length {
                result.is_valid = false;
                result.errors.push(format!("Input too short. Minimum length: {}", min_length));
            }
        }

        // Check pattern matching
        if let Some(pattern) = self.patterns.get(&rule.pattern) {
            if !pattern.is_match(input) {
                result.is_valid = false;
                result.errors.push(format!("Input does not match required pattern for type: {:?}", input_type));
            }
        }

        // Perform type-specific validation
        self.validate_specific_type(input, &input_type, &mut result)?;

        // Sanitize input if valid
        if result.is_valid {
            result.sanitized_value = Some(self.sanitize_input(input, &input_type));
        }

        Ok(result)
    }

    pub fn validate_multiple(&self, inputs: &[(String, InputType)]) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();
        
        for (input, input_type) in inputs {
            let result = self.validate(input, input_type.clone())?;
            results.push(result);
        }
        
        Ok(results)
    }

    pub fn is_safe_query(&self, query: &str) -> bool {
        // Check for SQL injection patterns
        let sql_injection_patterns = [
            r"(?i)\bunion\s+select\b",
            r"(?i)\bselect\s+.*\bfrom\b",
            r"(?i)\binsert\s+into\b",
            r"(?i)\bupdate\s+.*\bset\b",
            r"(?i)\bdelete\s+from\b",
            r"(?i)\bdrop\s+table\b",
            r"(?i)\balter\s+table\b",
            r"(?i)\bcreate\s+table\b",
            r"(?i)--",
            r"(?i)/\*.*\*/",
            r"(?i)\bexec\s*\(",
            r"(?i)\bexecute\s*\(",
        ];

        for pattern in &sql_injection_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                if regex.is_match(query) {
                    return false;
                }
            }
        }

        // Check for XSS patterns
        let xss_patterns = [
            r"(?i)<script[^>]*>",
            r"(?i)</script>",
            r"(?i)<iframe[^>]*>",
            r"(?i)<object[^>]*>",
            r"(?i)<embed[^>]*>",
            r"(?i)<form[^>]*>",
            r"(?i)javascript:",
            r"(?i)vbscript:",
            r"(?i)onload\s*=",
            r"(?i)onerror\s*=",
            r"(?i)onclick\s*=",
        ];

        for pattern in &xss_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                if regex.is_match(query) {
                    return false;
                }
            }
        }

        true
    }

    pub fn is_safe_path(&self, path: &str) -> bool {
        // Check for path traversal
        if path.contains("..") || path.contains("./") || path.contains("..\\") {
            return false;
        }

        // Check for null bytes
        if path.contains('\0') {
            return false;
        }

        // Check for control characters
        if path.chars().any(|c| c.is_control() && c != '\n' && c != '\r' && c != '\t') {
            return false;
        }

        // Check for reserved Windows filenames
        let reserved_names = [
            "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5",
            "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4",
            "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
        ];

        let path_upper = path.to_uppercase();
        for name in &reserved_names {
            if path_upper.contains(name) {
                return false;
            }
        }

        true
    }

    pub fn validate_file_content(&self, content: &[u8], file_type: &str) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            sanitized_value: None,
        };

        // Check file size
        if content.len() > self.security_config.max_request_size {
            result.is_valid = false;
            result.errors.push(format!("File too large. Maximum size: {} bytes", self.security_config.max_request_size));
            return Ok(result);
        }

        // Check for null bytes (potential binary content)
        if content.contains(&0) && !self.is_binary_type(file_type) {
            result.warnings.push("File contains null bytes, may be binary content".to_string());
        }

        // Check for potentially malicious content
        if self.contains_malicious_patterns(content) {
            result.is_valid = false;
            result.errors.push("File contains potentially malicious content".to_string());
        }

        // Validate specific file types
        match file_type.to_lowercase().as_str() {
            "json" => self.validate_json_content(content, &mut result)?,
            "xml" => self.validate_xml_content(content, &mut result)?,
            "csv" => self.validate_csv_content(content, &mut result)?,
            "html" => self.validate_html_content(content, &mut result)?,
            "js" | "javascript" => self.validate_javascript_content(content, &mut result)?,
            _ => {} // Default validation already performed
        }

        Ok(result)
    }

    fn initialize_default_rules(&mut self) -> Result<()> {
        // Query validation rule
        self.rules.insert(InputType::Query, ValidationRule {
            name: "query".to_string(),
            pattern: "query_pattern".to_string(),
            max_length: Some(self.security_config.max_query_length),
            min_length: Some(1),
            required: true,
            allow_empty: false,
            custom_validator: None,
        });

        // Path validation rule
        self.rules.insert(InputType::Path, ValidationRule {
            name: "path".to_string(),
            pattern: "path_pattern".to_string(),
            max_length: Some(self.security_config.max_path_length),
            min_length: Some(1),
            required: true,
            allow_empty: false,
            custom_validator: None,
        });

        // Email validation rule
        self.rules.insert(InputType::Email, ValidationRule {
            name: "email".to_string(),
            pattern: "email_pattern".to_string(),
            max_length: Some(254),
            min_length: Some(3),
            required: false,
            allow_empty: true,
            custom_validator: None,
        });

        // Username validation rule
        self.rules.insert(InputType::Username, ValidationRule {
            name: "username".to_string(),
            pattern: "username_pattern".to_string(),
            max_length: Some(50),
            min_length: Some(3),
            required: false,
            allow_empty: true,
            custom_validator: None,
        });

        // Password validation rule
        self.rules.insert(InputType::Password, ValidationRule {
            name: "password".to_string(),
            pattern: "password_pattern".to_string(),
            max_length: Some(128),
            min_length: Some(8),
            required: false,
            allow_empty: true,
            custom_validator: None,
        });

        // Filename validation rule
        self.rules.insert(InputType::FileName, ValidationRule {
            name: "filename".to_string(),
            pattern: "filename_pattern".to_string(),
            max_length: Some(255),
            min_length: Some(1),
            required: true,
            allow_empty: false,
            custom_validator: None,
        });

        // URL validation rule
        self.rules.insert(InputType::Url, ValidationRule {
            name: "url".to_string(),
            pattern: "url_pattern".to_string(),
            max_length: Some(2048),
            min_length: Some(3),
            required: false,
            allow_empty: true,
            custom_validator: None,
        });

        Ok(())
    }

    fn compile_patterns(&mut self) -> Result<()> {
        // Query pattern - allow most characters but block dangerous ones
        self.patterns.insert("query_pattern".to_string(), 
            Regex::new(r"^[a-zA-Z0-9\s\-_.,!?@#$%^&*()+=\[\]{}|\\:;\"'<>/~`]*$")?);

        // Path pattern - allow valid path characters
        self.patterns.insert("path_pattern".to_string(), 
            Regex::new(r"^[a-zA-Z0-9\s\-_./\\:]+$")?);

        // Email pattern
        self.patterns.insert("email_pattern".to_string(), 
            Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")?);

        // Username pattern - alphanumeric, underscore, hyphen
        self.patterns.insert("username_pattern".to_string(), 
            Regex::new(r"^[a-zA-Z0-9_-]+$")?);

        // Password pattern - require at least one uppercase, lowercase, digit, and special character
        self.patterns.insert("password_pattern".to_string(), 
            Regex::new(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]+$")?);

        // Filename pattern - no path separators or special characters
        self.patterns.insert("filename_pattern".to_string(), 
            Regex::new(r"^[a-zA-Z0-9\s\-_.,()]+$")?);

        // URL pattern
        self.patterns.insert("url_pattern".to_string(), 
            Regex::new(r"^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})?(?:/[a-zA-Z0-9._~:/?#[\]@!$&'()*+,;=%-]*)?$")?);

        Ok(())
    }

    fn validate_specific_type(&self, input: &str, input_type: &InputType, result: &mut ValidationResult) -> Result<()> {
        match input_type {
            InputType::Query => {
                if !self.is_safe_query(input) {
                    result.is_valid = false;
                    result.errors.push("Query contains potentially dangerous content".to_string());
                }
            }
            InputType::Path => {
                if !self.is_safe_path(input) {
                    result.is_valid = false;
                    result.errors.push("Path contains invalid or dangerous characters".to_string());
                }
            }
            InputType::Email => {
                // Additional email validation beyond regex
                if input.len() > 254 {
                    result.is_valid = false;
                    result.errors.push("Email address too long".to_string());
                }
                if input.chars().filter(|&c| c == '@').count() != 1 {
                    result.is_valid = false;
                    result.errors.push("Email must contain exactly one @ symbol".to_string());
                }
            }
            InputType::Password => {
                self.validate_password_strength(input, result);
            }
            InputType::FileName => {
                // Check for reserved names and characters
                if input.starts_with('.') {
                    result.warnings.push("Filename starts with dot (hidden file)".to_string());
                }
                if input.contains("..") {
                    result.is_valid = false;
                    result.errors.push("Filename contains invalid sequence".to_string());
                }
            }
            InputType::Url => {
                // Additional URL validation
                if !input.starts_with("http://") && !input.starts_with("https://") {
                    result.is_valid = false;
                    result.errors.push("URL must start with http:// or https://".to_string());
                }
            }
            InputType::Json => {
                // Validate JSON syntax
                if serde_json::from_str::<serde_json::Value>(input).is_err() {
                    result.is_valid = false;
                    result.errors.push("Invalid JSON format".to_string());
                }
            }
            _ => {} // Default validation already performed
        }

        Ok(())
    }

    fn validate_password_strength(&self, password: &str, result: &mut ValidationResult) {
        let mut score = 0;
        let mut recommendations = Vec::new();

        // Length check
        if password.len() >= 12 {
            score += 2;
        } else if password.len() >= 8 {
            score += 1;
        } else {
            recommendations.push("Use at least 8 characters".to_string());
        }

        // Character diversity
        if password.chars().any(|c| c.is_lowercase()) {
            score += 1;
        } else {
            recommendations.push("Include lowercase letters".to_string());
        }

        if password.chars().any(|c| c.is_uppercase()) {
            score += 1;
        } else {
            recommendations.push("Include uppercase letters".to_string());
        }

        if password.chars().any(|c| c.is_numeric()) {
            score += 1;
        } else {
            recommendations.push("Include numbers".to_string());
        }

        if password.chars().any(|c| "!@#$%^&*()_+-=[]{}|;:,.<>?".contains(c)) {
            score += 1;
        } else {
            recommendations.push("Include special characters".to_string());
        }

        // Common patterns
        if password.to_lowercase().contains("password") {
            score -= 2;
            recommendations.push("Avoid using 'password' in password".to_string());
        }

        if password.chars().collect::<Vec<_>>().windows(3).any(|w| w[0] == w[1] && w[1] == w[2]) {
            score -= 1;
            recommendations.push("Avoid repeating characters".to_string());
        }

        // Score interpretation
        if score < 4 {
            result.warnings.push("Password is weak".to_string());
        } else if score < 6 {
            result.warnings.push("Password is moderate".to_string());
        }

        for rec in recommendations {
            result.warnings.push(rec);
        }
    }

    fn sanitize_input(&self, input: &str, input_type: &InputType) -> String {
        match input_type {
            InputType::Query => {
                // Remove potentially dangerous characters
                input.chars()
                    .filter(|c| !c.is_control() || *c == '\n' || *c == '\r' || *c == '\t')
                    .collect()
            }
            InputType::HtmlContent => {
                // Basic HTML sanitization
                input
                    .replace("<script", "&lt;script")
                    .replace("</script>", "&lt;/script&gt;")
                    .replace("javascript:", "")
                    .replace("vbscript:", "")
                    .replace("onload=", "")
                    .replace("onerror=", "")
                    .replace("onclick=", "")
            }
            InputType::SqlParameter => {
                // SQL parameter sanitization
                input.replace("'", "''").replace(";", "")
            }
            _ => input.to_string()
        }
    }

    fn is_binary_type(&self, file_type: &str) -> bool {
        matches!(file_type.to_lowercase().as_str(), 
            "exe" | "dll" | "bin" | "so" | "dylib" | "png" | "jpg" | "jpeg" | "gif" | "bmp" | "ico" | "zip" | "rar" | "tar" | "gz")
    }

    fn contains_malicious_patterns(&self, content: &[u8]) -> bool {
        let content_str = String::from_utf8_lossy(content);
        
        // Check for executable signatures
        if content.len() >= 2 && content[0] == 0x4D && content[1] == 0x5A {
            return true; // PE/DOS header
        }

        if content.len() >= 4 && &content[0..4] == b"\x7fELF" {
            return true; // ELF header
        }

        // Check for script patterns
        let malicious_patterns = [
            "eval(", "exec(", "system(", "shell_exec(", "passthru(",
            "proc_open(", "popen(", "file_get_contents(", "file_put_contents(",
            "base64_decode(", "gzinflate(", "str_rot13(",
            "javascript:", "vbscript:", "data:text/html",
            "<?php", "<%", "<script", "<iframe", "<object", "<embed",
        ];

        for pattern in &malicious_patterns {
            if content_str.to_lowercase().contains(pattern) {
                return true;
            }
        }

        false
    }

    fn validate_json_content(&self, content: &[u8], result: &mut ValidationResult) -> Result<()> {
        let content_str = String::from_utf8_lossy(content);
        
        match serde_json::from_str::<serde_json::Value>(&content_str) {
            Ok(_) => {
                // Check for potentially dangerous JSON content
                if content_str.contains("__proto__") || content_str.contains("constructor") {
                    result.warnings.push("JSON contains prototype pollution patterns".to_string());
                }
            }
            Err(_) => {
                result.is_valid = false;
                result.errors.push("Invalid JSON format".to_string());
            }
        }

        Ok(())
    }

    fn validate_xml_content(&self, content: &[u8], result: &mut ValidationResult) -> Result<()> {
        let content_str = String::from_utf8_lossy(content);
        
        // Check for XML external entity attacks
        if content_str.contains("<!ENTITY") || content_str.contains("<!DOCTYPE") {
            result.warnings.push("XML contains entity definitions (potential XXE attack)".to_string());
        }

        // Check for XML bombs
        if content_str.matches("<!ENTITY").count() > 10 {
            result.is_valid = false;
            result.errors.push("XML contains too many entity definitions".to_string());
        }

        Ok(())
    }

    fn validate_csv_content(&self, content: &[u8], result: &mut ValidationResult) -> Result<()> {
        let content_str = String::from_utf8_lossy(content);
        
        // Check for CSV injection patterns
        let dangerous_prefixes = ["=", "+", "-", "@", "\t", "\r"];
        
        for line in content_str.lines() {
            for field in line.split(',') {
                let trimmed = field.trim();
                if dangerous_prefixes.iter().any(|prefix| trimmed.starts_with(prefix)) {
                    result.warnings.push("CSV contains potentially dangerous formula".to_string());
                    break;
                }
            }
        }

        Ok(())
    }

    fn validate_html_content(&self, content: &[u8], result: &mut ValidationResult) -> Result<()> {
        let content_str = String::from_utf8_lossy(content);
        
        // Check for dangerous HTML elements
        let dangerous_elements = [
            "<script", "<iframe", "<object", "<embed", "<form", "<meta",
            "<link", "<style", "<base", "<applet", "<frameset", "<frame"
        ];

        for element in &dangerous_elements {
            if content_str.to_lowercase().contains(element) {
                result.warnings.push(format!("HTML contains potentially dangerous element: {}", element));
            }
        }

        // Check for inline event handlers
        let event_handlers = [
            "onload=", "onerror=", "onclick=", "onmouseover=", "onkeydown=",
            "onsubmit=", "onchange=", "onfocus=", "onblur="
        ];

        for handler in &event_handlers {
            if content_str.to_lowercase().contains(handler) {
                result.warnings.push(format!("HTML contains inline event handler: {}", handler));
            }
        }

        Ok(())
    }

    fn validate_javascript_content(&self, content: &[u8], result: &mut ValidationResult) -> Result<()> {
        let content_str = String::from_utf8_lossy(content);
        
        // Check for dangerous JavaScript patterns
        let dangerous_patterns = [
            "eval(", "Function(", "setTimeout(", "setInterval(",
            "document.write(", "innerHTML", "outerHTML", "document.cookie",
            "location.href", "window.open(", "XMLHttpRequest", "fetch("
        ];

        for pattern in &dangerous_patterns {
            if content_str.contains(pattern) {
                result.warnings.push(format!("JavaScript contains potentially dangerous pattern: {}", pattern));
            }
        }

        Ok(())
    }
}