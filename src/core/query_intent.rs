use crate::error::{AppError, SearchError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use regex::Regex;

/// Intent labels for multi-label classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Intent {
    /// Find specific documents or files
    DocumentSearch,
    /// Search within document content
    ContentSearch,
    /// Time-based queries (recent, last week, etc.)
    TemporalSearch,
    /// Person-related searches
    PersonSearch,
    /// File type specific searches
    TypeSearch,
    /// Location-based searches
    LocationSearch,
    /// Size/quantity based searches
    SizeSearch,
    /// Project or tag-based searches
    ProjectSearch,
    /// Similar document requests
    SimilaritySearch,
    /// Question answering queries
    QuestionAnswering,
}

/// Named entity types detected in queries
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Date,
    Time,
    Location,
    Organization,
    FileType,
    Size,
    Project,
    Tag,
}

/// A detected named entity in the query
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub entity_type: EntityType,
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
    pub normalized_value: Option<String>,
}

/// Temporal expression with parsed date/time information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalExpression {
    pub original_text: String,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub relative_days: Option<i32>,
    pub confidence: f32,
}

/// Query classification and analysis results
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryIntent {
    /// Original user query
    pub original_query: String,
    /// Normalized and cleaned query text
    pub normalized_text: String,
    /// Detected intent labels with confidence scores
    pub intents: HashMap<Intent, f32>,
    /// Named entities found in the query
    pub entities: Vec<Entity>,
    /// Temporal expressions (dates, times, relative periods)
    pub temporal_expressions: Vec<TemporalExpression>,
    /// Query complexity score (0.0 to 1.0)
    pub complexity_score: f32,
    /// Suggested search strategy
    pub search_strategy: SearchStrategy,
    /// Processing timestamp
    pub processed_at: DateTime<Utc>,
}

/// Recommended search strategy based on intent analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SearchStrategy {
    /// Use full-text search only
    FullTextOnly,
    /// Use vector search only
    VectorOnly,
    /// Use hybrid search (FTS + vector)
    Hybrid {
        fts_weight: f32,
        vector_weight: f32,
    },
    /// Use multimodal search (text + images)
    Multimodal {
        text_weight: f32,
        image_weight: f32,
    },
    /// Complex multi-stage search
    MultiStage {
        stages: Vec<SearchStage>,
    },
}

/// Individual search stage for complex queries
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchStage {
    pub stage_type: SearchStageType,
    pub query: String,
    pub filters: Vec<SearchFilter>,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SearchStageType {
    FullText,
    Vector,
    Temporal,
    Entity,
    Similarity,
}

/// Search filters derived from query analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SearchFilter {
    DateRange {
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    },
    FileType {
        types: Vec<String>,
    },
    Person {
        names: Vec<String>,
    },
    Size {
        min_size: Option<u64>,
        max_size: Option<u64>,
    },
    Project {
        project_ids: Vec<String>,
    },
    Tags {
        tags: Vec<String>,
    },
}

/// Configuration for query intent classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryClassifierConfig {
    /// Enable intent classification
    pub enable_classification: bool,
    /// Enable named entity recognition
    pub enable_ner: bool,
    /// Enable temporal expression parsing
    pub enable_temporal: bool,
    /// Minimum confidence threshold for intents
    pub intent_threshold: f32,
    /// Minimum confidence threshold for entities
    pub entity_threshold: f32,
    /// Maximum query length to process
    pub max_query_length: usize,
    /// Enable query normalization
    pub enable_normalization: bool,
}

impl Default for QueryClassifierConfig {
    fn default() -> Self {
        Self {
            enable_classification: true,
            enable_ner: true,
            enable_temporal: true,
            intent_threshold: 0.5,
            entity_threshold: 0.6,
            max_query_length: 1000,
            enable_normalization: true,
        }
    }
}

/// Main query intent classifier
pub struct QueryIntentClassifier {
    config: QueryClassifierConfig,
    // Regex patterns for basic entity recognition
    person_patterns: Vec<Regex>,
    date_patterns: Vec<Regex>,
    file_type_patterns: Vec<Regex>,
    size_patterns: Vec<Regex>,
    temporal_patterns: Vec<Regex>,
    // Intent keywords for classification
    intent_keywords: HashMap<Intent, Vec<String>>,
}

impl QueryIntentClassifier {
    pub fn new(config: QueryClassifierConfig) -> Result<Self> {
        let mut classifier = Self {
            config,
            person_patterns: Vec::new(),
            date_patterns: Vec::new(),
            file_type_patterns: Vec::new(),
            size_patterns: Vec::new(),
            temporal_patterns: Vec::new(),
            intent_keywords: HashMap::new(),
        };

        classifier.initialize_patterns()?;
        classifier.initialize_intent_keywords();
        
        Ok(classifier)
    }

    /// Create classifier with default configuration
    pub fn default() -> Result<Self> {
        Self::new(QueryClassifierConfig::default())
    }

    /// Analyze query and return intent classification
    pub async fn analyze_query(&self, query: &str) -> Result<QueryIntent> {
        if query.len() > self.config.max_query_length {
            return Err(AppError::Search(SearchError::InvalidQuery(
                "Query too long".to_string()
            )));
        }

        let normalized_text = if self.config.enable_normalization {
            self.normalize_query(query)
        } else {
            query.to_string()
        };

        let mut intent = QueryIntent {
            original_query: query.to_string(),
            normalized_text: normalized_text.clone(),
            intents: HashMap::new(),
            entities: Vec::new(),
            temporal_expressions: Vec::new(),
            complexity_score: 0.0,
            search_strategy: SearchStrategy::FullTextOnly,
            processed_at: Utc::now(),
        };

        // Classify intents
        if self.config.enable_classification {
            intent.intents = self.classify_intents(&normalized_text);
        }

        // Extract entities (use original query to preserve capitalization)
        if self.config.enable_ner {
            intent.entities = self.extract_entities(query);
        }

        // Parse temporal expressions
        if self.config.enable_temporal {
            intent.temporal_expressions = self.parse_temporal_expressions(&normalized_text);
        }

        // Calculate complexity score
        intent.complexity_score = self.calculate_complexity(&intent);

        // Determine search strategy
        intent.search_strategy = self.determine_search_strategy(&intent);

        Ok(intent)
    }

    /// Initialize regex patterns for entity recognition
    fn initialize_patterns(&mut self) -> Result<()> {
        // Enhanced person patterns
        self.person_patterns = vec![
            // "from/by" + name patterns - match proper names after keywords
            Regex::new(r"\b(?:from|by|sent\s+by|created\s+by|authored\s+by|written\s+by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)")?,
            // "Dr./Mr./Ms." + name patterns - match titles with names
            Regex::new(r"\b(Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)")?,
            Regex::new(r"\b(Mr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)")?,
            Regex::new(r"\b(Ms\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)")?,
            Regex::new(r"\b(Mrs\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)")?,
            // First Last name patterns (standalone) - match two capitalized words
            Regex::new(r"\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b")?,
            // Name + action patterns - match names followed by actions
            Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:sent|created|wrote|shared)")?,
        ];

        // Date patterns (unchanged - working well)
        self.date_patterns = vec![
            Regex::new(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")?,
            Regex::new(r"\b(\d{4}-\d{2}-\d{2})\b")?,
            Regex::new(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b")?,
        ];

        // Enhanced file type patterns
        self.file_type_patterns = vec![
            // Standard extensions + "files"
            Regex::new(r"\b(pdf|doc|docx|txt|ppt|pptx|xls|xlsx|jpg|jpeg|png|gif|mp4|mp3|zip|rar)\s+files?\b")?,
            // Application names (case-insensitive)
            Regex::new(r"(?i)\b(excel|powerpoint|word|outlook|photoshop)\s+(?:files?|documents?)")?,
            // Application names standalone
            Regex::new(r"(?i)\b(excel|powerpoint|word)\s+(?:spreadsheets?|presentations?|documents?)")?,
            // Document types
            Regex::new(r"\b(spreadsheets?|presentations?|documents?|images?|videos?|emails?)\b")?,
            // Extension with dot
            Regex::new(r"\.([a-zA-Z0-9]{2,5})\s+files?\b")?,
            // Uppercase file types
            Regex::new(r"\b(PDF|DOC|DOCX|XLS|XLSX|PPT|PPTX)\s+files?\b")?,
        ];

        // Enhanced size patterns
        self.size_patterns = vec![
            // Descriptive sizes
            Regex::new(r"\b(small|large|big|huge|tiny|massive)\s+(?:files?|documents?)")?,
            // Comparative sizes (before and after file mention)
            Regex::new(r"\b(?:files?|documents?)\s+(larger|smaller|bigger)\s+than\s+(\d+)\s*(kb|mb|gb|bytes?)\b")?,
            Regex::new(r"\b(larger|smaller|bigger)\s+than\s+(\d+)\s*(kb|mb|gb|bytes?)")?,
            // Specific sizes
            Regex::new(r"\b(\d+)\s*(kb|mb|gb|tb|bytes?)\b")?,
            // Size ranges
            Regex::new(r"\b(?:over|above|more\s+than)\s+(\d+)\s*(kb|mb|gb|tb)\b")?,
            Regex::new(r"\b(?:under|below|less\s+than)\s+(\d+)\s*(kb|mb|gb|tb)\b")?,
        ];

        // Enhanced temporal patterns with improved detection
        self.temporal_patterns = vec![
            Regex::new(r"\b(today|yesterday|tomorrow)\b")?,
            Regex::new(r"\b(last|past|previous)\s+(week|month|year|day)\b")?,
            Regex::new(r"\b(this|current)\s+(week|month|year|day)\b")?,
            Regex::new(r"\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b")?,
            Regex::new(r"\brecent(ly)?\b")?,
            Regex::new(r"\b(last\s+month|past\s+month|previous\s+month)\b")?,
            Regex::new(r"\b(last\s+week|past\s+week|previous\s+week)\b")?,
        ];

        Ok(())
    }

    /// Initialize intent classification keywords
    fn initialize_intent_keywords(&mut self) {
        self.intent_keywords.insert(Intent::DocumentSearch, vec![
            "find".to_string(), "search".to_string(), "locate".to_string(), 
            "document".to_string(), "file".to_string(), "report".to_string(),
            "show".to_string(), "get".to_string(), "retrieve".to_string(),
            "fetch".to_string(), "pull".to_string(), "grab".to_string(),
        ]);

        self.intent_keywords.insert(Intent::ContentSearch, vec![
            "contains".to_string(), "mentions".to_string(), "about".to_string(),
            "content".to_string(), "text".to_string(), "says".to_string(),
            "discusses".to_string(), "covers".to_string(), "includes".to_string(),
            "talks about".to_string(), "refers to".to_string(),
        ]);

        self.intent_keywords.insert(Intent::TemporalSearch, vec![
            "recent".to_string(), "yesterday".to_string(), "today".to_string(),
            "last week".to_string(), "ago".to_string(), "when".to_string(),
            "lately".to_string(), "recently".to_string(), "past".to_string(),
            "previous".to_string(), "earlier".to_string(),
        ]);

        self.intent_keywords.insert(Intent::PersonSearch, vec![
            "from".to_string(), "by".to_string(), "sent by".to_string(),
            "created by".to_string(), "author".to_string(), "who".to_string(),
            "written by".to_string(), "authored by".to_string(), "made by".to_string(),
            "shared by".to_string(), "uploaded by".to_string(), "team".to_string(),
            "person".to_string(), "people".to_string(), "dr.".to_string(),
            "mr.".to_string(), "ms.".to_string(), "mrs.".to_string(),
        ]);

        self.intent_keywords.insert(Intent::TypeSearch, vec![
            "pdf".to_string(), "image".to_string(), "document".to_string(),
            "presentation".to_string(), "spreadsheet".to_string(), "video".to_string(),
            "excel".to_string(), "powerpoint".to_string(), "word".to_string(),
            "files".to_string(), "emails".to_string(), "photos".to_string(),
        ]);

        self.intent_keywords.insert(Intent::SimilaritySearch, vec![
            "similar".to_string(), "like".to_string(), "related".to_string(),
            "comparable".to_string(), "resembles".to_string(), "matches".to_string(),
            "same as".to_string(), "equivalent".to_string(), "analogous".to_string(),
            "alike".to_string(), "corresponding".to_string(),
        ]);

        self.intent_keywords.insert(Intent::QuestionAnswering, vec![
            "what is".to_string(), "what are".to_string(), "how to".to_string(),
            "how does".to_string(), "why does".to_string(), "why is".to_string(),
            "where is".to_string(), "when did".to_string(), "who is".to_string(),
            "what".to_string(), "how".to_string(), "why".to_string(),
            "where".to_string(), "when".to_string(), "who".to_string(),
            "explain".to_string(), "definition".to_string(), "meaning".to_string(),
        ]);

        self.intent_keywords.insert(Intent::SizeSearch, vec![
            "large".to_string(), "small".to_string(), "big".to_string(),
            "huge".to_string(), "tiny".to_string(), "massive".to_string(),
            "size".to_string(), "mb".to_string(), "gb".to_string(),
            "bytes".to_string(), "larger than".to_string(), "smaller than".to_string(),
        ]);
    }

    /// Normalize query text
    fn normalize_query(&self, query: &str) -> String {
        // Basic normalization - can be enhanced
        query.trim()
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,!?-_()[]{}\"'".contains(*c))
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
    }

    /// Classify query intents based on keywords and patterns with improved multi-intent support
    fn classify_intents(&self, query: &str) -> HashMap<Intent, f32> {
        let mut intents = HashMap::new();
        let query_lower = query.to_lowercase();

        for (intent, keywords) in &self.intent_keywords {
            let mut score = 0.0;
            let mut matches = 0;
            let mut keyword_coverage = 0.0;

            for keyword in keywords {
                if query_lower.contains(keyword) {
                    matches += 1;
                    // Weight longer keywords more heavily
                    let keyword_weight = (keyword.len() as f32 / query.len() as f32) * 
                                       (keyword.split_whitespace().count() as f32); // Multi-word bonus
                    score += keyword_weight;
                    keyword_coverage += keyword.len() as f32;
                }
            }

            if matches > 0 {
                // Enhanced scoring with keyword coverage and match count
                let base_confidence = score * (matches as f32).sqrt(); // Square root to prevent over-weighting
                let coverage_bonus = (keyword_coverage / query.len() as f32).min(0.3); // Max 30% bonus
                let confidence = (base_confidence + coverage_bonus).min(1.0);
                
                // Lower threshold for multi-intent detection
                let threshold = match intent {
                    Intent::QuestionAnswering | Intent::SimilaritySearch => 0.2, // More sensitive
                    _ => self.config.intent_threshold,
                };
                
                if confidence >= threshold {
                    intents.insert(intent.clone(), confidence);
                }
            }
        }

        // Special handling for question patterns
        self.detect_question_patterns(&query_lower, &mut intents);
        
        // Special handling for similarity patterns
        self.detect_similarity_patterns(&query_lower, &mut intents);

        // If no intents detected, try to infer from context
        if intents.is_empty() {
            if query_lower.contains("file") || query_lower.contains("document") {
                intents.insert(Intent::DocumentSearch, 0.6);
            } else {
                intents.insert(Intent::ContentSearch, 0.5);
            }
        }

        intents
    }

    /// Detect question patterns specifically
    fn detect_question_patterns(&self, query_lower: &str, intents: &mut HashMap<Intent, f32>) {
        let question_patterns = [
            r"\bwhat\s+is\b", r"\bwhat\s+are\b", r"\bwhat\s+does\b",
            r"\bhow\s+to\b", r"\bhow\s+does\b", r"\bhow\s+can\b",
            r"\bwhy\s+is\b", r"\bwhy\s+does\b", r"\bwhy\s+do\b",
            r"\bwhere\s+is\b", r"\bwhere\s+can\b", r"\bwhere\s+do\b",
            r"\bwhen\s+did\b", r"\bwhen\s+will\b", r"\bwhen\s+is\b",
            r"\bwho\s+is\b", r"\bwho\s+can\b", r"\bwho\s+did\b",
            r"^(?:what|how|why|where|when|who)\b", // Question word at start
            r"\?$", // Ends with question mark
        ];

        for pattern in &question_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                if regex.is_match(query_lower) {
                    let current_score = intents.get(&Intent::QuestionAnswering).unwrap_or(&0.0);
                    intents.insert(Intent::QuestionAnswering, (current_score + 0.3).min(1.0));
                    break;
                }
            }
        }
    }

    /// Detect similarity patterns specifically  
    fn detect_similarity_patterns(&self, query_lower: &str, intents: &mut HashMap<Intent, f32>) {
        let similarity_patterns = [
            r"\bsimilar\s+to\b", r"\blike\s+this\b", r"\brelated\s+to\b",
            r"\bsame\s+as\b", r"\bresembl", r"\bmatch", r"\bcompara",
            r"\balike\b", r"\bequivalent\b",
        ];

        for pattern in &similarity_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                if regex.is_match(query_lower) {
                    let current_score = intents.get(&Intent::SimilaritySearch).unwrap_or(&0.0);
                    intents.insert(Intent::SimilaritySearch, (current_score + 0.4).min(1.0));
                    break;
                }
            }
        }
    }

    /// Extract named entities from query with enhanced detection
    fn extract_entities(&self, query: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Extract persons with improved confidence scoring
        for pattern in &self.person_patterns {
            for capture in pattern.captures_iter(query) {
                if let Some(matched) = capture.get(1) {
                    // Calculate confidence based on pattern type and context
                    let confidence = if pattern.as_str().contains("Dr\\.|Mr\\.|Ms\\.|Mrs\\.") {
                        0.95 // High confidence for titles
                    } else if pattern.as_str().contains("(?:from|by|sent\\s+by)") {
                        0.85 // High confidence for explicit attribution
                    } else if matched.as_str().len() > 10 {
                        0.75 // Medium-high for longer names
                    } else {
                        0.6  // Lower for shorter potential names
                    };

                    entities.push(Entity {
                        entity_type: EntityType::Person,
                        text: matched.as_str().to_string(),
                        start: matched.start(),
                        end: matched.end(),
                        confidence,
                        normalized_value: Some(matched.as_str().to_title_case()),
                    });
                }
            }
        }

        // Extract file types with improved detection
        for pattern in &self.file_type_patterns {
            for capture in pattern.captures_iter(query) {
                if let Some(matched) = capture.get(1) {
                    let confidence = if ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"].contains(&matched.as_str().to_lowercase().as_str()) {
                        0.95 // High confidence for common office formats
                    } else if ["excel", "powerpoint", "word"].contains(&matched.as_str().to_lowercase().as_str()) {
                        0.9  // High confidence for application names
                    } else if ["spreadsheets", "presentations", "documents"].contains(&matched.as_str().to_lowercase().as_str()) {
                        0.8  // Good confidence for document categories
                    } else {
                        0.7  // Medium confidence for other types
                    };

                    entities.push(Entity {
                        entity_type: EntityType::FileType,
                        text: matched.as_str().to_string(),
                        start: matched.start(),
                        end: matched.end(),
                        confidence,
                        normalized_value: Some(self.normalize_file_type(matched.as_str())),
                    });
                }
            }
        }

        // Extract sizes with confidence scoring
        for pattern in &self.size_patterns {
            for capture in pattern.captures_iter(query) {
                if let Some(matched) = capture.get(1) {
                    let confidence = if matched.as_str().chars().any(|c| c.is_numeric()) {
                        0.9 // High confidence for numeric sizes
                    } else {
                        0.7 // Medium confidence for descriptive sizes
                    };

                    entities.push(Entity {
                        entity_type: EntityType::Size,
                        text: matched.as_str().to_string(),
                        start: matched.start(),
                        end: matched.end(),
                        confidence,
                        normalized_value: Some(matched.as_str().to_lowercase()),
                    });
                }
            }
        }

        // Remove duplicates and apply confidence threshold
        entities.sort_by(|a, b| a.start.cmp(&b.start));
        entities.dedup_by(|a, b| a.start == b.start && a.end == b.end);
        entities.retain(|e| e.confidence >= self.config.entity_threshold);
        
        entities
    }

    /// Normalize file type names for consistency
    fn normalize_file_type(&self, file_type: &str) -> String {
        match file_type.to_lowercase().as_str() {
            "excel" | "spreadsheets" | "spreadsheet" => "xlsx".to_string(),
            "powerpoint" | "presentations" | "presentation" => "pptx".to_string(),
            "word" | "documents" | "document" => "docx".to_string(),
            "photos" | "images" | "image" => "jpg".to_string(),
            "videos" | "video" => "mp4".to_string(),
            "emails" | "email" => "msg".to_string(),
            other => other.to_lowercase(),
        }
    }

    /// Parse temporal expressions
    fn parse_temporal_expressions(&self, query: &str) -> Vec<TemporalExpression> {
        let mut expressions = Vec::new();
        let now = Utc::now();

        for pattern in &self.temporal_patterns {
            for capture in pattern.captures_iter(query) {
                if let Some(matched) = capture.get(0) {
                    let text = matched.as_str().to_lowercase();
                    let mut expr = TemporalExpression {
                        original_text: matched.as_str().to_string(),
                        start_date: None,
                        end_date: None,
                        relative_days: None,
                        confidence: 0.7,
                    };

                    // Enhanced temporal parsing
                    if text.contains("today") {
                        expr.start_date = Some(now.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc());
                        expr.end_date = Some(now.date_naive().and_hms_opt(23, 59, 59).unwrap().and_utc());
                    } else if text.contains("yesterday") {
                        let yesterday = now.date_naive() - chrono::Duration::days(1);
                        expr.start_date = Some(yesterday.and_hms_opt(0, 0, 0).unwrap().and_utc());
                        expr.end_date = Some(yesterday.and_hms_opt(23, 59, 59).unwrap().and_utc());
                    } else if text.contains("last week") || text.contains("past week") {
                        expr.relative_days = Some(-7);
                        expr.start_date = Some(now - chrono::Duration::days(7));
                    } else if text.contains("last month") || text.contains("past month") {
                        expr.relative_days = Some(-30);
                        expr.start_date = Some(now - chrono::Duration::days(30));
                    } else if text.contains("recent") {
                        expr.relative_days = Some(-7);
                        expr.start_date = Some(now - chrono::Duration::days(7));
                    }

                    expressions.push(expr);
                }
            }
        }

        expressions
    }

    /// Calculate query complexity score
    fn calculate_complexity(&self, intent: &QueryIntent) -> f32 {
        let mut complexity = 0.0;

        // Base complexity from query length
        complexity += (intent.original_query.len() as f32 / 100.0).min(0.3);

        // Add complexity for multiple intents
        complexity += (intent.intents.len() as f32 * 0.1).min(0.3);

        // Add complexity for entities
        complexity += (intent.entities.len() as f32 * 0.05).min(0.2);

        // Add complexity for temporal expressions
        complexity += (intent.temporal_expressions.len() as f32 * 0.1).min(0.2);

        complexity.min(1.0)
    }

    /// Determine optimal search strategy based on sophisticated intent analysis
    fn determine_search_strategy(&self, intent: &QueryIntent) -> SearchStrategy {
        let has_similarity = intent.intents.contains_key(&Intent::SimilaritySearch);
        let has_qa = intent.intents.contains_key(&Intent::QuestionAnswering);
        let has_temporal = !intent.temporal_expressions.is_empty();
        let has_entities = !intent.entities.is_empty();
        let has_multiple_intents = intent.intents.len() > 1;
        let intent_count = intent.intents.len();

        // Check for specific high-confidence intents
        let similarity_confidence = intent.intents.get(&Intent::SimilaritySearch).unwrap_or(&0.0);
        let qa_confidence = intent.intents.get(&Intent::QuestionAnswering).unwrap_or(&0.0);

        // High-confidence similarity or QA queries -> Vector search
        if (*similarity_confidence > 0.6) || (*qa_confidence > 0.5) {
            return SearchStrategy::VectorOnly;
        }

        // Complex queries with multiple intents -> Multi-stage
        if intent.complexity_score > 0.8 || (has_multiple_intents && intent_count >= 3) {
            let mut stages = Vec::new();
            
            // Add appropriate stages based on intents
            if intent.intents.contains_key(&Intent::DocumentSearch) {
                stages.push(SearchStage {
                    stage_type: SearchStageType::FullText,
                    query: intent.normalized_text.clone(),
                    filters: self.extract_filters_from_intent(intent),
                    weight: 0.4,
                });
            }
            
            if has_similarity || has_qa {
                stages.push(SearchStage {
                    stage_type: SearchStageType::Vector,
                    query: intent.normalized_text.clone(),
                    filters: Vec::new(),
                    weight: 0.4,
                });
            }
            
            if has_temporal {
                stages.push(SearchStage {
                    stage_type: SearchStageType::Temporal,
                    query: intent.normalized_text.clone(),
                    filters: self.extract_filters_from_intent(intent),
                    weight: 0.2,
                });
            }

            return SearchStrategy::MultiStage { stages };
        }

        // Temporal or entity-rich queries -> Hybrid search
        if has_temporal || has_entities || has_multiple_intents {
            let fts_weight = if has_temporal || intent.intents.contains_key(&Intent::DocumentSearch) {
                0.7 // Favor FTS for temporal and document searches
            } else {
                0.5 // Balanced for entity searches
            };
            
            return SearchStrategy::Hybrid {
                fts_weight,
                vector_weight: 1.0 - fts_weight,
            };
        }

        // Simple content or document searches -> Full-text
        SearchStrategy::FullTextOnly
    }

    /// Extract search filters from query intent
    fn extract_filters_from_intent(&self, intent: &QueryIntent) -> Vec<SearchFilter> {
        let mut filters = Vec::new();

        // Add temporal filters
        if !intent.temporal_expressions.is_empty() {
            for expr in &intent.temporal_expressions {
                filters.push(SearchFilter::DateRange {
                    start: expr.start_date,
                    end: expr.end_date,
                });
            }
        }

        // Add file type filters
        let file_types: Vec<String> = intent.entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::FileType))
            .map(|e| e.normalized_value.as_ref().unwrap_or(&e.text).clone())
            .collect();
        
        if !file_types.is_empty() {
            filters.push(SearchFilter::FileType { types: file_types });
        }

        // Add person filters
        let people: Vec<String> = intent.entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::Person))
            .map(|e| e.normalized_value.as_ref().unwrap_or(&e.text).clone())
            .collect();
        
        if !people.is_empty() {
            filters.push(SearchFilter::Person { names: people });
        }

        // Add size filters (simplified for now)
        let has_size_entities = intent.entities
            .iter()
            .any(|e| matches!(e.entity_type, EntityType::Size));
        
        if has_size_entities {
            // Could be enhanced to parse actual size constraints
            filters.push(SearchFilter::Size {
                min_size: None,
                max_size: None,
            });
        }

        filters
    }
}

// Helper trait for string title case conversion
trait ToTitleCase {
    fn to_title_case(&self) -> String;
}

impl ToTitleCase for str {
    fn to_title_case(&self) -> String {
        self.split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
                }
            })
            .collect::<Vec<String>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_query_classification() {
        let classifier = QueryIntentClassifier::default().unwrap();
        
        let result = classifier.analyze_query("find documents by John from last week").await.unwrap();
        
        assert!(result.intents.contains_key(&Intent::DocumentSearch));
        assert!(result.intents.contains_key(&Intent::PersonSearch));
        assert!(result.intents.contains_key(&Intent::TemporalSearch));
        assert!(!result.entities.is_empty());
        assert!(!result.temporal_expressions.is_empty());
    }

    #[tokio::test]
    async fn test_similarity_search_detection() {
        let classifier = QueryIntentClassifier::default().unwrap();
        
        let result = classifier.analyze_query("find documents similar to this report").await.unwrap();
        
        assert!(result.intents.contains_key(&Intent::SimilaritySearch));
        matches!(result.search_strategy, SearchStrategy::VectorOnly);
    }

    #[tokio::test]
    async fn test_temporal_expression_parsing() {
        let classifier = QueryIntentClassifier::default().unwrap();
        
        let result = classifier.analyze_query("files from yesterday").await.unwrap();
        
        assert!(!result.temporal_expressions.is_empty());
        let expr = &result.temporal_expressions[0];
        assert!(expr.start_date.is_some());
        assert!(expr.end_date.is_some());
    }
}