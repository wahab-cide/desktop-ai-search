use crate::error::{AppError, IndexingError, Result};
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, NaiveDate};
use regex::Regex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryIntent {
    pub labels: Vec<IntentLabel>,
    pub entities: Vec<Entity>,
    pub normalized_text: String,
    pub confidence: f32,
    pub query_type: QueryType,
    pub temporal_constraints: Option<TemporalConstraint>,
    pub file_type_filters: Vec<String>,
    pub semantic_expansion: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum IntentLabel {
    Search,           // General search
    Find,            // Specific item lookup
    Filter,          // Apply filters/constraints
    Compare,         // Compare documents/items
    Summarize,       // Get summary/overview
    Timeline,        // Temporal/chronological view
    Similar,         // Find similar items
    Recent,          // Recent items focus
    Collaborative,   // Shared/team content
    Personal,        // Personal content only
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub text: String,
    pub entity_type: EntityType,
    pub start_pos: usize,
    pub end_pos: usize,
    pub confidence: f32,
    pub canonical_form: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Date,
    Time,
    FileType,
    Technology,
    Project,
    Topic,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QueryType {
    Simple,          // Single intent, no complex logic
    Compound,        // Multiple intents with AND/OR
    Conversational,  // Natural language question
    Filter,          // Primarily filtering
    Exploratory,     // Broad, discovery-oriented
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConstraint {
    pub constraint_type: TemporalType,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub relative_time: Option<RelativeTime>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalType {
    Absolute,     // Specific dates
    Relative,     // "last week", "yesterday" 
    Range,        // Date ranges
    Recurring,    // "every Monday"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativeTime {
    pub unit: TimeUnit,
    pub amount: u32,
    pub direction: TimeDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeUnit {
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeDirection {
    Past,     // "last week"
    Future,   // "next month"
    Exact,    // "this week"
}

pub struct AdvancedQueryProcessor {
    intent_classifier: IntentClassifier,
    entity_extractor: EntityExtractor,
    spell_corrector: SpellCorrector,
    query_expander: QueryExpander,
    temporal_parser: TemporalParser,
    user_dictionary: UserDictionary,
    classification_cache: lru::LruCache<String, QueryIntent>,
}

struct IntentClassifier {
    intent_patterns: HashMap<IntentLabel, Vec<Regex>>,
    compound_indicators: Vec<Regex>,
    confidence_thresholds: HashMap<IntentLabel, f32>,
}

struct EntityExtractor {
    person_patterns: Vec<Regex>,
    org_patterns: Vec<Regex>,
    date_patterns: Vec<Regex>,
    file_type_patterns: Vec<Regex>,
    tech_patterns: Vec<Regex>,
    common_names: HashSet<String>,
    common_orgs: HashSet<String>,
}

struct SpellCorrector {
    user_dictionary: HashSet<String>,
    common_corrections: HashMap<String, String>,
    edit_distance_threshold: usize,
    domain_vocabulary: HashSet<String>,
}

struct QueryExpander {
    expansion_rules: HashMap<String, Vec<String>>,
    synonym_map: HashMap<String, Vec<String>>,
    expansion_threshold: f32,
    max_expansions: usize,
}

struct TemporalParser {
    relative_patterns: HashMap<String, RelativeTime>,
    absolute_patterns: Vec<Regex>,
    date_formats: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct UserDictionary {
    tokens: HashMap<String, TokenInfo>,
    total_tokens: u64,
    last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct TokenInfo {
    frequency: u32,
    last_seen: DateTime<Utc>,
    context: Vec<String>, // Common contexts where this token appears
}

impl AdvancedQueryProcessor {
    pub fn new() -> Result<Self> {
        let intent_classifier = IntentClassifier::new();
        let entity_extractor = EntityExtractor::new();
        let spell_corrector = SpellCorrector::new();
        let query_expander = QueryExpander::new();
        let temporal_parser = TemporalParser::new();
        let user_dictionary = UserDictionary::new();
        
        // Initialize LRU cache for 1000 query classifications
        let classification_cache = lru::LruCache::new(
            std::num::NonZeroUsize::new(1000).unwrap()
        );

        Ok(Self {
            intent_classifier,
            entity_extractor,
            spell_corrector,
            query_expander,
            temporal_parser,
            user_dictionary,
            classification_cache,
        })
    }

    /// Main entry point: analyze query and extract intent, entities, and structure
    pub async fn analyze_query(&mut self, query: &str) -> Result<QueryIntent> {
        // Check cache first
        if let Some(cached_intent) = self.classification_cache.get(query) {
            return Ok(cached_intent.clone());
        }

        println!("Analyzing query: '{}'", query);

        // Step 1: Normalize and spell-check the query
        let normalized_query = self.normalize_and_correct(query).await?;
        
        // Step 2: Extract entities (people, dates, file types, etc.)
        let entities = self.entity_extractor.extract_entities(&normalized_query)?;
        
        // Step 3: Classify intent(s) - handle compound queries
        let (intent_labels, query_type, confidence) = self.intent_classifier.classify_intent(&normalized_query, &entities)?;
        
        // Step 4: Parse temporal constraints
        let temporal_constraints = self.temporal_parser.parse_temporal(&normalized_query)?;
        
        // Step 5: Extract file type filters
        let file_type_filters = self.extract_file_type_filters(&normalized_query, &entities);
        
        // Step 6: Generate semantic expansions (if appropriate)
        let semantic_expansion = self.query_expander.expand_query(&normalized_query, &entities).await?;
        
        let query_intent = QueryIntent {
            labels: intent_labels,
            entities,
            normalized_text: normalized_query,
            confidence,
            query_type,
            temporal_constraints,
            file_type_filters,
            semantic_expansion,
        };

        // Cache the result
        self.classification_cache.put(query.to_string(), query_intent.clone());
        
        Ok(query_intent)
    }

    /// Normalize text and apply spell correction
    async fn normalize_and_correct(&mut self, query: &str) -> Result<String> {
        // Basic normalization
        let mut normalized = query.trim().to_lowercase();
        
        // Remove extra whitespace
        let whitespace_regex = Regex::new(r"\s+").unwrap();
        normalized = whitespace_regex.replace_all(&normalized, " ").to_string();
        
        // Apply spell correction
        normalized = self.spell_corrector.correct_spelling(&normalized, &self.user_dictionary).await?;
        
        Ok(normalized)
    }

    /// Extract file type filters from query and entities
    fn extract_file_type_filters(&self, query: &str, entities: &[Entity]) -> Vec<String> {
        let mut file_types = Vec::new();
        
        // From entities
        for entity in entities {
            if matches!(entity.entity_type, EntityType::FileType) {
                file_types.push(entity.text.clone());
            }
        }
        
        // Common file type mentions in query
        let file_patterns = [
            (r"\bpdf\b", "pdf"),
            (r"\bdocx?\b", "doc"),
            (r"\bpptx?\b", "ppt"),
            (r"\bxlsx?\b", "xls"),
            (r"\bimage\b", "image"),
            (r"\bphoto\b", "image"),
            (r"\bvideo\b", "video"),
            (r"\baudio\b", "audio"),
            (r"\bspreadsheet\b", "xls"),
            (r"\bpresentation\b", "ppt"),
            (r"\bdocument\b", "doc"),
        ];
        
        for (pattern, file_type) in &file_patterns {
            let regex = Regex::new(pattern).unwrap();
            if regex.is_match(query) {
                file_types.push(file_type.to_string());
            }
        }
        
        file_types.sort();
        file_types.dedup();
        file_types
    }

    /// Update user dictionary with tokens from successful searches
    pub async fn update_user_dictionary(&mut self, query: &str, was_successful: bool) -> Result<()> {
        if was_successful {
            self.user_dictionary.add_tokens(query).await?;
        }
        Ok(())
    }

    /// Get statistics about query classification
    pub fn get_classification_stats(&self) -> ClassificationStats {
        ClassificationStats {
            cache_size: self.classification_cache.len(),
            cache_capacity: self.classification_cache.cap().get(),
            user_dictionary_size: self.user_dictionary.tokens.len(),
            last_dictionary_update: self.user_dictionary.last_updated,
        }
    }

    /// Clear classification cache
    pub fn clear_cache(&mut self) {
        self.classification_cache.clear();
    }
}

#[derive(Debug)]
pub struct ClassificationStats {
    pub cache_size: usize,
    pub cache_capacity: usize,
    pub user_dictionary_size: usize,
    pub last_dictionary_update: DateTime<Utc>,
}

impl IntentClassifier {
    fn new() -> Self {
        let mut intent_patterns = HashMap::new();
        
        // Search patterns
        intent_patterns.insert(IntentLabel::Search, vec![
            Regex::new(r"\bsearch\b").unwrap(),
            Regex::new(r"\blook\s+for\b").unwrap(),
            Regex::new(r"\bfind\s+all\b").unwrap(),
        ]);
        
        // Find patterns (specific lookup)
        intent_patterns.insert(IntentLabel::Find, vec![
            Regex::new(r"\bfind\b").unwrap(),
            Regex::new(r"\bwhere\s+is\b").unwrap(),
            Regex::new(r"\bshow\s+me\b").unwrap(),
            Regex::new(r"\bget\s+me\b").unwrap(),
        ]);
        
        // Filter patterns
        intent_patterns.insert(IntentLabel::Filter, vec![
            Regex::new(r"\bfilter\b").unwrap(),
            Regex::new(r"\bonly\b").unwrap(),
            Regex::new(r"\bexclude\b").unwrap(),
            Regex::new(r"\bwithout\b").unwrap(),
        ]);
        
        // Compare patterns
        intent_patterns.insert(IntentLabel::Compare, vec![
            Regex::new(r"\bcompare\b").unwrap(),
            Regex::new(r"\bdifference\b").unwrap(),
            Regex::new(r"\bversus\b").unwrap(),
            Regex::new(r"\bvs\b").unwrap(),
        ]);
        
        // Summarize patterns
        intent_patterns.insert(IntentLabel::Summarize, vec![
            Regex::new(r"\bsummar").unwrap(),
            Regex::new(r"\boverview\b").unwrap(),
            Regex::new(r"\bgist\b").unwrap(),
            Regex::new(r"\bmain\s+points\b").unwrap(),
        ]);
        
        // Timeline patterns
        intent_patterns.insert(IntentLabel::Timeline, vec![
            Regex::new(r"\btimeline\b").unwrap(),
            Regex::new(r"\bchronological\b").unwrap(),
            Regex::new(r"\bhistory\b").unwrap(),
            Regex::new(r"\bover\s+time\b").unwrap(),
        ]);
        
        // Similar patterns
        intent_patterns.insert(IntentLabel::Similar, vec![
            Regex::new(r"\bsimilar\b").unwrap(),
            Regex::new(r"\blike\s+this\b").unwrap(),
            Regex::new(r"\brelated\b").unwrap(),
            Regex::new(r"\balike\b").unwrap(),
        ]);
        
        // Recent patterns
        intent_patterns.insert(IntentLabel::Recent, vec![
            Regex::new(r"\brecent").unwrap(),
            Regex::new(r"\blast\b").unwrap(),
            Regex::new(r"\blatest\b").unwrap(),
            Regex::new(r"\bnew\b").unwrap(),
        ]);
        
        // Compound query indicators
        let compound_indicators = vec![
            Regex::new(r"\band\b").unwrap(),
            Regex::new(r"\bor\b").unwrap(),
            Regex::new(r"\bbut\b").unwrap(),
            Regex::new(r"\balso\b").unwrap(),
            Regex::new(r"\bplus\b").unwrap(),
        ];
        
        let mut confidence_thresholds = HashMap::new();
        confidence_thresholds.insert(IntentLabel::Search, 0.7);
        confidence_thresholds.insert(IntentLabel::Find, 0.8);
        confidence_thresholds.insert(IntentLabel::Filter, 0.75);
        confidence_thresholds.insert(IntentLabel::Compare, 0.85);
        confidence_thresholds.insert(IntentLabel::Recent, 0.8);

        Self {
            intent_patterns,
            compound_indicators,
            confidence_thresholds,
        }
    }

    fn classify_intent(&self, query: &str, entities: &[Entity]) -> Result<(Vec<IntentLabel>, QueryType, f32)> {
        let mut intent_scores: HashMap<IntentLabel, f32> = HashMap::new();
        
        // Score each intent based on patterns
        for (intent, patterns) in &self.intent_patterns {
            let mut score = 0.0;
            for pattern in patterns {
                if pattern.is_match(query) {
                    score += 1.0;
                }
            }
            if score > 0.0 {
                intent_scores.insert(intent.clone(), score);
            }
        }
        
        // Boost scores based on entities
        for entity in entities {
            match entity.entity_type {
                EntityType::Date | EntityType::Time => {
                    *intent_scores.entry(IntentLabel::Timeline).or_insert(0.0) += 0.5;
                    *intent_scores.entry(IntentLabel::Recent).or_insert(0.0) += 0.3;
                }
                EntityType::FileType => {
                    *intent_scores.entry(IntentLabel::Filter).or_insert(0.0) += 0.4;
                }
                EntityType::Person => {
                    *intent_scores.entry(IntentLabel::Collaborative).or_insert(0.0) += 0.3;
                }
                _ => {}
            }
        }
        
        // Determine if compound query
        let is_compound = self.compound_indicators.iter()
            .any(|indicator| indicator.is_match(query));
        
        let query_type = if is_compound {
            QueryType::Compound
        } else if query.contains('?') {
            QueryType::Conversational
        } else if intent_scores.contains_key(&IntentLabel::Filter) {
            QueryType::Filter
        } else if query.len() > 50 {
            QueryType::Exploratory
        } else {
            QueryType::Simple
        };
        
        // Select top intents
        let mut sorted_intents: Vec<_> = intent_scores.into_iter().collect();
        sorted_intents.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let max_intents = if is_compound { 3 } else { 2 };
        let intents: Vec<IntentLabel> = sorted_intents.into_iter()
            .take(max_intents)
            .filter(|(_, score)| *score >= 0.5)
            .map(|(intent, _)| intent)
            .collect();
        
        // Calculate overall confidence
        let confidence = if intents.is_empty() { 
            0.3 
        } else { 
            0.8 
        };
        
        let final_intents = if intents.is_empty() {
            vec![IntentLabel::Search] // Default fallback
        } else {
            intents
        };

        Ok((final_intents, query_type, confidence))
    }
}

impl EntityExtractor {
    fn new() -> Self {
        let person_patterns = vec![
            Regex::new(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b").unwrap(), // "John Smith"
            Regex::new(r"\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+\b").unwrap(), // "John A. Smith"
            Regex::new(r"\bfrom\s+([A-Z][a-z]+)\b").unwrap(), // "from John"
            Regex::new(r"\bby\s+([A-Z][a-z]+)\b").unwrap(), // "by Sarah"
        ];
        
        let org_patterns = vec![
            Regex::new(r"\b[A-Z][a-z]+\s+Inc\.?\b").unwrap(),
            Regex::new(r"\b[A-Z][a-z]+\s+Corp\.?\b").unwrap(),
            Regex::new(r"\b[A-Z][a-z]+\s+LLC\b").unwrap(),
        ];
        
        let date_patterns = vec![
            Regex::new(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b").unwrap(), // MM/DD/YYYY
            Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").unwrap(), // YYYY-MM-DD
            Regex::new(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b").unwrap(),
            Regex::new(r"\blast\s+(week|month|year)\b").unwrap(),
            Regex::new(r"\byesterday\b").unwrap(),
            Regex::new(r"\btoday\b").unwrap(),
        ];
        
        let file_type_patterns = vec![
            Regex::new(r"\b\w+\.(pdf|doc|docx|ppt|pptx|xls|xlsx|txt|md|png|jpg|jpeg|gif|mp4|mp3|zip)\b").unwrap(),
            Regex::new(r"\b(pdf|doc|docx|ppt|pptx|xls|xlsx|image|video|audio)\s+files?\b").unwrap(),
        ];
        
        let tech_patterns = vec![
            Regex::new(r"\b(Python|JavaScript|Rust|Java|C\+\+|SQL|HTML|CSS|React|Vue|Angular)\b").unwrap(),
            Regex::new(r"\b(API|REST|GraphQL|JSON|XML|YAML|Docker|Kubernetes|AWS|Azure|GCP)\b").unwrap(),
        ];
        
        // Common names and organizations (simplified - in production would be much larger)
        let common_names: HashSet<String> = [
            "john", "jane", "smith", "johnson", "williams", "brown", "jones", "garcia",
            "miller", "davis", "rodriguez", "martinez", "hernandez", "lopez", "gonzalez"
        ].iter().map(|s| s.to_string()).collect();
        
        let common_orgs: HashSet<String> = [
            "google", "microsoft", "apple", "amazon", "facebook", "meta", "tesla", 
            "netflix", "twitter", "linkedin", "github", "stackoverflow"
        ].iter().map(|s| s.to_string()).collect();

        Self {
            person_patterns,
            org_patterns,
            date_patterns,
            file_type_patterns,
            tech_patterns,
            common_names,
            common_orgs,
        }
    }

    fn extract_entities(&self, query: &str) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();
        
        // Extract persons
        for pattern in &self.person_patterns {
            for mat in pattern.find_iter(query) {
                let text = mat.as_str().to_string();
                let confidence = if self.common_names.contains(&text.to_lowercase()) { 0.9 } else { 0.7 };
                
                entities.push(Entity {
                    text,
                    entity_type: EntityType::Person,
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    confidence,
                    canonical_form: None,
                });
            }
        }
        
        // Extract dates
        for pattern in &self.date_patterns {
            for mat in pattern.find_iter(query) {
                entities.push(Entity {
                    text: mat.as_str().to_string(),
                    entity_type: EntityType::Date,
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    confidence: 0.8,
                    canonical_form: None,
                });
            }
        }
        
        // Extract file types
        for pattern in &self.file_type_patterns {
            for mat in pattern.find_iter(query) {
                entities.push(Entity {
                    text: mat.as_str().to_string(),
                    entity_type: EntityType::FileType,
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    confidence: 0.9,
                    canonical_form: None,
                });
            }
        }
        
        // Extract technology terms
        for pattern in &self.tech_patterns {
            for mat in pattern.find_iter(query) {
                entities.push(Entity {
                    text: mat.as_str().to_string(),
                    entity_type: EntityType::Technology,
                    start_pos: mat.start(),
                    end_pos: mat.end(),
                    confidence: 0.85,
                    canonical_form: None,
                });
            }
        }
        
        // Sort by position and remove overlaps
        entities.sort_by_key(|e| e.start_pos);
        entities = self.remove_overlapping_entities(entities);
        
        Ok(entities)
    }
    
    fn remove_overlapping_entities(&self, mut entities: Vec<Entity>) -> Vec<Entity> {
        let mut result = Vec::new();
        
        for entity in entities.drain(..) {
            let overlaps = result.iter().any(|existing: &Entity| {
                (entity.start_pos < existing.end_pos) && (entity.end_pos > existing.start_pos)
            });
            
            if !overlaps {
                result.push(entity);
            }
        }
        
        result
    }
}

impl SpellCorrector {
    fn new() -> Self {
        let mut common_corrections = HashMap::new();
        
        // Common typos
        common_corrections.insert("teh".to_string(), "the".to_string());
        common_corrections.insert("recieve".to_string(), "receive".to_string());
        common_corrections.insert("seperate".to_string(), "separate".to_string());
        common_corrections.insert("definately".to_string(), "definitely".to_string());
        common_corrections.insert("occured".to_string(), "occurred".to_string());
        
        // Domain-specific vocabulary
        let domain_vocabulary: HashSet<String> = [
            "javascript", "typescript", "python", "kubernetes", "docker", "aws", "azure",
            "react", "vue", "angular", "nodejs", "npm", "git", "github", "stackoverflow",
            "api", "rest", "graphql", "json", "xml", "yaml", "sql", "nosql", "mongodb"
        ].iter().map(|s| s.to_string()).collect();

        Self {
            user_dictionary: HashSet::new(),
            common_corrections,
            edit_distance_threshold: 2,
            domain_vocabulary,
        }
    }

    async fn correct_spelling(&self, query: &str, user_dict: &UserDictionary) -> Result<String> {
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut corrected_words = Vec::new();
        
        for word in words {
            let cleaned_word = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            
            // Skip if in user dictionary, domain vocabulary, or too short
            if cleaned_word.len() < 3 || 
               user_dict.tokens.contains_key(&cleaned_word) ||
               self.domain_vocabulary.contains(&cleaned_word) {
                corrected_words.push(word.to_string());
                continue;
            }
            
            // Check common corrections
            if let Some(correction) = self.common_corrections.get(&cleaned_word) {
                corrected_words.push(correction.clone());
                continue;
            }
            
            // For now, keep original word (in production, would implement edit distance)
            corrected_words.push(word.to_string());
        }
        
        Ok(corrected_words.join(" "))
    }
}

impl QueryExpander {
    fn new() -> Self {
        let mut synonym_map = HashMap::new();
        
        // File type synonyms
        synonym_map.insert("doc".to_string(), vec!["document".to_string(), "docx".to_string(), "word".to_string()]);
        synonym_map.insert("pic".to_string(), vec!["picture".to_string(), "image".to_string(), "photo".to_string()]);
        synonym_map.insert("vid".to_string(), vec!["video".to_string(), "movie".to_string(), "film".to_string()]);
        
        // Action synonyms
        synonym_map.insert("find".to_string(), vec!["search".to_string(), "locate".to_string(), "get".to_string()]);
        synonym_map.insert("show".to_string(), vec!["display".to_string(), "list".to_string(), "present".to_string()]);
        
        Self {
            expansion_rules: HashMap::new(),
            synonym_map,
            expansion_threshold: 0.6,
            max_expansions: 3,
        }
    }

    async fn expand_query(&self, query: &str, entities: &[Entity]) -> Result<Option<Vec<String>>> {
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut expansions = Vec::new();
        
        for word in words {
            let cleaned_word = word.to_lowercase();
            if let Some(synonyms) = self.synonym_map.get(&cleaned_word) {
                for synonym in synonyms.iter().take(2) { // Limit synonyms
                    expansions.push(synonym.clone());
                }
            }
        }
        
        if expansions.is_empty() {
            Ok(None)
        } else {
            expansions.truncate(self.max_expansions);
            Ok(Some(expansions))
        }
    }
}

impl TemporalParser {
    fn new() -> Self {
        let mut relative_patterns = HashMap::new();
        
        relative_patterns.insert("yesterday".to_string(), RelativeTime {
            unit: TimeUnit::Day,
            amount: 1,
            direction: TimeDirection::Past,
        });
        
        relative_patterns.insert("last week".to_string(), RelativeTime {
            unit: TimeUnit::Week,
            amount: 1,
            direction: TimeDirection::Past,
        });
        
        relative_patterns.insert("last month".to_string(), RelativeTime {
            unit: TimeUnit::Month,
            amount: 1,
            direction: TimeDirection::Past,
        });
        
        let absolute_patterns = vec![
            Regex::new(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b").unwrap(),
            Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").unwrap(),
        ];
        
        let date_formats = vec![
            "%m/%d/%Y".to_string(),
            "%Y-%m-%d".to_string(),
            "%B %d, %Y".to_string(),
        ];

        Self {
            relative_patterns,
            absolute_patterns,
            date_formats,
        }
    }

    fn parse_temporal(&self, query: &str) -> Result<Option<TemporalConstraint>> {
        // Check for relative time expressions
        for (pattern, relative_time) in &self.relative_patterns {
            if query.contains(pattern) {
                return Ok(Some(TemporalConstraint {
                    constraint_type: TemporalType::Relative,
                    start_date: None,
                    end_date: None,
                    relative_time: Some(relative_time.clone()),
                    confidence: 0.8,
                }));
            }
        }
        
        // Check for absolute dates
        for pattern in &self.absolute_patterns {
            if let Some(mat) = pattern.find(query) {
                // Try to parse the date
                let date_str = mat.as_str();
                for format in &self.date_formats {
                    if let Ok(naive_date) = NaiveDate::parse_from_str(date_str, format) {
                        let datetime = naive_date.and_hms_opt(0, 0, 0).unwrap().and_utc();
                        return Ok(Some(TemporalConstraint {
                            constraint_type: TemporalType::Absolute,
                            start_date: Some(datetime),
                            end_date: None,
                            relative_time: None,
                            confidence: 0.9,
                        }));
                    }
                }
            }
        }
        
        Ok(None)
    }
}

impl UserDictionary {
    fn new() -> Self {
        Self {
            tokens: HashMap::new(),
            total_tokens: 0,
            last_updated: Utc::now(),
        }
    }

    async fn add_tokens(&mut self, text: &str) -> Result<()> {
        let words: Vec<&str> = text.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() >= 3)
            .collect();
        
        for word in words {
            let word_lower = word.to_lowercase();
            let token_info = self.tokens.entry(word_lower).or_insert(TokenInfo {
                frequency: 0,
                last_seen: Utc::now(),
                context: Vec::new(),
            });
            
            token_info.frequency += 1;
            token_info.last_seen = Utc::now();
            self.total_tokens += 1;
        }
        
        self.last_updated = Utc::now();
        Ok(())
    }
}

impl Default for AdvancedQueryProcessor {
    fn default() -> Self {
        Self::new().unwrap()
    }
}