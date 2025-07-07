use crate::error::Result;
use crate::core::user_intelligence::{UserIntelligenceSystem, QuerySuggestion, SuggestionSource};
use crate::core::query_intent::{QueryIntentClassifier, QueryIntent};
use crate::core::ranking::SearchContext;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;

/// Configuration for search suggestion system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestionConfig {
    /// Maximum number of suggestions to return
    pub max_suggestions: usize,
    /// Minimum confidence threshold for suggestions
    pub min_confidence: f32,
    /// Enable semantic expansion of queries
    pub enable_semantic_expansion: bool,
    /// Enable typo correction
    pub enable_typo_correction: bool,
    /// Enable trending suggestions
    pub enable_trending: bool,
    /// Maximum edit distance for typo correction
    pub max_edit_distance: usize,
    /// Weight for personal history suggestions
    pub personal_weight: f32,
    /// Weight for global popularity suggestions
    pub popularity_weight: f32,
    /// Weight for contextual suggestions
    pub context_weight: f32,
    /// Weight for semantic suggestions
    pub semantic_weight: f32,
}

impl Default for SuggestionConfig {
    fn default() -> Self {
        Self {
            max_suggestions: 10,
            min_confidence: 0.1,
            enable_semantic_expansion: true,
            enable_typo_correction: true,
            enable_trending: true,
            max_edit_distance: 2,
            personal_weight: 0.35,
            popularity_weight: 0.25,
            context_weight: 0.25,
            semantic_weight: 0.15,
        }
    }
}

/// Enhanced query suggestion with additional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSuggestion {
    pub base: QuerySuggestion,
    pub suggestion_type: SuggestionType,
    pub keywords: Vec<String>,
    pub category: Option<String>,
    pub recent_usage_count: usize,
    pub trending_score: f32,
    pub semantic_similarity: f32,
    pub typo_corrected: bool,
    pub completion_metadata: CompletionMetadata,
}

/// Type of suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionType {
    Completion,     // Direct completion of user input
    Correction,     // Typo correction
    Expansion,      // Semantic expansion
    Historical,     // From user history
    Trending,       // Currently trending
    Contextual,     // Based on current context
    Related,        // Related to recent queries
}

/// Metadata about how the completion was generated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionMetadata {
    pub algorithm: CompletionAlgorithm,
    pub prefix_length: usize,
    pub suffix_added: String,
    pub confidence_factors: HashMap<String, f32>,
}

/// Algorithm used for completion
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompletionAlgorithm {
    PrefixTree,
    NGram,
    SemanticEmbedding,
    PatternMatching,
    Hybrid,
}

/// Adaptive search suggestion system
pub struct SearchSuggestionSystem {
    config: SuggestionConfig,
    user_intelligence: UserIntelligenceSystem,
    intent_classifier: QueryIntentClassifier,
    suggestion_index: SuggestionIndex,
    trending_tracker: TrendingTracker,
    semantic_expander: SemanticExpander,
    typo_corrector: TypoCorrector,
    completion_engine: CompletionEngine,
}

/// Index for fast suggestion lookup
struct SuggestionIndex {
    prefix_tree: PrefixTree,
    ngram_index: NGramIndex,
    frequency_map: HashMap<String, usize>,
    category_map: HashMap<String, String>,
    last_update: DateTime<Utc>,
}

/// Prefix tree for efficient completion
struct PrefixTree {
    root: PrefixNode,
}

#[derive(Default)]
struct PrefixNode {
    children: HashMap<char, PrefixNode>,
    is_complete: bool,
    frequency: usize,
    suggestions: Vec<String>,
}

/// N-gram based index for partial matching
struct NGramIndex {
    bigrams: HashMap<String, HashSet<String>>,
    trigrams: HashMap<String, HashSet<String>>,
}

/// Tracks trending queries and topics
struct TrendingTracker {
    time_windows: Vec<TimeWindow>,
    trending_queries: Vec<TrendingQuery>,
    topic_clusters: HashMap<String, Vec<String>>,
}

/// Time window for trend analysis
struct TimeWindow {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    query_counts: HashMap<String, usize>,
}

/// Trending query information
#[derive(Clone)]
struct TrendingQuery {
    query: String,
    trend_score: f32,
    velocity: f32, // Rate of increase
    peak_time: DateTime<Utc>,
}

/// Semantic query expansion system
struct SemanticExpander {
    synonym_map: HashMap<String, Vec<String>>,
    concept_graph: ConceptGraph,
    domain_vocabularies: HashMap<String, Vec<String>>,
}

/// Graph of related concepts
struct ConceptGraph {
    nodes: HashMap<String, ConceptNode>,
    edges: Vec<ConceptEdge>,
}

struct ConceptNode {
    concept: String,
    category: String,
    weight: f32,
}

struct ConceptEdge {
    from: String,
    to: String,
    relation_type: RelationType,
    strength: f32,
}

#[derive(Debug, Clone)]
enum RelationType {
    Synonym,
    Hypernym,  // More general
    Hyponym,   // More specific
    Related,
    Antonym,
}

/// Typo correction system
struct TypoCorrector {
    dictionary: HashSet<String>,
    common_mistakes: HashMap<String, String>,
    keyboard_layout: KeyboardLayout,
}

/// Keyboard layout for typo analysis
struct KeyboardLayout {
    adjacent_keys: HashMap<char, Vec<char>>,
}

/// Advanced completion engine
struct CompletionEngine {
    pattern_matcher: PatternMatcher,
    ml_predictor: Option<MLPredictor>,
}

/// Pattern-based completion
struct PatternMatcher {
    common_patterns: Vec<QueryPattern>,
    user_patterns: HashMap<Uuid, Vec<QueryPattern>>,
}

#[derive(Clone)]
struct QueryPattern {
    pattern: String,
    frequency: usize,
    placeholders: Vec<String>,
}

/// Machine learning based predictor (placeholder)
struct MLPredictor {
    model_path: String,
}

impl SearchSuggestionSystem {
    pub async fn new(
        config: SuggestionConfig,
        user_intelligence: UserIntelligenceSystem,
        intent_classifier: QueryIntentClassifier,
    ) -> Result<Self> {
        let suggestion_index = SuggestionIndex {
            prefix_tree: PrefixTree::new(),
            ngram_index: NGramIndex::new(),
            frequency_map: HashMap::new(),
            category_map: HashMap::new(),
            last_update: Utc::now(),
        };

        let trending_tracker = TrendingTracker {
            time_windows: vec![],
            trending_queries: vec![],
            topic_clusters: HashMap::new(),
        };

        let semantic_expander = SemanticExpander::new();
        let typo_corrector = TypoCorrector::new();
        let completion_engine = CompletionEngine::new();

        Ok(Self {
            config,
            user_intelligence,
            intent_classifier,
            suggestion_index,
            trending_tracker,
            semantic_expander,
            typo_corrector,
            completion_engine,
        })
    }

    /// Get suggestions for a partial query
    pub async fn get_suggestions(
        &self,
        user_id: Uuid,
        partial_query: &str,
        context: &SearchContext,
    ) -> Result<Vec<EnhancedSuggestion>> {
        let mut all_suggestions = Vec::new();

        // 1. Get basic suggestions from user intelligence
        let basic_suggestions = self.user_intelligence
            .get_query_suggestions(user_id, partial_query, context)
            .await?;

        // Convert basic to enhanced suggestions
        for basic in basic_suggestions {
            all_suggestions.push(self.enhance_suggestion(basic, partial_query).await?);
        }

        // 2. Add prefix-based completions
        let completions = self.get_prefix_completions(partial_query).await?;
        all_suggestions.extend(completions);

        // 3. Add typo corrections if enabled
        if self.config.enable_typo_correction {
            let corrections = self.get_typo_corrections(partial_query).await?;
            all_suggestions.extend(corrections);
        }

        // 4. Add semantic expansions if enabled
        if self.config.enable_semantic_expansion {
            let expansions = self.get_semantic_expansions(partial_query, context).await?;
            all_suggestions.extend(expansions);
        }

        // 5. Add trending suggestions if enabled
        if self.config.enable_trending {
            let trending = self.get_trending_suggestions(partial_query).await?;
            all_suggestions.extend(trending);
        }

        // 6. Add pattern-based completions
        let pattern_completions = self.get_pattern_completions(user_id, partial_query).await?;
        all_suggestions.extend(pattern_completions);

        // Rank and filter suggestions
        let ranked_suggestions = self.rank_suggestions(all_suggestions, user_id, context).await?;

        // Take top N suggestions
        Ok(ranked_suggestions.into_iter()
            .take(self.config.max_suggestions)
            .collect())
    }

    /// Record a query for learning
    pub async fn record_query(&mut self, user_id: Uuid, query: &str, selected: bool) -> Result<()> {
        // Update frequency map
        *self.suggestion_index.frequency_map.entry(query.to_string()).or_insert(0) += 1;

        // Update prefix tree
        self.suggestion_index.prefix_tree.insert(query);

        // Update n-gram index
        self.suggestion_index.ngram_index.add_query(query);

        // Update trending tracker
        self.trending_tracker.record_query(query);

        // Learn patterns if query was selected
        if selected {
            self.completion_engine.learn_pattern(user_id, query)?;
        }

        Ok(())
    }

    /// Update trending topics
    pub async fn update_trending(&mut self) -> Result<()> {
        self.trending_tracker.update_trends()?;
        Ok(())
    }

    // Private helper methods
    async fn enhance_suggestion(
        &self,
        basic: QuerySuggestion,
        partial_query: &str,
    ) -> Result<EnhancedSuggestion> {
        let keywords = self.extract_keywords(&basic.query);
        let category = self.suggestion_index.category_map.get(&basic.query).cloned();
        let recent_usage_count = self.suggestion_index.frequency_map.get(&basic.query).copied().unwrap_or(0);
        
        let trending_score = self.trending_tracker.get_trend_score(&basic.query);
        let semantic_similarity = self.calculate_semantic_similarity(partial_query, &basic.query);

        let completion_metadata = CompletionMetadata {
            algorithm: CompletionAlgorithm::Hybrid,
            prefix_length: partial_query.len(),
            suffix_added: if basic.query.starts_with(partial_query) && basic.query.len() > partial_query.len() {
                basic.query[partial_query.len()..].to_string()
            } else {
                String::new()
            },
            confidence_factors: self.get_confidence_factors(&basic),
        };

        let suggestion_type = match basic.source {
            SuggestionSource::Personal => SuggestionType::Historical,
            SuggestionSource::Popular => SuggestionType::Completion,
            SuggestionSource::Contextual => SuggestionType::Contextual,
            SuggestionSource::Trending => SuggestionType::Trending,
            SuggestionSource::Semantic => SuggestionType::Expansion,
        };

        Ok(EnhancedSuggestion {
            base: basic,
            suggestion_type,
            keywords,
            category,
            recent_usage_count,
            trending_score,
            semantic_similarity,
            typo_corrected: false,
            completion_metadata,
        })
    }

    async fn get_prefix_completions(&self, partial: &str) -> Result<Vec<EnhancedSuggestion>> {
        let completions = self.suggestion_index.prefix_tree.find_completions(partial, 10);
        let mut suggestions = Vec::new();

        for (completion, frequency) in completions {
            let confidence = self.calculate_completion_confidence(partial, &completion, frequency);
            
            if confidence >= self.config.min_confidence {
                let base = QuerySuggestion {
                    query: completion.clone(),
                    confidence,
                    source: SuggestionSource::Popular,
                    context_relevance: 0.5,
                };

                suggestions.push(self.enhance_suggestion(base, partial).await?);
            }
        }

        Ok(suggestions)
    }

    async fn get_typo_corrections(&self, partial: &str) -> Result<Vec<EnhancedSuggestion>> {
        let corrections = self.typo_corrector.correct(partial, self.config.max_edit_distance);
        let mut suggestions = Vec::new();

        for (correction, distance) in corrections {
            let confidence = 1.0 - (distance as f32 / self.config.max_edit_distance as f32) * 0.5;
            
            let base = QuerySuggestion {
                query: correction.clone(),
                confidence,
                source: SuggestionSource::Popular,
                context_relevance: 0.4,
            };

            let mut enhanced = self.enhance_suggestion(base, partial).await?;
            enhanced.suggestion_type = SuggestionType::Correction;
            enhanced.typo_corrected = true;
            
            suggestions.push(enhanced);
        }

        Ok(suggestions)
    }

    async fn get_semantic_expansions(
        &self,
        partial: &str,
        context: &SearchContext,
    ) -> Result<Vec<EnhancedSuggestion>> {
        let expansions = self.semantic_expander.expand(partial, context);
        let mut suggestions = Vec::new();

        for (expansion, similarity) in expansions {
            let base = QuerySuggestion {
                query: expansion.clone(),
                confidence: similarity * 0.8,
                source: SuggestionSource::Semantic,
                context_relevance: 0.6,
            };

            let mut enhanced = self.enhance_suggestion(base, partial).await?;
            enhanced.suggestion_type = SuggestionType::Expansion;
            enhanced.semantic_similarity = similarity;
            
            suggestions.push(enhanced);
        }

        Ok(suggestions)
    }

    async fn get_trending_suggestions(&self, partial: &str) -> Result<Vec<EnhancedSuggestion>> {
        let trending = self.trending_tracker.get_trending_matching(partial);
        let mut suggestions = Vec::new();

        for trending_query in trending {
            let base = QuerySuggestion {
                query: trending_query.query.clone(),
                confidence: trending_query.trend_score,
                source: SuggestionSource::Trending,
                context_relevance: 0.3,
            };

            let mut enhanced = self.enhance_suggestion(base, partial).await?;
            enhanced.suggestion_type = SuggestionType::Trending;
            enhanced.trending_score = trending_query.trend_score;
            
            suggestions.push(enhanced);
        }

        Ok(suggestions)
    }

    async fn get_pattern_completions(
        &self,
        user_id: Uuid,
        partial: &str,
    ) -> Result<Vec<EnhancedSuggestion>> {
        let patterns = self.completion_engine.complete_pattern(user_id, partial)?;
        let mut suggestions = Vec::new();

        for (completion, confidence) in patterns {
            let base = QuerySuggestion {
                query: completion.clone(),
                confidence,
                source: SuggestionSource::Personal,
                context_relevance: 0.7,
            };

            let mut enhanced = self.enhance_suggestion(base, partial).await?;
            enhanced.completion_metadata.algorithm = CompletionAlgorithm::PatternMatching;
            
            suggestions.push(enhanced);
        }

        Ok(suggestions)
    }

    async fn rank_suggestions(
        &self,
        mut suggestions: Vec<EnhancedSuggestion>,
        user_id: Uuid,
        context: &SearchContext,
    ) -> Result<Vec<EnhancedSuggestion>> {
        // Calculate final scores
        for suggestion in &mut suggestions {
            let personal_score = if matches!(suggestion.base.source, SuggestionSource::Personal) {
                1.0
            } else {
                0.3
            };

            let popularity_score = if suggestion.recent_usage_count > 0 {
                (suggestion.recent_usage_count as f32).ln() / 10.0
            } else {
                0.0
            };
            let context_score = suggestion.base.context_relevance;
            let semantic_score = suggestion.semantic_similarity;

            // Weighted combination
            let final_score = self.config.personal_weight * personal_score
                + self.config.popularity_weight * popularity_score
                + self.config.context_weight * context_score
                + self.config.semantic_weight * semantic_score;

            // Boost for typo corrections and trending
            let boost = match suggestion.suggestion_type {
                SuggestionType::Correction => 1.2,
                SuggestionType::Trending => 1.1,
                _ => 1.0,
            };

            suggestion.base.confidence = (final_score * boost).min(1.0);
        }

        // Sort by confidence
        suggestions.sort_by(|a, b| {
            b.base.confidence.partial_cmp(&a.base.confidence)
                .unwrap_or(Ordering::Equal)
        });

        // Remove duplicates
        let mut seen = HashSet::new();
        suggestions.retain(|s| seen.insert(s.base.query.clone()));

        Ok(suggestions)
    }

    fn extract_keywords(&self, query: &str) -> Vec<String> {
        query.split_whitespace()
            .filter(|word| word.len() > 3)
            .map(|word| word.to_lowercase())
            .collect()
    }

    fn calculate_semantic_similarity(&self, query1: &str, query2: &str) -> f32 {
        // Simple word overlap for now
        let words1: HashSet<&str> = query1.split_whitespace().collect();
        let words2: HashSet<&str> = query2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
    }

    fn calculate_completion_confidence(&self, partial: &str, completion: &str, frequency: usize) -> f32 {
        let length_ratio = partial.len() as f32 / completion.len() as f32;
        let frequency_score = (frequency as f32).ln() / 10.0;
        
        (length_ratio * 0.3 + frequency_score * 0.7).min(1.0)
    }

    fn get_confidence_factors(&self, suggestion: &QuerySuggestion) -> HashMap<String, f32> {
        let mut factors = HashMap::new();
        factors.insert("base_confidence".to_string(), suggestion.confidence);
        factors.insert("context_relevance".to_string(), suggestion.context_relevance);
        factors
    }
}

// Implementation of helper structs
impl PrefixTree {
    fn new() -> Self {
        Self {
            root: PrefixNode::default(),
        }
    }

    fn insert(&mut self, query: &str) {
        let mut current = &mut self.root;
        
        for ch in query.chars() {
            current = current.children.entry(ch).or_default();
            current.frequency += 1;
        }
        
        current.is_complete = true;
        current.suggestions.push(query.to_string());
    }

    fn find_completions(&self, prefix: &str, max_results: usize) -> Vec<(String, usize)> {
        let mut current = &self.root;
        
        // Navigate to prefix
        for ch in prefix.chars() {
            match current.children.get(&ch) {
                Some(node) => current = node,
                None => return vec![],
            }
        }
        
        // Collect completions
        let mut completions = BinaryHeap::new();
        self.collect_completions(current, prefix.to_string(), &mut completions);
        
        // Extract top results
        let mut results = Vec::new();
        while let Some(completion) = completions.pop() {
            results.push((completion.query, completion.frequency));
            if results.len() >= max_results {
                break;
            }
        }
        
        results
    }

    fn collect_completions(&self, node: &PrefixNode, prefix: String, heap: &mut BinaryHeap<CompletionEntry>) {
        if node.is_complete {
            heap.push(CompletionEntry {
                query: prefix.clone(),
                frequency: node.frequency,
            });
        }
        
        for (ch, child) in &node.children {
            let mut new_prefix = prefix.clone();
            new_prefix.push(*ch);
            self.collect_completions(child, new_prefix, heap);
        }
    }
}

#[derive(Eq)]
struct CompletionEntry {
    query: String,
    frequency: usize,
}

impl Ord for CompletionEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.frequency.cmp(&other.frequency)
    }
}

impl PartialOrd for CompletionEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for CompletionEntry {
    fn eq(&self, other: &Self) -> bool {
        self.frequency == other.frequency
    }
}

impl NGramIndex {
    fn new() -> Self {
        Self {
            bigrams: HashMap::new(),
            trigrams: HashMap::new(),
        }
    }

    fn add_query(&mut self, query: &str) {
        let chars: Vec<char> = query.chars().collect();
        
        // Add bigrams
        for window in chars.windows(2) {
            let bigram = window.iter().collect::<String>();
            self.bigrams.entry(bigram).or_default().insert(query.to_string());
        }
        
        // Add trigrams
        for window in chars.windows(3) {
            let trigram = window.iter().collect::<String>();
            self.trigrams.entry(trigram).or_default().insert(query.to_string());
        }
    }
}

impl TrendingTracker {
    fn record_query(&mut self, query: &str) {
        let now = Utc::now();
        
        // Add to current time window
        if let Some(window) = self.time_windows.last_mut() {
            if window.end > now {
                *window.query_counts.entry(query.to_string()).or_insert(0) += 1;
            }
        }
    }

    fn update_trends(&mut self) -> Result<()> {
        // Calculate trend scores based on time windows
        let mut query_trends = HashMap::new();
        
        for window in &self.time_windows {
            for (query, count) in &window.query_counts {
                let age = (Utc::now() - window.start).num_hours() as f32;
                let time_decay = (-age / 24.0).exp(); // Daily decay
                let score = (*count as f32) * time_decay;
                
                *query_trends.entry(query.clone()).or_insert(0.0) += score;
            }
        }
        
        // Update trending queries
        self.trending_queries = query_trends.into_iter()
            .map(|(query, score)| TrendingQuery {
                query,
                trend_score: score,
                velocity: 0.0, // Would calculate rate of change
                peak_time: Utc::now(),
            })
            .collect();
        
        // Sort by trend score
        self.trending_queries.sort_by(|a, b| {
            b.trend_score.partial_cmp(&a.trend_score).unwrap_or(Ordering::Equal)
        });
        
        Ok(())
    }

    fn get_trend_score(&self, query: &str) -> f32 {
        self.trending_queries.iter()
            .find(|tq| tq.query == query)
            .map(|tq| tq.trend_score)
            .unwrap_or(0.0)
    }

    fn get_trending_matching(&self, prefix: &str) -> Vec<TrendingQuery> {
        self.trending_queries.iter()
            .filter(|tq| tq.query.starts_with(prefix))
            .cloned()
            .collect()
    }
}

impl SemanticExpander {
    fn new() -> Self {
        let mut synonym_map = HashMap::new();
        // Add some common synonyms
        synonym_map.insert("search".to_string(), vec!["find".to_string(), "lookup".to_string(), "query".to_string()]);
        synonym_map.insert("file".to_string(), vec!["document".to_string(), "doc".to_string()]);
        synonym_map.insert("image".to_string(), vec!["picture".to_string(), "photo".to_string()]);
        
        let concept_graph = ConceptGraph {
            nodes: HashMap::new(),
            edges: vec![],
        };
        
        Self {
            synonym_map,
            concept_graph,
            domain_vocabularies: HashMap::new(),
        }
    }

    fn expand(&self, query: &str, _context: &SearchContext) -> Vec<(String, f32)> {
        let mut expansions = Vec::new();
        
        // Simple synonym expansion
        for word in query.split_whitespace() {
            if let Some(synonyms) = self.synonym_map.get(&word.to_lowercase()) {
                for synonym in synonyms {
                    let expanded = query.replace(word, synonym);
                    expansions.push((expanded, 0.8));
                }
            }
        }
        
        expansions
    }
}

impl TypoCorrector {
    fn new() -> Self {
        let mut dictionary = HashSet::new();
        // Add common words
        for word in &["machine", "learning", "search", "file", "document", "python", "data", "analysis"] {
            dictionary.insert(word.to_string());
        }
        
        let mut common_mistakes = HashMap::new();
        common_mistakes.insert("teh".to_string(), "the".to_string());
        common_mistakes.insert("recieve".to_string(), "receive".to_string());
        
        let keyboard_layout = KeyboardLayout {
            adjacent_keys: Self::build_qwerty_layout(),
        };
        
        Self {
            dictionary,
            common_mistakes,
            keyboard_layout,
        }
    }

    fn correct(&self, word: &str, max_distance: usize) -> Vec<(String, usize)> {
        let mut corrections = Vec::new();
        
        // Check common mistakes first
        if let Some(correction) = self.common_mistakes.get(word) {
            corrections.push((correction.clone(), 1));
            return corrections;
        }
        
        // Find words within edit distance
        for dict_word in &self.dictionary {
            let distance = self.edit_distance(word, dict_word);
            if distance <= max_distance && distance > 0 {
                corrections.push((dict_word.clone(), distance));
            }
        }
        
        corrections.sort_by_key(|(_, dist)| *dist);
        corrections
    }

    fn edit_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }
        
        for (i, c1) in s1.chars().enumerate() {
            for (j, c2) in s2.chars().enumerate() {
                let cost = if c1 == c2 { 0 } else { 1 };
                matrix[i + 1][j + 1] = std::cmp::min(
                    std::cmp::min(
                        matrix[i][j + 1] + 1,     // deletion
                        matrix[i + 1][j] + 1       // insertion
                    ),
                    matrix[i][j] + cost            // substitution
                );
            }
        }
        
        matrix[len1][len2]
    }

    fn build_qwerty_layout() -> HashMap<char, Vec<char>> {
        let mut layout = HashMap::new();
        
        // Simplified QWERTY adjacency
        layout.insert('q', vec!['w', 'a']);
        layout.insert('w', vec!['q', 'e', 's', 'a']);
        layout.insert('e', vec!['w', 'r', 'd', 's']);
        // ... would add full keyboard
        
        layout
    }
}

impl CompletionEngine {
    fn new() -> Self {
        Self {
            pattern_matcher: PatternMatcher {
                common_patterns: Self::load_common_patterns(),
                user_patterns: HashMap::new(),
            },
            ml_predictor: None,
        }
    }

    fn load_common_patterns() -> Vec<QueryPattern> {
        vec![
            QueryPattern {
                pattern: "find {} in {}".to_string(),
                frequency: 100,
                placeholders: vec!["term".to_string(), "location".to_string()],
            },
            QueryPattern {
                pattern: "{} tutorial".to_string(),
                frequency: 80,
                placeholders: vec!["topic".to_string()],
            },
            QueryPattern {
                pattern: "how to {}".to_string(),
                frequency: 90,
                placeholders: vec!["action".to_string()],
            },
        ]
    }

    fn complete_pattern(&self, user_id: Uuid, partial: &str) -> Result<Vec<(String, f32)>> {
        let mut completions = Vec::new();
        
        // Check user patterns first
        if let Some(user_patterns) = self.pattern_matcher.user_patterns.get(&user_id) {
            for pattern in user_patterns {
                if let Some(completion) = self.match_pattern(partial, pattern) {
                    let confidence = 0.9 * (pattern.frequency as f32 / 100.0);
                    completions.push((completion, confidence));
                }
            }
        }
        
        // Check common patterns
        for pattern in &self.pattern_matcher.common_patterns {
            if let Some(completion) = self.match_pattern(partial, pattern) {
                let confidence = 0.7 * (pattern.frequency as f32 / 100.0);
                completions.push((completion, confidence));
            }
        }
        
        Ok(completions)
    }

    fn match_pattern(&self, partial: &str, pattern: &QueryPattern) -> Option<String> {
        // Simple pattern matching (would be more sophisticated)
        let pattern_start = pattern.pattern.split(' ').next()?;
        if partial.starts_with(pattern_start) {
            Some(pattern.pattern.replace("{}", "..."))
        } else {
            None
        }
    }

    fn learn_pattern(&mut self, user_id: Uuid, query: &str) -> Result<()> {
        // Extract pattern from query (simplified)
        let pattern = self.extract_pattern(query);
        
        let user_patterns = self.pattern_matcher.user_patterns
            .entry(user_id)
            .or_default();
        
        // Update or add pattern
        if let Some(existing) = user_patterns.iter_mut().find(|p| p.pattern == pattern) {
            existing.frequency += 1;
        } else {
            user_patterns.push(QueryPattern {
                pattern,
                frequency: 1,
                placeholders: vec![],
            });
        }
        
        Ok(())
    }

    fn extract_pattern(&self, query: &str) -> String {
        // Simplified pattern extraction
        // Would use more sophisticated NLP in real implementation
        let words: Vec<&str> = query.split_whitespace().collect();
        if words.len() >= 3 {
            format!("{} {{}} {}", words[0], words[words.len() - 1])
        } else {
            query.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prefix_tree() {
        let mut tree = PrefixTree::new();
        tree.insert("machine learning");
        tree.insert("machine learning algorithms");
        tree.insert("machine vision");
        
        let completions = tree.find_completions("machine", 5);
        assert!(completions.len() >= 2);
        assert!(completions.iter().any(|(q, _)| q.contains("learning")));
    }

    #[tokio::test]
    async fn test_typo_correction() {
        let corrector = TypoCorrector::new();
        let corrections = corrector.correct("machin", 1);
        
        assert!(!corrections.is_empty());
        assert_eq!(corrections[0].0, "machine");
        assert_eq!(corrections[0].1, 1);
    }

    #[tokio::test]
    async fn test_semantic_expansion() {
        let expander = SemanticExpander::new();
        let context = SearchContext {
            session_id: Uuid::new_v4(),
            current_project: None,
            recent_documents: vec![],
            active_applications: vec![],
            search_history: vec![],
            timestamp: Utc::now(),
        };
        
        let expansions = expander.expand("search files", &context);
        assert!(!expansions.is_empty());
        assert!(expansions.iter().any(|(q, _)| q.contains("find")));
    }
}