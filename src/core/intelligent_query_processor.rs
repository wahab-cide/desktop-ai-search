use crate::error::Result;
use crate::core::advanced_query_processor::{AdvancedQueryProcessor, QueryIntent, IntentLabel, QueryType};
use crate::core::boolean_query_parser::{BooleanQueryParser, ParsedQuery as BooleanParsedQuery, QueryNode};
use crate::core::boolean_query_executor::{BooleanQueryExecutor, QueryExecutionResult};
use crate::database::Database;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Intelligent query processor that combines natural language understanding with boolean logic
/// This is the main entry point for complex query processing in Week 7
pub struct IntelligentQueryProcessor {
    nl_processor: AdvancedQueryProcessor,
    boolean_parser: BooleanQueryParser,
    boolean_executor: BooleanQueryExecutor,
    query_strategy_selector: QueryStrategySelector,
    integration_cache: lru::LruCache<String, IntegratedQueryResult>,
    processing_stats: ProcessingStatistics,
}

/// Result combining natural language understanding with boolean execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedQueryResult {
    pub original_query: String,
    pub detected_intent: QueryIntent,
    pub boolean_structure: Option<BooleanParsedQuery>,
    pub execution_result: Option<QueryExecutionResult>,
    pub processing_strategy: QueryProcessingStrategy,
    pub total_processing_time_ms: f64,
    pub confidence_score: f32,
    pub suggested_refinements: Vec<QueryRefinement>,
}

/// Strategy for processing different types of queries
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryProcessingStrategy {
    /// Pure natural language processing
    NaturalLanguageOnly,
    /// Pure boolean/structured query
    BooleanQueryOnly,
    /// Hybrid: NL understanding + boolean structure
    HybridIntelligent,
    /// Fallback when other strategies fail
    FallbackSearch,
}

/// Selector for determining optimal query processing strategy
struct QueryStrategySelector {
    boolean_indicators: Vec<String>,
    nl_indicators: Vec<String>,
    complexity_thresholds: StrategyThresholds,
}

#[derive(Debug)]
struct StrategyThresholds {
    boolean_confidence_threshold: f32,
    nl_confidence_threshold: f32,
    complexity_threshold: f32,
    hybrid_preference_score: f32,
}

/// Suggested query refinements for better results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRefinement {
    pub refinement_type: RefinementType,
    pub suggestion: String,
    pub explanation: String,
    pub estimated_improvement: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RefinementType {
    /// Add more specific terms
    AddSpecificity,
    /// Use boolean operators
    UseBooleanLogic,
    /// Specify fields explicitly
    UseFieldSpecifiers,
    /// Add temporal constraints
    AddTimeConstraints,
    /// Use phrase queries
    UsePhraseQueries,
    /// Simplify complex query
    SimplifyQuery,
    /// Fix syntax errors
    FixSyntax,
}

/// Processing statistics for monitoring and optimization
#[derive(Debug, Clone)]
struct ProcessingStatistics {
    total_queries_processed: u64,
    strategy_usage: HashMap<QueryProcessingStrategy, u64>,
    average_processing_time_ms: f64,
    success_rate: f32,
    cache_hit_rate: f32,
}

impl IntelligentQueryProcessor {
    pub fn new(database: Database) -> Result<Self> {
        let nl_processor = AdvancedQueryProcessor::new()?;
        let boolean_parser = BooleanQueryParser::new();
        let boolean_executor = BooleanQueryExecutor::new(database);
        
        let query_strategy_selector = QueryStrategySelector::new();
        
        let integration_cache = lru::LruCache::new(
            std::num::NonZeroUsize::new(500).unwrap()
        );
        
        let processing_stats = ProcessingStatistics {
            total_queries_processed: 0,
            strategy_usage: HashMap::new(),
            average_processing_time_ms: 0.0,
            success_rate: 1.0,
            cache_hit_rate: 0.0,
        };

        Ok(Self {
            nl_processor,
            boolean_parser,
            boolean_executor,
            query_strategy_selector,
            integration_cache,
            processing_stats,
        })
    }

    /// Main entry point: intelligently process any query with optimal strategy
    pub async fn process_query(&mut self, query: &str) -> Result<IntegratedQueryResult> {
        let start_time = std::time::Instant::now();
        println!("🧠 Processing intelligent query: '{}'", query);
        
        // Check cache first
        if let Some(cached_result) = self.integration_cache.get(query).cloned() {
            println!("  ✅ Cache hit for intelligent query");
            self.processing_stats.cache_hit_rate = 
                (self.processing_stats.cache_hit_rate * self.processing_stats.total_queries_processed as f32 + 1.0) 
                / (self.processing_stats.total_queries_processed + 1) as f32;
            
            self.processing_stats.total_queries_processed += 1;
            return Ok(cached_result);
        }

        // Step 1: Natural language understanding
        println!("  🔍 Analyzing natural language intent...");
        let detected_intent = self.nl_processor.analyze_query(query).await?;
        
        // Step 2: Determine optimal processing strategy
        println!("  🎯 Selecting processing strategy...");
        let strategy = self.query_strategy_selector.select_strategy(query, &detected_intent);
        println!("    Selected strategy: {:?}", strategy);
        
        // Step 3: Execute query based on selected strategy
        let mut boolean_structure = None;
        let mut execution_result = None;
        let mut confidence_score = detected_intent.confidence;
        
        match strategy {
            QueryProcessingStrategy::NaturalLanguageOnly => {
                // Pure NL processing - convert intent to search
                execution_result = Some(self.execute_nl_intent(&detected_intent).await?);
            },
            QueryProcessingStrategy::BooleanQueryOnly => {
                // Pure boolean processing
                let parsed_boolean = self.boolean_parser.parse(query)?;
                boolean_structure = Some(parsed_boolean.clone());
                execution_result = Some(self.boolean_executor.execute(&parsed_boolean).await?);
            },
            QueryProcessingStrategy::HybridIntelligent => {
                // Hybrid: use NL understanding to enhance boolean parsing
                execution_result = Some(self.execute_hybrid_strategy(query, &detected_intent).await?);
                confidence_score = (confidence_score + 0.8) / 2.0; // Boost confidence for hybrid
            },
            QueryProcessingStrategy::FallbackSearch => {
                // Fallback: simple text search
                execution_result = Some(self.execute_fallback_search(query).await?);
                confidence_score = 0.5; // Lower confidence for fallback
            },
        }
        
        // Step 4: Generate query refinement suggestions
        println!("  💡 Generating refinement suggestions...");
        let suggested_refinements = self.generate_refinement_suggestions(
            query, 
            &detected_intent, 
            &strategy,
            execution_result.as_ref()
        );
        
        let total_processing_time = start_time.elapsed().as_millis() as f64;
        
        let result = IntegratedQueryResult {
            original_query: query.to_string(),
            detected_intent,
            boolean_structure,
            execution_result,
            processing_strategy: strategy.clone(),
            total_processing_time_ms: total_processing_time,
            confidence_score,
            suggested_refinements,
        };
        
        // Cache the result
        self.integration_cache.put(query.to_string(), result.clone());
        
        // Update statistics
        self.update_processing_statistics(&strategy, total_processing_time, true);
        
        println!("  ✅ Query processing complete in {:.2}ms", total_processing_time);
        
        Ok(result)
    }

    /// Execute natural language intent as a search query
    async fn execute_nl_intent(&mut self, intent: &QueryIntent) -> Result<QueryExecutionResult> {
        println!("    Executing NL intent: {:?}", intent.labels);
        
        // Convert NL intent to boolean query structure
        let boolean_query = self.intent_to_boolean_query(intent);
        let parsed_boolean = self.boolean_parser.parse(&boolean_query)?;
        
        self.boolean_executor.execute(&parsed_boolean).await
    }

    /// Execute hybrid strategy combining NL understanding with boolean parsing
    async fn execute_hybrid_strategy(&mut self, query: &str, intent: &QueryIntent) -> Result<QueryExecutionResult> {
        println!("    Executing hybrid strategy");
        
        // Try to parse as boolean first
        let boolean_result = self.boolean_parser.parse(query);
        
        match boolean_result {
            Ok(parsed_boolean) => {
                // Successfully parsed as boolean - enhance with NL understanding
                let enhanced_query = self.enhance_boolean_with_intent(&parsed_boolean, intent)?;
                self.boolean_executor.execute(&enhanced_query).await
            },
            Err(_) => {
                // Failed to parse as boolean - use NL intent to construct boolean query
                let constructed_query = self.construct_boolean_from_intent(query, intent);
                let parsed_constructed = self.boolean_parser.parse(&constructed_query)?;
                self.boolean_executor.execute(&parsed_constructed).await
            }
        }
    }

    /// Execute fallback simple search
    async fn execute_fallback_search(&mut self, query: &str) -> Result<QueryExecutionResult> {
        println!("    Executing fallback search");
        
        // Simple word-based search as fallback
        let simple_query = format!("content:{}", query.replace(' ', " AND content:"));
        let parsed_simple = self.boolean_parser.parse(&simple_query)?;
        
        self.boolean_executor.execute(&parsed_simple).await
    }

    /// Convert natural language intent to boolean query string
    fn intent_to_boolean_query(&self, intent: &QueryIntent) -> String {
        let mut query_parts = Vec::new();
        
        // Add main content search
        if !intent.normalized_text.is_empty() {
            query_parts.push(format!("content:{}", intent.normalized_text));
        }
        
        // Add entity constraints
        for entity in &intent.entities {
            match entity.entity_type {
                crate::core::advanced_query_processor::EntityType::Person => {
                    query_parts.push(format!("author:{}", entity.text));
                },
                crate::core::advanced_query_processor::EntityType::FileType => {
                    query_parts.push(format!("type:{}", entity.text));
                },
                crate::core::advanced_query_processor::EntityType::Date => {
                    query_parts.push(format!("modified:[{} TO *]", entity.text));
                },
                _ => {
                    query_parts.push(format!("content:{}", entity.text));
                }
            }
        }
        
        // Add file type filters
        for file_type in &intent.file_type_filters {
            query_parts.push(format!("type:{}", file_type));
        }
        
        // Combine with appropriate operators based on intent
        if intent.labels.contains(&IntentLabel::Filter) {
            query_parts.join(" AND ")
        } else if intent.labels.contains(&IntentLabel::Compare) {
            query_parts.join(" OR ")
        } else {
            query_parts.join(" AND ")
        }
    }

    /// Enhance boolean query with natural language understanding
    fn enhance_boolean_with_intent(&self, boolean_query: &BooleanParsedQuery, intent: &QueryIntent) -> Result<BooleanParsedQuery> {
        // This would enhance the boolean query tree with insights from NL understanding
        // For now, return the original query
        Ok(boolean_query.clone())
    }

    /// Construct boolean query from natural language intent when parsing fails
    fn construct_boolean_from_intent(&self, original_query: &str, intent: &QueryIntent) -> String {
        // Intelligently construct a boolean query from failed parse + NL understanding
        let mut constructed = self.intent_to_boolean_query(intent);
        
        // If the original had boolean-like words, try to preserve them
        let boolean_words = ["AND", "OR", "NOT", "(", ")"];
        for word in boolean_words {
            if original_query.contains(word) {
                // Try to preserve boolean structure
                constructed = original_query.replace("and", "AND")
                    .replace("or", "OR")
                    .replace("not", "NOT");
                break;
            }
        }
        
        constructed
    }

    /// Generate suggestions for improving query results
    fn generate_refinement_suggestions(
        &self,
        query: &str,
        intent: &QueryIntent,
        strategy: &QueryProcessingStrategy,
        execution_result: Option<&QueryExecutionResult>,
    ) -> Vec<QueryRefinement> {
        let mut suggestions = Vec::new();
        
        // Suggest boolean operators if query has multiple concepts
        if query.split_whitespace().count() > 2 && !query.contains("AND") && !query.contains("OR") {
            suggestions.push(QueryRefinement {
                refinement_type: RefinementType::UseBooleanLogic,
                suggestion: format!("{}", query.replace(' ', " AND ")),
                explanation: "Use AND/OR operators to clarify relationships between terms".to_string(),
                estimated_improvement: 0.3,
            });
        }
        
        // Suggest field specifiers if no fields are used
        if !query.contains(':') && !intent.entities.is_empty() {
            for entity in &intent.entities {
                match entity.entity_type {
                    crate::core::advanced_query_processor::EntityType::Person => {
                        suggestions.push(QueryRefinement {
                            refinement_type: RefinementType::UseFieldSpecifiers,
                            suggestion: format!("author:{} AND {}", entity.text, query.replace(&entity.text, "").trim()),
                            explanation: format!("Search specifically in author field for '{}'", entity.text),
                            estimated_improvement: 0.4,
                        });
                    },
                    crate::core::advanced_query_processor::EntityType::FileType => {
                        suggestions.push(QueryRefinement {
                            refinement_type: RefinementType::UseFieldSpecifiers,
                            suggestion: format!("type:{} AND {}", entity.text, query.replace(&entity.text, "").trim()),
                            explanation: format!("Filter by file type '{}'", entity.text),
                            estimated_improvement: 0.5,
                        });
                    },
                    _ => {}
                }
            }
        }
        
        // Suggest phrase queries for multi-word terms
        if query.split_whitespace().count() >= 2 && !query.contains('"') {
            suggestions.push(QueryRefinement {
                refinement_type: RefinementType::UsePhraseQueries,
                suggestion: format!("\"{}\"", query),
                explanation: "Use phrase search to find exact word sequence".to_string(),
                estimated_improvement: 0.3,
            });
        }
        
        // Suggest temporal constraints if date entities are detected
        if let Some(temporal) = &intent.temporal_constraints {
            suggestions.push(QueryRefinement {
                refinement_type: RefinementType::AddTimeConstraints,
                suggestion: format!("{} AND modified:[{:?} TO *]", query, temporal.constraint_type),
                explanation: "Add temporal constraint to narrow results by date".to_string(),
                estimated_improvement: 0.4,
            });
        }
        
        // Suggest simplification if query is too complex
        if let Some(result) = execution_result {
            if result.query_plan.estimated_cost > 10.0 {
                suggestions.push(QueryRefinement {
                    refinement_type: RefinementType::SimplifyQuery,
                    suggestion: query.split_whitespace().take(3).collect::<Vec<_>>().join(" "),
                    explanation: "Simplify query to improve performance".to_string(),
                    estimated_improvement: 0.2,
                });
            }
        }
        
        // Sort by estimated improvement
        suggestions.sort_by(|a, b| b.estimated_improvement.partial_cmp(&a.estimated_improvement).unwrap());
        
        suggestions
    }

    /// Update processing statistics
    fn update_processing_statistics(&mut self, strategy: &QueryProcessingStrategy, processing_time: f64, success: bool) {
        self.processing_stats.total_queries_processed += 1;
        
        *self.processing_stats.strategy_usage.entry(strategy.clone()).or_insert(0) += 1;
        
        let total_queries = self.processing_stats.total_queries_processed as f64;
        self.processing_stats.average_processing_time_ms = 
            (self.processing_stats.average_processing_time_ms * (total_queries - 1.0) + processing_time) / total_queries;
        
        if success {
            self.processing_stats.success_rate = 
                ((self.processing_stats.success_rate as f64 * (total_queries - 1.0) + 1.0) / total_queries) as f32;
        } else {
            self.processing_stats.success_rate = 
                ((self.processing_stats.success_rate as f64 * (total_queries - 1.0)) / total_queries) as f32;
        }
    }

    /// Get processing statistics
    pub fn get_processing_statistics(&self) -> ProcessingStatistics {
        self.processing_stats.clone()
    }

    /// Clear caches
    pub fn clear_caches(&mut self) {
        self.integration_cache.clear();
        self.nl_processor.clear_cache();
        self.boolean_executor.clear_cache();
    }

    /// Get cache statistics
    pub fn get_cache_statistics(&self) -> HashMap<String, (usize, usize)> {
        let mut stats = HashMap::new();
        
        stats.insert(
            "integration_cache".to_string(),
            (self.integration_cache.len(), self.integration_cache.cap().get())
        );
        
        stats.insert(
            "nl_cache".to_string(),
            (self.nl_processor.get_classification_stats().cache_size, self.nl_processor.get_classification_stats().cache_capacity)
        );
        
        stats.insert(
            "boolean_cache".to_string(),
            self.boolean_executor.get_cache_stats()
        );
        
        stats
    }
}

impl QueryStrategySelector {
    fn new() -> Self {
        Self {
            boolean_indicators: vec![
                "AND".to_string(), "OR".to_string(), "NOT".to_string(),
                "(".to_string(), ")".to_string(), ":".to_string(),
                "\"".to_string(), "*".to_string(), "?".to_string(),
                "[".to_string(), "]".to_string(), "TO".to_string(),
            ],
            nl_indicators: vec![
                "find".to_string(), "search".to_string(), "show".to_string(),
                "what".to_string(), "where".to_string(), "when".to_string(),
                "who".to_string(), "how".to_string(), "why".to_string(),
                "can".to_string(), "could".to_string(), "would".to_string(),
            ],
            complexity_thresholds: StrategyThresholds {
                boolean_confidence_threshold: 0.7,
                nl_confidence_threshold: 0.6,
                complexity_threshold: 5.0,
                hybrid_preference_score: 0.8,
            },
        }
    }

    fn select_strategy(&self, query: &str, intent: &QueryIntent) -> QueryProcessingStrategy {
        let boolean_score = self.calculate_boolean_score(query);
        let nl_score = self.calculate_nl_score(query, intent);
        let complexity_score = intent.entities.len() as f32 + 
                             intent.file_type_filters.len() as f32 +
                             if intent.temporal_constraints.is_some() { 1.0 } else { 0.0 };
        
        println!("    Strategy scores - Boolean: {:.2}, NL: {:.2}, Complexity: {:.2}", 
                boolean_score, nl_score, complexity_score);

        // Decision logic for strategy selection
        if boolean_score > self.complexity_thresholds.boolean_confidence_threshold {
            if nl_score > self.complexity_thresholds.nl_confidence_threshold {
                QueryProcessingStrategy::HybridIntelligent
            } else {
                QueryProcessingStrategy::BooleanQueryOnly
            }
        } else if nl_score > self.complexity_thresholds.nl_confidence_threshold {
            QueryProcessingStrategy::NaturalLanguageOnly
        } else if complexity_score > self.complexity_thresholds.complexity_threshold {
            QueryProcessingStrategy::HybridIntelligent
        } else {
            QueryProcessingStrategy::FallbackSearch
        }
    }

    fn calculate_boolean_score(&self, query: &str) -> f32 {
        let mut score = 0.0;
        let total_indicators = self.boolean_indicators.len() as f32;
        
        for indicator in &self.boolean_indicators {
            if query.contains(indicator) {
                score += 1.0;
            }
        }
        
        score / total_indicators
    }

    fn calculate_nl_score(&self, query: &str, intent: &QueryIntent) -> f32 {
        let mut score = 0.0;
        let query_lower = query.to_lowercase();
        
        // Check for natural language indicators
        for indicator in &self.nl_indicators {
            if query_lower.contains(indicator) {
                score += 0.1;
            }
        }
        
        // Factor in intent confidence
        score += intent.confidence * 0.5;
        
        // Factor in complexity of intent analysis
        if !intent.entities.is_empty() {
            score += 0.2;
        }
        
        if intent.temporal_constraints.is_some() {
            score += 0.2;
        }
        
        if intent.labels.len() > 1 {
            score += 0.1;
        }
        
        score.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_intelligent_processing() {
        let database = Database::new("test.db").unwrap();
        let mut processor = IntelligentQueryProcessor::new(database).unwrap();
        
        let result = processor.process_query("find documents by john").await.unwrap();
        
        assert_eq!(result.original_query, "find documents by john");
        assert!(result.confidence_score > 0.0);
        assert!(!result.suggested_refinements.is_empty());
    }

    #[tokio::test]
    async fn test_boolean_strategy_selection() {
        let database = Database::new("test.db").unwrap();
        let mut processor = IntelligentQueryProcessor::new(database).unwrap();
        
        let result = processor.process_query("author:john AND type:pdf").await.unwrap();
        
        assert!(matches!(result.processing_strategy, QueryProcessingStrategy::BooleanQueryOnly | QueryProcessingStrategy::HybridIntelligent));
    }

    #[tokio::test]
    async fn test_refinement_suggestions() {
        let database = Database::new("test.db").unwrap();
        let mut processor = IntelligentQueryProcessor::new(database).unwrap();
        
        let result = processor.process_query("machine learning python").await.unwrap();
        
        // Should suggest boolean operators or field specifiers
        assert!(!result.suggested_refinements.is_empty());
        
        let has_boolean_suggestion = result.suggested_refinements.iter()
            .any(|r| r.refinement_type == RefinementType::UseBooleanLogic);
        
        assert!(has_boolean_suggestion);
    }
}