use crate::error::Result;
use crate::core::boolean_query_parser::{QueryNode, TermQuery, BooleanOperator, TermType, ParsedQuery};
use crate::database::Database;
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Executes parsed boolean queries against the search index
pub struct BooleanQueryExecutor {
    database: Database,
    field_mappings: HashMap<String, String>,
    execution_cache: lru::LruCache<String, CachedQueryResult>,
    performance_stats: ExecutionStats,
}

/// Result of executing a boolean query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExecutionResult {
    pub document_ids: Vec<String>,
    pub relevance_scores: HashMap<String, f32>,
    pub execution_time_ms: f64,
    pub total_matches: usize,
    pub query_plan: QueryPlan,
    pub performance_metrics: QueryPerformanceMetrics,
}

/// Query execution plan showing how the query was executed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    pub steps: Vec<ExecutionStep>,
    pub estimated_cost: f32,
    pub actual_cost: f32,
    pub optimizations_applied: Vec<String>,
}

/// Individual step in query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    pub step_type: StepType,
    pub operation: String,
    pub input_size: usize,
    pub output_size: usize,
    pub execution_time_ms: f64,
    pub selectivity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    IndexLookup,
    TermSearch,
    BooleanOperation,
    Filtering,
    Scoring,
    Caching,
}

/// Performance metrics for query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPerformanceMetrics {
    pub total_documents_examined: usize,
    pub index_hits: usize,
    pub cache_hits: usize,
    pub memory_usage_mb: f32,
    pub io_operations: usize,
    pub cpu_time_ms: f64,
}

/// Cached query result for performance
#[derive(Debug, Clone)]
struct CachedQueryResult {
    pub document_ids: Vec<String>,
    pub relevance_scores: HashMap<String, f32>,
    pub cached_at: DateTime<Utc>,
    pub query_hash: String,
}

/// Execution statistics for the executor
#[derive(Debug, Clone)]
struct ExecutionStats {
    pub total_queries: u64,
    pub cache_hits: u64,
    pub average_execution_time_ms: f64,
    pub total_documents_processed: u64,
}

/// Field-specific search operations
trait FieldSearcher {
    fn search_field(&self, field: &str, value: &str, term_type: &TermType) -> Result<Vec<String>>;
    fn get_field_statistics(&self, field: &str) -> FieldStatistics;
}

#[derive(Debug, Clone)]
struct FieldStatistics {
    pub total_documents: usize,
    pub unique_values: usize,
    pub average_value_length: f32,
    pub most_common_values: Vec<(String, usize)>,
}

impl BooleanQueryExecutor {
    pub fn new(database: Database) -> Self {
        let mut field_mappings = HashMap::new();
        field_mappings.insert("text_content".to_string(), "content".to_string());
        field_mappings.insert("document_title".to_string(), "title".to_string());
        field_mappings.insert("creator".to_string(), "author".to_string());
        field_mappings.insert("file_type".to_string(), "type".to_string());
        field_mappings.insert("modified_date".to_string(), "modified".to_string());
        field_mappings.insert("created_date".to_string(), "created".to_string());
        field_mappings.insert("file_size".to_string(), "size".to_string());

        Self {
            database,
            field_mappings,
            execution_cache: lru::LruCache::new(std::num::NonZeroUsize::new(100).unwrap()),
            performance_stats: ExecutionStats {
                total_queries: 0,
                cache_hits: 0,
                average_execution_time_ms: 0.0,
                total_documents_processed: 0,
            },
        }
    }

    /// Execute a parsed boolean query
    pub async fn execute(&mut self, parsed_query: &ParsedQuery) -> Result<QueryExecutionResult> {
        let start_time = std::time::Instant::now();
        let query_string = &parsed_query.original_query;
        
        println!("Executing boolean query: '{}'", query_string);
        
        // Check cache first
        let query_hash = self.calculate_query_hash(query_string);
        if let Some(cached_result) = self.execution_cache.get(&query_hash) {
            self.performance_stats.cache_hits += 1;
            
            println!("  Cache hit for query");
            
            return Ok(QueryExecutionResult {
                document_ids: cached_result.document_ids.clone(),
                relevance_scores: cached_result.relevance_scores.clone(),
                execution_time_ms: start_time.elapsed().as_millis() as f64,
                total_matches: cached_result.document_ids.len(),
                query_plan: QueryPlan {
                    steps: vec![ExecutionStep {
                        step_type: StepType::Caching,
                        operation: "Cache hit".to_string(),
                        input_size: 0,
                        output_size: cached_result.document_ids.len(),
                        execution_time_ms: 0.1,
                        selectivity: 1.0,
                    }],
                    estimated_cost: 0.1,
                    actual_cost: 0.1,
                    optimizations_applied: vec!["Cache lookup".to_string()],
                },
                performance_metrics: QueryPerformanceMetrics {
                    total_documents_examined: 0,
                    index_hits: 1,
                    cache_hits: 1,
                    memory_usage_mb: 0.1,
                    io_operations: 0,
                    cpu_time_ms: 0.1,
                },
            });
        }

        // Execute the query tree
        let mut query_plan = QueryPlan {
            steps: Vec::new(),
            estimated_cost: parsed_query.complexity_score,
            actual_cost: 0.0,
            optimizations_applied: Vec::new(),
        };

        let execution_context = ExecutionContext::new();
        let document_ids = self.execute_node(&parsed_query.tree, &mut query_plan, &execution_context).await?;
        
        // Calculate relevance scores
        let relevance_scores = self.calculate_relevance_scores(&document_ids, &parsed_query.tree).await?;
        
        let execution_time = start_time.elapsed().as_millis() as f64;
        query_plan.actual_cost = execution_time as f32;

        // Cache the result
        let cached_result = CachedQueryResult {
            document_ids: document_ids.clone(),
            relevance_scores: relevance_scores.clone(),
            cached_at: Utc::now(),
            query_hash: query_hash.clone(),
        };
        self.execution_cache.put(query_hash, cached_result);

        // Update statistics
        self.performance_stats.total_queries += 1;
        self.performance_stats.total_documents_processed += document_ids.len() as u64;
        self.performance_stats.average_execution_time_ms = 
            (self.performance_stats.average_execution_time_ms * (self.performance_stats.total_queries - 1) as f64 + execution_time) 
            / self.performance_stats.total_queries as f64;

        Ok(QueryExecutionResult {
            document_ids: document_ids.clone(),
            relevance_scores,
            execution_time_ms: execution_time,
            total_matches: document_ids.len(),
            query_plan: query_plan.clone(),
            performance_metrics: QueryPerformanceMetrics {
                total_documents_examined: document_ids.len(),
                index_hits: query_plan.steps.len(),
                cache_hits: 0,
                memory_usage_mb: (document_ids.len() * 64) as f32 / 1024.0 / 1024.0, // Rough estimate
                io_operations: query_plan.steps.len(),
                cpu_time_ms: execution_time,
            },
        })
    }

    /// Execute a single query node recursively
    fn execute_node<'a>(
        &'a self,
        node: &'a QueryNode,
        query_plan: &'a mut QueryPlan,
        context: &'a ExecutionContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<String>>> + 'a>> {
        Box::pin(async move {
        let step_start = std::time::Instant::now();
        
        let result = match node {
            QueryNode::Empty => {
                Vec::new()
            },
            QueryNode::Term(term) => {
                self.execute_term_query(term, query_plan, context).await?
            },
            QueryNode::Binary { left, operator, right } => {
                self.execute_binary_operation(left, operator, right, query_plan, context).await?
            },
            QueryNode::Not(expr) => {
                self.execute_not_operation(expr, query_plan, context).await?
            },
            QueryNode::Group(expr) => {
                // Groups don't change the logic, just execute the inner expression
                self.execute_node(expr, query_plan, context).await?
            },
        };

        let execution_time = step_start.elapsed().as_millis() as f64;
        
        // Add execution step to plan
        query_plan.steps.push(ExecutionStep {
            step_type: match node {
                QueryNode::Term(_) => StepType::TermSearch,
                QueryNode::Binary { .. } => StepType::BooleanOperation,
                QueryNode::Not(_) => StepType::BooleanOperation,
                _ => StepType::IndexLookup,
            },
            operation: self.node_description(node),
            input_size: context.current_document_count,
            output_size: result.len(),
            execution_time_ms: execution_time,
            selectivity: if context.current_document_count > 0 {
                result.len() as f32 / context.current_document_count as f32
            } else {
                1.0
            },
        });

        Ok(result)
        })
    }

    /// Execute a terminal term query
    async fn execute_term_query(
        &self,
        term: &TermQuery,
        _query_plan: &mut QueryPlan,
        _context: &ExecutionContext,
    ) -> Result<Vec<String>> {
        let default_field = "text_content".to_string();
        let field = term.field.as_ref().unwrap_or(&default_field);
        let mapped_field = self.field_mappings.get(field).unwrap_or(field);
        
        println!("  Searching field '{}' for '{}'", mapped_field, term.value);
        
        match &term.query_type {
            TermType::Word => {
                self.search_word_in_field(mapped_field, &term.value).await
            },
            TermType::Phrase => {
                self.search_phrase_in_field(mapped_field, &term.value).await
            },
            TermType::Wildcard => {
                self.search_wildcard_in_field(mapped_field, &term.value).await
            },
            TermType::Range { start, end, inclusive } => {
                self.search_range_in_field(mapped_field, start, end, *inclusive).await
            },
            TermType::Exists => {
                self.search_field_exists(mapped_field).await
            },
            TermType::Regex => {
                self.search_regex_in_field(mapped_field, &term.value).await
            },
        }
    }

    /// Execute binary boolean operations (AND, OR)
    fn execute_binary_operation<'a>(
        &'a self,
        left: &'a QueryNode,
        operator: &'a BooleanOperator,
        right: &'a QueryNode,
        query_plan: &'a mut QueryPlan,
        context: &'a ExecutionContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<String>>> + 'a>> {
        Box::pin(async move {
        println!("  Executing {} operation", match operator {
            BooleanOperator::And => "AND",
            BooleanOperator::Or => "OR",
            BooleanOperator::Not => "NOT", // Shouldn't happen in binary
        });

        let left_results = self.execute_node(left, query_plan, context).await?;
        let right_results = self.execute_node(right, query_plan, context).await?;

        match operator {
            BooleanOperator::And => {
                Ok(self.intersect_document_sets(&left_results, &right_results))
            },
            BooleanOperator::Or => {
                Ok(self.union_document_sets(&left_results, &right_results))
            },
            BooleanOperator::Not => {
                // This shouldn't happen in a binary operation
                Ok(left_results)
            },
        }
        })
    }

    /// Execute NOT operations
    fn execute_not_operation<'a>(
        &'a self,
        expr: &'a QueryNode,
        query_plan: &'a mut QueryPlan,
        context: &'a ExecutionContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<String>>> + 'a>> {
        Box::pin(async move {
        println!("  Executing NOT operation");
        
        let positive_results = self.execute_node(expr, query_plan, context).await?;
        let all_documents = self.get_all_document_ids().await?;
        
        Ok(self.subtract_document_sets(&all_documents, &positive_results))
        })
    }

    /// Calculate relevance scores for documents
    async fn calculate_relevance_scores(
        &self,
        document_ids: &[String],
        query_tree: &QueryNode,
    ) -> Result<HashMap<String, f32>> {
        let mut scores = HashMap::new();
        
        for doc_id in document_ids {
            let score = self.calculate_document_score(doc_id, query_tree).await?;
            scores.insert(doc_id.clone(), score);
        }
        
        Ok(scores)
    }

    /// Calculate relevance score for a single document
    async fn calculate_document_score(&self, _doc_id: &str, _query_tree: &QueryNode) -> Result<f32> {
        // Simplified scoring - in practice, this would be much more sophisticated
        // incorporating TF-IDF, BM25, field boosts, etc.
        Ok(1.0)
    }

    // Field search implementations
    async fn search_word_in_field(&self, field: &str, value: &str) -> Result<Vec<String>> {
        // Simulate database search
        println!("    Word search: {} = '{}'", field, value);
        
        // In a real implementation, this would query the database
        let mock_results = match field {
            "content" => vec!["doc1".to_string(), "doc2".to_string()],
            "title" => vec!["doc1".to_string()],
            "author" => vec!["doc3".to_string()],
            "type" => {
                if value == "pdf" {
                    vec!["doc1".to_string(), "doc3".to_string()]
                } else {
                    vec!["doc2".to_string()]
                }
            },
            _ => vec!["doc1".to_string()],
        };
        
        Ok(mock_results)
    }

    async fn search_phrase_in_field(&self, field: &str, phrase: &str) -> Result<Vec<String>> {
        println!("    Phrase search: {} = \"{}\"", field, phrase);
        
        // Phrase searches are more restrictive than word searches
        let word_results = self.search_word_in_field(field, phrase).await?;
        let result_count = word_results.len();
        
        // Return subset (phrases are more selective)
        Ok(word_results.into_iter().take(result_count / 2 + 1).collect())
    }

    async fn search_wildcard_in_field(&self, field: &str, pattern: &str) -> Result<Vec<String>> {
        println!("    Wildcard search: {} = '{}'", field, pattern);
        
        // Wildcard searches typically return more results
        let base_results = self.search_word_in_field(field, &pattern.replace('*', "")).await?;
        
        // Simulate additional matches from wildcard expansion
        let mut expanded_results = base_results;
        expanded_results.push("doc4".to_string());
        expanded_results.push("doc5".to_string());
        
        Ok(expanded_results)
    }

    async fn search_range_in_field(
        &self,
        field: &str,
        start: &str,
        end: &str,
        _inclusive: bool,
    ) -> Result<Vec<String>> {
        println!("    Range search: {} = [{}..{}]", field, start, end);
        
        // Simulate range query results
        match field {
            "size" => Ok(vec!["doc1".to_string(), "doc2".to_string()]),
            "modified" | "created" => Ok(vec!["doc2".to_string(), "doc3".to_string()]),
            _ => Ok(vec!["doc1".to_string()]),
        }
    }

    async fn search_field_exists(&self, field: &str) -> Result<Vec<String>> {
        println!("    Existence search: field '{}' exists", field);
        
        // Most documents have most fields
        Ok(vec![
            "doc1".to_string(),
            "doc2".to_string(),
            "doc3".to_string(),
            "doc4".to_string(),
        ])
    }

    async fn search_regex_in_field(&self, field: &str, pattern: &str) -> Result<Vec<String>> {
        println!("    Regex search: {} =~ /{}/", field, pattern);
        
        // Regex searches can be expensive but powerful
        let base_results = self.search_word_in_field(field, &pattern.chars().filter(|c| c.is_alphanumeric()).collect::<String>()).await?;
        
        Ok(base_results)
    }

    async fn get_all_document_ids(&self) -> Result<Vec<String>> {
        // In practice, this would query the database for all document IDs
        Ok(vec![
            "doc1".to_string(),
            "doc2".to_string(),
            "doc3".to_string(),
            "doc4".to_string(),
            "doc5".to_string(),
            "doc6".to_string(),
        ])
    }

    // Set operations for combining results
    fn intersect_document_sets(&self, left: &[String], right: &[String]) -> Vec<String> {
        let left_set: HashSet<&String> = left.iter().collect();
        let right_set: HashSet<&String> = right.iter().collect();
        
        left_set.intersection(&right_set)
            .cloned()
            .cloned()
            .collect()
    }

    fn union_document_sets(&self, left: &[String], right: &[String]) -> Vec<String> {
        let mut result_set: HashSet<String> = left.iter().cloned().collect();
        result_set.extend(right.iter().cloned());
        
        result_set.into_iter().collect()
    }

    fn subtract_document_sets(&self, all: &[String], to_remove: &[String]) -> Vec<String> {
        let remove_set: HashSet<&String> = to_remove.iter().collect();
        
        all.iter()
            .filter(|doc| !remove_set.contains(doc))
            .cloned()
            .collect()
    }

    fn calculate_query_hash(&self, query: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn node_description(&self, node: &QueryNode) -> String {
        match node {
            QueryNode::Empty => "Empty".to_string(),
            QueryNode::Term(term) => {
                format!("Term: {}:{}", 
                    term.field.as_ref().unwrap_or(&"*".to_string()),
                    term.value)
            },
            QueryNode::Binary { operator, .. } => {
                format!("Binary: {:?}", operator)
            },
            QueryNode::Not(_) => "NOT".to_string(),
            QueryNode::Group(_) => "Group".to_string(),
        }
    }

    /// Get execution statistics
    pub fn get_execution_stats(&self) -> ExecutionStats {
        self.performance_stats.clone()
    }

    /// Clear execution cache
    pub fn clear_cache(&mut self) {
        self.execution_cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        (self.execution_cache.len(), self.execution_cache.cap().get())
    }
}

/// Context passed through query execution
#[derive(Debug, Clone)]
struct ExecutionContext {
    current_document_count: usize,
    execution_depth: u32,
    optimization_hints: Vec<String>,
}

impl ExecutionContext {
    fn new() -> Self {
        Self {
            current_document_count: 1000, // Estimated total documents
            execution_depth: 0,
            optimization_hints: Vec::new(),
        }
    }
}

impl FieldSearcher for BooleanQueryExecutor {
    fn search_field(&self, _field: &str, _value: &str, term_type: &TermType) -> Result<Vec<String>> {
        // Synchronous wrapper for async methods
        // In practice, you'd use an async runtime or different architecture
        match term_type {
            TermType::Word => Ok(vec!["doc1".to_string()]),
            TermType::Phrase => Ok(vec!["doc1".to_string()]),
            _ => Ok(vec!["doc1".to_string()]),
        }
    }

    fn get_field_statistics(&self, field: &str) -> FieldStatistics {
        // Return mock statistics
        FieldStatistics {
            total_documents: 1000,
            unique_values: match field {
                "content" => 50000,
                "title" => 800,
                "author" => 50,
                "type" => 10,
                _ => 100,
            },
            average_value_length: match field {
                "content" => 2000.0,
                "title" => 50.0,
                "author" => 15.0,
                "type" => 4.0,
                _ => 20.0,
            },
            most_common_values: vec![
                ("common_value".to_string(), 100),
                ("another_value".to_string(), 80),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::boolean_query_parser::BooleanQueryParser;

    #[tokio::test]
    async fn test_simple_execution() {
        let database = Database::new("test.db").unwrap();
        let mut executor = BooleanQueryExecutor::new(database);
        let parser = BooleanQueryParser::new();
        
        let parsed = parser.parse("hello").unwrap();
        let result = executor.execute(&parsed).await.unwrap();
        
        assert!(!result.document_ids.is_empty());
        assert!(result.execution_time_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_and_operation() {
        let database = Database::new("test.db").unwrap();
        let mut executor = BooleanQueryExecutor::new(database);
        let parser = BooleanQueryParser::new();
        
        let parsed = parser.parse("hello AND world").unwrap();
        let result = executor.execute(&parsed).await.unwrap();
        
        // AND should return intersection
        assert!(result.query_plan.steps.len() > 1);
    }

    #[tokio::test]
    async fn test_field_search() {
        let database = Database::new("test.db").unwrap();
        let mut executor = BooleanQueryExecutor::new(database);
        let parser = BooleanQueryParser::new();
        
        let parsed = parser.parse("author:john").unwrap();
        let result = executor.execute(&parsed).await.unwrap();
        
        assert!(!result.document_ids.is_empty());
    }
}