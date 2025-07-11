use crate::error::{AppError, SearchError, Result};
use crate::database::{Database, operations::{SearchResult as DatabaseSearchResult, SimilarChunk}};
use crate::core::embedding_manager::EmbeddingManager;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use regex::Regex;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq)]
pub enum SearchMode {
    /// Precise: Prefers exact keyword matches, minimal semantic expansion
    Precise,
    /// Balanced: Equal weight between keyword and semantic search
    Balanced,
    /// Exploratory: Emphasizes semantic similarity, conceptual exploration
    Exploratory,
}

impl Default for SearchMode {
    fn default() -> Self {
        SearchMode::Balanced
    }
}

#[derive(Debug, Clone)]
pub struct QueryAnalysis {
    /// Raw query string
    pub query: String,
    /// Detected keywords/phrases for FTS search
    pub keywords: Vec<String>,
    /// Boolean operators detected (AND, OR, NOT)
    pub has_boolean_operators: bool,
    /// Quoted phrases for exact matching
    pub quoted_phrases: Vec<String>,
    /// Estimated complexity (affects search strategy)
    pub complexity_score: f32,
    /// Suggested search mode based on query characteristics
    pub suggested_mode: SearchMode,
    /// Whether query appears to be seeking specific factual information
    pub is_factual_query: bool,
    /// Whether query appears to be exploratory/conceptual
    pub is_conceptual_query: bool,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk_id: String,
    pub document_id: Uuid,
    pub content: String,
    pub relevance_score: f32,
    pub source: SearchResultSource,
    pub highlighted_content: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SearchResultSource {
    FullTextSearch,
    SemanticSearch,
    HybridFusion,
}

#[derive(Debug, Clone)]
pub struct HybridSearchConfig {
    /// FTS weight in final scoring (0.0 - 1.0)
    pub fts_weight: f32,
    /// Vector search weight in final scoring (0.0 - 1.0)  
    pub vector_weight: f32,
    /// Minimum similarity threshold for vector search
    pub vector_threshold: f32,
    /// Maximum results per search type before fusion
    pub max_results_per_type: usize,
    /// Final result limit after fusion
    pub final_result_limit: usize,
    /// Whether to enable result caching
    pub enable_caching: bool,
    /// RRF rank fusion parameter (typically 60)
    pub rrf_k: f32,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            fts_weight: 0.4,
            vector_weight: 0.6,
            vector_threshold: 0.3,
            max_results_per_type: 20,
            final_result_limit: 10,
            enable_caching: true,
            rrf_k: 60.0,
        }
    }
}

impl HybridSearchConfig {
    pub fn for_mode(mode: SearchMode) -> Self {
        match mode {
            SearchMode::Precise => Self {
                fts_weight: 0.8,
                vector_weight: 0.2,
                vector_threshold: 0.5,
                max_results_per_type: 15,
                final_result_limit: 8,
                rrf_k: 60.0,
                ..Default::default()
            },
            SearchMode::Balanced => Self::default(),
            SearchMode::Exploratory => Self {
                fts_weight: 0.2,
                vector_weight: 0.8,
                vector_threshold: 0.2,
                max_results_per_type: 25,
                final_result_limit: 12,
                rrf_k: 60.0,
                ..Default::default()
            },
        }
    }
}

pub struct HybridSearchEngine {
    database: Arc<Database>,
    embedding_manager: Option<Arc<Mutex<EmbeddingManager>>>,
    config: HybridSearchConfig,
    query_analyzer: QueryAnalyzer,
    result_cache: Option<lru::LruCache<String, Vec<SearchResult>>>,
}

impl HybridSearchEngine {
    pub fn new(database: Arc<Database>) -> Self {
        Self {
            database,
            embedding_manager: None,
            config: HybridSearchConfig::default(),
            query_analyzer: QueryAnalyzer::new(),
            result_cache: Some(lru::LruCache::new(std::num::NonZeroUsize::new(100).unwrap())),
        }
    }

    pub fn with_config(mut self, config: HybridSearchConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn set_embedding_manager(&mut self, embedding_manager: Arc<Mutex<EmbeddingManager>>) {
        self.embedding_manager = Some(embedding_manager);
    }

    pub fn database(&self) -> &Arc<Database> {
        &self.database
    }

    pub fn set_mode(&mut self, mode: SearchMode) {
        self.config = HybridSearchConfig::for_mode(mode);
    }

    /// Main search entry point - automatically routes based on query analysis
    pub async fn search(&mut self, query: &str) -> Result<Vec<SearchResult>> {
        // Analyze query to determine optimal search strategy
        let analysis = self.query_analyzer.analyze_query(query);
        
        // Check cache first if enabled
        if self.config.enable_caching {
            if let Some(ref mut cache) = self.result_cache {
                let cache_key = format!("{}:{:?}", query, analysis.suggested_mode);
                if let Some(cached_results) = cache.get(&cache_key) {
                    return Ok(cached_results.clone());
                }
            }
        }

        // Adjust config based on query analysis
        let search_config = HybridSearchConfig::for_mode(analysis.suggested_mode.clone());
        
        let results = match analysis.suggested_mode {
            SearchMode::Precise if analysis.has_boolean_operators || !analysis.quoted_phrases.is_empty() => {
                // Use primarily FTS for precise, structured queries
                self.search_fts_primary(&analysis, &search_config).await?
            }
            SearchMode::Exploratory if analysis.is_conceptual_query => {
                // Use primarily semantic search for conceptual queries  
                self.search_semantic_primary(&analysis, &search_config).await?
            }
            _ => {
                // Use hybrid fusion for balanced queries
                self.search_hybrid_fusion(&analysis, &search_config).await?
            }
        };

        // Cache results if enabled
        if self.config.enable_caching {
            if let Some(ref mut cache) = self.result_cache {
                let cache_key = format!("{}:{:?}", query, analysis.suggested_mode);
                cache.put(cache_key, results.clone());
            }
        }

        Ok(results)
    }

    /// Search with explicit mode override
    pub async fn search_with_mode(&mut self, query: &str, mode: SearchMode) -> Result<Vec<SearchResult>> {
        let mut analysis = self.query_analyzer.analyze_query(query);
        analysis.suggested_mode = mode;
        
        let search_config = HybridSearchConfig::for_mode(analysis.suggested_mode.clone());
        
        match analysis.suggested_mode {
            SearchMode::Precise => self.search_fts_primary(&analysis, &search_config).await,
            SearchMode::Exploratory => self.search_semantic_primary(&analysis, &search_config).await,
            SearchMode::Balanced => self.search_hybrid_fusion(&analysis, &search_config).await,
        }
    }

    /// FTS-primary search for precise queries
    async fn search_fts_primary(&self, analysis: &QueryAnalysis, config: &HybridSearchConfig) -> Result<Vec<SearchResult>> {
        // Build FTS query from analysis
        let fts_query = self.build_fts_query(analysis);
        
        // Execute FTS search
        let fts_results = self.database.search_documents(&fts_query, config.max_results_per_type)?;
        
        // Convert to SearchResult format
        let mut results: Vec<SearchResult> = fts_results.into_iter().map(|result| {
            SearchResult {
                chunk_id: result.chunk_id,
                document_id: result.document_id,
                content: result.content,
                relevance_score: result.relevance_score * config.fts_weight,
                source: SearchResultSource::FullTextSearch,
                highlighted_content: result.highlighted_content,
            }
        }).collect();

        // Optionally enhance with semantic results if embedding manager available
        if let Some(ref embedding_manager) = self.embedding_manager {
            if let Ok(semantic_results) = self.search_semantic(&analysis.query, config.max_results_per_type / 2, embedding_manager).await {
                // Add semantic results with lower weight
                for mut result in semantic_results {
                    result.relevance_score *= config.vector_weight * 0.3; // Reduced weight for FTS-primary mode
                    result.source = SearchResultSource::SemanticSearch;
                    results.push(result);
                }
            }
        }

        // Sort by relevance and limit
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(config.final_result_limit);

        Ok(results)
    }

    /// Semantic-primary search for exploratory queries
    async fn search_semantic_primary(&self, analysis: &QueryAnalysis, config: &HybridSearchConfig) -> Result<Vec<SearchResult>> {
        let Some(ref embedding_manager) = self.embedding_manager else {
            // Fallback to FTS if no embedding manager
            return self.search_fts_primary(analysis, config).await;
        };

        // Execute semantic search
        let mut results = self.search_semantic(&analysis.query, config.max_results_per_type, embedding_manager).await?;
        
        // Apply vector weight
        for result in &mut results {
            result.relevance_score *= config.vector_weight;
            result.source = SearchResultSource::SemanticSearch;
        }

        // Optionally enhance with FTS results
        let fts_query = self.build_fts_query(analysis);
        if let Ok(fts_results) = self.database.search_documents(&fts_query, config.max_results_per_type / 2) {
            for fts_result in fts_results {
                let mut result = SearchResult {
                    chunk_id: fts_result.chunk_id,
                    document_id: fts_result.document_id,
                    content: fts_result.content,
                    relevance_score: fts_result.relevance_score * config.fts_weight * 0.3, // Reduced weight
                    source: SearchResultSource::FullTextSearch,
                    highlighted_content: fts_result.highlighted_content,
                };
                results.push(result);
            }
        }

        // Sort by relevance and limit
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(config.final_result_limit);

        Ok(results)
    }

    /// Hybrid fusion using Reciprocal Rank Fusion (RRF)
    async fn search_hybrid_fusion(&self, analysis: &QueryAnalysis, config: &HybridSearchConfig) -> Result<Vec<SearchResult>> {
        // Execute both search types in parallel
        let fts_query = self.build_fts_query(analysis);
        let fts_future = self.database.search_documents(&fts_query, config.max_results_per_type);
        
        let semantic_future = if let Some(ref embedding_manager) = self.embedding_manager {
            Some(self.search_semantic(&analysis.query, config.max_results_per_type, embedding_manager))
        } else {
            None
        };

        // Collect FTS results
        let fts_results = fts_future?;
        
        // Collect semantic results
        let semantic_results = if let Some(future) = semantic_future {
            future.await.unwrap_or_else(|_| Vec::new())
        } else {
            Vec::new()
        };

        // Apply Reciprocal Rank Fusion
        let fused_results = self.apply_rrf(
            fts_results,
            semantic_results,
            config.rrf_k,
            config.fts_weight,
            config.vector_weight,
        );

        // Limit final results
        let mut final_results = fused_results;
        final_results.truncate(config.final_result_limit);

        Ok(final_results)
    }

    /// Execute semantic search using embedding manager
    async fn search_semantic(&self, query: &str, limit: usize, embedding_manager: &Arc<Mutex<EmbeddingManager>>) -> Result<Vec<SearchResult>> {
        // Generate query embedding
        let manager = embedding_manager.lock().await;
        let query_embeddings = manager.generate_embeddings(&[query.to_string()]).await
            .map_err(|e| AppError::Search(SearchError::VectorSearch(e.to_string())))?;
        drop(manager);

        let Some(query_embedding) = query_embeddings.first() else {
            return Ok(Vec::new());
        };

        // Search for similar chunks
        let similar_chunks = self.database.find_similar_chunks(
            query_embedding,
            limit,
            self.config.vector_threshold,
        )?;

        // Convert to SearchResult format
        let results = similar_chunks.into_iter().map(|chunk| {
            SearchResult {
                chunk_id: chunk.chunk_id,
                document_id: chunk.document_id,
                content: chunk.content,
                relevance_score: chunk.similarity_score,
                source: SearchResultSource::SemanticSearch,
                highlighted_content: None, // TODO: Implement semantic highlighting
            }
        }).collect();

        Ok(results)
    }

    /// Build FTS query string from query analysis
    fn build_fts_query(&self, analysis: &QueryAnalysis) -> String {
        let mut query_parts = Vec::new();

        // Add quoted phrases with high priority (exact phrase search)
        for phrase in &analysis.quoted_phrases {
            query_parts.push(format!("\"{}\"", phrase));
        }

        // Add keywords with proper FTS5 syntax
        if !analysis.keywords.is_empty() {
            if analysis.has_boolean_operators {
                // Use keywords as-is if boolean operators detected
                query_parts.push(analysis.keywords.join(" "));
            } else {
                // For natural language queries like "machine learning", 
                // try both phrase search and AND search
                if analysis.keywords.len() == 2 && !analysis.is_factual_query {
                    // For two-word terms, try phrase first, then individual terms
                    let phrase = analysis.keywords.join(" ");
                    let individual_terms = analysis.keywords.join(" AND ");
                    query_parts.push(format!("\"{}\" OR ({})", phrase, individual_terms));
                } else if analysis.keywords.len() <= 3 {
                    // For short queries, use phrase search first
                    let phrase = analysis.keywords.join(" ");
                    query_parts.push(format!("\"{}\"", phrase));
                } else {
                    // For longer queries, use AND logic
                    let keyword_query = analysis.keywords.join(" AND ");
                    query_parts.push(keyword_query);
                }
            }
        }

        // Fallback to simple phrase search for original query
        if query_parts.is_empty() {
            // For simple queries, use phrase search
            if analysis.query.split_whitespace().count() <= 3 {
                query_parts.push(format!("\"{}\"", analysis.query));
            } else {
                query_parts.push(analysis.query.clone());
            }
        }

        query_parts.join(" ")
    }

    /// Apply Reciprocal Rank Fusion to combine FTS and semantic results
    fn apply_rrf(
        &self,
        fts_results: Vec<DatabaseSearchResult>,
        semantic_results: Vec<SearchResult>,
        k: f32,
        fts_weight: f32,
        vector_weight: f32,
    ) -> Vec<SearchResult> {
        let mut score_map: HashMap<String, (SearchResult, f32)> = HashMap::new();

        // Process FTS results
        for (rank, fts_result) in fts_results.into_iter().enumerate() {
            let rrf_score = fts_weight / (k + (rank + 1) as f32);
            let search_result = SearchResult {
                chunk_id: fts_result.chunk_id.clone(),
                document_id: fts_result.document_id,
                content: fts_result.content,
                relevance_score: fts_result.relevance_score,
                source: SearchResultSource::HybridFusion,
                highlighted_content: fts_result.highlighted_content,
            };
            
            score_map.insert(fts_result.chunk_id, (search_result, rrf_score));
        }

        // Process semantic results
        for (rank, semantic_result) in semantic_results.into_iter().enumerate() {
            let rrf_score = vector_weight / (k + (rank + 1) as f32);
            
            if let Some((existing_result, existing_score)) = score_map.get_mut(&semantic_result.chunk_id) {
                // Combine scores if chunk exists in both result sets
                *existing_score += rrf_score;
                existing_result.source = SearchResultSource::HybridFusion;
                // Keep the higher individual relevance score
                if semantic_result.relevance_score > existing_result.relevance_score {
                    existing_result.relevance_score = semantic_result.relevance_score;
                }
            } else {
                // Add new result
                let mut search_result = semantic_result;
                search_result.source = SearchResultSource::HybridFusion;
                score_map.insert(search_result.chunk_id.clone(), (search_result, rrf_score));
            }
        }

        // Convert to final results and sort by RRF score
        let mut final_results: Vec<SearchResult> = score_map.into_iter()
            .map(|(_, (mut result, rrf_score))| {
                result.relevance_score = rrf_score;
                result
            })
            .collect();

        final_results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        final_results
    }

    pub fn clear_cache(&mut self) {
        if let Some(ref mut cache) = self.result_cache {
            cache.clear();
        }
    }

    pub fn get_cache_stats(&self) -> (usize, usize) {
        if let Some(ref cache) = self.result_cache {
            (cache.len(), cache.cap().get())
        } else {
            (0, 0)
        }
    }
}

/// Query analysis for determining optimal search strategy
pub struct QueryAnalyzer {
    boolean_operators: Regex,
    quoted_phrases: Regex,
    question_words: Vec<&'static str>,
    factual_indicators: Vec<&'static str>,
    conceptual_indicators: Vec<&'static str>,
}

impl QueryAnalyzer {
    pub fn new() -> Self {
        Self {
            boolean_operators: Regex::new(r"\b(AND|OR|NOT)\b").unwrap(),
            quoted_phrases: Regex::new(r#""([^"]*)""#).unwrap(),
            question_words: vec!["what", "when", "where", "who", "why", "how", "which"],
            factual_indicators: vec!["define", "definition", "meaning", "explain", "describe", "list", "name"],
            conceptual_indicators: vec!["similar", "like", "related", "about", "concept", "idea", "theory", "approach"],
        }
    }

    pub fn analyze_query(&self, query: &str) -> QueryAnalysis {
        let query_lower = query.to_lowercase();
        
        // Extract quoted phrases
        let quoted_phrases: Vec<String> = self.quoted_phrases
            .captures_iter(query)
            .map(|cap| cap[1].to_string())
            .collect();

        // Check for boolean operators
        let has_boolean_operators = self.boolean_operators.is_match(query);

        // Extract keywords (excluding quoted content and boolean operators)
        let mut keywords = Vec::new();
        let mut remaining_query = query.to_string();
        
        // Remove quoted phrases
        for phrase in &quoted_phrases {
            remaining_query = remaining_query.replace(&format!("\"{}\"", phrase), "");
        }
        
        // Remove boolean operators and extract meaningful words
        let keyword_text = self.boolean_operators.replace_all(&remaining_query, " ");
        keywords.extend(
            keyword_text
                .split_whitespace()
                .filter(|word| word.len() > 2 && !word.chars().all(|c| !c.is_alphabetic()))
                .map(|word| word.to_lowercase())
        );

        // Determine query characteristics
        let is_factual_query = self.factual_indicators.iter()
            .any(|&indicator| query_lower.contains(indicator)) ||
            self.question_words.iter()
                .any(|&word| query_lower.starts_with(word));

        let is_conceptual_query = self.conceptual_indicators.iter()
            .any(|&indicator| query_lower.contains(indicator)) ||
            (!is_factual_query && keywords.len() >= 3);

        // Calculate complexity score
        let mut complexity_score = 0.0;
        complexity_score += keywords.len() as f32 * 0.1;
        complexity_score += quoted_phrases.len() as f32 * 0.3;
        if has_boolean_operators { complexity_score += 0.4; }
        if is_factual_query { complexity_score += 0.2; }
        if is_conceptual_query { complexity_score += 0.3; }
        
        // Determine suggested search mode
        let suggested_mode = if has_boolean_operators || !quoted_phrases.is_empty() || 
                               (is_factual_query && complexity_score < 0.5) {
            SearchMode::Precise
        } else if is_conceptual_query || complexity_score > 0.8 {
            SearchMode::Exploratory
        } else {
            SearchMode::Balanced
        };

        QueryAnalysis {
            query: query.to_string(),
            keywords,
            has_boolean_operators,
            quoted_phrases,
            complexity_score,
            suggested_mode,
            is_factual_query,
            is_conceptual_query,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_analysis_factual() {
        let analyzer = QueryAnalyzer::new();
        
        let analysis = analyzer.analyze_query("What is machine learning?");
        assert!(analysis.is_factual_query);
        assert_eq!(analysis.suggested_mode, SearchMode::Precise);
        assert!(analysis.keywords.contains(&"machine".to_string()));
        assert!(analysis.keywords.contains(&"learning".to_string()));
    }

    #[test]
    fn test_query_analysis_conceptual() {
        let analyzer = QueryAnalyzer::new();
        
        let analysis = analyzer.analyze_query("similar approaches to neural network training");
        assert!(analysis.is_conceptual_query);
        assert_eq!(analysis.suggested_mode, SearchMode::Exploratory);
        assert!(analysis.keywords.len() >= 3);
    }

    #[test]
    fn test_query_analysis_boolean() {
        let analyzer = QueryAnalyzer::new();
        
        let analysis = analyzer.analyze_query("machine learning AND neural networks");
        assert!(analysis.has_boolean_operators);
        assert_eq!(analysis.suggested_mode, SearchMode::Precise);
    }

    #[test]
    fn test_query_analysis_quoted() {
        let analyzer = QueryAnalyzer::new();
        
        let analysis = analyzer.analyze_query("\"deep learning\" applications");
        assert_eq!(analysis.quoted_phrases, vec!["deep learning"]);
        assert_eq!(analysis.suggested_mode, SearchMode::Precise);
    }

    #[test]
    fn test_query_analysis_balanced() {
        let analyzer = QueryAnalyzer::new();
        
        let analysis = analyzer.analyze_query("machine learning algorithms");
        assert!(!analysis.is_factual_query);
        assert!(!analysis.is_conceptual_query);
        assert_eq!(analysis.suggested_mode, SearchMode::Balanced);
    }
}