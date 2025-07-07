use crate::error::{AppError, Result};
use crate::core::query_intent::QueryIntent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;

/// Feature vector for learning-to-rank model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingFeatures {
    /// Text-based BM25 relevance score
    pub text_bm25: f32,
    /// Vector cosine similarity score
    pub cosine_vec: f32,
    /// Recency decay factor (1.0 = today, approaches 0)
    pub recency_decay: f32,
    /// User interaction frequency with this document
    pub user_frequency: f32,
    /// Document quality score based on completeness, formatting
    pub doc_quality: f32,
    /// Boolean: is this document in the same project as current context
    pub same_project_flag: f32,
    /// Diversity penalty (similarity to already-selected results)
    pub diversity_penalty: f32,
    /// Query-document intent alignment score
    pub intent_alignment: f32,
    /// File type preference score for this query type
    pub type_preference: f32,
    /// Document size normalization factor
    pub size_factor: f32,
}

/// Result with ranking information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedResult {
    pub document_id: Uuid,
    pub chunk_id: Option<Uuid>,
    pub title: String,
    pub snippet: String,
    pub file_path: String,
    pub file_type: String,
    pub relevance_score: f32,
    pub features: RankingFeatures,
    pub ranking_explanation: Option<RankingExplanation>,
}

/// Explanation of why a result was ranked at this position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingExplanation {
    pub primary_signals: Vec<(String, f32)>,
    pub feature_contributions: HashMap<String, f32>,
    pub ranking_model: String,
    pub confidence: f32,
}

/// Configuration for ranking algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingConfig {
    /// Recency decay time constant (days)
    pub recency_tau: f32,
    /// Diversity lambda parameter for MMR
    pub diversity_lambda: f32,
    /// Feature weights for linear combination fallback
    pub feature_weights: HashMap<String, f32>,
    /// Model-specific parameters
    pub model_params: HashMap<String, serde_json::Value>,
    /// Enable learning-to-rank model
    pub enable_ltr: bool,
    /// Enable diversity selection
    pub enable_diversity: bool,
}

impl Default for RankingConfig {
    fn default() -> Self {
        let mut feature_weights = HashMap::new();
        feature_weights.insert("text_bm25".to_string(), 0.4);
        feature_weights.insert("cosine_vec".to_string(), 0.3);
        feature_weights.insert("recency_decay".to_string(), 0.15);
        feature_weights.insert("user_frequency".to_string(), 0.1);
        feature_weights.insert("doc_quality".to_string(), 0.05);

        Self {
            recency_tau: 30.0, // 30-day decay
            diversity_lambda: 0.5,
            feature_weights,
            model_params: HashMap::new(),
            enable_ltr: true,
            enable_diversity: true,
        }
    }
}

/// User interaction tracking for personalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInteraction {
    pub document_id: Uuid,
    pub interaction_type: InteractionType,
    pub timestamp: DateTime<Utc>,
    pub query_context: Option<String>,
    pub session_id: Uuid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InteractionType {
    /// User clicked on the result
    Click,
    /// User opened the document
    Open,
    /// User spent significant time viewing the document
    Dwell,
    /// User bookmarked or starred the document
    Bookmark,
    /// User shared or copied the document
    Share,
}

/// Context information for current search session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchContext {
    pub session_id: Uuid,
    pub current_project: Option<String>,
    pub recent_documents: Vec<Uuid>,
    pub active_applications: Vec<String>,
    pub search_history: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

/// Advanced ranking system with learning-to-rank capabilities
pub struct AdvancedRanker {
    config: RankingConfig,
    user_interactions: HashMap<Uuid, Vec<UserInteraction>>,
    document_stats: HashMap<Uuid, DocumentStats>,
    current_context: Option<SearchContext>,
    // LTR model would be loaded here (placeholder for now)
    ltr_model: Option<Box<dyn LTRModel>>,
}

/// Document statistics for quality scoring
#[derive(Debug, Clone, Default)]
pub struct DocumentStats {
    pub total_views: u32,
    pub unique_viewers: u32,
    pub average_dwell_time: f32,
    pub bookmark_count: u32,
    pub share_count: u32,
    pub last_accessed: Option<DateTime<Utc>>,
    pub quality_score: f32,
}

/// Trait for learning-to-rank models
pub trait LTRModel: Send + Sync {
    fn predict(&self, features: &RankingFeatures) -> Result<f32>;
    fn predict_batch(&self, features: &[RankingFeatures]) -> Result<Vec<f32>>;
    fn explain(&self, features: &RankingFeatures) -> Result<RankingExplanation>;
}

/// Simple linear model implementation for cold-start
pub struct LinearRankingModel {
    weights: HashMap<String, f32>,
    bias: f32,
}

impl LinearRankingModel {
    pub fn new(weights: HashMap<String, f32>) -> Self {
        Self {
            weights,
            bias: 0.0,
        }
    }

    pub fn from_config(config: &RankingConfig) -> Self {
        Self::new(config.feature_weights.clone())
    }
}

impl LTRModel for LinearRankingModel {
    fn predict(&self, features: &RankingFeatures) -> Result<f32> {
        let score = self.weights.get("text_bm25").unwrap_or(&0.0) * features.text_bm25
            + self.weights.get("cosine_vec").unwrap_or(&0.0) * features.cosine_vec
            + self.weights.get("recency_decay").unwrap_or(&0.0) * features.recency_decay
            + self.weights.get("user_frequency").unwrap_or(&0.0) * features.user_frequency
            + self.weights.get("doc_quality").unwrap_or(&0.0) * features.doc_quality
            + self.weights.get("same_project_flag").unwrap_or(&0.0) * features.same_project_flag
            + self.weights.get("intent_alignment").unwrap_or(&0.0) * features.intent_alignment
            + self.weights.get("type_preference").unwrap_or(&0.0) * features.type_preference
            + self.bias;

        Ok(score.max(0.0).min(1.0)) // Clamp to [0, 1]
    }

    fn predict_batch(&self, features: &[RankingFeatures]) -> Result<Vec<f32>> {
        features.iter().map(|f| self.predict(f)).collect()
    }

    fn explain(&self, features: &RankingFeatures) -> Result<RankingExplanation> {
        let mut contributions = HashMap::new();
        
        contributions.insert("text_bm25".to_string(), 
            self.weights.get("text_bm25").unwrap_or(&0.0) * features.text_bm25);
        contributions.insert("cosine_vec".to_string(), 
            self.weights.get("cosine_vec").unwrap_or(&0.0) * features.cosine_vec);
        contributions.insert("recency_decay".to_string(), 
            self.weights.get("recency_decay").unwrap_or(&0.0) * features.recency_decay);
        contributions.insert("user_frequency".to_string(), 
            self.weights.get("user_frequency").unwrap_or(&0.0) * features.user_frequency);

        // Find top contributing features
        let mut primary_signals: Vec<(String, f32)> = contributions.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        primary_signals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        primary_signals.truncate(3);

        Ok(RankingExplanation {
            primary_signals,
            feature_contributions: contributions,
            ranking_model: "LinearModel".to_string(),
            confidence: 0.7, // Fixed confidence for linear model
        })
    }
}

impl AdvancedRanker {
    pub fn new(config: RankingConfig) -> Self {
        let ltr_model: Option<Box<dyn LTRModel>> = if config.enable_ltr {
            Some(Box::new(LinearRankingModel::from_config(&config)))
        } else {
            None
        };

        Self {
            config,
            user_interactions: HashMap::new(),
            document_stats: HashMap::new(),
            current_context: None,
            ltr_model,
        }
    }

    /// Extract ranking features for a document given query context
    pub fn extract_features(
        &self,
        document_id: &Uuid,
        query_intent: &QueryIntent,
        text_score: f32,
        vector_score: f32,
        document_timestamp: DateTime<Utc>,
        file_type: &str,
        file_size: u64,
    ) -> Result<RankingFeatures> {
        let now = Utc::now();
        
        // Calculate recency decay
        let days_old = (now - document_timestamp).num_days() as f32;
        let recency_decay = (-days_old / self.config.recency_tau).exp();

        // Calculate user frequency
        let user_frequency = self.user_interactions
            .get(document_id)
            .map(|interactions| {
                let recent_interactions = interactions.iter()
                    .filter(|i| (now - i.timestamp) < Duration::days(30))
                    .count() as f32;
                (recent_interactions / 10.0).min(1.0) // Normalize to [0, 1]
            })
            .unwrap_or(0.0);

        // Get document quality score
        let doc_quality = self.document_stats
            .get(document_id)
            .map(|stats| stats.quality_score)
            .unwrap_or(0.5); // Default neutral quality

        // Calculate same project flag
        let same_project_flag = self.current_context
            .as_ref()
            .and_then(|ctx| ctx.current_project.as_ref())
            .map(|_| 1.0) // Simplified - would check actual project membership
            .unwrap_or(0.0);

        // Calculate intent alignment (simplified)
        let intent_alignment = self.calculate_intent_alignment(query_intent, file_type);

        // Calculate type preference based on query intent
        let type_preference = self.calculate_type_preference(query_intent, file_type);

        // Size factor (normalize file size)
        let size_factor = self.normalize_file_size(file_size);

        Ok(RankingFeatures {
            text_bm25: text_score,
            cosine_vec: vector_score,
            recency_decay,
            user_frequency,
            doc_quality,
            same_project_flag,
            diversity_penalty: 0.0, // Will be calculated during MMR
            intent_alignment,
            type_preference,
            size_factor,
        })
    }

    /// Calculate alignment between query intent and document characteristics
    fn calculate_intent_alignment(&self, query_intent: &QueryIntent, file_type: &str) -> f32 {
        use crate::core::query_intent::Intent;
        
        let mut weighted_alignment = 0.0;
        let mut total_confidence = 0.0;
        
        if query_intent.intents.is_empty() {
            return 0.5; // Neutral alignment
        }

        for (intent, confidence) in &query_intent.intents {
            let type_match = match intent {
                Intent::DocumentSearch => match file_type {
                    "pdf" | "doc" | "docx" | "txt" => 0.95,
                    "md" | "rtf" => 0.85,
                    "jpg" | "png" | "jpeg" => 0.3,
                    _ => 0.4,
                },
                Intent::TypeSearch => {
                    // For type search, check if the query mentions this file type
                    let query_lower = query_intent.original_query.to_lowercase();
                    if query_lower.contains(file_type) || 
                       query_lower.contains(&file_type.to_uppercase()) ||
                       (file_type == "pdf" && (query_lower.contains("pdf") || query_lower.contains("document"))) ||
                       (file_type == "doc" && query_lower.contains("document")) ||
                       (file_type == "docx" && query_lower.contains("document")) {
                        1.0
                    } else {
                        0.6
                    }
                },
                Intent::QuestionAnswering => match file_type {
                    "pdf" | "doc" | "docx" | "txt" | "md" => 0.9,
                    "rtf" => 0.7,
                    _ => 0.3,
                },
                Intent::SimilaritySearch => 0.8, // Good for all document types
                Intent::PersonSearch => match file_type {
                    "msg" | "eml" => 0.95,
                    "pdf" | "doc" | "docx" => 0.8,
                    _ => 0.4,
                },
                Intent::ContentSearch => match file_type {
                    "pdf" | "doc" | "docx" | "txt" | "md" => 0.9,
                    _ => 0.5,
                },
                Intent::TemporalSearch => 0.7, // Temporal queries work for all file types
                _ => 0.6, // Default for other intents
            };
            
            weighted_alignment += confidence * type_match;
            total_confidence += confidence;
        }

        if total_confidence > 0.0 {
            (weighted_alignment / total_confidence).min(1.0)
        } else {
            0.5
        }
    }

    /// Calculate type preference based on query characteristics
    fn calculate_type_preference(&self, query_intent: &QueryIntent, file_type: &str) -> f32 {
        // Simplified type preference calculation
        let base_preference = match file_type {
            "pdf" => 0.8,  // Generally preferred for document searches
            "doc" | "docx" => 0.7,
            "txt" | "md" => 0.6,
            "jpg" | "png" => 0.5,
            _ => 0.4,
        };

        // Boost preference if file type mentioned in query
        let query_text = &query_intent.original_query.to_lowercase();
        let type_mentioned = query_text.contains(file_type) || 
            query_text.contains(&file_type.to_uppercase());
        
        if type_mentioned {
            (base_preference + 0.3f32).min(1.0)
        } else {
            base_preference
        }
    }

    /// Normalize file size to a 0-1 factor
    fn normalize_file_size(&self, file_size: u64) -> f32 {
        // Logarithmic normalization - prefer medium-sized files
        let size_mb = file_size as f32 / (1024.0 * 1024.0);
        let log_size = (size_mb + 1.0).ln();
        
        // Peak preference around 1-10MB (log scale ~0-2.3)
        let optimal_log_size = 2.0;
        let distance = (log_size - optimal_log_size).abs();
        
        // Gaussian-like preference
        (-distance * distance / 4.0).exp()
    }

    /// Rank results using the configured model
    pub async fn rank_results(
        &self,
        mut results: Vec<RankedResult>,
        query_intent: &QueryIntent,
    ) -> Result<Vec<RankedResult>> {
        if results.is_empty() {
            return Ok(results);
        }

        // Extract features for all results
        let features: Vec<RankingFeatures> = results.iter()
            .map(|r| r.features.clone())
            .collect();

        // Get model predictions
        let scores = if let Some(ref model) = self.ltr_model {
            model.predict_batch(&features)?
        } else {
            // Fallback to simple linear combination
            features.iter()
                .map(|f| self.linear_combination_score(f))
                .collect::<Result<Vec<_>>>()?
        };

        // Update relevance scores
        for (result, score) in results.iter_mut().zip(scores.iter()) {
            result.relevance_score = *score;
            
            // Add explanation if requested
            if let Some(ref model) = self.ltr_model {
                result.ranking_explanation = model.explain(&result.features).ok();
            }
        }

        // Apply diversity selection if enabled
        if self.config.enable_diversity {
            results = self.apply_mmr_diversity(results, query_intent).await?;
        }

        // Sort by final relevance score
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    /// Simple linear combination fallback scoring
    fn linear_combination_score(&self, features: &RankingFeatures) -> Result<f32> {
        let score = self.config.feature_weights.get("text_bm25").unwrap_or(&0.0) * features.text_bm25
            + self.config.feature_weights.get("cosine_vec").unwrap_or(&0.0) * features.cosine_vec
            + self.config.feature_weights.get("recency_decay").unwrap_or(&0.0) * features.recency_decay
            + self.config.feature_weights.get("user_frequency").unwrap_or(&0.0) * features.user_frequency
            + self.config.feature_weights.get("doc_quality").unwrap_or(&0.0) * features.doc_quality;

        Ok(score.max(0.0).min(1.0))
    }

    /// Apply Maximal Marginal Relevance (MMR) for diversity
    async fn apply_mmr_diversity(
        &self,
        mut results: Vec<RankedResult>,
        _query_intent: &QueryIntent,
    ) -> Result<Vec<RankedResult>> {
        if results.len() <= 1 {
            return Ok(results);
        }

        let lambda = self.config.diversity_lambda;
        let mut selected = Vec::new();
        let mut remaining = results;

        // Select first result (highest relevance)
        if !remaining.is_empty() {
            selected.push(remaining.remove(0));
        }

        // Iteratively select results balancing relevance and diversity
        while !remaining.is_empty() && selected.len() < 50 { // Limit to prevent infinite loops
            let mut best_idx = 0;
            let mut best_score = f32::NEG_INFINITY;

            for (idx, candidate) in remaining.iter().enumerate() {
                // Calculate maximum similarity to already selected results
                let max_similarity = selected.iter()
                    .map(|selected_result| {
                        self.calculate_similarity(candidate, selected_result)
                    })
                    .fold(0.0f32, f32::max);

                // MMR formula: λ * relevance - (1-λ) * max_similarity
                let mmr_score = lambda * candidate.relevance_score - (1.0 - lambda) * max_similarity;

                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = idx;
                }
            }

            selected.push(remaining.remove(best_idx));
        }

        // Add any remaining results
        selected.extend(remaining);

        Ok(selected)
    }

    /// Calculate similarity between two results for diversity
    fn calculate_similarity(&self, result1: &RankedResult, result2: &RankedResult) -> f32 {
        // Simple similarity based on cosine vector scores and file types
        let vector_sim = (result1.features.cosine_vec - result2.features.cosine_vec).abs();
        let type_sim = if result1.file_type == result2.file_type { 1.0 } else { 0.0 };
        
        // Weighted combination
        0.7 * (1.0 - vector_sim) + 0.3 * type_sim
    }

    /// Record user interaction for learning
    pub fn record_interaction(&mut self, interaction: UserInteraction) {
        self.user_interactions
            .entry(interaction.document_id)
            .or_insert_with(Vec::new)
            .push(interaction);
    }

    /// Update search context
    pub fn update_context(&mut self, context: SearchContext) {
        self.current_context = Some(context);
    }

    /// Update document statistics
    pub fn update_document_stats(&mut self, document_id: Uuid, stats: DocumentStats) {
        self.document_stats.insert(document_id, stats);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::query_intent::{QueryIntentClassifier, Intent};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_ranking_feature_extraction() {
        let config = RankingConfig::default();
        let ranker = AdvancedRanker::new(config);
        
        let doc_id = Uuid::new_v4();
        let mut query_intent = QueryIntent {
            original_query: "find pdf documents".to_string(),
            normalized_text: "find pdf documents".to_string(),
            intents: HashMap::new(),
            entities: Vec::new(),
            temporal_expressions: Vec::new(),
            complexity_score: 0.3,
            search_strategy: crate::core::query_intent::SearchStrategy::FullTextOnly,
            processed_at: Utc::now(),
        };
        query_intent.intents.insert(Intent::DocumentSearch, 0.8);
        query_intent.intents.insert(Intent::TypeSearch, 0.9);

        let features = ranker.extract_features(
            &doc_id,
            &query_intent,
            0.8, // text_score
            0.6, // vector_score
            Utc::now() - Duration::days(5), // 5 days old
            "pdf",
            1024 * 1024, // 1MB
        ).unwrap();

        assert!(features.text_bm25 == 0.8);
        assert!(features.cosine_vec == 0.6);
        assert!(features.recency_decay > 0.8); // Should be high for recent document
        assert!(features.intent_alignment > 0.8); // Good alignment for PDF search
        assert!(features.type_preference > 0.8); // High preference for PDF
    }

    #[tokio::test]
    async fn test_linear_ranking_model() {
        let mut weights = HashMap::new();
        weights.insert("text_bm25".to_string(), 0.5);
        weights.insert("cosine_vec".to_string(), 0.3);
        weights.insert("recency_decay".to_string(), 0.2);

        let model = LinearRankingModel::new(weights);
        
        let features = RankingFeatures {
            text_bm25: 0.8,
            cosine_vec: 0.6,
            recency_decay: 0.9,
            user_frequency: 0.1,
            doc_quality: 0.7,
            same_project_flag: 1.0,
            diversity_penalty: 0.0,
            intent_alignment: 0.8,
            type_preference: 0.9,
            size_factor: 0.8,
        };

        let score = model.predict(&features).unwrap();
        assert!(score > 0.0 && score <= 1.0);
        
        // Test explanation
        let explanation = model.explain(&features).unwrap();
        assert!(explanation.primary_signals.len() <= 3);
        assert!(explanation.feature_contributions.contains_key("text_bm25"));
    }

    #[tokio::test]
    async fn test_mmr_diversity_selection() {
        let config = RankingConfig::default();
        let ranker = AdvancedRanker::new(config);

        let query_intent = QueryIntent {
            original_query: "test query".to_string(),
            normalized_text: "test query".to_string(),
            intents: HashMap::new(),
            entities: Vec::new(),
            temporal_expressions: Vec::new(),
            complexity_score: 0.3,
            search_strategy: crate::core::query_intent::SearchStrategy::FullTextOnly,
            processed_at: Utc::now(),
        };

        let results = vec![
            RankedResult {
                document_id: Uuid::new_v4(),
                chunk_id: None,
                title: "Document 1".to_string(),
                snippet: "Test content 1".to_string(),
                file_path: "/test1.pdf".to_string(),
                file_type: "pdf".to_string(),
                relevance_score: 0.9,
                features: RankingFeatures {
                    text_bm25: 0.9,
                    cosine_vec: 0.8,
                    recency_decay: 0.9,
                    user_frequency: 0.1,
                    doc_quality: 0.7,
                    same_project_flag: 0.0,
                    diversity_penalty: 0.0,
                    intent_alignment: 0.8,
                    type_preference: 0.9,
                    size_factor: 0.8,
                },
                ranking_explanation: None,
            },
            RankedResult {
                document_id: Uuid::new_v4(),
                chunk_id: None,
                title: "Document 2".to_string(),
                snippet: "Test content 2".to_string(),
                file_path: "/test2.pdf".to_string(),
                file_type: "pdf".to_string(),
                relevance_score: 0.8,
                features: RankingFeatures {
                    text_bm25: 0.8,
                    cosine_vec: 0.9, // Very similar to first result
                    recency_decay: 0.8,
                    user_frequency: 0.2,
                    doc_quality: 0.6,
                    same_project_flag: 0.0,
                    diversity_penalty: 0.0,
                    intent_alignment: 0.7,
                    type_preference: 0.9,
                    size_factor: 0.7,
                },
                ranking_explanation: None,
            },
        ];

        let ranked_results = ranker.rank_results(results, &query_intent).await.unwrap();
        
        assert_eq!(ranked_results.len(), 2);
        // First result should still be highest (highest relevance)
        assert!(ranked_results[0].relevance_score >= ranked_results[1].relevance_score);
    }
}