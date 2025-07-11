use crate::error::Result;
use crate::core::ranking::{RankedResult, RankingFeatures};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Cluster of related search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultCluster {
    pub cluster_id: Uuid,
    pub cluster_label: String,
    pub cluster_keywords: Vec<String>,
    pub results: Vec<RankedResult>,
    pub centroid_features: Option<RankingFeatures>,
    pub coherence_score: f32,
    pub created_at: DateTime<Utc>,
}

/// Group of duplicate documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    pub group_id: Uuid,
    pub primary_result: RankedResult,
    pub duplicates: Vec<DuplicateDocument>,
    pub similarity_threshold: f32,
    pub created_at: DateTime<Utc>,
}

/// Duplicate document information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateDocument {
    pub document_id: Uuid,
    pub file_path: String,
    pub similarity_score: f32,
    pub duplicate_type: DuplicateType,
    pub metadata: HashMap<String, String>,
}

/// Types of document duplication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DuplicateType {
    /// Exact content match (same hash)
    Exact,
    /// Near-duplicate (minor formatting differences)
    NearDuplicate,
    /// Same content, different format (PDF vs DOCX)
    FormatVariant,
    /// Version of the same document
    Version,
}

/// Document summary information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSummary {
    pub document_id: Uuid,
    pub extractive_summary: String,
    pub abstractive_summary: Option<String>,
    pub key_topics: Vec<String>,
    pub summary_length: usize,
    pub confidence_score: f32,
    pub generation_method: SummaryMethod,
    pub created_at: DateTime<Utc>,
}

/// Method used to generate summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SummaryMethod {
    /// TextRank-based extractive summarization
    TextRank,
    /// TF-IDF based extraction
    TfIdf,
    /// LLM-generated abstractive summary
    LlmAbstractive,
    /// Hybrid approach
    Hybrid,
}

/// Highlighted search result snippet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighlightedSnippet {
    pub text: String,
    pub highlights: Vec<Highlight>,
    pub context_before: String,
    pub context_after: String,
    pub snippet_score: f32,
    pub chunk_id: Option<Uuid>,
}

/// Individual highlight in text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Highlight {
    pub start: usize,
    pub end: usize,
    pub highlight_type: HighlightType,
    pub relevance_score: f32,
}

/// Type of highlighting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HighlightType {
    /// Exact query term match
    ExactMatch,
    /// Semantic similarity match
    SemanticMatch,
    /// Entity match (person, date, etc.)
    EntityMatch,
    /// Important keyword
    KeywordMatch,
}

/// Configuration for result processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultProcessingConfig {
    /// Enable clustering of results
    pub enable_clustering: bool,
    /// Enable duplicate detection
    pub enable_deduplication: bool,
    /// Enable automatic summarization
    pub enable_summarization: bool,
    /// Similarity threshold for clustering
    pub clustering_threshold: f32,
    /// Similarity threshold for duplicate detection
    pub duplicate_threshold: f32,
    /// Maximum cluster size
    pub max_cluster_size: usize,
    /// Maximum summary length (words)
    pub max_summary_length: usize,
    /// Enable intelligent highlighting
    pub enable_highlighting: bool,
}

impl Default for ResultProcessingConfig {
    fn default() -> Self {
        Self {
            enable_clustering: true,
            enable_deduplication: true,
            enable_summarization: true,
            clustering_threshold: 0.3,
            duplicate_threshold: 0.85,
            max_cluster_size: 10,
            max_summary_length: 150,
            enable_highlighting: true,
        }
    }
}

/// Advanced result processing system
pub struct ResultProcessor {
    config: ResultProcessingConfig,
    cluster_cache: HashMap<String, Vec<ResultCluster>>,
    duplicate_cache: HashMap<Uuid, DuplicateGroup>,
    summary_cache: HashMap<Uuid, DocumentSummary>,
}

impl ResultProcessor {
    pub fn new(config: ResultProcessingConfig) -> Self {
        Self {
            config,
            cluster_cache: HashMap::new(),
            duplicate_cache: HashMap::new(),
            summary_cache: HashMap::new(),
        }
    }

    /// Process search results with clustering, deduplication, and summarization
    pub async fn process_results(
        &mut self,
        results: Vec<RankedResult>,
        query: &str,
    ) -> Result<ProcessedResults> {
        let start_time = std::time::Instant::now();
        let original_count = results.len();

        // Step 1: Detect and group duplicates
        let deduplicated_results = if self.config.enable_deduplication {
            self.detect_duplicates(results).await?
        } else {
            results
        };
        let deduplicated_count = deduplicated_results.len();

        // Step 2: Cluster related results
        let clusters = if self.config.enable_clustering {
            self.cluster_results(&deduplicated_results, query).await?
        } else {
            vec![]
        };

        // Step 3: Generate summaries for top results
        let summaries = if self.config.enable_summarization {
            self.generate_summaries(&deduplicated_results).await?
        } else {
            HashMap::new()
        };

        // Step 4: Create highlighted snippets
        let mut highlighted_results = deduplicated_results;
        if self.config.enable_highlighting {
            self.create_highlighted_snippets(&mut highlighted_results, query).await?;
        }

        let processing_time = start_time.elapsed();

        Ok(ProcessedResults {
            results: highlighted_results,
            clusters,
            duplicate_groups: self.get_duplicate_groups(),
            summaries,
            processing_time_ms: processing_time.as_millis() as u32,
            total_results: original_count,
            deduplicated_count,
        })
    }

    /// Detect duplicate documents using multiple similarity measures
    pub async fn detect_duplicates(&mut self, results: Vec<RankedResult>) -> Result<Vec<RankedResult>> {
        let mut unique_results = Vec::new();
        let mut processed_hashes = HashSet::new();
        
        for result in results {
            let content_hash = self.calculate_content_hash(&result);
            
            if processed_hashes.contains(&content_hash) {
                // Find existing group and add as duplicate
                self.add_to_duplicate_group(&result, &content_hash).await?;
            } else {
                // Check for near-duplicates using similarity
                let is_near_duplicate = self.check_near_duplicate(&result, &unique_results).await?;
                
                if !is_near_duplicate {
                    processed_hashes.insert(content_hash);
                    unique_results.push(result);
                }
            }
        }

        Ok(unique_results)
    }

    /// Calculate content hash for duplicate detection
    fn calculate_content_hash(&self, result: &RankedResult) -> String {
        use sha2::{Sha256, Digest};
        
        // Create hash from normalized content
        let normalized_content = result.snippet.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>();
        
        let mut hasher = Sha256::new();
        hasher.update(normalized_content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Check if result is a near-duplicate of existing results
    async fn check_near_duplicate(
        &self,
        candidate: &RankedResult,
        existing_results: &[RankedResult],
    ) -> Result<bool> {
        for existing in existing_results {
            let similarity = self.calculate_text_similarity(
                &candidate.snippet,
                &existing.snippet,
            );
            
            if similarity > self.config.duplicate_threshold {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Calculate text similarity using improved Jaccard + TF-IDF similarity
    fn calculate_text_similarity(&self, text1: &str, text2: &str) -> f32 {
        // Normalize and tokenize text
        let words1 = self.tokenize_and_normalize(text1);
        let words2 = self.tokenize_and_normalize(text2);
        
        // Jaccard similarity on word sets
        let set1: HashSet<&str> = words1.iter().map(|s| s.as_str()).collect();
        let set2: HashSet<&str> = words2.iter().map(|s| s.as_str()).collect();
        
        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        
        let jaccard = if union == 0 { 0.0 } else { intersection as f32 / union as f32 };
        
        // TF-IDF cosine similarity
        let cosine = self.calculate_cosine_text_similarity(&words1, &words2);
        
        // Combine both measures
        0.6 * jaccard + 0.4 * cosine
    }
    
    /// Tokenize and normalize text for better similarity calculation
    fn tokenize_and_normalize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 2 && !self.is_stop_word(word))
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|word| !word.is_empty())
            .collect()
    }
    
    /// Calculate cosine similarity between word frequency vectors
    fn calculate_cosine_text_similarity(&self, words1: &[String], words2: &[String]) -> f32 {
        let mut word_counts1 = HashMap::new();
        let mut word_counts2 = HashMap::new();
        
        for word in words1 {
            *word_counts1.entry(word.clone()).or_insert(0) += 1;
        }
        for word in words2 {
            *word_counts2.entry(word.clone()).or_insert(0) += 1;
        }
        
        if word_counts1.is_empty() || word_counts2.is_empty() {
            return 0.0;
        }
        
        // Get all unique words
        let all_words: HashSet<String> = word_counts1.keys()
            .chain(word_counts2.keys())
            .cloned()
            .collect();
        
        // Calculate dot product and magnitudes
        let mut dot_product = 0.0;
        let mut magnitude1 = 0.0;
        let mut magnitude2 = 0.0;
        
        for word in all_words {
            let freq1 = *word_counts1.get(&word).unwrap_or(&0) as f32;
            let freq2 = *word_counts2.get(&word).unwrap_or(&0) as f32;
            
            dot_product += freq1 * freq2;
            magnitude1 += freq1 * freq1;
            magnitude2 += freq2 * freq2;
        }
        
        if magnitude1 == 0.0 || magnitude2 == 0.0 {
            0.0
        } else {
            dot_product / (magnitude1.sqrt() * magnitude2.sqrt())
        }
    }
    
    /// Calculate vector similarity based on embedding features
    fn calculate_vector_similarity(&self, result1: &RankedResult, result2: &RankedResult) -> f32 {
        // Use the actual cosine similarity scores to estimate vector similarity
        let avg_score1 = (result1.features.cosine_vec + result1.features.text_bm25) / 2.0;
        let avg_score2 = (result2.features.cosine_vec + result2.features.text_bm25) / 2.0;
        
        // If both have similar relevance scores, they might be semantically similar
        let score_similarity = 1.0 - (avg_score1 - avg_score2).abs();
        
        // Also consider other feature similarities
        let intent_similarity = 1.0 - (result1.features.intent_alignment - result2.features.intent_alignment).abs();
        let quality_similarity = 1.0 - (result1.features.doc_quality - result2.features.doc_quality).abs();
        
        // Weighted combination
        0.5 * score_similarity + 0.3 * intent_similarity + 0.2 * quality_similarity
    }
    
    /// Calculate semantic similarity based on content analysis
    fn calculate_semantic_similarity(&self, result1: &RankedResult, result2: &RankedResult) -> f32 {
        // Extract key terms and concepts
        let terms1 = self.extract_key_terms(&format!("{} {}", result1.title, result1.snippet));
        let terms2 = self.extract_key_terms(&format!("{} {}", result2.title, result2.snippet));
        
        if terms1.is_empty() || terms2.is_empty() {
            return 0.0;
        }
        
        // Calculate overlap of key concepts
        let set1: HashSet<&str> = terms1.iter().map(|s| s.as_str()).collect();
        let set2: HashSet<&str> = terms2.iter().map(|s| s.as_str()).collect();
        
        let intersection = set1.intersection(&set2).count();
        let min_size = set1.len().min(set2.len());
        
        if min_size == 0 {
            0.0
        } else {
            intersection as f32 / min_size as f32
        }
    }
    
    /// Extract key terms for semantic analysis
    fn extract_key_terms(&self, text: &str) -> Vec<String> {
        let words = self.tokenize_and_normalize(text);
        let mut word_counts = HashMap::new();
        
        for word in &words {
            if word.len() > 4 { // Focus on longer, more meaningful words
                *word_counts.entry(word.clone()).or_insert(0) += 1;
            }
        }
        
        // Sort by frequency and take top terms
        let mut terms: Vec<(String, usize)> = word_counts.into_iter().collect();
        terms.sort_by(|a, b| b.1.cmp(&a.1));
        
        terms.into_iter()
            .take(5) // Top 5 key terms
            .map(|(word, _)| word)
            .collect()
    }
    
    /// Calculate path similarity (documents in same folder are more related)
    fn calculate_path_similarity(&self, path1: &str, path2: &str) -> f32 {
        let parts1: Vec<&str> = path1.split('/').filter(|s| !s.is_empty()).collect();
        let parts2: Vec<&str> = path2.split('/').filter(|s| !s.is_empty()).collect();
        
        if parts1.is_empty() || parts2.is_empty() {
            return 0.0;
        }
        
        // Count common path components
        let mut common_components = 0;
        let min_len = parts1.len().min(parts2.len());
        
        for i in 0..min_len {
            if parts1[i] == parts2[i] {
                common_components += 1;
            } else {
                break; // Stop at first difference
            }
        }
        
        // Bonus if files are in the exact same directory
        if parts1.len() > 1 && parts2.len() > 1 {
            let dir1 = &parts1[..parts1.len()-1];
            let dir2 = &parts2[..parts2.len()-1];
            if dir1 == dir2 {
                return 0.9; // High similarity for same directory
            }
        }
        
        // Calculate similarity based on common path prefix
        common_components as f32 / parts1.len().max(parts2.len()) as f32
    }

    /// Add result to duplicate group
    async fn add_to_duplicate_group(
        &mut self,
        result: &RankedResult,
        content_hash: &str,
    ) -> Result<()> {
        // Simplified - would find existing group by hash and add duplicate
        let duplicate_doc = DuplicateDocument {
            document_id: result.document_id,
            file_path: result.file_path.clone(),
            similarity_score: 1.0, // Exact match
            duplicate_type: DuplicateType::Exact,
            metadata: HashMap::new(),
        };

        // In real implementation, would update existing group
        Ok(())
    }

    /// Cluster results using simple vector similarity
    pub async fn cluster_results(
        &mut self,
        results: &[RankedResult],
        query: &str,
    ) -> Result<Vec<ResultCluster>> {
        if results.len() < 2 {
            return Ok(vec![]);
        }

        let mut clusters = Vec::new();
        let mut used_indices = HashSet::new();

        for (i, result) in results.iter().enumerate() {
            if used_indices.contains(&i) {
                continue;
            }

            let mut cluster_results = vec![result.clone()];
            used_indices.insert(i);

            // Find similar results to cluster together
            for (j, other_result) in results.iter().enumerate() {
                if i == j || used_indices.contains(&j) {
                    continue;
                }

                let similarity = self.calculate_cluster_similarity(result, other_result);
                
                if similarity > self.config.clustering_threshold && 
                   cluster_results.len() < self.config.max_cluster_size {
                    cluster_results.push(other_result.clone());
                    used_indices.insert(j);
                }
            }

            // Only create cluster if it has multiple results
            if cluster_results.len() > 1 {
                let cluster = self.create_cluster(cluster_results, query).await?;
                clusters.push(cluster);
            }
        }

        Ok(clusters)
    }

    /// Calculate similarity between two results for clustering
    fn calculate_cluster_similarity(&self, result1: &RankedResult, result2: &RankedResult) -> f32 {
        // 1. Enhanced text similarity (TF-IDF cosine + Jaccard)
        let snippet_sim = self.calculate_enhanced_text_similarity(&result1.snippet, &result2.snippet);
        let title_sim = self.calculate_enhanced_text_similarity(&result1.title, &result2.title);
        let text_sim = 0.7 * snippet_sim + 0.3 * title_sim;
        
        // 2. Vector similarity (use the actual vector scores)
        let vector_sim = self.calculate_vector_similarity(result1, result2);
        
        // 3. File type similarity (boost for same type)
        let type_sim = if result1.file_type == result2.file_type { 0.8 } else { 0.0 };
        
        // 4. Semantic similarity based on content analysis
        let semantic_sim = self.calculate_semantic_similarity(result1, result2);
        
        // 5. Path similarity (documents in same folder are more likely related)
        let path_sim = self.calculate_path_similarity(&result1.file_path, &result2.file_path);
        
        // 6. Feature-based similarity (using ranking features)
        let feature_sim = self.calculate_feature_similarity(result1, result2);
        
        // Weighted combination with emphasis on content and semantics
        let base_similarity = 0.35 * text_sim + 0.20 * vector_sim + 0.25 * semantic_sim + 0.10 * type_sim + 0.10 * path_sim;
        
        // Add feature similarity as a boost
        (base_similarity + 0.15 * feature_sim).min(1.0)
    }

    /// Enhanced text similarity combining TF-IDF cosine similarity and Jaccard
    fn calculate_enhanced_text_similarity(&self, text1: &str, text2: &str) -> f32 {
        // Get clean, normalized tokens
        let tokens1 = self.tokenize_and_normalize(text1);
        let tokens2 = self.tokenize_and_normalize(text2);
        
        if tokens1.is_empty() || tokens2.is_empty() {
            return 0.0;
        }
        
        // 1. TF-IDF Cosine Similarity
        let tfidf_sim = self.calculate_tfidf_cosine_similarity(&tokens1, &tokens2);
        
        // 2. Jaccard similarity on tokens
        let set1: HashSet<&str> = tokens1.iter().map(|s| s.as_str()).collect();
        let set2: HashSet<&str> = tokens2.iter().map(|s| s.as_str()).collect();
        
        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        
        let jaccard_sim = if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        };
        
        // 3. N-gram similarity (bigrams)
        let ngram_sim = self.calculate_ngram_similarity(&tokens1, &tokens2, 2);
        
        // Weighted combination: emphasize semantic similarity but include lexical overlap
        0.5 * tfidf_sim + 0.3 * jaccard_sim + 0.2 * ngram_sim
    }

    /// Calculate TF-IDF cosine similarity between token lists
    fn calculate_tfidf_cosine_similarity(&self, tokens1: &[String], tokens2: &[String]) -> f32 {
        // Build vocabulary
        let mut vocab = HashSet::new();
        tokens1.iter().for_each(|t| { vocab.insert(t.as_str()); });
        tokens2.iter().for_each(|t| { vocab.insert(t.as_str()); });
        
        if vocab.is_empty() {
            return 0.0;
        }
        
        // Calculate term frequencies
        let mut tf1 = HashMap::new();
        let mut tf2 = HashMap::new();
        
        for token in tokens1 {
            *tf1.entry(token.as_str()).or_insert(0) += 1;
        }
        for token in tokens2 {
            *tf2.entry(token.as_str()).or_insert(0) += 1;
        }
        
        // Convert to TF vectors (simple term frequency)
        let mut vec1 = Vec::new();
        let mut vec2 = Vec::new();
        
        for term in &vocab {
            let freq1 = *tf1.get(term).unwrap_or(&0) as f32;
            let freq2 = *tf2.get(term).unwrap_or(&0) as f32;
            
            vec1.push(freq1);
            vec2.push(freq2);
        }
        
        // Calculate cosine similarity
        self.cosine_similarity(&vec1, &vec2)
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        if vec1.len() != vec2.len() {
            return 0.0;
        }
        
        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Calculate n-gram similarity between token lists
    fn calculate_ngram_similarity(&self, tokens1: &[String], tokens2: &[String], n: usize) -> f32 {
        if tokens1.len() < n || tokens2.len() < n {
            return 0.0;
        }
        
        let ngrams1: HashSet<Vec<&str>> = tokens1.windows(n)
            .map(|window| window.iter().map(|s| s.as_str()).collect())
            .collect();
        
        let ngrams2: HashSet<Vec<&str>> = tokens2.windows(n)
            .map(|window| window.iter().map(|s| s.as_str()).collect())
            .collect();
        
        let intersection = ngrams1.intersection(&ngrams2).count();
        let union = ngrams1.union(&ngrams2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Calculate feature-based similarity using ranking features
    fn calculate_feature_similarity(&self, result1: &RankedResult, result2: &RankedResult) -> f32 {
        let f1 = &result1.features;
        let f2 = &result2.features;
        
        // Calculate similarity for each feature
        let bm25_sim = 1.0 - (f1.text_bm25 - f2.text_bm25).abs();
        let cosine_sim = 1.0 - (f1.cosine_vec - f2.cosine_vec).abs();
        let recency_sim = 1.0 - (f1.recency_decay - f2.recency_decay).abs().min(1.0);
        let quality_sim = 1.0 - (f1.doc_quality - f2.doc_quality).abs();
        let intent_sim = 1.0 - (f1.intent_alignment - f2.intent_alignment).abs();
        let type_pref_sim = 1.0 - (f1.type_preference - f2.type_preference).abs();
        
        // Weighted combination of feature similarities
        0.3 * bm25_sim + 0.25 * cosine_sim + 0.15 * intent_sim + 0.15 * quality_sim + 0.1 * type_pref_sim + 0.05 * recency_sim
    }

    /// Create a cluster from grouped results
    async fn create_cluster(
        &self,
        results: Vec<RankedResult>,
        query: &str,
    ) -> Result<ResultCluster> {
        let cluster_id = Uuid::new_v4();
        
        // Generate cluster label from common terms
        let cluster_label = self.generate_cluster_label(&results, query);
        
        // Extract common keywords
        let cluster_keywords = self.extract_cluster_keywords(&results);
        
        // Calculate centroid features
        let centroid_features = self.calculate_centroid_features(&results);
        
        // Calculate cluster coherence
        let coherence_score = self.calculate_cluster_coherence(&results);

        Ok(ResultCluster {
            cluster_id,
            cluster_label,
            cluster_keywords,
            results,
            centroid_features: Some(centroid_features),
            coherence_score,
            created_at: Utc::now(),
        })
    }

    /// Generate a descriptive label for the cluster
    fn generate_cluster_label(&self, results: &[RankedResult], query: &str) -> String {
        // Extract common words from titles and snippets
        let mut word_counts = HashMap::new();
        
        for result in results {
            let all_text = format!("{} {}", result.title, result.snippet);
            for word in all_text.split_whitespace() {
                let word = word.to_lowercase();
                if word.len() > 3 && !self.is_stop_word(&word) {
                    *word_counts.entry(word).or_insert(0) += 1;
                }
            }
        }

        // Find most common meaningful words
        let mut word_freq: Vec<(String, usize)> = word_counts.into_iter().collect();
        word_freq.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Take top 2-3 words for label
        let label_words: Vec<String> = word_freq
            .into_iter()
            .take(2)
            .map(|(word, _)| self.capitalize_first(&word))
            .collect();

        if label_words.is_empty() {
            format!("Related to '{}'", query)
        } else {
            label_words.join(" & ")
        }
    }

    /// Extract keywords that are common across cluster results
    fn extract_cluster_keywords(&self, results: &[RankedResult]) -> Vec<String> {
        let mut word_counts = HashMap::new();
        let min_frequency = (results.len() / 2).max(1); // Word must appear in at least half the results
        
        for result in results {
            let mut doc_words = HashSet::new();
            let all_text = format!("{} {}", result.title, result.snippet);
            
            for word in all_text.split_whitespace() {
                let word = word.to_lowercase();
                if word.len() > 3 && !self.is_stop_word(&word) {
                    doc_words.insert(word);
                }
            }
            
            for word in doc_words {
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        word_counts
            .into_iter()
            .filter(|(_, count)| *count >= min_frequency)
            .map(|(word, _)| word)
            .take(5)
            .collect()
    }

    /// Calculate centroid features for cluster
    fn calculate_centroid_features(&self, results: &[RankedResult]) -> RankingFeatures {
        let count = results.len() as f32;
        
        let avg_text_bm25 = results.iter().map(|r| r.features.text_bm25).sum::<f32>() / count;
        let avg_cosine_vec = results.iter().map(|r| r.features.cosine_vec).sum::<f32>() / count;
        let avg_recency = results.iter().map(|r| r.features.recency_decay).sum::<f32>() / count;
        let avg_user_freq = results.iter().map(|r| r.features.user_frequency).sum::<f32>() / count;
        let avg_quality = results.iter().map(|r| r.features.doc_quality).sum::<f32>() / count;

        RankingFeatures {
            text_bm25: avg_text_bm25,
            cosine_vec: avg_cosine_vec,
            recency_decay: avg_recency,
            user_frequency: avg_user_freq,
            doc_quality: avg_quality,
            same_project_flag: 0.0,
            diversity_penalty: 0.0,
            intent_alignment: results.iter().map(|r| r.features.intent_alignment).sum::<f32>() / count,
            type_preference: results.iter().map(|r| r.features.type_preference).sum::<f32>() / count,
            size_factor: results.iter().map(|r| r.features.size_factor).sum::<f32>() / count,
        }
    }

    /// Calculate how coherent the cluster is
    fn calculate_cluster_coherence(&self, results: &[RankedResult]) -> f32 {
        if results.len() < 2 {
            return 1.0;
        }

        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                total_similarity += self.calculate_cluster_similarity(&results[i], &results[j]);
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_similarity / comparisons as f32
        } else {
            0.0
        }
    }

    /// Generate summaries for top results
    async fn generate_summaries(
        &mut self,
        results: &[RankedResult],
    ) -> Result<HashMap<Uuid, DocumentSummary>> {
        let mut summaries = HashMap::new();
        
        // Only summarize top results to avoid computational cost
        let top_results = results.iter().take(5);
        
        for result in top_results {
            if let Some(summary) = self.summary_cache.get(&result.document_id) {
                summaries.insert(result.document_id, summary.clone());
            } else {
                let summary = self.generate_extractive_summary(result).await?;
                self.summary_cache.insert(result.document_id, summary.clone());
                summaries.insert(result.document_id, summary);
            }
        }

        Ok(summaries)
    }

    /// Generate extractive summary using TextRank-like algorithm
    pub async fn generate_extractive_summary(&self, result: &RankedResult) -> Result<DocumentSummary> {
        let sentences = self.split_into_sentences(&result.snippet);
        
        if sentences.is_empty() {
            return Ok(DocumentSummary {
                document_id: result.document_id,
                extractive_summary: result.snippet.clone(),
                abstractive_summary: None,
                key_topics: vec![],
                summary_length: result.snippet.split_whitespace().count(),
                confidence_score: 0.5,
                generation_method: SummaryMethod::TextRank,
                created_at: Utc::now(),
            });
        }

        // Score sentences based on word frequency and position
        let scored_sentences = self.score_sentences(&sentences);
        
        // Select top sentences for summary
        let summary_sentences = self.select_summary_sentences(scored_sentences);
        let extractive_summary = summary_sentences.join(" ");
        
        // Extract key topics
        let key_topics = self.extract_key_topics(&result.snippet);

        Ok(DocumentSummary {
            document_id: result.document_id,
            extractive_summary,
            abstractive_summary: None,
            key_topics,
            summary_length: summary_sentences.iter().map(|s| s.split_whitespace().count()).sum(),
            confidence_score: 0.7,
            generation_method: SummaryMethod::TextRank,
            created_at: Utc::now(),
        })
    }

    /// Split text into sentences
    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        text.split(". ")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() > 10)
            .collect()
    }

    /// Score sentences for extractive summarization
    fn score_sentences(&self, sentences: &[String]) -> Vec<(String, f32)> {
        let mut scored = Vec::new();
        
        for (i, sentence) in sentences.iter().enumerate() {
            let mut score = 0.0;
            
            // Position bonus (earlier sentences get higher scores)
            score += 1.0 / (i as f32 + 1.0);
            
            // Length bonus (prefer medium-length sentences)
            let word_count = sentence.split_whitespace().count();
            if word_count >= 10 && word_count <= 30 {
                score += 0.5;
            }
            
            // Keyword density bonus
            let meaningful_words = sentence.split_whitespace()
                .filter(|word| word.len() > 3 && !self.is_stop_word(word))
                .count();
            score += meaningful_words as f32 / word_count as f32;
            
            scored.push((sentence.clone(), score));
        }
        
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Select best sentences for summary
    fn select_summary_sentences(&self, scored_sentences: Vec<(String, f32)>) -> Vec<String> {
        let mut selected = Vec::new();
        let mut word_count = 0;
        
        for (sentence, _score) in scored_sentences {
            let sentence_words = sentence.split_whitespace().count();
            if word_count + sentence_words <= self.config.max_summary_length {
                selected.push(sentence);
                word_count += sentence_words;
            }
            
            if selected.len() >= 3 { // Max 3 sentences
                break;
            }
        }
        
        selected
    }

    /// Extract key topics from text
    fn extract_key_topics(&self, text: &str) -> Vec<String> {
        let mut word_counts = HashMap::new();
        
        for word in text.split_whitespace() {
            let word = word.to_lowercase();
            if word.len() > 4 && !self.is_stop_word(&word) {
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        let mut topics: Vec<(String, usize)> = word_counts.into_iter().collect();
        topics.sort_by(|a, b| b.1.cmp(&a.1));
        
        topics.into_iter()
            .take(3)
            .map(|(word, _)| self.capitalize_first(&word))
            .collect()
    }

    /// Create highlighted snippets for search results
    async fn create_highlighted_snippets(
        &self,
        results: &mut Vec<RankedResult>,
        query: &str,
    ) -> Result<()> {
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        
        for result in results {
            result.snippet = self.highlight_snippet(&result.snippet, &query_terms);
        }
        
        Ok(())
    }

    /// Add highlighting to snippet text
    pub fn highlight_snippet(&self, text: &str, query_terms: &[&str]) -> String {
        let mut highlighted = text.to_string();
        
        for term in query_terms {
            let _term_lower = term.to_lowercase();
            let pattern = regex::Regex::new(&format!(r"(?i)\b{}\b", regex::escape(term))).unwrap();
            highlighted = pattern.replace_all(&highlighted, |caps: &regex::Captures| {
                format!("<mark>{}</mark>", &caps[0])
            }).to_string();
        }
        
        highlighted
    }

    /// Get all duplicate groups
    fn get_duplicate_groups(&self) -> Vec<DuplicateGroup> {
        self.duplicate_cache.values().cloned().collect()
    }

    /// Check if word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        matches!(word, "the" | "a" | "an" | "and" | "or" | "but" | "in" | "on" | "at" | "to" | "for" | "of" | "with" | "by" | "is" | "are" | "was" | "were" | "be" | "been" | "have" | "has" | "had" | "do" | "does" | "did" | "will" | "would" | "could" | "should" | "may" | "might" | "can" | "this" | "that" | "these" | "those")
    }

    /// Capitalize first letter of word
    fn capitalize_first(&self, word: &str) -> String {
        let mut chars = word.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        }
    }
}

/// Final processed search results
#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessedResults {
    /// Main search results with highlighting
    pub results: Vec<RankedResult>,
    /// Result clusters
    pub clusters: Vec<ResultCluster>,
    /// Duplicate groups that were merged
    pub duplicate_groups: Vec<DuplicateGroup>,
    /// Document summaries
    pub summaries: HashMap<Uuid, DocumentSummary>,
    /// Processing time in milliseconds
    pub processing_time_ms: u32,
    /// Total number of original results
    pub total_results: usize,
    /// Number of results after deduplication
    pub deduplicated_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ranking::RankingFeatures;

    #[tokio::test]
    async fn test_duplicate_detection() {
        let mut processor = ResultProcessor::new(ResultProcessingConfig::default());
        
        let results = vec![
            create_test_result("doc1", "This is a test document about machine learning"),
            create_test_result("doc2", "This is a test document about machine learning"), // Duplicate
            create_test_result("doc3", "Different content about artificial intelligence"),
        ];

        let deduplicated = processor.detect_duplicates(results).await.unwrap();
        assert_eq!(deduplicated.len(), 2); // Should remove one duplicate
    }

    #[tokio::test]
    async fn test_result_clustering() {
        let mut processor = ResultProcessor::new(ResultProcessingConfig::default());
        
        let results = vec![
            create_test_result("doc1", "Machine learning algorithms for data science"),
            create_test_result("doc2", "Deep learning neural networks and AI"),
            create_test_result("doc3", "Python programming tutorial for beginners"),
            create_test_result("doc4", "Advanced machine learning techniques"),
        ];

        let clusters = processor.cluster_results(&results, "machine learning").await.unwrap();
        assert!(!clusters.is_empty());
        
        // Should cluster ML-related documents together
        if let Some(cluster) = clusters.first() {
            assert!(cluster.results.len() >= 2);
            assert!(cluster.cluster_label.contains("Machine") || cluster.cluster_label.contains("Learning"));
        }
    }

    #[tokio::test]
    async fn test_summarization() {
        let mut processor = ResultProcessor::new(ResultProcessingConfig::default());
        
        let result = create_test_result("doc1", 
            "Machine learning is a subset of artificial intelligence. It involves training algorithms on data. \
             These algorithms can then make predictions on new data. The field has many applications including \
             computer vision, natural language processing, and robotics.");

        let summary = processor.generate_extractive_summary(&result).await.unwrap();
        
        assert!(!summary.extractive_summary.is_empty());
        assert!(summary.summary_length <= 150); // Respects max length
        assert!(!summary.key_topics.is_empty());
    }

    fn create_test_result(title: &str, snippet: &str) -> RankedResult {
        RankedResult {
            document_id: Uuid::new_v4(),
            chunk_id: None,
            title: title.to_string(),
            snippet: snippet.to_string(),
            file_path: format!("/test/{}.txt", title),
            file_type: "txt".to_string(),
            relevance_score: 0.8,
            features: RankingFeatures {
                text_bm25: 0.8,
                cosine_vec: 0.7,
                recency_decay: 0.9,
                user_frequency: 0.1,
                doc_quality: 0.7,
                same_project_flag: 0.0,
                diversity_penalty: 0.0,
                intent_alignment: 0.8,
                type_preference: 0.8,
                size_factor: 0.8,
            },
            ranking_explanation: None,
        }
    }
}