use crate::error::Result;
use crate::core::ranking::{UserInteraction, InteractionType, SearchContext};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// User behavior patterns and preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: Uuid,
    pub created_at: DateTime<Utc>,
    pub last_active: DateTime<Utc>,
    pub search_preferences: SearchPreferences,
    pub interaction_patterns: InteractionPatterns,
    pub document_preferences: DocumentPreferences,
    pub temporal_patterns: TemporalPatterns,
    pub context_preferences: ContextPreferences,
}

/// User's search behavior preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPreferences {
    /// Preferred file types (weighted by usage frequency)
    pub file_type_preferences: HashMap<String, f32>,
    /// Preferred search strategies based on query types
    pub strategy_preferences: HashMap<String, f32>,
    /// Common search domains/topics
    pub topic_preferences: HashMap<String, f32>,
    /// Preferred result ranking factors
    pub ranking_preferences: RankingPreferences,
    /// Average results per page user typically views
    pub typical_results_viewed: usize,
    /// Preferred snippet length
    pub preferred_snippet_length: usize,
}

/// User's ranking factor preferences learned from interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingPreferences {
    /// Weight for recency (higher = prefers newer documents)
    pub recency_weight: f32,
    /// Weight for document quality
    pub quality_weight: f32,
    /// Weight for exact text matches vs semantic similarity
    pub text_vs_semantic_weight: f32,
    /// Preference for diversity vs relevance
    pub diversity_preference: f32,
    /// Context/project locality preference
    pub context_locality_weight: f32,
}

/// Patterns in user interaction behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPatterns {
    /// Average time spent on search results
    pub average_dwell_time: f32,
    /// Click-through rates by result position
    pub position_click_rates: HashMap<usize, f32>,
    /// Common query reformulation patterns
    pub reformulation_patterns: Vec<QueryReformulation>,
    /// Bounce rate (quick return to search)
    pub bounce_rate: f32,
    /// Multi-document session patterns
    pub session_patterns: SessionPatterns,
}

/// Query reformulation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryReformulation {
    pub original_query: String,
    pub reformulated_query: String,
    pub success_rate: f32,
    pub frequency: usize,
}

/// Session-level interaction patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionPatterns {
    /// Average queries per session
    pub avg_queries_per_session: f32,
    /// Average session duration
    pub avg_session_duration: f32,
    /// Common multi-query patterns
    pub query_sequences: Vec<QuerySequence>,
    /// Task completion indicators
    pub completion_patterns: Vec<CompletionPattern>,
}

/// Sequence of related queries in a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySequence {
    pub queries: Vec<String>,
    pub frequency: usize,
    pub success_rate: f32,
    pub typical_duration: f32,
}

/// Pattern indicating task completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionPattern {
    pub final_action: CompletionAction,
    pub query_pattern: String,
    pub success_indicators: Vec<String>,
}

/// Actions that indicate task completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompletionAction {
    DocumentOpen,
    DocumentBookmark,
    LongDwell,
    DocumentShare,
    SessionEnd,
}

/// Document type and content preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentPreferences {
    /// Preferred document sizes (in KB)
    pub size_preferences: SizePreference,
    /// Language preferences
    pub language_preferences: HashMap<String, f32>,
    /// Content freshness preferences
    pub freshness_preferences: FreshnessPreference,
    /// Source/author preferences
    pub source_preferences: HashMap<String, f32>,
}

/// Document size preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizePreference {
    pub min_preferred_size: u64,
    pub max_preferred_size: u64,
    pub optimal_size: u64,
}

/// Content freshness preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreshnessPreference {
    /// How much user values recent content (0.0 = doesn't care, 1.0 = only recent)
    pub recency_importance: f32,
    /// Acceptable age for content (in days)
    pub max_acceptable_age: u32,
    /// Optimal content age (in days)
    pub optimal_age: u32,
}

/// Time-based usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPatterns {
    /// Usage patterns by hour of day
    pub hourly_patterns: HashMap<u8, f32>,
    /// Usage patterns by day of week
    pub daily_patterns: HashMap<u8, f32>,
    /// Seasonal/monthly patterns
    pub monthly_patterns: HashMap<u8, f32>,
    /// Search behavior differences by time
    pub time_based_preferences: TimeBasedPreferences,
}

/// How search behavior changes by time of day/week
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBasedPreferences {
    /// Morning vs evening search patterns
    pub time_of_day_preferences: HashMap<String, f32>,
    /// Weekday vs weekend preferences
    pub weekday_preferences: HashMap<String, f32>,
}

/// Context and environment preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPreferences {
    /// Preferred contexts/projects for different query types
    pub project_associations: HashMap<String, String>,
    /// Application context preferences
    pub app_context_preferences: HashMap<String, f32>,
    /// Multi-tasking patterns
    pub multitasking_patterns: Vec<MultitaskingPattern>,
}

/// Pattern of using multiple applications together
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultitaskingPattern {
    pub applications: Vec<String>,
    pub frequency: usize,
    pub typical_search_queries: Vec<String>,
}

/// Configuration for user intelligence system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserIntelligenceConfig {
    /// Enable personalization features
    pub enable_personalization: bool,
    /// Enable interaction tracking
    pub enable_tracking: bool,
    /// Minimum interactions before personalization kicks in
    pub min_interactions_for_personalization: usize,
    /// How long to retain user data (in days)
    pub data_retention_days: u32,
    /// Learning rate for preference updates
    pub learning_rate: f32,
    /// Decay rate for older interactions
    pub temporal_decay_rate: f32,
}

impl Default for UserIntelligenceConfig {
    fn default() -> Self {
        Self {
            enable_personalization: true,
            enable_tracking: true,
            min_interactions_for_personalization: 10,
            data_retention_days: 365,
            learning_rate: 0.1,
            temporal_decay_rate: 0.99,
        }
    }
}

/// User intelligence system for personalization and adaptation
pub struct UserIntelligenceSystem {
    config: UserIntelligenceConfig,
    user_profiles: HashMap<Uuid, UserProfile>,
    active_sessions: HashMap<Uuid, SearchSession>,
    interaction_buffer: Vec<UserInteraction>,
    global_patterns: GlobalUsagePatterns,
}

/// Active search session tracking
#[derive(Debug, Clone)]
pub struct SearchSession {
    pub session_id: Uuid,
    pub user_id: Uuid,
    pub started_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub queries: Vec<SessionQuery>,
    pub context: SearchContext,
    pub current_task: Option<InferredTask>,
}

/// Query within a search session
#[derive(Debug, Clone)]
pub struct SessionQuery {
    pub query: String,
    pub timestamp: DateTime<Utc>,
    pub results_shown: usize,
    pub interactions: Vec<UserInteraction>,
    pub reformulated_from: Option<String>,
}

/// Inferred user task/intent
#[derive(Debug, Clone)]
pub struct InferredTask {
    pub task_type: TaskType,
    pub confidence: f32,
    pub context_clues: Vec<String>,
    pub estimated_completion: f32,
}

/// Types of tasks users might be performing
#[derive(Debug, Clone)]
pub enum TaskType {
    Research,
    Troubleshooting,
    Learning,
    Reference,
    Creation,
    Verification,
    Comparison,
}

/// Global usage patterns across all users
#[derive(Debug, Clone)]
pub struct GlobalUsagePatterns {
    pub popular_queries: HashMap<String, usize>,
    pub trending_topics: Vec<TrendingTopic>,
    pub common_reformulations: HashMap<String, Vec<String>>,
    pub seasonal_patterns: HashMap<String, f32>,
}

/// Trending topic information
#[derive(Debug, Clone)]
pub struct TrendingTopic {
    pub topic: String,
    pub trend_score: f32,
    pub related_queries: Vec<String>,
    pub time_period: String,
}

impl UserIntelligenceSystem {
    pub fn new(config: UserIntelligenceConfig) -> Self {
        Self {
            config,
            user_profiles: HashMap::new(),
            active_sessions: HashMap::new(),
            interaction_buffer: Vec::new(),
            global_patterns: GlobalUsagePatterns {
                popular_queries: HashMap::new(),
                trending_topics: Vec::new(),
                common_reformulations: HashMap::new(),
                seasonal_patterns: HashMap::new(),
            },
        }
    }

    /// Start a new search session for a user
    pub fn start_session(&mut self, user_id: Uuid, context: SearchContext) -> Uuid {
        let session_id = Uuid::new_v4();
        let session = SearchSession {
            session_id,
            user_id,
            started_at: Utc::now(),
            last_activity: Utc::now(),
            queries: Vec::new(),
            context,
            current_task: None,
        };
        
        self.active_sessions.insert(session_id, session);
        session_id
    }

    /// Record a user interaction
    pub async fn record_interaction(
        &mut self,
        interaction: UserInteraction,
        session_id: Option<Uuid>,
    ) -> Result<()> {
        // Add to buffer for batch processing
        self.interaction_buffer.push(interaction.clone());
        
        // Update active session if provided
        if let Some(sid) = session_id {
            // Clone session for task inference to avoid borrowing issues
            let needs_task_update = if let Some(session) = self.active_sessions.get_mut(&sid) {
                session.last_activity = Utc::now();
                
                // Add to current query if exists
                if let Some(last_query) = session.queries.last_mut() {
                    last_query.interactions.push(interaction.clone());
                }
                
                true
            } else { false };
            
            // Update task inference separately
            if needs_task_update {
                self.update_task_inference_for_session(sid, &interaction).await?;
            }
        }
        
        // Process interaction for learning
        self.process_interaction_for_learning(&interaction).await?;
        
        Ok(())
    }

    /// Record a search query in a session
    pub async fn record_query(
        &mut self,
        session_id: Uuid,
        query: String,
        results_count: usize,
    ) -> Result<()> {
        // Check reformulation outside of mutable borrow
        let reformulated_from = if let Some(session) = self.active_sessions.get(&session_id) {
            self.detect_reformulation(&session.queries, &query)
        } else { None };
        
        if let Some(session) = self.active_sessions.get_mut(&session_id) {
            let session_query = SessionQuery {
                query: query.clone(),
                timestamp: Utc::now(),
                results_shown: results_count,
                interactions: Vec::new(),
                reformulated_from,
            };
            
            session.queries.push(session_query);
            session.last_activity = Utc::now();
        }
        
        // Update global patterns
        *self.global_patterns.popular_queries.entry(query).or_insert(0) += 1;
        
        // Learn from query patterns after session update
        self.learn_from_query_pattern_for_session(session_id).await?;
        
        Ok(())
    }

    /// Get personalized ranking weights for a user
    pub async fn get_personalized_ranking_weights(&self, user_id: Uuid) -> Result<HashMap<String, f32>> {
        let mut weights = HashMap::new();
        
        if let Some(profile) = self.user_profiles.get(&user_id) {
            let prefs = &profile.search_preferences.ranking_preferences;
            
            weights.insert("recency".to_string(), prefs.recency_weight);
            weights.insert("quality".to_string(), prefs.quality_weight);
            weights.insert("text_match".to_string(), prefs.text_vs_semantic_weight);
            weights.insert("diversity".to_string(), prefs.diversity_preference);
            weights.insert("context".to_string(), prefs.context_locality_weight);
        } else {
            // Default weights for new users
            weights.insert("recency".to_string(), 0.15);
            weights.insert("quality".to_string(), 0.05);
            weights.insert("text_match".to_string(), 0.6);
            weights.insert("diversity".to_string(), 0.5);
            weights.insert("context".to_string(), 0.2);
        }
        
        Ok(weights)
    }

    /// Get query suggestions based on user patterns
    pub async fn get_query_suggestions(
        &self,
        user_id: Uuid,
        partial_query: &str,
        context: &SearchContext,
    ) -> Result<Vec<QuerySuggestion>> {
        let mut suggestions = Vec::new();
        
        // Personal query history suggestions
        if let Some(profile) = self.user_profiles.get(&user_id) {
            suggestions.extend(self.get_personal_suggestions(profile, partial_query, context).await?);
        }
        
        // Global popular queries
        suggestions.extend(self.get_popular_suggestions(partial_query).await?);
        
        // Context-aware suggestions
        suggestions.extend(self.get_contextual_suggestions(partial_query, context).await?);
        
        // Sort by relevance and confidence
        suggestions.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Take top 10 suggestions
        suggestions.truncate(10);
        
        Ok(suggestions)
    }

    /// End a search session and process learnings
    pub async fn end_session(&mut self, session_id: Uuid) -> Result<()> {
        if let Some(session) = self.active_sessions.remove(&session_id) {
            // Process session for learning
            self.process_session_for_learning(&session).await?;
            
            // Update user profile
            self.update_user_profile_from_session(&session).await?;
        }
        
        Ok(())
    }

    /// Get user's current preferences for result filtering
    pub async fn get_user_preferences(&self, user_id: Uuid) -> Result<UserSearchPreferences> {
        if let Some(profile) = self.user_profiles.get(&user_id) {
            Ok(UserSearchPreferences {
                preferred_file_types: profile.search_preferences.file_type_preferences.clone(),
                max_results: profile.search_preferences.typical_results_viewed,
                snippet_length: profile.search_preferences.preferred_snippet_length,
                ranking_weights: self.get_personalized_ranking_weights(user_id).await?,
            })
        } else {
            // Default preferences for new users
            Ok(UserSearchPreferences::default())
        }
    }

    // Private helper methods
    async fn process_interaction_for_learning(&mut self, interaction: &UserInteraction) -> Result<()> {
        // Get user_id from session context or create a default user
        let user_id = if let Some(session) = self.active_sessions.get(&interaction.session_id) {
            session.user_id
        } else {
            // Create a default user if session not found
            Uuid::new_v4()
        };
        
        // Create profile if doesn't exist
        if !self.user_profiles.contains_key(&user_id) {
            let profile = self.create_default_profile();
            self.user_profiles.insert(user_id, profile);
        }
        
        // Clone the profile for modification to avoid borrowing issues
        let mut profile = self.user_profiles.get(&user_id).unwrap().clone();
        
        // Learn from interaction type
        match interaction.interaction_type {
            InteractionType::Click => {
                // Update click-through patterns
                self.update_click_patterns(&mut profile, interaction).await?;
            },
            InteractionType::Dwell => {
                // Update dwell time patterns
                self.update_dwell_patterns(&mut profile, interaction).await?;
            },
            InteractionType::Bookmark => {
                // Strong positive signal
                self.update_preference_signals(&mut profile, interaction, 1.0).await?;
            },
            _ => {}
        }
        
        // Update the profile back
        self.user_profiles.insert(user_id, profile);
        
        Ok(())
    }

    async fn update_task_inference_for_session(&mut self, session_id: Uuid, interaction: &UserInteraction) -> Result<()> {
        if let Some(session) = self.active_sessions.get_mut(&session_id) {
            // Simple task inference based on interaction patterns
            let task_type = match interaction.interaction_type {
                InteractionType::Click if session.queries.len() > 3 => TaskType::Research,
                InteractionType::Dwell if session.queries.len() == 1 => TaskType::Reference,
                InteractionType::Bookmark => TaskType::Learning,
                _ => TaskType::Reference,
            };
            
            session.current_task = Some(InferredTask {
                task_type,
                confidence: 0.7,
                context_clues: vec!["interaction_pattern".to_string()],
                estimated_completion: 0.5,
            });
        }
        
        Ok(())
    }

    async fn update_task_inference_simple(&mut self, session: &mut SearchSession, interaction: &UserInteraction) -> Result<()> {
        // Simple task inference based on interaction patterns
        let task_type = match interaction.interaction_type {
            InteractionType::Click if session.queries.len() > 3 => TaskType::Research,
            InteractionType::Dwell if session.queries.len() == 1 => TaskType::Reference,
            InteractionType::Bookmark => TaskType::Learning,
            _ => TaskType::Reference,
        };
        
        session.current_task = Some(InferredTask {
            task_type,
            confidence: 0.7,
            context_clues: vec!["interaction_pattern".to_string()],
            estimated_completion: 0.5,
        });
        
        Ok(())
    }

    fn detect_reformulation(&self, queries: &[SessionQuery], new_query: &str) -> Option<String> {
        if let Some(last_query) = queries.last() {
            let similarity = self.calculate_query_similarity(&last_query.query, new_query);
            if similarity > 0.5 && similarity < 0.95 {
                return Some(last_query.query.clone());
            }
        }
        None
    }

    fn calculate_query_similarity(&self, query1: &str, query2: &str) -> f32 {
        // Simple word overlap similarity
        let words1: std::collections::HashSet<&str> = query1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = query2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
    }

    async fn learn_from_query_pattern_for_session(&mut self, session_id: Uuid) -> Result<()> {
        if let Some(session) = self.active_sessions.get(&session_id) {
            // Analyze query patterns and update learning
            if session.queries.len() >= 2 {
                // Look for reformulation patterns
                let last_two = &session.queries[session.queries.len()-2..];
                if let [_prev, current] = last_two {
                    if let Some(reformulated_from) = &current.reformulated_from {
                        // Record successful reformulation pattern
                        self.global_patterns.common_reformulations
                            .entry(reformulated_from.clone())
                            .or_insert_with(Vec::new)
                            .push(current.query.clone());
                    }
                }
            }
        }
        
        Ok(())
    }

    async fn learn_from_query_pattern_simple(&mut self, session: &SearchSession) -> Result<()> {
        // Analyze query patterns and update learning
        if session.queries.len() >= 2 {
            // Look for reformulation patterns
            let last_two = &session.queries[session.queries.len()-2..];
            if let [_prev, current] = last_two {
                if let Some(reformulated_from) = &current.reformulated_from {
                    // Record successful reformulation pattern
                    self.global_patterns.common_reformulations
                        .entry(reformulated_from.clone())
                        .or_insert_with(Vec::new)
                        .push(current.query.clone());
                }
            }
        }
        
        Ok(())
    }

    async fn get_personal_suggestions(&self, profile: &UserProfile, partial: &str, _context: &SearchContext) -> Result<Vec<QuerySuggestion>> {
        let mut suggestions = Vec::new();
        
        // Look through user's topic preferences for matches
        for (topic, weight) in &profile.search_preferences.topic_preferences {
            if topic.starts_with(partial) {
                suggestions.push(QuerySuggestion {
                    query: topic.clone(),
                    confidence: weight * 0.8, // Personal suggestions get high confidence
                    source: SuggestionSource::Personal,
                    context_relevance: 0.9,
                });
            }
        }
        
        Ok(suggestions)
    }

    async fn get_popular_suggestions(&self, partial: &str) -> Result<Vec<QuerySuggestion>> {
        let mut suggestions = Vec::new();
        
        for (query, count) in &self.global_patterns.popular_queries {
            if query.starts_with(partial) {
                let confidence = (*count as f32).ln() / 10.0; // Log scale for popularity
                suggestions.push(QuerySuggestion {
                    query: query.clone(),
                    confidence: confidence.min(1.0),
                    source: SuggestionSource::Popular,
                    context_relevance: 0.5,
                });
            }
        }
        
        Ok(suggestions)
    }

    async fn get_contextual_suggestions(&self, partial: &str, context: &SearchContext) -> Result<Vec<QuerySuggestion>> {
        let mut suggestions = Vec::new();
        
        // Context-based suggestions based on recent documents or current project
        if let Some(project) = &context.current_project {
            // Create project-specific suggestions
            suggestions.push(QuerySuggestion {
                query: format!("{} in {}", partial, project),
                confidence: 0.7,
                source: SuggestionSource::Contextual,
                context_relevance: 0.95,
            });
        }
        
        Ok(suggestions)
    }

    async fn process_session_for_learning(&mut self, session: &SearchSession) -> Result<()> {
        // Extract patterns from the completed session
        let _session_duration = (session.last_activity - session.started_at).num_seconds() as f32;
        let query_count = session.queries.len();
        
        // Update global session patterns
        if query_count > 0 {
            // This would update learning models in a real implementation
        }
        
        Ok(())
    }

    async fn update_user_profile_from_session(&mut self, session: &SearchSession) -> Result<()> {
        // Create default profile if needed
        if !self.user_profiles.contains_key(&session.user_id) {
            let default_profile = self.create_default_profile();
            self.user_profiles.insert(session.user_id, default_profile);
        }
        
        // Update profile
        if let Some(profile) = self.user_profiles.get_mut(&session.user_id) {
            profile.last_active = Utc::now();
            
            // Update interaction patterns based on session
            if !session.queries.is_empty() {
                let _avg_interactions = session.queries.iter()
                    .map(|q| q.interactions.len())
                    .sum::<usize>() as f32 / session.queries.len() as f32;
                
                // Simple learning update
                profile.interaction_patterns.session_patterns.avg_queries_per_session = 
                    profile.interaction_patterns.session_patterns.avg_queries_per_session * 0.9 + 
                    session.queries.len() as f32 * 0.1;
            }
        }
        
        Ok(())
    }

    fn create_default_profile(&self) -> UserProfile {
        let mut file_type_preferences = HashMap::new();
        file_type_preferences.insert("pdf".to_string(), 0.8);
        file_type_preferences.insert("doc".to_string(), 0.7);
        file_type_preferences.insert("txt".to_string(), 0.6);
        
        UserProfile {
            user_id: Uuid::new_v4(),
            created_at: Utc::now(),
            last_active: Utc::now(),
            search_preferences: SearchPreferences {
                file_type_preferences,
                strategy_preferences: HashMap::new(),
                topic_preferences: HashMap::new(),
                ranking_preferences: RankingPreferences {
                    recency_weight: 0.15,
                    quality_weight: 0.05,
                    text_vs_semantic_weight: 0.6,
                    diversity_preference: 0.5,
                    context_locality_weight: 0.2,
                },
                typical_results_viewed: 10,
                preferred_snippet_length: 200,
            },
            interaction_patterns: InteractionPatterns {
                average_dwell_time: 30.0,
                position_click_rates: HashMap::new(),
                reformulation_patterns: Vec::new(),
                bounce_rate: 0.2,
                session_patterns: SessionPatterns {
                    avg_queries_per_session: 2.5,
                    avg_session_duration: 120.0,
                    query_sequences: Vec::new(),
                    completion_patterns: Vec::new(),
                },
            },
            document_preferences: DocumentPreferences {
                size_preferences: SizePreference {
                    min_preferred_size: 1024,
                    max_preferred_size: 1024 * 1024 * 10, // 10MB
                    optimal_size: 1024 * 100, // 100KB
                },
                language_preferences: HashMap::new(),
                freshness_preferences: FreshnessPreference {
                    recency_importance: 0.3,
                    max_acceptable_age: 365,
                    optimal_age: 30,
                },
                source_preferences: HashMap::new(),
            },
            temporal_patterns: TemporalPatterns {
                hourly_patterns: HashMap::new(),
                daily_patterns: HashMap::new(),
                monthly_patterns: HashMap::new(),
                time_based_preferences: TimeBasedPreferences {
                    time_of_day_preferences: HashMap::new(),
                    weekday_preferences: HashMap::new(),
                },
            },
            context_preferences: ContextPreferences {
                project_associations: HashMap::new(),
                app_context_preferences: HashMap::new(),
                multitasking_patterns: Vec::new(),
            },
        }
    }

    // Placeholder methods for specific learning algorithms
    async fn update_click_patterns(&mut self, _profile: &mut UserProfile, _interaction: &UserInteraction) -> Result<()> {
        // Update click position preferences
        Ok(())
    }

    async fn update_dwell_patterns(&mut self, _profile: &mut UserProfile, _interaction: &UserInteraction) -> Result<()> {
        // Update dwell time patterns
        Ok(())
    }

    async fn update_preference_signals(&mut self, _profile: &mut UserProfile, _interaction: &UserInteraction, _signal_strength: f32) -> Result<()> {
        // Update document/content preferences
        Ok(())
    }
}

/// Query suggestion with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySuggestion {
    pub query: String,
    pub confidence: f32,
    pub source: SuggestionSource,
    pub context_relevance: f32,
}

/// Source of a query suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionSource {
    Personal,
    Popular,
    Contextual,
    Trending,
    Semantic,
}

/// User's search preferences for current session
#[derive(Debug, Clone)]
pub struct UserSearchPreferences {
    pub preferred_file_types: HashMap<String, f32>,
    pub max_results: usize,
    pub snippet_length: usize,
    pub ranking_weights: HashMap<String, f32>,
}

impl Default for UserSearchPreferences {
    fn default() -> Self {
        let mut file_types = HashMap::new();
        file_types.insert("pdf".to_string(), 0.8);
        file_types.insert("doc".to_string(), 0.7);
        file_types.insert("txt".to_string(), 0.6);
        
        let mut ranking_weights = HashMap::new();
        ranking_weights.insert("recency".to_string(), 0.15);
        ranking_weights.insert("quality".to_string(), 0.05);
        ranking_weights.insert("text_match".to_string(), 0.6);
        ranking_weights.insert("diversity".to_string(), 0.5);
        ranking_weights.insert("context".to_string(), 0.2);
        
        Self {
            preferred_file_types: file_types,
            max_results: 10,
            snippet_length: 200,
            ranking_weights,
        }
    }
}