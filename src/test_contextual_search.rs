use crate::error::Result;
use crate::core::contextual_search_engine::{
    ContextualSearchEngine, UserProfile, SearchPreferences, QueryPatterns, 
    ResultPreferences, TemporalPatterns, ContextualQuery, UserContextManager
};
use crate::core::adaptive_learning_engine::{
    AdaptiveLearningEngine, UserLearningModel, LearningState, BehavioralPatterns
};
use crate::core::intelligent_query_processor::IntelligentQueryProcessor;
use crate::database::Database;
use std::collections::HashMap;
use chrono::{Utc, Duration};
use uuid::Uuid;

/// Comprehensive test suite for contextual and personalized search functionality
pub async fn test_contextual_search_pipeline() -> Result<()> {
    println!("🧪 Testing Contextual & Personalized Search Pipeline");
    println!("===================================================");
    
    // Test 1: Contextual Search Engine Creation
    println!("\n1️⃣ Testing Contextual Search Engine Initialization...");
    let database = Database::new("contextual_test.db")?;
    let intelligent_processor = IntelligentQueryProcessor::new(database.clone())?;
    
    let mut contextual_engine = ContextualSearchEngine::new(intelligent_processor)?;
    println!("   ✅ Contextual search engine initialized successfully");
    
    // Test 2: User Profile Management
    println!("\n2️⃣ Testing User Profile Management...");
    test_user_profile_management(&mut contextual_engine).await?;
    
    // Test 3: Behavioral Pattern Learning
    println!("\n3️⃣ Testing Behavioral Pattern Learning...");
    test_behavioral_pattern_learning(&mut contextual_engine).await?;
    
    // Test 4: Contextual Query Processing
    println!("\n4️⃣ Testing Contextual Query Processing...");
    test_contextual_query_processing(&mut contextual_engine).await?;
    
    // Test 5: Personalized Ranking
    println!("\n5️⃣ Testing Personalized Ranking...");
    test_personalized_ranking(&mut contextual_engine).await?;
    
    // Test 6: Session Continuity
    println!("\n6️⃣ Testing Session Continuity...");
    test_session_continuity(&mut contextual_engine).await?;
    
    // Test 7: Adaptive Learning
    println!("\n7️⃣ Testing Adaptive Learning Engine...");
    test_adaptive_learning_engine().await?;
    
    // Test 8: Collaborative Filtering
    println!("\n8️⃣ Testing Collaborative Filtering...");
    test_collaborative_filtering(&mut contextual_engine).await?;
    
    println!("\n✅ All contextual search tests passed!");
    Ok(())
}

/// Test basic functionality of contextual search components
pub fn test_contextual_search_basic_functionality() -> Result<()> {
    println!("🔧 Testing Contextual Search Basic Functionality");
    println!("===============================================");
    
    // Test 1: User Profile Creation
    println!("\n1️⃣ Testing User Profile Creation...");
    let user_id = Uuid::new_v4();
    let user_profile = UserProfile {
        user_id,
        created_at: Utc::now(),
        last_active: Utc::now(),
        total_searches: 0,
        search_preferences: SearchPreferences {
            preferred_file_types: HashMap::new(),
            preferred_authors: HashMap::new(),
            preferred_time_ranges: Vec::new(),
            query_complexity_preference: 0.5,
            boolean_logic_usage: 0.3,
            natural_language_usage: 0.7,
            average_query_length: 5.0,
            refinement_frequency: 0.2,
        },
        domain_expertise: HashMap::new(),
        query_patterns: QueryPatterns {
            common_terms: HashMap::new(),
            common_combinations: HashMap::new(),
            temporal_keywords: HashMap::new(),
            field_usage_patterns: HashMap::new(),
            intent_patterns: HashMap::new(),
            seasonal_patterns: HashMap::new(),
        },
        result_preferences: ResultPreferences {
            preferred_result_count: 10,
            click_through_patterns: HashMap::new(),
            dwell_time_patterns: HashMap::new(),
            relevance_feedback_history: Vec::new(),
            result_format_preferences: HashMap::new(),
            snippet_length_preference: 200,
        },
        temporal_patterns: TemporalPatterns {
            active_hours: vec![0.0; 24],
            active_days: vec![0.0; 7],
            seasonal_activity: vec![0.0; 12],
            query_frequency_by_hour: HashMap::new(),
            topic_shifts_by_time: HashMap::new(),
        },
        collaboration_preferences: crate::core::contextual_search_engine::CollaborationPreferences {
            share_search_history: false,
            accept_recommendations: true,
            contribute_to_collective_intelligence: true,
            privacy_level: crate::core::contextual_search_engine::PrivacyLevel::Moderate,
            trusted_users: Vec::new(),
            blocked_users: Vec::new(),
        },
    };
    println!("   ✅ User profile created with ID: {}", user_id);
    
    // Test 2: Learning Model Creation
    println!("\n2️⃣ Testing Learning Model Creation...");
    let learning_model = UserLearningModel {
        user_id,
        model_version: 1,
        created_at: Utc::now(),
        last_updated: Utc::now(),
        learning_state: LearningState::ColdStart,
        feature_weights: HashMap::new(),
        preference_embeddings: vec![0.0; 100],
        behavioral_patterns: BehavioralPatterns {
            query_reformulation_patterns: Vec::new(),
            click_through_patterns: crate::core::adaptive_learning_engine::ClickThroughPatterns {
                average_ctr_by_position: vec![0.8, 0.6, 0.4, 0.3, 0.2],
                ctr_by_document_type: HashMap::new(),
                ctr_by_query_type: HashMap::new(),
                time_to_click_distribution: Vec::new(),
                multi_click_patterns: Vec::new(),
                result_abandonment_signals: Vec::new(),
            },
            browsing_patterns: crate::core::adaptive_learning_engine::BrowsingPatterns {
                session_duration_patterns: Vec::new(),
                navigation_patterns: Vec::new(),
                result_exploration_depth: 0.0,
                back_button_usage: 0.0,
                scroll_behavior: crate::core::adaptive_learning_engine::ScrollBehavior {
                    average_scroll_depth: 0.0,
                    fast_scroll_frequency: 0.0,
                    pause_patterns: Vec::new(),
                },
                tab_switching_patterns: Vec::new(),
            },
            temporal_usage_patterns: crate::core::adaptive_learning_engine::TemporalUsagePatterns {
                peak_activity_hours: Vec::new(),
                weekend_vs_weekday: 0.0,
                seasonal_variations: HashMap::new(),
                query_spacing_patterns: Vec::new(),
                burst_activity_detection: Vec::new(),
            },
            domain_expertise_evolution: HashMap::new(),
            collaboration_patterns: crate::core::adaptive_learning_engine::CollaborationPatterns {
                sharing_frequency: 0.0,
                recommendation_acceptance_rate: 0.0,
                feedback_quality_score: 0.0,
                community_contribution_level: 0.0,
            },
            learning_velocity: crate::core::adaptive_learning_engine::LearningVelocity {
                preference_change_rate: 0.0,
                adaptation_speed: 0.0,
                concept_drift_sensitivity: 0.0,
                learning_plateau_detection: false,
            },
        },
        adaptation_history: std::collections::VecDeque::new(),
        performance_metrics: crate::core::adaptive_learning_engine::ModelPerformanceMetrics {
            accuracy_score: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            ndcg_score: 0.0,
            user_satisfaction_score: 0.0,
            learning_progress_rate: 0.0,
        },
        confidence_scores: HashMap::new(),
    };
    println!("   ✅ Learning model created in {} state", 
             match learning_model.learning_state {
                 LearningState::ColdStart => "Cold Start",
                 LearningState::Bootstrapping => "Bootstrapping",
                 LearningState::Active => "Active",
                 LearningState::Stable => "Stable",
                 LearningState::Adapting => "Adapting",
                 LearningState::Degraded => "Degraded",
             });
    
    // Test 3: Context Manager Initialization
    println!("\n3️⃣ Testing Context Manager Initialization...");
    let context_manager = UserContextManager {
        user_profiles: HashMap::new(),
        active_sessions: HashMap::new(),
        context_cache: lru::LruCache::new(std::num::NonZeroUsize::new(100).unwrap()),
    };
    println!("   ✅ Context manager initialized with empty state");
    
    println!("\n✅ All basic functionality tests passed!");
    Ok(())
}

async fn test_user_profile_management(engine: &mut ContextualSearchEngine) -> Result<()> {
    let user_id = Uuid::new_v4();
    
    // Create new user profile
    engine.create_user_profile(user_id).await?;
    println!("   ✅ New user profile created");
    
    // Update user preferences
    let mut preferences = HashMap::new();
    preferences.insert("pdf".to_string(), 0.8);
    preferences.insert("docx".to_string(), 0.6);
    
    engine.update_user_preferences(user_id, preferences).await?;
    println!("   ✅ User preferences updated");
    
    // Track search behavior
    let query = ContextualQuery {
        user_id,
        session_id: "session_1".to_string(),
        original_query: "machine learning tutorials".to_string(),
        timestamp: Utc::now(),
        context: HashMap::new(),
        previous_queries: Vec::new(),
        expected_intent: None,
    };
    
    engine.track_search_behavior(&query).await?;
    println!("   ✅ Search behavior tracked");
    
    // Retrieve user profile
    let profile = engine.get_user_profile(user_id).await?;
    assert_eq!(profile.user_id, user_id);
    println!("   ✅ User profile retrieved successfully");
    
    Ok(())
}

async fn test_behavioral_pattern_learning(engine: &mut ContextualSearchEngine) -> Result<()> {
    let user_id = Uuid::new_v4();
    engine.create_user_profile(user_id).await?;
    
    // Simulate multiple search sessions
    for i in 0..10 {
        let query = ContextualQuery {
            user_id,
            session_id: format!("session_{}", i),
            original_query: format!("python tutorial {}", i),
            timestamp: Utc::now() - Duration::hours(i as i64),
            context: HashMap::new(),
            previous_queries: Vec::new(),
            expected_intent: None,
        };
        
        engine.track_search_behavior(&query).await?;
        
        // Simulate result interactions
        let interaction = crate::core::contextual_search_engine::ResultInteraction {
            user_id,
            query_id: format!("query_{}", i),
            document_id: format!("doc_{}", i % 3),
            interaction_type: crate::core::contextual_search_engine::InteractionType::Click,
            timestamp: Utc::now() - Duration::hours(i as i64),
            dwell_time_seconds: Some(30 + i * 10),
            scroll_depth: Some(0.8),
            follow_up_actions: Vec::new(),
        };
        
        engine.record_result_interaction(interaction).await?;
    }
    
    println!("   ✅ Behavioral patterns learned from 10 search sessions");
    
    // Analyze learned patterns
    let patterns = engine.analyze_user_patterns(user_id).await?;
    assert!(!patterns.common_terms.is_empty());
    println!("   ✅ Pattern analysis completed: {} common terms identified", 
             patterns.common_terms.len());
    
    Ok(())
}

async fn test_contextual_query_processing(engine: &mut ContextualSearchEngine) -> Result<()> {
    let user_id = Uuid::new_v4();
    engine.create_user_profile(user_id).await?;
    
    // Build up some context through previous searches
    let previous_queries = vec![
        "python programming",
        "machine learning basics",
        "neural networks tutorial",
    ];
    
    for (i, prev_query) in previous_queries.iter().enumerate() {
        let query = ContextualQuery {
            user_id,
            session_id: "contextual_session".to_string(),
            original_query: prev_query.to_string(),
            timestamp: Utc::now() - Duration::minutes((previous_queries.len() - i) as i64 * 10),
            context: HashMap::new(),
            previous_queries: Vec::new(),
            expected_intent: None,
        };
        
        engine.track_search_behavior(&query).await?;
    }
    
    // Process a contextual query
    let contextual_query = ContextualQuery {
        user_id,
        session_id: "contextual_session".to_string(),
        original_query: "advanced techniques".to_string(),
        timestamp: Utc::now(),
        context: HashMap::new(),
        previous_queries: previous_queries.iter().map(|s| s.to_string()).collect(),
        expected_intent: None,
    };
    
    let result = engine.process_contextual_query(contextual_query).await?;
    
    println!("   ✅ Contextual query processed successfully");
    println!("      Query expansion applied: {}", !result.expanded_terms.is_empty());
    println!("      Context influence: {:.2}", result.context_influence_score);
    println!("      Total results: {}", result.document_ids.len());
    
    Ok(())
}

async fn test_personalized_ranking(engine: &mut ContextualSearchEngine) -> Result<()> {
    let user_id = Uuid::new_v4();
    engine.create_user_profile(user_id).await?;
    
    // Simulate user preferring certain document types and authors
    let mut preferences = HashMap::new();
    preferences.insert("pdf".to_string(), 0.9);
    preferences.insert("tutorial".to_string(), 0.8);
    
    engine.update_user_preferences(user_id, preferences).await?;
    
    // Create mock search results
    let document_ids = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
    
    // Test personalized ranking
    let ranked_results = engine.apply_personalized_ranking(
        user_id,
        &document_ids.iter().map(|s| s.to_string()).collect(),
        "machine learning"
    ).await?;
    
    println!("   ✅ Personalized ranking applied");
    println!("      Original order: {:?}", document_ids);
    println!("      Personalized order: {:?}", ranked_results.document_ids);
    println!("      Ranking confidence: {:.2}", ranked_results.ranking_confidence);
    
    assert_eq!(ranked_results.document_ids.len(), document_ids.len());
    
    Ok(())
}

async fn test_session_continuity(engine: &mut ContextualSearchEngine) -> Result<()> {
    let user_id = Uuid::new_v4();
    let session_id = "continuity_session";
    
    engine.create_user_profile(user_id).await?;
    
    // Start a search session
    engine.start_search_session(user_id, session_id.to_string()).await?;
    println!("   ✅ Search session started");
    
    // Perform multiple related queries in the session
    let session_queries = vec![
        "python basics",
        "python functions",
        "python classes",
        "python modules",
    ];
    
    for (i, query_text) in session_queries.iter().enumerate() {
        let query = ContextualQuery {
            user_id,
            session_id: session_id.to_string(),
            original_query: query_text.to_string(),
            timestamp: Utc::now() - Duration::minutes((session_queries.len() - i) as i64 * 5),
            context: HashMap::new(),
            previous_queries: session_queries[..i].iter().map(|s| s.to_string()).collect(),
            expected_intent: None,
        };
        
        let result = engine.process_contextual_query(query).await?;
        println!("      Query {}: '{}' -> {} results", i + 1, query_text, result.document_ids.len());
    }
    
    // Test session context retrieval
    let session_context = engine.get_session_context(session_id).await?;
    assert_eq!(session_context.query_history.len(), session_queries.len());
    println!("   ✅ Session context maintained across {} queries", session_queries.len());
    
    // End the session
    engine.end_search_session(session_id).await?;
    println!("   ✅ Search session ended successfully");
    
    Ok(())
}

async fn test_adaptive_learning_engine() -> Result<()> {
    let mut learning_engine = AdaptiveLearningEngine::new()?;
    
    let user_id = Uuid::new_v4();
    
    // Initialize user learning model
    learning_engine.initialize_user_model(user_id).await?;
    println!("   ✅ User learning model initialized");
    
    // Simulate learning from user feedback
    for i in 0..20 {
        let feedback = crate::core::contextual_search_engine::RelevanceFeedback {
            user_id,
            query: format!("query_{}", i),
            document_id: format!("doc_{}", i % 5),
            relevance_score: if i % 3 == 0 { 5.0 } else { 3.0 },
            feedback_type: crate::core::contextual_search_engine::FeedbackType::Explicit,
            timestamp: Utc::now() - Duration::hours(i as i64),
            context: HashMap::new(),
        };
        
        learning_engine.process_user_feedback(feedback).await?;
    }
    
    println!("   ✅ Processed 20 feedback instances");
    
    // Test model adaptation
    let adaptation_result = learning_engine.adapt_user_model(user_id).await?;
    println!("   ✅ Model adaptation completed");
    println!("      Performance improvement: {:.2}%", adaptation_result.performance_improvement * 100.0);
    println!("      Learning state: {:?}", adaptation_result.new_learning_state);
    
    // Test concept drift detection
    let drift_detected = learning_engine.detect_concept_drift(user_id).await?;
    println!("   ✅ Concept drift detection: {}", if drift_detected { "Drift detected" } else { "No drift" });
    
    // Test cold start handling
    let new_user_id = Uuid::new_v4();
    let cold_start_recommendations = learning_engine.handle_cold_start(new_user_id).await?;
    println!("   ✅ Cold start handled: {} recommendations generated", 
             cold_start_recommendations.len());
    
    Ok(())
}

async fn test_collaborative_filtering(engine: &mut ContextualSearchEngine) -> Result<()> {
    // Create multiple users with similar interests
    let user_ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
    
    for user_id in &user_ids {
        engine.create_user_profile(*user_id).await?;
        
        // Simulate similar search patterns
        for i in 0..5 {
            let query = ContextualQuery {
                user_id: *user_id,
                session_id: format!("collab_session_{}", i),
                original_query: format!("machine learning topic {}", i),
                timestamp: Utc::now() - Duration::hours(i as i64),
                context: HashMap::new(),
                previous_queries: Vec::new(),
                expected_intent: None,
            };
            
            engine.track_search_behavior(&query).await?;
        }
    }
    
    println!("   ✅ Created 5 users with similar search patterns");
    
    // Test collaborative recommendations
    let target_user = user_ids[0];
    let recommendations = engine.get_collaborative_recommendations(target_user).await?;
    
    println!("   ✅ Collaborative filtering completed");
    println!("      Recommendations generated: {}", recommendations.len());
    
    if !recommendations.is_empty() {
        println!("      Top recommendation: {}", recommendations[0].content);
        println!("      Confidence score: {:.2}", recommendations[0].confidence_score);
    }
    
    // Test similar user identification
    let similar_users = engine.find_similar_users(target_user, 3).await?;
    println!("   ✅ Similar users identified: {}", similar_users.len());
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_full_contextual_pipeline() {
        if let Err(e) = test_contextual_search_pipeline().await {
            panic!("Contextual search pipeline test failed: {}", e);
        }
    }

    #[test]
    fn test_basic_contextual_functionality() {
        if let Err(e) = test_contextual_search_basic_functionality() {
            panic!("Basic contextual functionality test failed: {}", e);
        }
    }

    #[tokio::test]
    async fn test_user_learning_progression() {
        let mut learning_engine = AdaptiveLearningEngine::new().unwrap();
        let user_id = Uuid::new_v4();
        
        // Test learning state progression
        learning_engine.initialize_user_model(user_id).await.unwrap();
        
        // Should start in ColdStart state
        let initial_model = learning_engine.get_user_model(user_id).await.unwrap();
        assert_eq!(initial_model.learning_state, LearningState::ColdStart);
        
        // After some feedback, should progress to Bootstrapping
        for i in 0..10 {
            let feedback = crate::core::contextual_search_engine::RelevanceFeedback {
                user_id,
                query: format!("test_query_{}", i),
                document_id: format!("test_doc_{}", i),
                relevance_score: 4.0,
                feedback_type: crate::core::contextual_search_engine::FeedbackType::Explicit,
                timestamp: Utc::now(),
                context: HashMap::new(),
            };
            
            learning_engine.process_user_feedback(feedback).await.unwrap();
        }
        
        learning_engine.adapt_user_model(user_id).await.unwrap();
        
        let updated_model = learning_engine.get_user_model(user_id).await.unwrap();
        assert!(matches!(updated_model.learning_state, 
                        LearningState::Bootstrapping | LearningState::Active));
    }
}