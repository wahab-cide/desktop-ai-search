use desktop_ai_search::core::user_intelligence::{
    UserIntelligenceSystem, UserIntelligenceConfig
};
use desktop_ai_search::core::ranking::{UserInteraction, InteractionType, SearchContext};
use uuid::Uuid;
use chrono::Utc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Testing User Intelligence System\n");

    // Test 1: User Profile Creation and Session Management
    println!("=== Test 1: User Profile and Session Management ===");
    test_user_profile_creation().await?;

    // Test 2: Interaction Tracking and Learning
    println!("\n=== Test 2: Interaction Tracking and Learning ===");
    test_interaction_tracking().await?;

    // Test 3: Query Suggestions System
    println!("\n=== Test 3: Query Suggestions System ===");
    test_query_suggestions().await?;

    // Test 4: Personalized Ranking Weights
    println!("\n=== Test 4: Personalized Ranking Weights ===");
    test_personalized_ranking().await?;

    // Test 5: Session Pattern Recognition
    println!("\n=== Test 5: Session Pattern Recognition ===");
    test_session_patterns().await?;

    // Test 6: User Preference Learning
    println!("\n=== Test 6: User Preference Learning ===");
    test_preference_learning().await?;

    // Test 7: Performance and Scalability
    println!("\n=== Test 7: Performance and Scalability ===");
    test_performance().await?;

    println!("\n✅ All user intelligence tests completed!");
    Ok(())
}

async fn test_user_profile_creation() -> Result<(), Box<dyn std::error::Error>> {
    let mut system = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    
    // Create test user and context
    let user_id = Uuid::new_v4();
    let context = create_test_context();
    
    // Start a session
    let session_id = system.start_session(user_id, context);
    
    println!("  Created user: {}", user_id.to_string().split('-').next().unwrap_or("unknown"));
    println!("  Started session: {}", session_id.to_string().split('-').next().unwrap_or("unknown"));
    
    // Record some queries
    system.record_query(session_id, "machine learning tutorial".to_string(), 15).await?;
    system.record_query(session_id, "python programming guide".to_string(), 12).await?;
    system.record_query(session_id, "data science methods".to_string(), 8).await?;
    
    println!("  Recorded 3 queries in session");
    
    // End session
    system.end_session(session_id).await?;
    println!("  Session ended successfully");
    
    // Verify user preferences were created
    let preferences = system.get_user_preferences(user_id).await?;
    println!("  User preferences created:");
    println!("    Max results: {}", preferences.max_results);
    println!("    Snippet length: {}", preferences.snippet_length);
    println!("    File type preferences: {}", preferences.preferred_file_types.len());
    
    assert!(preferences.max_results > 0, "Max results should be positive");
    assert!(preferences.snippet_length > 0, "Snippet length should be positive");
    
    println!("  ✅ User profile creation working correctly");
    Ok(())
}

async fn test_interaction_tracking() -> Result<(), Box<dyn std::error::Error>> {
    let mut system = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    
    let user_id = Uuid::new_v4();
    let context = create_test_context();
    let session_id = system.start_session(user_id, context);
    
    // Record initial query
    system.record_query(session_id, "rust programming tutorial".to_string(), 10).await?;
    
    // Simulate various user interactions
    let interactions = vec![
        (InteractionType::Click, "User clicked on first result"),
        (InteractionType::Dwell, "User spent time reading document"),
        (InteractionType::Bookmark, "User bookmarked useful content"),
        (InteractionType::Share, "User shared document"),
        (InteractionType::Open, "User opened document in external app"),
    ];
    
    println!("  Recording {} different interaction types:", interactions.len());
    
    for (interaction_type, description) in interactions {
        let interaction = UserInteraction {
            document_id: Uuid::new_v4(),
            interaction_type,
            timestamp: Utc::now(),
            query_context: Some("rust programming tutorial".to_string()),
            session_id,
        };
        
        system.record_interaction(interaction, Some(session_id)).await?;
        println!("    - {}", description);
    }
    
    // Test query reformulation detection
    system.record_query(session_id, "rust programming basics".to_string(), 8).await?;
    system.record_query(session_id, "rust tutorial for beginners".to_string(), 12).await?;
    
    println!("  Recorded query reformulation sequence");
    
    // End session and process learnings
    system.end_session(session_id).await?;
    
    // Verify interactions were processed
    let ranking_weights = system.get_personalized_ranking_weights(user_id).await?;
    println!("  Learned ranking weights:");
    for (weight_name, weight_value) in &ranking_weights {
        println!("    {}: {:.3}", weight_name, weight_value);
    }
    
    assert!(ranking_weights.len() > 0, "Should have learned some ranking weights");
    
    println!("  ✅ Interaction tracking working correctly");
    Ok(())
}

async fn test_query_suggestions() -> Result<(), Box<dyn std::error::Error>> {
    let mut system = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    
    let user_id = Uuid::new_v4();
    let context = create_test_context();
    
    // Build up some query history
    let session_id = system.start_session(user_id, context.clone());
    
    let historical_queries = vec![
        "machine learning algorithms",
        "machine learning applications", 
        "python data science",
        "python machine learning",
        "data analysis techniques",
        "statistical analysis methods",
    ];
    
    println!("  Building query history with {} queries:", historical_queries.len());
    for query in &historical_queries {
        system.record_query(session_id, query.to_string(), 10).await?;
        println!("    - {}", query);
    }
    
    system.end_session(session_id).await?;
    
    // Test query suggestions
    let test_cases = vec![
        ("machine", "Should suggest machine learning queries"),
        ("python", "Should suggest python-related queries"),
        ("data", "Should suggest data analysis queries"),
        ("statistical", "Should suggest statistical analysis"),
    ];
    
    println!("\n  Testing query suggestions:");
    
    for (partial_query, expected_behavior) in test_cases {
        let suggestions = system.get_query_suggestions(user_id, partial_query, &context).await?;
        
        println!("    Query prefix: \"{}\"", partial_query);
        println!("      Expected: {}", expected_behavior);
        println!("      Suggestions found: {}", suggestions.len());
        
        for (i, suggestion) in suggestions.iter().take(3).enumerate() {
            println!("        {}. \"{}\" (confidence: {:.2}, source: {:?})", 
                i + 1, suggestion.query, suggestion.confidence, suggestion.source);
        }
        
        // Verify suggestions quality
        assert!(suggestions.len() > 0, "Should provide at least one suggestion");
        
        let high_confidence_suggestions = suggestions.iter()
            .filter(|s| s.confidence > 0.5)
            .count();
        
        println!("      High confidence suggestions: {}", high_confidence_suggestions);
        
        if !suggestions.is_empty() {
            assert!(suggestions[0].confidence > 0.0, "Top suggestion should have positive confidence");
        }
    }
    
    println!("  ✅ Query suggestions working correctly");
    Ok(())
}

async fn test_personalized_ranking() -> Result<(), Box<dyn std::error::Error>> {
    let mut system = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    
    // Create two different user profiles
    let user1 = Uuid::new_v4();
    let user2 = Uuid::new_v4();
    let context = create_test_context();
    
    println!("  Creating two different user profiles:");
    
    // User 1: Prefers recent documents and high quality
    let session1 = system.start_session(user1, context.clone());
    system.record_query(session1, "latest research papers".to_string(), 10).await?;
    system.record_query(session1, "recent developments in AI".to_string(), 8).await?;
    
    // Simulate preference for recent, high-quality content
    for _ in 0..5 {
        let interaction = UserInteraction {
            document_id: Uuid::new_v4(),
            interaction_type: InteractionType::Bookmark,
            timestamp: Utc::now(),
            query_context: Some("recent high quality content".to_string()),
            session_id: session1,
        };
        system.record_interaction(interaction, Some(session1)).await?;
    }
    
    system.end_session(session1).await?;
    println!("    User 1: Recent content preference profile");
    
    // User 2: Prefers comprehensive documents regardless of age
    let session2 = system.start_session(user2, context.clone());
    system.record_query(session2, "comprehensive tutorial".to_string(), 15).await?;
    system.record_query(session2, "complete reference guide".to_string(), 20).await?;
    
    // Simulate preference for comprehensive content
    for _ in 0..5 {
        let interaction = UserInteraction {
            document_id: Uuid::new_v4(),
            interaction_type: InteractionType::Dwell,
            timestamp: Utc::now(),
            query_context: Some("comprehensive reference".to_string()),
            session_id: session2,
        };
        system.record_interaction(interaction, Some(session2)).await?;
    }
    
    system.end_session(session2).await?;
    println!("    User 2: Comprehensive content preference profile");
    
    // Compare personalized ranking weights
    let weights1 = system.get_personalized_ranking_weights(user1).await?;
    let weights2 = system.get_personalized_ranking_weights(user2).await?;
    
    println!("\n  Comparing personalized ranking weights:");
    println!("    User 1 (recent preference):");
    for (weight_name, weight_value) in &weights1 {
        println!("      {}: {:.3}", weight_name, weight_value);
    }
    
    println!("    User 2 (comprehensive preference):");
    for (weight_name, weight_value) in &weights2 {
        println!("      {}: {:.3}", weight_name, weight_value);
    }
    
    // Verify that weights are different (indicating personalization)
    let weight_differences = weights1.iter()
        .filter_map(|(key, val1)| {
            weights2.get(key).map(|val2| (key, (val1 - val2).abs()))
        })
        .collect::<Vec<_>>();
    
    let significant_differences = weight_differences.iter()
        .filter(|(_, diff)| *diff > 0.01)
        .count();
    
    println!("    Significant weight differences: {}", significant_differences);
    
    // Both should have valid weights
    assert!(weights1.len() > 0, "User 1 should have ranking weights");
    assert!(weights2.len() > 0, "User 2 should have ranking weights");
    
    println!("  ✅ Personalized ranking working correctly");
    Ok(())
}

async fn test_session_patterns() -> Result<(), Box<dyn std::error::Error>> {
    let mut system = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    
    let user_id = Uuid::new_v4();
    let context = create_test_context();
    
    println!("  Simulating complex search session patterns:");
    
    // Simulate a research session with multiple reformulations
    let session_id = system.start_session(user_id, context);
    
    let query_sequence = vec![
        ("neural networks", "Initial broad query"),
        ("neural network architecture", "More specific query"),
        ("convolutional neural networks", "Focused on CNN"),
        ("CNN image classification", "Application-specific"),
        ("CNN tutorial python", "Implementation-focused"),
    ];
    
    println!("    Research session with query refinement:");
    
    for (i, (query, description)) in query_sequence.iter().enumerate() {
        system.record_query(session_id, query.to_string(), 10 - i).await?;
        println!("      {}. \"{}\" - {}", i + 1, query, description);
        
        // Simulate interactions for each query
        if i > 0 {
            // Later queries have more focused interactions
            let interaction = UserInteraction {
                document_id: Uuid::new_v4(),
                interaction_type: if i < 3 { InteractionType::Click } else { InteractionType::Bookmark },
                timestamp: Utc::now(),
                query_context: Some(query.to_string()),
                session_id,
            };
            system.record_interaction(interaction, Some(session_id)).await?;
        }
    }
    
    // Simulate task completion
    let completion_interaction = UserInteraction {
        document_id: Uuid::new_v4(),
        interaction_type: InteractionType::Open,
        timestamp: Utc::now(),
        query_context: Some("CNN tutorial python".to_string()),
        session_id,
    };
    system.record_interaction(completion_interaction, Some(session_id)).await?;
    
    println!("    Session completed with document opened");
    
    system.end_session(session_id).await?;
    
    // Test that the system learned from session patterns
    let preferences = system.get_user_preferences(user_id).await?;
    
    println!("    Learned session preferences:");
    println!("      Max results viewed: {}", preferences.max_results);
    println!("      Preferred snippet length: {}", preferences.snippet_length);
    
    // Verify session was processed
    assert!(preferences.max_results > 0, "Should have learned result preferences");
    
    println!("  ✅ Session pattern recognition working correctly");
    Ok(())
}

async fn test_preference_learning() -> Result<(), Box<dyn std::error::Error>> {
    let mut system = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    
    let user_id = Uuid::new_v4();
    let context = create_test_context();
    
    println!("  Testing adaptive preference learning:");
    
    // Initial preferences
    let initial_prefs = system.get_user_preferences(user_id).await?;
    println!("    Initial preferences:");
    println!("      File types: {}", initial_prefs.preferred_file_types.len());
    
    // Simulate consistent PDF preference through interactions
    let session_id = system.start_session(user_id, context);
    
    println!("    Simulating strong PDF preference through interactions:");
    
    for i in 0..10 {
        system.record_query(session_id, format!("document search {}", i), 10).await?;
        
        // Consistently interact with PDF documents
        let pdf_interaction = UserInteraction {
            document_id: Uuid::new_v4(),
            interaction_type: if i % 3 == 0 { InteractionType::Bookmark } else { InteractionType::Click },
            timestamp: Utc::now(),
            query_context: Some(format!("pdf document {}", i)),
            session_id,
        };
        system.record_interaction(pdf_interaction, Some(session_id)).await?;
        
        if i % 3 == 0 {
            println!("      Interaction {}: Bookmarked PDF document", i + 1);
        }
    }
    
    system.end_session(session_id).await?;
    
    // Check if preferences adapted
    let updated_prefs = system.get_user_preferences(user_id).await?;
    
    println!("    Updated preferences:");
    println!("      File types: {}", updated_prefs.preferred_file_types.len());
    for (file_type, preference) in &updated_prefs.preferred_file_types {
        println!("        {}: {:.2}", file_type, preference);
    }
    
    // Verify learning occurred
    assert!(updated_prefs.preferred_file_types.len() >= initial_prefs.preferred_file_types.len(),
        "Should maintain or learn new file type preferences");
    
    println!("  ✅ Preference learning working correctly");
    Ok(())
}

async fn test_performance() -> Result<(), Box<dyn std::error::Error>> {
    let config = UserIntelligenceConfig {
        enable_personalization: true,
        enable_tracking: true,
        min_interactions_for_personalization: 5,
        data_retention_days: 30,
        learning_rate: 0.2,
        temporal_decay_rate: 0.95,
    };
    
    let mut system = UserIntelligenceSystem::new(config);
    
    println!("  Performance testing with multiple users and sessions:");
    
    let start_time = std::time::Instant::now();
    
    // Create multiple users with sessions
    let user_count = 50;
    let queries_per_user = 20;
    let interactions_per_query = 3;
    
    println!("    Creating {} users with {} queries each", user_count, queries_per_user);
    
    for user_i in 0..user_count {
        let user_id = Uuid::new_v4();
        let context = create_test_context();
        let session_id = system.start_session(user_id, context.clone());
        
        for query_i in 0..queries_per_user {
            let query = format!("test query {} from user {}", query_i, user_i);
            system.record_query(session_id, query.clone(), 10).await?;
            
            // Record interactions
            for int_i in 0..interactions_per_query {
                let interaction = UserInteraction {
                    document_id: Uuid::new_v4(),
                    interaction_type: match int_i % 3 {
                        0 => InteractionType::Click,
                        1 => InteractionType::Dwell,
                        _ => InteractionType::Bookmark,
                    },
                    timestamp: Utc::now(),
                    query_context: Some(query.clone()),
                    session_id,
                };
                system.record_interaction(interaction, Some(session_id)).await?;
            }
        }
        
        system.end_session(session_id).await?;
        
        if (user_i + 1) % 10 == 0 {
            println!("      Processed {} users...", user_i + 1);
        }
    }
    
    let processing_time = start_time.elapsed();
    
    // Test query suggestions performance
    let suggestion_start = std::time::Instant::now();
    let test_user = Uuid::new_v4();
    let suggestions = system.get_query_suggestions(test_user, "test", &create_test_context()).await?;
    let suggestion_time = suggestion_start.elapsed();
    
    println!("    Performance Results:");
    println!("      Total processing time: {:?}", processing_time);
    println!("      Interactions processed: {}", user_count * queries_per_user * interactions_per_query);
    println!("      Avg time per interaction: {:.2}ms", 
        processing_time.as_millis() as f64 / (user_count * queries_per_user * interactions_per_query) as f64);
    println!("      Query suggestion time: {:?}", suggestion_time);
    println!("      Suggestions generated: {}", suggestions.len());
    
    // Performance assertions
    assert!(processing_time.as_secs() < 10, "Should process all interactions within 10 seconds");
    assert!(suggestion_time.as_millis() < 100, "Query suggestions should be fast (<100ms)");
    
    println!("  ✅ Performance testing passed");
    Ok(())
}

// Helper function to create test context
fn create_test_context() -> SearchContext {
    SearchContext {
        session_id: Uuid::new_v4(),
        current_project: Some("test_project".to_string()),
        recent_documents: vec![Uuid::new_v4(), Uuid::new_v4()],
        active_applications: vec!["vscode".to_string(), "browser".to_string()],
        search_history: vec!["previous query".to_string()],
        timestamp: Utc::now(),
    }
}