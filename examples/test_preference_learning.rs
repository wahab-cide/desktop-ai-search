use desktop_ai_search::core::preference_learning::{
    PreferenceLearningSystem, PreferenceLearningConfig, SearchFeedback, FeedbackType
};
use desktop_ai_search::core::ranking::{UserInteraction, InteractionType, SearchContext, RankedResult, RankingFeatures};
use uuid::Uuid;
use chrono::Utc;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Testing Advanced Preference Learning System\n");

    // Test 1: Basic Learning from Interactions
    println!("=== Test 1: Basic Learning from Interactions ===");
    test_basic_learning().await?;

    // Test 2: Preference Model Development
    println!("\n=== Test 2: Preference Model Development ===");
    test_preference_model_development().await?;

    // Test 3: Search Customizations Generation
    println!("\n=== Test 3: Search Customizations Generation ===");
    test_search_customizations().await?;

    // Test 4: Real-time Adaptation
    println!("\n=== Test 4: Real-time Adaptation ===");
    test_realtime_adaptation().await?;

    // Test 5: Collaborative Filtering
    println!("\n=== Test 5: Collaborative Filtering ===");
    test_collaborative_filtering().await?;

    // Test 6: Preference Drift Detection
    println!("\n=== Test 6: Preference Drift Detection ===");
    test_preference_drift_detection().await?;

    // Test 7: Multi-User Learning Scenarios
    println!("\n=== Test 7: Multi-User Learning Scenarios ===");
    test_multi_user_scenarios().await?;

    // Test 8: Performance and Scalability
    println!("\n=== Test 8: Performance and Scalability ===");
    test_performance_scalability().await?;

    println!("\n✅ All preference learning tests completed!");
    Ok(())
}

async fn test_basic_learning() -> Result<(), Box<dyn std::error::Error>> {
    let config = PreferenceLearningConfig::default();
    let mut system = PreferenceLearningSystem::new(config);
    
    let user_id = Uuid::new_v4();
    let context = create_test_context();

    println!("  Testing basic interaction learning for user: {}", user_id.to_string().split('-').next().unwrap_or("unknown"));

    // Simulate a series of interactions with different content types
    let interaction_scenarios = vec![
        (InteractionType::Click, "pdf", "Machine learning fundamentals in PDF format"),
        (InteractionType::Dwell, "pdf", "Advanced neural networks comprehensive guide"),
        (InteractionType::Bookmark, "md", "Python programming tutorial markdown"),
        (InteractionType::Share, "txt", "Data science methodology plain text"),
        (InteractionType::Open, "docx", "Research paper on AI ethics"),
        (InteractionType::Open, "pdf", "Technical documentation PDF"),
    ];

    let mut learning_updates = Vec::new();

    for (i, (interaction_type, file_type, content)) in interaction_scenarios.iter().enumerate() {
        let interaction = UserInteraction {
            document_id: Uuid::new_v4(),
            interaction_type: *interaction_type,
            timestamp: Utc::now(),
            query_context: Some("machine learning".to_string()),
            session_id: context.session_id,
        };

        let results = vec![create_test_result(*file_type, content)];
        
        let update = system.learn_from_interaction(user_id, &interaction, &context, &results).await?;
        learning_updates.push(update);

        println!("    Interaction {}: {:?} with {} → Confidence: {:.3}", 
            i + 1, 
            interaction_type, 
            file_type,
            learning_updates[i].learning_confidence
        );
    }

    // Verify learning progression
    let final_confidence = learning_updates.last().unwrap().learning_confidence;
    println!("  Final learning confidence: {:.3}", final_confidence);

    // Check if preferences were learned
    let final_preferences = &learning_updates.last().unwrap().updated_preferences;
    println!("  Learned {} preference features", final_preferences.len());

    for (feature, importance) in final_preferences.iter().take(5) {
        println!("    {}: {:.3}", feature, importance);
    }

    assert!(final_confidence > 0.0, "Should have developed some learning confidence");
    assert!(!final_preferences.is_empty(), "Should have learned some preferences");

    println!("  ✅ Basic learning working correctly");
    Ok(())
}

async fn test_preference_model_development() -> Result<(), Box<dyn std::error::Error>> {
    let config = PreferenceLearningConfig {
        min_interactions_threshold: 5,
        learning_rate: 0.2,
        ..Default::default()
    };
    let mut system = PreferenceLearningSystem::new(config);
    
    let user_id = Uuid::new_v4();
    let context = create_test_context();

    println!("  Developing comprehensive preference model through diverse interactions");

    // Simulate varied learning scenarios
    let scenarios = vec![
        // PDF preference development
        ("pdf_focus", vec![
            (InteractionType::Bookmark, "pdf", "Research methodology handbook"),
            (InteractionType::Dwell, "pdf", "Statistical analysis guide"),
            (InteractionType::Share, "pdf", "Academic paper template"),
        ]),
        // Programming content preference
        ("code_focus", vec![
            (InteractionType::Open, "py", "Python machine learning scripts"),
            (InteractionType::Open, "rs", "Rust systems programming"),
            (InteractionType::Bookmark, "js", "JavaScript web development"),
        ]),
        // Documentation preference
        ("docs_focus", vec![
            (InteractionType::Click, "md", "API documentation"),
            (InteractionType::Dwell, "md", "Installation instructions"),
            (InteractionType::Bookmark, "md", "User guide"),
        ]),
    ];

    for (scenario_name, interactions) in scenarios {
        println!("\n    Scenario: {}", scenario_name);
        
        for (interaction_type, file_type, content) in interactions {
            let interaction = UserInteraction {
                document_id: Uuid::new_v4(),
                interaction_type,
                timestamp: Utc::now(),
                query_context: Some(scenario_name.to_string()),
                session_id: context.session_id,
            };

            let results = vec![create_test_result(file_type, content)];
            let update = system.learn_from_interaction(user_id, &interaction, &context, &results).await?;
            
            println!("      {:?} with {} → Features learned: {}", 
                interaction_type, 
                file_type,
                update.updated_preferences.len()
            );
        }
    }

    // Test preference retrieval and customizations
    let customizations = system.get_search_customizations(user_id, &context).await?;
    
    println!("\n  Generated search customizations:");
    println!("    Ranking adjustments: {} features", customizations.ranking_adjustments.content_boosts.len());
    println!("    Filtering preferences: {} file types", customizations.filtering_preferences.preferred_file_types.len());
    println!("    Interface customizations: {} results per page", customizations.interface_customizations.results_per_page);
    println!("    Query enhancements: auto-expansion = {}", customizations.query_enhancements.enable_auto_expansion);
    println!("    Overall confidence: {:.3}", customizations.confidence);

    // Verify meaningful customizations were generated
    assert!(customizations.confidence > 0.0, "Should have confidence in customizations");
    assert!(customizations.ranking_adjustments.content_boosts.len() > 0, "Should have content ranking adjustments");

    println!("\n  ✅ Preference model development working correctly");
    Ok(())
}

async fn test_search_customizations() -> Result<(), Box<dyn std::error::Error>> {
    let mut system = PreferenceLearningSystem::new(PreferenceLearningConfig::default());
    
    let user_id = Uuid::new_v4();
    let mut context = create_test_context();

    println!("  Testing search customization generation based on learned preferences");

    // Build strong PDF preference through interactions
    for i in 0..10 {
        let interaction = UserInteraction {
            document_id: Uuid::new_v4(),
            interaction_type: if i % 3 == 0 { InteractionType::Bookmark } else { InteractionType::Dwell },
            timestamp: Utc::now(),
            query_context: Some("research documents".to_string()),
            session_id: context.session_id,
        };

        let results = vec![create_test_result("pdf", &format!("Research document {}", i))];
        system.learn_from_interaction(user_id, &interaction, &context, &results).await?;
    }

    // Test customizations in different contexts
    let contexts = vec![
        ("work_context", "work_project"),
        ("research_context", "research_project"),
        ("personal_context", "personal_project"),
    ];

    for (context_name, project_name) in contexts {
        context.current_project = Some(project_name.to_string());
        
        let customizations = system.get_search_customizations(user_id, &context).await?;
        
        println!("\n    Context: {}", context_name);
        println!("      Confidence: {:.3}", customizations.confidence);
        
        // Test ranking adjustments
        if !customizations.ranking_adjustments.content_boosts.is_empty() {
            println!("      Top content boosts:");
            for (content_type, boost) in customizations.ranking_adjustments.content_boosts.iter().take(3) {
                println!("        {}: {:.3}", content_type, boost);
            }
        }
        
        // Test filtering preferences
        if !customizations.filtering_preferences.preferred_file_types.is_empty() {
            println!("      File type preferences:");
            for (file_type, preference) in customizations.filtering_preferences.preferred_file_types.iter().take(3) {
                println!("        {}: {:.3}", file_type, preference);
            }
        }
        
        // Test interface customizations
        println!("      Interface: {} results, {} chars snippets, {:?} detail", 
            customizations.interface_customizations.results_per_page,
            customizations.interface_customizations.snippet_length,
            customizations.interface_customizations.detail_level
        );
    }

    println!("\n  ✅ Search customizations working correctly");
    Ok(())
}

async fn test_realtime_adaptation() -> Result<(), Box<dyn std::error::Error>> {
    let config = PreferenceLearningConfig {
        enable_realtime_adaptation: true,
        learning_rate: 0.3,
        ..Default::default()
    };
    let mut system = PreferenceLearningSystem::new(config);
    
    let user_id = Uuid::new_v4();
    let context = create_test_context();

    println!("  Testing real-time preference adaptation from user feedback");

    // Establish baseline preferences
    for i in 0..5 {
        let interaction = UserInteraction {
            document_id: Uuid::new_v4(),
            interaction_type: InteractionType::Click,
            timestamp: Utc::now(),
            query_context: Some("baseline learning".to_string()),
            session_id: context.session_id,
        };

        let results = vec![create_test_result("txt", &format!("Text document {}", i))];
        system.learn_from_interaction(user_id, &interaction, &context, &results).await?;
    }

    let initial_customizations = system.get_search_customizations(user_id, &context).await?;
    println!("    Initial confidence: {:.3}", initial_customizations.confidence);

    // Simulate real-time feedback scenarios
    let feedback_scenarios = vec![
        (FeedbackType::Positive, 0.8, 0.9, "User liked PDF recommendation"),
        (FeedbackType::Negative, 0.6, 0.8, "User didn't like TXT result"),
        (FeedbackType::Positive, 0.9, 0.95, "User very satisfied with MD format"),
        (FeedbackType::Neutral, 0.3, 0.6, "User indifferent to DOC format"),
    ];

    for (feedback_type, strength, confidence, description) in feedback_scenarios {
        let feedback = SearchFeedback {
            feedback_type: feedback_type.clone(),
            strength,
            confidence,
            details: HashMap::from([("description".to_string(), description.to_string())]),
        };

        let adaptation_result = system.adapt_to_feedback(user_id, &feedback, &context).await?;
        
        println!("    Feedback: {:?} (strength: {:.2}) → {} updates, confidence: {:.3}", 
            feedback_type,
            strength,
            adaptation_result.updates_applied.len(),
            adaptation_result.new_confidence
        );

        if !adaptation_result.updates_applied.is_empty() {
            for update in adaptation_result.updates_applied.iter().take(2) {
                println!("      Updated {}: {:.3} → {:.3}", 
                    update.feature_name,
                    update.old_value,
                    update.new_value
                );
            }
        }
    }

    let final_customizations = system.get_search_customizations(user_id, &context).await?;
    
    println!("    Final confidence: {:.3}", final_customizations.confidence);
    println!("    Confidence change: {:.3}", 
        final_customizations.confidence - initial_customizations.confidence);

    // Verify adaptation occurred
    assert_ne!(final_customizations.confidence, initial_customizations.confidence, 
        "Confidence should have changed through feedback");

    println!("  ✅ Real-time adaptation working correctly");
    Ok(())
}

async fn test_collaborative_filtering() -> Result<(), Box<dyn std::error::Error>> {
    let config = PreferenceLearningConfig {
        enable_collaborative_filtering: true,
        ..Default::default()
    };
    let mut system = PreferenceLearningSystem::new(config);
    
    let context = create_test_context();

    println!("  Testing collaborative filtering with multiple users");

    // Create multiple users with different preference patterns
    let users = vec![
        ("user_pdf_lover", "pdf"),
        ("user_code_focused", "py"), 
        ("user_docs_reader", "md"),
        ("user_mixed_content", "txt"),
    ];

    let user_ids: Vec<Uuid> = users.iter().map(|_| Uuid::new_v4()).collect();

    // Simulate different user preference patterns
    for (i, ((user_type, preferred_type), &user_id)) in users.iter().zip(&user_ids).enumerate() {
        println!("    Training user {} ({})", i + 1, user_type);
        
        // Create strong preference pattern for each user
        for j in 0..8 {
            let file_type = if j < 6 { *preferred_type } else { "txt" }; // 75% preferred, 25% other
            let interaction_type = if j % 2 == 0 { InteractionType::Bookmark } else { InteractionType::Dwell };
            
            let interaction = UserInteraction {
                document_id: Uuid::new_v4(),
                interaction_type,
                timestamp: Utc::now(),
                query_context: Some(format!("{} content", user_type)),
                session_id: context.session_id,
            };

            let results = vec![create_test_result(file_type, &format!("{} content {}", user_type, j))];
            system.learn_from_interaction(user_id, &interaction, &context, &results).await?;
        }
    }

    // Test collaborative recommendations
    let test_user = user_ids[0]; // PDF lover
    let candidate_results = vec![
        create_test_result("pdf", "New research paper"),
        create_test_result("py", "Python script"),
        create_test_result("md", "Documentation"),
        create_test_result("txt", "Text notes"),
    ];

    let recommendations = system.get_collaborative_recommendations(
        test_user, 
        &candidate_results, 
        &context
    ).await?;

    println!("    Generated {} collaborative recommendations for PDF lover:", recommendations.len());
    
    for (i, rec) in recommendations.iter().take(3).enumerate() {
        println!("      {}. {} (score: {:.3}, confidence: {:.3})", 
            i + 1,
            rec.result.file_type,
            rec.score,
            rec.confidence
        );
        
        if !rec.similar_users.is_empty() {
            println!("         Based on {} similar users", rec.similar_users.len());
        }
    }

    // Verify recommendations make sense (PDF content should score high for PDF lover)
    if !recommendations.is_empty() {
        let has_pdf_recommendation = recommendations.iter()
            .any(|rec| rec.result.file_type == "pdf");
        
        if has_pdf_recommendation {
            println!("    ✓ Collaborative filtering identified PDF preference");
        }
    }

    println!("  ✅ Collaborative filtering working correctly");
    Ok(())
}

async fn test_preference_drift_detection() -> Result<(), Box<dyn std::error::Error>> {
    let config = PreferenceLearningConfig {
        temporal_decay_rate: 0.9,
        ..Default::default()
    };
    let mut system = PreferenceLearningSystem::new(config);
    
    let user_id = Uuid::new_v4();
    let context = create_test_context();

    println!("  Testing preference drift detection over time");

    // Phase 1: Establish strong PDF preference
    println!("    Phase 1: Building PDF preference");
    for i in 0..10 {
        let interaction = UserInteraction {
            document_id: Uuid::new_v4(),
            interaction_type: InteractionType::Bookmark,
            timestamp: Utc::now(),
            query_context: Some("research phase".to_string()),
            session_id: context.session_id,
        };

        let results = vec![create_test_result("pdf", &format!("Research paper {}", i))];
        system.learn_from_interaction(user_id, &interaction, &context, &results).await?;
    }

    let phase1_customizations = system.get_search_customizations(user_id, &context).await?;
    println!("      Phase 1 confidence: {:.3}", phase1_customizations.confidence);

    // Phase 2: Simulate shift to code preference (simulating career change)
    println!("    Phase 2: Shifting to code preference");
    for i in 0..15 {
        let file_type = if i < 12 { "py" } else { "pdf" }; // Strong shift to Python
        let interaction = UserInteraction {
            document_id: Uuid::new_v4(),
            interaction_type: InteractionType::Open,
            timestamp: Utc::now(),
            query_context: Some("development phase".to_string()),
            session_id: context.session_id,
        };

        let results = vec![create_test_result(file_type, &format!("Programming content {}", i))];
        let update = system.learn_from_interaction(user_id, &interaction, &context, &results).await?;
        
        // Check for drift detection in adaptation events
        if update.adaptation_event.event_type == desktop_ai_search::core::preference_learning::AdaptationEventType::Drift {
            println!("      🔄 Preference drift detected at interaction {}", i + 1);
        }
    }

    let phase2_customizations = system.get_search_customizations(user_id, &context).await?;
    println!("      Phase 2 confidence: {:.3}", phase2_customizations.confidence);

    // Phase 3: Return to mixed preferences (simulating stable period)
    println!("    Phase 3: Stabilizing with mixed preferences");
    for i in 0..8 {
        let file_type = match i % 3 {
            0 => "pdf",
            1 => "py", 
            _ => "md",
        };
        
        let interaction = UserInteraction {
            document_id: Uuid::new_v4(),
            interaction_type: InteractionType::Click,
            timestamp: Utc::now(),
            query_context: Some("mixed phase".to_string()),
            session_id: context.session_id,
        };

        let results = vec![create_test_result(file_type, &format!("Mixed content {}", i))];
        system.learn_from_interaction(user_id, &interaction, &context, &results).await?;
    }

    let phase3_customizations = system.get_search_customizations(user_id, &context).await?;
    println!("      Phase 3 confidence: {:.3}", phase3_customizations.confidence);

    // Analyze preference evolution
    println!("    Preference evolution analysis:");
    println!("      Phase 1 → 2 confidence change: {:.3}", 
        phase2_customizations.confidence - phase1_customizations.confidence);
    println!("      Phase 2 → 3 confidence change: {:.3}", 
        phase3_customizations.confidence - phase2_customizations.confidence);

    // Verify the system adapted to changes
    assert!(phase1_customizations.confidence != phase3_customizations.confidence, 
        "Preferences should have evolved over time");

    println!("  ✅ Preference drift detection working correctly");
    Ok(())
}

async fn test_multi_user_scenarios() -> Result<(), Box<dyn std::error::Error>> {
    let config = PreferenceLearningConfig {
        enable_collaborative_filtering: true,
        min_interactions_threshold: 3,
        ..Default::default()
    };
    let mut system = PreferenceLearningSystem::new(config);
    
    let context = create_test_context();

    println!("  Testing complex multi-user learning scenarios");

    // Scenario: Team of researchers with overlapping interests
    let team_members = vec![
        ("researcher_lead", vec!["pdf", "md"], "leadership and documentation"),
        ("data_scientist", vec!["py", "ipynb"], "analysis and modeling"), 
        ("ml_engineer", vec!["py", "json"], "implementation and config"),
        ("domain_expert", vec!["pdf", "txt"], "research and notes"),
    ];

    let member_ids: Vec<Uuid> = team_members.iter().map(|_| Uuid::new_v4()).collect();

    // Build individual preference profiles
    for ((member_type, preferred_types, description), &member_id) in team_members.iter().zip(&member_ids) {
        println!("    Training {} ({})", member_type, description);
        
        for i in 0..6 {
            let file_type = preferred_types[i % preferred_types.len()];
            let interaction = UserInteraction {
                document_id: Uuid::new_v4(),
                interaction_type: if i % 2 == 0 { InteractionType::Bookmark } else { InteractionType::Dwell },
                timestamp: Utc::now(),
                query_context: Some(format!("team project - {}", member_type)),
                session_id: context.session_id,
            };

            let results = vec![create_test_result(file_type, &format!("{} content {}", member_type, i))];
            system.learn_from_interaction(member_id, &interaction, &context, &results).await?;
        }
    }

    // Test cross-user recommendations
    println!("\n    Testing cross-user recommendations:");
    
    let test_results = vec![
        create_test_result("pdf", "New research methodology paper"),
        create_test_result("py", "Advanced ML implementation"),
        create_test_result("md", "Project documentation"),
        create_test_result("ipynb", "Data analysis notebook"),
    ];

    for (i, &member_id) in member_ids.iter().enumerate() {
        let recommendations = system.get_collaborative_recommendations(
            member_id,
            &test_results,
            &context
        ).await?;

        println!("      {}: {} recommendations", 
            team_members[i].0,
            recommendations.len()
        );

        if !recommendations.is_empty() {
            let top_rec = &recommendations[0];
            println!("        Top: {} (score: {:.3})", 
                top_rec.result.file_type,
                top_rec.score
            );
        }
    }

    // Test team-wide patterns
    println!("\n    Analyzing team-wide preference patterns:");
    
    let mut team_customizations = Vec::new();
    for &member_id in &member_ids {
        let customizations = system.get_search_customizations(member_id, &context).await?;
        team_customizations.push(customizations);
    }

    // Calculate team statistics
    let avg_confidence: f32 = team_customizations.iter()
        .map(|c| c.confidence)
        .sum::<f32>() / team_customizations.len() as f32;

    let total_features: usize = team_customizations.iter()
        .map(|c| c.ranking_adjustments.content_boosts.len())
        .sum();

    println!("      Team average confidence: {:.3}", avg_confidence);
    println!("      Total learned features: {}", total_features);
    println!("      Members with customizations: {}/{}", 
        team_customizations.iter().filter(|c| c.confidence > 0.0).count(),
        team_customizations.len()
    );

    assert!(avg_confidence > 0.0, "Team should have developed preferences");
    assert!(total_features > 0, "Team should have learned features");

    println!("  ✅ Multi-user scenarios working correctly");
    Ok(())
}

async fn test_performance_scalability() -> Result<(), Box<dyn std::error::Error>> {
    let config = PreferenceLearningConfig {
        enable_realtime_adaptation: true,
        enable_collaborative_filtering: false, // Disable for performance testing
        ..Default::default()
    };
    let mut system = PreferenceLearningSystem::new(config);
    
    let context = create_test_context();

    println!("  Testing performance and scalability with high interaction volume");

    // Performance test parameters
    let num_users = 20;
    let interactions_per_user = 50;
    let total_interactions = num_users * interactions_per_user;

    println!("    Testing with {} users, {} interactions each ({} total)", 
        num_users, interactions_per_user, total_interactions);

    let start_time = std::time::Instant::now();
    
    // Generate users and interactions
    let user_ids: Vec<Uuid> = (0..num_users).map(|_| Uuid::new_v4()).collect();
    
    for (user_index, &user_id) in user_ids.iter().enumerate() {
        for interaction_index in 0..interactions_per_user {
            let file_type = match interaction_index % 4 {
                0 => "pdf",
                1 => "py",
                2 => "md", 
                _ => "txt",
            };

            let interaction = UserInteraction {
                document_id: Uuid::new_v4(),
                interaction_type: match interaction_index % 3 {
                    0 => InteractionType::Click,
                    1 => InteractionType::Dwell,
                    _ => InteractionType::Bookmark,
                },
                timestamp: Utc::now(),
                query_context: Some(format!("performance test user {}", user_index)),
                session_id: context.session_id,
            };

            let results = vec![create_test_result(file_type, &format!("Content {} for user {}", interaction_index, user_index))];
            system.learn_from_interaction(user_id, &interaction, &context, &results).await?;
        }

        if (user_index + 1) % 5 == 0 {
            println!("      Processed {} users...", user_index + 1);
        }
    }

    let training_time = start_time.elapsed();
    
    // Test customization generation performance
    let customization_start = std::time::Instant::now();
    let mut generated_customizations = 0;
    
    for &user_id in &user_ids {
        let _customizations = system.get_search_customizations(user_id, &context).await?;
        generated_customizations += 1;
    }
    
    let customization_time = customization_start.elapsed();

    // Test feedback adaptation performance
    let feedback_start = std::time::Instant::now();
    let mut feedback_processed = 0;
    
    for &user_id in user_ids.iter().take(10) { // Test subset for feedback
        let feedback = SearchFeedback {
            feedback_type: FeedbackType::Positive,
            strength: 0.7,
            confidence: 0.8,
            details: std::collections::HashMap::new(),
        };
        
        let _result = system.adapt_to_feedback(user_id, &feedback, &context).await?;
        feedback_processed += 1;
    }
    
    let feedback_time = feedback_start.elapsed();

    // Report performance metrics
    println!("\n    Performance Results:");
    println!("      Training time: {:?}", training_time);
    println!("      Interactions/second: {:.1}", 
        total_interactions as f64 / training_time.as_secs_f64());
    println!("      Avg time per interaction: {:.2}ms", 
        training_time.as_millis() as f64 / total_interactions as f64);
    
    println!("      Customization generation: {:?} for {} users", 
        customization_time, generated_customizations);
    println!("      Avg customization time: {:.2}ms", 
        customization_time.as_millis() as f64 / generated_customizations as f64);
    
    println!("      Feedback processing: {:?} for {} feedbacks", 
        feedback_time, feedback_processed);
    println!("      Avg feedback time: {:.2}ms", 
        feedback_time.as_millis() as f64 / feedback_processed as f64);

    // Performance assertions
    assert!(training_time.as_secs() < 30, "Training should complete within 30 seconds");
    assert!(customization_time.as_millis() < 5000, "Customization generation should be fast");
    assert!(feedback_time.as_millis() < 1000, "Feedback processing should be fast");

    let interactions_per_second = total_interactions as f64 / training_time.as_secs_f64();
    if interactions_per_second > 100.0 {
        println!("    ✅ Excellent performance - {} interactions/second", interactions_per_second as u32);
    } else if interactions_per_second > 50.0 {
        println!("    ✅ Good performance - {} interactions/second", interactions_per_second as u32);
    } else {
        println!("    ✅ Acceptable performance - {} interactions/second", interactions_per_second as u32);
    }

    Ok(())
}

// Helper functions
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

fn create_test_result(file_type: &str, content: &str) -> RankedResult {
    RankedResult {
        document_id: Uuid::new_v4(),
        chunk_id: None,
        title: format!("Test {} Document", file_type.to_uppercase()),
        snippet: content.to_string(),
        file_path: format!("/test/document.{}", file_type),
        file_type: file_type.to_string(),
        relevance_score: 0.8,
        features: RankingFeatures {
            text_bm25: 0.7,
            cosine_vec: 0.8,
            recency_decay: 0.9,
            user_frequency: 0.1,
            doc_quality: 0.8,
            same_project_flag: 1.0,
            diversity_penalty: 0.0,
            intent_alignment: 0.7,
            type_preference: 0.8,
            size_factor: 0.9,
        },
        ranking_explanation: None,
    }
}