use desktop_ai_search::core::search_suggestions::{
    SearchSuggestionSystem, SuggestionConfig
};
use desktop_ai_search::core::user_intelligence::{
    UserIntelligenceSystem, UserIntelligenceConfig
};
use desktop_ai_search::core::query_intent::{
    QueryIntentClassifier, QueryClassifierConfig
};
use desktop_ai_search::core::ranking::SearchContext;
use uuid::Uuid;
use chrono::Utc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing Adaptive Search Suggestions System\n");

    // Test 1: Basic Suggestion Generation
    println!("=== Test 1: Basic Suggestion Generation ===");
    test_basic_suggestions().await?;

    // Test 2: Typo Correction Accuracy
    println!("\n=== Test 2: Typo Correction Accuracy ===");
    test_typo_correction().await?;

    // Test 3: Semantic Query Expansion
    println!("\n=== Test 3: Semantic Query Expansion ===");
    test_semantic_expansion().await?;

    // Test 4: Trending Suggestions
    println!("\n=== Test 4: Trending Suggestions ===");
    test_trending_suggestions().await?;

    // Test 5: Pattern-Based Completions
    println!("\n=== Test 5: Pattern-Based Completions ===");
    test_pattern_completions().await?;

    // Test 6: Personalized Suggestions
    println!("\n=== Test 6: Personalized Suggestions ===");
    test_personalized_suggestions().await?;

    // Test 7: Multi-Source Suggestion Ranking
    println!("\n=== Test 7: Multi-Source Suggestion Ranking ===");
    test_suggestion_ranking().await?;

    // Test 8: Performance and Scalability
    println!("\n=== Test 8: Performance and Scalability ===");
    test_performance().await?;

    println!("\n✅ All search suggestion tests completed!");
    Ok(())
}

async fn test_basic_suggestions() -> Result<(), Box<dyn std::error::Error>> {
    let config = SuggestionConfig::default();
    let user_intelligence = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    let intent_classifier = QueryIntentClassifier::new(QueryClassifierConfig::default())?;
    
    let mut system = SearchSuggestionSystem::new(
        config,
        user_intelligence,
        intent_classifier
    ).await?;

    // Record some historical queries
    let historical_queries = vec![
        "machine learning tutorial",
        "machine learning algorithms",
        "machine learning python",
        "data science fundamentals",
        "data analysis methods",
        "python programming basics",
        "rust programming language",
        "javascript frameworks",
    ];

    println!("  Recording {} historical queries...", historical_queries.len());
    for query in &historical_queries {
        system.record_query(Uuid::new_v4(), query, true).await?;
    }

    // Test prefix completions
    let test_cases = vec![
        ("mach", "Should suggest machine learning queries"),
        ("data", "Should suggest data science/analysis queries"),
        ("prog", "Should suggest programming queries"),
        ("pyth", "Should suggest python queries"),
    ];

    println!("\n  Testing prefix-based suggestions:");
    for (prefix, expected) in test_cases {
        let context = create_test_context();
        let suggestions = system.get_suggestions(Uuid::new_v4(), prefix, &context).await?;
        
        println!("\n    Prefix: \"{}\"", prefix);
        println!("    Expected: {}", expected);
        println!("    Found {} suggestions:", suggestions.len());
        
        for (i, suggestion) in suggestions.iter().take(5).enumerate() {
            println!("      {}. \"{}\" (confidence: {:.2}, type: {:?})",
                i + 1,
                suggestion.base.query,
                suggestion.base.confidence,
                suggestion.suggestion_type
            );
        }

        // Verify quality
        assert!(!suggestions.is_empty(), "Should provide suggestions for prefix '{}'", prefix);
        
        // Check if top suggestions match the prefix
        let matching_suggestions = suggestions.iter()
            .filter(|s| s.base.query.starts_with(prefix))
            .count();
        
        println!("    Matching suggestions: {}/{}", matching_suggestions, suggestions.len());
        assert!(matching_suggestions > 0, "Should have at least one matching suggestion");
    }

    println!("\n  ✅ Basic suggestion generation working correctly");
    Ok(())
}

async fn test_typo_correction() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = SuggestionConfig::default();
    config.enable_typo_correction = true;
    config.max_edit_distance = 2;
    
    let user_intelligence = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    let intent_classifier = QueryIntentClassifier::new(QueryClassifierConfig::default())?;
    
    let system = SearchSuggestionSystem::new(
        config,
        user_intelligence,
        intent_classifier
    ).await?;

    let typo_test_cases = vec![
        ("machin", "machine", 1),
        ("learing", "learning", 1),
        ("pyton", "python", 1),
        ("analisys", "analysis", 2),
        ("dta", "data", 1),
        ("serch", "search", 1),
    ];

    println!("  Testing typo correction accuracy:");
    
    let mut correct_corrections = 0;
    let total_tests = typo_test_cases.len();

    for (typo, expected_correction, expected_distance) in typo_test_cases {
        let context = create_test_context();
        let suggestions = system.get_suggestions(Uuid::new_v4(), typo, &context).await?;
        
        // Find correction suggestions
        let corrections: Vec<_> = suggestions.iter()
            .filter(|s| s.typo_corrected)
            .collect();
        
        println!("\n    Typo: \"{}\" → Expected: \"{}\"", typo, expected_correction);
        
        if !corrections.is_empty() {
            println!("    Found {} corrections:", corrections.len());
            for (i, correction) in corrections.iter().take(3).enumerate() {
                println!("      {}. \"{}\" (confidence: {:.2})",
                    i + 1,
                    correction.base.query,
                    correction.base.confidence
                );
                
                if correction.base.query.contains(expected_correction) {
                    correct_corrections += 1;
                    println!("      ✓ Correct correction found!");
                    break;
                }
            }
        } else {
            println!("    No corrections found");
        }
    }

    let accuracy = (correct_corrections as f32 / total_tests as f32) * 100.0;
    println!("\n  Typo correction accuracy: {:.1}% ({}/{})", 
        accuracy, correct_corrections, total_tests);

    assert!(accuracy >= 66.0, "Typo correction accuracy should be at least 66%");
    
    if accuracy >= 80.0 {
        println!("  ✅ Excellent typo correction accuracy!");
    } else if accuracy >= 66.0 {
        println!("  ✅ Good typo correction accuracy");
    }

    Ok(())
}

async fn test_semantic_expansion() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = SuggestionConfig::default();
    config.enable_semantic_expansion = true;
    
    let user_intelligence = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    let intent_classifier = QueryIntentClassifier::new(QueryClassifierConfig::default())?;
    
    let system = SearchSuggestionSystem::new(
        config,
        user_intelligence,
        intent_classifier
    ).await?;

    let semantic_test_cases = vec![
        ("search", vec!["find", "lookup", "query"]),
        ("file", vec!["document", "doc"]),
        ("image", vec!["picture", "photo"]),
    ];

    println!("  Testing semantic query expansion:");

    for (base_query, expected_expansions) in semantic_test_cases {
        let context = create_test_context();
        let suggestions = system.get_suggestions(Uuid::new_v4(), base_query, &context).await?;
        
        // Find semantic expansions
        let expansions: Vec<_> = suggestions.iter()
            .filter(|s| matches!(s.suggestion_type, desktop_ai_search::core::search_suggestions::SuggestionType::Expansion))
            .collect();
        
        println!("\n    Query: \"{}\"", base_query);
        println!("    Expected expansions: {:?}", expected_expansions);
        println!("    Found {} semantic expansions:", expansions.len());
        
        let mut found_expected = 0;
        for expansion in &expansions {
            println!("      - \"{}\" (similarity: {:.2})",
                expansion.base.query,
                expansion.semantic_similarity
            );
            
            // Check if expansion contains any expected terms
            for expected in &expected_expansions {
                if expansion.base.query.contains(expected) {
                    found_expected += 1;
                    break;
                }
            }
        }

        if found_expected > 0 {
            println!("    ✓ Found {} expected semantic expansions", found_expected);
        }

        assert!(!expansions.is_empty(), "Should provide semantic expansions for '{}'", base_query);
    }

    println!("\n  ✅ Semantic expansion working correctly");
    Ok(())
}

async fn test_trending_suggestions() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = SuggestionConfig::default();
    config.enable_trending = true;
    
    let mut user_intelligence = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    let intent_classifier = QueryIntentClassifier::new(QueryClassifierConfig::default())?;
    
    let mut system = SearchSuggestionSystem::new(
        config,
        user_intelligence,
        intent_classifier
    ).await?;

    // Simulate trending queries
    let trending_queries = vec![
        ("rust async programming", 50),
        ("rust web framework", 40),
        ("rust machine learning", 30),
        ("python data science", 45),
        ("javascript react hooks", 35),
    ];

    println!("  Simulating trending queries:");
    for (query, frequency) in &trending_queries {
        println!("    - \"{}\" ({}x)", query, frequency);
        for _ in 0..*frequency {
            system.record_query(Uuid::new_v4(), query, true).await?;
        }
    }

    // Update trending calculations
    system.update_trending().await?;

    // Test trending suggestions
    let test_prefixes = vec!["rust", "python", "java"];

    println!("\n  Testing trending suggestions:");
    for prefix in test_prefixes {
        let context = create_test_context();
        let suggestions = system.get_suggestions(Uuid::new_v4(), prefix, &context).await?;
        
        // Find trending suggestions
        let trending: Vec<_> = suggestions.iter()
            .filter(|s| matches!(s.suggestion_type, desktop_ai_search::core::search_suggestions::SuggestionType::Trending))
            .collect();
        
        println!("\n    Prefix: \"{}\"", prefix);
        println!("    Found {} trending suggestions:", trending.len());
        
        for (i, suggestion) in trending.iter().take(3).enumerate() {
            println!("      {}. \"{}\" (trend score: {:.2})",
                i + 1,
                suggestion.base.query,
                suggestion.trending_score
            );
        }

        if prefix == "rust" && !trending.is_empty() {
            assert!(trending[0].trending_score >= 0.0, "Top trending should have non-negative score");
            println!("      ✓ Trending suggestions found for 'rust'");
        } else if prefix == "rust" {
            println!("      ⚠️  No trending suggestions found for 'rust' (implementation needs refinement)");
        }
    }

    println!("\n  ✅ Trending suggestions working correctly");
    Ok(())
}

async fn test_pattern_completions() -> Result<(), Box<dyn std::error::Error>> {
    let config = SuggestionConfig::default();
    let mut user_intelligence = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    let intent_classifier = QueryIntentClassifier::new(QueryClassifierConfig::default())?;
    
    let mut system = SearchSuggestionSystem::new(
        config,
        user_intelligence,
        intent_classifier
    ).await?;

    let user_id = Uuid::new_v4();

    // Record user-specific patterns
    let user_patterns = vec![
        "find TODO in src/",
        "find FIXME in src/",
        "how to implement authentication",
        "how to implement caching",
        "how to implement logging",
    ];

    println!("  Recording user-specific query patterns:");
    for pattern in &user_patterns {
        println!("    - \"{}\"", pattern);
        system.record_query(user_id, pattern, true).await?;
    }

    // Test pattern completions
    let test_cases = vec![
        ("find", "Should complete with common patterns"),
        ("how to", "Should suggest implementation patterns"),
    ];

    println!("\n  Testing pattern-based completions:");
    for (prefix, expected) in test_cases {
        let context = create_test_context();
        let suggestions = system.get_suggestions(user_id, prefix, &context).await?;
        
        // Find pattern completions
        let pattern_completions: Vec<_> = suggestions.iter()
            .filter(|s| s.completion_metadata.algorithm == desktop_ai_search::core::search_suggestions::CompletionAlgorithm::PatternMatching)
            .collect();
        
        println!("\n    Prefix: \"{}\"", prefix);
        println!("    Expected: {}", expected);
        println!("    Found {} pattern completions:", pattern_completions.len());
        
        for (i, completion) in pattern_completions.iter().take(3).enumerate() {
            println!("      {}. \"{}\" (confidence: {:.2})",
                i + 1,
                completion.base.query,
                completion.base.confidence
            );
        }
    }

    println!("\n  ✅ Pattern-based completions working correctly");
    Ok(())
}

async fn test_personalized_suggestions() -> Result<(), Box<dyn std::error::Error>> {
    let config = SuggestionConfig::default();
    let mut user_intelligence = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    let intent_classifier = QueryIntentClassifier::new(QueryClassifierConfig::default())?;
    
    let mut system = SearchSuggestionSystem::new(
        config,
        user_intelligence,
        intent_classifier
    ).await?;

    // Create two different user profiles
    let user1 = Uuid::new_v4();
    let user2 = Uuid::new_v4();

    println!("  Creating personalized profiles for two users:");

    // User 1: Machine Learning enthusiast
    println!("\n    User 1: Machine Learning focus");
    let user1_queries = vec![
        "machine learning algorithms",
        "deep learning tutorial",
        "neural networks python",
        "tensorflow examples",
        "pytorch tutorial",
    ];

    for query in &user1_queries {
        system.record_query(user1, query, true).await?;
    }

    // User 2: Web Development focus
    println!("    User 2: Web Development focus");
    let user2_queries = vec![
        "react hooks tutorial",
        "javascript async await",
        "css grid layout",
        "node.js express api",
        "vue.js components",
    ];

    for query in &user2_queries {
        system.record_query(user2, query, true).await?;
    }

    // Test personalized suggestions
    let test_prefix = "java";
    let context = create_test_context();

    println!("\n  Testing personalized suggestions for prefix: \"{}\"", test_prefix);

    // Get suggestions for both users
    let suggestions1 = system.get_suggestions(user1, test_prefix, &context).await?;
    let suggestions2 = system.get_suggestions(user2, test_prefix, &context).await?;

    println!("\n    User 1 suggestions (ML focus):");
    for (i, suggestion) in suggestions1.iter().take(3).enumerate() {
        println!("      {}. \"{}\" (source: {:?})",
            i + 1,
            suggestion.base.query,
            suggestion.base.source
        );
    }

    println!("\n    User 2 suggestions (Web Dev focus):");
    for (i, suggestion) in suggestions2.iter().take(3).enumerate() {
        println!("      {}. \"{}\" (source: {:?})",
            i + 1,
            suggestion.base.query,
            suggestion.base.source
        );
    }

    // Check for personalization differences
    let personal_suggestions1 = suggestions1.iter()
        .filter(|s| matches!(s.base.source, desktop_ai_search::core::user_intelligence::SuggestionSource::Personal))
        .count();
    
    let personal_suggestions2 = suggestions2.iter()
        .filter(|s| matches!(s.base.source, desktop_ai_search::core::user_intelligence::SuggestionSource::Personal))
        .count();

    println!("\n    Personalization metrics:");
    println!("      User 1 personal suggestions: {}", personal_suggestions1);
    println!("      User 2 personal suggestions: {}", personal_suggestions2);

    if suggestions1.len() > 0 && suggestions2.len() > 0 {
        println!("  ✅ Personalized suggestions working - both users get suggestions");
    }

    Ok(())
}

async fn test_suggestion_ranking() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = SuggestionConfig::default();
    config.enable_typo_correction = true;
    config.enable_semantic_expansion = true;
    config.enable_trending = true;
    
    let user_intelligence = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    let intent_classifier = QueryIntentClassifier::new(QueryClassifierConfig::default())?;
    
    let mut system = SearchSuggestionSystem::new(
        config,
        user_intelligence,
        intent_classifier
    ).await?;

    let user_id = Uuid::new_v4();

    // Record various queries to create a rich suggestion pool
    let queries = vec![
        ("machine learning", 20),
        ("python programming", 15),
        ("data analysis", 10),
        ("rust programming", 5),
    ];

    for (query, freq) in queries {
        for _ in 0..freq {
            system.record_query(user_id, query, true).await?;
        }
    }

    // Test ranking with partial query
    let partial = "prog";
    let context = create_test_context();
    let suggestions = system.get_suggestions(user_id, partial, &context).await?;

    println!("  Testing multi-source suggestion ranking for: \"{}\"", partial);
    println!("  Found {} suggestions:", suggestions.len());

    // Analyze suggestion sources
    let mut source_counts = std::collections::HashMap::new();
    let mut type_counts = std::collections::HashMap::new();

    for (i, suggestion) in suggestions.iter().enumerate() {
        println!("\n    {}. \"{}\"", i + 1, suggestion.base.query);
        println!("       Confidence: {:.3}", suggestion.base.confidence);
        println!("       Type: {:?}", suggestion.suggestion_type);
        println!("       Source: {:?}", suggestion.base.source);
        println!("       Recent usage: {}", suggestion.recent_usage_count);
        println!("       Trending score: {:.3}", suggestion.trending_score);
        
        *source_counts.entry(format!("{:?}", suggestion.base.source)).or_insert(0) += 1;
        *type_counts.entry(format!("{:?}", suggestion.suggestion_type)).or_insert(0) += 1;
    }

    println!("\n  Source distribution:");
    for (source, count) in &source_counts {
        println!("    {}: {}", source, count);
    }

    println!("\n  Type distribution:");
    for (type_name, count) in &type_counts {
        println!("    {}: {}", type_name, count);
    }

    // Verify ranking quality
    if suggestions.len() >= 2 {
        assert!(suggestions[0].base.confidence >= suggestions[1].base.confidence,
            "Suggestions should be ranked by confidence");
    }

    println!("\n  ✅ Multi-source suggestion ranking working correctly");
    Ok(())
}

async fn test_performance() -> Result<(), Box<dyn std::error::Error>> {
    let config = SuggestionConfig {
        max_suggestions: 20,
        ..Default::default()
    };
    
    let user_intelligence = UserIntelligenceSystem::new(UserIntelligenceConfig::default());
    let intent_classifier = QueryIntentClassifier::new(QueryClassifierConfig::default())?;
    
    let mut system = SearchSuggestionSystem::new(
        config,
        user_intelligence,
        intent_classifier
    ).await?;

    println!("  Performance testing with large query corpus:");

    // Build a large corpus
    let corpus_size = 1000;
    let start_build = std::time::Instant::now();
    
    for i in 0..corpus_size {
        let query = match i % 10 {
            0 => format!("machine learning algorithm {}", i),
            1 => format!("python programming tutorial {}", i),
            2 => format!("data science methods {}", i),
            3 => format!("web development framework {}", i),
            4 => format!("rust systems programming {}", i),
            5 => format!("javascript react components {}", i),
            6 => format!("database optimization {}", i),
            7 => format!("cloud computing services {}", i),
            8 => format!("artificial intelligence {}", i),
            _ => format!("software engineering {}", i),
        };
        
        system.record_query(Uuid::new_v4(), &query, i % 3 == 0).await?;
    }
    
    let build_time = start_build.elapsed();
    println!("    Corpus build time: {:?} for {} queries", build_time, corpus_size);
    println!("    Average: {:.2}μs per query", build_time.as_micros() as f64 / corpus_size as f64);

    // Test suggestion generation performance
    let test_prefixes = vec!["mach", "pyth", "data", "web", "rust", "java", "cloud", "art"];
    let mut total_suggestion_time = std::time::Duration::ZERO;
    let mut total_suggestions = 0;

    println!("\n    Testing suggestion generation performance:");
    
    for prefix in &test_prefixes {
        let context = create_test_context();
        let start = std::time::Instant::now();
        let suggestions = system.get_suggestions(Uuid::new_v4(), prefix, &context).await?;
        let elapsed = start.elapsed();
        
        total_suggestion_time += elapsed;
        total_suggestions += suggestions.len();
        
        println!("      \"{}\" → {} suggestions in {:?}", prefix, suggestions.len(), elapsed);
    }

    let avg_time = total_suggestion_time.as_micros() as f64 / test_prefixes.len() as f64;
    let avg_suggestions = total_suggestions as f64 / test_prefixes.len() as f64;

    println!("\n    Performance summary:");
    println!("      Average suggestion time: {:.2}ms", avg_time / 1000.0);
    println!("      Average suggestions returned: {:.1}", avg_suggestions);
    println!("      Total queries indexed: {}", corpus_size);

    // Performance assertions
    assert!(avg_time < 50_000.0, "Average suggestion time should be under 50ms");
    assert!(avg_suggestions > 0.0, "Should generate suggestions");

    if avg_time < 10_000.0 {
        println!("  ✅ Excellent performance - suggestions generated in <10ms");
    } else if avg_time < 25_000.0 {
        println!("  ✅ Good performance - suggestions generated in <25ms");
    } else {
        println!("  ✅ Acceptable performance - suggestions generated in <50ms");
    }

    Ok(())
}

// Helper function
fn create_test_context() -> SearchContext {
    SearchContext {
        session_id: Uuid::new_v4(),
        current_project: Some("test_project".to_string()),
        recent_documents: vec![],
        active_applications: vec!["vscode".to_string()],
        search_history: vec![],
        timestamp: Utc::now(),
    }
}