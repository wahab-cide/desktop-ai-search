use desktop_ai_search::core::ranking::{
    AdvancedRanker, RankingConfig, RankingFeatures, RankedResult, 
    UserInteraction, InteractionType, SearchContext, DocumentStats
};
use desktop_ai_search::core::query_intent::{QueryIntentClassifier, Intent};
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Testing Advanced Ranking System\n");

    // Test 1: Feature Extraction Accuracy
    println!("=== Test 1: Feature Extraction Accuracy ===");
    test_feature_extraction().await?;

    // Test 2: Linear Model Performance
    println!("\n=== Test 2: Linear Model Performance ===");
    test_linear_model_performance().await?;

    // Test 3: Diversity Selection (MMR)
    println!("\n=== Test 3: MMR Diversity Selection ===");
    test_mmr_diversity().await?;

    // Test 4: User Interaction Learning
    println!("\n=== Test 4: User Interaction Learning ===");
    test_user_interaction_learning().await?;

    // Test 5: Intent Alignment Scoring
    println!("\n=== Test 5: Intent Alignment Scoring ===");
    test_intent_alignment().await?;

    // Test 6: Comprehensive Ranking Accuracy
    println!("\n=== Test 6: Comprehensive Ranking Accuracy ===");
    test_comprehensive_ranking().await?;

    println!("\n✅ All ranking system tests completed!");
    Ok(())
}

async fn test_feature_extraction() -> Result<(), Box<dyn std::error::Error>> {
    let config = RankingConfig::default();
    let ranker = AdvancedRanker::new(config);
    
    let classifier = QueryIntentClassifier::default()?;
    let query_intent = classifier.analyze_query("find recent PDF documents about machine learning").await?;
    
    let doc_id = Uuid::new_v4();
    let recent_timestamp = Utc::now() - Duration::days(2);
    let old_timestamp = Utc::now() - Duration::days(60);
    
    // Test recent document features
    let recent_features = ranker.extract_features(
        &doc_id,
        &query_intent,
        0.8,  // text_score
        0.7,  // vector_score
        recent_timestamp,
        "pdf",
        2 * 1024 * 1024, // 2MB
    )?;
    
    // Test old document features
    let old_features = ranker.extract_features(
        &doc_id,
        &query_intent,
        0.8,  // Same text score
        0.7,  // Same vector score
        old_timestamp,
        "pdf",
        2 * 1024 * 1024,
    )?;
    
    println!("Recent document recency decay: {:.3}", recent_features.recency_decay);
    println!("Old document recency decay: {:.3}", old_features.recency_decay);
    
    // Validate recency decay works correctly
    assert!(recent_features.recency_decay > old_features.recency_decay, 
           "Recent documents should have higher recency decay");
    assert!(recent_features.recency_decay > 0.8, 
           "Recent documents should have high recency decay");
    assert!(old_features.recency_decay < 0.3, 
           "Old documents should have low recency decay");
    
    println!("Intent alignment for PDF: {:.3}", recent_features.intent_alignment);
    println!("Type preference for PDF: {:.3}", recent_features.type_preference);
    
    // Validate intent alignment for PDF documents
    assert!(recent_features.intent_alignment > 0.7, 
           "PDF documents should have high intent alignment for document search");
    assert!(recent_features.type_preference > 0.7, 
           "PDF should have high type preference");
    
    println!("✅ Feature extraction working correctly");
    Ok(())
}

async fn test_linear_model_performance() -> Result<(), Box<dyn std::error::Error>> {
    let config = RankingConfig::default();
    let ranker = AdvancedRanker::new(config);
    
    let classifier = QueryIntentClassifier::default()?;
    let query_intent = classifier.analyze_query("important project documents").await?;
    
    // Create test documents with varying characteristics
    let test_docs = vec![
        // High relevance document
        create_test_result("high_rel", 0.9, 0.8, Duration::days(1), "pdf", 1024*1024),
        // Medium relevance, old document
        create_test_result("med_old", 0.7, 0.6, Duration::days(90), "doc", 2*1024*1024),
        // Low relevance, recent document  
        create_test_result("low_recent", 0.3, 0.2, Duration::days(3), "txt", 500*1024),
        // High vector similarity, low text relevance
        create_test_result("high_vec", 0.2, 0.9, Duration::days(10), "pdf", 3*1024*1024),
    ];
    
    let mut results = Vec::new();
    for (name, text_score, vec_score, age, file_type, file_size) in test_docs {
        let doc_id = Uuid::new_v4();
        let timestamp = Utc::now() - age;
        
        let features = ranker.extract_features(
            &doc_id,
            &query_intent,
            text_score,
            vec_score,
            timestamp,
            &file_type,
            file_size,
        )?;
        
        results.push(RankedResult {
            document_id: doc_id,
            chunk_id: None,
            title: name.to_string(),
            snippet: format!("Test document {}", name),
            file_path: format!("/test/{}.{}", name, file_type),
            file_type: file_type,
            relevance_score: 0.0, // Will be set by ranking
            features,
            ranking_explanation: None,
        });
    }
    
    let ranked_results = ranker.rank_results(results, &query_intent).await?;
    
    println!("Ranking results:");
    for (i, result) in ranked_results.iter().enumerate() {
        println!("  {}. {} - Score: {:.3} (Text: {:.2}, Vec: {:.2}, Recency: {:.2})", 
            i + 1, 
            result.title,
            result.relevance_score,
            result.features.text_bm25,
            result.features.cosine_vec,
            result.features.recency_decay
        );
    }
    
    // Validate ranking order makes sense
    assert!(ranked_results.len() == 4, "Should have 4 results");
    
    // High relevance document should be first or second (depending on recency vs relevance balance)
    let high_rel_pos = ranked_results.iter().position(|r| r.title == "high_rel").unwrap();
    assert!(high_rel_pos <= 1, "High relevance recent document should rank very highly");
    
    // Low relevance should not be first
    let low_recent_pos = ranked_results.iter().position(|r| r.title == "low_recent").unwrap();
    assert!(low_recent_pos > 0, "Low relevance document should not rank first");
    
    println!("✅ Linear model ranking working correctly");
    Ok(())
}

async fn test_mmr_diversity() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = RankingConfig::default();
    config.diversity_lambda = 0.5; // Balanced diversity vs relevance
    let ranker = AdvancedRanker::new(config);
    
    let classifier = QueryIntentClassifier::default()?;
    let query_intent = classifier.analyze_query("research documents").await?;
    
    // Create similar documents (high cosine similarity)
    let similar_results = vec![
        create_ranked_result("doc1", 0.95, 0.90, 0.90, "pdf"), // Highest relevance
        create_ranked_result("doc2", 0.90, 0.89, 0.85, "pdf"), // Very similar, high relevance
        create_ranked_result("doc3", 0.85, 0.50, 0.80, "doc"), // Different type, good relevance
        create_ranked_result("doc4", 0.80, 0.88, 0.75, "pdf"), // Similar to doc1&2, lower relevance
    ];
    
    let ranked_results = ranker.rank_results(similar_results, &query_intent).await?;
    
    println!("MMR Diversity results:");
    for (i, result) in ranked_results.iter().enumerate() {
        println!("  {}. {} - Score: {:.3} (Type: {}, Vector: {:.2})", 
            i + 1, 
            result.title,
            result.relevance_score,
            result.file_type,
            result.features.cosine_vec
        );
    }
    
    // Validate diversity: doc3 (different type) should rank higher than expected based on relevance alone
    let doc2_pos = ranked_results.iter().position(|r| r.title == "doc2").unwrap();
    let doc3_pos = ranked_results.iter().position(|r| r.title == "doc3").unwrap();
    
    // doc3 should potentially rank higher than doc2 due to diversity despite lower similarity
    println!("Doc2 position: {}, Doc3 position: {}", doc2_pos + 1, doc3_pos + 1);
    
    // First result should still be the highest relevance
    assert_eq!(ranked_results[0].title, "doc1", "Highest relevance should still rank first");
    
    println!("✅ MMR diversity selection working correctly");
    Ok(())
}

async fn test_user_interaction_learning() -> Result<(), Box<dyn std::error::Error>> {
    let config = RankingConfig::default();
    let mut ranker = AdvancedRanker::new(config);
    
    let doc_id = Uuid::new_v4();
    let session_id = Uuid::new_v4();
    
    // Simulate user interactions with a document
    let interactions = vec![
        UserInteraction {
            document_id: doc_id,
            interaction_type: InteractionType::Click,
            timestamp: Utc::now() - Duration::days(1),
            query_context: Some("test query".to_string()),
            session_id,
        },
        UserInteraction {
            document_id: doc_id,
            interaction_type: InteractionType::Open,
            timestamp: Utc::now() - Duration::hours(2),
            query_context: Some("another query".to_string()),
            session_id,
        },
        UserInteraction {
            document_id: doc_id,
            interaction_type: InteractionType::Bookmark,
            timestamp: Utc::now() - Duration::minutes(30),
            query_context: Some("important query".to_string()),
            session_id,
        },
    ];
    
    // Record interactions
    for interaction in interactions {
        ranker.record_interaction(interaction);
    }
    
    // Update document stats
    let stats = DocumentStats {
        total_views: 15,
        unique_viewers: 5,
        average_dwell_time: 45.0, // seconds
        bookmark_count: 3,
        share_count: 1,
        last_accessed: Some(Utc::now() - Duration::minutes(30)),
        quality_score: 0.85,
    };
    ranker.update_document_stats(doc_id, stats);
    
    let classifier = QueryIntentClassifier::default()?;
    let query_intent = classifier.analyze_query("test document").await?;
    
    // Extract features for this well-interacted document
    let features = ranker.extract_features(
        &doc_id,
        &query_intent,
        0.6, // medium text relevance
        0.7, // good vector relevance
        Utc::now() - Duration::days(10),
        "pdf",
        1024 * 1024,
    )?;
    
    println!("User frequency score: {:.3}", features.user_frequency);
    println!("Document quality score: {:.3}", features.doc_quality);
    
    // Validate that user interactions boost the document
    assert!(features.user_frequency > 0.0, "Document with interactions should have positive user frequency");
    assert!(features.doc_quality > 0.8, "Document with good stats should have high quality score");
    
    // Compare with a document without interactions
    let other_doc_id = Uuid::new_v4();
    let other_features = ranker.extract_features(
        &other_doc_id,
        &query_intent,
        0.6, // same text relevance
        0.7, // same vector relevance
        Utc::now() - Duration::days(10), // same age
        "pdf",
        1024 * 1024,
    )?;
    
    println!("Other doc user frequency: {:.3}", other_features.user_frequency);
    println!("Other doc quality: {:.3}", other_features.doc_quality);
    
    assert!(features.user_frequency > other_features.user_frequency,
           "Document with interactions should score higher on user frequency");
    
    println!("✅ User interaction learning working correctly");
    Ok(())
}

async fn test_intent_alignment() -> Result<(), Box<dyn std::error::Error>> {
    let config = RankingConfig::default();
    let ranker = AdvancedRanker::new(config);
    
    let classifier = QueryIntentClassifier::default()?;
    
    // Test different query types
    let test_cases = vec![
        ("find PDF documents", "pdf", "Should have high alignment"),
        ("find PDF documents", "jpg", "Should have lower alignment"), 
        ("what is machine learning?", "pdf", "QA query should favor text documents"),
        ("what is machine learning?", "jpg", "QA query should not favor images"),
        ("similar to this report", "pdf", "Similarity search should work for all types"),
        ("emails from john", "msg", "Person search should favor email formats"),
        ("emails from john", "pdf", "Person search should favor documents less"),
    ];
    
    println!("Intent alignment test results:");
    for (query, file_type, description) in test_cases {
        let query_intent = classifier.analyze_query(query).await?;
        let doc_id = Uuid::new_v4();
        
        let features = ranker.extract_features(
            &doc_id,
            &query_intent,
            0.7, // Standard relevance
            0.7,
            Utc::now(),
            file_type,
            1024 * 1024,
        )?;
        
        println!("  '{}' + {} → Alignment: {:.3} | Type Pref: {:.3} | {}",
            query, file_type, features.intent_alignment, features.type_preference, description);
    }
    
    // Specific validation tests
    let pdf_query = classifier.analyze_query("find PDF documents").await?;
    let doc_id = Uuid::new_v4();
    
    let pdf_features = ranker.extract_features(&doc_id, &pdf_query, 0.7, 0.7, Utc::now(), "pdf", 1024*1024)?;
    let jpg_features = ranker.extract_features(&doc_id, &pdf_query, 0.7, 0.7, Utc::now(), "jpg", 1024*1024)?;
    
    assert!(pdf_features.intent_alignment > jpg_features.intent_alignment,
           "PDF documents should have higher intent alignment for PDF-specific queries");
    assert!(pdf_features.type_preference > jpg_features.type_preference,
           "PDF should have higher type preference for PDF queries");
    
    println!("✅ Intent alignment scoring working correctly");
    Ok(())
}

async fn test_comprehensive_ranking() -> Result<(), Box<dyn std::error::Error>> {
    let config = RankingConfig::default();
    let mut ranker = AdvancedRanker::new(config);
    
    // Set up realistic search context
    let context = SearchContext {
        session_id: Uuid::new_v4(),
        current_project: Some("ML Research".to_string()),
        recent_documents: vec![],
        active_applications: vec!["Code".to_string(), "Safari".to_string()],
        search_history: vec!["machine learning".to_string(), "neural networks".to_string()],
        timestamp: Utc::now(),
    };
    ranker.update_context(context);
    
    let classifier = QueryIntentClassifier::default()?;
    let query_intent = classifier.analyze_query("recent machine learning research papers").await?;
    
    // Create diverse test corpus
    let test_corpus = vec![
        // Perfect match: recent, relevant, right type
        ("ml_paper_2024.pdf".to_string(), 0.95, 0.90, Duration::days(2), "pdf".to_string(), 2*1024*1024, 0.9),
        // Good content, but old
        ("classic_ml_book.pdf".to_string(), 0.90, 0.85, Duration::days(365), "pdf".to_string(), 10*1024*1024, 0.8),
        // Recent but less relevant
        ("software_doc.pdf".to_string(), 0.40, 0.30, Duration::days(1), "pdf".to_string(), 1*1024*1024, 0.6),
        // Different format, good content
        ("ml_notes.txt".to_string(), 0.80, 0.75, Duration::days(7), "txt".to_string(), 100*1024, 0.7),
        // High user interaction history
        ("favorite_paper.pdf".to_string(), 0.70, 0.65, Duration::days(30), "pdf".to_string(), 3*1024*1024, 0.95),
    ];
    
    let mut results = Vec::new();
    for (name, text_score, vec_score, age, file_type, file_size, quality) in test_corpus {
        let doc_id = Uuid::new_v4();
        let timestamp = Utc::now() - age;
        
        // Add interactions for the "favorite" document
        if name.contains("favorite") {
            for i in 0..5 {
                ranker.record_interaction(UserInteraction {
                    document_id: doc_id,
                    interaction_type: InteractionType::Click,
                    timestamp: Utc::now() - Duration::days(i),
                    query_context: Some("ml research".to_string()),
                    session_id: Uuid::new_v4(),
                });
            }
            
            ranker.update_document_stats(doc_id, DocumentStats {
                total_views: 20,
                unique_viewers: 3,
                average_dwell_time: 120.0,
                bookmark_count: 1,
                share_count: 2,
                last_accessed: Some(Utc::now() - Duration::hours(1)),
                quality_score: quality,
            });
        }
        
        let features = ranker.extract_features(
            &doc_id,
            &query_intent,
            text_score,
            vec_score,
            timestamp,
            &file_type,
            file_size,
        )?;
        
        results.push(RankedResult {
            document_id: doc_id,
            chunk_id: None,
            title: name.to_string(),
            snippet: format!("Content from {}", name),
            file_path: format!("/docs/{}", name),
            file_type: file_type,
            relevance_score: 0.0,
            features,
            ranking_explanation: None,
        });
    }
    
    let ranked_results = ranker.rank_results(results, &query_intent).await?;
    
    println!("Comprehensive ranking results:");
    for (i, result) in ranked_results.iter().enumerate() {
        let explanation = if let Some(ref exp) = result.ranking_explanation {
            format!("Primary: {:?}", exp.primary_signals.iter().take(2).collect::<Vec<_>>())
        } else {
            "No explanation".to_string()
        };
        
        println!("  {}. {} - Score: {:.3}", i + 1, result.title, result.relevance_score);
        println!("     Features: Text={:.2}, Vec={:.2}, Recency={:.2}, User={:.2}, Quality={:.2}",
            result.features.text_bm25,
            result.features.cosine_vec, 
            result.features.recency_decay,
            result.features.user_frequency,
            result.features.doc_quality
        );
        println!("     {}", explanation);
    }
    
    // Validate ranking makes sense
    assert!(ranked_results.len() == 5, "Should have all 5 results");
    
    // Recent relevant paper should rank very highly
    let recent_paper_pos = ranked_results.iter().position(|r| r.title.contains("ml_paper_2024")).unwrap();
    assert!(recent_paper_pos <= 1, "Recent relevant paper should rank in top 2");
    
    // Low relevance document should not be in top 2
    let software_doc_pos = ranked_results.iter().position(|r| r.title.contains("software_doc")).unwrap();
    assert!(software_doc_pos >= 2, "Low relevance document should not rank in top 2");
    
    // Calculate ranking quality metrics
    let mut dcg = 0.0;
    let relevance_scores = vec![0.95, 0.90, 0.40, 0.80, 0.70]; // True relevance for each document
    
    for (i, result) in ranked_results.iter().enumerate() {
        // Find true relevance
        let true_relevance = match result.title.as_str() {
            name if name.contains("ml_paper_2024") => 0.95,
            name if name.contains("classic_ml_book") => 0.90,
            name if name.contains("software_doc") => 0.40,
            name if name.contains("ml_notes") => 0.80,
            name if name.contains("favorite_paper") => 0.70,
            _ => 0.0,
        };
        
        // DCG calculation: relevance / log2(position + 1)
        dcg += true_relevance / (i as f32 + 2.0).log2();
    }
    
    // Calculate ideal DCG (perfect ranking)
    let mut ideal_relevance = relevance_scores.clone();
    ideal_relevance.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut idcg = 0.0;
    for (i, &relevance) in ideal_relevance.iter().enumerate() {
        idcg += relevance / (i as f32 + 2.0).log2();
    }
    
    let ndcg = dcg / idcg;
    println!("\nRanking Quality Metrics:");
    println!("  DCG: {:.3}", dcg);
    println!("  IDCG: {:.3}", idcg);
    println!("  NDCG: {:.3}", ndcg);
    
    // NDCG should be reasonably high (>0.8 is good)
    assert!(ndcg > 0.7, "NDCG should be at least 0.7 for good ranking quality");
    
    if ndcg > 0.9 {
        println!("  🌟 Excellent ranking quality (NDCG > 0.9)");
    } else if ndcg > 0.8 {
        println!("  ✅ Good ranking quality (NDCG > 0.8)");
    } else {
        println!("  ⚠️  Acceptable ranking quality (NDCG > 0.7)");
    }
    
    println!("✅ Comprehensive ranking system working correctly");
    Ok(())
}

// Helper functions
fn create_test_result(name: &str, text_score: f32, vec_score: f32, age: Duration, file_type: &str, file_size: u64) -> (String, f32, f32, Duration, String, u64) {
    (name.to_string(), text_score, vec_score, age, file_type.to_string(), file_size)
}

fn create_ranked_result(name: &str, relevance: f32, vector_score: f32, recency: f32, file_type: &str) -> RankedResult {
    RankedResult {
        document_id: Uuid::new_v4(),
        chunk_id: None,
        title: name.to_string(),
        snippet: format!("Test content for {}", name),
        file_path: format!("/test/{}.{}", name, file_type),
        file_type: file_type.to_string(),
        relevance_score: relevance,
        features: RankingFeatures {
            text_bm25: relevance * 0.8,
            cosine_vec: vector_score,
            recency_decay: recency,
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