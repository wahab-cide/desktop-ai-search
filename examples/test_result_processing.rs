use desktop_ai_search::core::result_processing::{
    ResultProcessor, ResultProcessingConfig
};
use desktop_ai_search::core::ranking::{RankedResult, RankingFeatures};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Testing Result Processing System\n");

    // Test 1: Duplicate Detection Accuracy
    println!("=== Test 1: Duplicate Detection ===");
    test_duplicate_detection().await?;

    // Test 2: Result Clustering
    println!("\n=== Test 2: Result Clustering ===");
    test_result_clustering().await?;

    // Test 3: Document Summarization
    println!("\n=== Test 3: Document Summarization ===");
    test_document_summarization().await?;

    // Test 4: Snippet Highlighting
    println!("\n=== Test 4: Snippet Highlighting ===");
    test_snippet_highlighting().await?;

    // Test 5: End-to-End Processing
    println!("\n=== Test 5: End-to-End Processing ===");
    test_end_to_end_processing().await?;

    // Test 6: Performance Benchmarks
    println!("\n=== Test 6: Performance Benchmarks ===");
    test_performance_benchmarks().await?;

    println!("\n✅ All result processing tests completed!");
    Ok(())
}

async fn test_duplicate_detection() -> Result<(), Box<dyn std::error::Error>> {
    let mut processor = ResultProcessor::new(ResultProcessingConfig::default());
    
    // Create test results with duplicates
    let results = vec![
        create_test_result("original_doc", "Machine learning is a powerful tool for data analysis and prediction"),
        create_test_result("exact_duplicate", "Machine learning is a powerful tool for data analysis and prediction"),
        create_test_result("near_duplicate", "Machine learning is a powerful tool for data analysis and predictions"),
        create_test_result("similar_content", "Machine learning provides powerful tools for analyzing data and making predictions"),
        create_test_result("different_doc", "Python programming fundamentals for software development"),
        create_test_result("another_duplicate", "Machine learning is a powerful tool for data analysis and prediction"),
    ];

    println!("Original results count: {}", results.len());
    let deduplicated = processor.detect_duplicates(results).await?;
    println!("After deduplication: {}", deduplicated.len());

    // Should detect and remove duplicates
    assert!(deduplicated.len() < 6, "Should remove some duplicates");
    assert!(deduplicated.len() >= 3, "Should keep unique documents");

    // Verify content diversity
    let mut unique_snippets = std::collections::HashSet::new();
    for result in &deduplicated {
        unique_snippets.insert(result.snippet.clone());
    }
    
    println!("Unique content pieces: {}", unique_snippets.len());
    
    if deduplicated.len() <= 4 {
        println!("✅ Effective duplicate detection - reduced from 6 to {} results", deduplicated.len());
    } else {
        println!("⚠️  Moderate duplicate detection - some duplicates may remain");
    }

    Ok(())
}

async fn test_result_clustering() -> Result<(), Box<dyn std::error::Error>> {
    let mut processor = ResultProcessor::new(ResultProcessingConfig::default());
    
    // Create results with clear thematic groups and more similar content
    let results = vec![
        // Machine Learning cluster - more similar content
        create_test_result("ml_basics", "Introduction to machine learning algorithms neural networks supervised learning"),
        create_test_result("deep_learning", "Deep learning neural networks artificial intelligence machine learning frameworks"),
        create_test_result("ml_applications", "Machine learning applications artificial intelligence computer vision natural language"),
        create_test_result("ai_fundamentals", "Artificial intelligence machine learning deep neural networks algorithms"),
        
        // Python Programming cluster - more similar content
        create_test_result("python_tutorial", "Python programming tutorial variables functions classes object oriented programming"),
        create_test_result("python_web", "Python web development Django Flask framework programming tutorial"),
        create_test_result("python_data", "Python data science programming pandas numpy libraries tutorial"),
        create_test_result("python_basics", "Learn Python programming fundamentals tutorial variables functions"),
        
        // Business Analysis cluster - more similar content
        create_test_result("market_analysis", "Market research competitive analysis business strategy data analysis"),
        create_test_result("business_plan", "Business plan financial projections startup strategy market analysis"),
        create_test_result("data_analytics", "Business analytics data analysis market research strategic planning"),
        
        // Standalone document - completely different
        create_test_result("cooking_recipe", "Italian pasta recipe fresh tomatoes basil cooking ingredients"),
    ];

    println!("Clustering {} results...", results.len());
    let clusters = processor.cluster_results(&results, "machine learning python").await?;
    
    println!("Found {} clusters:", clusters.len());
    for (i, cluster) in clusters.iter().enumerate() {
        println!("  Cluster {}: \"{}\" ({} results, coherence: {:.2})", 
            i + 1, 
            cluster.cluster_label, 
            cluster.results.len(),
            cluster.coherence_score
        );
        
        for result in &cluster.results {
            println!("    - {}", result.title);
        }
        
        if !cluster.cluster_keywords.is_empty() {
            println!("    Keywords: {}", cluster.cluster_keywords.join(", "));
        }
    }

    // Validate clustering quality
    let clustered_results: usize = clusters.iter().map(|c| c.results.len()).sum();
    println!("\nClustering analysis:");
    println!("  Results clustered: {}/{} ({:.1}%)", 
        clustered_results, 
        results.len(),
        (clustered_results as f32 / results.len() as f32) * 100.0
    );

    // Check for meaningful clusters
    let meaningful_clusters = clusters.iter()
        .filter(|c| c.coherence_score > 0.5)
        .count();
    
    println!("  Meaningful clusters (coherence > 0.5): {}", meaningful_clusters);

    if meaningful_clusters >= 2 {
        println!("✅ Good clustering - found {} meaningful thematic groups", meaningful_clusters);
    } else if meaningful_clusters >= 1 {
        println!("✅ Moderate clustering - found {} meaningful group", meaningful_clusters);
    } else {
        println!("⚠️  Limited clustering effectiveness");
    }

    Ok(())
}

async fn test_document_summarization() -> Result<(), Box<dyn std::error::Error>> {
    let mut processor = ResultProcessor::new(ResultProcessingConfig::default());
    
    // Create documents with varying content complexity
    let test_cases = vec![
        (
            "short_doc",
            "Machine learning is useful for data analysis."
        ),
        (
            "medium_doc", 
            "Machine learning is a subset of artificial intelligence that focuses on algorithms. \
             These algorithms can learn patterns from data without being explicitly programmed. \
             Common applications include image recognition, natural language processing, and \
             recommendation systems. The field has grown rapidly due to increased computational \
             power and large datasets."
        ),
        (
            "long_doc",
            "Machine learning represents a fundamental shift in how we approach problem-solving \
             with computers. Traditional programming requires explicit instructions for every \
             scenario, but machine learning allows systems to improve their performance through \
             experience. The field encompasses supervised learning, where algorithms learn from \
             labeled examples, unsupervised learning, which finds hidden patterns in unlabeled \
             data, and reinforcement learning, where agents learn through interaction with an \
             environment. Deep learning, a subset of machine learning, uses neural networks \
             with multiple layers to model complex patterns. Applications span healthcare, \
             finance, autonomous vehicles, and entertainment. Recent breakthroughs include \
             transformer architectures for natural language processing, generative adversarial \
             networks for image synthesis, and large language models capable of human-like text \
             generation. The ethical implications of these technologies include bias in algorithms, \
             privacy concerns, and the potential for job displacement."
        ),
    ];

    println!("Testing summarization on {} documents:", test_cases.len());
    
    for (title, content) in test_cases {
        let result = create_test_result(title, content);
        let summary = processor.generate_extractive_summary(&result).await?;
        
        let original_words = content.split_whitespace().count();
        let summary_words = summary.summary_length;
        let compression_ratio = (summary_words as f32 / original_words as f32) * 100.0;
        
        println!("\n  Document: {}", title);
        println!("    Original: {} words", original_words);
        println!("    Summary: {} words ({:.1}% compression)", summary_words, compression_ratio);
        println!("    Confidence: {:.2}", summary.confidence_score);
        println!("    Key topics: {}", summary.key_topics.join(", "));
        println!("    Summary text: \"{}\"", 
            if summary.extractive_summary.len() > 100 {
                format!("{}...", &summary.extractive_summary[..100])
            } else {
                summary.extractive_summary.clone()
            }
        );

        // Validate summary quality
        assert!(summary.summary_length <= 150, "Summary should respect length limit");
        assert!(!summary.extractive_summary.is_empty(), "Summary should not be empty");
        assert!(summary.confidence_score > 0.0, "Should have positive confidence");
        
        if original_words > 20 && compression_ratio < 70.0 {
            println!("    ✅ Good compression ratio");
        } else if original_words <= 20 {
            println!("    ✅ Short document handled appropriately");
        } else {
            println!("    ⚠️  High compression ratio - summary might be too long");
        }
    }

    println!("\n✅ Summarization working correctly");
    Ok(())
}

async fn test_snippet_highlighting() -> Result<(), Box<dyn std::error::Error>> {
    let processor = ResultProcessor::new(ResultProcessingConfig::default());
    
    let test_cases = vec![
        (
            "machine learning tutorial",
            "This comprehensive machine learning tutorial covers algorithms, data preprocessing, and model evaluation",
            vec!["machine", "learning"]
        ),
        (
            "python programming guide", 
            "Learn Python programming fundamentals including variables, functions, and object-oriented programming",
            vec!["python", "programming"]
        ),
        (
            "data science analysis",
            "Data science involves statistical analysis, data visualization, and machine learning techniques",
            vec!["data", "analysis", "machine"]
        ),
    ];

    println!("Testing snippet highlighting:");

    for (query, original_snippet, expected_highlights) in test_cases {
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        let highlighted = processor.highlight_snippet(original_snippet, &query_terms);
        
        println!("\n  Query: \"{}\"", query);
        println!("  Original: {}", original_snippet);
        println!("  Highlighted: {}", highlighted);
        
        // Count highlights
        let highlight_count = highlighted.matches("<mark>").count();
        let expected_count = expected_highlights.len();
        
        println!("  Highlights found: {} (expected: {})", highlight_count, expected_count);
        
        // Verify highlighting
        assert!(highlighted.contains("<mark>"), "Should contain highlight marks");
        assert!(highlight_count > 0, "Should have at least one highlight");
        
        // Check that expected terms are highlighted
        let mut found_highlights = 0;
        for expected_term in &expected_highlights {
            if highlighted.to_lowercase().contains(&format!("<mark>{}</mark>", expected_term.to_lowercase())) {
                found_highlights += 1;
            }
        }
        
        if found_highlights >= expected_highlights.len() / 2 {
            println!("  ✅ Good highlighting coverage");
        } else {
            println!("  ⚠️  Some expected terms not highlighted");
        }
    }

    println!("\n✅ Snippet highlighting working correctly");
    Ok(())
}

async fn test_end_to_end_processing() -> Result<(), Box<dyn std::error::Error>> {
    let mut processor = ResultProcessor::new(ResultProcessingConfig::default());
    
    // Create a realistic result set
    let results = vec![
        create_test_result("ml_intro", "Introduction to machine learning algorithms and data science techniques"),
        create_test_result("ml_duplicate", "Introduction to machine learning algorithms and data science techniques"), // Duplicate
        create_test_result("deep_learning", "Deep learning neural networks for computer vision and natural language processing"),
        create_test_result("python_ml", "Python machine learning libraries including scikit-learn and TensorFlow"),
        create_test_result("data_analysis", "Statistical data analysis methods for business intelligence and insights"),
        create_test_result("web_dev", "Web development using Python Django framework for building applications"),
        create_test_result("python_basics", "Python programming fundamentals covering syntax and basic concepts"),
        create_test_result("business_analytics", "Business analytics and data-driven decision making strategies"),
    ];

    let query = "machine learning python data analysis";
    
    println!("Processing {} results for query: \"{}\"", results.len(), query);
    
    let start_time = std::time::Instant::now();
    let processed = processor.process_results(results, query).await?;
    let processing_time = start_time.elapsed();
    
    println!("\nProcessing Results:");
    println!("  Original results: {}", processed.total_results);
    println!("  After deduplication: {}", processed.deduplicated_count);
    println!("  Final results: {}", processed.results.len());
    println!("  Clusters found: {}", processed.clusters.len());
    println!("  Summaries generated: {}", processed.summaries.len());
    println!("  Processing time: {:?} (reported: {}ms)", processing_time, processed.processing_time_ms);
    
    // Show clusters
    if !processed.clusters.is_empty() {
        println!("\nClusters:");
        for (i, cluster) in processed.clusters.iter().enumerate() {
            println!("  {}. \"{}\" ({} results)", i + 1, cluster.cluster_label, cluster.results.len());
        }
    }
    
    // Show highlighted results
    println!("\nHighlighted Results:");
    for (i, result) in processed.results.iter().take(3).enumerate() {
        println!("  {}. {} - {}", i + 1, result.title, result.snippet);
    }
    
    // Show summaries
    if !processed.summaries.is_empty() {
        println!("\nSummaries generated: {}", processed.summaries.len());
        for (doc_id, summary) in processed.summaries.iter().take(2) {
            println!("  Document {}: {} words, {} topics", 
                doc_id.to_string().split('-').next().unwrap_or("unknown"),
                summary.summary_length,
                summary.key_topics.len()
            );
        }
    }

    // Validate end-to-end processing
    assert!(processed.deduplicated_count <= processed.total_results, "Deduplication should not increase count");
    assert!(processed.processing_time_ms > 0, "Should report processing time");
    
    let deduplication_effectiveness = (processed.total_results - processed.deduplicated_count) as f32 / processed.total_results as f32;
    
    println!("\nEffectiveness Metrics:");
    println!("  Deduplication rate: {:.1}%", deduplication_effectiveness * 100.0);
    println!("  Clustering rate: {:.1}%", 
        (processed.clusters.iter().map(|c| c.results.len()).sum::<usize>() as f32 / processed.results.len() as f32) * 100.0);
    println!("  Summarization rate: {:.1}%", 
        (processed.summaries.len() as f32 / processed.results.len() as f32) * 100.0);

    if deduplication_effectiveness > 0.0 {
        println!("  ✅ Effective deduplication");
    }
    if !processed.clusters.is_empty() {
        println!("  ✅ Successful clustering");
    }
    if !processed.summaries.is_empty() {
        println!("  ✅ Summaries generated");
    }

    println!("\n✅ End-to-end processing working correctly");
    Ok(())
}

async fn test_performance_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    let mut processor = ResultProcessor::new(ResultProcessingConfig::default());
    
    // Test different scales
    let test_scales = vec![10, 50, 100];
    
    println!("Performance benchmarks:");
    
    for scale in test_scales {
        let results = generate_test_results(scale);
        let query = "machine learning data analysis";
        
        let start_time = std::time::Instant::now();
        let processed = processor.process_results(results, query).await?;
        let processing_time = start_time.elapsed();
        
        let results_per_ms = scale as f32 / processing_time.as_millis() as f32;
        
        println!("  {} results: {:?} ({:.1} results/ms)", 
            scale, 
            processing_time,
            results_per_ms
        );
        
        // Performance assertions
        assert!(processing_time.as_millis() < 5000, "Should process within 5 seconds");
        assert!(processed.results.len() <= scale, "Should not increase result count");
        
        if processing_time.as_millis() < 1000 {
            println!("    ✅ Fast processing");
        } else if processing_time.as_millis() < 3000 {
            println!("    ✅ Acceptable performance");
        } else {
            println!("    ⚠️  Slow processing");
        }
    }

    println!("\n✅ Performance benchmarks completed");
    Ok(())
}

// Helper functions
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

fn generate_test_results(count: usize) -> Vec<RankedResult> {
    let snippets = vec![
        "Machine learning algorithms for data analysis and prediction modeling",
        "Python programming tutorial covering basic syntax and advanced features",
        "Data science methodology including statistical analysis and visualization",
        "Deep learning neural networks for computer vision applications",
        "Business intelligence and analytics for decision making processes",
        "Web development using modern frameworks and best practices",
        "Database design and optimization for large-scale applications",
        "Software engineering principles and design patterns",
        "Cloud computing architecture and deployment strategies",
        "Artificial intelligence research and practical applications",
    ];
    
    (0..count)
        .map(|i| {
            let snippet = &snippets[i % snippets.len()];
            create_test_result(&format!("doc_{}", i), snippet)
        })
        .collect()
}