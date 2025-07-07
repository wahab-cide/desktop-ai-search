use desktop_ai_search::core::query_intent::{QueryIntentClassifier, QueryClassifierConfig};
use desktop_ai_search::core::cached_query_classifier::{CachedQueryClassifier, QueryClassificationService, CachedClassifierConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Testing Query Intent Classification System\n");

    // Test 1: Basic Query Classification
    println!("=== Test 1: Basic Query Classification ===");
    test_basic_classification().await?;

    // Test 2: Cache Performance
    println!("\n=== Test 2: Cache Performance ===");
    test_cache_performance().await?;

    // Test 3: Entity Recognition
    println!("\n=== Test 3: Entity Recognition ===");
    test_entity_recognition().await?;

    // Test 4: Temporal Parsing
    println!("\n=== Test 4: Temporal Expression Parsing ===");
    test_temporal_parsing().await?;

    // Test 5: Search Strategy Selection
    println!("\n=== Test 5: Search Strategy Selection ===");
    test_search_strategy_selection().await?;

    // Test 6: Service Integration
    println!("\n=== Test 6: Service Integration ===");
    test_service_integration().await?;

    println!("\n✅ All tests completed successfully!");
    Ok(())
}

async fn test_basic_classification() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = QueryIntentClassifier::default()?;
    
    let test_queries = vec![
        "find documents by John",
        "search for presentations about AI",
        "what is machine learning?",
        "recent files from last week",
        "similar to this report",
        "PDF files larger than 5MB",
    ];

    for query in test_queries {
        let result = classifier.analyze_query(query).await?;
        println!("Query: \"{}\"", query);
        println!("  Intents: {:?}", result.intents.keys().collect::<Vec<_>>());
        println!("  Entities: {}", result.entities.len());
        println!("  Complexity: {:.2}", result.complexity_score);
        println!("  Strategy: {:?}", result.search_strategy);
        println!();
    }

    Ok(())
}

async fn test_cache_performance() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = CachedQueryClassifier::default().await?;
    
    let test_query = "find documents by John from last week";
    
    println!("Testing cache performance with query: \"{}\"", test_query);
    
    // First call - cache miss
    let start = std::time::Instant::now();
    let result1 = classifier.analyze_query(test_query).await?;
    let first_duration = start.elapsed();
    
    // Second call - cache hit
    let start = std::time::Instant::now();
    let result2 = classifier.analyze_query(test_query).await?;
    let second_duration = start.elapsed();
    
    // Third call - cache hit
    let start = std::time::Instant::now();
    let result3 = classifier.analyze_query(test_query).await?;
    let third_duration = start.elapsed();
    
    println!("First call (cache miss): {:?}", first_duration);
    println!("Second call (cache hit): {:?}", second_duration);
    println!("Third call (cache hit): {:?}", third_duration);
    
    // Verify results are the same
    println!("Results identical: {}", 
        std::ptr::eq(result1.as_ref(), result2.as_ref()) && 
        std::ptr::eq(result2.as_ref(), result3.as_ref())
    );
    
    // Check cache stats
    let stats = classifier.get_cache_stats().await;
    println!("Cache stats:");
    println!("  Hits: {}", stats.hit_count);
    println!("  Misses: {}", stats.miss_count);
    println!("  Hit rate: {:.1}%", stats.hit_rate * 100.0);
    println!("  Entries: {}", stats.entry_count);

    Ok(())
}

async fn test_entity_recognition() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = QueryIntentClassifier::default()?;
    
    let test_queries = vec![
        "documents from John Smith",
        "files sent by Jane Doe last Friday",
        "PDF files larger than 10MB",
        "presentations created by team lead",
        "images from vacation.jpg",
    ];

    for query in test_queries {
        let result = classifier.analyze_query(query).await?;
        println!("Query: \"{}\"", query);
        for entity in &result.entities {
            println!("  Entity: {:?} = \"{}\" (confidence: {:.2})", 
                entity.entity_type, entity.text, entity.confidence);
        }
        if result.entities.is_empty() {
            println!("  No entities detected");
        }
        println!();
    }

    Ok(())
}

async fn test_temporal_parsing() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = QueryIntentClassifier::default()?;
    
    let test_queries = vec![
        "files from yesterday",
        "documents from last week",
        "recent presentations",
        "files modified today",
        "reports from last month",
        "3 days ago",
    ];

    for query in test_queries {
        let result = classifier.analyze_query(query).await?;
        println!("Query: \"{}\"", query);
        for expr in &result.temporal_expressions {
            println!("  Temporal: \"{}\" (confidence: {:.2})", 
                expr.original_text, expr.confidence);
            if let Some(start) = expr.start_date {
                println!("    Start: {}", start.format("%Y-%m-%d %H:%M"));
            }
            if let Some(end) = expr.end_date {
                println!("    End: {}", end.format("%Y-%m-%d %H:%M"));
            }
            if let Some(days) = expr.relative_days {
                println!("    Relative: {} days", days);
            }
        }
        if result.temporal_expressions.is_empty() {
            println!("  No temporal expressions detected");
        }
        println!();
    }

    Ok(())
}

async fn test_search_strategy_selection() -> Result<(), Box<dyn std::error::Error>> {
    let classifier = QueryIntentClassifier::default()?;
    
    let test_cases = vec![
        ("find documents", "Should use FullTextOnly"),
        ("what is artificial intelligence?", "Should use VectorOnly (QA)"),
        ("similar to this report", "Should use VectorOnly (similarity)"),
        ("files from John last week", "Should use Hybrid (person + temporal)"),
        ("complex query with multiple entities and temporal expressions from various people", "Should use MultiStage (high complexity)"),
    ];

    for (query, expected) in test_cases {
        let result = classifier.analyze_query(query).await?;
        println!("Query: \"{}\"", query);
        println!("  Expected: {}", expected);
        println!("  Strategy: {:?}", result.search_strategy);
        println!("  Complexity: {:.2}", result.complexity_score);
        println!();
    }

    Ok(())
}

async fn test_service_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating QueryClassificationService...");
    let service = QueryClassificationService::default().await?;
    
    println!("Service ready: {}", service.is_ready());
    
    // Test single query
    let query = "find recent presentations by the marketing team";
    let result = service.classify_query(query).await?;
    println!("Single query result:");
    println!("  Query: \"{}\"", query);
    println!("  Intents: {}", result.intents.len());
    println!("  Entities: {}", result.entities.len());
    
    // Test batch processing
    let batch_queries = vec![
        "recent documents".to_string(),
        "files by John".to_string(),
        "PDF presentations".to_string(),
        "similar reports".to_string(),
    ];
    
    println!("\nBatch processing {} queries...", batch_queries.len());
    let start = std::time::Instant::now();
    let batch_results = service.classify_queries_batch(&batch_queries).await?;
    let duration = start.elapsed();
    
    println!("Batch processing completed in {:?}", duration);
    println!("Average time per query: {:?}", duration / batch_queries.len() as u32);
    
    for (i, result) in batch_results.iter().enumerate() {
        println!("  Query {}: {} intents, {} entities", 
            i + 1, result.intents.len(), result.entities.len());
    }
    
    // Performance stats
    let stats = service.get_performance_stats().await;
    println!("\nFinal performance stats:");
    println!("  Total requests: {}", stats.hit_count + stats.miss_count);
    println!("  Cache hit rate: {:.1}%", stats.hit_rate * 100.0);
    println!("  Cache entries: {}", stats.entry_count);

    Ok(())
}