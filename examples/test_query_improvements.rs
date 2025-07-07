use desktop_ai_search::core::query_intent::{QueryIntentClassifier, Intent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing Query Intent Improvements\n");

    let classifier = QueryIntentClassifier::default()?;

    // Test specific intent detection improvements
    test_intent_improvements(&classifier).await?;
    test_entity_improvements(&classifier).await?;
    test_complex_queries(&classifier).await?;

    Ok(())
}

async fn test_intent_improvements(classifier: &QueryIntentClassifier) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Intent Detection Improvements ===");
    
    let test_cases = vec![
        ("what is machine learning?", vec![Intent::QuestionAnswering], "Should detect QA intent"),
        ("how does AI work?", vec![Intent::QuestionAnswering], "Should detect QA intent"),
        ("why is this happening?", vec![Intent::QuestionAnswering], "Should detect QA intent"),
        
        ("similar to this document", vec![Intent::SimilaritySearch], "Should detect similarity intent"),
        ("find related files", vec![Intent::SimilaritySearch], "Should detect similarity intent"),
        ("like this report", vec![Intent::SimilaritySearch], "Should detect similarity intent"),
        
        ("find PDF files by John", vec![Intent::DocumentSearch, Intent::PersonSearch, Intent::TypeSearch], "Should detect multiple intents"),
        ("recent presentations from team", vec![Intent::DocumentSearch, Intent::TemporalSearch], "Should detect multiple intents"),
    ];

    for (query, expected_intents, description) in test_cases {
        let result = classifier.analyze_query(query).await?;
        let detected_intents: Vec<&Intent> = result.intents.keys().collect();
        
        println!("Query: \"{}\"", query);
        println!("  Expected: {:?}", expected_intents);
        println!("  Detected: {:?}", detected_intents);
        println!("  Description: {}", description);
        
        // Check if we got the main expected intents
        let matches: usize = expected_intents.iter()
            .map(|intent| if result.intents.contains_key(intent) { 1 } else { 0 })
            .sum();
        let score = matches as f32 / expected_intents.len() as f32;
        println!("  Match Score: {:.1}% ({}/{})", score * 100.0, matches, expected_intents.len());
        
        if score >= 0.5 {
            println!("  ✅ Good detection");
        } else {
            println!("  ⚠️  Could be improved");
        }
        println!();
    }

    Ok(())
}

async fn test_entity_improvements(classifier: &QueryIntentClassifier) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Entity Recognition Improvements ===");
    
    let test_cases = vec![
        ("documents from John Smith", "Person", "John Smith"),
        ("files sent by Jane Doe", "Person", "Jane Doe"),
        ("reports by Dr. Johnson", "Person", "Dr. Johnson"),
        ("presentations from Sarah Wilson last week", "Person", "Sarah Wilson"),
        
        ("PDF files larger than 5MB", "FileType", "PDF"),
        ("Excel spreadsheets", "FileType", "Excel"),
        ("PowerPoint presentations", "FileType", "PowerPoint"),
        
        ("files larger than 10MB", "Size", "10MB"),
        ("small documents", "Size", "small"),
        ("huge files", "Size", "huge"),
    ];

    for (query, expected_type, expected_text) in test_cases {
        let result = classifier.analyze_query(query).await?;
        
        println!("Query: \"{}\"", query);
        println!("  Expected: {} = \"{}\"", expected_type, expected_text);
        
        if result.entities.is_empty() {
            println!("  Detected: No entities");
            println!("  ⚠️  Expected entity not found");
        } else {
            for entity in &result.entities {
                println!("  Detected: {:?} = \"{}\" (confidence: {:.2})", 
                    entity.entity_type, entity.text, entity.confidence);
            }
            
            // Check if we found the expected entity type
            let found_expected = result.entities.iter().any(|e| {
                let type_match = match expected_type {
                    "Person" => matches!(e.entity_type, desktop_ai_search::core::query_intent::EntityType::Person),
                    "FileType" => matches!(e.entity_type, desktop_ai_search::core::query_intent::EntityType::FileType),
                    "Size" => matches!(e.entity_type, desktop_ai_search::core::query_intent::EntityType::Size),
                    _ => false,
                };
                type_match && e.text.to_lowercase().contains(&expected_text.to_lowercase())
            });
            
            if found_expected {
                println!("  ✅ Found expected entity");
            } else {
                println!("  ⚠️  Expected entity not found");
            }
        }
        println!();
    }

    Ok(())
}

async fn test_complex_queries(classifier: &QueryIntentClassifier) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Complex Query Analysis ===");
    
    let complex_queries = vec![
        "find recent PDF presentations by John Smith about machine learning from last month",
        "what are the best practices for AI development in documents sent by the research team?",
        "show me Excel files similar to the quarterly report that were modified yesterday",
        "locate large PowerPoint files created by marketing team members in the past week",
    ];

    for query in complex_queries {
        let result = classifier.analyze_query(query).await?;
        
        println!("Query: \"{}\"", query);
        println!("  Intents: {} ({})", result.intents.len(), 
            result.intents.keys().map(|i| format!("{:?}", i)).collect::<Vec<_>>().join(", "));
        println!("  Entities: {} ({})", result.entities.len(),
            result.entities.iter().map(|e| format!("{:?}:{}", e.entity_type, e.text)).collect::<Vec<_>>().join(", "));
        println!("  Temporal: {}", result.temporal_expressions.len());
        println!("  Complexity: {:.2}", result.complexity_score);
        println!("  Strategy: {:?}", result.search_strategy);
        
        // Analyze complexity appropriateness
        let expected_complexity = if result.intents.len() >= 3 || result.entities.len() >= 2 || !result.temporal_expressions.is_empty() {
            "High (>0.6)"
        } else if result.intents.len() >= 2 || result.entities.len() >= 1 {
            "Medium (0.3-0.6)"
        } else {
            "Low (<0.3)"
        };
        
        println!("  Expected complexity: {}", expected_complexity);
        
        if result.complexity_score > 0.6 {
            println!("  ✅ High complexity properly detected");
        } else if result.complexity_score > 0.3 {
            println!("  ✅ Medium complexity detected");
        } else {
            println!("  ⚠️  Complexity might be underestimated");
        }
        println!();
    }

    Ok(())
}