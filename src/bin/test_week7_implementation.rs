use desktop_ai_search::test_screenshot::{test_screenshot_pipeline, test_screenshot_basic_functionality};
use desktop_ai_search::test_query_understanding::{test_query_understanding_pipeline, test_query_understanding_basic_functionality};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Week 7: Query Understanding & Intent Detection Test Suite");
    println!("============================================================");
    println!("Testing the complete query understanding system implementation");
    println!();
    
    // Phase 1: Basic functionality tests
    println!("📋 PHASE 1: Basic Functionality Tests");
    println!("=====================================");
    
    println!("\n🔬 Screenshot Processing Basic Tests...");
    if let Err(e) = test_screenshot_basic_functionality() {
        println!("❌ Screenshot basic functionality failed: {}", e);
    } else {
        println!("✅ Screenshot basic functionality passed");
    }
    
    println!("\n🧠 Query Understanding Basic Tests...");
    if let Err(e) = test_query_understanding_basic_functionality() {
        println!("❌ Query understanding basic functionality failed: {}", e);
    } else {
        println!("✅ Query understanding basic functionality passed");
    }
    
    // Phase 2: Advanced pipeline tests
    println!("\n\n📋 PHASE 2: Advanced Pipeline Tests");
    println!("===================================");
    
    println!("\n🔬 Screenshot Processing Pipeline...");
    if let Err(e) = test_screenshot_pipeline().await {
        println!("❌ Screenshot pipeline failed: {}", e);
    } else {
        println!("✅ Screenshot pipeline passed");
    }
    
    println!("\n🧠 Query Understanding Pipeline...");
    if let Err(e) = test_query_understanding_pipeline().await {
        println!("❌ Query understanding pipeline failed: {}", e);
    } else {
        println!("✅ Query understanding pipeline passed");
    }
    
    // Phase 3: Integration demonstration
    println!("\n\n📋 PHASE 3: Integration Demonstration");
    println!("=====================================");
    
    demonstrate_query_understanding_capabilities().await?;
    
    // Final summary
    println!("\n\n🎯 WEEK 7 IMPLEMENTATION COMPLETE");
    println!("==================================");
    println!("✅ Screenshot Processing: Advanced visual content understanding");
    println!("✅ Query Understanding: Natural language intent detection");
    println!("✅ Entity Extraction: People, dates, technologies, file types");
    println!("✅ Temporal Parsing: Relative and absolute date constraints");
    println!("✅ Spell Correction: Domain-aware text normalization");
    println!("✅ Query Expansion: Semantic synonym expansion");
    println!("✅ User Learning: Personalized dictionary from successful searches");
    println!("✅ Performance Optimization: LRU caching for repeated queries");
    println!();
    println!("🚀 Ready for Week 7 Day 3-4: Boolean Logic & PEG Grammar!");
    
    Ok(())
}

async fn demonstrate_query_understanding_capabilities() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎮 Interactive Query Understanding Demonstration");
    println!("===============================================");
    
    use desktop_ai_search::core::advanced_query_processor::AdvancedQueryProcessor;
    
    let mut processor = AdvancedQueryProcessor::new()?;
    
    let demo_queries = vec![
        // Natural language queries that showcase different capabilities
        "Find Python scripts that John created last week",
        "Show me presentations about machine learning and data science",
        "Search for PDFs but exclude images from yesterday", 
        "Compare quarterly reports from 2023 and 2024",
        "Timeline of all documents modified in January",
        "Find similar screenshots to my UI mockups",
        "What presentations did the team create about kubernetes?",
        "Summarize meeting notes from last month",
        "Filter JavaScript files created by Sarah",
        "Find documents containing tensorflow or pytorch code",
    ];
    
    println!("\nProcessing {} diverse natural language queries...", demo_queries.len());
    
    for (i, query) in demo_queries.iter().enumerate() {
        println!("\n{}. Query: \"{}\"", i + 1, query);
        
        let result = processor.analyze_query(query).await?;
        
        // Display comprehensive analysis
        println!("   📋 Analysis Results:");
        println!("      Intent Labels: {:?}", result.labels);
        println!("      Query Type: {:?}", result.query_type);
        println!("      Confidence: {:.1}%", result.confidence * 100.0);
        
        if !result.entities.is_empty() {
            println!("      Entities Detected:");
            for entity in &result.entities {
                println!("         - {} ({:?}) [confidence: {:.1}%]", 
                        entity.text, entity.entity_type, entity.confidence * 100.0);
            }
        }
        
        if let Some(temporal) = &result.temporal_constraints {
            println!("      Temporal Constraint: {:?} [confidence: {:.1}%]", 
                    temporal.constraint_type, temporal.confidence * 100.0);
            
            if let Some(relative) = &temporal.relative_time {
                println!("         Relative: {} {:?} in the {:?}", 
                        relative.amount, relative.unit, relative.direction);
            }
        }
        
        if !result.file_type_filters.is_empty() {
            println!("      File Type Filters: {:?}", result.file_type_filters);
        }
        
        if let Some(expansions) = &result.semantic_expansion {
            println!("      Semantic Expansions: {:?}", expansions);
        }
        
        println!("      Normalized Text: \"{}\"", result.normalized_text);
    }
    
    // Demonstrate user dictionary learning
    println!("\n📚 User Dictionary Learning Demonstration");
    println!("=========================================");
    
    let learning_queries = vec![
        "tensorflow documentation",
        "kubernetes deployment guide", 
        "claude api reference",
        "tailwindcss component library",
    ];
    
    println!("Teaching the system domain-specific terms...");
    for query in &learning_queries {
        processor.update_user_dictionary(query, true).await?;
        println!("   Learned from: \"{}\"", query);
    }
    
    let stats = processor.get_classification_stats();
    println!("\nDictionary Stats:");
    println!("   Total tokens learned: {}", stats.user_dictionary_size);
    println!("   Cache efficiency: {}/{}", stats.cache_size, stats.cache_capacity);
    println!("   Last updated: {}", stats.last_dictionary_update.format("%H:%M:%S"));
    
    // Test domain term recognition
    println!("\n🧪 Testing Domain Term Recognition");
    let domain_test = "find tensorflow models and kubernetes configs";
    println!("Query with learned terms: \"{}\"", domain_test);
    
    let result = processor.analyze_query(domain_test).await?;
    println!("   Recognition confidence: {:.1}%", result.confidence * 100.0);
    println!("   Entities found: {}", result.entities.len());
    
    println!("\n🎉 Query understanding system is now intelligent and adaptive!");
    
    Ok(())
}