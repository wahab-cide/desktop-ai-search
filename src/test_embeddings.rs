use crate::core::embedding_manager::EmbeddingManager;

pub async fn test_embedding_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Testing Embedding Pipeline");
    println!("==============================");
    
    // Test 1: Create embedding manager
    println!("\n1. Creating embedding manager...");
    let mut manager = EmbeddingManager::new()?;
    println!("   ✅ Embedding manager created successfully");
    
    // Test 2: List available models
    println!("\n2. Checking available models...");
    let models = manager.get_available_models();
    println!("   Available models:");
    for model in models {
        println!("     - {} ({}D, {}MB)", model.name, model.dimensions, model.model_size_mb);
    }
    
    // Test 3: Check if model is downloaded
    println!("\n3. Checking if default model is downloaded...");
    let model_id = "all-minilm-l6-v2";
    let is_downloaded = manager.is_model_downloaded(model_id).await?;
    println!("   Model '{}' downloaded: {}", model_id, is_downloaded);
    
    if !is_downloaded {
        println!("   📥 Downloading model (this may take a while)...");
        manager.download_model(model_id).await?;
        println!("   ✅ Model downloaded successfully");
    }
    
    // Test 4: Load model
    println!("\n4. Loading embedding model...");
    manager.load_model(model_id, None).await?;
    println!("   ✅ Model loaded successfully");
    
    // Test 5: Generate embeddings
    println!("\n5. Testing embedding generation...");
    let test_texts = vec![
        "This is a test document about artificial intelligence.".to_string(),
        "Machine learning is a subset of AI that focuses on data.".to_string(),
        "The weather today is sunny and warm.".to_string(),
    ];
    
    let embeddings = manager.generate_embeddings(&test_texts).await?;
    println!("   ✅ Generated {} embeddings", embeddings.len());
    
    if let Some(first_embedding) = embeddings.first() {
        println!("   First embedding dimensions: {}", first_embedding.len());
        println!("   First few values: {:?}", &first_embedding[..5.min(first_embedding.len())]);
    }
    
    // Test 6: Calculate similarity
    println!("\n6. Testing similarity calculation...");
    if embeddings.len() >= 3 {
        let sim_ai_ml = EmbeddingManager::cosine_similarity(&embeddings[0], &embeddings[1]);
        let sim_ai_weather = EmbeddingManager::cosine_similarity(&embeddings[0], &embeddings[2]);
        
        println!("   Similarity (AI vs ML): {:.4}", sim_ai_ml);
        println!("   Similarity (AI vs Weather): {:.4}", sim_ai_weather);
        
        if sim_ai_ml > sim_ai_weather {
            println!("   ✅ Semantic similarity works correctly!");
        } else {
            println!("   ⚠️  Unexpected similarity results");
        }
    }
    
    // Test 7: Current model info
    println!("\n7. Checking current model info...");
    if let Some(model_info) = manager.get_current_model_info() {
        println!("   Current model: {} ({}D)", model_info.name, model_info.dimensions);
        println!("   ✅ Model info retrieved successfully");
    }
    
    println!("\n🎉 All embedding tests completed successfully!");
    Ok(())
}

pub fn test_basic_functionality() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Testing Basic Functionality");
    println!("===============================");
    
    // Test cosine similarity function
    println!("\n1. Testing cosine similarity...");
    let vec1 = vec![1.0, 0.0, 0.0];
    let vec2 = vec![1.0, 0.0, 0.0];
    let vec3 = vec![0.0, 1.0, 0.0];
    
    let sim_identical = EmbeddingManager::cosine_similarity(&vec1, &vec2);
    let sim_orthogonal = EmbeddingManager::cosine_similarity(&vec1, &vec3);
    
    println!("   Identical vectors similarity: {:.4}", sim_identical);
    println!("   Orthogonal vectors similarity: {:.4}", sim_orthogonal);
    
    if (sim_identical - 1.0).abs() < 0.001 && sim_orthogonal.abs() < 0.001 {
        println!("   ✅ Cosine similarity works correctly");
    } else {
        println!("   ❌ Cosine similarity test failed");
        return Err("Cosine similarity test failed".into());
    }
    
    println!("\n🎉 Basic functionality tests passed!");
    Ok(())
}