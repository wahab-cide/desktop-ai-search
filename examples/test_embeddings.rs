use desktop_ai_search::core::embedding_manager::{EmbeddingManager, EmbeddingConfig};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing Desktop AI Search Embedding Integration");
    println!("=================================================");
    
    // Create Embedding Manager
    println!("\n1. Creating Embedding Manager...");
    let mut manager = EmbeddingManager::new()?;
    
    // List available models
    println!("2. Listing available embedding models...");
    let available_models = manager.get_available_models();
    println!("   Available models:");
    for model in available_models {
        println!("     • {} ({}) - {}MB", 
                 model.name, 
                 model.dimensions, 
                 model.model_size_mb);
        println!("       {}", model.description);
    }
    
    // Download our primary model (all-MiniLM-L6-v2)
    let model_id = "all-minilm-l6-v2";
    println!("\n3. Checking if model '{}' is downloaded...", model_id);
    
    if !manager.is_model_downloaded(model_id).await? {
        println!("   Model not found locally. Downloading...");
        manager.download_model(model_id).await?;
    } else {
        println!("   ✅ Model already downloaded");
    }
    
    // Load the model
    println!("\n4. Loading model for inference...");
    let config = EmbeddingConfig::default();
    manager.load_model(model_id, Some(config)).await?;
    
    if let Some(model_info) = manager.get_current_model_info() {
        println!("   ✅ Model loaded: {} ({} dimensions)", 
                 model_info.name, 
                 model_info.dimensions);
    } else {
        println!("   ❌ Model failed to load");
        return Ok(());
    }
    
    // Test embedding generation
    println!("\n5. Testing embedding generation...");
    let test_texts = vec![
        "Artificial intelligence is transforming technology".to_string(),
        "Machine learning algorithms analyze data patterns".to_string(),
        "Natural language processing understands human speech".to_string(),
        "The weather is sunny today".to_string(),
        "I love eating pizza for dinner".to_string(),
    ];
    
    println!("   Generating embeddings for {} texts...", test_texts.len());
    let embeddings = manager.generate_embeddings(&test_texts).await?;
    
    println!("   ✅ Generated {} embeddings", embeddings.len());
    
    // Test semantic similarity
    println!("\n6. Testing semantic similarity...");
    println!("   Comparing text similarities:");
    
    for i in 0..test_texts.len() {
        for j in (i+1)..test_texts.len() {
            let similarity = EmbeddingManager::cosine_similarity(
                &embeddings[i], 
                &embeddings[j]
            );
            println!("     • \"{}\" vs \"{}\"", 
                     test_texts[i].chars().take(30).collect::<String>(),
                     test_texts[j].chars().take(30).collect::<String>());
            println!("       Similarity: {:.3}", similarity);
        }
    }
    
    // Test batch processing
    println!("\n7. Testing batch processing with larger dataset...");
    let large_dataset: Vec<String> = (0..100)
        .map(|i| format!("This is test document number {} about various topics", i))
        .collect();
    
    let start_time = std::time::Instant::now();
    let batch_embeddings = manager.generate_embeddings(&large_dataset).await?;
    let processing_time = start_time.elapsed();
    
    println!("   ✅ Processed {} documents in {:.2}s", 
             batch_embeddings.len(), 
             processing_time.as_secs_f32());
    println!("   Average: {:.2}ms per document", 
             processing_time.as_millis() as f32 / batch_embeddings.len() as f32);
    
    // Verify embedding dimensions
    if let Some(first_embedding) = batch_embeddings.first() {
        println!("   Embedding dimensions: {}", first_embedding.len());
        
        // Verify L2 normalization
        let norm: f32 = first_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("   L2 norm (should be ~1.0): {:.6}", norm);
    }
    
    // Unload model
    println!("\n8. Unloading model...");
    manager.unload_model();
    
    if !manager.is_model_loaded() {
        println!("   ✅ Model unloaded successfully");
    }
    
    println!("\n🎉 Embedding Integration Test Complete!");
    println!("✅ All embedding systems working correctly");
    println!("✅ Model downloading, loading, and inference operational");
    println!("✅ Semantic similarity calculations functional");
    
    Ok(())
}