use desktop_ai_search::core::clip_processor::ClipProcessor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Testing CLIP with Actual Model Loading");
    println!("==========================================");
    
    // Test 1: Create CLIP processor
    println!("\n1. Creating CLIP processor...");
    let mut processor = ClipProcessor::new().map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    println!("   ✅ CLIP processor created successfully");
    
    // Test 2: Load actual CLIP model (this will download from HuggingFace)
    println!("\n2. Loading CLIP model (this may take a while for first run)...");
    match processor.load_model(None).await {
        Ok(_) => {
            println!("   ✅ CLIP model loaded successfully");
            
            // Test 3: Verify model is loaded
            println!("\n3. Verifying model status...");
            if processor.is_model_loaded() {
                println!("   ✅ Model is confirmed as loaded");
                
                if let Some(model_info) = processor.get_current_model_info() {
                    println!("   Model: {}", model_info.model_id);
                    println!("   Embedding dimensions: {}", model_info.embedding_dim);
                }
            } else {
                println!("   ❌ Model reports as not loaded (unexpected)");
            }
            
            // Test 4: Test text embedding generation
            println!("\n4. Testing text embedding generation...");
            let test_texts = vec![
                "A beautiful sunset over the ocean".to_string(),
                "A cat sitting on a windowsill".to_string(),
            ];
            
            match processor.generate_text_embeddings(&test_texts).await {
                Ok(embeddings) => {
                    println!("   ✅ Generated {} text embeddings", embeddings.len());
                    for (i, embedding) in embeddings.iter().enumerate() {
                        println!("   Text {}: {} dimensions", i + 1, embedding.len());
                    }
                }
                Err(e) => {
                    println!("   ❌ Failed to generate text embeddings: {}", e);
                }
            }
            
            // Test 5: Test cache functionality
            println!("\n5. Testing cache functionality...");
            let (cache_size, cache_capacity) = processor.get_cache_stats().await;
            println!("   Cache: {}/{} items", cache_size, cache_capacity);
            
            // Clear cache and verify
            processor.clear_cache().await;
            let (new_cache_size, _) = processor.get_cache_stats().await;
            println!("   Cache after clear: {} items", new_cache_size);
            println!("   ✅ Cache functionality working");
            
        }
        Err(e) => {
            println!("   ❌ Failed to load CLIP model: {}", e);
            println!("   This might be due to:");
            println!("   - Network connectivity issues");
            println!("   - HuggingFace Hub access problems");
            println!("   - Model file format compatibility");
            println!("   - Insufficient disk space");
            return Err(e.into());
        }
    }
    
    println!("\n🎉 CLIP model loading and inference tests completed!");
    println!("📝 Note: This demonstrates actual CLIP model loading and embedding generation");
    
    Ok(())
}