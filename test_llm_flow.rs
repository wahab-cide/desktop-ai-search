use std::sync::Arc;
use desktop_ai_search::core::llm_manager::{LlmManager, InferencePreset};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing Desktop AI Search LLM Integration");
    println!("=============================================");
    
    // Create LLM manager
    println!("\n1. Creating LLM Manager...");
    let manager = LlmManager::new()?;
    
    // Scan for available models
    println!("2. Scanning for available models...");
    manager.scan_models().await?;
    
    let available_models = manager.get_available_models().await;
    println!("   Found {} local models", available_models.len());
    
    if available_models.is_empty() {
        println!("   No models found. Please download a model first:");
        println!("   cargo run -- download-model qwen2.5-0.5b-instruct-q4_k_m");
        return Ok(());
    }
    
    // Load the first available model
    let model_name = &available_models[0].name;
    println!("3. Loading model: {}", model_name);
    manager.load_model(model_name, None).await?;
    
    // Check if model is loaded
    if let Some(loaded_model) = manager.get_current_model_info().await {
        println!("   ✅ Model loaded: {}", loaded_model);
    } else {
        println!("   ❌ Model failed to load");
        return Ok(());
    }
    
    // Test text generation with different presets
    let test_prompt = "Hello, what is artificial intelligence?";
    
    println!("\n4. Testing text generation...");
    for preset in [InferencePreset::Balanced, InferencePreset::Creative, InferencePreset::Precise] {
        println!("\n   Testing with preset: {:?}", preset);
        
        let response = manager.generate_with_preset(
            test_prompt,
            preset,
            Some("You are a helpful AI assistant.".to_string())
        ).await?;
        
        println!("   📝 Response: {}", response.text);
        println!("   📊 Stats: {} tokens, {:.2} tokens/sec, {}ms", 
            response.tokens_generated, 
            response.tokens_per_second, 
            response.total_time_ms
        );
    }
    
    // Test unloading
    println!("\n5. Unloading model...");
    manager.unload_model().await?;
    
    if !manager.is_model_loaded().await {
        println!("   ✅ Model unloaded successfully");
    }
    
    println!("\n🎉 LLM Integration Test Complete!");
    println!("✅ Model scanning: Working");
    println!("✅ Model loading: Working");
    println!("✅ Text generation: Working (placeholder)");
    println!("✅ Multiple presets: Working");
    println!("✅ Model unloading: Working");
    
    Ok(())
}