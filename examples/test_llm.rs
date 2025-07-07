use desktop_ai_search::core::llm_manager::{LlmManager, InferencePreset};
use tokio;

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
    
    // Test text generation
    let test_prompt = "Hello, what is artificial intelligence?";
    println!("\n4. Testing text generation...");
    println!("   Prompt: {}", test_prompt);
    
    let response = manager.generate_with_preset(
        test_prompt,
        InferencePreset::Balanced,
        Some("You are a helpful AI assistant.".to_string())
    ).await?;
    
    println!("\n   📝 Generated Response:");
    println!("   {}", response.text);
    println!("\n   📊 Generation Statistics:");
    println!("     • Tokens generated: {}", response.tokens_generated);
    println!("     • Tokens per second: {:.2}", response.tokens_per_second);
    println!("     • Total time: {}ms", response.total_time_ms);
    println!("     • Stop reason: {:?}", response.stop_reason);
    
    // Test different presets
    println!("\n5. Testing Creative preset...");
    let creative_response = manager.generate_with_preset(
        "Tell me about space exploration",
        InferencePreset::Creative,
        None
    ).await?;
    
    println!("   📝 Creative Response: {}", creative_response.text);
    
    // Unload model
    println!("\n6. Unloading model...");
    manager.unload_model().await?;
    
    if !manager.is_model_loaded().await {
        println!("   ✅ Model unloaded successfully");
    }
    
    println!("\n🎉 LLM Integration Test Complete!");
    println!("✅ All systems working correctly with placeholder inference");
    
    Ok(())
}