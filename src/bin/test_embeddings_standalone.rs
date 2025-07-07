use desktop_ai_search::test_embeddings::{test_basic_functionality, test_embedding_pipeline};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing Embedding Pipeline");
    println!("==============================");
    
    // Run basic functionality tests first
    if let Err(e) = test_basic_functionality() {
        eprintln!("❌ Basic functionality tests failed: {}", e);
        return Err(e);
    }
    
    // Run full embedding pipeline tests
    if let Err(e) = test_embedding_pipeline().await {
        eprintln!("❌ Embedding pipeline tests failed: {}", e);
        return Err(e);
    }
    
    println!("\n🎉 All embedding tests completed successfully!");
    Ok(())
}