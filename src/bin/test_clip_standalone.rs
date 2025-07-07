use desktop_ai_search::test_clip::{test_clip_basic_functionality, test_clip_pipeline};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Testing CLIP Integration");
    println!("============================");
    
    // Run basic functionality tests first
    if let Err(e) = test_clip_basic_functionality() {
        eprintln!("❌ Basic CLIP functionality tests failed: {}", e);
        return Err(e);
    }
    
    // Run full CLIP pipeline tests
    if let Err(e) = test_clip_pipeline().await {
        eprintln!("❌ CLIP pipeline tests failed: {}", e);
        return Err(e);
    }
    
    println!("\n🎉 All CLIP tests completed successfully!");
    Ok(())
}