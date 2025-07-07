use desktop_ai_search::commands::{indexing, search};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing Desktop AI Search Integration\n");
    
    // Test 1: Index some test documents
    println!("=== Test 1: Indexing Documents ===");
    
    let test_files = [
        "/tmp/test-docs/rust-programming.txt",
        "/tmp/test-docs/machine-learning.txt", 
        "/tmp/test-docs/web-development.txt"
    ];
    
    for file_path in &test_files {
        if Path::new(file_path).exists() {
            match indexing::index_file(file_path.to_string()).await {
                Ok(()) => println!("✓ Successfully indexed: {}", file_path),
                Err(e) => println!("✗ Failed to index {}: {}", file_path, e),
            }
        } else {
            println!("✗ File not found: {}", file_path);
        }
    }
    
    // Check indexing status
    let (indexed, total) = indexing::get_indexing_status().await?;
    println!("📊 Indexing Status: {}/{} files", indexed, total);
    
    println!();
    
    // Test 2: Search for different queries
    println!("=== Test 2: Search Functionality ===");
    
    let test_queries = [
        "rust programming",
        "machine learning",
        "web development",
        "python",
        "memory safety",
        "neural networks",
        "JavaScript frameworks"
    ];
    
    for query in &test_queries {
        println!("\n🔍 Searching for: '{}'", query);
        match search::search_documents(query.to_string()).await {
            Ok(results) => {
                if results.is_empty() {
                    println!("   No results found");
                } else {
                    println!("   Found {} results:", results.len());
                    for (i, result) in results.iter().take(3).enumerate() {
                        println!("   {}. {} (score: {:.2})", 
                               i + 1, 
                               result.content.chars().take(80).collect::<String>(),
                               result.score);
                    }
                }
            }
            Err(e) => println!("   ✗ Search failed: {}", e),
        }
    }
    
    println!("\n✅ Testing complete!");
    Ok(())
}