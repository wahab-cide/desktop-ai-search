use desktop_ai_search::database::Database;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Testing Simple Database Creation");
    println!("==================================");
    
    // Test basic database creation
    println!("\n1. Creating database...");
    let db_path = "test_simple.db";
    
    match Database::new(db_path) {
        Ok(_db) => {
            println!("   ✅ Database created successfully");
            
            // Clean up
            std::fs::remove_file(db_path).ok();
            println!("   ✅ Database test completed");
        }
        Err(e) => {
            println!("   ❌ Database creation failed: {}", e);
            std::fs::remove_file(db_path).ok();
            return Err(e.into());
        }
    }
    
    Ok(())
}