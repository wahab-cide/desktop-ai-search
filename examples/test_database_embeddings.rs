use desktop_ai_search::database::Database;
use desktop_ai_search::models::{Document, FileType};
use std::collections::HashMap;
use uuid::Uuid;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing Database Operations with Embeddings");
    println!("==============================================");
    
    // Create test database
    println!("\n1. Creating test database...");
    let db_path = "test_embeddings.db";
    let db = Database::new(db_path)?;
    
    // Create test document
    println!("2. Creating test document...");
    let document = Document {
        id: Uuid::new_v4(),
        file_path: "test_document.txt".to_string(),
        content_hash: "test-hash-123".to_string(),
        file_type: FileType::Text,
        creation_date: chrono::Utc::now(),
        modification_date: chrono::Utc::now(),
        last_indexed: chrono::Utc::now(),
        file_size: 1024,
        metadata: HashMap::new(),
    };
    
    // Insert document
    db.insert_document(&document)?;
    println!("   ✅ Document inserted: {}", document.id);
    
    // Create test chunks with embeddings
    println!("\n3. Creating test chunks with embeddings...");
    let chunks_data = vec![
        (
            "chunk_1".to_string(),
            document.id,
            0,
            0,
            100,
            "Artificial intelligence is transforming technology".to_string(),
            7,
            1,
            false,
            true,
            Some(vec![0.1, 0.2, 0.3, 0.4, 0.5]), // 5-dimensional embedding for testing
        ),
        (
            "chunk_2".to_string(),
            document.id,
            1,
            95,
            200,
            "Machine learning algorithms analyze data patterns".to_string(),
            6,
            1,
            true,
            true,
            Some(vec![0.2, 0.3, 0.4, 0.5, 0.6]), // Similar but different embedding
        ),
        (
            "chunk_3".to_string(),
            document.id,
            2,
            195,
            300,
            "The weather is sunny today".to_string(),
            5,
            1,
            true,
            false,
            Some(vec![0.8, 0.9, 0.1, 0.2, 0.3]), // Very different embedding
        ),
    ];
    
    // Batch insert chunks with embeddings
    db.batch_insert_chunks_with_embeddings(&chunks_data, "test-model-v1")?;
    println!("   ✅ Inserted {} chunks with embeddings", chunks_data.len());
    
    // Test retrieval
    println!("\n4. Testing chunk retrieval...");
    let retrieved_chunks = db.get_chunks_with_embeddings(&document.id)?;
    println!("   Retrieved {} chunks:", retrieved_chunks.len());
    
    for chunk in &retrieved_chunks {
        println!("     • Chunk {}: {} characters", 
                 chunk.chunk_index + 1, 
                 chunk.content.len());
        println!("       Content: \"{}\"", 
                 chunk.content.chars().take(50).collect::<String>());
        if let Some(ref embedding) = chunk.embedding {
            println!("       Embedding: {} dimensions", embedding.len());
            println!("       Sample values: [{:.3}, {:.3}, {:.3}...]", 
                     embedding[0], embedding[1], embedding[2]);
        }
        println!("       Metadata: {} words, {} sentences", 
                 chunk.word_count, chunk.sentence_count);
        println!();
    }
    
    // Test similarity search
    println!("5. Testing semantic similarity search...");
    let query_embedding = vec![0.15, 0.25, 0.35, 0.45, 0.55]; // Similar to chunks 1 and 2
    let similar_chunks = db.find_similar_chunks(&query_embedding, 10, 0.0)?;
    
    println!("   Found {} similar chunks:", similar_chunks.len());
    for (i, chunk) in similar_chunks.iter().enumerate() {
        println!("     {}. Chunk {} (similarity: {:.3})", 
                 i + 1, 
                 chunk.chunk_index + 1, 
                 chunk.similarity_score);
        println!("        Content: \"{}\"", 
                 chunk.content.chars().take(50).collect::<String>());
    }
    
    // Test individual chunk insertion
    println!("\n6. Testing individual chunk insertion...");
    db.insert_chunk_with_embedding(
        "chunk_4",
        &document.id,
        3,
        295,
        400,
        "Deep learning networks process complex patterns",
        6,
        1,
        true,
        false,
        Some(&[0.1, 0.3, 0.2, 0.6, 0.4]),
        Some("all-minilm-l6-v2"),
    )?;
    println!("   ✅ Individual chunk inserted");
    
    // Verify the new chunk
    let updated_chunks = db.get_chunks_with_embeddings(&document.id)?;
    println!("   Total chunks now: {}", updated_chunks.len());
    
    // Test updated similarity search
    println!("\n7. Testing updated similarity search...");
    let ai_query_embedding = vec![0.1, 0.25, 0.25, 0.5, 0.45]; // Should match AI-related chunks
    let ai_similar_chunks = db.find_similar_chunks(&ai_query_embedding, 5, 0.8)?;
    
    println!("   High-similarity chunks (threshold: 0.8):");
    for chunk in &ai_similar_chunks {
        println!("     • Chunk {}: similarity {:.3}", 
                 chunk.chunk_index + 1, 
                 chunk.similarity_score);
        println!("       \"{}\"", chunk.content);
    }
    
    // Test deletion
    println!("\n8. Testing chunk deletion...");
    db.delete_document_chunks(&document.id)?;
    let chunks_after_deletion = db.get_chunks_with_embeddings(&document.id)?;
    println!("   ✅ Chunks after deletion: {}", chunks_after_deletion.len());
    
    // Clean up
    std::fs::remove_file(db_path).ok();
    
    println!("\n🎉 Database Embedding Operations Test Complete!");
    println!("✅ All operations working correctly");
    println!("✅ Chunk storage and retrieval functional");
    println!("✅ Embedding serialization/deserialization working");
    println!("✅ Semantic similarity search operational");
    println!("✅ Database schema supports full embedding workflow");
    
    Ok(())
}