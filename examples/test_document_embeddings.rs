use desktop_ai_search::core::document_processor::{DocumentProcessor, ProcessingOptions};
use desktop_ai_search::models::{Document, FileType};
use std::collections::HashMap;
use uuid::Uuid;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing Document Processing with Embeddings");
    println!("============================================");
    
    // Create a document processor
    println!("\n1. Creating Document Processor...");
    let options = ProcessingOptions::default();
    let mut processor = DocumentProcessor::new(options);
    
    // Set up embedding generation
    println!("2. Setting up embedding model...");
    processor.set_embedding_manager("all-minilm-l6-v2").await?;
    
    if processor.has_embedding_manager() {
        println!("   ✅ Embedding manager initialized");
    } else {
        println!("   ❌ Failed to initialize embedding manager");
        return Ok(());
    }
    
    // Create a test document
    println!("\n3. Creating test document...");
    let test_document = Document {
        id: Uuid::new_v4(),
        file_path: "test_content.txt".to_string(),
        file_type: FileType::Text,
        file_size: 0,
        content_hash: "test-hash".to_string(),
        creation_date: chrono::Utc::now(),
        modification_date: chrono::Utc::now(),
        last_indexed: chrono::Utc::now(),
        metadata: HashMap::new(),
    };
    
    // Create test content file
    let test_content = r#"
    Artificial intelligence (AI) is revolutionizing how we interact with technology. 
    Machine learning algorithms can analyze vast amounts of data to identify patterns and make predictions.
    
    Natural language processing enables computers to understand and generate human language.
    Computer vision allows machines to interpret and understand visual information from the world.
    
    Deep learning networks, inspired by the human brain, can solve complex problems in image recognition, 
    speech processing, and game playing. These technologies are transforming industries from healthcare 
    to transportation, finance to entertainment.
    
    The future of AI holds immense potential for solving humanity's greatest challenges, from climate 
    change to disease detection and treatment. However, it also raises important questions about 
    ethics, privacy, and the future of work.
    "#;
    
    // Write test content to temporary file
    std::fs::write("test_content.txt", test_content)?;
    
    // Process the document
    println!("4. Processing document with embedding generation...");
    let start_time = std::time::Instant::now();
    
    let result = processor.process_document(&test_document, "test_content.txt").await?;
    
    let processing_time = start_time.elapsed();
    
    println!("   ✅ Document processed successfully");
    println!("   📊 Processing Statistics:");
    println!("     • Processing time: {:.2}s", processing_time.as_secs_f32());
    println!("     • Chunks created: {}", result.chunks.len());
    println!("     • Content length: {} characters", result.extraction_result.content.len());
    println!("     • Requires OCR: {}", result.requires_ocr);
    println!("     • Requires transcription: {}", result.requires_transcription);
    
    // Analyze the chunks and their embeddings
    println!("\n5. Analyzing chunks and embeddings...");
    let mut chunks_with_embeddings = 0;
    let mut total_embedding_dimensions = 0;
    
    for (i, chunk) in result.chunks.iter().enumerate() {
        println!("   Chunk {}: {} characters", i + 1, chunk.content.len());
        println!("     Content preview: \"{}...\"", 
                 chunk.content.chars().take(60).collect::<String>());
        
        if let Some(ref embedding) = chunk.embedding {
            chunks_with_embeddings += 1;
            total_embedding_dimensions = embedding.len();
            println!("     ✅ Embedding: {} dimensions", embedding.len());
            
            // Show first few embedding values
            let preview: Vec<String> = embedding.iter()
                .take(5)
                .map(|x| format!("{:.4}", x))
                .collect();
            println!("     Embedding preview: [{}...]", preview.join(", "));
        } else {
            println!("     ❌ No embedding generated");
        }
        println!();
    }
    
    println!("📈 Embedding Summary:");
    println!("   • Chunks with embeddings: {}/{}", chunks_with_embeddings, result.chunks.len());
    println!("   • Embedding dimensions: {}", total_embedding_dimensions);
    
    // Test semantic similarity between chunks
    if chunks_with_embeddings >= 2 {
        println!("\n6. Testing semantic similarity between chunks...");
        use desktop_ai_search::core::embedding_manager::EmbeddingManager;
        
        for i in 0..(result.chunks.len().min(3)) {
            for j in (i+1)..(result.chunks.len().min(3)) {
                if let (Some(ref emb1), Some(ref emb2)) = 
                    (&result.chunks[i].embedding, &result.chunks[j].embedding) {
                    
                    let similarity = EmbeddingManager::cosine_similarity(emb1, emb2);
                    println!("   Chunk {} vs Chunk {}: similarity = {:.3}", 
                             i + 1, j + 1, similarity);
                }
            }
        }
    }
    
    // Clean up test file
    std::fs::remove_file("test_content.txt").ok();
    
    println!("\n🎉 Document Processing with Embeddings Test Complete!");
    
    if chunks_with_embeddings == result.chunks.len() && chunks_with_embeddings > 0 {
        println!("✅ All chunks successfully processed with embeddings");
        println!("✅ Semantic search capabilities are now operational");
    } else {
        println!("⚠️  Some chunks missing embeddings");
    }
    
    Ok(())
}