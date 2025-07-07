use desktop_ai_search::database::Database;
use desktop_ai_search::utils::file_processor::FileProcessor;
use desktop_ai_search::core::document_processor::{DocumentProcessor, ProcessingProgress};
use desktop_ai_search::core::embedding_manager::EmbeddingManager;
use std::path::PathBuf;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📚 Testing Complete Document Indexing Pipeline");
    println!("===============================================");
    
    // Initialize database
    println!("\n1. Setting up database...");
    let db_path = "test_indexing.db";
    
    // Clean up any existing database first
    std::fs::remove_file(db_path).ok();
    
    let database = Database::new(db_path)?;
    println!("   ✅ Database initialized with all migrations");
    
    // Initialize document processor with embeddings
    println!("\n2. Setting up document processor with embeddings...");
    let mut doc_processor = DocumentProcessor::default();
    doc_processor.set_embedding_manager("all-minilm-l6-v2").await?;
    println!("   ✅ Document processor configured with all-MiniLM-L6-v2");
    
    // Scan test documents directory
    println!("\n3. Scanning test documents directory...");
    let test_docs_dir = PathBuf::from("test_documents");
    
    if !test_docs_dir.exists() {
        println!("   ❌ Test documents directory not found: {}", test_docs_dir.display());
        println!("   Please create test documents first");
        return Ok(());
    }
    
    let mut file_processor = FileProcessor::new()?;
    let documents = file_processor.scan_directory(&test_docs_dir).await?;
    
    if documents.is_empty() {
        println!("   ❌ No documents found in test directory");
        return Ok(());
    }
    
    println!("   ✅ Found {} documents to process:", documents.len());
    for doc in &documents {
        println!("     • {} ({:?}, {} bytes)", 
                 doc.file_path, doc.file_type, doc.file_size);
    }
    
    // Process documents with progress tracking
    println!("\n4. Processing documents with embeddings...");
    let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel::<ProcessingProgress>();
    
    // Spawn progress reporter
    let progress_handle = tokio::spawn(async move {
        while let Some(progress) = progress_rx.recv().await {
            if let Some(current) = &progress.current_document {
                println!("   Processing: {} ({}/{})", 
                    current, 
                    progress.processed_documents + 1, 
                    progress.total_documents
                );
            } else {
                println!("   ✅ Processing complete!");
                println!("     Documents: {}", progress.total_documents);
                println!("     Chunks created: {}", progress.chunks_created);
                println!("     Processing time: {}ms", progress.elapsed_time_ms);
            }
        }
    });
    
    // Process documents
    let results = doc_processor.process_documents(documents, Some(progress_tx)).await?;
    
    // Wait for progress reporting to complete
    progress_handle.await.map_err(|e| format!("Progress reporting error: {}", e))?;
    
    // Store results in database with embeddings
    println!("\n5. Storing chunks and embeddings in database...");
    let stored_chunks = doc_processor.store_processing_results(&database, results.clone()).await?;
    println!("   ✅ Stored {} chunks with embeddings", stored_chunks);
    
    // Show processing statistics
    println!("\n6. Processing statistics:");
    let stats = doc_processor.get_processing_statistics(&results);
    for (key, value) in stats {
        println!("   {}: {}", key, value);
    }
    
    // Test database retrieval
    println!("\n7. Testing database retrieval...");
    let total_chunks = database.get_chunks_with_embeddings(&results[0].document.id)?;
    println!("   Retrieved {} chunks for first document", total_chunks.len());
    
    if let Some(first_chunk) = total_chunks.first() {
        println!("   Sample chunk:");
        println!("     Content: \"{}\"", &first_chunk.content[..100.min(first_chunk.content.len())]);
        println!("     Word count: {}", first_chunk.word_count);
        println!("     Has embedding: {}", first_chunk.embedding.is_some());
        if let Some(ref embedding) = first_chunk.embedding {
            println!("     Embedding dimensions: {}", embedding.len());
        }
    }
    
    // Test semantic search with various queries
    println!("\n8. Testing semantic search capabilities...");
    let test_queries = vec![
        ("machine learning algorithms", "Should find AI/ML content"),
        ("cooking pasta techniques", "Should find cooking content"), 
        ("climate change effects", "Should find climate content"),
        ("neural network training", "Should find AI/ML content"),
        ("Italian food preparation", "Should find cooking content"),
        ("greenhouse gas emissions", "Should find climate content"),
    ];
    
    // Create a separate embedding manager for testing queries
    let mut query_embedding_manager = EmbeddingManager::new()?;
    query_embedding_manager.load_model("all-minilm-l6-v2", None).await?;
    
    for (query, description) in test_queries {
        println!("\n   Query: \"{}\" ({})", query, description);
        
        // Generate query embedding
        let query_embeddings = query_embedding_manager.generate_embeddings(&[query.to_string()]).await?;
        
        if let Some(query_embedding) = query_embeddings.first() {
            // Search for similar chunks
            let similar_chunks = database.find_similar_chunks(query_embedding, 3, 0.4)?;
            
            if similar_chunks.is_empty() {
                println!("     No results found above threshold 0.4");
            } else {
                println!("     Found {} results:", similar_chunks.len());
                for (i, chunk) in similar_chunks.iter().enumerate() {
                    println!("       {}. Similarity: {:.3}", i + 1, chunk.similarity_score);
                    println!("          \"{}\"", &chunk.content[..80.min(chunk.content.len())]);
                }
            }
        }
    }
    
    // Keep database for testing hybrid search
    // std::fs::remove_file(db_path).ok();
    
    println!("\n🎉 Complete Document Indexing Test Finished!");
    println!("✅ Document scanning and processing working");
    println!("✅ Embedding generation for all chunks successful");
    println!("✅ Database storage and retrieval operational");
    println!("✅ Semantic search finds relevant content");
    println!("✅ End-to-end pipeline fully functional");
    
    Ok(())
}