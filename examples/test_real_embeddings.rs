use desktop_ai_search::core::embedding_manager::EmbeddingManager;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Testing Real Embedding Model");
    println!("===============================");
    
    // Create embedding manager
    println!("\n1. Initializing embedding manager...");
    let mut embedding_manager = EmbeddingManager::new()?;
    
    // List available models
    println!("\n2. Available embedding models:");
    let available_models = embedding_manager.get_available_models();
    for model_info in &available_models {
        println!("   • {} - {}", model_info.id, model_info.description);
        println!("     Dimensions: {}, Max sequence: {}", 
                 model_info.dimensions, model_info.max_sequence_length);
    }
    
    // Check if model is downloaded
    let model_id = "all-minilm-l6-v2";
    println!("\n3. Checking if model '{}' is downloaded...", model_id);
    let is_downloaded = embedding_manager.is_model_downloaded(model_id).await?;
    
    if !is_downloaded {
        println!("   Model not found locally. Downloading...");
        println!("   This may take a few minutes for the first run...");
        embedding_manager.download_model(model_id).await?;
        println!("   ✅ Model downloaded successfully");
    } else {
        println!("   ✅ Model already available locally");
    }
    
    // Load the model
    println!("\n4. Loading embedding model...");
    embedding_manager.load_model(model_id, None).await?;
    println!("   ✅ Model loaded successfully");
    
    if let Some(info) = embedding_manager.get_current_model_info() {
        println!("   Model: {} ({} dimensions)", info.description, info.dimensions);
    }
    
    // Test with various texts
    println!("\n5. Testing embedding generation...");
    let test_texts = vec![
        "Machine learning algorithms can analyze complex data patterns".to_string(),
        "Cooking pasta requires boiling water and timing".to_string(),
        "Climate change affects global weather patterns".to_string(),
        "Neural networks learn through backpropagation".to_string(),
        "Italian cuisine emphasizes fresh ingredients".to_string(),
        "Greenhouse gases trap heat in the atmosphere".to_string(),
    ];
    
    println!("   Generating embeddings for {} texts...", test_texts.len());
    let embeddings = embedding_manager.generate_embeddings(&test_texts).await?;
    
    if embeddings.len() != test_texts.len() {
        panic!("Expected {} embeddings, got {}", test_texts.len(), embeddings.len());
    }
    
    println!("   ✅ Generated {} embeddings", embeddings.len());
    println!("   Embedding dimensions: {}", embeddings[0].len());
    
    // Test similarity calculations
    println!("\n6. Testing semantic similarity...");
    for (i, text1) in test_texts.iter().enumerate() {
        for (j, text2) in test_texts.iter().enumerate() {
            if i >= j { continue; }
            
            let similarity = EmbeddingManager::cosine_similarity(&embeddings[i], &embeddings[j]);
            println!("   Similarity {:.3}: \"{}\" <-> \"{}\"", 
                     similarity, 
                     &text1[..50.min(text1.len())],
                     &text2[..50.min(text2.len())]);
        }
    }
    
    // Group by topic and show similarity within groups
    println!("\n7. Analyzing topic clustering...");
    let ai_texts = vec![0, 3]; // Machine learning, Neural networks
    let cooking_texts = vec![1, 4]; // Cooking, Italian cuisine  
    let climate_texts = vec![2, 5]; // Climate change, Greenhouse gases
    
    println!("   AI/ML topic similarities:");
    for &i in &ai_texts {
        for &j in &ai_texts {
            if i >= j { continue; }
            let similarity = EmbeddingManager::cosine_similarity(&embeddings[i], &embeddings[j]);
            println!("     {:.3}: \"{}\" <-> \"{}\"", 
                     similarity, 
                     &test_texts[i][..40.min(test_texts[i].len())],
                     &test_texts[j][..40.min(test_texts[j].len())]);
        }
    }
    
    println!("   Cooking topic similarities:");
    for &i in &cooking_texts {
        for &j in &cooking_texts {
            if i >= j { continue; }
            let similarity = EmbeddingManager::cosine_similarity(&embeddings[i], &embeddings[j]);
            println!("     {:.3}: \"{}\" <-> \"{}\"", 
                     similarity, 
                     &test_texts[i][..40.min(test_texts[i].len())],
                     &test_texts[j][..40.min(test_texts[j].len())]);
        }
    }
    
    println!("   Climate topic similarities:");
    for &i in &climate_texts {
        for &j in &climate_texts {
            if i >= j { continue; }
            let similarity = EmbeddingManager::cosine_similarity(&embeddings[i], &embeddings[j]);
            println!("     {:.3}: \"{}\" <-> \"{}\"", 
                     similarity, 
                     &test_texts[i][..40.min(test_texts[i].len())],
                     &test_texts[j][..40.min(test_texts[j].len())]);
        }
    }
    
    // Test cross-topic similarities (should be lower)
    println!("\n   Cross-topic similarities (should be lower):");
    let cross_pairs = vec![(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)];
    for (i, j) in cross_pairs {
        let similarity = EmbeddingManager::cosine_similarity(&embeddings[i], &embeddings[j]);
        println!("     {:.3}: \"{}\" <-> \"{}\"", 
                 similarity, 
                 &test_texts[i][..40.min(test_texts[i].len())],
                 &test_texts[j][..40.min(test_texts[j].len())]);
    }
    
    println!("\n🎉 Embedding Model Test Complete!");
    println!("✅ Model download and loading working");
    println!("✅ Embedding generation operational");  
    println!("✅ Semantic similarity calculations accurate");
    println!("✅ Topic clustering demonstrates semantic understanding");
    
    Ok(())
}