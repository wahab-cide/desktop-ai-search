#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod core;
mod models;
mod utils;
mod error;
mod database;
mod cli;
mod app;
mod test_screenshot;
mod test_query_understanding;

use commands::{search, indexing, files, ai, embeddings};

#[tokio::main]
async fn main() {
    // Check if running as CLI
    if let Some(cli_command) = cli::parse_args() {
        if let Err(e) = cli::execute_cli_command(cli_command).await {
            eprintln!("CLI command failed: {}", e);
            std::process::exit(1);
        }
        return;
    }
    
    // Run as Tauri app
    tauri::Builder::default()
        .setup(|app| {
            app::setup_app(app)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Search commands
            search::search_documents,
            search::search_with_filters,
            search::get_search_suggestions,
            
            // Indexing commands
            indexing::index_file,
            indexing::index_directory, 
            indexing::get_indexing_status,
            indexing::pause_indexing,
            indexing::resume_indexing,
            
            // File commands
            files::get_file_content,
            files::get_file_metadata,
            
            // AI commands
            ai::init_ai_system,
            ai::list_ai_models,
            ai::load_ai_model,
            ai::process_ai_query,
            ai::generate_query_suggestions,
            ai::analyze_search_results,
            ai::get_ai_system_info,
            
            // Embedding commands
            embeddings::list_embedding_models,
            embeddings::download_embedding_model,
            embeddings::load_embedding_model,
            embeddings::generate_embeddings,
            embeddings::calculate_text_similarity,
            embeddings::get_embedding_status,
            embeddings::regenerate_all_embeddings,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}