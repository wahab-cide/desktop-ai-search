#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod core;
mod models;
mod utils;
mod error;
mod database;
mod cli;

use commands::{search, indexing, files};

fn main() {
    // Check if running as CLI
    if let Some(cli_command) = cli::parse_args() {
        if let Err(e) = cli::execute_cli_command(cli_command) {
            eprintln!("CLI command failed: {}", e);
            std::process::exit(1);
        }
        return;
    }
    
    // For now, just print help if no CLI command
    cli::print_usage();
    
    // TODO: Re-enable Tauri app once all modules are implemented
    // tauri::Builder::default()
    //     .invoke_handler(tauri::generate_handler![
    //         search::search_documents,
    //         indexing::index_file,
    //         indexing::get_indexing_status,
    //         files::get_file_content
    //     ])
    //     .run(tauri::generate_context!())
    //     .expect("error while running tauri application");
}