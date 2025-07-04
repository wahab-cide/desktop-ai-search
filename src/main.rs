#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod core;
mod models;
mod utils;

use commands::{search, indexing, files};

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            search::search_documents,
            indexing::index_file,
            indexing::get_indexing_status,
            files::get_file_content
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}