#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod core;
mod models;
mod utils;
mod error;
mod database;
mod cli;
mod app;
mod config;
mod recovery;
mod cache;
mod monitoring;

use commands::{search, search_v2, indexing, files, ai, embeddings, health, monitoring as monitoring_commands};
use commands::cache as cache_commands;
use config::AppConfig;

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
            search::rebuild_search_index,
            
            // Advanced search commands (v2)
            search_v2::search,
            search_v2::get_search_suggestions_v2,
            search_v2::get_file_type_counts,
            search_v2::browse_files_by_type,
            
            // Indexing commands
            indexing::index_file,
            indexing::index_directory,
            indexing::index_directory_incremental,
            indexing::get_indexing_status,
            indexing::get_indexing_statistics,
            indexing::reset_indexing_state,
            indexing::start_background_indexing,
            indexing::test_indexing_params,
            indexing::test_camel_case,
            indexing::index_directory_simple,
            indexing::cleanup_missing_files,
            indexing::reset_database,
            
            // File commands
            files::get_file_content,
            files::open_file_in_default_app,
            files::show_file_in_folder,
            
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
            
            // Health and recovery commands
            health::get_health_status,
            health::get_system_metrics,
            health::trigger_recovery,
            health::reset_circuit_breaker,
            health::get_recovery_suggestions,
            
            // Cache management commands
            cache_commands::get_cache_status,
            cache_commands::clear_all_caches,
            cache_commands::clear_cache_type,
            cache_commands::optimize_caches,
            cache_commands::get_search_cache_stats,
            cache_commands::get_embedding_cache_stats,
            cache_commands::get_model_memory_usage,
            cache_commands::invalidate_search_cache_for_file,
            cache_commands::invalidate_embedding_cache_for_model,
            cache_commands::get_cache_performance_metrics,
            cache_commands::update_cache_config,
            
            // Monitoring commands
            monitoring_commands::get_system_health,
            monitoring_commands::get_performance_metrics,
            monitoring_commands::get_performance_summary,
            monitoring_commands::record_metric,
            monitoring_commands::get_metrics_range,
            monitoring_commands::get_active_alerts,
            monitoring_commands::create_alert,
            monitoring_commands::acknowledge_alert,
            monitoring_commands::resolve_alert,
            monitoring_commands::get_alert_stats,
            monitoring_commands::add_alert_rule,
            monitoring_commands::remove_alert_rule,
            monitoring_commands::get_alert_rules,
            monitoring_commands::record_performance_event,
            monitoring_commands::export_telemetry,
            monitoring_commands::get_telemetry_stats,
            monitoring_commands::update_telemetry_privacy_settings,
            monitoring_commands::get_telemetry_privacy_settings,
            monitoring_commands::clear_telemetry_data,
            monitoring_commands::export_metrics,
            monitoring_commands::get_aggregated_metric,
            monitoring_commands::get_metric_names,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}