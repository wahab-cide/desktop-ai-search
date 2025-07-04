use std::fs;

#[tauri::command]
pub async fn get_file_content(path: String) -> Result<String, String> {
    fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file: {}", e))
}