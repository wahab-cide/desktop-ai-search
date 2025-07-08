use std::fs;
use std::path::Path;

#[tauri::command]
pub async fn get_file_content(path: String) -> Result<String, String> {
    fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file: {}", e))
}

#[tauri::command]
pub async fn open_file_in_default_app(file_path: String) -> Result<String, String> {
    let path = Path::new(&file_path);
    
    // Check if file exists
    if !path.exists() {
        return Err(format!("File not found: {}", file_path));
    }
    
    // Open file with default system application
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&file_path)
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }
    
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", &file_path])
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }
    
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&file_path)
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }
    
    Ok(format!("Opened file: {}", file_path))
}

#[tauri::command]
pub async fn show_file_in_folder(file_path: String) -> Result<String, String> {
    let path = Path::new(&file_path);
    
    // Check if file exists
    if !path.exists() {
        return Err(format!("File not found: {}", file_path));
    }
    
    // Show file in system file manager
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .args(["-R", &file_path])
            .spawn()
            .map_err(|e| format!("Failed to show file in Finder: {}", e))?;
    }
    
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .args(["/select,", &file_path])
            .spawn()
            .map_err(|e| format!("Failed to show file in Explorer: {}", e))?;
    }
    
    #[cfg(target_os = "linux")]
    {
        // Try to use the file manager, fallback to opening the parent directory
        if let Some(parent) = path.parent() {
            std::process::Command::new("xdg-open")
                .arg(parent)
                .spawn()
                .map_err(|e| format!("Failed to show file in file manager: {}", e))?;
        } else {
            return Err("Could not determine parent directory".to_string());
        }
    }
    
    Ok(format!("Showed file in folder: {}", file_path))
}