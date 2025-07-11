use std::fs;
use std::path::{Path, PathBuf};
use std::env;

#[tauri::command]
pub async fn get_file_content(path: String) -> Result<String, String> {
    fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file: {}", e))
}

#[tauri::command]
pub async fn open_file_in_default_app(file_path: String) -> Result<String, String> {
    println!("🔍 open_file_in_default_app called with: {}", file_path);
    println!("🔍 Current working directory: {:?}", env::current_dir());
    
    // Convert to absolute path if relative
    let path = if file_path.starts_with("./") || file_path.starts_with("../") || !Path::new(&file_path).is_absolute() {
        // Try to resolve relative to the app data directory or current directory
        let mut absolute_path = None;
        
        // First, try relative to the app's data directory
        if let Ok(app_dir) = env::var("APPDIR") {
            let candidate = Path::new(&app_dir).join(&file_path);
            if candidate.exists() {
                absolute_path = Some(candidate);
            }
        }
        
        // If not found, try relative to current directory
        if absolute_path.is_none() {
            if let Ok(current_dir) = env::current_dir() {
                println!("🔍 Current directory: {:?}", current_dir);
                let candidate = current_dir.join(&file_path);
                println!("🔍 Trying path: {:?}, exists: {}", candidate, candidate.exists());
                if candidate.exists() {
                    absolute_path = Some(candidate);
                }
            }
        }
        
        // If still not found, try cleaning the path and checking again
        if absolute_path.is_none() {
            let cleaned_path = file_path.trim_start_matches("./");
            if let Ok(current_dir) = env::current_dir() {
                let candidate = current_dir.join(cleaned_path);
                if candidate.exists() {
                    absolute_path = Some(candidate);
                }
            }
        }
        
        match absolute_path {
            Some(p) => p,
            None => return Err(format!("File not found: {} (tried multiple locations)", file_path))
        }
    } else {
        PathBuf::from(&file_path)
    };
    
    // Check if file exists
    if !path.exists() {
        return Err(format!("File not found: {:?}", path));
    }
    
    let file_path_str = path.to_string_lossy().to_string();
    
    // Open file with default system application
    #[cfg(target_os = "macos")]
    {
        println!("🔍 Attempting to open file with macOS 'open' command");
        println!("🔍 File path: {}", file_path_str);
        
        let output = std::process::Command::new("open")
            .arg(&file_path_str)
            .output()
            .map_err(|e| {
                println!("❌ Failed to execute 'open' command: {}", e);
                format!("Failed to execute 'open' command: {}", e)
            })?;
            
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("❌ 'open' command failed: {}", stderr);
            return Err(format!("'open' command failed: {}", stderr));
        }
        
        println!("✅ Successfully opened file");
    }
    
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", &file_path_str])
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }
    
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&file_path_str)
            .spawn()
            .map_err(|e| format!("Failed to open file: {}", e))?;
    }
    
    Ok(format!("Opened file: {}", file_path_str))
}

#[tauri::command]
pub async fn show_file_in_folder(file_path: String) -> Result<String, String> {
    // Convert to absolute path if relative
    let path = if file_path.starts_with("./") || file_path.starts_with("../") || !Path::new(&file_path).is_absolute() {
        // Try to resolve relative to the app data directory or current directory
        let mut absolute_path = None;
        
        // First, try relative to the app's data directory
        if let Ok(app_dir) = env::var("APPDIR") {
            let candidate = Path::new(&app_dir).join(&file_path);
            if candidate.exists() {
                absolute_path = Some(candidate);
            }
        }
        
        // If not found, try relative to current directory
        if absolute_path.is_none() {
            if let Ok(current_dir) = env::current_dir() {
                println!("🔍 Current directory: {:?}", current_dir);
                let candidate = current_dir.join(&file_path);
                println!("🔍 Trying path: {:?}, exists: {}", candidate, candidate.exists());
                if candidate.exists() {
                    absolute_path = Some(candidate);
                }
            }
        }
        
        // If still not found, try cleaning the path and checking again
        if absolute_path.is_none() {
            let cleaned_path = file_path.trim_start_matches("./");
            if let Ok(current_dir) = env::current_dir() {
                let candidate = current_dir.join(cleaned_path);
                if candidate.exists() {
                    absolute_path = Some(candidate);
                }
            }
        }
        
        match absolute_path {
            Some(p) => p,
            None => return Err(format!("File not found: {} (tried multiple locations)", file_path))
        }
    } else {
        PathBuf::from(&file_path)
    };
    
    // Check if file exists
    if !path.exists() {
        return Err(format!("File not found: {:?}", path));
    }
    
    let file_path_str = path.to_string_lossy().to_string();
    
    // Show file in system file manager
    #[cfg(target_os = "macos")]
    {
        println!("🔍 Attempting to show file in Finder with 'open -R' command");
        println!("🔍 File path: {}", file_path_str);
        
        let output = std::process::Command::new("open")
            .args(["-R", &file_path_str])
            .output()
            .map_err(|e| {
                println!("❌ Failed to execute 'open -R' command: {}", e);
                format!("Failed to execute 'open -R' command: {}", e)
            })?;
            
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("❌ 'open -R' command failed: {}", stderr);
            return Err(format!("'open -R' command failed: {}", stderr));
        }
        
        println!("✅ Successfully showed file in Finder");
    }
    
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .args(["/select,", &file_path_str])
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
    
    Ok(format!("Showed file in folder: {}", file_path_str))
}