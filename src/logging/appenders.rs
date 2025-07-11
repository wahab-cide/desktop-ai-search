use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Mutex;

use chrono::{DateTime, Utc};

use crate::error::Result;
use super::{LogAppender, LogEntry, LogFormatter};
use super::formatters::{JsonFormatter, TextFormatter};

pub struct ConsoleAppender {
    formatter: Box<dyn LogFormatter>,
}

pub struct FileAppender {
    file: Mutex<BufWriter<File>>,
    path: PathBuf,
    max_size: u64,
    max_files: u32,
    current_size: Mutex<u64>,
    formatter: Box<dyn LogFormatter>,
}

pub struct RotatingFileAppender {
    base_path: PathBuf,
    max_size: u64,
    max_files: u32,
    current_file: Mutex<Option<BufWriter<File>>>,
    current_size: Mutex<u64>,
    formatter: Box<dyn LogFormatter>,
}

pub struct StructuredFileAppender {
    file: Mutex<BufWriter<File>>,
    path: PathBuf,
    max_size: u64,
    max_files: u32,
    current_size: Mutex<u64>,
}

impl ConsoleAppender {
    pub fn new() -> Self {
        Self {
            formatter: Box::new(TextFormatter::new()),
        }
    }

    pub fn with_formatter(formatter: Box<dyn LogFormatter>) -> Self {
        Self { formatter }
    }
}

impl LogAppender for ConsoleAppender {
    fn append(&self, entry: &LogEntry) -> Result<()> {
        let formatted = self.formatter.format(entry);
        
        match entry.level {
            super::LogLevel::Error => eprintln!("{}", formatted),
            super::LogLevel::Warn => eprintln!("{}", formatted),
            _ => println!("{}", formatted),
        }
        
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        use std::io::{stdout, stderr};
        stdout().flush()?;
        stderr().flush()?;
        Ok(())
    }
}

impl FileAppender {
    pub fn new(path: PathBuf, max_size: u64, max_files: u32) -> Result<Self> {
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        let current_size = file.metadata()?.len();
        let buf_writer = BufWriter::new(file);

        Ok(Self {
            file: Mutex::new(buf_writer),
            path,
            max_size,
            max_files,
            current_size: Mutex::new(current_size),
            formatter: Box::new(JsonFormatter::new()),
        })
    }

    pub fn with_formatter(mut self, formatter: Box<dyn LogFormatter>) -> Self {
        self.formatter = formatter;
        self
    }

    fn rotate_if_needed(&self) -> Result<()> {
        let current_size = *self.current_size.lock().unwrap();
        
        if current_size >= self.max_size {
            self.rotate_files()?;
        }
        
        Ok(())
    }

    fn rotate_files(&self) -> Result<()> {
        // Close current file
        {
            let mut file = self.file.lock().unwrap();
            file.flush()?;
        }

        // Rotate existing files
        for i in (1..self.max_files).rev() {
            let old_path = if i == 1 {
                self.path.clone()
            } else {
                self.path.with_extension(format!("{}.{}", 
                    self.path.extension().unwrap_or_default().to_string_lossy(),
                    i - 1
                ))
            };

            let new_path = self.path.with_extension(format!("{}.{}", 
                self.path.extension().unwrap_or_default().to_string_lossy(),
                i
            ));

            if old_path.exists() {
                std::fs::rename(&old_path, &new_path)?;
            }
        }

        // Create new file
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;

        *self.file.lock().unwrap() = BufWriter::new(file);
        *self.current_size.lock().unwrap() = 0;

        Ok(())
    }
}

impl LogAppender for FileAppender {
    fn append(&self, entry: &LogEntry) -> Result<()> {
        self.rotate_if_needed()?;
        
        let formatted = self.formatter.format(entry);
        let line = format!("{}\n", formatted);
        
        {
            let mut file = self.file.lock().unwrap();
            file.write_all(line.as_bytes())?;
            file.flush()?;
        }

        {
            let mut current_size = self.current_size.lock().unwrap();
            *current_size += line.len() as u64;
        }

        Ok(())
    }

    fn flush(&self) -> Result<()> {
        let mut file = self.file.lock().unwrap();
        file.flush()?;
        Ok(())
    }
}

impl RotatingFileAppender {
    pub fn new(base_path: PathBuf, max_size: u64, max_files: u32) -> Result<Self> {
        // Create parent directories if they don't exist
        if let Some(parent) = base_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        Ok(Self {
            base_path,
            max_size,
            max_files,
            current_file: Mutex::new(None),
            current_size: Mutex::new(0),
            formatter: Box::new(JsonFormatter::new()),
        })
    }

    pub fn with_formatter(mut self, formatter: Box<dyn LogFormatter>) -> Self {
        self.formatter = formatter;
        self
    }

    fn get_current_file_path(&self) -> PathBuf {
        let now = Utc::now();
        let timestamp = now.format("%Y%m%d_%H%M%S").to_string();
        
        let mut path = self.base_path.clone();
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let ext = path.extension().unwrap_or_default().to_string_lossy();
        
        path.set_file_name(format!("{}_{}.{}", stem, timestamp, ext));
        path
    }

    fn ensure_current_file(&self) -> Result<()> {
        let mut current_file = self.current_file.lock().unwrap();
        let mut current_size = self.current_size.lock().unwrap();
        
        if current_file.is_none() || *current_size >= self.max_size {
            // Close current file if it exists
            if current_file.is_some() {
                current_file.as_mut().unwrap().flush()?;
            }

            // Create new file
            let file_path = self.get_current_file_path();
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&file_path)?;

            *current_file = Some(BufWriter::new(file));
            *current_size = 0;

            // Clean up old files
            self.cleanup_old_files()?;
        }

        Ok(())
    }

    fn cleanup_old_files(&self) -> Result<()> {
        let parent = self.base_path.parent().unwrap();
        let stem = self.base_path.file_stem().unwrap_or_default().to_string_lossy();
        let ext = self.base_path.extension().unwrap_or_default().to_string_lossy();
        
        let mut files = Vec::new();
        
        // Find all log files matching the pattern
        for entry in std::fs::read_dir(parent)? {
            let entry = entry?;
            let path = entry.path();
            
            if let Some(file_name) = path.file_name() {
                let file_name_str = file_name.to_string_lossy();
                if file_name_str.starts_with(&*stem) && file_name_str.ends_with(&*ext) {
                    if let Ok(metadata) = entry.metadata() {
                        files.push((path, metadata.modified().unwrap_or(std::time::UNIX_EPOCH)));
                    }
                }
            }
        }
        
        // Sort by modification time (newest first)
        files.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Remove excess files
        for (path, _) in files.iter().skip(self.max_files as usize) {
            std::fs::remove_file(path)?;
        }
        
        Ok(())
    }
}

impl LogAppender for RotatingFileAppender {
    fn append(&self, entry: &LogEntry) -> Result<()> {
        self.ensure_current_file()?;
        
        let formatted = self.formatter.format(entry);
        let line = format!("{}\n", formatted);
        
        {
            let mut current_file = self.current_file.lock().unwrap();
            let mut current_size = self.current_size.lock().unwrap();
            
            if let Some(ref mut file) = current_file.as_mut() {
                file.write_all(line.as_bytes())?;
                file.flush()?;
                *current_size += line.len() as u64;
            }
        }

        Ok(())
    }

    fn flush(&self) -> Result<()> {
        let mut current_file = self.current_file.lock().unwrap();
        if let Some(ref mut file) = current_file.as_mut() {
            file.flush()?;
        }
        Ok(())
    }
}

impl StructuredFileAppender {
    pub fn new(path: PathBuf, max_size: u64, max_files: u32) -> Result<Self> {
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        let current_size = file.metadata()?.len();
        let buf_writer = BufWriter::new(file);

        Ok(Self {
            file: Mutex::new(buf_writer),
            path,
            max_size,
            max_files,
            current_size: Mutex::new(current_size),
        })
    }

    fn rotate_if_needed(&self) -> Result<()> {
        let current_size = *self.current_size.lock().unwrap();
        
        if current_size >= self.max_size {
            self.rotate_files()?;
        }
        
        Ok(())
    }

    fn rotate_files(&self) -> Result<()> {
        // Close current file
        {
            let mut file = self.file.lock().unwrap();
            file.flush()?;
        }

        // Rotate existing files
        for i in (1..self.max_files).rev() {
            let old_path = if i == 1 {
                self.path.clone()
            } else {
                self.path.with_extension(format!("{}.{}", 
                    self.path.extension().unwrap_or_default().to_string_lossy(),
                    i - 1
                ))
            };

            let new_path = self.path.with_extension(format!("{}.{}", 
                self.path.extension().unwrap_or_default().to_string_lossy(),
                i
            ));

            if old_path.exists() {
                std::fs::rename(&old_path, &new_path)?;
            }
        }

        // Create new file
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;

        *self.file.lock().unwrap() = BufWriter::new(file);
        *self.current_size.lock().unwrap() = 0;

        Ok(())
    }
}

impl LogAppender for StructuredFileAppender {
    fn append(&self, entry: &LogEntry) -> Result<()> {
        self.rotate_if_needed()?;
        
        // Write structured log entry as JSON
        let json = serde_json::to_string(entry)?;
        let line = format!("{}\n", json);
        
        {
            let mut file = self.file.lock().unwrap();
            file.write_all(line.as_bytes())?;
            file.flush()?;
        }

        {
            let mut current_size = self.current_size.lock().unwrap();
            *current_size += line.len() as u64;
        }

        Ok(())
    }

    fn flush(&self) -> Result<()> {
        let mut file = self.file.lock().unwrap();
        file.flush()?;
        Ok(())
    }
}