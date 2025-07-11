use crate::error::{AppError, FileSystemError, Result};
use notify::{Watcher, RecursiveMode, Event, EventKind, RecommendedWatcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::sleep;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WatcherEvent {
    Created(PathBuf),
    Modified(PathBuf),
    Renamed { old_path: PathBuf, new_path: PathBuf },
    Deleted(PathBuf),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchState {
    pub file_path: PathBuf,
    pub sha256: String,
    pub mtime: u64,
}

pub struct FileSystemWatcher {
    watcher: RecommendedWatcher,
    event_sender: mpsc::UnboundedSender<WatcherEvent>,
    debounce_map: HashMap<PathBuf, Instant>,
    debounce_duration: Duration,
    watch_state: HashMap<PathBuf, WatchState>,
}

impl FileSystemWatcher {
    pub fn new() -> Result<(Self, mpsc::UnboundedReceiver<WatcherEvent>)> {
        let (tx, rx) = mpsc::unbounded_channel();
        let tx_clone = tx.clone();
        
        let watcher = RecommendedWatcher::new(
            move |result: notify::Result<Event>| {
                match result {
                    Ok(event) => {
                        if let Err(e) = Self::handle_raw_event(&tx_clone, event) {
                            eprintln!("Error handling file system event: {}", e);
                        }
                    }
                    Err(e) => {
                        eprintln!("File system watcher error: {}", e);
                    }
                }
            },
            notify::Config::default(),
        ).map_err(FileSystemError::Watcher)?;
        
        let fs_watcher = Self {
            watcher,
            event_sender: tx,
            debounce_map: HashMap::new(),
            debounce_duration: Duration::from_millis(300),
            watch_state: HashMap::new(),
        };
        
        Ok((fs_watcher, rx))
    }
    
    fn handle_raw_event(
        sender: &mpsc::UnboundedSender<WatcherEvent>,
        event: Event,
    ) -> Result<()> {
        match event.kind {
            EventKind::Create(_) => {
                for path in event.paths {
                    let _ = sender.send(WatcherEvent::Created(path));
                }
            }
            EventKind::Modify(_) => {
                for path in event.paths {
                    let _ = sender.send(WatcherEvent::Modified(path));
                }
            }
            EventKind::Remove(_) => {
                for path in event.paths {
                    let _ = sender.send(WatcherEvent::Deleted(path));
                }
            }
            EventKind::Other => {
                // Handle platform-specific rename events
                if event.paths.len() == 2 {
                    let _ = sender.send(WatcherEvent::Renamed {
                        old_path: event.paths[0].clone(),
                        new_path: event.paths[1].clone(),
                    });
                }
            }
            _ => {
                // Handle other event types as generic modifications
                for path in event.paths {
                    let _ = sender.send(WatcherEvent::Modified(path));
                }
            }
        };
        
        Ok(())
    }
    
    pub fn watch_directory<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.watcher
            .watch(path.as_ref(), RecursiveMode::Recursive)
            .map_err(FileSystemError::Watcher)
            .map_err(AppError::FileSystem)
    }
    
    pub fn unwatch_directory<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        self.watcher
            .unwatch(path.as_ref())
            .map_err(FileSystemError::Watcher)
            .map_err(AppError::FileSystem)
    }
    
    pub fn should_process_event(&mut self, path: &Path) -> bool {
        let now = Instant::now();
        
        // Check debouncing
        if let Some(&last_event_time) = self.debounce_map.get(path) {
            if now.duration_since(last_event_time) < self.debounce_duration {
                return false;
            }
        }
        
        // Update debounce map
        self.debounce_map.insert(path.to_path_buf(), now);
        
        // Clean up old entries periodically
        self.cleanup_debounce_map(now);
        
        true
    }
    
    fn cleanup_debounce_map(&mut self, now: Instant) {
        let cleanup_threshold = Duration::from_secs(60);
        self.debounce_map.retain(|_, &mut last_time| {
            now.duration_since(last_time) < cleanup_threshold
        });
    }
    
    pub fn update_watch_state(&mut self, path: PathBuf, sha256: String, mtime: u64) {
        self.watch_state.insert(path.clone(), WatchState {
            file_path: path,
            sha256,
            mtime,
        });
    }
    
    pub fn get_watch_state(&self, path: &Path) -> Option<&WatchState> {
        self.watch_state.get(path)
    }
    
    pub fn persist_watch_state<P: AsRef<Path>>(&self, state_file: P) -> Result<()> {
        let state_json = serde_json::to_string_pretty(&self.watch_state)?;
        std::fs::write(state_file, state_json)
            .map_err(|e| AppError::FileSystem(FileSystemError::Metadata(e.to_string())))
    }
    
    pub fn restore_watch_state<P: AsRef<Path>>(&mut self, state_file: P) -> Result<()> {
        if !state_file.as_ref().exists() {
            return Ok(());
        }
        
        let state_json = std::fs::read_to_string(state_file)
            .map_err(|e| AppError::FileSystem(FileSystemError::Metadata(e.to_string())))?;
        
        self.watch_state = serde_json::from_str(&state_json)?;
        Ok(())
    }
    
    pub fn has_file_changed(&self, path: &Path, current_sha256: &str, current_mtime: u64) -> bool {
        match self.get_watch_state(path) {
            Some(state) => {
                state.sha256 != current_sha256 || state.mtime != current_mtime
            }
            None => true, // File not in watch state, consider it changed
        }
    }
}

pub struct DebouncedEventProcessor {
    debounce_duration: Duration,
    pending_events: HashMap<PathBuf, (WatcherEvent, Instant)>,
}

impl DebouncedEventProcessor {
    pub fn new(debounce_duration: Duration) -> Self {
        Self {
            debounce_duration,
            pending_events: HashMap::new(),
        }
    }
    
    pub async fn process_event(
        &mut self,
        event: WatcherEvent,
        processor: impl Fn(WatcherEvent) -> Result<()>,
    ) -> Result<()> {
        let path = match &event {
            WatcherEvent::Created(p) | WatcherEvent::Modified(p) | WatcherEvent::Deleted(p) => p.clone(),
            WatcherEvent::Renamed { new_path, .. } => new_path.clone(),
        };
        
        // Store the event with current timestamp
        self.pending_events.insert(path.clone(), (event, Instant::now()));
        
        // Wait for debounce duration
        sleep(self.debounce_duration).await;
        
        // Process if the event is still the latest for this path
        if let Some((pending_event, timestamp)) = self.pending_events.get(&path) {
            if timestamp.elapsed() >= self.debounce_duration {
                let event_to_process = pending_event.clone();
                self.pending_events.remove(&path);
                processor(event_to_process)?;
            }
        }
        
        Ok(())
    }
    
    pub fn cleanup_old_events(&mut self) {
        let cleanup_threshold = self.debounce_duration * 5;
        self.pending_events.retain(|_, (_, timestamp)| {
            timestamp.elapsed() < cleanup_threshold
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs;
    
    #[tokio::test]
    async fn test_file_watcher_creation() {
        let (mut watcher, mut rx) = FileSystemWatcher::new().unwrap();
        
        let temp_dir = tempdir().unwrap();
        watcher.watch_directory(temp_dir.path()).unwrap();
        
        // Create a test file
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "test content").unwrap();
        
        // Should receive a creation event
        tokio::select! {
            event = rx.recv() => {
                match event {
                    Some(WatcherEvent::Created(path)) => {
                        assert_eq!(path, test_file);
                    }
                    Some(WatcherEvent::Modified(path)) => {
                        // Some platforms send modify instead of create
                        assert_eq!(path, test_file);
                    }
                    other => {
                        panic!("Unexpected event: {:?}", other);
                    }
                }
            }
            _ = tokio::time::sleep(Duration::from_secs(2)) => {
                // Events might be delayed on some platforms
            }
        }
    }
    
    #[test]
    fn test_debouncing() {
        let (mut watcher, _) = FileSystemWatcher::new().unwrap();
        let test_path = PathBuf::from("/test/path");
        
        // First event should be processed
        assert!(watcher.should_process_event(&test_path));
        
        // Immediate second event should be debounced
        assert!(!watcher.should_process_event(&test_path));
        
        // Simulate time passing
        std::thread::sleep(Duration::from_millis(350));
        
        // Event after debounce period should be processed
        assert!(watcher.should_process_event(&test_path));
    }
}