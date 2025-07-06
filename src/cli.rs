use crate::database::Database;
use crate::error::Result;
use std::path::PathBuf;
use std::env;

pub enum CliCommand {
    InitDb { path: PathBuf },
    Migrate { path: PathBuf },
    Backup { db_path: PathBuf, backup_path: PathBuf },
    Check { path: PathBuf },
    Optimize { path: PathBuf },
    Stats { path: PathBuf },
}

pub fn parse_args() -> Option<CliCommand> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        return None;
    }
    
    match args[1].as_str() {
        "init-db" => {
            let path = args.get(2)
                .map(|p| PathBuf::from(p))
                .unwrap_or_else(|| PathBuf::from("./search.db"));
            Some(CliCommand::InitDb { path })
        }
        "migrate" => {
            let path = args.get(2)
                .map(|p| PathBuf::from(p))
                .unwrap_or_else(|| PathBuf::from("./search.db"));
            Some(CliCommand::Migrate { path })
        }
        "backup" => {
            if args.len() < 4 {
                eprintln!("Usage: backup <db_path> <backup_path>");
                return None;
            }
            Some(CliCommand::Backup {
                db_path: PathBuf::from(&args[2]),
                backup_path: PathBuf::from(&args[3]),
            })
        }
        "check" => {
            let path = args.get(2)
                .map(|p| PathBuf::from(p))
                .unwrap_or_else(|| PathBuf::from("./search.db"));
            Some(CliCommand::Check { path })
        }
        "optimize" => {
            let path = args.get(2)
                .map(|p| PathBuf::from(p))
                .unwrap_or_else(|| PathBuf::from("./search.db"));
            Some(CliCommand::Optimize { path })
        }
        "stats" => {
            let path = args.get(2)
                .map(|p| PathBuf::from(p))
                .unwrap_or_else(|| PathBuf::from("./search.db"));
            Some(CliCommand::Stats { path })
        }
        _ => None,
    }
}

pub fn execute_cli_command(command: CliCommand) -> Result<()> {
    match command {
        CliCommand::InitDb { path } => {
            println!("Initializing database at: {}", path.display());
            let _db = Database::new(&path)?;
            println!("Database initialized successfully");
            Ok(())
        }
        CliCommand::Migrate { path } => {
            println!("Running migrations on: {}", path.display());
            let _db = Database::new(&path)?;
            println!("Migrations completed successfully");
            Ok(())
        }
        CliCommand::Backup { db_path, backup_path } => {
            println!("Backing up {} to {}", db_path.display(), backup_path.display());
            let db = Database::new(&db_path)?;
            db.backup(&backup_path)?;
            println!("Backup completed successfully");
            Ok(())
        }
        CliCommand::Check { path } => {
            println!("Checking database integrity: {}", path.display());
            let db = Database::new(&path)?;
            let is_healthy = db.health_check()?;
            if is_healthy {
                println!("Database is healthy");
            } else {
                println!("Database has issues");
            }
            Ok(())
        }
        CliCommand::Optimize { path } => {
            println!("Optimizing database: {}", path.display());
            let db = Database::new(&path)?;
            db.optimize()?;
            println!("Database optimized successfully");
            Ok(())
        }
        CliCommand::Stats { path } => {
            println!("Database statistics for: {}", path.display());
            let db = Database::new(&path)?;
            let count = db.get_document_count()?;
            let metadata = db.get_indexing_status()?;
            
            println!("Total documents: {}", count);
            println!("Indexing status: {:?}", metadata.indexing_status);
            println!("Last full index: {}", metadata.last_full_index);
            println!("Performance metrics: {:?}", metadata.performance_metrics);
            
            Ok(())
        }
    }
}

pub fn print_usage() {
    println!("Desktop AI Search CLI");
    println!("Usage:");
    println!("  init-db [path]          - Initialize database");
    println!("  migrate [path]          - Run database migrations");
    println!("  backup <db> <backup>    - Create database backup");
    println!("  check [path]            - Check database integrity");
    println!("  optimize [path]         - Optimize database performance");
    println!("  stats [path]            - Show database statistics");
    println!();
    println!("Default database path: ./search.db");
}