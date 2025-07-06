-- Initial schema for desktop AI search

-- Documents table
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL,
    file_type TEXT NOT NULL,
    creation_date INTEGER NOT NULL,
    modification_date INTEGER NOT NULL,
    last_indexed INTEGER NOT NULL,
    file_size INTEGER NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

-- FTS5 external content table
CREATE VIRTUAL TABLE search_index USING fts5(
    content,
    content='documents',
    content_rowid='rowid'
);

-- Indexing status table
CREATE TABLE indexing_status (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    total_files INTEGER NOT NULL DEFAULT 0,
    indexed_files INTEGER NOT NULL DEFAULT 0,
    last_update INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'idle',
    errors TEXT NOT NULL DEFAULT '{}',
    performance_metrics TEXT NOT NULL DEFAULT '{}'
);

-- Initialize indexing status
INSERT INTO indexing_status (id) VALUES (1);