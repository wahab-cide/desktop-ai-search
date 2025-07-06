-- Add vector embeddings support

CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    model_version TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);