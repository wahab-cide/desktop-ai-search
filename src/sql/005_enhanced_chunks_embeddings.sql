-- Enhanced chunks and embeddings schema for semantic search

-- Drop existing embeddings table to recreate with better structure
DROP TABLE IF EXISTS embeddings;

-- Enhanced document_chunks table with additional metadata
ALTER TABLE document_chunks ADD COLUMN word_count INTEGER DEFAULT 0;
ALTER TABLE document_chunks ADD COLUMN sentence_count INTEGER DEFAULT 0;
ALTER TABLE document_chunks ADD COLUMN overlap_start BOOLEAN DEFAULT FALSE;
ALTER TABLE document_chunks ADD COLUMN overlap_end BOOLEAN DEFAULT FALSE;

-- Create enhanced embeddings table with better structure
CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    model_id TEXT NOT NULL DEFAULT 'all-minilm-l6-v2',
    embedding_dimensions INTEGER NOT NULL DEFAULT 384,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Create indices for performance
CREATE INDEX idx_embeddings_chunk_id ON embeddings(chunk_id);
CREATE INDEX idx_embeddings_document_id_new ON embeddings(document_id);
CREATE INDEX idx_embeddings_model_id ON embeddings(model_id);
CREATE INDEX idx_chunks_word_count ON document_chunks(word_count);
CREATE INDEX idx_chunks_content_length ON document_chunks(LENGTH(content));

-- Create a view that joins chunks with their embeddings for easy querying
CREATE VIEW chunks_with_embeddings AS
SELECT 
    c.id as chunk_id,
    c.document_id,
    c.chunk_index,
    c.start_char,
    c.end_char,
    c.content,
    c.word_count,
    c.sentence_count,
    c.overlap_start,
    c.overlap_end,
    c.created_at,
    e.id as embedding_id,
    e.embedding,
    e.model_id,
    e.embedding_dimensions
FROM document_chunks c
LEFT JOIN embeddings e ON c.id = e.chunk_id;

-- Create semantic search support table for storing similarity search results
CREATE TABLE semantic_search_cache (
    id TEXT PRIMARY KEY,
    query_hash TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE
);

CREATE INDEX idx_semantic_cache_query ON semantic_search_cache(query_hash);
CREATE INDEX idx_semantic_cache_score ON semantic_search_cache(similarity_score DESC);
CREATE INDEX idx_semantic_cache_expires ON semantic_search_cache(expires_at);