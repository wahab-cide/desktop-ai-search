-- Add FTS5 full-text search for document chunks

-- Create FTS5 virtual table for chunk content
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    document_id UNINDEXED,
    chunk_id UNINDEXED,
    content='document_chunks',
    content_rowid='rowid',
    tokenize='porter unicode61 remove_diacritics 1'
);

-- Triggers to keep FTS5 index synchronized with document_chunks
CREATE TRIGGER chunks_fts_insert AFTER INSERT ON document_chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, document_id, chunk_id) 
    VALUES (new.rowid, new.content, new.document_id, new.id);
END;

CREATE TRIGGER chunks_fts_delete AFTER DELETE ON document_chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, document_id, chunk_id) 
    VALUES('delete', old.rowid, old.content, old.document_id, old.id);
END;

CREATE TRIGGER chunks_fts_update AFTER UPDATE ON document_chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, document_id, chunk_id) 
    VALUES('delete', old.rowid, old.content, old.document_id, old.id);
    INSERT INTO chunks_fts(rowid, content, document_id, chunk_id) 
    VALUES (new.rowid, new.content, new.document_id, new.id);
END;

-- Rebuild FTS5 index with existing data
INSERT INTO chunks_fts(rowid, content, document_id, chunk_id)
SELECT rowid, content, document_id, id FROM document_chunks;