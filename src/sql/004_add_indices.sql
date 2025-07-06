-- Add performance indices

CREATE INDEX idx_documents_file_path ON documents(file_path);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);
CREATE INDEX idx_documents_last_indexed ON documents(last_indexed);
CREATE INDEX idx_documents_file_type ON documents(file_type);
CREATE INDEX idx_documents_modification_date ON documents(modification_date);

CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_chunks_chunk_index ON document_chunks(document_id, chunk_index);

CREATE INDEX idx_embeddings_document_id ON embeddings(document_id);
CREATE INDEX idx_embeddings_model_version ON embeddings(model_version);