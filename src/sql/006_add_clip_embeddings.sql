-- Migration 6: Add CLIP embeddings support for multimodal search

-- Table for storing image metadata and paths
CREATE TABLE images (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    format TEXT,
    content_hash TEXT NOT NULL,
    creation_date INTEGER NOT NULL,
    modification_date INTEGER NOT NULL,
    last_indexed INTEGER NOT NULL,
    metadata TEXT DEFAULT '{}', -- JSON metadata (EXIF, etc.)
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

-- Table for storing CLIP embeddings for images
CREATE TABLE image_embeddings (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    embedding BLOB NOT NULL, -- Binary f32 array
    embedding_dimensions INTEGER NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
);

-- Table for storing CLIP embeddings for text queries (cache)
CREATE TABLE text_embeddings (
    id TEXT PRIMARY KEY,
    text_hash TEXT NOT NULL UNIQUE, -- SHA256 of text content
    text_content TEXT NOT NULL,
    model_id TEXT NOT NULL,
    embedding BLOB NOT NULL, -- Binary f32 array
    embedding_dimensions INTEGER NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

-- Table for storing image-text similarity results cache
CREATE TABLE image_text_similarities (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    query_text TEXT NOT NULL,
    model_id TEXT NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    UNIQUE(image_id, text_hash, model_id)
);

-- Indices for performance
CREATE INDEX idx_images_file_path ON images(file_path);
CREATE INDEX idx_images_content_hash ON images(content_hash);
CREATE INDEX idx_images_modification_date ON images(modification_date);
CREATE INDEX idx_images_format ON images(format);

CREATE INDEX idx_image_embeddings_image_id ON image_embeddings(image_id);
CREATE INDEX idx_image_embeddings_model_id ON image_embeddings(model_id);
CREATE INDEX idx_image_embeddings_created_at ON image_embeddings(created_at);

CREATE INDEX idx_text_embeddings_text_hash ON text_embeddings(text_hash);
CREATE INDEX idx_text_embeddings_model_id ON text_embeddings(model_id);

CREATE INDEX idx_similarities_image_id ON image_text_similarities(image_id);
CREATE INDEX idx_similarities_text_hash ON image_text_similarities(text_hash);
CREATE INDEX idx_similarities_similarity_score ON image_text_similarities(similarity_score DESC);
CREATE INDEX idx_similarities_model_id ON image_text_similarities(model_id);

-- View for images with embeddings
CREATE VIEW images_with_embeddings AS
SELECT 
    i.*,
    ie.model_id as embedding_model_id,
    ie.embedding,
    ie.embedding_dimensions,
    ie.created_at as embedding_created_at
FROM images i
LEFT JOIN image_embeddings ie ON i.id = ie.image_id;

-- View for multimodal search results
CREATE VIEW multimodal_search_results AS
SELECT 
    i.id as image_id,
    i.file_path,
    i.file_name,
    i.format,
    i.width,
    i.height,
    its.similarity_score,
    its.query_text,
    its.model_id,
    its.created_at as search_date
FROM images i
JOIN image_text_similarities its ON i.id = its.image_id
ORDER BY its.similarity_score DESC;