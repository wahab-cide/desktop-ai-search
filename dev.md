# Desktop AI Search Engine: Complete Development Roadmap

## Project Overview

### What We're Building

A **privacy-first, on-device AI search engine** that serves as your personal digital assistant for finding anything across your entire digital life. Think of it as "Google for your personal data" but running entirely on your local machine with zero cloud dependencies.

### Core Product Vision

Instead of relying on cloud-based AI services that require uploading your sensitive documents, emails, and personal files, this application processes everything locally using state-of-the-art AI models. Users can ask natural language questions like "Find that tax document my accountant emailed last March" or "Show me the recipe screenshot I took last weekend" and get instant, accurate results across all their personal data.

### Key Differentiators

**Complete Privacy**: All AI processing happens on-device. No data ever leaves your computer, making it safe for lawyers, doctors, journalists, and anyone handling sensitive information.

**Multimodal Intelligence**: Unlike traditional file search tools, this leverages AI to understand the *content* of documents, images, emails, screenshots, and even audio files through OCR, image recognition, and speech transcription.

**Natural Language Interface**: Users interact through conversational queries rather than keyword searches, making it intuitive for non-technical users while powerful enough for complex research tasks.

**Contextual Understanding**: The AI understands relationships between different data types - connecting a person mentioned in an email to photos they appear in, or linking meeting notes to related documents.

### Technical Architecture

**Desktop-First Strategy**: Starting with macOS to take advantage of Metal acceleration and the target user base of knowledge workers and professionals who prioritize privacy and are willing to pay for premium tools.

**Local AI Stack**: Utilizes quantized large language models (Mistral 7B, Phi-3) running via llama.cpp, combined with specialized models for embeddings (MiniLM), OCR (PaddleOCR), image understanding (CLIP), and speech recognition (Whisper).

**Hybrid Search Engine**: Combines traditional full-text search with vector similarity search for semantic understanding, enabling both precise keyword matching and conceptual queries.

**Real-Time Indexing**: Continuously monitors the file system for changes, automatically indexing new content in the background without user intervention.

### Target Market

**Primary Users**: Privacy-conscious professionals including lawyers, journalists, researchers, consultants, and executives who work with sensitive documents and need powerful search capabilities across their digital workspace.

**Secondary Market**: General knowledge workers who have accumulated large amounts of personal data and struggle with existing search tools that can't understand context or content.

### Business Model

**Freemium Desktop Application**: Basic search functionality with premium features like larger context windows, advanced AI models, and enterprise integrations.

**Professional/Enterprise Licenses**: Enhanced versions for organizations requiring deployment across teams with additional security and compliance features.

### Success Criteria

The application succeeds when users can find any piece of information from their digital life in under 3 seconds using natural language, with 95%+ accuracy, while maintaining complete data privacy and requiring minimal user configuration or maintenance.

This represents a fundamental shift from cloud-dependent AI assistants to truly private, local intelligence that grows smarter the more personal data it can access and understand.

---

## Tech Stack

### Core Architecture
- **Framework**: Tauri (Rust backend + React/Solid.js frontend)
- **Language**: Rust for backend, TypeScript for frontend
- **UI Library**: Solid.js with Tailwind CSS
- **Build Tool**: Vite for frontend bundling

### Backend Stack

#### AI/ML Components
- **LLM Runtime**: `llama.cpp` with Rust bindings
- **Models**: Mistral 7B, Phi-3, Llama 3 (quantized via GGUF)
- **Embeddings**: all-MiniLM-L6-v2 or all-mpnet-base-v2
- **OCR**: PaddleOCR (Python subprocess)
- **Speech-to-Text**: Whisper (local)
- **Image Understanding**: CLIP for image-text matching

#### Storage & Search
- **Database**: SQLite with FTS5 for full-text search
- **Vector Store**: Chroma (with Qdrant as optional upgrade)
- **File Monitoring**: `notify` crate for filesystem watching
- **Caching**: In-memory caching with disk persistence

#### Document Processing
- **PDF**: `lopdf` crate with OCR fallback
- **Office Docs**: `docx-rs` for Word documents
- **Images**: Image processing with CLIP embeddings
- **Audio**: Whisper transcription with timestamp mapping

### Frontend Stack

#### UI Framework
- **Base**: Solid.js for reactive UI
- **Styling**: Tailwind CSS v4
- **State Management**: Solid stores and signals
- **Routing**: @solidjs/router (if needed for settings)

#### Tauri Integration
- **API Communication**: @tauri-apps/api for backend calls
- **File System**: @tauri-apps/plugin-fs for file access
- **Platform Integration**: Native system tray, global hotkeys

### Platform-Specific Optimizations

#### macOS (Primary Target)
- **GPU Acceleration**: Metal integration via llama.cpp
- **Authentication**: TouchID/FaceID via LocalAuthentication
- **System Integration**: Spotlight, Services menu, Alfred workflows
- **Email**: Mail.app database access

#### Cross-Platform Support (Future)
- **Windows**: DirectML acceleration
- **Linux**: CUDA/OpenCL support
- **File Systems**: Platform-specific file access patterns

### Development Tools
- **Build System**: Cargo for Rust, npm/Vite for frontend
- **Testing**: `cargo test`, property-based testing with `proptest`
- **Profiling**: `perf`, `flamegraph` for performance analysis
- **Error Handling**: `anyhow` and `thiserror` for robust error management

### Key Dependencies

#### Rust Crates
```toml
# Core
tauri = "2.0"
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }

# Database & Storage
rusqlite = { version = "0.29", features = ["bundled"] }
r2d2 = "0.8"
r2d2_sqlite = "0.22"

# File System & Monitoring
notify = "6.0"
walkdir = "2.0"
mime_guess = "2.0"

# AI/ML
llama-cpp-rs = "0.1"  # or similar binding
candle-core = "0.3"
candle-transformers = "0.3"

# Document Processing
lopdf = "0.26"
docx-rs = "0.4"

# Error Handling
anyhow = "1.0"
thiserror = "1.0"
```

#### Frontend Dependencies
```json
{
  "dependencies": {
    "solid-js": "^1.9.5",
    "@solidjs/router": "^0.32.0",
    "@tauri-apps/api": "^2.6.0",
    "@tauri-apps/plugin-fs": "^2.4.0"
  },
  "devDependencies": {
    "vite": "^6.0.0",
    "vite-plugin-solid": "^2.11.6",
    "tailwindcss": "^4.0.7",
    "@tailwindcss/vite": "^4.0.7",
    "typescript": "^5.7.2"
  }
}
```

### Architecture Benefits

#### Performance
- **On-device processing** eliminates network latency
- **Metal/GPU acceleration** for AI inference
- **Efficient vector search** with approximate algorithms
- **Incremental indexing** minimizes resource usage

#### Privacy & Security
- **Zero cloud dependencies** - all data stays local
- **Encrypted storage** with biometric authentication
- **Sandboxed execution** via Tauri security model
- **Audit logging** for data access tracking

#### Scalability
- **Modular architecture** allows easy feature additions
- **Cross-platform foundation** for future expansion
- **Plugin system ready** for third-party integrations
- **Configurable models** for different hardware tiers

---

## Development Timeline

### Phase 1: Core Backend Infrastructure (Weeks 1-3)

#### Week 1: Data Models & Storage

**Day 1-2: Design Core Data Structures**
- Create `Document` struct with fields: id (UUID), file_path, content_hash (SHA256), file_type enum, creation_date, modification_date, last_indexed, file_size, and optional metadata HashMap
- Design `SearchResult` struct containing document_id, relevance_score (f64), matched_content snippets with character positions, result_type enum (exact, semantic, fuzzy), and ranking factors
- Implement `IndexMetadata` struct tracking indexing_status, total_documents, last_full_index, error_counts, and performance metrics
- Add proper serialization/deserialization using serde for all structs
- Create comprehensive error types using `thiserror` covering DatabaseError, FileSystemError, IndexingError, and SearchError variants

**Day 3-4: SQLite Database Schema**
- Design `documents` table with columns: id (TEXT PRIMARY KEY), file_path (TEXT UNIQUE), content_hash (TEXT), file_type (TEXT), creation_date (INTEGER), modification_date (INTEGER), last_indexed (INTEGER), file_size (INTEGER), metadata (TEXT as JSON)
- Create `search_index` FTS5 table as "external content" table referencing documents to avoid content duplication: `CREATE VIRTUAL TABLE search_index USING fts5(content, content='documents', content_rowid='rowid')`
- Design separate vector storage strategy: evaluate Qdrant-RS for file-based vector storage vs. keeping embeddings in separate SQLite database to prevent main DB bloat
- Create `indexing_status` table: id (INTEGER PRIMARY KEY), total_files (INTEGER), indexed_files (INTEGER), last_update (INTEGER), errors (TEXT as JSON)
- Implement database migrations using `rusqlite_migration` crate with versioned schema updates
- Configure WAL mode and `PRAGMA synchronous = NORMAL` for performance from day one

**Day 5-7: Database Operations & Error Handling**
- Set up connection pooling using `r2d2` with SQLite for thread-safe database access
- Implement CRUD operations with proper prepared statements and parameter binding
- Create transaction-based batch operations for bulk inserts/updates during indexing
- Add database backup and recovery mechanisms with automatic corruption detection
- Implement comprehensive error handling with context preservation using `anyhow` for error chains
- Create database health checks and automatic repair functionality for corrupted indexes
- Build CLI management tool using `cargo xtask` with commands: `init-db`, `migrate`, `backup`, `check`, `optimize` for development and CI automation
- Implement weekly `PRAGMA optimize` scheduling to maintain query performance as data grows
- Create database maintenance routines including VACUUM scheduling and integrity checks

#### Week 2: File System Integration

**Day 1-2: File System Monitoring**
- Integrate `notify` crate for cross-platform file system watching with recursive directory monitoring
- Create unified `WatcherEvent` enum (Created, Modified, Renamed(old, new), Deleted) to normalize OS-specific behaviors between FSEvents (macOS) and ReadDirectoryChangesW (Windows)
- Implement path-based debouncing using `HashMap<PathBuf, Instant>` to handle atomic-save operations (MS Word, Pages) that emit rapid remove→create sequences
- Add handling for file moves, renames, and deletions with proper cleanup of stale database entries using `std::fs::metadata` calls to resolve FSEvents ambiguity
- Implement watch state persistence with `watch_state` table storing (file_path, sha256, mtime) for diff calculation across app restarts
- Create `--watch-dry-run` mode for debugging filters and rename detection during development
- Implement graceful shutdown and restart of file watchers with state persistence

**Day 3-4: File Type Detection & Processing**
- Use `infer` crate for magic-byte signature detection (500+ formats) instead of `mime_guess` for reliable content type identification
- Implement ZIP container inspection for office formats (.docx, .pages) by checking `[Content_Types].xml` or `_rels/.rels` with memory-buffered parsing
- Create comprehensive `FileType` enum with `#[non_exhaustive]`, `FromStr`, and `Display` implementations for future expansion
- Design `FileClassificationResult` struct containing `{type, is_text, should_embed, confidence}` for consistent downstream processing
- Add special handling for HEIC/AVIF images common on macOS with graceful fallback for unsupported codecs
- Implement `tokio::time::timeout` wrapper (3-second threshold) for all file type detection to prevent hanging on malformed files or ZIP bombs
- Add file size limits and processing timeouts to prevent hanging on large files

**Day 5-7: Metadata Extraction & Deduplication**
- Implement tiered hashing strategy: check (size, mtime) first → hash first+last 1MB → full SHA256 only for small files (<10MB) or hash mismatches
- Create `FileHashCache` (in-memory + SQLite persistence) keyed by (file_path, mtime) to prevent unnecessary rehashing during repeated scans
- Extract file system metadata (creation date, modification date, size, permissions) using `std::fs::metadata`
- Implement symbolic link handling with `std::fs::canonicalize()` and visited set using `HashSet<(device_id, inode)>` to prevent traversal loops
- Create extended metadata extraction queue for EXIF (images) and ID3 (audio) tags stored as JSON in metadata column, processed on "slow queue" to avoid blocking simple files
- Add configuration system using `config` crate with TOML/JSON file support for user preferences including `--max-depth` traversal limits
- Implement smart deduplication logic that preserves file location history while avoiding content reprocessing
- Store device_id + inode (Unix) or FileID (Windows/macOS) to track renamed/moved files across locations

#### Week 3: Document Processing Pipeline

**Day 1-2: OCR Integration**
- Isolate PaddleOCR in gRPC microservice launched via `Process::spawn` to keep main Tauri bundle pure-Rust and avoid notarization complications
- Implement Rust-side preprocessing pipeline: grayscale → adaptive threshold → deskew using `fast_image_resize` + `opencv-rust` for 30% speed improvement
- Create bounded semaphore for OCR concurrency (`physical_cpus / 2` or `available_gpu_memory / model_size`) to prevent VRAM thrashing
- Implement fused confidence scoring: aggregate PaddleOCR per-line confidence to page-level, drop pages below 0.70 threshold, store quality scores in metadata JSON
- Add image buffer recycling with `Arc<[u8]>` pool and WebP thumbnail caching for re-OCR scenarios
- Cache `sha256(page_bytes)` to skip identical pages across documents (common with scanned forms)
- Add `--ocr-max-workers N` CLI flag and `--ocr-quality-threshold` configuration options
- Implement parallel OCR processing for multi-page documents with memory management and proper error recovery

**Day 3-4: Audio Processing with Whisper**
- Integrate `candle-whisper` (pure Rust) for Metal acceleration on Apple Silicon without Python dependencies
- Use `ffmpeg` only for format normalization: `ffmpeg -i … -ar 16000 -ac 1 -f wav -` then process with Rust
- Implement streaming transcription with 30-second windows to prevent RAM overrun on long recordings
- Track `byte_offset → timestamp` mapping for precise snippet positioning and UI jump-to functionality
- Implement language detection using Whisper's `<|language|>` token logits with `whatlang-rs` fallback for short clips
- Create energy-based VAD + k-means speaker embeddings for basic speaker diarization (2-5 speakers max)
- Store `(start_ms, end_ms, speaker_id)` alongside transcript chunks for UI speaker highlighting
- Implement lazy model loading: ship `tiny.en` and `base.en` by default, fetch larger models to `~/Library/Application Support/aisearch/models/`
- Add `--max-audio-duration` (default 2h) and write partial transcripts every 5 minutes for crash recovery
- Create model manager UI for downloading and managing different Whisper model sizes

**Day 5-7: Document Format Processing & Pipeline Architecture**
- Use `lopdf` for PDF text extraction with `pdfium-render` fallback for malformed PDFs, maintaining per-page `extraction_mode` enum (Text, ImageOCR, FallbackImageOCR)
- Enhance `docx-rs` integration to parse `<w:pStyle>` hierarchy for semantic chunking at heading boundaries
- Add support for Markdown (`pulldown-cmark`) and HTML (`html2text`) with whitespace normalization and boilerplate removal
- Implement sentence-aware chunking: ~1k tokens with 15% overlap (`start += chunk_len × 0.85`), storing `chunk_id`, `start_char`, `end_char` for precise highlighting
- Create hierarchical progress tracking with `indicatif`: document → page → task levels with exponential moving average ETA calculation
- Implement per-document SQL transactions with rollback on failure, preserving progress on successful documents
- Add graceful shutdown via `ctrlc` crate: flush current batch, commit, and exit cleanly
- Create `--preview-index` mode (limit 20 files) for testing new pipelines without long waits
- Implement comprehensive error logging to `index_errors.log` with full backtraces for debugging
- Add scheduled maintenance: nightly `VACUUM` + `PRAGMA optimize` when on AC power
- Feed processed chunks to both FTS5 external-content table and vector store for unified hybrid retrieval via SQL JOIN

### Phase 2: AI/ML Integration (Weeks 4-6)

#### Week 4: Local LLM Setup

**Day 1-2: llama.cpp Integration**
- Set up `llama-cpp-rs` Rust bindings with single-threaded Tokio task architecture using `tokio::sync::mpsc` request queue to prevent Metal command-buffer deadlocks
- Implement "one-active-model" guard: drop previous `llama_context`, call `metal_fflush`, wait 16ms for VRAM release before loading next model
- Create model storage in `~/Library/Application Support/aisearch/models/<name>/<sha256>/` with integrity verification using `sha256sum -c`
- Ship YAML model manifest (`models.yml`) with `name, size_gb, quant, sha256, license` metadata for offline model management
- Add model download and verification system with checksum validation and automatic updates stored in database
- Implement `aisearch models pull mistral --quant q4` CLI command with GUI wrapper for power users
- Create GGUF model format validation and compatibility checking with proper error handling

**Day 3-4: Model Configuration & Memory Management**
- Implement runtime VRAM probing using `MTLDevice.recommendedMaxWorkingSetSize`, reserve 350MB for OS, compute optimal `n_gpu_layers` based on available memory
- Create model profiles storing computed layer counts to skip VRAM probing on subsequent launches
- Design preset system (`creative`, `balanced`, `precise`) mapping to parameter tuples (temperature, top-p, top-k) with advanced drawer for raw values
- Implement dual fixed-context architecture: short (2k tokens) and long (8k tokens) contexts to avoid expensive `llama_kv_cache_seq_shift` reallocations
- Add context dispatch logic based on tokenized prompt + retrieved chunks length for deterministic latency
- Create OOM protection: poll `sysinfo::System::refresh_memory()` every 250ms, yield generation if free RAM < 500MB threshold
- Implement context GC (drop/recreate) taking <1s for Q4 models as recovery mechanism
- Add `--prewarm N` flag running `llama_eval` on 16-token dummy prompt to upload Metal shaders and hide first-call latency
- Persist "last used model + params" for instant restoration on next boot

**Day 5-7: Inference Pipeline & Performance Optimization**
- Create async streaming inference using `async_stream::stream!` wrapper around blocking `llama_cpp_rs::Inference::next_token()` calls
- Implement 3-token buffering before UI push to smooth bursty token emission without perceptible delay
- Design data-driven prompt template system with SQLite `prompt_templates` table: `(name, system_prefix, user_prefix, assistant_prefix, stop_tokens)`
- Add inference caching: hash `(template_name, prompt_sha256)` to cache first 128 tokens with LRU eviction (~2.5MB total memory impact)
- Implement time-sliced batching using `llama_kv_cache_seq_dup` for ~1.3× throughput improvement while preserving low latency per stream
- Create comprehensive metrics tracking: `tokens/s`, `ms_per_token`, `load_ms`, `ctx_mem_mb` logged to `inference_stats_rolling` table (10k row cap)
- Add two-pass safety filtering: (1) profanity/PII regex patterns, (2) user-configurable block-list with token redaction and warning tags
- Implement `tokio::time::timeout` (30s default) for generation timeouts to prevent runaway loops
- Add `/v1/checkpoint` IPC for pause/resume generation without losing context state
- Capture Metal performance counters (`MTLCommandBuffer.gpuStartTime`) for GPU profiling diagnostics
- Create performance dashboard with sparklines from metrics database for system monitoring

#### Week 5: Embedding & Vector Search

**Day 1-2: Sentence Transformer Integration**
- Set up local sentence transformer models using `candle-transformers` for Metal acceleration and self-contained binary, with ONNX Runtime as fallback for unsupported models
- Implement exact preprocessing pipeline matching model training: smart quote conversion, unicode NFKC normalization, emoji filtering, precise attention mask logic to prevent >0.05 cosine score degradation
- Create adaptive batch sizing: `batch = min(max_tokens / seq_len, max_batch_cap)` clipped at `seq_len = chunks.max()` to avoid padding 128-token snippets to 1k tokens
- Add embedding generation with proper batching to optimize GPU utilization (≈64 × 1k-token chunks in 8GB VRAM)
- Implement RocksDB-based embedding cache using `(model_rev, text_sha256)` → embedding mapping for crash-resistant, high-write performance
- Create `embeddings_meta` table storing `{model_id, dim, created_at}` for version compatibility checking and upgrade blocking on dimension mismatches
- Apply stop-word removal before embedding for FTS alignment, reducing vector store size by 10-15%

**Day 3-4: Vector Database Setup**
- Integrate Qdrant-RS or valdb (Rust-native) instead of Python-based Chroma to avoid notarization complications and 150MB+ app size bloat
- Implement quarterly index sharding (`2025-Q3`, `2025-Q4`) with multi-shard search to avoid full rebuilds during corpus growth
- Create two-stage vector storage: raw embeddings + PQ compression (8×8-bit sub-vectors) with fallback to exact cosine for close results
- Store PQ codebooks once per model version in `vector_codebooks/` directory (~12kB each for 384-dim embeddings)
- Implement incremental updates with delta HNSW for O(log n) inserts and nightly merge operations to maintain recall quality
- Add HNSW indexing for ≤2M vectors with IVF-PQ fallback for larger datasets requiring coarse codebook retraining
- Create parallel FTS5 "keywords" table keyed by `doc_id, chunk_id` for hybrid fusion support
- Implement `aisearch index stats` CLI showing `num_vectors`, `mean_degree`, `recall@10` validation metrics
- Add `truncate cascade` script for vector DB cleanup during testing to prevent disk bloat

**Day 5-7: Hybrid Search Implementation**
- Implement query routing logic based on tokenized analysis: `keyword_density = keywords/total_tokens` and `len_tokens` with heuristic routing table:
  - `len_tokens < 4 AND keyword_density > 0.55` → FTS-only
  - Named entities, dates → Hybrid
  - `len_tokens > 20` → Vector → Re-rank
- Create Reciprocal Rank Fusion (RRF) with `1/(k + rank)` formula, tunable k=60, configurable `alpha_text` and `alpha_vec` weights stored in config table
- Integrate tiny-bi-e5-v2 cross-encoder (40MB, 256-dim) for top-30 result re-ranking with <70ms latency cap, lazy-loaded on demand
- Add search mode presets: *Precise* (FTS-heavy), *Balanced*, *Exploratory* (vector-heavy) adjusting RRF weights, ANN k-value, and cross-encoder usage
- Implement comprehensive analytics logging: `{query_sha256, route, latency_ms, hits_ft, hits_vec, rerank_delta, click_id}` with 180-day TTL
- Create nightly analytics job computing P@5 and NDCG metrics to detect relevance regressions
- Add query result caching for last 20 searches using `(query, fused_ids)` mapping for iterative refinement scenarios
- Implement failsafe fallback: vector DB panic → seamless FTS-only mode to prevent blank search results
- Create search result re-ranking using cross-encoder models for improved relevance with configurable on/off toggle

#### Week 6: Multimodal Processing

**Day 1-2: CLIP Integration for Image Understanding**
- Set up CLIP model using `candle-clip` with ViT-B/32 bundled weights, download pre-quantized GGUF checkpoints for ViT-L/14 (636MB fp16) with metadata manifest `{dim, patch_size, mean, std}`
- Implement Metal-accelerated image preprocessing using `metal-petal` or `fast_image_resize` with CoreGraphics zero-copy mapping for 2× performance over CPU resize
- Create runtime GPU memory measurement: `max_batch = (vram_free - 256MB) / (tensor_size × 1.4)` with bounded channel coordination between PDF OCR and CLIP processing
- Add dual caching: CLIP tensor embeddings + ~400px JPEG thumbnails for instantaneous gallery views
- Store `clip_version` alongside embeddings for cache invalidation during model swaps
- Implement LRU RAM cache using `moka` for last 256 embeddings to optimize Photos.app-style timeline scrolling
- Pre-compute and persist image↔text dot-products for document captions to accelerate reciprocal search
- Create image-to-text and text-to-image search capabilities with temperature-scaled cosine similarity for image-only mode

**Day 3-4: Screenshot and Visual Content Processing**
- Implement dual-pass OCR strategy: (1) `tesseract --psm 7` on down-scaled image for >5% ASCII hit rate detection, (2) PaddleOCR for full layout-aware processing only when needed
- Create screenshot detection via heuristics (UI grey bars, exact resolution bins) with perceptual hash (phash) deduplication before heavy OCR
- Integrate `Donut-lite` or `LayoutLM v3` via ONNX for layout analysis outputting `Vec<LayoutBlock>` with `{type, bbox, text, confidence}` for structured content extraction
- Add diagram/chart classification using `py-matplotlib-chart-classifier` (40MB Python subprocess) for coarse labeling (bar/line/pie charts)
- Implement visual similarity search with image-only CLIP index using temperature-scaled cosine similarity to avoid mixed-modal drift
- Create solid-color border detection and cropping to improve CLIP accuracy by removing UI chrome
- Extract and store screenshot EXIF metadata (`CaptureDate`, `AppName`) for timeline queries, leveraging Mac screenshot embedding
- Add `mode = ImageOnly` API flag for pure image-to-image similarity search bypassing text embeddings

**Day 5-7: Cross-Modal Entity Linking & Timeline Search**
- Design micro-ontology with `entity` table `{id UUID, kind ENUM}` and `mention` table `{entity_id, doc_id, chunk_id, modality ENUM, start, end, confidence}` for unified ID space
- Integrate HF `ontonotes-bert` NER (5ms per chunk) for text mentions, extract photo mentions from EXIF GPS→reverse-geocode and filename heuristics
- Implement Rust rule-engine using `esso` crate for linkage rules allowing logic updates without LLM recompilation
- Create temporal indexing: store entities/dates in `tsvector` + `start_epoch, end_epoch` for range scans, use `chrono-natural-language` for parsing with LLM fallback for complex spans
- Build bidirectional adjacency list `{entity_a, entity_b, doc_count, first_seen, last_seen}` with HNSW graph using PMI edge weights for co-occurrence queries
- Implement HDBSCAN clustering over text embeddings with `cluster_id` assignment, auto-generated cluster labels from top TF-IDF tokens stored in `cluster_meta`
- Create visual timeline using Solid.js + `vis-timeline` backed by `/timeline?start=&end=` API returning `{date, type, icon, doc_id}` with 5k event pagination
- Add timeline zoom slider with exponential steps (day→week→month) and hover prefetching of CLIP thumbnails + NER snippets
- Implement mini-map for geolocated photos using `leaflet-tiny` (offline) and timeline node clustering for dense event management

### Phase 3: Advanced Search Features (Weeks 7-9)

#### Week 7: Natural Language Processing

**Day 1-2: Query Understanding & Intent Detection**
- Implement multi-label intent classification using MiniLM-128 (<3ms inference) for compound requests ("Find slides John sent last Friday AND photos we took afterwards")
- Integrate `stanza-lite` or distilled spaCy model fine-tuned on emails/chat text to reduce 350MB language model download and avoid GPU allocation clashes
- Create dynamic spell-correction whitelist: inject tokens appearing ≥10× in document corpus into custom dictionary for personalized corrections avoiding domain word ("llama.cpp", "Okta") false fixes
- Implement opt-in query expansion with per-route thresholds: expand only when FTS hit-count <K and ANN recall <R to maintain precision
- Add `moka` cache for last 1000 `(query → classification)` mappings to skip re-computing NER on every keystroke
- Create `user_dictionary` table `{token, freq}` feeding both spell-check and expansion systems
- Output structured `QueryIntent {labels: Vec<Intent>, ner: Vec<Entity>, normalized_text}` to search router preventing duplicate processing

**Day 3-4: Advanced Query Processing**
- Implement PEG grammar using `pest` or `nom` for natural-language boolean logic: `expr = term (AND|OR term)* (NOT term)*` with `term = phrase | entity | date_range | size_range`
- Create temporal understanding with `chrono-nlp` for standard phrases plus lookup table for fiscal/calendar events (Easter, quarter-end) with LLM fallback only when lookup fails
- Add quantitative filter parsing using `rust-sechkell` for number handling ("five hundred") plus unit conversion table normalizing to `bytes`, `pages` for range scan filters
- Implement natural-language refinement suggestions based on Week 5 analytics: suggest filter removal when recall <20% with active filters
- Create `STOPWORDS_BOOL` set preventing boolean keywords in filenames ("and.jpg") from triggering parser
- Add debug AST logging (`RUST_LOG=debug`) for grammar mis-parse visibility
- Include `query_version` column in analytics for A/B testing grammar updates without full releases
- Store unit-normalized values in document metadata for efficient range filtering

**Day 5-7: Contextual & Personalized Search**
- Implement transparent scored feature vector: `score = α×base_relevance + β×recency_boost + γ×access_frequency + δ×same_project_bonus` with user-configurable coefficients in JSON settings
- Create exponential recency decay `e^(-t/τ)` with configurable τ (default 30 days) and expose personalization sliders in "Advanced → Personalisation" UI
- Add current project detection from: (1) active window bundle ID via Tauri plugin, (2) calendar events ±2h, (3) folder browsed ≥3× in 24h with confidence >0.6 threshold
- Implement search session management: allocate `session_id` on search start, cache last N `(query, entity, doc_id)` triples, boost overlapping entities by 1 logit if next query within 5min shares ≥1 entity
- Create local-first telemetry: record `(query_sha256, doc_clicked, latency, personalised_score)` in SQLite with hourly aggregation and opt-in anonymized upload
- Add "Incognito search" mode zeroing all personalization weights for privacy-conscious queries
- Implement `priority_channel` for ranking tasks ensuring interactive searches preempt background indexing
- Start with coefficients `β=0.15, γ=0.10, δ=0.25` and log `score_components` for feature importance analysis
- Create `current_context(project_id?)` cache updating every 10s via OS event hook with 30min session timeout

#### Week 8: Search Intelligence

**Day 1-2: Advanced Ranking Algorithm**
- Implement learning-to-rank (LTR) model using LightGBM or XGBoost (1-2MB binary, 10k docs <5ms) with feature vector: `{text_bm25, cosine_vec, recency_decay, user_recency_clicks, doc_quality, same_project_flag, diversity_penalty}`
- Create cold-start strategy: seed LTR with synthetic labels (BM25 top-k + vector top-k as pseudo-positives) blending 80% synthetic + 20% real clicks, gradually annealing to 100% real
- Implement file importance scoring: `longevity × access_frequency × explicit_stars` with min-max normalization and bucket encoding for tree model compatibility
- Serialize LightGBM models to JSON in `ranking_models/version/` directory with hot-swap capability via file watching
- Add MMR diversity selection with λ=0.5: iteratively select `max(rank_score - λ·sim_to_selected_max)` for result diversification
- Create A/B testing framework: maintain dual rankers, assign 10% sessions to experimental model, log `(model_id, rank@1, click@1)` telemetry
- Implement comprehensive performance metrics tracking (tokens/second, latency, memory usage) with automated benchmarking

**Day 3-4: Result Processing & Organization**
- Pre-cluster corpus weekly using Top2Vec/BERTopic (HDBSCAN over embeddings) storing `cluster_id → top_keywords`, enabling query-time `GROUP BY` clustering
- Implement hybrid duplicate detection: exact SHA-256 + fuzzy simhash on text with v-stack UI for version selection (PDF, DOCX, email attachment)
- Maintain single vector entry per duplicate group (lowest doc_id) with `duplicates` table for full context restoration
- Create tiered summarization: extractive text-rank on chunk windows → abstractive LLM fallback for >1024 chars, cache by `doc_sha256 + model_rev`
- Implement intelligent highlighting: store `start_char, end_char` from chunks, render 30-word windows with highlighted terms, fallback to ANN-nearest token for pure vector hits
- Run duplicate detection during indexing rather than query time for performance
- Persist `cluster_top_sentence` (highest TF-IDF) for instant cluster previews
- Create pre-computed "golden snippets" (first paragraph + headings) as LLM summarization timeout fallback

**Day 5-7: User Experience Intelligence**
- Implement encrypted search history using separate SQLCipher DB (`~/Library/Application Support/aisearch/history.db`) with TouchID-unlocked AES-256 key for privacy protection
- Create saved searches as persisted AST + ranking config with background Tokio scheduler (24h default interval) and throttled notifications (3/day max)
- Build tag system as first-class entities: `tag(id, name)`, `doc_tag(doc_id, tag_id)` with autocomplete leveraging query suggestion trie
- Implement da-trie autocomplete: lowercase prefixes with `{top_score, doc_freq}` payload, incremental updates, pre-computed Damerau-1 edits as ghost nodes for zero-cost fuzzy search
- Create analytics dashboard with hourly aggregation into materialized views, `/metrics` endpoint serving sparklines via `uPlot` (opt-in only)
- Add "Auto-clean after 90 days" toggle for automatic search history deletion beyond configurable threshold
- Implement power-aware background tasks: saved search cron runs only on AC power + idle state (`pmset -g batt`)
- Create "why did this rank here?" debug popover showing feature weights breakdown for ML ranking transparency and user trust
- Add comprehensive search history with privacy controls and automatic cleanup policies

#### Week 9: Performance Optimization

**Day 1-2: Indexing Performance**
- Profile indexing pipeline using `cargo flamegraph` and macOS `sample` (Instruments-CLI) for high-granularity GPU traces targeting hashing and OCR/Whisper subprocess hotspots
- Implement intelligent parallelization: limit `lopdf`/`pdfium` to `min(physical_cores/2, 4)` threads to prevent heap contention, gate OCR workers with semaphore tied to `sysctl hw.perflevel0_availcpus`
- Create robust incremental indexing comparing `inode + hash + page_count` rather than `mtime` alone, use APFS `fsid + fileid` for rename detection without rehashing
- Implement storage compression: weekly SQLite `VACUUM ... INTO 'tmp'` replacement, bit-packing on PQ codes using `bincode::encode_from_slice::<u8,4>()` for 30% storage reduction
- Pre-allocate document worker pool sized via `sysinfo::System::total_memory()` with 2GB RAM safety margin, persist configuration in `settings.json`
- Add "quiet hours" scheduler: time-slice heavy indexing to App Nap windows (macOS idle triggers) for silent operation during active use
- Create `--reindex-changed` CLI with inode map scanning reducing cold-start diff from minutes to seconds

**Day 3-4: Search Performance**
- Optimize HNSW parameters: reduce `M` from 16→12 and `efConstruction` from 200→64 for <2% recall@10 loss with halved build/query time
- Enable SIMD dot-product acceleration: compile Qdrant-RS with `RUSTFLAGS="-C target-feature=+neon"` for 1.6× speedup on 384-dim vectors
- Implement intelligent result caching keyed on `(query_sha256, personalization_hash)` with pre-warming of last 5 queries on app launch (350ms→120ms perceived latency)
- Add cache invalidation on filesystem events affecting associated `doc_id` entries for consistency
- Create covering indices: `fts_idx(doc_id, rank)` and `embeddings_idx(chunk_id, cluster_id)`, rewrite large IN-clauses to `JOIN (VALUES ...)` for SQLite optimization
- Configure connection pooling with `r2d2`, `busy_timeout=50ms`, and WAL mode preventing FTS write blocking
- Maintain query latency histogram: auto-lower ANN `efSearch` if 95th percentile >600ms trading recall for speed
- Implement prepared-statement caching (`rusqlite::CachedStatement`) for top 10 read queries eliminating query-plan compile overhead
- Add `/health/search` IPC endpoint returning P50/P95 latencies for UI "search slow" toast notifications

**Day 5-7: System Resource Management**
- Implement adaptive resource allocation with `mach_host_statistics64` sampling every 5s updating `ResourceState` struct broadcast to worker pools
- Create memory pressure management: evict oldest inference/embedding cache entries and unload inactive models via `llama_cpp_unload` when RAM usage ≥70%
- Add thermal management using `IOReport` CPU package temperature monitoring: throttle background tasks if temp >85°C or battery <30% on AC disconnect
- Implement token-bucket rate limiting for OCR/ANN threads balancing burst completion with thermal envelope constraints
- Create pause/resume system: tag long-running tasks with `tokio::sync::watch` cancellation tokens triggered by `kIOMessageSystemWillSleep` and power management callbacks
- Build comprehensive monitoring: persist 1-minute aggregates `{cpu%, ram_mb, gpu%, tokens/s, io_read_MB/s}` in `perf_metrics` table (7-day TTL) with SLA breach detection
- Add optional Prometheus export on `localhost:8123/metrics` for power user Grafana dashboards
- Create performance overlay (⌥⌘P) with live CPU/GPU/memory bars for debugging and bug reports
- Use libdispatch QoS (`QOS_CLASS_UTILITY`) for automatic macOS background task deprioritization
- Set model warm-cache duration to 30min with automatic unloading on idle timeout for RAM management

### Phase 4: Frontend Development (Weeks 10-12)

#### Week 10: Core UI Components

**Day 1-2: Main Search Interface**
- Design reactive search bar with streaming suggestions using dual debouncing: `requestAnimationFrame` + 120ms delay preventing >8 suggestion rounds/sec and DOM thrashing
- Implement virtualized query history dropdown (Solid Virt or custom `for: each`) for >200 rows preventing WebKit list diff main thread spikes
- Create faceted filter panel with `createStore({ filters: ... })` for structural diff optimization, avoiding dozens of separate signal updates
- Build windowed infinite scroll with intersection observers: maintain ±50 results in DOM, flush off-screen nodes preventing 300MB+ heap snapshots in Tauri WebView
- Add keyboard navigation: ⌘K for search focus, ⌘/⌥ ↑/↓ for suggestion cycling with `@tauri-apps/plugin-global-shortcut` preparation
- Use Tailwind `sticky + backdrop-blur` for native-feeling search bar with smooth suggestions overlay
- Implement incremental metadata loading: skeleton cards first → lazy-fetch thumbnails via `Promise.allSettled` reducing perceived load by ~200ms

**Day 3-4: Rich Result Previews**
- Implement thumbnail generation: raster PDF first pages to WebP during indexing, store blob paths in SQLite to eliminate filesystem seek latency
- Create syntax highlighting optimization: pre-tokenize first 500 LOC using `tree-sitter` in Rust, send ready-to-highlight HTML snippets reducing Prism.js main thread cost to near-zero
- Build rich-text preview system: transform DOCX/RTF to HTML fragments during indexing, cache as `.html` files, display in sandboxed `<iframe sandbox>` preventing CSS bleed
- Add audio/video preview with `controlsList="nodownload"` for privacy, implement lazy waveform rendering using `wavesurfer.js` (decode on hover only)
- Create SVG spritesheet for 50+ file-type icons reducing 60KB compared to icon fonts
- Implement IndexedDB thumbnail caching (browser cache headers ignored in Tauri WebView)
- Add server-side Markdown rendering: use `pulldown-cmark` in Rust for HTML generation, stream to UI cutting initial paint by 70%

**Day 5-7: Interactive Features**
- Implement drag-and-drop with macOS sandbox compatibility: declare `NSFilesAndFoldersUsageDescription` in `Info.plist`, queue dropped files into existing ingest daemon with real-time progress toasts
- Create efficient batch selection using `Set` in Solid signal with `classList={{ "ring-2 ring-brand": selected.has(id) }}` for boolean-only diffing preventing re-render storms
- Build export functionality: prioritize CSV/JSON sync operations, implement PDF export via `print-to-PDF` in off-screen WebView or `printpdf` crate with progress streaming
- Add responsive design with filter sidebar collapse to bottom sheet at `window.innerWidth < 950px` for portrait mode support
- Implement command pattern undo stack: store inverse operation closures, expose ⌘Z + 10s "Undo" snackbar for destructive action recovery
- Add user preference persistence in `localStorage` keyed by app version for sort/filter state restoration
- Integrate `@solid-primitives/clipboard` for quick copy-path actions without custom implementation
- Create asynchronous folder enumeration using `tokio::fs::read_dir` with incremental count display preventing main thread blocking

#### Week 11: Advanced UI Features

**Day 1-2: File Preview System**
- Implement progressive tile rendering using `requestAnimationFrame`: decode only visible tiles at target zoom level preventing WebView memory saturation on high-DPI images
- Create virtualized thumbnail strip with intersection observers for >1000-page documents, pre-render low-res thumbnails during indexing to avoid UI freezing
- Build annotation system with page-relative `{x, y, w, h}` coordinates stored in SQLite table, apply as separate `<canvas>` overlay avoiding raster mutation
- Add side-by-side comparison with Rust diff pass: compute changed pages/diff hunks server-side, UI scrolls to matching anchors for performance
- Implement native pinch-zoom support leveraging macOS trackpad gestures, add `⌥← / ⌥→` keyboard shortcuts for page navigation
- Create breadcrumb navigation stack `{doc_id, scroll_pos}` in Solid store enabling one-animation *Back* transitions for nested previews
- Store `lastVisitedPage` per document in `localStorage` for session restoration across app launches

**Day 3-4: Timeline and Visualization**
- Implement server-side event binning: aggregate thousands of events into hourly/daily buckets, down-sample to ≤5000 nodes preventing SVG performance issues
- Create headless date-picker using `@thisbeyond/solid-datepicker` with focus trapping for accessibility, avoiding unreliable Tauri WebView `<input type="date">` polyfills
- Build relationship graphs with server-side ForceAtlas2 layout (Rust port): send fixed coordinates to UI preventing CPU-intensive client-side force-directed re-layouts
- Add search analytics charts using static Canvas (uPlot, Chart.js) avoiding WebGL GPU contention with Metal LLM inference
- Implement timeline zoom levels (day ↔ week ↔ month) via API re-querying rather than client-side re-binning for performance
- Cache `timeline?start=&end=` payloads in IndexedDB enabling instant back-and-forth navigation
- Create server-side tag cloud computation: pre-calculate top 100 tags with counts, use CSS sizing (`font-size: calc(0.8rem + 0.2rem * weight)`) eliminating JS scaling loops

**Day 5-7: Advanced Workflow Features**
- Build collections and tagging with query AST extension `IN_COLLECTION(x)` and indexed `doc_collection(doc_id, collection_id)` table preventing N+1 joins
- Implement search alerts using Tauri tray process + macOS LaunchAgent for background operation surviving app quit (opt-in feature)
- Create bulk operations with Rust worker processes: stream 500MB+ moves/deletes via IPC events, implement optimistic UI updates with failure rollback
- Add workflow automation with finite rule tree compilation: evaluate "if file.type = PDF AND contains 'invoice' → add tag 'Finance'" once per doc post-indexing rather than per query
- Build collections UI with drag-select → C key → modal workflow, persist manual ordering with `ordinal` column for user-controlled sorting
- Implement Focus mode awareness: query `defaults -currentHost read -g FocusStatus` and queue notifications until mode ends
- Create bulk diff view leveraging Day 1-2 compare engine: process multiple doc pairs sequentially with diff buffer recycling
- Add rule compilation system preventing combinatorial explosion of auto-tag conditions through optimized evaluation trees

#### Week 12: User Experience Polish

**Day 1-2: Keyboard Shortcuts & Power User Features**
- Implement customizable key bindings with `{action: Shortcut}` JSON persistence in `settings.db`, merge on version upgrades preserving user preferences while adding new action defaults
- Add global shortcuts using `@tauri-apps/plugin-global-shortcut` with graceful conflict handling: wrap register calls in `Result`, expose conflicts in *Preferences ▶ Shortcuts* panel
- Create command palette with client-side fuzzy matching using `fuse.js` for 0ms latency, pre-index commands locally eliminating IPC round-trips for thousands of commands
- Implement scoped Vim navigation mode: dedicated `div` with `contentEditable=false`, bail on ⌘/Ctrl modifiers preserving standard copy/paste operations
- Build auto-generated cheat sheet modal (`?` key) from key-map JSON maintaining single source of truth for shortcut documentation
- Add search-syntax snippets as ghost text (`type:pdf last:30d`) in search bar enabling passive power filter learning
- Create trackpad gesture support: two-finger swipe left/right for result page navigation with `requestAnimationFrame` throttling preventing jank

**Day 3-4: Search History & Personalization**
- Implement complete query reconstruction persisting `query_ast`, `ranking_config_hash`, `filters_json` per history row avoiding expensive recomputation
- Create tiered history retention: full detail 90 days → summary stats 365 days → purge, with automatic rotation scheduling
- Build personalized dashboard with pre-materialized `freq_view`: nightly computation writing JSON to disk for instant UI rendering avoiding slow SQLite GROUP BY on launch
- Add *Pause History* toggle beside search bar: when disabled, skip all writes rather than marking rows private for true privacy protection
- Create bookmark system with `bookmarks(doc_id, tag_id, note, created_at)` table, expose tag chips in autocomplete feeding discovery workflows
- Implement suggestion engine using offline Apriori rules in Rust: "You often open invoices on the 1st—want a Smart-Filter?" with cheap computation
- Add "Forget all data on quit" option for dev/beta users enabling throw-away testing sessions

**Day 5-7: Onboarding & Accessibility**
- Create accessible interactive tours using ARIA-live polite regions with Escape dismissal at any step, avoiding focus-stealing issues in WebView environments
- Implement deferred tooltip mounting with `setTimeout(0)` post-hydration preventing Solid hydration mismatches and portal rendering conflicts
- Build comprehensive i18n infrastructure using runtime-loaded JSON bundles (`i18next-solid`) wrapping all static strings to prevent late localization complexity
- Add robust dark/light mode support: respect `prefers-color-scheme` with manual override persistence, include `<meta name="color-scheme" content="dark light">` for correct form control rendering
- Generate build-time focus order maps (`data-tour-order`) enabling auto-walking tour elements without hard-coded selectors
- Implement screen-reader announcements: `aria-live="assertive"` for "X results loaded" on each search query with proper live region management
- Ship initial locale support for en, es, de with complete infrastructure even if partially translated, establishing foundation for launch
- Add contextual help system with tooltips and guided tutorials accessible via keyboard navigation and screen reader compatibility
- Create "Send feedback" functionality pre-filling emails with device info + build hash for rich bug reporting during beta testing

### Phase 5: Integration & Platform Features (Weeks 13-15)

#### Week 13: Email Integration

**Day 1-2: macOS Mail.app Integration**
- Implement MailKit plugin architecture (macOS 13+) for structured JSON events with proper privacy settings respect, fallback to AppleScript for ≤macOS 12 with TCC prompt handling
- Navigate Mail.app sandbox protection: require Full-Disk Access entitlement `com.apple.security.files.user-selected.read-write`, provide guided "Privacy ▶ Full Disk Access" user flow
- Create streaming EMLX parser using `mailparse` crate: handle MIME encoding + separate Info-plist with flags (junk, replied, forwarded), avoid loading >25MB attachments into RAM
- Implement HTML tracker protection: strip `<img src=http(s)>` on ingest or set `referrerpolicy="no-referrer"` for preview iframe security
- Use read-only SQLite queries on Envelope Index: `sqlite3 -readonly 'select rowid, mailboxes.date_sent, flags …'` for delta sync avoiding mtime scanning
- Persist `email_source_id` triple (account_uuid, mailbox_id, message_id) enabling trivial cross-account deduplication
- Convert HTML → Markdown using `pulldown-cmark` during ingest for unified text chunker reuse and embedded webview avoidance

**Day 3-4: Thunderbird Integration**
- Handle dual storage formats: streaming mbox parser with custom `BufReader` keeping <20MB resident for 4GB+ files, Maildir inode+mtime scanning watching for hard links across `cur/` and `new/`
- Implement character encoding detection using `encoding_rs` to sniff and transcode mixed ISO-8859-* to UTF-8 preventing cosine similarity degradation
- Create IMAP connection pooling with exponential back-off: disconnect after 5min idle on battery, respect Gmail's 15 connection limit per account
- Add intelligent attachment handling: decode only indexable types (PDF, DOCX, images), peek ZIP filenames without deep extraction avoiding ZIP bombs
- Store IMAP UIDVALIDITY + UID pairs enabling incremental sync via `UID SEARCH UID > last_seen` queries
- Use `mbox-cat` Rust crate for zero-copy message iterators with mmap on macOS for fast reads of large mbox files
- Build "Link account via IMAP/POP3" wizard: server/port/SSL configuration with app-specific password documentation for Gmail/iCloud

**Day 5-7: Email-Specific Features**
- Build contact graph management: canonicalize contacts in `contact(id, email, name)` table mapping aliases (`john@corp`, `john.doe@corp`) during ingest to control graph explosion
- Implement robust thread reconstruction: use `In-Reply-To` headers with fallback to `(References, Subject, time_diff < 36h)` heuristic, persist SHA256-generated `thread_id` from earliest message_id
- Create email body NLP optimization: strip signatures below `-- ` and quoted replies matching "On DATE, NAME wrote:" regex before chunking for focused embeddings
- Add privacy/redaction system: mark chunks containing SSN, DOB mm/dd/yyyy, or ≥5 consecutive digits as *sensitive*, require TouchID for full text view, exclude from suggestion snippets
- Implement email-specific query filters: `from:, to:, subject:, has:attachment, in:thread:` prefixes in query grammar with autocomplete surfacing
- Create conversation flow view reusing side-by-side diff component: left=previous message, right=current reply with `diff-match-patch` highlighting
- Add smart notifications: generate tray icon badge for unread emails matching saved searches, suppress Focus mode toasts
- Build comprehensive email search with privacy controls and automatic redaction for sensitive content

#### Week 14: Browser Integration

**Day 1-2: Browser History & Bookmarks**
- Access browser SQLite databases with proper sandboxing: Safari `~/Library/Safari/History.db`, Chrome `~/Library/Application Support/Google/Chrome/Default/History`, Firefox `places.sqlite` requiring Full Disk Access entitlement
- Implement database locking workaround: use SQLite readonly mode with retry strategy and backoff, copy to temp location for reading while browsers are running
- Create timestamp normalization: convert browser-specific epochs (Chrome WebKit microseconds since 1601, Safari Core Data timestamps) to UTC during ingest preventing temporal search bugs
- Respect privacy mode exclusions: explicitly exclude Incognito/Private browsing sessions without guessing or inference, honor default browser privacy behavior
- Extract favicon and page title during ingest for rich search result display with visual context
- Use `uuid_v5(URL)` as canonical bookmark ID preventing duplication across imports and browser switches
- Link downloads to browsing events: match file paths + timestamps ±10s building session trails ("Saw article, downloaded dataset")

**Day 3-4: Web Content Processing**
- Implement tiered page extraction: `html2text` + `select` crate for basic DOM parsing, escalate to headless Chromium (`puppeteer`/`playwright` Node subprocess) for JS-heavy sites (Notion, Medium)
- Create screenshot capture using `puppeteer` or `chromium --headless --screenshot`: persist images with SHA256(URL+timestamp) filenames for deduplication, store 320px width thumbnails balancing disk usage with preview utility
- Add intelligent change detection: hash normalized DOM text (not full HTML) ignoring tracking scripts/ads, use `difference` crate for diffs, notify on >15% content changes over time
- Surface "This page changed since your last view" in search previews for research and reference material tracking
- Implement selective archiving using `wget --mirror` for offline retrieval of disappearing or paywalled content
- Analyze internal/external link relationships building cross-document connections: "This PDF links to bookmarked article from last June"
- Create webpage content extraction using headless browser or HTML parsing with fallback strategies for dynamic content

**Day 5-7: Browser-Specific Search Features**
- Design `search_scope` enum (`Bookmarks`, `History`, `Downloads`, `WebpageArchives`, `Auto`) with fast route-switching in query parser AST for custom search modes
- Implement URL pattern recognition for automatic tagging: `github.com/<user>/<repo>` → Project, `docs.google.com/` → Document, `*.pdf` → Reference material enabling intelligent categorization
- Add website categorization using Open Source datasets (Common Crawl domains, Mozilla categories) with domain regex heuristics fallback avoiding heavy ML overhead
- Create embedding-based bookmark auto-tagging: apply BERTopic or KMeans clustering suggesting tags like "AI Research", "Health", "Shopping" based on content analysis
- Build smart auto-suggestions: surface queries like "Recent pages from Medium" or "Downloaded PDFs from last week" based on user typing patterns
- Implement bulk bookmark tagging via clustering interface similar to Raindrop.io for efficient organization
- Add visual browsing analytics: pie charts or tag clouds showing topic breakdown for digital mindfulness insights
- Create privacy controls for web browsing data with granular exclusions (exclude Facebook, YouTube) and opt-in only access

#### Week 15: System Integration

**Day 1-2: System Tray & Global Access**
- Implement native system tray using NSStatusItem on main AppKit thread via `objc::runtime` and `tao`'s main-thread channel preventing silent failures in Tauri environment
- Create robust global hotkey registration with `try_register()` loop handling conflicts with Apple system shortcuts, surface conflicts in *Preferences ▶ Shortcuts* for user resolution
- Build mini search window using borderless `NSPanel` with `level = .floating` and `collectionBehavior = .canJoinAllSpaces` preventing desktop switching flicker
- Implement screenshot/text capture with `CGDisplayStream` handling Screen Recording permission prompts on Ventura+, provide guided two-step flow with cached permission status
- Add color-coded tray status indicators: green=idle, yellow=indexing, red=paused (battery/thermal) for visual system state feedback
- Create quick-capture functionality: ⌥Esc for selected text via `pbpaste`, store to clipboard DB with instant indexing queue
- Use `NSUserNotification` for lightweight toasts, escalate to `UNUserNotificationCenter` only for action buttons (macOS 14+)

**Day 3-4: macOS System Integration**
- Create Spotlight plugin using Obj-C/Swift `mdimporter` with `com.apple.security.files.user-selected.read-write` entitlement, expose `kMDItemContentType = "ai.privateSearch"` calling back to Rust binary via XPC for snippet retrieval
- Implement Services menu integration: register app service in Info.plist running in separate process, keep Service lightweight dumping selected text to temp file with IPC to main app
- Generate Finder smart folders via `.savedSearch` files pointing to `x-aisearch://query=<base64_ast>`, provide command palette "Create Smart Folder from search" action
- Build Alfred/LaunchBar workflow: package script filter calling `aisearch --alfred-json '{query}'` returning Alfred JSON, cache results in `~/Library/Caches/AISearch/alfred.json` for <100ms latency
- Add Shortcuts app integration via XPC dictionary: `Run Query`, `Summarise URL`, `Index File` using `AppIntents` (Swift wrapper) or `NSXPCConnection` (Rust fallback)
- Pre-register `x-aisearch://` URL scheme enabling Spotlight and Quick Actions deep-linking to specific searches
- Provide command-line symlink: `ln -s /Applications/AISearch.app/Contents/MacOS/aisearch /usr/local/bin/aisearch` for power user access

**Day 5-7: Advanced System Features**
- Implement power-efficient clipboard history: use `CGEventTapCreate` with `kCGEventOtherMouseDown` mask instead of 300ms polling preventing ~1%/h battery drain, activate only when clipboard viewer open
- Create system event correlation using `EventKit` (user grant required) and `LaunchServices LSApplicationWorkspace` + `FSEvents` for app focus, merge via `(timestamp, app, file)` table indexing only ≥5s focus time
- Build encrypted backup system: `tar + zstd` archive wrapped with `age` (Rust), store AES-256 recipient key in macOS keychain with TouchID restore prompts
- Add crash reporting with `Sentry`/`Bugsnag` using `before_send` hook stripping PII from filenames/queries, local breadcrumbs only with opt-in upload
- Trigger backups only on AC power + idle + Wi-Fi (non-metered SSID) respecting system power state
- Schedule automatic optimization via `pmset schedule` during PowerNap windows: `aisearch self-optimize --vacuum --reindex-delta` running even with lid closed
- Implement crash auto-recovery: detect `crash.flag` on launch, reopen last query in Safe Mode (indexing paused, no models loaded) for data retrieval access

### Phase 6: Security & Privacy (Weeks 16-17)

#### Week 16: Data Protection

**Day 1-2: Local Encryption Implementation**
- Implement Argon2id key derivation (ops-limit 2¹⁹, mem-limit 64MB) with 16-byte salt and stored parameters in header, avoiding PBKDF2 performance/security trade-offs
- Create hierarchical key architecture: derive 32-byte master encryption key (MEK) once per login sealed in Keychain, envelope-encrypt per-file FEKs (file encryption keys) with MEK eliminating Keychain round-trips per file
- Configure SQLCipher with 256-bit raw key mode and `PRAGMA cipher_page_size = 8192` minimizing write-amplification, ensure FTS5 compilation with `SQLITE_HAS_CODEC` for search compatibility
- Implement secure memory handling: `sodium_memzero` on plaintext buffers after vectorization preventing jemalloc arena memory scraping during LLM embedding generation
- Cache MEK in locked memory (`mlock`) with generation counter bumped on TouchID re-auth for stale reference invalidation
- Add file format versioning: prefix encrypted files with `b"AIS1"` magic + header JSON for future-proof format evolution
- Maintain unencrypted public metadata table (doc_id, title, mime, size) enabling fast FTS list views without secret material access

**Day 3-4: Authentication & Access Control**
- Implement LocalAuthentication with per-action `LAContext` recreation preventing "context invalidated" crashes from window focus loss, cache only for single actions
- Create robust session timeout using `DistributedNotificationCenter` listening for `com.apple.screenIsLocked`/`...Unlocked` expiring sessions immediately on lid close avoiding silent access during sleep
- Build multi-user support with APFS Secure Enclave per-user keys: separate stores under `~/Library/Application Support/aisearch/<uid>`, bind Keychain items to `kSecAttrAccessibleWhenUnlockedThisDeviceOnly` preventing Fast-User-Switching leaks
- Add backup encryption with salt + version + KDF params preventing silent data corruption on fresh machine restores with user-exported keys
- Provide "Require auth after..." slider (0, 5, 15 min, immediate) adjusting session generation TTL for user-controlled security/convenience balance
- Implement CLI `aisearch lock` command for script integration and manual security enforcement
- Add optional passphrase fallback for biometric failures (cold fingers, external keyboards) maintaining access reliability

**Day 5-7: Security Monitoring & Audit**
- Implement document-granularity audit logging: (timestamp, action, doc_id, actor_session) with daily rotation into LZ4-compressed files encrypted with MEK preventing log explosion
- Create intrusion detection using token-bucket algorithm: >N sensitive docs in M seconds or failed decryption attempts triggering immediate re-auth + alert for on-device attacker protection
- Add crypto-erasure secure deletion: encrypt FEK with ephemeral key, secure delete via FEK + header wipe making data cryptographically unrecoverable (APFS snapshot-resistant)
- Implement signed auto-updates: Developer ID + Hardened Runtime + timestamp with SHA-256 and Ed25519 signature verification in update JSON manifest, fallback to last-good build on code-signature failure
- Build "Security Health" dashboard: green=all good, yellow=pending update, red=audit log >1GB or unresolved intrusion flag for visual security status
- Add local notification on new device first decryption enabling instant detection of unexpected restores without remote dependencies
- Create `export-audit` command redacting doc titles while preserving timestamps for compliance reporting without content revelation

#### Week 17: Privacy Controls

**Day 1-2: Granular Privacy Settings**
- Build privacy control panel with granular settings for data types and locations
- Implement directory and file type exclusion lists with pattern matching
- Create automatic privacy detection for sensitive file types (tax documents, medical records)
- Add content-based privacy filtering using keyword and pattern detection
- Implement privacy impact assessment for new features and data sources

**Day 3-4: Data Management & Retention**
- Create automatic data retention policies with configurable timeframes
- Implement data minimization with automatic cleanup of unused indexes and caches
- Add data portability features with encrypted export functionality
- Create privacy-preserving analytics that don't expose personal information
- Implement right-to-be-forgotten functionality with comprehensive data removal

**Day 5-7: Privacy Dashboard & Transparency**
- Build comprehensive privacy dashboard showing all indexed data and access patterns
- Create data usage visualization and statistics with privacy-preserving aggregation
- Implement transparency reports showing what data is collected and how it's used
- Add privacy preference migration and backup/restore functionality
- Create privacy education and best practices guidance within the application

### Phase 7: Testing & Quality Assurance (Weeks 18-19)

#### Week 18: Automated Testing

**Day 1-2: Unit Testing Framework**
- Set up comprehensive unit testing using `cargo test` with custom test harnesses
- Create mock implementations for file system operations and external dependencies
- Implement property-based testing using `proptest` for search algorithms and data structures
- Add performance regression testing with automated benchmarking
- Create test data generation for various file types and content scenarios

**Day 3-4: Integration Testing**
- Build end-to-end testing framework covering full indexing and search workflows
- Create test environments with controlled datasets and predictable outcomes
- Implement API testing for all Tauri commands with edge case coverage
- Add database migration testing and schema validation
- Create cross-platform testing setup for future Windows/Linux support

**Day 5-7: Performance & Load Testing**
- Implement stress testing with large datasets (100k+ documents) and concurrent operations
- Create memory leak detection and resource usage monitoring during testing
- Add performance profiling integration with automated bottleneck detection
- Implement search accuracy testing with ground truth datasets and relevance scoring
- Create automated UI testing using Tauri's testing capabilities with realistic user workflows

#### Week 19: Manual Testing & Quality Assurance

**Day 1-2: Comprehensive Manual Testing**
- Conduct systematic testing across all supported file types with various content scenarios
- Test edge cases including corrupted files, permission issues, and disk space limitations
- Perform usability testing with realistic user workflows and complex search scenarios
- Test accessibility features with screen readers and keyboard-only navigation
- Validate privacy controls and data isolation with sensitive document handling

**Day 3-4: Performance Validation**
- Test application performance on different hardware configurations (various RAM, CPU, GPU)
- Validate search accuracy with domain-specific documents and technical content
- Test system resource usage under various load conditions and extended usage periods
- Perform thermal and battery impact testing on laptop configurations
- Validate backup and restore functionality with large datasets

**Day 5-7: Security & Privacy Testing**
- Conduct security testing including authentication bypass attempts and data access validation
- Test encryption implementation with key rotation and recovery scenarios
- Perform privacy validation ensuring no data leakage or unauthorized access
- Test crash recovery and data corruption handling with various failure scenarios
- Validate error handling and user experience during network connectivity issues and system limitations

### Phase 8: Deployment & Distribution (Weeks 20-21)

#### Week 20: Build System & Distribution

**Day 1-2: Automated Build Pipeline**
- Set up GitHub Actions workflow for automated building, testing, and artifact generation
- Implement cross-compilation support for different macOS architectures (Intel, Apple Silicon)
- Create reproducible builds with dependency locking and checksum verification
- Add automated testing integration with build pipeline and failure notification
- Implement build artifact signing and notarization for macOS security requirements

**Day 3-4: Code Signing & Notarization**
- Configure Apple Developer account and certificates for code signing
- Implement automated code signing in build pipeline with secure credential management
- Set up notarization process for macOS Gatekeeper compatibility
- Create installer package using `create-dmg` or native macOS installer tools
- Add automatic update framework using Tauri's updater with signature verification

**Day 5-7: Distribution Infrastructure**
- Set up release management with semantic versioning and changelog generation
- Create download infrastructure with CDN distribution and analytics
- Implement crash reporting using `sentry` or similar with privacy-preserving data collection
- Add telemetry collection (opt-in) for usage analytics and performance monitoring
- Create beta testing distribution channel with automatic update management

#### Week 21: Launch Preparation

**Day 1-2: Documentation & Support**
- Create comprehensive user manual with screenshots and video tutorials
- Build developer documentation for future contributors and API reference
- Implement in-app help system with contextual assistance and troubleshooting guides
- Create FAQ and troubleshooting documentation based on beta testing feedback
- Set up community support channels and feedback collection systems

**Day 3-4: Marketing & User Education**
- Build product landing page with feature demonstrations and privacy explanations
- Create marketing materials emphasizing privacy benefits and use cases
- Develop case studies for target user personas (lawyers, journalists, researchers)
- Implement user onboarding sequence with progressive feature introduction
- Create social media and community outreach strategy

**Day 5-7: Launch Strategy & Monitoring**
- Plan phased rollout strategy starting with limited beta users
- Set up monitoring dashboards for application performance, user engagement, and error tracking
- Create incident response plan for critical issues and rapid hotfix deployment
- Implement user feedback collection and prioritization system
- Prepare customer support processes and escalation procedures

### Phase 9: Post-Launch & Iteration (Ongoing)

#### Immediate Post-Launch (Weeks 22-24)

**Week 22: User Feedback Integration**
- Monitor user feedback channels and prioritize feature requests based on usage data
- Implement rapid bug fixing workflow with automated testing and deployment
- Create user survey system for satisfaction measurement and feature validation
- Add analytics dashboard for tracking key performance indicators and user behavior
- Implement A/B testing framework for UI/UX improvements

**Week 23: Performance Optimization**
- Analyze real-world usage patterns and optimize for common workflows
- Implement performance improvements based on user hardware configurations
- Add intelligent defaults and configuration optimization based on user behavior
- Create proactive performance monitoring and automatic optimization suggestions
- Implement resource usage optimization for battery life and thermal management

**Week 24: Feature Enhancement**
- Prioritize and implement most-requested features from user feedback
- Add integration with additional applications based on user needs
- Implement workflow automation features for common user tasks
- Create advanced search operators and query syntax based on power user requests
- Add collaboration features for team environments

#### Long-term Roadmap (Months 7-12)

**Cross-Platform Expansion**
- Plan and begin Windows port with DirectML integration and Windows-specific features
- Design Linux support with various distribution compatibility and X11/Wayland support
- Create unified codebase architecture supporting multiple platforms efficiently
- Implement platform-specific optimizations while maintaining feature parity

**Mobile Companion Development**
- Design mobile app architecture as companion to desktop application
- Implement secure synchronization between desktop and mobile with end-to-end encryption
- Create mobile-optimized search interface with voice input and camera integration
- Add location-based search and context awareness for mobile scenarios

**Advanced AI Features**
- Implement document summarization and insight generation using larger language models
- Add conversation-style interaction for complex research tasks
- Create intelligent document organization and automatic tagging based on content analysis
- Implement predictive search and proactive information delivery

**Enterprise & Team Features**
- Design multi-user support with role-based access control and team collaboration
- Implement centralized administration and policy management for organizations
- Add compliance features for regulated industries (HIPAA, GDPR, SOX)
- Create enterprise integration with existing document management and collaboration systems

## Success Metrics

### Technical Metrics
- **Search Performance**: Sub-500ms response time for 95% of queries
- **Indexing Speed**: Process 1000 documents per minute on standard hardware
- **Accuracy**: 95%+ relevance for natural language queries
- **Resource Usage**: <2GB RAM baseline, <10% CPU during background indexing

### User Experience Metrics
- **User Retention**: 80% monthly active users after 3 months
- **Search Success Rate**: 90% of searches result in file access within 30 seconds
- **Feature Adoption**: 70% of users utilize multimodal search within first week
- **Support Tickets**: <5% of users require support contact

### Business Metrics
- **Market Penetration**: 10,000 active users in first 6 months
- **Conversion Rate**: 15% free-to-paid conversion
- **Customer Satisfaction**: 4.5+ star rating across platforms
- **Word-of-Mouth Growth**: 40% of new users from referrals

## Risk Mitigation

### Technical Risks
- **Performance Issues**: Regular profiling and optimization cycles
- **Model Updates**: Backward compatibility testing and gradual rollouts
- **Security Vulnerabilities**: Regular security audits and penetration testing
- **Platform Changes**: Monitor OS updates and maintain compatibility

### Business Risks
- **Competition**: Focus on privacy differentiation and superior user experience
- **Market Changes**: Flexible architecture allowing feature pivots
- **Regulatory**: Proactive compliance with data protection regulations
- **Funding**: Milestone-based development with early revenue generation

This roadmap provides a comprehensive guide for building a privacy-first, on-device AI search engine that meets the demanding requirements of professional users while maintaining the highest standards of security and performance.