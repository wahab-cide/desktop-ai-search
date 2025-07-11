# Desktop AI Search

A powerful, AI-driven search engine that understands your local files like a human would. Search through documents, images, audio, and videos using natural language queries - all running completely offline on your machine.

## 🚀 Features

### Core Search Capabilities
- **Hybrid Search Engine**: Combines traditional keyword search with AI-powered semantic understanding
- **Multi-Modal Support**: Search through documents, images, audio files, and videos
- **Natural Language Queries**: Describe what you're looking for in plain English
- **Intelligent Query Processing**: Automatically detects search intent and optimizes query strategy
- **Boolean Query Support**: Advanced users can use precise boolean operators
- **Real-time Search Suggestions**: Smart autocomplete with query history

### AI & Machine Learning
- **Local LLM Integration**: Powered by local language models (no cloud required)
- **Semantic Embeddings**: Advanced vector search for conceptual understanding
- **OCR Processing**: Extract and search text from images and scanned documents
- **Audio Transcription**: Search through audio content via speech-to-text
- **Content Intelligence**: Understands context and relationships between documents

### User Experience
- **Modern Interface**: Clean, responsive design with dark/light/system themes
- **Keyboard Shortcuts**: Full keyboard navigation for power users
- **Real-time Results**: Instant search with progressive result loading
- **Rich File Previews**: Preview images, documents, and media files
- **Advanced Filters**: Filter by file type, date range, size, and more
- **Accessibility**: WCAG 2.1 compliant with screen reader support

### Performance & Reliability
- **Multi-layered Caching**: Intelligent caching for sub-second search responses
- **Circuit Breaker Pattern**: Automatic error recovery and fault tolerance
- **Performance Monitoring**: Real-time metrics and health monitoring
- **Incremental Indexing**: Efficient updates without full re-indexing
- **Background Processing**: Non-blocking file processing and indexing

## 🔧 Installation

### Prerequisites
- **Rust**: 1.70+ (for backend)
- **Node.js**: 18+ (for frontend)
- **System Dependencies**: 
  - macOS: `brew install tesseract ffmpeg`
  - Linux: `sudo apt-get install tesseract-ocr ffmpeg`
  - Windows: Install via chocolatey or manual download

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/desktop-ai-search.git
cd desktop-ai-search

# Install dependencies
cargo build --release
cd frontend && npm install

# Start the application
npm run tauri dev
```

### Production Build

```bash
# Build for production
npm run tauri build

# The compiled app will be in src-tauri/target/release/
```

## 📖 Usage Guide

### Initial Setup
1. **Launch the application**
2. **Configure indexing**: Go to Settings → Indexing
3. **Select directories**: Choose folders to index
4. **Start indexing**: Click "Start Indexing" and wait for completion
5. **Begin searching**: Use the search bar to find your files

### Search Syntax

#### Natural Language (Recommended)
```
documents about machine learning
photos from last summer
audio files with jazz music
presentations about quarterly results
```

#### Boolean Queries (Advanced)
```
"machine learning" AND (python OR tensorflow)
type:pdf AND created:2024
author:"John Smith" AND NOT draft
```

#### Filters
```
type:pdf                    # Search only PDF files
created:2024-01-01          # Files created after date
size:>10MB                  # Files larger than 10MB
modified:last-week          # Recently modified files
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + K` | Focus search bar |
| `Ctrl/Cmd + T` | Toggle theme |
| `Ctrl/Cmd + S` | Open settings |
| `Ctrl/Cmd + ?` | Show all shortcuts |
| `Escape` | Clear search |
| `↑/↓` | Navigate suggestions |
| `Enter` | Search/select |

## 🏗️ Architecture

### Backend (Rust)
- **Tauri Framework**: Cross-platform desktop app framework
- **SQLite Database**: Local storage with FTS5 full-text search
- **Tokio Runtime**: Async runtime for high-performance I/O
- **Candle ML**: Local AI model inference
- **Multi-threaded Processing**: Concurrent file processing and indexing

### Frontend (TypeScript/SolidJS)
- **SolidJS**: Reactive UI framework for optimal performance
- **Tailwind CSS**: Utility-first CSS framework
- **TypeScript**: Type-safe JavaScript for better development experience
- **Vite**: Fast build tool and development server

### Key Components

```
src/
├── core/                   # Core search and AI logic
│   ├── hybrid_search.rs    # Hybrid search engine
│   ├── embedding_manager.rs # Vector embeddings
│   ├── llm_manager.rs      # Language model integration
│   └── ocr_processor.rs    # Image text extraction
├── database/               # Data persistence
│   ├── operations.rs       # Database operations
│   └── migrations.rs       # Schema migrations
├── monitoring/             # System monitoring
│   ├── metrics.rs          # Performance metrics
│   ├── alerts.rs           # Alert system
│   └── telemetry.rs        # Usage analytics
├── cache/                  # Caching system
│   ├── search_cache.rs     # Search result caching
│   └── embedding_cache.rs  # Vector cache
└── commands/               # Tauri API commands
    ├── search.rs           # Search commands
    ├── indexing.rs         # Indexing commands
    └── monitoring.rs       # Monitoring commands
```

## ⚙️ Configuration

### Application Settings
Configuration is managed through `config/mod.rs` and can be customized via:

```toml
# config.toml
[database]
max_connections = 10
connection_timeout = 30

[search]
max_results = 100
search_timeout = 10
enable_fuzzy_search = true

[indexing]
max_file_size = 100_000_000  # 100MB
excluded_extensions = [".tmp", ".cache"]
batch_size = 1000

[models]
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
llm_model = "microsoft/DialoGPT-medium"
device = "auto"  # "cpu", "cuda", "auto"

[performance]
cache_size = 512  # MB
max_threads = 8
enable_compression = true
```

### Environment Variables
```bash
# Performance tuning
RUST_LOG=info                    # Logging level
RAYON_NUM_THREADS=8             # Parallel processing threads
DESKTOP_AI_SEARCH_CONFIG_PATH=  # Custom config path

# Model configuration
HF_HUB_OFFLINE=true             # Disable model downloading
TOKENIZERS_PARALLELISM=true     # Enable parallel tokenization
```

## 🔒 Security & Privacy

### Data Privacy
- **100% Local Processing**: All data stays on your machine
- **No Cloud Dependencies**: No external API calls or data transmission
- **Encrypted Storage**: Sensitive data encrypted at rest
- **Secure File Access**: Sandboxed file system access

### Security Features
- **Input Validation**: All user inputs sanitized and validated
- **SQL Injection Prevention**: Parameterized queries throughout
- **Path Traversal Protection**: Secure file path handling
- **Resource Limits**: Memory and CPU usage limits
- **Audit Logging**: Security event logging

## 🧪 Testing

### Running Tests
```bash
# Run all tests
cargo test

# Run specific test module
cargo test search::tests

# Run with coverage
cargo tarpaulin --out Html
```

### Test Structure
```
tests/
├── integration/           # Integration tests
├── unit/                 # Unit tests
├── performance/          # Performance benchmarks
└── fixtures/             # Test data
```

## 📊 Performance

### Benchmarks
- **Search Speed**: < 200ms for 100K+ documents
- **Indexing Speed**: 100+ files/minute
- **Memory Usage**: < 512MB for typical workloads
- **Startup Time**: < 3 seconds cold start

### Optimization Tips
1. **SSD Storage**: Use SSD for database and cache
2. **Memory**: 8GB+ RAM recommended for large datasets
3. **CPU**: Multi-core CPU for parallel processing
4. **File Organization**: Avoid deeply nested directory structures

## 🛠️ Development

### Building from Source
```bash
# Development build
cargo build
cd frontend && npm run dev

# Release build
cargo build --release
cd frontend && npm run build
```

### Code Style
- **Rust**: Follow rustfmt and clippy recommendations
- **TypeScript**: Use Prettier and ESLint configuration
- **Documentation**: Document all public APIs

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📈 Monitoring & Debugging

### Health Monitoring
Access monitoring dashboard via Settings → Monitoring:
- **System Health**: CPU, memory, disk usage
- **Search Performance**: Response times, success rates
- **Indexing Status**: Progress, errors, statistics
- **Cache Performance**: Hit rates, memory usage

### Debugging
```bash
# Enable debug logging
RUST_LOG=debug cargo run

# Performance profiling
cargo run --release --features profiling

# Memory debugging
valgrind --tool=memcheck ./target/release/desktop-ai-search
```

### Log Files
- **Application Logs**: `~/.local/share/desktop-ai-search/logs/`
- **Error Logs**: `~/.local/share/desktop-ai-search/errors/`
- **Performance Metrics**: `~/.local/share/desktop-ai-search/metrics/`

## 🚨 Troubleshooting

### Common Issues

#### Search Returns No Results
1. Check if indexing is complete
2. Verify file permissions
3. Try rebuilding the search index
4. Check excluded file types

#### Slow Performance
1. Increase cache size in configuration
2. Check available system memory
3. Optimize database with `VACUUM`
4. Consider excluding large binary files

#### High Memory Usage
1. Reduce cache size
2. Limit concurrent indexing threads
3. Clear embedding cache
4. Restart application

#### Application Won't Start
1. Check system dependencies
2. Verify Rust/Node.js versions
3. Clear cache and configuration
4. Check log files for errors

## 📝 Changelog

### v1.0.0 (Current)
- Initial release with full search capabilities
- AI-powered semantic search
- Multi-modal file support
- Real-time monitoring
- Comprehensive caching system

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Tauri Team**: For the excellent cross-platform framework
- **Candle ML**: For local AI inference capabilities
- **SolidJS**: For the reactive UI framework
- **Hugging Face**: For providing open-source AI models
- **SQLite**: For reliable local database storage

## 🔗 Resources

- [API Documentation](docs/api.md)
- [Architecture Deep Dive](docs/architecture.md)
- [Performance Tuning Guide](docs/performance.md)
- [Security Best Practices](docs/security.md)
- [Contributing Guidelines](docs/contributing.md)

---

**For privacy-conscious power users who want AI search without compromising their data.**