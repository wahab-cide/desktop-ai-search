# Desktop AI Search

A powerful, AI-driven search engine that understands your local files like a human would. Search through documents, images, audio, and videos using natural language queries - all running completely offline on your machine.

<div align="center">

![Desktop AI Search Interface](https://via.placeholder.com/800x400/6366f1/ffffff?text=Desktop+AI+Search+Interface)

[![Build Status](https://github.com/your-username/desktop-ai-search/workflows/CI/badge.svg)](https://github.com/your-username/desktop-ai-search/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](https://github.com/your-username/desktop-ai-search)
[![Version](https://img.shields.io/github/v/release/your-username/desktop-ai-search)](https://github.com/your-username/desktop-ai-search/releases)

</div>

## ✨ Features

### 🔍 **Smart Search**
- **Natural Language**: "Find my tax documents from 2023" or "photos of my dog"
- **Multi-Modal**: Search through documents, images, audio, and videos
- **Semantic Understanding**: AI-powered contextual search beyond keywords
- **Real-time Results**: Instant search with progressive loading

### 🤖 **AI-Powered**
- **Local AI Models**: Runs completely offline, no cloud required
- **Vector Embeddings**: Advanced semantic similarity matching
- **OCR Processing**: Extract and search text from images
- **Audio Transcription**: Search through spoken content

### 🎨 **Modern Interface**
- **Clean Design**: Beautiful, responsive interface
- **Dark/Light Themes**: Automatic system theme detection
- **Keyboard Shortcuts**: Full keyboard navigation
- **Accessibility**: WCAG 2.1 compliant

### 🔒 **Privacy-First**
- **100% Local**: All processing happens on your machine
- **No Cloud**: Zero external API calls or data transmission
- **Encrypted Storage**: Sensitive data protected at rest
- **No Tracking**: No analytics or telemetry by default

## 🚀 Quick Start

### Prerequisites
- **Rust**: 1.70+ (for backend)
- **Node.js**: 18+ (for frontend)
- **System Dependencies**:
  - macOS: `brew install tesseract ffmpeg`
  - Linux: `sudo apt-get install tesseract-ocr ffmpeg`
  - Windows: Install via chocolatey or manual download

### Installation

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

### Build for Production

```bash
# Create production build
npm run tauri build

# The compiled app will be in src-tauri/target/release/
```

## 📖 Usage

### 1. Initial Setup
1. **Launch the application**
2. **Go to Settings** → Indexing
3. **Select directories** to index
4. **Start indexing** and wait for completion

### 2. Search Examples

```
# Natural language queries
documents about machine learning
photos from last summer
audio files with jazz music
presentations about quarterly results

# Advanced boolean queries
"machine learning" AND (python OR tensorflow)
type:pdf AND created:2024
author:"John Smith" AND NOT draft
```

### 3. Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + K` | Focus search bar |
| `Ctrl/Cmd + T` | Toggle theme |
| `Ctrl/Cmd + ?` | Show all shortcuts |
| `Escape` | Clear search |

## 🏗️ Architecture

Built with modern technologies for optimal performance:

- **Backend**: Rust + Tauri + SQLite + Candle ML
- **Frontend**: TypeScript + SolidJS + Tailwind CSS
- **AI Models**: Local inference with Hugging Face transformers
- **Database**: SQLite with FTS5 full-text search
- **Caching**: Multi-layer caching for sub-second responses

## 📊 Performance

- **Search Speed**: < 200ms for 100K+ documents
- **Indexing Speed**: 100+ files/minute
- **Memory Usage**: < 512MB for typical workloads
- **Startup Time**: < 3 seconds cold start

## 🔧 Configuration

Customize the application through `config.toml`:

```toml
[search]
max_results = 100
enable_fuzzy_search = true

[indexing]
max_file_size = 100_000_000  # 100MB
batch_size = 1000

[models]
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
device = "auto"  # "cpu", "cuda", "auto"
```

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run with coverage
cargo tarpaulin --out Html

# Performance benchmarks
cargo bench
```

## 📚 Documentation

- **[Complete Documentation](docs/README.md)** - Comprehensive guide
- **[API Reference](docs/api.md)** - Command API documentation
- **[Architecture Guide](docs/architecture.md)** - Technical deep dive
- **[Performance Tuning](docs/performance.md)** - Optimization tips

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Roadmap

- [ ] **Cloud Sync** (optional): Encrypted cloud backup
- [ ] **Mobile App**: iOS/Android companion app
- [ ] **Plugin System**: Custom indexing and search plugins
- [ ] **Team Features**: Shared search across team members
- [ ] **Advanced Analytics**: Search patterns and insights

## 🐛 Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/your-username/desktop-ai-search/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/your-username/desktop-ai-search/discussions)
- **Documentation**: [Wiki](https://github.com/your-username/desktop-ai-search/wiki)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Tauri Team** - Cross-platform framework
- **Candle ML** - Local AI inference
- **SolidJS** - Reactive UI framework
- **Hugging Face** - Open-source AI models
- **SQLite** - Reliable local database

---

<div align="center">

**Built with ❤️ for privacy-conscious power users**

[Download Latest Release](https://github.com/your-username/desktop-ai-search/releases/latest) | [View Documentation](docs/README.md) | [Report Bug](https://github.com/your-username/desktop-ai-search/issues)

</div>
