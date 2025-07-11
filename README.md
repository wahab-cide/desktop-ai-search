# Desktop AI Search

A local document search application with AI-powered features.

## Features

- Index documents from local directories
- Full-text search with FTS5
- AI-powered semantic search using embeddings
- Support for PDF, DOCX, TXT, MD, and image files
- OCR support for images and scanned documents
- Dark/light theme
- File preview and navigation

## Tech Stack

- Backend: Rust with Tauri
- Frontend: SolidJS with TypeScript
- Database: SQLite with FTS5
- AI: Local LLM and embedding models

## Development

### Prerequisites

- Rust (latest stable)
- Node.js (v16+)
- Cargo and npm/pnpm

### Setup

1. Clone the repository
2. Install frontend dependencies: `cd frontend && npm install`
3. Run development server: `cargo tauri dev`

### Build

Run `cargo tauri build` to create a production build.

## Usage

1. Launch the application
2. Click the settings icon to access the indexing panel
3. Select a directory to index
4. Use the search bar to find documents
5. Click on results to open files or show in folder

## Configuration

Copy `config.example.toml` to your app data directory and modify as needed.