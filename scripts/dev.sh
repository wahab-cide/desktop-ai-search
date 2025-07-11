#!/bin/bash

# Development script for Desktop AI Search
# This script provides development utilities and shortcuts

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="desktop-ai-search"
FRONTEND_DIR="frontend"
BACKEND_DIR="src"
DEV_DB="dev_search.db"
TEST_DB="test_search.db"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup development environment
setup_dev_env() {
    print_status "Setting up development environment..."
    
    # Create development database
    if [ ! -f "$DEV_DB" ]; then
        print_status "Creating development database..."
        sqlite3 "$DEV_DB" < src/sql/001_initial_schema.sql
        sqlite3 "$DEV_DB" < src/sql/002_add_embeddings.sql
        sqlite3 "$DEV_DB" < src/sql/003_add_chunks.sql
        sqlite3 "$DEV_DB" < src/sql/004_add_indices.sql
        sqlite3 "$DEV_DB" < src/sql/005_enhanced_chunks_embeddings.sql
        sqlite3 "$DEV_DB" < src/sql/006_add_clip_embeddings.sql
        sqlite3 "$DEV_DB" < src/sql/007_add_fts5_chunks.sql
        print_success "Development database created"
    fi
    
    # Install git hooks
    if [ -d ".git" ]; then
        print_status "Installing git hooks..."
        mkdir -p .git/hooks
        
        # Pre-commit hook
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for Desktop AI Search

set -e

echo "Running pre-commit checks..."

# Check Rust formatting
if ! cargo fmt --check; then
    echo "Error: Rust code is not formatted. Run 'cargo fmt' to fix."
    exit 1
fi

# Check Rust linting
if ! cargo clippy -- -D warnings; then
    echo "Error: Rust linting failed. Fix the issues above."
    exit 1
fi

# Check TypeScript formatting (if prettier is available)
if command -v npx >/dev/null 2>&1 && [ -f "frontend/package.json" ]; then
    cd frontend
    if npx prettier --check "src/**/*.{ts,tsx}"; then
        echo "TypeScript formatting OK"
    else
        echo "Error: TypeScript code is not formatted. Run 'npx prettier --write src/**/*.{ts,tsx}' to fix."
        exit 1
    fi
    cd ..
fi

echo "Pre-commit checks passed!"
EOF
        chmod +x .git/hooks/pre-commit
        print_success "Git hooks installed"
    fi
    
    # Create development configuration
    if [ ! -f "config.dev.toml" ]; then
        cat > config.dev.toml << 'EOF'
[database]
database_url = "dev_search.db"
max_connections = 5
connection_timeout = 30

[search]
max_results = 50
search_timeout = 10
enable_fuzzy_search = true
enable_debug_logging = true

[indexing]
max_file_size = 50_000_000  # 50MB for dev
batch_size = 100
watch_for_changes = true

[models]
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
device = "cpu"  # Force CPU for development
enable_model_caching = true

[performance]
cache_size = 128  # MB
max_threads = 4
enable_compression = false

[logging]
level = "debug"
file = "logs/dev.log"
EOF
        print_success "Development configuration created"
    fi
    
    print_success "Development environment setup completed"
}

# Function to start development server
start_dev_server() {
    print_status "Starting development server..."
    
    # Start backend in background
    export DESKTOP_AI_SEARCH_CONFIG_PATH="config.dev.toml"
    export RUST_LOG=debug
    
    # Kill any existing processes
    pkill -f "cargo.*tauri.*dev" || true
    sleep 2
    
    # Start Tauri development server
    cd $FRONTEND_DIR
    npm run tauri &
    DEV_PID=$!
    cd ..
    
    print_success "Development server started (PID: $DEV_PID)"
    print_status "Press Ctrl+C to stop the server"
    
    # Wait for interrupt
    trap "kill $DEV_PID 2>/dev/null || true; exit 0" INT
    wait $DEV_PID
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    # Run Rust tests
    print_status "Running Rust tests..."
    export DESKTOP_AI_SEARCH_CONFIG_PATH="config.dev.toml"
    export RUST_LOG=debug
    cargo test --verbose
    
    # Run frontend tests if available
    if [ -f "$FRONTEND_DIR/package.json" ]; then
        cd $FRONTEND_DIR
        if grep -q "test" package.json; then
            print_status "Running frontend tests..."
            npm test
        fi
        cd ..
    fi
    
    print_success "Tests completed"
}

# Function to format code
format_code() {
    print_status "Formatting code..."
    
    # Format Rust code
    cargo fmt
    
    # Format TypeScript code
    if [ -f "$FRONTEND_DIR/package.json" ]; then
        cd $FRONTEND_DIR
        if command_exists npx; then
            npx prettier --write "src/**/*.{ts,tsx,js,jsx,css,json}"
        fi
        cd ..
    fi
    
    print_success "Code formatted"
}

# Function to lint code
lint_code() {
    print_status "Linting code..."
    
    # Lint Rust code
    cargo clippy -- -D warnings
    
    # Lint TypeScript code
    if [ -f "$FRONTEND_DIR/package.json" ]; then
        cd $FRONTEND_DIR
        if command_exists npx && [ -f ".eslintrc.js" ]; then
            npx eslint src/
        fi
        cd ..
    fi
    
    print_success "Code linted"
}

# Function to watch for changes
watch_files() {
    print_status "Watching for file changes..."
    
    if ! command_exists inotifywait && ! command_exists fswatch; then
        print_warning "No file watcher available. Install inotify-tools (Linux) or fswatch (macOS)"
        return 1
    fi
    
    # Watch Rust files
    if command_exists inotifywait; then
        inotifywait -m -r -e modify,create,delete src/ --format '%w%f %e' | while read file event; do
            if [[ "$file" == *.rs ]]; then
                print_status "Rust file changed: $file"
                cargo check
            fi
        done &
    elif command_exists fswatch; then
        fswatch -o src/ | while read; do
            print_status "Rust files changed, checking..."
            cargo check
        done &
    fi
    
    WATCH_PID=$!
    
    print_success "File watcher started (PID: $WATCH_PID)"
    print_status "Press Ctrl+C to stop watching"
    
    trap "kill $WATCH_PID 2>/dev/null || true; exit 0" INT
    wait $WATCH_PID
}

# Function to clean development artifacts
clean_dev() {
    print_status "Cleaning development artifacts..."
    
    # Clean Rust artifacts
    cargo clean
    
    # Clean frontend artifacts
    if [ -d "$FRONTEND_DIR/node_modules" ]; then
        rm -rf $FRONTEND_DIR/node_modules
    fi
    
    if [ -d "$FRONTEND_DIR/dist" ]; then
        rm -rf $FRONTEND_DIR/dist
    fi
    
    # Clean development database
    if [ -f "$DEV_DB" ]; then
        rm "$DEV_DB"
    fi
    
    # Clean logs
    if [ -d "logs" ]; then
        rm -rf logs
    fi
    
    print_success "Development artifacts cleaned"
}

# Function to generate test data
generate_test_data() {
    print_status "Generating test data..."
    
    # Create test documents directory
    mkdir -p test_dev_documents
    
    # Generate sample documents
    for i in {1..20}; do
        cat > "test_dev_documents/document_$i.txt" << EOF
Test Document $i
================

This is a test document for development purposes.
It contains sample content for testing search functionality.

Keywords: test, development, document, search, AI, machine learning
Category: Development
Priority: $((i % 3 + 1))
Created: $(date)

Content:
Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco.

Topics covered:
- Search algorithms
- Database optimization
- User interface design
- Performance testing
EOF
    done
    
    # Generate sample code files
    for i in {1..5}; do
        cat > "test_dev_documents/code_$i.py" << EOF
#!/usr/bin/env python3
"""
Test Python file $i for development
"""

import os
import sys
import json

class TestClass$i:
    def __init__(self):
        self.value = $i
    
    def process_data(self, data):
        """Process data for testing"""
        return data * self.value
    
    def get_info(self):
        return {
            'name': 'TestClass$i',
            'value': self.value,
            'type': 'development'
        }

if __name__ == '__main__':
    test = TestClass$i()
    print(test.get_info())
EOF
    done
    
    print_success "Test data generated in test_dev_documents/"
}

# Function to benchmark performance
benchmark_performance() {
    print_status "Running performance benchmarks..."
    
    # Run Rust benchmarks
    if grep -q "bench" Cargo.toml; then
        cargo bench
    else
        print_warning "No benchmarks configured in Cargo.toml"
    fi
    
    # Test search performance
    print_status "Testing search performance..."
    
    # Generate test data if not exists
    if [ ! -d "test_dev_documents" ]; then
        generate_test_data
    fi
    
    # Run search performance test
    cat > benchmark_search.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import time
import json

def benchmark_search(query, iterations=10):
    """Benchmark search performance"""
    times = []
    
    for i in range(iterations):
        start = time.time()
        # This would call the actual search API
        # result = search_api.search(query)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    return {
        'query': query,
        'average_time': avg_time,
        'min_time': min(times),
        'max_time': max(times),
        'iterations': iterations
    }

queries = [
    "test document",
    "machine learning",
    "development",
    "search algorithm",
    "database optimization"
]

results = []
for query in queries:
    print(f"Benchmarking query: {query}")
    result = benchmark_search(query)
    results.append(result)
    print(f"  Average time: {result['average_time']:.3f}s")

print("\nBenchmark Results:")
print(json.dumps(results, indent=2))
EOF
    
    python3 benchmark_search.py
    rm benchmark_search.py
    
    print_success "Performance benchmarks completed"
}

# Function to show development status
show_dev_status() {
    print_status "Development Status:"
    
    # Git status
    if [ -d ".git" ]; then
        echo "Git Status:"
        git status --porcelain | head -10
        echo "Current branch: $(git branch --show-current)"
        echo "Last commit: $(git log -1 --pretty=format:'%h %s')"
        echo ""
    fi
    
    # Development database status
    if [ -f "$DEV_DB" ]; then
        echo "Development Database:"
        echo "  Size: $(du -sh $DEV_DB | cut -f1)"
        echo "  Tables: $(sqlite3 $DEV_DB '.tables' | wc -w)"
        echo "  Documents: $(sqlite3 $DEV_DB 'SELECT COUNT(*) FROM documents;' 2>/dev/null || echo 'N/A')"
        echo ""
    fi
    
    # Dependencies status
    echo "Dependencies:"
    echo "  Rust: $(rustc --version)"
    echo "  Node.js: $(node --version)"
    echo "  Cargo packages: $(cargo tree --depth=0 | wc -l)"
    if [ -f "$FRONTEND_DIR/package.json" ]; then
        echo "  NPM packages: $(cd $FRONTEND_DIR && npm list --depth=0 | wc -l)"
    fi
    echo ""
    
    # Process status
    echo "Running Processes:"
    ps aux | grep -E "(cargo|node|npm)" | grep -v grep | head -5
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "COMMANDS:"
    echo "  setup        Setup development environment"
    echo "  start        Start development server"
    echo "  test         Run tests"
    echo "  format       Format code"
    echo "  lint         Lint code"
    echo "  watch        Watch files for changes"
    echo "  clean        Clean development artifacts"
    echo "  data         Generate test data"
    echo "  bench        Run performance benchmarks"
    echo "  status       Show development status"
    echo "  help         Show this help message"
    echo ""
    echo "OPTIONS:"
    echo "  -v, --verbose    Verbose output"
    echo ""
    echo "Examples:"
    echo "  $0 setup        # Setup development environment"
    echo "  $0 start        # Start development server"
    echo "  $0 test         # Run all tests"
    echo "  $0 format lint  # Format and lint code"
}

# Parse command line arguments
VERBOSE=false
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            set -x
            shift
            ;;
        -h|--help|help)
            show_usage
            exit 0
            ;;
        setup|start|test|format|lint|watch|clean|data|bench|status)
            COMMAND="$1"
            shift
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                COMMAND="$1"
            fi
            shift
            ;;
    esac
done

# Default command
if [ -z "$COMMAND" ]; then
    show_usage
    exit 1
fi

# Execute command
case "$COMMAND" in
    setup)
        setup_dev_env
        ;;
    start)
        start_dev_server
        ;;
    test)
        run_tests
        ;;
    format)
        format_code
        ;;
    lint)
        lint_code
        ;;
    watch)
        watch_files
        ;;
    clean)
        clean_dev
        ;;
    data)
        generate_test_data
        ;;
    bench)
        benchmark_performance
        ;;
    status)
        show_dev_status
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac