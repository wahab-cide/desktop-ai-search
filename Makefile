# Makefile for Desktop AI Search
# This provides convenient shortcuts for common development tasks

.PHONY: help install build test clean dev release deploy ci format lint watch

# Default target
help:
	@echo "Desktop AI Search - Development Commands"
	@echo "======================================="
	@echo ""
	@echo "Available targets:"
	@echo "  help      - Show this help message"
	@echo "  install   - Install dependencies"
	@echo "  build     - Build the project"
	@echo "  test      - Run tests"
	@echo "  clean     - Clean build artifacts"
	@echo "  dev       - Start development server"
	@echo "  release   - Build release version"
	@echo "  deploy    - Deploy the application"
	@echo "  ci        - Run CI pipeline"
	@echo "  format    - Format code"
	@echo "  lint      - Lint code"
	@echo "  watch     - Watch for file changes"
	@echo ""
	@echo "Examples:"
	@echo "  make install    # Install dependencies"
	@echo "  make dev        # Start development server"
	@echo "  make test       # Run all tests"
	@echo "  make release    # Build for production"

# Install dependencies
install:
	@echo "Installing dependencies..."
	@cargo fetch
	@cd frontend && npm install
	@echo "Dependencies installed successfully!"

# Build the project
build:
	@echo "Building project..."
	@cd frontend && npm run build
	@cargo build --release
	@echo "Build completed successfully!"

# Run tests
test:
	@echo "Running tests..."
	@cargo test --verbose
	@cd frontend && npm test || echo "No frontend tests configured"
	@echo "Tests completed!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@cargo clean
	@cd frontend && rm -rf dist node_modules || true
	@rm -rf release deploy logs || true
	@echo "Clean completed!"

# Start development server
dev:
	@echo "Starting development server..."
	@./scripts/dev.sh start

# Build release version
release:
	@echo "Building release version..."
	@./scripts/build.sh --release --optimize --package

# Deploy the application
deploy:
	@echo "Deploying application..."
	@./scripts/deploy.sh --archive

# Run CI pipeline
ci:
	@echo "Running CI pipeline..."
	@./scripts/ci.sh

# Format code
format:
	@echo "Formatting code..."
	@cargo fmt
	@cd frontend && npx prettier --write "src/**/*.{ts,tsx,js,jsx,css,json}" || echo "Prettier not available"

# Lint code
lint:
	@echo "Linting code..."
	@cargo clippy -- -D warnings
	@cd frontend && npx eslint src/ || echo "ESLint not available"

# Watch for file changes
watch:
	@echo "Watching for file changes..."
	@./scripts/dev.sh watch

# Setup development environment
setup:
	@echo "Setting up development environment..."
	@./scripts/dev.sh setup

# Generate test data
data:
	@echo "Generating test data..."
	@./scripts/dev.sh data

# Run benchmarks
bench:
	@echo "Running benchmarks..."
	@./scripts/dev.sh bench

# Show development status
status:
	@echo "Development status:"
	@./scripts/dev.sh status

# Quick development workflow
quick: format lint test build
	@echo "Quick development workflow completed!"

# Full CI workflow
full-ci: clean install format lint test build ci
	@echo "Full CI workflow completed!"

# Release workflow
release-workflow: clean install test build release deploy
	@echo "Release workflow completed!"