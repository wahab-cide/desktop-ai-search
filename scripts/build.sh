#!/bin/bash

# Build script for Desktop AI Search
# This script handles both development and production builds

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
BUILD_DIR="dist"
RELEASE_DIR="release"
PROFILE="release"
PLATFORM=$(uname -s)
ARCH=$(uname -m)

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

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Rust
    if ! command_exists rustc; then
        print_error "Rust is not installed. Please install Rust from https://rustup.rs/"
        exit 1
    fi
    
    # Check Node.js
    if ! command_exists node; then
        print_error "Node.js is not installed. Please install Node.js from https://nodejs.org/"
        exit 1
    fi
    
    # Check npm
    if ! command_exists npm; then
        print_error "npm is not installed. Please install npm."
        exit 1
    fi
    
    # Check Tauri CLI
    if ! command_exists cargo-tauri; then
        print_warning "Tauri CLI not found. Installing..."
        cargo install tauri-cli
    fi
    
    # Check system dependencies
    case "$PLATFORM" in
        "Darwin")
            if ! command_exists brew; then
                print_warning "Homebrew not found. Some features may not work."
            fi
            ;;
        "Linux")
            if ! command_exists apt-get && ! command_exists yum && ! command_exists pacman; then
                print_warning "No supported package manager found. Some features may not work."
            fi
            ;;
    esac
    
    print_success "Prerequisites check completed"
}

# Function to clean build artifacts
clean_build() {
    print_status "Cleaning build artifacts..."
    
    # Clean Rust artifacts
    if [ -d "target" ]; then
        rm -rf target/
    fi
    
    # Clean frontend artifacts
    if [ -d "$FRONTEND_DIR/dist" ]; then
        rm -rf $FRONTEND_DIR/dist/
    fi
    
    if [ -d "$FRONTEND_DIR/node_modules" ]; then
        rm -rf $FRONTEND_DIR/node_modules/
    fi
    
    # Clean release directory
    if [ -d "$RELEASE_DIR" ]; then
        rm -rf $RELEASE_DIR/
    fi
    
    print_success "Build artifacts cleaned"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Install Rust dependencies
    print_status "Installing Rust dependencies..."
    cargo fetch
    
    # Install frontend dependencies
    print_status "Installing frontend dependencies..."
    cd $FRONTEND_DIR
    npm ci
    cd ..
    
    print_success "Dependencies installed"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    # Run Rust tests
    print_status "Running Rust tests..."
    cargo test --release
    
    # Run frontend tests (if any)
    if [ -f "$FRONTEND_DIR/package.json" ] && grep -q "test" "$FRONTEND_DIR/package.json"; then
        print_status "Running frontend tests..."
        cd $FRONTEND_DIR
        npm test
        cd ..
    fi
    
    print_success "Tests completed"
}

# Function to build frontend
build_frontend() {
    print_status "Building frontend..."
    
    cd $FRONTEND_DIR
    
    # Set production environment
    export NODE_ENV=production
    
    # Build frontend
    npm run build
    
    # Check if build was successful
    if [ ! -d "dist" ]; then
        print_error "Frontend build failed - dist directory not found"
        exit 1
    fi
    
    cd ..
    
    print_success "Frontend build completed"
}

# Function to build backend
build_backend() {
    print_status "Building backend..."
    
    # Build with optimizations
    RUSTFLAGS="-C target-cpu=native" cargo build --release
    
    # Check if build was successful
    if [ ! -f "target/release/$PROJECT_NAME" ] && [ ! -f "target/release/$PROJECT_NAME.exe" ]; then
        print_error "Backend build failed - executable not found"
        exit 1
    fi
    
    print_success "Backend build completed"
}

# Function to build Tauri app
build_tauri() {
    print_status "Building Tauri application..."
    
    # Build Tauri app
    cargo tauri build --release
    
    print_success "Tauri build completed"
}

# Function to optimize build
optimize_build() {
    print_status "Optimizing build..."
    
    # Strip debug symbols from binary (Linux/macOS)
    if [ "$PLATFORM" != "MSYS_NT" ] && [ "$PLATFORM" != "MINGW64_NT" ]; then
        if [ -f "target/release/$PROJECT_NAME" ]; then
            strip target/release/$PROJECT_NAME
        fi
    fi
    
    # Compress static assets
    if command_exists gzip; then
        find $FRONTEND_DIR/dist -type f \( -name "*.js" -o -name "*.css" -o -name "*.html" \) -exec gzip -k {} \;
    fi
    
    print_success "Build optimization completed"
}

# Function to package release
package_release() {
    print_status "Packaging release..."
    
    # Create release directory
    mkdir -p $RELEASE_DIR
    
    # Package based on platform
    case "$PLATFORM" in
        "Darwin")
            if [ -d "target/release/bundle/macos" ]; then
                cp -r target/release/bundle/macos/* $RELEASE_DIR/
            fi
            ;;
        "Linux")
            if [ -d "target/release/bundle/appimage" ]; then
                cp target/release/bundle/appimage/*.AppImage $RELEASE_DIR/
            fi
            if [ -d "target/release/bundle/deb" ]; then
                cp target/release/bundle/deb/*.deb $RELEASE_DIR/
            fi
            ;;
        "MSYS_NT"*|"MINGW64_NT"*)
            if [ -d "target/release/bundle/msi" ]; then
                cp target/release/bundle/msi/*.msi $RELEASE_DIR/
            fi
            ;;
    esac
    
    # Copy additional files
    cp README.md $RELEASE_DIR/
    cp -r docs/ $RELEASE_DIR/
    
    # Create checksums
    cd $RELEASE_DIR
    if command_exists sha256sum; then
        sha256sum * > checksums.txt
    elif command_exists shasum; then
        shasum -a 256 * > checksums.txt
    fi
    cd ..
    
    print_success "Release packaged in $RELEASE_DIR/"
}

# Function to show build info
show_build_info() {
    print_status "Build Information:"
    echo "  Platform: $PLATFORM"
    echo "  Architecture: $ARCH"
    echo "  Profile: $PROFILE"
    echo "  Rust version: $(rustc --version)"
    echo "  Node.js version: $(node --version)"
    echo "  npm version: $(npm --version)"
    echo "  Build timestamp: $(date)"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help       Show this help message"
    echo "  -c, --clean      Clean build artifacts before building"
    echo "  -t, --test       Run tests before building"
    echo "  -d, --dev        Build in development mode"
    echo "  -r, --release    Build in release mode (default)"
    echo "  -p, --package    Package the release"
    echo "  -f, --frontend   Build frontend only"
    echo "  -b, --backend    Build backend only"
    echo "  -o, --optimize   Optimize build (strip, compress)"
    echo "  -v, --verbose    Verbose output"
    echo ""
    echo "Examples:"
    echo "  $0                    # Build in release mode"
    echo "  $0 -c -t -p          # Clean, test, build, and package"
    echo "  $0 -d -f             # Build frontend in development mode"
    echo "  $0 -r -o -p          # Build optimized release and package"
}

# Parse command line arguments
CLEAN=false
TEST=false
DEV=false
PACKAGE=false
FRONTEND_ONLY=false
BACKEND_ONLY=false
OPTIMIZE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -t|--test)
            TEST=true
            shift
            ;;
        -d|--dev)
            DEV=true
            PROFILE="dev"
            shift
            ;;
        -r|--release)
            PROFILE="release"
            shift
            ;;
        -p|--package)
            PACKAGE=true
            shift
            ;;
        -f|--frontend)
            FRONTEND_ONLY=true
            shift
            ;;
        -b|--backend)
            BACKEND_ONLY=true
            shift
            ;;
        -o|--optimize)
            OPTIMIZE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            set -x
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main build process
main() {
    print_status "Starting build process for $PROJECT_NAME"
    show_build_info
    
    # Check prerequisites
    check_prerequisites
    
    # Clean if requested
    if [ "$CLEAN" = true ]; then
        clean_build
    fi
    
    # Install dependencies
    install_dependencies
    
    # Run tests if requested
    if [ "$TEST" = true ]; then
        run_tests
    fi
    
    # Build components
    if [ "$FRONTEND_ONLY" = true ]; then
        build_frontend
    elif [ "$BACKEND_ONLY" = true ]; then
        build_backend
    else
        build_frontend
        build_backend
        build_tauri
    fi
    
    # Optimize if requested
    if [ "$OPTIMIZE" = true ]; then
        optimize_build
    fi
    
    # Package if requested
    if [ "$PACKAGE" = true ]; then
        package_release
    fi
    
    print_success "Build process completed successfully!"
    
    # Show final output
    if [ "$PACKAGE" = true ]; then
        print_status "Release artifacts:"
        ls -la $RELEASE_DIR/
    else
        print_status "Build artifacts:"
        ls -la target/release/
    fi
}

# Run main function
main "$@"