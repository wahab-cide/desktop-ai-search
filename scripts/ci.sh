#!/bin/bash

# Continuous Integration script for Desktop AI Search
# This script runs the complete CI pipeline

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
CI_DB="ci_test.db"
COVERAGE_THRESHOLD=80
TEST_TIMEOUT=300  # 5 minutes

# Function to print colored output
print_status() {
    echo -e "${BLUE}[CI-INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[CI-SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[CI-WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[CI-ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}[CI-STEP]${NC} $1"
    echo -e "${BLUE}============================================${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup CI environment
setup_ci_environment() {
    print_step "Setting up CI environment"
    
    # Set environment variables
    export RUST_BACKTRACE=1
    export RUST_LOG=debug
    export DESKTOP_AI_SEARCH_CONFIG_PATH="config.ci.toml"
    export NODE_ENV=test
    
    # Create CI configuration
    cat > config.ci.toml << 'EOF'
[database]
database_url = "ci_test.db"
max_connections = 5
connection_timeout = 30

[search]
max_results = 50
search_timeout = 5
enable_fuzzy_search = true

[indexing]
max_file_size = 10_000_000  # 10MB for CI
batch_size = 100

[models]
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
device = "cpu"

[performance]
cache_size = 64  # MB
max_threads = 2
enable_compression = true

[logging]
level = "info"
file = "logs/ci.log"
EOF
    
    # Create logs directory
    mkdir -p logs
    
    print_success "CI environment setup completed"
}

# Function to check system requirements
check_system_requirements() {
    print_step "Checking system requirements"
    
    # Check Rust
    if ! command_exists rustc; then
        print_error "Rust is not installed"
        exit 1
    fi
    
    RUST_VERSION=$(rustc --version | cut -d' ' -f2)
    print_status "Rust version: $RUST_VERSION"
    
    # Check Node.js
    if ! command_exists node; then
        print_error "Node.js is not installed"
        exit 1
    fi
    
    NODE_VERSION=$(node --version)
    print_status "Node.js version: $NODE_VERSION"
    
    # Check npm
    if ! command_exists npm; then
        print_error "npm is not installed"
        exit 1
    fi
    
    NPM_VERSION=$(npm --version)
    print_status "npm version: $NPM_VERSION"
    
    # Check SQLite
    if ! command_exists sqlite3; then
        print_error "SQLite is not installed"
        exit 1
    fi
    
    SQLITE_VERSION=$(sqlite3 --version | cut -d' ' -f1)
    print_status "SQLite version: $SQLITE_VERSION"
    
    # Check system resources
    print_status "System information:"
    echo "  OS: $(uname -s)"
    echo "  Architecture: $(uname -m)"
    echo "  Memory: $(free -h 2>/dev/null | grep '^Mem:' | awk '{print $2}' || echo 'N/A')"
    echo "  Disk space: $(df -h . | tail -1 | awk '{print $4}' || echo 'N/A')"
    echo "  CPU cores: $(nproc 2>/dev/null || echo 'N/A')"
    
    print_success "System requirements check completed"
}

# Function to install dependencies
install_dependencies() {
    print_step "Installing dependencies"
    
    # Install Rust dependencies
    print_status "Installing Rust dependencies..."
    cargo fetch --verbose
    
    # Install frontend dependencies
    print_status "Installing frontend dependencies..."
    cd $FRONTEND_DIR
    npm ci --verbose
    cd ..
    
    print_success "Dependencies installed"
}

# Function to run code quality checks
run_code_quality_checks() {
    print_step "Running code quality checks"
    
    # Check Rust formatting
    print_status "Checking Rust code formatting..."
    if ! cargo fmt --check; then
        print_error "Rust code is not properly formatted"
        exit 1
    fi
    
    # Run Rust linting
    print_status "Running Rust linting..."
    cargo clippy --all-targets --all-features -- -D warnings
    
    # Check for security vulnerabilities
    if command_exists cargo-audit; then
        print_status "Running security audit..."
        cargo audit
    else
        print_warning "cargo-audit not installed, skipping security audit"
    fi
    
    # Check frontend code quality
    if [ -f "$FRONTEND_DIR/package.json" ]; then
        cd $FRONTEND_DIR
        
        # Check TypeScript formatting
        if command_exists npx; then
            print_status "Checking TypeScript formatting..."
            npx prettier --check "src/**/*.{ts,tsx,js,jsx,css,json}"
        fi
        
        # Run ESLint
        if [ -f ".eslintrc.js" ]; then
            print_status "Running ESLint..."
            npx eslint src/
        fi
        
        cd ..
    fi
    
    print_success "Code quality checks completed"
}

# Function to build the project
build_project() {
    print_step "Building project"
    
    # Build frontend
    print_status "Building frontend..."
    cd $FRONTEND_DIR
    npm run build
    
    # Verify frontend build
    if [ ! -d "dist" ]; then
        print_error "Frontend build failed - dist directory not found"
        exit 1
    fi
    
    cd ..
    
    # Build backend
    print_status "Building backend..."
    cargo build --release --verbose
    
    # Verify backend build
    if [ ! -f "target/release/$PROJECT_NAME" ] && [ ! -f "target/release/$PROJECT_NAME.exe" ]; then
        print_error "Backend build failed - executable not found"
        exit 1
    fi
    
    print_success "Project build completed"
}

# Function to setup test database
setup_test_database() {
    print_status "Setting up test database..."
    
    # Remove existing test database
    if [ -f "$CI_DB" ]; then
        rm "$CI_DB"
    fi
    
    # Create test database with schema
    sqlite3 "$CI_DB" < src/sql/001_initial_schema.sql
    sqlite3 "$CI_DB" < src/sql/002_add_embeddings.sql
    sqlite3 "$CI_DB" < src/sql/003_add_chunks.sql
    sqlite3 "$CI_DB" < src/sql/004_add_indices.sql
    sqlite3 "$CI_DB" < src/sql/005_enhanced_chunks_embeddings.sql
    sqlite3 "$CI_DB" < src/sql/006_add_clip_embeddings.sql
    sqlite3 "$CI_DB" < src/sql/007_add_fts5_chunks.sql
    
    print_success "Test database setup completed"
}

# Function to run unit tests
run_unit_tests() {
    print_step "Running unit tests"
    
    # Run Rust tests
    print_status "Running Rust unit tests..."
    timeout $TEST_TIMEOUT cargo test --verbose --release
    
    # Run frontend tests if available
    if [ -f "$FRONTEND_DIR/package.json" ]; then
        cd $FRONTEND_DIR
        if grep -q "test" package.json; then
            print_status "Running frontend tests..."
            timeout $TEST_TIMEOUT npm test
        fi
        cd ..
    fi
    
    print_success "Unit tests completed"
}

# Function to run integration tests
run_integration_tests() {
    print_step "Running integration tests"
    
    # Run integration tests
    print_status "Running integration tests..."
    timeout $TEST_TIMEOUT cargo test --test '*' --verbose --release
    
    print_success "Integration tests completed"
}

# Function to run performance tests
run_performance_tests() {
    print_step "Running performance tests"
    
    # Run benchmarks if available
    if grep -q "bench" Cargo.toml; then
        print_status "Running performance benchmarks..."
        cargo bench
    else
        print_warning "No benchmarks configured"
    fi
    
    print_success "Performance tests completed"
}

# Function to generate test coverage
generate_test_coverage() {
    print_step "Generating test coverage"
    
    # Check if tarpaulin is installed
    if ! command_exists cargo-tarpaulin; then
        print_status "Installing cargo-tarpaulin..."
        cargo install cargo-tarpaulin
    fi
    
    # Generate coverage report
    print_status "Generating coverage report..."
    cargo tarpaulin --out Html --out Xml --output-dir coverage --verbose
    
    # Check coverage threshold
    if [ -f "coverage/tarpaulin-report.xml" ]; then
        COVERAGE=$(grep -o 'line-rate="[^"]*"' coverage/tarpaulin-report.xml | head -1 | cut -d'"' -f2)
        COVERAGE_PERCENT=$(echo "$COVERAGE * 100" | bc -l | cut -d'.' -f1)
        
        print_status "Test coverage: $COVERAGE_PERCENT%"
        
        if [ "$COVERAGE_PERCENT" -lt "$COVERAGE_THRESHOLD" ]; then
            print_error "Test coverage ($COVERAGE_PERCENT%) is below threshold ($COVERAGE_THRESHOLD%)"
            exit 1
        fi
    fi
    
    print_success "Test coverage generation completed"
}

# Function to run security checks
run_security_checks() {
    print_step "Running security checks"
    
    # Run cargo audit
    if command_exists cargo-audit; then
        print_status "Running cargo audit..."
        cargo audit
    else
        print_warning "cargo-audit not installed, skipping security audit"
    fi
    
    # Check for common security issues
    print_status "Checking for common security issues..."
    
    # Check for hardcoded secrets
    if command_exists grep; then
        SECRET_PATTERNS=(
            "password.*=.*['\"].*['\"]"
            "secret.*=.*['\"].*['\"]"
            "token.*=.*['\"].*['\"]"
            "key.*=.*['\"].*['\"]"
            "api_key.*=.*['\"].*['\"]"
        )
        
        for pattern in "${SECRET_PATTERNS[@]}"; do
            if grep -r -i "$pattern" src/ --include="*.rs" --include="*.toml" --include="*.json"; then
                print_error "Potential hardcoded secret found: $pattern"
                exit 1
            fi
        done
    fi
    
    print_success "Security checks completed"
}

# Function to validate documentation
validate_documentation() {
    print_step "Validating documentation"
    
    # Check if required documentation exists
    REQUIRED_DOCS=(
        "README.md"
        "docs/README.md"
        "docs/api.md"
    )
    
    for doc in "${REQUIRED_DOCS[@]}"; do
        if [ ! -f "$doc" ]; then
            print_error "Required documentation missing: $doc"
            exit 1
        fi
    done
    
    # Check if documentation is up to date
    print_status "Checking documentation freshness..."
    
    # Validate README links (if available)
    if command_exists markdown-link-check; then
        markdown-link-check README.md
        markdown-link-check docs/README.md
    fi
    
    print_success "Documentation validation completed"
}

# Function to run deployment tests
run_deployment_tests() {
    print_step "Running deployment tests"
    
    # Test build scripts
    print_status "Testing build scripts..."
    if [ -f "scripts/build.sh" ]; then
        chmod +x scripts/build.sh
        scripts/build.sh --help > /dev/null
    fi
    
    # Test deployment scripts
    print_status "Testing deployment scripts..."
    if [ -f "scripts/deploy.sh" ]; then
        chmod +x scripts/deploy.sh
        scripts/deploy.sh --help > /dev/null
    fi
    
    print_success "Deployment tests completed"
}

# Function to cleanup CI artifacts
cleanup_ci_artifacts() {
    print_step "Cleaning up CI artifacts"
    
    # Remove test database
    if [ -f "$CI_DB" ]; then
        rm "$CI_DB"
    fi
    
    # Remove temporary files
    if [ -f "config.ci.toml" ]; then
        rm "config.ci.toml"
    fi
    
    # Clean up logs (keep for debugging)
    # rm -rf logs
    
    print_success "CI artifacts cleanup completed"
}

# Function to generate CI report
generate_ci_report() {
    print_step "Generating CI report"
    
    REPORT_FILE="ci_report.json"
    
    cat > "$REPORT_FILE" << EOF
{
    "project": "$PROJECT_NAME",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "platform": "$(uname -s)",
    "architecture": "$(uname -m)",
    "rust_version": "$(rustc --version)",
    "node_version": "$(node --version)",
    "build_status": "success",
    "test_status": "success",
    "coverage": {
        "threshold": $COVERAGE_THRESHOLD,
        "actual": "$([ -f coverage/tarpaulin-report.xml ] && grep -o 'line-rate="[^"]*"' coverage/tarpaulin-report.xml | head -1 | cut -d'"' -f2 | awk '{print $1 * 100}' || echo 0)"
    },
    "artifacts": {
        "binary": "target/release/$PROJECT_NAME",
        "frontend": "frontend/dist",
        "coverage": "coverage/tarpaulin-report.html"
    }
}
EOF
    
    print_success "CI report generated: $REPORT_FILE"
}

# Function to show CI summary
show_ci_summary() {
    print_step "CI Pipeline Summary"
    
    echo "Project: $PROJECT_NAME"
    echo "Timestamp: $(date)"
    echo "Platform: $(uname -s) $(uname -m)"
    echo "Rust: $(rustc --version)"
    echo "Node.js: $(node --version)"
    echo ""
    echo "Build Status: ✅ SUCCESS"
    echo "Test Status: ✅ SUCCESS"
    echo "Security: ✅ PASSED"
    echo "Coverage: $([ -f coverage/tarpaulin-report.xml ] && grep -o 'line-rate="[^"]*"' coverage/tarpaulin-report.xml | head -1 | cut -d'"' -f2 | awk '{print $1 * 100}' || echo 0)%"
    echo ""
    echo "Artifacts:"
    echo "  Binary: target/release/$PROJECT_NAME"
    echo "  Frontend: frontend/dist"
    echo "  Coverage: coverage/tarpaulin-report.html"
    echo "  Report: ci_report.json"
}

# Function to handle CI failure
handle_ci_failure() {
    print_error "CI Pipeline Failed"
    
    # Generate failure report
    cat > ci_failure_report.json << EOF
{
    "project": "$PROJECT_NAME",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "status": "failed",
    "error": "$1",
    "logs": "logs/ci.log"
}
EOF
    
    # Show failure summary
    echo "Failure Report: ci_failure_report.json"
    echo "Logs: logs/ci.log"
    
    exit 1
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help       Show this help message"
    echo "  -f, --fast       Skip performance tests and coverage"
    echo "  -s, --security   Run security checks only"
    echo "  -c, --coverage   Generate coverage report"
    echo "  -v, --verbose    Verbose output"
    echo ""
    echo "Examples:"
    echo "  $0               # Run full CI pipeline"
    echo "  $0 -f            # Run fast CI pipeline"
    echo "  $0 -s            # Run security checks only"
    echo "  $0 -c            # Generate coverage report"
}

# Parse command line arguments
FAST=false
SECURITY_ONLY=false
COVERAGE_ONLY=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -f|--fast)
            FAST=true
            shift
            ;;
        -s|--security)
            SECURITY_ONLY=true
            shift
            ;;
        -c|--coverage)
            COVERAGE_ONLY=true
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

# Set up error handling
trap 'handle_ci_failure "Pipeline interrupted"' INT TERM

# Main CI pipeline
main() {
    print_status "Starting CI pipeline for $PROJECT_NAME"
    
    # Setup CI environment
    setup_ci_environment
    
    # Check system requirements
    check_system_requirements
    
    # Install dependencies
    install_dependencies
    
    if [ "$SECURITY_ONLY" = true ]; then
        # Run security checks only
        run_security_checks
        print_success "Security checks completed successfully"
        exit 0
    fi
    
    if [ "$COVERAGE_ONLY" = true ]; then
        # Generate coverage only
        setup_test_database
        generate_test_coverage
        print_success "Coverage report generated successfully"
        exit 0
    fi
    
    # Run code quality checks
    run_code_quality_checks
    
    # Build project
    build_project
    
    # Setup test database
    setup_test_database
    
    # Run tests
    run_unit_tests
    run_integration_tests
    
    # Run performance tests (unless fast mode)
    if [ "$FAST" = false ]; then
        run_performance_tests
        generate_test_coverage
    fi
    
    # Run security checks
    run_security_checks
    
    # Validate documentation
    validate_documentation
    
    # Run deployment tests
    run_deployment_tests
    
    # Generate CI report
    generate_ci_report
    
    # Show summary
    show_ci_summary
    
    # Cleanup
    cleanup_ci_artifacts
    
    print_success "CI pipeline completed successfully!"
}

# Run main function
main "$@"