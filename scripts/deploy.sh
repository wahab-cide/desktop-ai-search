#!/bin/bash

# Deployment script for Desktop AI Search
# This script handles deployment to various platforms and environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="desktop-ai-search"
VERSION=$(grep '^version' Cargo.toml | sed 's/version = "\(.*\)"/\1/')
PLATFORM=$(uname -s)
ARCH=$(uname -m)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RELEASE_DIR="release"
DEPLOY_DIR="deploy"

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

# Function to create deployment directory
create_deploy_dir() {
    print_status "Creating deployment directory..."
    
    DEPLOY_PATH="$DEPLOY_DIR/${PROJECT_NAME}_${VERSION}_${PLATFORM}_${ARCH}_${TIMESTAMP}"
    mkdir -p "$DEPLOY_PATH"
    
    print_success "Deployment directory created: $DEPLOY_PATH"
}

# Function to copy release artifacts
copy_artifacts() {
    print_status "Copying release artifacts..."
    
    if [ ! -d "$RELEASE_DIR" ]; then
        print_error "Release directory not found. Please run build script first."
        exit 1
    fi
    
    # Copy all release files
    cp -r $RELEASE_DIR/* "$DEPLOY_PATH/"
    
    # Copy additional deployment files
    cp Cargo.toml "$DEPLOY_PATH/"
    cp tauri.conf.json "$DEPLOY_PATH/"
    
    print_success "Artifacts copied to deployment directory"
}

# Function to create installation script
create_install_script() {
    print_status "Creating installation script..."
    
    case "$PLATFORM" in
        "Darwin")
            cat > "$DEPLOY_PATH/install.sh" << 'EOF'
#!/bin/bash

# Installation script for Desktop AI Search (macOS)

set -e

INSTALL_DIR="/Applications"
APP_NAME="Desktop AI Search"

echo "Installing Desktop AI Search..."

# Check if app bundle exists
if [ -d "$APP_NAME.app" ]; then
    # Copy app to Applications
    cp -r "$APP_NAME.app" "$INSTALL_DIR/"
    
    # Make executable
    chmod +x "$INSTALL_DIR/$APP_NAME.app/Contents/MacOS/$APP_NAME"
    
    echo "Installation completed successfully!"
    echo "You can now find Desktop AI Search in your Applications folder."
else
    echo "Error: Application bundle not found."
    exit 1
fi
EOF
            ;;
        "Linux")
            cat > "$DEPLOY_PATH/install.sh" << 'EOF'
#!/bin/bash

# Installation script for Desktop AI Search (Linux)

set -e

INSTALL_DIR="/usr/local/bin"
DESKTOP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons"

echo "Installing Desktop AI Search..."

# Check if we have AppImage
if ls *.AppImage 1> /dev/null 2>&1; then
    # Install AppImage
    APP_IMAGE=$(ls *.AppImage | head -1)
    sudo cp "$APP_IMAGE" "$INSTALL_DIR/desktop-ai-search"
    sudo chmod +x "$INSTALL_DIR/desktop-ai-search"
    
    # Create desktop entry
    mkdir -p "$DESKTOP_DIR"
    cat > "$DESKTOP_DIR/desktop-ai-search.desktop" << 'DESKTOP_EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=Desktop AI Search
Comment=AI-powered local file search
Exec=desktop-ai-search
Icon=desktop-ai-search
Terminal=false
Categories=Utility;
DESKTOP_EOF
    
    echo "Installation completed successfully!"
    echo "You can now run 'desktop-ai-search' from the command line."
    
elif ls *.deb 1> /dev/null 2>&1; then
    # Install DEB package
    DEB_FILE=$(ls *.deb | head -1)
    sudo dpkg -i "$DEB_FILE"
    
    echo "Installation completed successfully!"
    echo "You can now find Desktop AI Search in your applications menu."
else
    echo "Error: No suitable installation package found."
    exit 1
fi
EOF
            ;;
        "MSYS_NT"*|"MINGW64_NT"*)
            cat > "$DEPLOY_PATH/install.bat" << 'EOF'
@echo off

REM Installation script for Desktop AI Search (Windows)

echo Installing Desktop AI Search...

REM Check if MSI installer exists
if exist *.msi (
    for %%f in (*.msi) do (
        echo Installing %%f...
        msiexec /i "%%f" /qn
        echo Installation completed successfully!
        echo You can now find Desktop AI Search in your Start Menu.
        goto :end
    )
) else (
    echo Error: MSI installer not found.
    pause
    exit /b 1
)

:end
pause
EOF
            ;;
    esac
    
    # Make install script executable
    chmod +x "$DEPLOY_PATH/install.sh" 2>/dev/null || true
    
    print_success "Installation script created"
}

# Function to create uninstall script
create_uninstall_script() {
    print_status "Creating uninstall script..."
    
    case "$PLATFORM" in
        "Darwin")
            cat > "$DEPLOY_PATH/uninstall.sh" << 'EOF'
#!/bin/bash

# Uninstall script for Desktop AI Search (macOS)

set -e

INSTALL_DIR="/Applications"
APP_NAME="Desktop AI Search"

echo "Uninstalling Desktop AI Search..."

# Remove app from Applications
if [ -d "$INSTALL_DIR/$APP_NAME.app" ]; then
    rm -rf "$INSTALL_DIR/$APP_NAME.app"
    echo "Application removed from Applications folder."
fi

# Remove user data (optional)
read -p "Remove user data and settings? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$HOME/Library/Application Support/com.desktop-ai-search.app"
    rm -rf "$HOME/.local/share/desktop-ai-search"
    echo "User data removed."
fi

echo "Uninstall completed successfully!"
EOF
            ;;
        "Linux")
            cat > "$DEPLOY_PATH/uninstall.sh" << 'EOF'
#!/bin/bash

# Uninstall script for Desktop AI Search (Linux)

set -e

INSTALL_DIR="/usr/local/bin"
DESKTOP_DIR="$HOME/.local/share/applications"

echo "Uninstalling Desktop AI Search..."

# Remove binary
if [ -f "$INSTALL_DIR/desktop-ai-search" ]; then
    sudo rm "$INSTALL_DIR/desktop-ai-search"
    echo "Binary removed from $INSTALL_DIR."
fi

# Remove desktop entry
if [ -f "$DESKTOP_DIR/desktop-ai-search.desktop" ]; then
    rm "$DESKTOP_DIR/desktop-ai-search.desktop"
    echo "Desktop entry removed."
fi

# Remove user data (optional)
read -p "Remove user data and settings? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$HOME/.local/share/desktop-ai-search"
    echo "User data removed."
fi

echo "Uninstall completed successfully!"
EOF
            ;;
        "MSYS_NT"*|"MINGW64_NT"*)
            cat > "$DEPLOY_PATH/uninstall.bat" << 'EOF'
@echo off

REM Uninstall script for Desktop AI Search (Windows)

echo Uninstalling Desktop AI Search...

REM Use Windows Programs and Features to uninstall
echo Opening Programs and Features...
appwiz.cpl

echo Please locate and uninstall "Desktop AI Search" from the list.
pause
EOF
            ;;
    esac
    
    # Make uninstall script executable
    chmod +x "$DEPLOY_PATH/uninstall.sh" 2>/dev/null || true
    
    print_success "Uninstall script created"
}

# Function to create deployment manifest
create_manifest() {
    print_status "Creating deployment manifest..."
    
    cat > "$DEPLOY_PATH/MANIFEST.txt" << EOF
Desktop AI Search Deployment Manifest
=====================================

Version: $VERSION
Platform: $PLATFORM
Architecture: $ARCH
Build Date: $(date)
Deployment Date: $(date)

Files Included:
---------------
EOF
    
    # List all files in deployment directory
    cd "$DEPLOY_PATH"
    find . -type f -exec ls -la {} \; >> MANIFEST.txt
    cd - > /dev/null
    
    print_success "Deployment manifest created"
}

# Function to create README for deployment
create_deployment_readme() {
    print_status "Creating deployment README..."
    
    cat > "$DEPLOY_PATH/README.txt" << EOF
Desktop AI Search v$VERSION
==========================

Thank you for downloading Desktop AI Search!

SYSTEM REQUIREMENTS:
- Platform: $PLATFORM
- Architecture: $ARCH
- RAM: 8GB+ recommended
- Storage: 2GB+ free space
- Dependencies: See docs/README.md

INSTALLATION:
1. Extract all files to a directory
2. Run the installation script:
   - Linux/macOS: ./install.sh
   - Windows: install.bat
3. Follow the on-screen instructions

FIRST RUN:
1. Launch Desktop AI Search
2. Go to Settings → Indexing
3. Select directories to index
4. Start indexing process
5. Begin searching!

DOCUMENTATION:
- Complete documentation: docs/README.md
- API Reference: docs/api.md
- Troubleshooting: docs/README.md#troubleshooting

SUPPORT:
- GitHub Issues: https://github.com/your-username/desktop-ai-search/issues
- Documentation: https://github.com/your-username/desktop-ai-search/wiki

UNINSTALL:
To uninstall, run:
- Linux/macOS: ./uninstall.sh
- Windows: uninstall.bat

Built with ❤️ for privacy-conscious users
EOF
    
    print_success "Deployment README created"
}

# Function to generate checksums
generate_checksums() {
    print_status "Generating checksums..."
    
    cd "$DEPLOY_PATH"
    
    # Generate SHA256 checksums
    if command_exists sha256sum; then
        sha256sum * > SHA256SUMS.txt
    elif command_exists shasum; then
        shasum -a 256 * > SHA256SUMS.txt
    fi
    
    # Generate MD5 checksums
    if command_exists md5sum; then
        md5sum * > MD5SUMS.txt
    elif command_exists md5; then
        md5 * > MD5SUMS.txt
    fi
    
    cd - > /dev/null
    
    print_success "Checksums generated"
}

# Function to create archive
create_archive() {
    print_status "Creating deployment archive..."
    
    ARCHIVE_NAME="${PROJECT_NAME}_${VERSION}_${PLATFORM}_${ARCH}.tar.gz"
    
    cd "$DEPLOY_DIR"
    tar -czf "$ARCHIVE_NAME" "$(basename "$DEPLOY_PATH")"
    cd - > /dev/null
    
    print_success "Archive created: $DEPLOY_DIR/$ARCHIVE_NAME"
}

# Function to validate deployment
validate_deployment() {
    print_status "Validating deployment..."
    
    # Check if all required files exist
    local required_files=(
        "README.txt"
        "MANIFEST.txt"
        "install.sh"
        "uninstall.sh"
        "SHA256SUMS.txt"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$DEPLOY_PATH/$file" ]; then
            print_warning "Missing file: $file"
        fi
    done
    
    # Verify checksums
    cd "$DEPLOY_PATH"
    if [ -f "SHA256SUMS.txt" ]; then
        if command_exists sha256sum; then
            if sha256sum -c SHA256SUMS.txt >/dev/null 2>&1; then
                print_success "Checksum verification passed"
            else
                print_warning "Checksum verification failed"
            fi
        fi
    fi
    cd - > /dev/null
    
    print_success "Deployment validation completed"
}

# Function to upload to GitHub releases (optional)
upload_to_github() {
    print_status "Uploading to GitHub releases..."
    
    if ! command_exists gh; then
        print_warning "GitHub CLI not installed. Skipping GitHub upload."
        return
    fi
    
    # Check if we're in a git repository
    if [ ! -d ".git" ]; then
        print_warning "Not in a git repository. Skipping GitHub upload."
        return
    fi
    
    # Create release if it doesn't exist
    if ! gh release view "v$VERSION" >/dev/null 2>&1; then
        print_status "Creating GitHub release v$VERSION..."
        gh release create "v$VERSION" \
            --title "Desktop AI Search v$VERSION" \
            --notes "Release notes for version $VERSION" \
            --draft
    fi
    
    # Upload archive
    ARCHIVE_NAME="${PROJECT_NAME}_${VERSION}_${PLATFORM}_${ARCH}.tar.gz"
    if [ -f "$DEPLOY_DIR/$ARCHIVE_NAME" ]; then
        gh release upload "v$VERSION" "$DEPLOY_DIR/$ARCHIVE_NAME"
        print_success "Archive uploaded to GitHub releases"
    fi
}

# Function to show deployment summary
show_deployment_summary() {
    print_status "Deployment Summary:"
    echo "  Version: $VERSION"
    echo "  Platform: $PLATFORM"
    echo "  Architecture: $ARCH"
    echo "  Deployment Path: $DEPLOY_PATH"
    echo "  Files deployed: $(find "$DEPLOY_PATH" -type f | wc -l)"
    echo "  Total size: $(du -sh "$DEPLOY_PATH" | cut -f1)"
    
    if [ -f "$DEPLOY_DIR/${PROJECT_NAME}_${VERSION}_${PLATFORM}_${ARCH}.tar.gz" ]; then
        echo "  Archive: $DEPLOY_DIR/${PROJECT_NAME}_${VERSION}_${PLATFORM}_${ARCH}.tar.gz"
        echo "  Archive size: $(du -sh "$DEPLOY_DIR/${PROJECT_NAME}_${VERSION}_${PLATFORM}_${ARCH}.tar.gz" | cut -f1)"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help       Show this help message"
    echo "  -a, --archive    Create deployment archive"
    echo "  -g, --github     Upload to GitHub releases"
    echo "  -v, --validate   Validate deployment"
    echo "  -c, --clean      Clean previous deployments"
    echo "  --verbose        Verbose output"
    echo ""
    echo "Examples:"
    echo "  $0               # Basic deployment"
    echo "  $0 -a -g         # Deploy with archive and GitHub upload"
    echo "  $0 -v -a         # Validate and create archive"
}

# Parse command line arguments
ARCHIVE=false
GITHUB=false
VALIDATE=false
CLEAN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -a|--archive)
            ARCHIVE=true
            shift
            ;;
        -g|--github)
            GITHUB=true
            shift
            ;;
        -v|--validate)
            VALIDATE=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        --verbose)
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

# Main deployment process
main() {
    print_status "Starting deployment process for $PROJECT_NAME v$VERSION"
    
    # Clean previous deployments if requested
    if [ "$CLEAN" = true ]; then
        print_status "Cleaning previous deployments..."
        rm -rf "$DEPLOY_DIR"
    fi
    
    # Create deployment directory
    create_deploy_dir
    
    # Copy artifacts
    copy_artifacts
    
    # Create installation scripts
    create_install_script
    create_uninstall_script
    
    # Create deployment documentation
    create_manifest
    create_deployment_readme
    
    # Generate checksums
    generate_checksums
    
    # Validate deployment if requested
    if [ "$VALIDATE" = true ]; then
        validate_deployment
    fi
    
    # Create archive if requested
    if [ "$ARCHIVE" = true ]; then
        create_archive
    fi
    
    # Upload to GitHub if requested
    if [ "$GITHUB" = true ]; then
        upload_to_github
    fi
    
    # Show summary
    show_deployment_summary
    
    print_success "Deployment process completed successfully!"
}

# Check if release directory exists
if [ ! -d "$RELEASE_DIR" ]; then
    print_error "Release directory not found. Please run the build script first."
    exit 1
fi

# Run main function
main "$@"