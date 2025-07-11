# Dockerfile for Desktop AI Search
# Multi-stage build for optimized production image

# Build stage
FROM rust:1.70-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    tesseract-ocr \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy Cargo files
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src
COPY frontend ./frontend
COPY build.rs ./

# Build frontend
WORKDIR /app/frontend
RUN npm ci --only=production
RUN npm run build

# Build backend
WORKDIR /app
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/desktop-ai-search /app/desktop-ai-search

# Copy frontend assets
COPY --from=builder /app/frontend/dist /app/frontend/dist

# Copy configuration
COPY --from=builder /app/src/sql /app/sql

# Create directories for app data
RUN mkdir -p /app/data /app/logs /app/cache \
    && chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Expose port (if needed for web interface)
EXPOSE 8080

# Set environment variables
ENV RUST_LOG=info
ENV DESKTOP_AI_SEARCH_DATA_DIR=/app/data
ENV DESKTOP_AI_SEARCH_LOG_DIR=/app/logs
ENV DESKTOP_AI_SEARCH_CACHE_DIR=/app/cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/desktop-ai-search --version || exit 1

# Default command
CMD ["/app/desktop-ai-search"]