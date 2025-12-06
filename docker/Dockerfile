# NanoARB Trading Engine - Multi-stage Dockerfile
# Optimized for minimal latency and small image size

# =============================================================================
# Stage 1: Build Environment
# =============================================================================
FROM rust:1.75-bookworm AS builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./
COPY crates/nano-core/Cargo.toml crates/nano-core/
COPY crates/nano-feed/Cargo.toml crates/nano-feed/
COPY crates/nano-lob/Cargo.toml crates/nano-lob/
COPY crates/nano-model/Cargo.toml crates/nano-model/
COPY crates/nano-backtest/Cargo.toml crates/nano-backtest/
COPY crates/nano-strategy/Cargo.toml crates/nano-strategy/
COPY crates/nano-gateway/Cargo.toml crates/nano-gateway/

# Create dummy source files for dependency caching
RUN mkdir -p crates/nano-core/src && echo "pub fn main() {}" > crates/nano-core/src/lib.rs && \
    mkdir -p crates/nano-feed/src && echo "pub fn main() {}" > crates/nano-feed/src/lib.rs && \
    mkdir -p crates/nano-lob/src && echo "pub fn main() {}" > crates/nano-lob/src/lib.rs && \
    mkdir -p crates/nano-model/src && echo "pub fn main() {}" > crates/nano-model/src/lib.rs && \
    mkdir -p crates/nano-backtest/src && echo "pub fn main() {}" > crates/nano-backtest/src/lib.rs && \
    mkdir -p crates/nano-strategy/src && echo "pub fn main() {}" > crates/nano-strategy/src/lib.rs && \
    mkdir -p crates/nano-gateway/src && echo "fn main() {}" > crates/nano-gateway/src/main.rs

# Build dependencies only (for caching)
RUN cargo build --release 2>/dev/null || true

# Copy actual source code
COPY crates/ crates/

# Touch source files to trigger rebuild
RUN find crates -name "*.rs" -exec touch {} \;

# Build the actual application
RUN cargo build --release --bin nanoarb

# Strip debug symbols for smaller binary
RUN strip /build/target/release/nanoarb

# =============================================================================
# Stage 2: Runtime Environment
# =============================================================================
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 nanoarb
USER nanoarb
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/nanoarb /app/nanoarb

# Copy default configuration
COPY --chown=nanoarb:nanoarb docker/config.toml /app/config.toml

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/models

# Expose metrics port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

# Set environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV NANOARB_CONFIG=/app/config.toml

# Entry point
ENTRYPOINT ["/app/nanoarb"]
CMD ["--config", "/app/config.toml"]

