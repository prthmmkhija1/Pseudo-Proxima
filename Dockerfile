# Proxima Dockerfile
# Build: docker build -t proxima:latest .
# Run: docker run --rm proxima --help

# =============================================================================
# Builder Stage
# =============================================================================
FROM python:3.14-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md CHANGELOG.md ./
COPY src/ ./src/

# Install Proxima with core dependencies
RUN pip install --no-cache-dir --user .

# =============================================================================
# Runtime Stage
# =============================================================================
FROM python:3.14-slim AS runtime

# Labels
LABEL org.opencontainers.image.title="Proxima"
LABEL org.opencontainers.image.description="Intelligent Quantum Simulation Orchestration Framework"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/proxima-project/proxima"
LABEL org.opencontainers.image.licenses="MIT"

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create data directories
RUN mkdir -p /app/circuits /app/results /app/exports

# Create non-root user for production use (optional)
# RUN useradd --create-home --uid 1000 proxima
# USER proxima

# Set environment variables
ENV PROXIMA_CONFIG=/app/proxima.yaml
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
ENTRYPOINT ["proxima"]
CMD ["--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD proxima --version || exit 1
