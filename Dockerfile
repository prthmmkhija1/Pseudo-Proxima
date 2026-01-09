# =============================================================================
# Proxima Agent - Multi-stage Dockerfile
# Intelligent Quantum Simulation Orchestration Framework
# =============================================================================

# Stage 1: Build stage
FROM python:3.14-slim as builder

# Set build arguments
ARG PROXIMA_VERSION=0.1.0

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src/ ./src/
COPY configs/ ./configs/

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel build

# Install the package
RUN pip install -e ".[all]"

# =============================================================================
# Stage 2: Production runtime
# =============================================================================
FROM python:3.14-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Proxima specific
    PROXIMA_HOME=/home/proxima \
    PROXIMA_CONFIG_DIR=/home/proxima/.proxima \
    PROXIMA_LOG_LEVEL=info \
    # Virtual environment
    PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd --gid 1000 proxima && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash proxima

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy configuration files
COPY --chown=proxima:proxima configs/ /app/configs/

# Set working directory
WORKDIR /app

# Create config directory
RUN mkdir -p /home/proxima/.proxima && \
    chown -R proxima:proxima /home/proxima/.proxima

# Switch to non-root user
USER proxima

# Copy default configuration
COPY --chown=proxima:proxima configs/default.yaml /home/proxima/.proxima/config.yaml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD proxima --version || exit 1

# Default command
ENTRYPOINT ["proxima"]
CMD ["--help"]

# Labels for container metadata
LABEL org.opencontainers.image.title="Proxima Agent" \
    org.opencontainers.image.description="Intelligent Quantum Simulation Orchestration Framework" \
    org.opencontainers.image.version="${PROXIMA_VERSION}" \
    org.opencontainers.image.source="https://github.com/proxima-project/proxima" \
    org.opencontainers.image.licenses="MIT" \
    org.opencontainers.image.vendor="Proxima Team"

# =============================================================================
# Stage 3: Development image (optional)
# =============================================================================
FROM runtime as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-mock pytest-cov black ruff mypy

USER proxima

# Override for development
CMD ["bash"]
