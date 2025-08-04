# Flight Operations Optimiser Dockerfile
# Multi-stage build for optimised production image

# Use Python 3.11 slim image as base
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data output models reports logs cache mlruns

# Set permissions
RUN chmod +x scripts/*.sh || true

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov pytest-mock black flake8 mypy

# Set development environment
ENV ENVIRONMENT=development
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 3000

# Default command for development
CMD ["python", "-m", "dagster", "dev"]

# Production stage
FROM base as production

# Create non-root user for security
RUN groupadd -r flightops && useradd -r -g flightops flightops

# Change ownership of app directory
RUN chown -R flightops:flightops /app

# Switch to non-root user
USER flightops

# Set production environment
ENV ENVIRONMENT=production
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for production
CMD ["python", "-m", "src.main"]

# ML training stage
FROM base as ml-training

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    psutil

# Set ML training environment
ENV ENVIRONMENT=training
ENV PYTHONPATH=/app

# Create Jupyter configuration
RUN jupyter notebook --generate-config

# Expose Jupyter port
EXPOSE 8888

# Default command for ML training
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Optimisation stage
FROM base as optimisation

# Install Gurobi optimiser (requires license)
# Note: This is a placeholder - actual Gurobi installation requires license
RUN echo "Gurobi installation would go here with proper license"

# Set optimisation environment
ENV ENVIRONMENT=optimisation
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000

# Default command for optimisation
CMD ["python", "-m", "src.models.optimisation.scheduler"]

# Testing stage
FROM base as testing

# Install testing dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-mock \
    pytest-xdist \
    coverage

# Set testing environment
ENV ENVIRONMENT=testing
ENV PYTHONPATH=/app

# Copy test files
COPY tests/ tests/

# Default command for testing
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=term"] 