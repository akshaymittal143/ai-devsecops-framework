# Multi-stage build for threat detection service
FROM python:3.9-slim as builder

# Security: Run as non-root user
RUN useradd -m -u 1000 app
WORKDIR /app

# Install build dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Second stage: Runtime
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY config/ config/

# Create log directory
RUN mkdir -p logs

# Set environment variables
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/config/settings.yaml

# Security: Run as non-root user
RUN useradd -m -u 1000 app && \
    chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run service
CMD ["python", "src/main.py"]