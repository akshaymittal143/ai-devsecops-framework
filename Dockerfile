# Multi-stage build for threat detection service
FROM python:3.8-slim as builder

# Security: Run as non-root user
RUN useradd -m -u 1000 app
WORKDIR /app

# Install build dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Second stage: Runtime
FROM python:3.8-slim

# Copy only necessary files from builder
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/

# Security: Run as non-root user
RUN useradd -m -u 1000 app && \
    chown -R app:app /app
USER app

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "-m", "src.main"]