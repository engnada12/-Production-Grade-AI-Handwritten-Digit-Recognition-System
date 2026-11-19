# ==============================================================================
# Multi-Stage Production Dockerfile for AI Handwritten Digit Recognizer
# Security-Hardened, Non-Root User, Ultra-Fast Startup
# ==============================================================================

# Stage 1: Build & Dependency Resolution
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --user -r requirements.txt


# Stage 2: Minimal Runtime Environment
FROM python:3.11-slim AS runtime

# Security: Create non-root system user and group
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application source code
COPY --chown=appuser:appgroup model_handler.py server.py ./
COPY --chown=appuser:appgroup static/ ./static/
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup data/ ./data/

# Switch to non-root user
USER appuser

# Expose HTTP port
EXPOSE 8081

# Healthcheck definition
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8081/api/health')" || exit 1

# Start production server
CMD ["python", "server.py", "8081"]
