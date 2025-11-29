# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12.0
FROM python:${PYTHON_VERSION}-slim as base

# 1. Environment Setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# 2. Dependencies (Cached Layer)
# Install system dependencies required for building wheels or specialized libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade --default-timeout=1000 -r requirements.txt
# 5. Security: Create User FIRST
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --no-create-home appuser

# 3. Model Injection (Copy AND set permissions at the same time)
COPY --chown=appuser:appgroup onnx_model_optimized ./onnx_model_optimized

# 4. Application Code (Copy AND set permissions)
COPY --chown=appuser:appgroup src ./src

# Switch User
USER appuser

# 6. Default Command (Can be overridden)
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host=0.0.0.0", "--port=8000"]