# Use Python 3.10 slim (Standard for HF Spaces)
FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies + Redis Server
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user && \
    mkdir -p /var/run/redis /var/log/redis && \
    chown -R user:user /var/run/redis /var/log/redis /var/lib/redis

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Install Dependencies
# Copy only requirements first to leverage Docker cache
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 4. Copy Code & Model
# We copy everything in the current directory (src, onnx_model_optimized, alembic)
COPY --chown=user . .

# 5. Make start script executable
RUN chmod +x start.sh

# 6. Expose the mandatory Hugging Face port
EXPOSE 7860

# 7. Run the start script
CMD ["./start.sh"]