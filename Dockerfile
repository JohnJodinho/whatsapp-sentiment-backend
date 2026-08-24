# Python 3.11 slim image for production deployment
FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies + Redis Server
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    redis-server \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user && \
    mkdir -p /var/run/redis /var/log/redis /app/data /app/models && \
    chown -R user:user /var/run/redis /var/log/redis /var/lib/redis /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 2. Install Dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 3. Copy Application Code & Models
COPY --chown=user . .

# 4. Make start script executable
RUN chmod +x start.sh

# 5. Run start script with dynamic $PORT binding
CMD ["./start.sh"]