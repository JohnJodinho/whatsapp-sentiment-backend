# SentimentScope Backend

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![ONNX](https://img.shields.io/badge/ONNX-005C84?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai/)
[![Celery](https://img.shields.io/badge/celery-%23a9cc54.svg?style=for-the-badge&logo=celery&logoColor=fdd835)](https://docs.celeryq.dev/en/stable/)

An enterprise-grade, asynchronous backend system for WhatsApp chat analytics, RAG-based natural language querying, and highly-optimized ONNX sentiment analysis running on CPU.

## 🎯 Core Capabilities
- **Chat Intelligence:** Processes raw WhatsApp exports into structured analytical models (Messages, Segments, Participants).
- **Optimized Sentiment Analysis:** Leverages HuggingFace Optimum with an ONNX-quantized classification model to analyze large chat volumes asynchronously via Celery using optimized CPU multi-threading.
- **RAG & Embeddings (pgvector):** Generates and stores 1536-dimensional vectors via Langchain & `pgvector` (`ivfflat` indexing) for advanced conversational memory and contextual querying.
- **Real-Time Dashboards & SSE:** Uses Server-Sent Events (SSE) and Redis Pub/Sub to deliver real-time progress updates during heavy computation tasks.

---

## 🏗 System Architecture & Data Flow

### Architecture Topology

```mermaid
graph TD
    %% Clients
    Client[Client / Web UI]
    
    %% API Layer
    subgraph API Layer [FastAPI Application]
        RouterAuth[Auth Router]
        RouterUploads[Uploads & Parsing Router]
        RouterChats[Chat Management Router]
        RouterRAG[RAG & Query Router]
        RouterSSE[SSE & WebSocket Router]
        RouterDash[Dashboard Router]
    end

    %% Services Layer
    subgraph Services Layer
        AuthSvc[Security & Auth Service]
        ParseSvc[Chat Parser & Structurer]
        EmbedSvc[Embedding Service]
        RAGSvc[LangChain RAG Service]
        DashSvc[Analytics Aggregator]
    end

    %% Workers Layer
    subgraph Workers Layer [Celery Background Workers]
        SentWorker[Sentiment Worker]
        EmbedWorker[Embedding Generation Worker]
    end

    %% Data & Inference Layer
    subgraph Infrastructure
        DB[(PostgreSQL + pgvector)]
        Redis[(Redis Pub/Sub & Broker)]
        ONNX[ONNX Quantized Model CPU]
    end

    %% Connections
    Client -->|REST API / SSE| RouterAuth
    Client --> RouterUploads
    Client --> RouterChats
    Client --> RouterRAG
    Client --> RouterSSE
    Client --> RouterDash

    RouterAuth --> AuthSvc
    RouterUploads --> ParseSvc
    RouterRAG --> RAGSvc
    RouterDash --> DashSvc
    
    ParseSvc --> DB
    DashSvc --> DB
    AuthSvc --> DB

    %% Worker Triggers
    RouterChats -.->|Enqueues Task| Redis
    Redis -.->|Consumes Task| SentWorker
    Redis -.->|Consumes Task| EmbedWorker

    %% Worker Processing
    SentWorker -->|Inference Batching| ONNX
    SentWorker -->|Writes Scores| DB
    SentWorker -->|Publishes Progress| Redis
    
    EmbedWorker -->|Writes Vectors| DB
    
    %% RAG Query
    RAGSvc -->|Vector Similarity Search| DB
```

### Sentiment Analysis Execution Sequence

```mermaid
sequenceDiagram
    participant User as Web UI
    participant API as FastAPI Router
    participant DB as PostgreSQL
    participant Redis as Redis Pub/Sub & Broker
    participant Celery as Celery Worker
    participant ONNX as ONNX Inference

    User->>API: POST /chat/{id}/analyze
    API->>DB: Set chat status to "processing"
    API->>Redis: Enqueue sentiment analysis task
    API-->>User: 202 Accepted (Task ID)
    
    User->>API: GET /chat/{id}/progress/stream
    API->>Redis: Subscribe to "chat_progress_{id}"
    API-->>User: SSE Connection Established

    Celery->>Redis: Dequeue task
    Celery->>DB: Stream unscored messages (Batch size: 100)
    loop Every Batch
        Celery->>ONNX: Text classification (Batch size: 32)
        ONNX-->>Celery: Prediction Scores
        Celery->>DB: Insert MessageSentiment records
        Celery->>DB: db.commit()
        Celery->>DB: Fetch progress %
        Celery->>Redis: r.publish("chat_progress_{id}", payload)
    end
    
    Redis-->>API: Receive progress payload
    API-->>User: Emit SSE Event (percent complete)
    
    Celery->>DB: Update chat status to "completed"
    Celery->>Redis: r.publish("chat_progress_{id}", {status: "done"})
    Redis-->>API: 
    API-->>User: Emit SSE Event (done)
    User->>API: Close SSE Connection
```

---

## 📂 Project Directory Structure

```text
whatsapp_sentiment_backend/
├── alembic/                    # SQLAlchemy migration scripts
├── alembic.ini                 # Alembic configuration
├── Dockerfile                  # API Container build instructions
├── compose.yaml                # Multi-container local orchestration
├── pytest.ini                  # PyTest configuration
├── start.sh                    # Container entrypoint script
├── requirements.txt            # Python dependencies
├── models/                     # ML Assets (GitIgnored)
│   └── onnx_model_optimized/   # HuggingFace Optimum quantized ONNX models
└── src/
    └── app/                    # Primary FastAPI Application Module
        ├── __init__.py
        ├── main.py             # FastAPI App Factory & Lifespan
        ├── config.py           # Pydantic BaseSettings config map
        ├── logging_config.py   # Centralized logger setup
        ├── limiter.py          # Rate limiting middleware
        ├── security.py         # JWT and dependency injections
        ├── schemas.py          # Pydantic models for request/response
        ├── models.py           # SQLAlchemy declarative base models
        ├── crud.py             # Database query operations
        ├── db/                 # DB connections and session makers
        ├── routers/            # API Route handlers (auth, chats, rag, sentiment)
        ├── services/           # Core business logic and Celery Workers
        └── utils/              # Parsers, formatters, and helpers
```

---

## 🚀 Prerequisites & Environment Setup

- **Python:** 3.12+ (if running locally outside Docker)
- **Database:** PostgreSQL with the `pgvector` extension installed.
- **Message Broker:** Redis server.
- **Machine Learning Models:** The optimized ONNX model must be placed in `models/onnx_model_optimized/`.

Create a `.env` file in the root directory:

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/whatsapp_sentiment

# Redis / Celery
CELERY_BROKER_URL=redis://localhost:6379/0

# Authentication
SECRET_KEY=your_jwt_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

---

## 💻 Installation & Execution

### Option 1: Docker Compose (Recommended)
This spins up the FastAPI API, Celery worker, PostgreSQL (with pgvector), and Redis.

```bash
# Ensure models are placed in models/onnx_model_optimized/
docker-compose up --build -d
```

### Option 2: Local Development Environment

1. **Virtual Environment Setup:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **Database Migrations:**
   ```bash
   alembic upgrade head
   ```

3. **Run the API:**
   ```bash
   uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Run the Celery Worker (in a separate terminal):**
   ```bash
   # Linux/macOS
   celery -A src.app.services.sentiment_worker worker --loglevel=info

   # Windows (requires gevent/solo pool)
   celery -A src.app.services.sentiment_worker worker --loglevel=info --pool=solo
   ```

---

## 📖 API & Pipeline Reference

### Key Endpoints

- `POST /auth/login`: Authenticate and receive a JWT.
- `POST /uploads/chat`: Upload and parse raw exported `.txt` WhatsApp chats into DB objects.
- `GET /chats/{id}/status`: Fetch current extraction and embedding pipeline status.
- `POST /chat/{id}/analyze`: Trigger the Celery worker to perform sentiment analysis over parsed segments.
- `GET /chat/{id}/progress/stream`: Subscribe to an SSE stream yielding live progression % for the sentiment batch process.
- `GET /chats/{id}/dashboard`: Fetch holistic chat analytics, KPI metrics, sentiment distributions, and timelines.
- `POST /chat/{id}/query/streamed`: Execute a Langchain RAG query directly against the chat's vector embeddings, yielding a streamed Markdown response.

### Background Pipelines (Celery)
The `analyze_sentiment_task` handles intensive CPU-bound inference. It accesses the `onnxruntime` backend configured specifically for `CPUExecutionProvider` using max thread optimization (`intra_op_num_threads = 0`). Results are committed dynamically using pagination and batch sizes of `32` to ensure consistent and fast database I/O.