# SentimentScope Backend

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=chroma&logoColor=white)](https://www.trychroma.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005C84?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Celery](https://img.shields.io/badge/celery-%23a9cc54.svg?style=for-the-badge&logo=celery&logoColor=fdd835)](https://docs.celeryq.dev/en/stable/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

An asynchronous, production-grade AI backend for WhatsApp chat analytics, multi-tiered Agentic RAG, and CPU-optimized local INT8 ONNX inference for sentiment analysis and dense embeddings.

---

## 🎯 Core Capabilities

- **Chat Intelligence & Parsing:** Ingests raw WhatsApp `.txt` exports and converts unstructured conversation logs into structured relational models (Messages, Time Segments, Sender Segments, Participants).
- **Nigerian Pidgin & English Sentiment Analysis:** Powered by **`JohnAlbarkaIbrahim/afroxlmr-mini-nigerian-sentiment`** (`v1/onnx_int8/`) running on ONNX Runtime CPU (`CPUExecutionProvider`) with dynamic `id2label` mapping and Celery batch execution.
- **Zero External Latency Embeddings:** Computes 384-dimensional dense vectors locally using **`Davlan/afro-xlmr-mini` INT8 ONNX** with attention-masked mean pooling and L2 normalization.
- **Unified Vector Storage (ChromaDB / Chroma Cloud):** `VectorStore` abstraction supporting local persistent storage (`PersistentClient`) and stateless cloud deployments (`CloudClient`).
- **High-Throughput Groq LLM Engine:**
  - **`openai/gpt-oss-120b`**: Primary RAG answer synthesis and reasoning.
  - **`openai/gpt-oss-20b`**: Conversational contextualizer and resilient fallback.
  - **`qwen/qwen3.6-27b`**: Query intent classification, metadata filter extraction, and safe parameter-bound SQL generation.
- **Real-Time Streaming & Observability:** Server-Sent Events (SSE) and Redis Pub/Sub for live token streaming and task progress tracking.

---

## 🏛 System Architecture & Topology

```mermaid
graph TD
    Client["Client / Web UI"]

    subgraph APILayer ["FastAPI Application Layer"]
        AuthRouter["Auth & Security Router"]
        UploadRouter["Uploads & Parsing Router"]
        ChatRouter["Chat Lifecycle Router"]
        RAGRouter["Agentic RAG Streaming Router"]
        SSERouter["SSE Progress Router"]
        DashRouter["Dashboard Metrics Router"]
    end

    subgraph ServiceLayer ["Business Logic & Orchestration"]
        AuthSvc["Security & JWT Service"]
        ParseSvc["WhatsApp Parser Engine"]
        RouterSvc["Multi-Tier Router Service"]
        RetrieverSvc["VectorStore Retriever"]
        SummarySvc["Groq Summary Service"]
        CleanupSvc["User/Chat Cleanup Service"]
    end

    subgraph InferenceLayer ["Local CPU ONNX Runtime"]
        SentimentModel["Sentiment Classifier: JohnAlbarkaIbrahim/afroxlmr-mini-nigerian-sentiment (INT8 ONNX)"]
        EmbeddingModel["Dense Embedder: Davlan/afro-xlmr-mini (INT8 ONNX, 384d)"]
    end

    subgraph GenerationLayer ["Groq Cloud Engine"]
        GroqRouter["Router & SQL: qwen/qwen3.6-27b"]
        GroqPrimary["Primary Synthesis: openai/gpt-oss-120b"]
        GroqFallback["Fallback LLM: openai/gpt-oss-20b"]
    end

    subgraph StorageLayer ["Persistence & Caching"]
        PostgreSQL[("Supabase PostgreSQL / Relational DB")]
        Redis[("Redis Broker & Pub/Sub")]
        VectorDB[("ChromaDB: Local Persistent / Chroma Cloud")]
    end

    subgraph WorkerLayer ["Asynchronous Celery Workers"]
        SentimentWorker["Sentiment Processing Worker"]
        EmbeddingWorker["Vector Ingestion Worker"]
    end

    Client --> AuthRouter
    Client --> UploadRouter
    Client --> ChatRouter
    Client --> RAGRouter
    Client --> SSERouter
    Client --> DashRouter

    UploadRouter --> ParseSvc
    ParseSvc --> PostgreSQL

    ChatRouter -->|Enqueue Task| Redis
    Redis --> SentimentWorker
    Redis --> EmbeddingWorker

    SentimentWorker --> SentimentModel
    SentimentWorker --> PostgreSQL
    SentimentWorker --> Redis

    EmbeddingWorker --> EmbeddingModel
    EmbeddingWorker --> VectorDB

    RAGRouter --> RouterSvc
    RouterSvc --> GroqRouter
    RouterSvc --> RetrieverSvc
    RetrieverSvc --> EmbeddingModel
    RetrieverSvc --> VectorDB
    RouterSvc --> GroqPrimary
    GroqPrimary -. Fallback .-> GroqFallback
    RouterSvc --> PostgreSQL
```

---

## 🔍 Multi-Tier Agentic RAG Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor User as User
    participant API as FastAPI Router
    participant LLMRouter as Groq Router (qwen3.6-27b)
    participant SQL as Supabase PostgreSQL
    participant Embed as Local ONNX Embedder
    participant Chroma as ChromaDB / Chroma Cloud
    participant Synth as Groq Synthesizer (gpt-oss-120b)

    User->>API: POST /chat/{id}/query/streamed (Question + Analytics JSON)
    
    rect rgb(240, 248, 255)
        Note over API: Tier 1: Fast Trap (Regex Greeting Check)
        API-->>User: Instant greeting stream (if greeting matched)
    end

    API->>LLMRouter: Contextualize & Classify Intent (SQL / Vector / Dashboard / Hybrid)
    
    alt SQL or Hybrid Query
        API->>LLMRouter: Generate Safe Parameter-Bound SQL (:chat_id)
        LLMRouter-->>API: Validated SQL Query
        API->>SQL: Execute Read-Only Query with :chat_id parameter
        SQL-->>API: Tabular Aggregation Results
    end

    alt Vector Search or Hybrid Query
        API->>LLMRouter: Extract Date & Sender Metadata Filters
        LLMRouter-->>API: Extracted Filters (time_ranges, sender_names)
        API->>Embed: Embed Query (Davlan/afro-xlmr-mini INT8 ONNX 384d)
        Embed-->>API: 384-dimensional Normalized Dense Vector
        API->>Chroma: Query with Vector + Scoped Metadata Filter
        Chroma-->>API: Top-K Document Chunks
    end

    API->>Synth: Synthesize Grounded Response (Context + Anti-Leak Prompt)
    Synth-->>User: SSE Token Stream (data: "...")
    API->>SQL: Asynchronously Save Conversation Turn & Cited Sources
```

---

## 🧩 Component Architecture Breakdown

### 1. API & Routing Layer (`src/app/api/`)
- **`auth.py`**: User registration, login, token refresh, and JWT validation.
- **`chat.py`**: Chat creation, deletion, cancelation, and background job triggering.
- **`uploads.py`**: WhatsApp `.txt` file streaming upload, regex format parsing, and database population.
- **`query.py`**: Server-Sent Events (SSE) streaming endpoint for RAG chat querying.
- **`dashboard.py`**: Aggregated analytics and KPIs for general chat and sentiment visualizations.
- **`sse.py`**: Real-time progress broadcasting for long-running Celery background jobs.

### 2. Core Services Layer (`src/app/services/`)
- **`router_service.py`**: Multi-tiered RAG orchestrator with regex fast trap, Groq query routing, filter extraction, SQL generation, and SSE response streaming.
- **`vector_store.py`**: Abstract `VectorStore` interface and concrete `ChromaVectorStore` supporting `local` (PersistentClient) and `cloud` (CloudClient/HttpClient) modes.
- **`embedding_service.py`**: CPU-quantized INT8 ONNX embedder for `Davlan/afro-xlmr-mini` with attention mean-pooling and L2 normalization (384 dimensions).
- **`sentiment_service.py`**: CPU-quantized INT8 ONNX classifier for `JohnAlbarkaIbrahim/afroxlmr-mini-nigerian-sentiment` with dynamic `id2label` mapping.
- **`llm_factory.py`**: Resilient `ChatGroq` model factory with automatic fallback (`openai/gpt-oss-120b` $\rightarrow$ `openai/gpt-oss-20b`) and exponential backoff retry.
- **`summary_service.py`**: Groq-powered segment summarization and key topic extraction.
- **`cleanup_service.py`**: Background cleanup and vector deletion for deleted chats and expired accounts.

### 3. Background Workers Layer (`src/app/services/`)
- **`sentiment_worker.py`**: Celery worker consuming `sentiment` queue for message sentiment classification and segment aggregation.
- **`embedding_worker.py`**: Celery worker consuming `embeddings` queue for text segment vectorization and ChromaDB upsertion.

---

## ⚙️ Model Specifications & Routing Strategy

| Role | Provider / Engine | Model ID / Path | Dimensions / Quantization | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Sentiment Analysis** | Local ONNX Runtime CPU | `JohnAlbarkaIbrahim/afroxlmr-mini-nigerian-sentiment` | INT8 ONNX (`v1/onnx_int8/`) | Classifies Nigerian Pidgin & English messages into Positive, Negative, Neutral |
| **Dense Embeddings** | Local ONNX Runtime CPU | `Davlan/afro-xlmr-mini` | 384 dimensions, INT8 ONNX | Generates vector embeddings for chat segments and RAG semantic retrieval |
| **Router & SQL Agent** | Groq Cloud | `qwen/qwen3.6-27b` | Open Weights | Fast intent routing, metadata filter extraction, and safe SQL query generation |
| **Primary Synthesis** | Groq Cloud | `openai/gpt-oss-120b` | Open Weights | Master RAG answer generation, reasoning, and context synthesis |
| **Fallback Synthesis** | Groq Cloud | `openai/gpt-oss-20b` | Open Weights | Resilient failover LLM and conversational contextualizer |

---

## 🗄 Vector Storage Strategy

The backend uses a clean **`VectorStore` abstraction** allowing seamless switching between environments:

- **Local Mode (`CHROMA_MODE=local`)**:
  - Uses `chromadb.PersistentClient(path="./data/chroma")`.
  - Stores SQLite metadata and HNSW binary indices on disk for offline local development and unit testing.
- **Cloud Mode (`CHROMA_MODE=cloud`)**:
  - Uses `chromadb.CloudClient` / `HttpClient`.
  - Authenticates via `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE`.
  - Enables zero-filesystem, stateless container deployments on Heroku Eco Dynos.

---

## 🔐 Environment Configuration

Create a `.env` file in the project root:

```env
# =================================================================
# Database & Cache
# =================================================================
DATABASE_URL=postgresql+asyncpg://postgres.your-project:password@aws-0-eu-west-1.pooler.supabase.com:6543/postgres
CELERY_BROKER_URL=redis://localhost:6379/0

# =================================================================
# Groq LLM Configuration
# =================================================================
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL_PRIMARY=openai/gpt-oss-120b
GROQ_MODEL_FALLBACK=openai/gpt-oss-20b
GROQ_MODEL_ROUTER=qwen/qwen3.6-27b

# =================================================================
# Local ONNX Inference Models
# =================================================================
SENTIMENT_MODEL_REPO=JohnAlbarkaIbrahim/afroxlmr-mini-nigerian-sentiment
SENTIMENT_MODEL_SUBFOLDER=v1/onnx_int8
SENTIMENT_MODEL_DIR=./models/sentiment_onnx_int8

EMBEDDING_MODEL_REPO=Davlan/afro-xlmr-mini
EMBEDDING_MODEL_DIR=./models/afro_mini_onnx_int8

# =================================================================
# ChromaDB Vector Store
# =================================================================
CHROMA_MODE=cloud                  # 'local' or 'cloud'
CHROMA_COLLECTION_NAME=chat_embeddings
CHROMA_PERSIST_DIRECTORY=./data/chroma

# Chroma Cloud (Required when CHROMA_MODE=cloud)
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=your_tenant_id
CHROMA_DATABASE=your_database_name

# =================================================================
# Authentication & Security
# =================================================================
SECRET_KEY=your_secure_jwt_secret_key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_DAYS=7
```

---

## 🛠 Local Setup & Testing

### 1. Install Dependencies
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
# Run all unit tests
pytest tests/ -v

# Run configuration and vector store tests
pytest tests/test_config.py tests/test_vector_store.py -v
```

### 3. Run FastAPI Application
```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Run Celery Workers
```bash
# Linux/macOS
celery -A src.app.celery_app worker -Q sentiment,embeddings --loglevel=info

# Windows
celery -A src.app.celery_app worker -Q sentiment,embeddings --loglevel=info --pool=solo
```

---

## 🚢 Docker & Production Deployment (Heroku)

The backend is fully containerized and configured for Heroku Eco Dynos:

### Build and Run Locally with Docker
```bash
docker build -t whatsapp-sentiment-backend .
docker run -p 8000:8000 -e PORT=8000 --env-file .env whatsapp-sentiment-backend
```

### Deploy to Heroku Container Registry
```bash
heroku login
heroku container:login
heroku container:push web --app your-heroku-app-name
heroku container:release web --app your-heroku-app-name
```