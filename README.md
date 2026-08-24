# SentimentScope — Enterprise-Grade Agentic RAG & Multi-Lingual Sentiment Intelligence Engine

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=chroma&logoColor=white)](https://www.trychroma.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005C84?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![Celery](https://img.shields.io/badge/Celery-%23a9cc54.svg?style=for-the-badge&logo=celery&logoColor=fdd835)](https://docs.celeryq.dev/en/stable/)
[![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![Docker](https://img.shields.io/badge/Docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

**SentimentScope** is a high-throughput, cloud-native backend engine engineered for conversational data processing, low-latency multi-lingual NLP inference, and multi-tier agentic retrieval-augmented generation (RAG). 

The platform pairs **local INT8-quantized ONNX models running on CPU** (for zero external embedding latency and cost-effective sentiment classification) with a **Groq-accelerated multi-model LLM generation tier**, backed by an asynchronous event-driven Celery/Redis pipeline and scalable PostgreSQL/ChromaDB storage.

---

## 🏛 System Architecture & Topology

```mermaid
graph TD
    Client["Client Application / Web Frontend"]

    subgraph APILayer ["FastAPI Ingestion & Orchestration Layer"]
        AuthRouter["Auth & Security Router"]
        UploadRouter["Chat Ingestion & Stream Parser"]
        ChatRouter["Chat Lifecycle & Management"]
        RAGRouter["Agentic RAG Engine (SSE Stream)"]
        SSERouter["Live Progress Event Stream"]
        DashRouter["Analytics & Aggregations"]
    end

    subgraph ServiceLayer ["Orchestration & Business Logic Layer"]
        AuthSvc["JWT Auth & Security Service"]
        ParseSvc["Log Ingestion & Structuring Engine"]
        RouterSvc["Hierarchical Agentic Router"]
        RetrieverSvc["Scoped VectorStore Retriever"]
        SummarySvc["Groq LLM Summarization Service"]
        CleanupSvc["Asynchronous Lifecycle & Cleanup"]
    end

    subgraph InferenceLayer ["Local MLOps: Quantized ONNX Runtime (CPU)"]
        SentimentModel["Sentiment Classifier: JohnAlbarkaIbrahim/afroxlmr-mini-nigerian-sentiment (INT8 ONNX)"]
        EmbeddingModel["Dense Embedder: Davlan/afro-xlmr-mini (INT8 ONNX, 384d, L2 Norm)"]
    end

    subgraph GenerationLayer ["High-Throughput Groq LLM Fleet"]
        GroqRouter["Router & SQL Generator: qwen/qwen3.6-27b"]
        GroqPrimary["Primary Synthesis Engine: openai/gpt-oss-120b"]
        GroqFallback["Resilient Fallback Engine: openai/gpt-oss-20b"]
    end

    subgraph StorageLayer ["Persistence & Feature Store Layer"]
        PostgreSQL[("PostgreSQL: Relational OLTP & Analytics")]
        Redis[("Redis: Celery Broker & Pub/Sub")]
        VectorDB[("ChromaDB: Local Persistent / Chroma Cloud")]
    end

    subgraph WorkerLayer ["Asynchronous Distributed Worker Fleet (Celery)"]
        SentimentWorker["Batch Sentiment Classification Worker"]
        EmbeddingWorker["Vector Embedding & Indexing Worker"]
    end

    Client --> AuthRouter
    Client --> UploadRouter
    Client --> ChatRouter
    Client --> RAGRouter
    Client --> SSERouter
    Client --> DashRouter

    UploadRouter --> ParseSvc
    ParseSvc --> PostgreSQL

    ChatRouter -->|Dispatch Task| Redis
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
    GroqPrimary -. Automatic Failover .-> GroqFallback
    RouterSvc --> PostgreSQL
```

---

## ⚡ Core Engineering Highlights

### 1. Data Engineering & Asynchronous ETL Pipelines
- **Stream-Safe Ingestion:** Custom regex streaming parser processes WhatsApp conversation logs, normalizing variations in ISO timestamps, sender aliases, systemic system notifications, and multi-line message boundaries.
- **Relational Dimensional Modeling:** Deconstructs chats into normalized schemas: `chats`, `participants`, `messages`, `time_segments`, `sender_segments`, and granular sentiment scores.
- **Event-Driven Distributed Workloads:** Celery workers process long-running batch NLP jobs across dedicated queues (`sentiment`, `embeddings`), publishing real-time telemetry back to clients via Redis Pub/Sub and SSE.

### 2. High-Performance Local MLOps (CPU-Quantized ONNX)
- **Zero API Dependency for Embeddings:** Instead of relying on rate-limited, expensive external embedding APIs, the system utilizes local **`Davlan/afro-xlmr-mini` INT8 ONNX**.
- **Vector Pooling & Normalization:** Performs attention-masked mean pooling followed by $L_2$ vector normalization to generate deterministic 384-dimensional dense vectors:
  $$\mathbf{e} = \frac{\sum_{i=1}^{L} m_i \mathbf{h}_i}{\sum_{i=1}^{L} m_i}, \quad \hat{\mathbf{e}} = \frac{\mathbf{e}}{\|\mathbf{e}\|_2}$$
- **Domain-Adapted Multi-Lingual Sentiment Classifier:** Integrated **`JohnAlbarkaIbrahim/afroxlmr-mini-nigerian-sentiment`** INT8 ONNX model, purpose-trained for Nigerian Pidgin, Hausa, Yoruba, Igbo, and English mixed-code conversational semantics.

### 3. Multi-Tier Agentic RAG Architecture
- **Hierarchical Request Routing:** Employs a multi-stage decision graph to route queries with minimal compute overhead:
  - **Tier 1 (Fast Trap):** Sub-millisecond regex matching for conversational greetings and generic boilerplate.
  - **Tier 2 (Analytical SQL):** Dynamic translation of quantitative queries (e.g., *"Who sent the most messages in October?"*) into parameterized, safe PostgreSQL queries.
  - **Tier 3 (Scoped Semantic Retrieval):** Extraction of temporal and sender metadata filters to execute filtered dense vector search against ChromaDB.
  - **Tier 4 (Grounded Anti-Leak Synthesis):** Assembles verified tabular results and top-$K$ semantic context chunks into a grounded LLM prompt, enforcing strict anti-leak and hallucination guardrails.

---

## 🔄 End-to-End RAG Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor User as Client
    participant API as FastAPI Router
    participant LLMRouter as Groq Router (qwen3.6-27b)
    participant SQL as PostgreSQL (OLTP/Analytics)
    participant Embed as Local ONNX Embedder (384d)
    participant Chroma as ChromaDB Vector Store
    participant Synth as Groq Synthesizer (gpt-oss-120b)

    User->>API: POST /chat/{id}/query/streamed (Question + Analytics State)
    
    rect rgb(240, 248, 255)
        Note over API: Tier 1: Fast Trap (Regex Greeting Check)
        API-->>User: Instant greeting stream (if greeting matched)
    end

    API->>LLMRouter: Contextualize & Classify Intent (SQL / Vector / Dashboard / Hybrid)
    
    alt Quantitative / Aggregation Query
        API->>LLMRouter: Generate Safe Parameter-Bound SQL (:chat_id)
        LLMRouter-->>API: Validated SQL Query
        API->>SQL: Execute Read-Only Query with :chat_id binding
        SQL-->>API: Tabular Aggregation Results
    end

    alt Semantic / Conversational Search Query
        API->>LLMRouter: Extract Date Ranges & Sender Metadata Filters
        LLMRouter-->>API: Structured Metadata Filters
        API->>Embed: Compute Query Dense Vector (Davlan/afro-xlmr-mini INT8)
        Embed-->>API: 384d Normalized Embedding
        API->>Chroma: Execute Query with Vector + Scoped Metadata Filter
        Chroma-->>API: Top-K Grounded Context Passages
    end

    API->>Synth: Stream Grounded Context with Strict Anti-Leak System Prompt
    Synth-->>User: Server-Sent Events (SSE) Token Stream (`data: "..."`)
    API->>SQL: Asynchronously Persist Turn & Cited Document Sources
```

---

## 🔬 Model Fleet & Inference Matrix

| Component | Architecture / Provider | Model Identifier | Precision & Dimensions | Operational Role |
| :--- | :--- | :--- | :--- | :--- |
| **Sentiment Inference** | ONNX Runtime CPU | `JohnAlbarkaIbrahim/afroxlmr-mini-nigerian-sentiment` | INT8 Quantized (`v1/onnx_int8/`) | Classifies Nigerian Pidgin & English messages into Positive, Negative, and Neutral. |
| **Dense Vector Embeddings** | ONNX Runtime CPU | `Davlan/afro-xlmr-mini` | 384-dimensional, INT8 Quantized | Local vectorization of chat segments for semantic similarity and hybrid retrieval. |
| **Query Intent & SQL Router** | Groq Cloud Engine | `qwen/qwen3.6-27b` | Open Weights | Rapid intent classification, filter parameter extraction, and safe SQL statement synthesis. |
| **Primary RAG Synthesizer** | Groq Cloud Engine | `openai/gpt-oss-120b` | Open Weights | Master reasoning engine generating coherent, source-attributed conversational answers. |
| **Fallback & Contextualizer** | Groq Cloud Engine | `openai/gpt-oss-20b` | Open Weights | Conversational turn re-writing and automated failover layer during network spikes. |

---

## 🗃 Storage & Persistence Design

```
                                  ┌──────────────────────────────┐
                                  │   Application Service Layer  │
                                  └──────────────┬───────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
        ┌───────────────────────────────┐                 ┌───────────────────────────────┐
        │  PostgreSQL (Relational OLTP) │                 │      VectorStore Interface    │
        ├───────────────────────────────┤                 ├───────────────────────────────┤
        │ • chats                       │                 │ Local Mode: Persistent SQLite │
        │ • participants                │                 │ Cloud Mode: Chroma Cloud API  │
        │ • messages & sentiments       │                 │ • 384d Dense Vector Search    │
        │ • time_segments & metrics     │                 │ • Scoped $and / $or Metadata  │
        │ • chat_history & citations    │                 │ • Per-Chat Index Isolation    │
        └───────────────────────────────┘                 └───────────────────────────────┘
```

- **Relational Storage:** Relational integrity enforced via SQLAlchemy models, index optimization on `(chat_id, timestamp)`, and foreign-key cascading.
- **Abstract Vector Storage Layer:** Polymorphic `VectorStore` ABC decoupling business logic from underlying vector database implementations, supporting seamless migration between **Local Persistent ChromaDB** (`./data/chroma`) and **Multi-Tenant Chroma Cloud**.

---

## 🛡 Security Threat Modeling & Prompt Injection Defense

1. **SQL Injection Defense:**
   - LLM-generated queries are strictly validated against an AST-level syntax filter.
   - Prohibits destructive DDL/DML keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `GRANT`).
   - Restricts queries to whitelist tables (`messages`, `time_segments`, `participants`, `message_sentiments`).
   - Enforces parameter binding (`:chat_id`) at execution time; cross-tenant query attempts are systematically rejected.
2. **System Prompt Hardening:**
   - Enforces strict role boundary isolation to prevent prompt jailbreaks.
   - Enforces explicit ground-truth citation rules to eliminate hallucinated sources.

---

## 🔮 Future Architectural Roadmap: Deterministic Agentic Tool Calling

To further elevate system safety, eliminate unstructured prompt parsing, and defend against advanced prompt injection vectors, the following architectural upgrades are actively planned:

```mermaid
graph LR
    UserQuery["User Input Query"] --> AgentSupervisor["Agent Supervisor / LangGraph / LangChain"]
    
    subgraph DeterministicToolExecution ["Deterministic Structured Tool Dispatch"]
        AgentSupervisor -->|Strict Pydantic Schema| ToolSQL["SQL Query Tool (Safe Sandboxed Read Engine)"]
        AgentSupervisor -->|Strict Pydantic Schema| ToolVector["Vector Retrieval Tool (Chroma Filter Engine)"]
        AgentSupervisor -->|Strict Pydantic Schema| ToolDash["Dashboard Metric Tool (Aggregator Engine)"]
    end

    ToolSQL --> ExecutionGuard["AST Query Validator & RBAC Sandbox"]
    ToolVector --> ChromaIndex["ChromaDB Scoped Collection"]
    ToolDash --> MetricEngine["Pre-aggregated Analytics Store"]
```

### Planned Tool Calling Capabilities:
1. **Pydantic-Enforced Function Calling:**
   - Migrate SQL generation and dashboard exploration to native Groq tool calling (`tools` API).
   - Replaces JSON string parsing with strongly-typed parameter validation schemas.
2. **Deterministic SQL Tool Sandbox:**
   - Encapsulate database interactions inside an isolated `ExecuteSafeSQLQuery` tool.
   - Tool verifies tenant isolation cryptographically before submitting queries through read-only database connections.
3. **Multi-Agent Supervisor Hierarchy:**
   - Implement stateful agent graphs (e.g., using LangGraph) to orchestrate multi-step reasoning:
     - Step 1: Tool Selection & Argument Validation
     - Step 2: Parallel Tool Execution (SQL aggregation + Semantic Search)
     - Step 3: Synthesis & Hallucination Verification

---

## 📂 Repository Structure

```
├── src/
│   ├── app/
│   │   ├── api/                     # FastAPI Route Definitions
│   │   │   ├── auth.py              # User registration & JWT authentication
│   │   │   ├── chat.py              # Chat lifecycle & task orchestration
│   │   │   ├── uploads.py           # Streamed file uploads & parser trigger
│   │   │   ├── query.py             # Server-Sent Events (SSE) RAG query stream
│   │   │   ├── dashboard.py         # Tabular aggregations & KPI metrics
│   │   │   └── sse.py               # Celery task progress broadcasting
│   │   ├── services/                # Core Business Logic & Inferences
│   │   │   ├── router_service.py    # Multi-tier RAG routing & synthesis
│   │   │   ├── vector_store.py      # Abstract VectorStore & ChromaDB implementation
│   │   │   ├── embedding_service.py # Local INT8 ONNX dense embedder (384d)
│   │   │   ├── sentiment_service.py # Local INT8 ONNX Nigerian sentiment classifier
│   │   │   ├── llm_factory.py       # Groq model factory with exponential fallback
│   │   │   ├── summary_service.py   # Groq-powered segment summarization
│   │   │   ├── parser.py            # Stream-based WhatsApp chat log parser
│   │   │   ├── sentiment_worker.py  # Celery batch sentiment processing worker
│   │   │   ├── embedding_worker.py  # Celery batch vector embedding worker
│   │   │   └── cleanup_service.py   # User and chat lifecycle data pruning
│   │   ├── db/                      # SQLAlchemy Engine, Base, and Session
│   │   ├── models/                  # Relational ORM Database Models
│   │   ├── schemas/                 # Pydantic Request / Response Contracts
│   │   ├── config.py                # Pydantic Settings & Environment Validation
│   │   ├── celery_app.py            # Celery Distributed Task Queue Instance
│   │   └── main.py                  # Application Factory & Middleware Configuration
├── tests/                           # Pytest Test Suite
│   ├── test_config.py               # Settings & environment validation tests
│   ├── test_vector_store.py         # ChromaDB Local CRUD & inference tests
│   └── live/                        # Integration test harnesses
├── Dockerfile                       # Multi-stage production container specification
├── start.sh                         # Dynamic runtime startup script
├── requirements.txt                 # Pinned application dependencies
├── .env.example                     # Environment template with parameter documentation
└── README.md                        # Architectural documentation
```

---

## ⚙️ Environment Configuration

Refer to [`.env.example`](./.env.example) for the full list of configuration options:

```env
# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/sentiment_db

# ChromaDB Vector Store
CHROMA_MODE=local                      # 'local' or 'cloud'
CHROMA_COLLECTION_NAME=sentiment-scope
CHROMA_PERSIST_DIRECTORY=./data/chroma

# Chroma Cloud (Required when CHROMA_MODE=cloud)
CHROMA_HOST=api.trychroma.com
CHROMA_API_KEY=ck-your_chroma_api_key
CHROMA_TENANT=your_tenant_id
CHROMA_DATABASE=your_database_name

# Groq LLM Engine
GROQ_API_KEY=gsk_your_groq_api_key
GROQ_MODEL_PRIMARY=openai/gpt-oss-120b
GROQ_MODEL_FALLBACK=openai/gpt-oss-20b
GROQ_MODEL_ROUTER=qwen/qwen3.6-27b

# Local ONNX Models
SENTIMENT_MODEL_REPO=JohnAlbarkaIbrahim/afroxlmr-mini-nigerian-sentiment
SENTIMENT_MODEL_SUBFOLDER=v1/onnx_int8
EMBEDDING_MODEL_REPO=Davlan/afro-xlmr-mini

# Celery & Redis
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Security
SECRET_KEY=your_jwt_secret_key
JWT_ALGORITHM=HS256
```

---

## 🚀 Setup & Local Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/JohnJodinho/whatsapp-sentiment-backend.git
cd whatsapp-sentiment-backend

# Initialize virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Start Development Server
```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Launch Celery Workers
```bash
# Unix / macOS:
celery -A src.app.celery_app worker -Q sentiment,embeddings --loglevel=info

# Windows (Single-Process Pool):
celery -A src.app.celery_app worker -Q sentiment,embeddings --loglevel=info --pool=solo
```

### 5. Run via Docker
```bash
docker build -t sentimentscope-backend .
docker run -p 8000:8000 --env-file .env sentimentscope-backend
```