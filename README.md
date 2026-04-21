# 🔍 RAG Knowledge Base QA System

> Hybrid Retrieval · Query Rewriting · Multi-turn Dialogue · Evaluation Loop

---

## 🚀 Features

| Module                  | Implementation                              | Description                                                              |
| ----------------------- | ------------------------------------------- | ------------------------------------------------------------------------ |
| **Hybrid Retrieval**    | ChromaDB + BM25 + RRF                       | Dense vector search + sparse BM25 → fused via Reciprocal Rank Fusion     |
| **Query Rewriting**     | LLM-based rewriting + multi-turn completion | Resolves coreference, expands keywords, optional sub-query decomposition |
| **Multi-turn Dialogue** | Session Manager (LRU)                       | Context truncation, session isolation, concurrent support                |
| **Evaluation Loop**     | LLM-as-Judge                                | Faithfulness / Answer Relevance / Context Precision / Overall Score      |
| **Chinese PDF Support** | PyMuPDF + jieba                             | Chinese-friendly parsing, semantic chunking, tokenization                |
| **Streaming Output**    | SSE / AsyncGenerator                        | Real-time token streaming                                                |
| **LLM Compatibility**   | Anthropic / OpenAI                          | Supports Claude and any OpenAI-compatible APIs                           |

---

## 📁 Project Structure

```
rag-project/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py              # Configuration management
│   ├── api/
│   │   ├── chat.py            # Chat API (GET/POST /api/chat)
│   │   ├── upload.py          # Upload API (POST /api/documents/upload)
│   │   └── eval.py            # Evaluation API (POST /api/eval)
│   ├── core/
│   │   ├── document_processor.py  # PDF parsing + smart chunking
│   │   ├── embedder.py            # Sentence-Transformers wrapper
│   │   ├── vector_store.py        # ChromaDB vector store
│   │   ├── bm25_store.py          # BM25 + jieba sparse index
│   │   ├── retriever.py           # RRF hybrid retriever
│   │   ├── query_rewriter.py      # Query rewriting module
│   │   ├── llm_client.py          # LLM client (Anthropic/OpenAI)
│   │   ├── generator.py           # RAG generator
│   │   ├── evaluator.py           # Evaluation module
│   │   └── session_manager.py     # Multi-turn session manager
│   └── models/
│       └── schemas.py             # Pydantic schemas
├── frontend/
│   └── index.html                # Built-in frontend UI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚡ Quick Start

### Option 1: Docker Deployment (Recommended)

**1. Clone & Setup**

```bash
git clone <repo-url>
cd rag-project

cp .env.example .env
```

**2. Configure `.env`**

```env
# Choose LLM provider
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx

# Or use OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-xxxxxxxx
# LLM_MODEL=gpt-4o
```

**3. Build & Run**

```bash
docker compose up -d --build

# View logs
docker compose logs -f rag-app
```

**4. Access Services**

* Frontend UI: [http://localhost:8000](http://localhost:8000)
* API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env

# Create data directories
mkdir -p data/chroma data/uploads

# Update .env paths
# CHROMA_PERSIST_DIR=./data/chroma
# UPLOAD_DIR=./data/uploads

# Run server
uvicorn app.main:app --reload --port 8000
```

---

## 📡 API Endpoints

### Upload Document

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@your_document.pdf"
```

### Chat (QA)

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key concepts mentioned in the document?",
    "session_id": "user-001",
    "enable_rewrite": true,
    "top_k": 5
  }'
```

### Streaming Chat

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain this technical solution in detail",
    "session_id": "user-001"
  }'
```

### Evaluate Answer Quality

```bash
curl -X POST http://localhost:8000/api/eval \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is a Transformer model?",
    "answer": "A Transformer is a neural network architecture based on attention...",
    "contexts": ["...retrieved document chunks..."],
    "ground_truth": "...reference answer (optional)..."
  }'
```

### Session Management

```bash
# Get history
curl http://localhost:8000/api/chat/history/user-001

# Clear history
curl -X DELETE http://localhost:8000/api/chat/history/user-001
```

---

## 🧠 RRF Algorithm

```
RRF(d) = Σᵢ  wᵢ / (k + rankᵢ(d))

Where:
  d      = document
  wᵢ     = weight of retriever i (default: vector:BM25 = 0.6:0.4)
  k      = smoothing constant (default: 60)
  rankᵢ  = rank of document in retriever i

Pipeline:
  Query → [Vector Search top-15]  ──┐
                                    ├─→ RRF Fusion → top-5 results
  Query → [BM25 Search top-15] ─────┘
```

---

## 🔄 Query Rewriting Pipeline

```
User: "What are its advantages?"
History: [User: What is RAG? Assistant: RAG is...]

        ↓ LLM rewriting

Rewritten:
"What are the main advantages of RAG (Retrieval-Augmented Generation)?"

        ↓ Retrieval

Sub-queries:
["RAG advantages", "benefits of retrieval-augmented generation", "RAG vs fine-tuning"]
```

---

## 📊 Evaluation Metrics

| Metric                | Meaning                                      | Method                                   |
| --------------------- | -------------------------------------------- | ---------------------------------------- |
| **Faithfulness**      | Is the answer grounded in retrieved context? | LLM-as-Judge                             |
| **Answer Relevance**  | Does the answer address the question?        | LLM-as-Judge                             |
| **Context Precision** | Are retrieved documents useful?              | LLM-as-Judge                             |
| **Context Recall**    | Is necessary info retrieved?                 | Keyword coverage (requires ground truth) |
| **Overall Score**     | Aggregated metric                            | Weighted average                         |


---

## 🛠 Tech Stack

* **Backend**: FastAPI + Uvicorn
* **Vector Store**: ChromaDB (persistent)
* **Sparse Retrieval**: rank-bm25 + jieba
* **Embedding**: paraphrase-multilingual-MiniLM-L12-v2
* **PDF Parsing**: PyMuPDF
* **LLM**: Anthropic Claude / OpenAI
* **Containerization**: Docker + Docker Compose

---

If you want, I can also help you **optimize this README for recruiter impact** (e.g., add system design highlights, QPS, scalability, and engineering depth) — that’s usually what makes it stand out for AI startup roles.
