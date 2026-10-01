# Production-Grade Agentic RAG Assistant

A complete production-ready Agentic Knowledge Assistant built with **LangChain**, **LangGraph**, **Qdrant Cloud**, **EURI LLM**, and a rich interactive frontend.

## Architecture Overview

```
Query → Understand → Route → Retrieve (Hybrid) → Grade (CRAG+Self-RAG) → Generate → Evaluate → Final Answer
                  ↓                     ↑                                        ↓
               Web/Calc            Rewrite Query ←─────────────────────── Poor Quality
```

### Key Technologies

| Component | Technology |
|-----------|-----------|
| LLM | EURI API (`gemini-3.5-flash-lite`) |
| Framework | LangChain + LangGraph |
| Vector DB | Qdrant Cloud |
| Checkpoints | MongoDB |
| PDF Extraction | Docling + PyMuPDF |
| Embeddings | EURI Embeddings API |
| BM25 Search | rank_bm25 |
| Reranking | sentence-transformers CrossEncoder |
| Tracing | LangSmith |
| API | FastAPI |

### Advanced RAG Concepts Implemented

- **CRAG** (Corrective RAG): Evaluates retrieval quality, falls back to web search if docs are irrelevant
- **Self-RAG**: IsRET (should retrieve?), IsREL (relevant?), IsSUP (supported?), IsUSE (useful?)
- **Hybrid Search**: BM25 + Semantic (vector) with configurable alpha blending
- **RRF** (Reciprocal Rank Fusion): Merges ranked lists from multiple retrievers
- **Cross-Encoder Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM Reranking**: Optional LLM-based relevance scoring
- **Multiple Chunking**: Fixed, Recursive, Semantic, Sentence, Markdown, Paragraph
- **Metadata-Aware Retrieval**: Page numbers, file hashes, source tracking

### LangGraph Workflow Nodes

1. `understand_query` — Intent analysis, key concept extraction
2. `route` — Decides tool: vectorstore / web search / calculator / direct LLM
3. `retrieve` — Hybrid search (BM25 + semantic) → RRF → cross-encoder rerank
4. `web_search` — Tavily/DuckDuckGo fallback
5. `calculator` — Safe math evaluation
6. `grade_documents` — CRAG + Self-RAG IsREL grading
7. `generate` — RAG generation with inline citations
8. `evaluate` — Hallucination + answer relevance + Self-RAG IsSUP/IsUSE
9. `rewrite_query` — LLM query reformulation (max 3 retries)
10. `final_answer` — Format and return with citations

## Installation

### Prerequisites
- Python 3.11+
- Qdrant Cloud account (free tier available)
- MongoDB Atlas or local MongoDB
- EURI API key (euron.one)
- LangSmith account (optional, for tracing)
- Tavily API key (optional, for web search)

### Setup

```bash
# 1. Clone and enter directory
git clone <your-repo-url>
cd production-agentic-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Start the server
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Windows Quick Start
```
start.bat
```

### Docker
```bash
cp .env.example .env
# Edit .env
docker-compose up -d
```

## Configuration (`.env`)

```env
# Required
EURI_API_KEY=your_euri_api_key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_key
MONGODB_URI=mongodb+srv://...

# Optional
LANGCHAIN_API_KEY=your_langsmith_key   # For tracing
TAVILY_API_KEY=your_tavily_key         # For web search
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/v1/health` | Health check |
| `GET`  | `/api/v1/health/components` | Component health |
| `POST` | `/api/v1/ingest/upload` | Upload & ingest documents |
| `GET`  | `/api/v1/ingest/strategies` | List chunking strategies |
| `POST` | `/api/v1/query` | Run agentic RAG query |
| `POST` | `/api/v1/query/stream` | Streaming SSE query |
| `GET`  | `/api/v1/query/history/{session_id}` | Session history |

### Example Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the uploaded documents?"}'
```

### Example Upload

```bash
curl -X POST http://localhost:8000/api/v1/ingest/upload \
  -F "file=@document.pdf" \
  -F "chunking_strategy=recursive" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200"
```

## Frontend

- **`/`** — Main UI with file upload, chat interface, and execution trace
- **`/architecture`** — Interactive architecture visualizer with:
  - SVG workflow graph (clickable nodes)
  - Step-by-step playback with 4 demo scenarios
  - State variable tracker
  - Live query demo
  - Offline mode (fully embedded demo data)

## Project Structure

```
├── backend/
│   └── app/
│       ├── core/          # Config, LLM client, Embeddings
│       ├── ingestion/     # Docling + PyMuPDF + Chunking strategies
│       ├── retrieval/     # Qdrant, BM25, Hybrid Search, RRF, Reranker
│       ├── rag/           # CRAG, Self-RAG, Graders
│       ├── agents/        # LangGraph State, Nodes, Graph, Tools
│       ├── checkpoints/   # MongoDB checkpoint saver
│       └── api/routes/    # FastAPI routes
├── frontend/
│   ├── index.html         # Main UI
│   └── architecture.html  # Architecture visualizer
├── .env.example
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## LangSmith Tracing

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to enable full LangSmith tracing. Each query creates a traced run showing all node executions, LLM calls, and grading decisions.

## Submission

- **GitHub**: Push this repository (never commit `.env` — it's in `.gitignore`)
- **YouTube**: Record a walkthrough covering:
  1. Architecture overview (architecture.html)
  2. Document ingestion pipeline
  3. Hybrid search + RRF + reranking
  4. LangGraph state and node transitions
  5. CRAG and Self-RAG in action
  6. Hallucination grading
  7. Live end-to-end demo

## License

MIT
