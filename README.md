# Knowledge Agent

[中文](README_zh.md)

A personal knowledge agent with persistent memory, built on LangGraph. It can research topics via web search, ingest documents into a vector knowledge base, and recall stored knowledge across sessions.

> Inspired by [Letta](https://github.com/letta-ai/letta)'s three-tier memory architecture and [Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)'s agent workflow design.

## Features

- **Three-Tier Memory System** — Core Memory (always in context), Archival Memory (vector long-term storage), Recall Memory (conversation history)
- **Hybrid Search + Re-ranking** — Archival retrieval combines vector similarity (DashScope embeddings) with BM25 keyword matching (PostgreSQL tsvector + GIN index), followed by cross-encoder re-ranking (`BAAI/bge-reranker-v2-m3`) for improved result ordering
- **Core Memory Auto-Compression** — Blocks approaching their character limit are automatically distilled by the LLM, preserving key facts while staying within bounds
- **Deep Research** — Multi-loop web search with automatic gap analysis and follow-up queries
- **Document Ingestion** — Import URLs, PDFs, or plain text; automatic chunking and vectorization. Supports two strategies: fixed-size splitting (paragraph-first, 800-char chunks) and **semantic chunking** (embedding-based boundary detection using cosine similarity between adjacent sentences)
- **Intent Routing** — LLM-classified modes: Chat, Research, Recall, Memory Edit, Ingest
- **Chat with Knowledge Retrieval** — Chat mode automatically searches Archival Memory for relevant context before generating a response
- **Persistent Knowledge** — Research findings are stored in PostgreSQL + pgvector and survive across sessions
- **Memory Pipeline (mem0-style)** — Automatic fact extraction, cosine dedup, conflict resolution, and consolidation after every conversation turn
- **Document Library** — Full document storage (documents table) separate from chunked retrieval index, with document-level browsing and dedup
- **Conversation History Sidebar** — Left panel for switching between past conversation threads
- **Smart Deduplication** — LLM-assisted batch dedup for archival memory with cosine similarity candidate filtering
- **Full-Stack UI** — React frontend with real-time agent activity visualization

## Architecture

### Three-Tier Memory

```
┌─────────────────────────────────────────────────────┐
│                    Agent State                       │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Core Memory  │  │ Archival Mem  │  │ Recall Mem │ │
│  │ (in context) │  │ (long-term)   │  │ (history)  │ │
│  │              │  │              │  │            │ │
│  │ persona      │  │ pgvector     │  │ pgvector   │ │
│  │ human        │  │ cosine sim   │  │ semantic   │ │
│  │ knowledge_   │  │ + metadata   │  │ search     │ │
│  │   focus      │  │              │  │            │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
```

- **Core Memory** — Editable blocks injected into the system prompt every turn. The agent can self-edit them. Blocks approaching their character limit (≥80%) are automatically compressed by the LLM to stay within bounds.
- **Archival Memory** — Long-term semantic storage in PostgreSQL + pgvector. Research findings and ingested documents are stored as 1024-dim embeddings. Retrieval uses hybrid search: vector cosine similarity (70% weight) combined with BM25 keyword matching (30% weight) via PostgreSQL `tsvector` + GIN index.
- **Recall Memory** — Conversation history with semantic search via pgvector. Automatically populated after each turn and queried during recall_memory for cross-session context.

### Agent Graph

```
START → route_intent
  ├── "ingest" → ingest_document → save_to_archival → respond → memory_pipeline → [consolidate?] → END
  └── (其他)   → recall_memory → evaluate_recall
                   ├── (memory sufficient) → respond → memory_pipeline → [consolidate?] → END
                   └── (memory insufficient) → generate_query → [web_research × N] → reflection
                         ├── (gaps) → [web_research] → reflection (loop)
                         └── (sufficient) → save_to_archival → respond → memory_pipeline → [consolidate?] → END
```

After `recall_memory`, the `evaluate_recall` node uses an LLM to judge whether the retrieved memory content is sufficient to answer the user's question. If sufficient, it skips web search and responds directly. If insufficient, it proceeds to web research. The mem0-style `memory_pipeline` extracts facts after every response, deduplicates against existing archival memory, and upserts new knowledge. Consolidation (Union-Find clustering + LLM merge) runs every N turns to deduplicate stored facts.

### Document Ingestion Pipeline

```
User provides URL / plain text / file path
  ↓
ingest_document node
  ├── URL     → httpx fetch → BeautifulSoup extract
  ├── Text    → direct processing
  └── File    → read content
  ↓
Chunking (configurable strategy)
  ├── "fixed"    → chunk_text(): paragraph → sentence → character splitting (800 chars, 100 overlap)
  └── "semantic" → chunk_text_semantic(): sentence split → embed → cosine similarity → break at low-sim points
                   ├── Merge segments below min_chunk_size (200 chars)
                   ├── Split segments above max_chunk_size (1500 chars)
                   └── Add overlap between consecutive chunks
  ↓
Documents table (full text) + ArchivalMemory (chunks with document_id)
  ↓
DashScope Embedding batch vectorization
  ↓
PostgreSQL + pgvector (dedup via UNIQUE INDEX on title+source)
```

**Semantic Chunking** detects topic shifts by computing cosine similarity between adjacent sentence embeddings. When similarity drops below a threshold (default 0.5), a chunk boundary is inserted. This produces chunks that are semantically coherent rather than arbitrarily sized. Configure via environment variables: `CHUNK_STRATEGY=semantic`, `CHUNK_SIMILARITY_THRESHOLD=0.5`, `CHUNK_MIN_SIZE=200`, `CHUNK_MAX_SIZE=1500`.

## Tech Stack

| Component | Technology | Notes |
|---|---|---|
| Agent Framework | LangGraph StateGraph | State-driven agent workflow |
| LLM | Any OpenAI-compatible API | Configurable via `.env` |
| Embedding | DashScope `text-embedding-v3` | 1024-dim vectors |
| Vector Store | PostgreSQL 16 + pgvector | Docker deployment, cosine similarity |
| Full-Text Search | PostgreSQL tsvector + GIN | BM25 keyword matching, hybrid with vector |
| Web Search | Tavily API | Advanced search mode |
| Document Processing | pypdf + BeautifulSoup + httpx | PDF / web / text extraction |
| Frontend | React 19 + Vite 6 + Tailwind CSS 4 | shadcn/ui components |

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose

### 1. Start the Database

```bash
cd backend
docker compose up -d
```

This starts PostgreSQL 16 with the pgvector extension on port 5432.

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# LLM (OpenAI-compatible API — works with OpenAI, DeepSeek, MiMo, etc.)
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Embedding (DashScope)
DASHSCOPE_API_KEY=your-dashscope-key
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v3

# Web Search (Tavily)
TAVILY_API_KEY=your-tavily-key

# PostgreSQL
DATABASE_URL=postgresql://knowledge_agent:knowledge_agent@localhost:5432/knowledge_agent

# LangSmith (optional — for tracing & observability)
LANGSMITH_API_KEY=your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=knowledge-agent
```

### 3. Install Backend Dependencies

```bash
cd backend
pip install -e .
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 5. Start the Services

**Backend (LangGraph dev server):**

```bash
cd backend
langgraph dev
```

Runs at `http://localhost:2024` by default.

**Frontend (Vite dev server):**

```bash
cd frontend
npm run dev
```

Runs at `http://localhost:5173` by default.

**CLI mode (no frontend needed):**

```bash
cd backend
python examples/cli_chat.py
```

### 6. LangSmith Tracing (Optional)

[LangSmith](https://smith.langchain.com/) provides tracing and observability for your agent. To enable:

1. Create an account at [smith.langchain.com](https://smith.langchain.com/) and get your API key
2. Set the env vars in `.env`:
   ```env
   LANGSMITH_API_KEY=lsv2_pt_xxxxx
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=knowledge-agent
   ```
3. Restart `langgraph dev` — all LLM calls, tool invocations, and graph executions will be traced automatically

View traces at `https://smith.langchain.com/` under your project. Each run shows the full execution graph with node-level timing, input/output, and token usage.

## Testing & Evaluation

### Recall Benchmark

Tests hybrid search (vector + BM25) recall on a small built-in dataset:

```bash
cd backend
python scripts/test_recall.py
```

### LoCoMo Evaluation

[LoCoMo](https://github.com/snap-research/locomo) is a long conversation memory benchmark. It measures Recall@K (R@1, R@3, R@5, R@10) across 5 question types: single-session, multi-session, temporal, adversarial, and event-summary.

**Setup:** Download the dataset (not committed to git):

```bash
cd backend
mkdir -p data
# Option A: HuggingFace (requires network access to huggingface.co)
python -c "from datasets import load_dataset; ds = load_dataset('KimmoZZZ/locomo', split='test'); import json; json.dump([dict(d) for d in ds], open('data/locomo.json','w'), ensure_ascii=False, indent=2)"

# Option B: Use hf-mirror.com (for restricted networks)
HF_ENDPOINT=https://hf-mirror.com python -c "from datasets import load_dataset; ds = load_dataset('KimmoZZZ/locomo', split='test'); import json; json.dump([dict(d) for d in ds], open('data/locomo.json','w'), ensure_ascii=False, indent=2)"
```

**Run evaluation:**

```bash
# Quick test — baseline strategy, 1 sample (~2min)
python scripts/test_locomo.py --local data/locomo.json --limit 1

# Full dataset — baseline strategy (raw turns, no LLM extraction)
python scripts/test_locomo.py --local data/locomo.json

# Compare all 3 strategies (slow — requires LLM calls for extraction)
python scripts/test_locomo.py --local data/locomo.json --strategy all

# Single strategy
python scripts/test_locomo.py --local data/locomo.json --strategy baseline
python scripts/test_locomo.py --local data/locomo.json --strategy extracted
python scripts/test_locomo.py --local data/locomo.json --strategy hybrid
```

**Strategies:**
- **baseline** — Raw conversation turns stored directly (fast, no LLM calls)
- **extracted** — Facts extracted via LLM, then stored (tests mem0 pipeline)
- **hybrid** — Both raw turns and extracted facts in the same index

Output is a Recall@K table broken down by question type. Requires PostgreSQL to be running (`docker compose up -d`).

### Archival Memory Management

Manage research summaries, ingested documents, and other archival entries:

```bash
cd backend

# Show entry counts and age per namespace
python scripts/manage_archival.py stats

# List entries in a namespace
python scripts/manage_archival.py list --namespace research
python scripts/manage_archival.py list --namespace ingested --limit 10

# Find and merge duplicate entries (cosine similarity >= 0.85)
python scripts/manage_archival.py dedup --namespace research --dry-run
python scripts/manage_archival.py dedup --namespace research

# Remove entries older than N days
python scripts/manage_archival.py cleanup --namespace research --days 30 --dry-run
python scripts/manage_archival.py cleanup --namespace research --days 30

# Full consolidation (dedup + merge via LLM)
python scripts/manage_archival.py consolidate --namespace research

# Smart dedup (cosine candidates + batched LLM judgment)
python scripts/manage_archival.py smart-dedup --namespace research --dry-run
python scripts/manage_archival.py smart-dedup --namespace research

# Delete all entries in a namespace
python scripts/manage_archival.py drop --namespace locomo_baseline --dry-run
python scripts/manage_archival.py drop --namespace locomo_baseline
```

## Project Structure

```
knowledge-agent/
├── backend/
│   ├── pyproject.toml
│   ├── docker-compose.yml            # PostgreSQL + pgvector
│   ├── .env.example
│   ├── langgraph.json
│   ├── examples/
│   │   └── cli_chat.py               # CLI entry point
│   ├── scripts/
│   │   ├── test_recall.py            # Hybrid search recall benchmark
│   │   ├── test_locomo.py            # LoCoMo dataset evaluation (Recall@K)
│   │   └── manage_archival.py        # Archival Memory management (stats/dedup/cleanup)
│   └── src/agent/
│       ├── graph.py                  # Main LangGraph definition
│       ├── state.py                  # AgentState TypedDict
│       ├── configuration.py          # Runtime configuration
│       ├── prompts.py                # Prompt templates
│       ├── memory/
│       │   ├── block.py              # Block model (inspired by Letta)
│       │   ├── core_memory.py        # CoreMemory + compile()
│       │   └── tools.py              # Memory edit + archival tools
│       ├── storage/
│       │   ├── embedding.py          # DashScope embedding wrapper
│       │   ├── archival.py           # ArchivalMemory (pgvector) + documents table
│       │   ├── recall.py             # RecallMemory
│       │   ├── reranker.py           # Cross-encoder re-ranking (BAAI/bge-reranker-v2-m3)
│       │   └── ingestion.py          # Document ingestion pipeline
│       └── nodes/
│           ├── memory_manager.py     # Intent routing + archival save
│           ├── memory_pipeline.py    # mem0-style extract → dedup → upsert
│           ├── researcher.py         # Search + research + reflection
│           └── responder.py          # Response generation
└── frontend/
    └── src/
        ├── App.tsx                   # Main app with LangGraph streaming + thread management
        └── components/
            ├── InputForm.tsx         # Mode selector (Chat/Research/Memory/Ingest)
            ├── WelcomeScreen.tsx
            ├── ChatMessagesView.tsx
            ├── DocumentLibrary.tsx   # Knowledge base browser (documents + archival entries)
            ├── ThreadSidebar.tsx     # Conversation history sidebar
            └── ActivityTimeline.tsx
```

## CLI Commands

| Command | Description |
|---|---|
| `/memory` | View Core Memory blocks |
| `/search <query>` | Search Archival Memory |
| `/ingest <url_or_path>` | Ingest a document (URL or file path) |
| `/quit` | Exit |

## Roadmap

- [x] **Cross-encoder Re-ranking** — Added `BAAI/bge-reranker-v2-m3` re-ranking stage after hybrid retrieval, with automatic fallback to cosine-only ordering when the model is unavailable.
- [x] **Recall Memory Integration** — Conversation history stored in recall_memory with semantic search, displayed in Document Library's Recall tab, and included in hybrid archival retrieval.
- [x] **Document Library Output** — Full document storage (documents table) with document-level browsing, namespace filtering, and dedup via unique index. Ingested documents show title, chunk count, and expandable full content.
- [x] **Conversation History Sidebar** — Left panel with thread listing via LangGraph SDK, supporting thread switching and new chat creation.
- [x] **Smart Deduplication** — `manage_archival.py smart-dedup` command: cosine similarity candidate filtering (≥0.85) with batched LLM judgment (5 pairs/call) and auto-merge for high-confidence duplicates (≥0.95).
- [x] **Improved Text Chunking** — Added embedding-based semantic chunking (`chunk_text_semantic`) as an alternative to fixed-size splitting. Computes cosine similarity between adjacent sentence embeddings and breaks at semantic boundaries (low similarity points). Configurable threshold, min/max chunk size, and overlap. Selectable via `CHUNK_STRATEGY=semantic` env var or `strategy` parameter.
- [ ] **Memory Pipeline Optimization** — Current mem0-style pipeline (extract → dedup → upsert) runs after every response turn, adding latency. Fact extraction quality depends heavily on LLM capability; dedup uses a fixed cosine threshold (0.8) which may miss near-duplicates or over-merge distinct facts. Explore: batched/async pipeline execution, adaptive dedup thresholds, fact confidence scoring, and conflict resolution strategies beyond simple merge/skip.

## Acknowledgments

This project draws design inspiration from:

- **[Letta](https://github.com/letta-ai/letta)** (formerly MemGPT) — The three-tier memory architecture (Core / Archival / Recall), Block-based memory model, and self-editing memory concept are adapted from Letta's design. The implementation is independent and simplified for LangGraph usage.
- **[Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)** — The fullstack agent pattern (LangGraph backend + React frontend with streaming), research loop with reflection, and activity timeline UI are inspired by this Google Gemini quickstart project.

## License

[MIT](LICENSE)
