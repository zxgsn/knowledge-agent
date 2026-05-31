# Knowledge Agent

[中文](README_zh.md)

A personal knowledge agent with persistent memory, built on LangGraph. It can research topics via web search, ingest documents into a vector knowledge base, and recall stored knowledge across sessions.

> Inspired by [Letta](https://github.com/letta-ai/letta)'s three-tier memory architecture and [Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)'s agent workflow design.

## Features

- **Three-Tier Memory** — Core Memory (in-context blocks), Archival Memory (pgvector + BM25 hybrid search), Recall Memory (conversation history with semantic search)
- **Hybrid Retrieval Pipeline** — Vector similarity + BM25 keyword matching + cross-encoder re-ranking (`BAAI/bge-reranker-v2-m3`) + MMR diversity filtering + HyDE (hypothetical answer embedding)
- **Selective Memory Capture** — mem0-style pipeline: LLM judgment -> conflict resolution -> structured ADD/UPDATE/DELETE operations, with session context windows
- **Deep Research** — Multi-loop web search with automatic gap analysis and follow-up queries
- **Document Ingestion** — URL, PDF, or plain text; semantic or fixed-size chunking
- **Intent Routing** — Chat, Research, Recall, Memory Edit, Ingest modes
- **Core Memory Tools** — Self-editing blocks with undo history and auto-compression
- **Proactive Memory** — Pushes relevant memories at conversation start without being asked
- **Thread Isolation** — Recall memory scoped per session via `thread_id`
- **Full-Stack UI** — React frontend with streaming, document library, and conversation sidebar
- **Temporal Query Support** — Time-aware retrieval using temporal metadata extraction, date-range filtering, and absolute date indexing
- **Version History & Rollback** — Snapshot-before-mutate pattern across all three memory tiers (Archival, Recall, Core Memory) with diff view and one-click restore
- **Bulk Conflict Resolution** — Batch approve or reject multiple pending conflict reviews at once
- **Memory Analytics Dashboard** — Visualization of importance distribution, top entities, namespace breakdown, age distribution, and source trust scores
- **Automated Source Credibility** — Source trust scoring based on origin type (research_summary > manual > conversation) with visual trust indicators
- **Context Engineering** — Token budget management with priority-based truncation, automatic conversation summarization for long sessions, retrieval dedup, and source attribution via `[ref:ID]` citations
- **Dynamic Context Allocation** — Retrieval volume automatically adjusts based on query complexity (word count, question depth, multi-part queries)
- **Inline Edit for All Namespaces** — Edit entries across all archival namespaces (manual, research, ingested, etc.) directly from the Document Library
- **Knowledge Graph** — In-memory entity-relationship graph built from archival memories with interactive force-directed visualization

## Demo

### Memory Retrieval

![Memory Retrieval](images/memory_retrieved.png)

## Architecture

### Agent Graph

```
START -> route_intent
  |-- "ingest" -> ingest_document -> respond -> memory_pipeline -> [consolidate?] -> END
  +-- (other)  -> recall_memory -> evaluate_recall
                   |-- (sufficient) -> respond -> memory_pipeline -> [consolidate?] -> END
                   +-- (insufficient) -> generate_query -> [web_research x N] -> reflection
                         |-- (gaps) -> loop
                         +-- (sufficient) -> save_to_archival -> respond -> memory_pipeline -> [consolidate?] -> END
```

`evaluate_recall` judges whether retrieved memory is sufficient -- skipping web search when it is. `memory_pipeline` extracts facts after every response with conflict resolution; `consolidate` merges duplicates via embedding clustering every N turns.

### Retrieval Pipeline

```
Query -> [Rewrite] -> [HyDE] -> Search (vector + BM25, dynamic limit) -> Re-rank (cross-encoder) -> [MMR diversity] -> Results
```

- **Query Rewriting** — Resolves ambiguous/referential queries using conversation history
- **HyDE** — Generates a hypothetical answer, embeds it instead of the raw query
- **Dynamic Context Allocation** — Adjusts retrieval limit (3-8 results) based on query complexity
- **MMR** — Balances relevance and diversity in final results

### Memory Pipeline

```
User + Assistant messages
  -> LLM judgment (memorable?)
  -> Search existing memories
  -> Extract ADD/UPDATE/DELETE operations
  -> Conflict resolution (FACT_CONFLICT_PROMPT)
  -> Execute operations (with version snapshots)
  -> Store session context window (every N turns)
```

### Conversation Summarization

When message count exceeds a configurable threshold (default: 20), older turns are compressed into a summary via LLM. The summary is stored in state and prepended as context, keeping only recent turns as raw text.

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph StateGraph |
| LLM | Any OpenAI-compatible API |
| Embedding | BAAI/bge-m3 (1024-dim, local sentence-transformers, with LRU cache) |
| Vector Store | PostgreSQL 16 + pgvector |
| Full-Text Search | PostgreSQL tsvector + GIN |
| Re-ranking | BAAI/bge-reranker-v2-m3 (cross-encoder) |
| Web Search | Tavily API |
| Frontend | React 19 + Vite 6 + Tailwind CSS 4 + shadcn/ui |

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

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

```env
# LLM (OpenAI-compatible -- works with OpenAI, DeepSeek, MiMo, etc.)
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Embedding (local BAAI/bge-m3, auto-downloaded on first use)
EMBEDDING_MODEL=BAAI/bge-m3

# Web Search (Tavily)
TAVILY_API_KEY=your-tavily-key

# PostgreSQL (port 7432 maps to container 5432)
DATABASE_URL=postgresql://knowledge_agent:knowledge_agent@localhost:7432/knowledge_agent

# LangSmith (optional)
LANGSMITH_API_KEY=your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=knowledge-agent
```

### 3. Install & Run

```bash
# Backend
cd backend
pip install -e .
langgraph dev

# Frontend
cd frontend
npm install
npm run dev
```

Backend runs at `http://localhost:2024`, frontend at `http://localhost:5173`.

**CLI mode** (no frontend): `cd backend && python examples/cli_chat.py`

## Testing & Evaluation

```bash
cd backend

# Unit tests
python -m pytest tests/ -q

# Hybrid search recall benchmark
python scripts/test_recall.py

# LoCoMo evaluation (Recall@K across question types)
python scripts/test_locomo.py --local data/locomo.json --limit 1        # quick
python scripts/test_locomo.py --local data/locomo.json                   # full
python scripts/test_locomo.py --local data/locomo.json --strategy all    # compare strategies
```

### Archival Memory Management

```bash
python scripts/manage_archival.py stats                                  # entry counts per namespace
python scripts/manage_archival.py list --namespace research               # list entries
python scripts/manage_archival.py dedup --namespace research --dry-run    # find duplicates
python scripts/manage_archival.py smart-dedup --namespace research        # LLM-assisted dedup
python scripts/manage_archival.py cleanup --namespace research --days 30  # remove old entries
python scripts/manage_archival.py consolidate --namespace research        # full consolidation
```

## Project Structure

```
knowledge-agent/
├── backend/
│   ├── src/
│   │   ├── agent/
│   │   │   ├── graph.py                  # LangGraph definition
│   │   │   ├── state.py                  # AgentState TypedDict
│   │   │   ├── configuration.py          # Runtime configuration
│   │   │   ├── prompts.py                # Prompt templates
│   │   │   ├── db/
│   │   │   │   ├── _base.py              # Shared helpers (dedup, MMR, cosine)
│   │   │   │   ├── archival.py           # Archival memory CRUD + structured indexes
│   │   │   │   ├── recall.py             # Recall memory search, save, cleanup
│   │   │   │   ├── versioning.py         # Version history (archival + recall + core memory)
│   │   │   │   ├── conflicts.py          # Conflict review queue + resolution
│   │   │   │   ├── documents.py          # Document management + content-hash dedup
│   │   │   │   ├── analytics.py          # Analytics stats + source trust scoring
│   │   │   │   └── importance.py         # Importance scoring + weighted search
│   │   │   ├── memory/
│   │   │   │   ├── block.py              # Block model
│   │   │   │   ├── core_memory.py        # CoreMemory + undo history + compression
│   │   │   │   ├── importance.py         # Importance score calculation
│   │   │   │   ├── knowledge_graph.py    # In-memory entity-relationship graph
│   │   │   │   └── tools.py              # Memory edit + archival tools
│   │   │   ├── storage/
│   │   │   │   ├── embedding.py          # BGE-M3 local embedding + LRU cache
│   │   │   │   ├── reranker.py           # Cross-encoder re-ranking
│   │   │   │   └── ingestion.py          # Document ingestion pipeline
│   │   │   └── nodes/
│   │   │       ├── memory_manager.py     # Intent routing + recall + dynamic context allocation
│   │   │       ├── memory_pipeline.py    # Selective capture + consolidation
│   │   │       ├── researcher.py         # Web research + reflection
│   │   │       └── responder.py          # Response generation + summarization + token budget
│   │   └── api_server.py                 # REST API (FastAPI) for frontend
│   ├── scripts/
│   │   ├── test_recall.py
│   │   ├── test_locomo.py
│   │   └── manage_archival.py
│   ├── tests/                            # Unit tests
│   ├── examples/cli_chat.py
│   ├── config.yaml                       # Feature flags and tuning
│   └── docker-compose.yml                # PostgreSQL + pgvector
├── docs/                                  # Project documentation
│   ├── architecture/                      # System overview, graph flow, data model
│   ├── design-decisions/                  # ADRs (memory, HyDE, selective capture, reranker, MMR)
│   ├── modules/                           # Module-level documentation
│   ├── api/                               # REST API reference
│   ├── deployment/                        # Setup guide
│   └── roadmap.md                         # Future extensibility plan
└── frontend/src/
    ├── App.tsx
    └── components/
        ├── ChatMessagesView.tsx           # Chat UI with streaming
        ├── DocumentLibrary.tsx            # Document/archival/recall browser + version history
        ├── MemorySearchPanel.tsx          # Advanced memory search
        ├── MemoryAnalytics.tsx            # Analytics dashboard
        ├── KnowledgeGraphView.tsx         # Interactive knowledge graph visualization
        ├── CoreMemoryPanel.tsx            # Core memory blocks viewer
        ├── ActivityTimeline.tsx           # Memory operations timeline
        ├── ThreadSidebar.tsx              # Session/thread selector
        ├── InputForm.tsx                  # Message input
        └── WelcomeScreen.tsx              # Landing page
```

## Configuration

All feature flags and tuning parameters are in `backend/config.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `summarize_threshold` | 20 | Summarize older turns when message count exceeds this |
| `summarize_keep_recent` | 10 | Recent messages kept as raw text during summarization |
| `memory_dedup_threshold` | 0.8 | Cosine similarity threshold for fact deduplication |
| `rerank_enabled` | true | Cross-encoder re-ranking |
| `hyde_enabled` | true | HyDE hypothetical document embeddings |
| `mmr_enabled` | true | MMR diversity filtering |
| `proactive_memory_enabled` | true | Push relevant memories at conversation start |
| `structured_extraction` | true | Entity and temporal extraction from facts |
| `document_enrichment` | true | LLM-assisted chunk enrichment during ingestion |

## Acknowledgments

- **[Letta](https://github.com/letta-ai/letta)** -- Three-tier memory architecture, Block model, self-editing memory concept.
- **[Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)** -- Fullstack agent pattern, research loop with reflection, activity timeline UI.

## License

[MIT](LICENSE)
