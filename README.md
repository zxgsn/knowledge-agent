# Knowledge Agent

[中文](README_zh.md)

A personal knowledge agent with persistent memory, built on LangGraph. It can research topics via web search, ingest documents into a vector knowledge base, and recall stored knowledge across sessions.

> Inspired by [Letta](https://github.com/letta-ai/letta)'s three-tier memory architecture and [Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)'s agent workflow design.

## Features

- **Three-Tier Memory** — Core Memory (in-context blocks), Archival Memory (pgvector + BM25 hybrid search), Recall Memory (conversation history with semantic search)
- **Hybrid Retrieval Pipeline** — Vector similarity + BM25 keyword matching + cross-encoder re-ranking (`BAAI/bge-reranker-v2-m3`) + MMR diversity filtering + HyDE (hypothetical answer embedding)
- **Selective Memory Capture** — mem0-style pipeline: LLM judgment → conflict resolution → structured ADD/UPDATE/DELETE operations, with session context windows
- **Deep Research** — Multi-loop web search with automatic gap analysis and follow-up queries
- **Document Ingestion** — URL, PDF, or plain text; semantic or fixed-size chunking
- **Intent Routing** — Chat, Research, Recall, Memory Edit, Ingest modes
- **Core Memory Tools** — Self-editing blocks with undo history and auto-compression
- **Proactive Memory** — Pushes relevant memories at conversation start without being asked
- **Thread Isolation** — Recall memory scoped per session via `thread_id`
- **Full-Stack UI** — React frontend with streaming, document library, and conversation sidebar

## Demo

### Memory Retrieval

![Memory Retrieval](images/memory_retrieved.png)

## Architecture

### Agent Graph

```
START → route_intent
  ├── "ingest" → ingest_document → respond → memory_pipeline → [consolidate?] → END
  └── (other)  → recall_memory → evaluate_recall
                   ├── (sufficient) → respond → memory_pipeline → [consolidate?] → END
                   └── (insufficient) → generate_query → [web_research × N] → reflection
                         ├── (gaps) → loop
                         └── (sufficient) → save_to_archival → respond → memory_pipeline → [consolidate?] → END
```

`evaluate_recall` judges whether retrieved memory is sufficient — skipping web search when it is. `memory_pipeline` extracts facts after every response with conflict resolution; `consolidate` merges duplicates via embedding clustering every N turns.

### Retrieval Pipeline

```
Query → [Rewrite] → [HyDE] → Search (vector + BM25) → Re-rank (cross-encoder) → [MMR diversity] → Results
```

- **Query Rewriting** — Resolves ambiguous/referential queries using conversation history
- **HyDE** — Generates a hypothetical answer, embeds it instead of the raw query
- **MMR** — Balances relevance and diversity in final results

### Memory Pipeline

```
User + Assistant messages
  → LLM judgment (memorable?)
  → Search existing memories
  → Extract ADD/UPDATE/DELETE operations
  → Conflict resolution (FACT_CONFLICT_PROMPT)
  → Execute operations
  → Store session context window (every N turns)
```

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph StateGraph |
| LLM | Any OpenAI-compatible API |
| Embedding | DashScope `text-embedding-v3` (1024-dim, with LRU cache) |
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
# LLM (OpenAI-compatible — works with OpenAI, DeepSeek, MiMo, etc.)
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
│   ├── src/agent/
│   │   ├── graph.py                  # LangGraph definition
│   │   ├── state.py                  # AgentState TypedDict
│   │   ├── configuration.py          # Runtime configuration
│   │   ├── prompts.py                # Prompt templates
│   │   ├── db.py                     # Centralized DB helpers (archival + recall)
│   │   ├── memory/
│   │   │   ├── block.py              # Block model
│   │   │   ├── core_memory.py        # CoreMemory + undo history
│   │   │   └── tools.py              # Memory edit + archival tools
│   │   ├── storage/
│   │   │   ├── embedding.py          # DashScope embedding + LRU cache
│   │   │   ├── reranker.py           # Cross-encoder re-ranking
│   │   │   └── ingestion.py          # Document ingestion pipeline
│   │   └── nodes/
│   │       ├── memory_manager.py     # Intent routing + recall + archival save
│   │       ├── memory_pipeline.py    # Selective capture + consolidation
│   │       ├── researcher.py         # Web research + reflection
│   │       └── responder.py          # Response generation
│   ├── scripts/
│   │   ├── test_recall.py
│   │   ├── test_locomo.py
│   │   └── manage_archival.py
│   ├── tests/                        # 174 unit tests
│   └── examples/cli_chat.py
└── frontend/src/
    ├── App.tsx
    └── components/
```

## Roadmap

- **Edit support for all namespaces** — Currently only `manual` entries can be edited from the Document Library UI. Extend inline edit to `ingested`, `research`, and other archival namespaces.
- **Version diff view** — Side-by-side comparison between two archival versions, highlighting added/removed/changed text for easier review.
- **Bulk conflict resolution** — Allow approving or rejecting multiple pending conflict reviews at once, with a batch action UI.
- **Recall & Core Memory version tracking** — Extend the snapshot-before-mutate pattern to Recall memory and Core Memory blocks, enabling full rollback across all three tiers.
- **Memory analytics dashboard** — Visualize memory quality metrics: total entries by namespace, conflict resolution rate, version history depth, source trust distribution, and storage growth over time.
- **Automated source credibility** — Auto-detect source reliability from content patterns (e.g., peer-reviewed papers, official docs vs. blog posts) instead of relying solely on the `metadata.source` field.
- **Context engineering improvements**:
  - **Token budget management** — Count and cap total tokens injected into the LLM (history, retrieval results, system prompt) to prevent context window overflow.
  - **Message summarization** — Compress early conversation turns into summaries, keeping only recent turns as raw text, to avoid context loss or token waste in long conversations.
  - **Retrieval dedup & conflict resolution** — Deduplicate Archival and Recall results before injection; annotate or resolve conflicting information.
  - **Context source attribution** — Attach structured metadata (source, timestamp, confidence) to injected retrieval results so the LLM can judge reliability and recency.
  - **Dynamic context allocation** — Adjust retrieval and injection volume based on query complexity: fewer results for simple questions, more for complex ones.

## Acknowledgments

- **[Letta](https://github.com/letta-ai/letta)** — Three-tier memory architecture, Block model, self-editing memory concept.
- **[Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)** — Fullstack agent pattern, research loop with reflection, activity timeline UI.

## License

[MIT](LICENSE)
