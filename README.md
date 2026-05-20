# Knowledge Agent

[中文](README_zh.md)

A personal knowledge agent with persistent memory, built on LangGraph. It can research topics via web search, ingest documents into a vector knowledge base, and recall stored knowledge across sessions.

> Inspired by [Letta](https://github.com/letta-ai/letta)'s three-tier memory architecture and [Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)'s agent workflow design.

## Features

- **Three-Tier Memory System** — Core Memory (always in context), Archival Memory (vector long-term storage), Recall Memory (conversation history)
- **Hybrid Search** — Archival retrieval combines vector similarity (DashScope embeddings) with BM25 keyword matching (PostgreSQL tsvector + GIN index) for better recall
- **Core Memory Auto-Compression** — Blocks approaching their character limit are automatically distilled by the LLM, preserving key facts while staying within bounds
- **Deep Research** — Multi-loop web search with automatic gap analysis and follow-up queries
- **Document Ingestion** — Import URLs, PDFs, or plain text; automatic chunking and vectorization
- **Intent Routing** — LLM-classified modes: Chat, Research, Recall, Memory Edit, Ingest
- **Chat with Knowledge Retrieval** — Chat mode automatically searches Archival Memory for relevant context before generating a response
- **Persistent Knowledge** — Research findings are stored in PostgreSQL + pgvector and survive across sessions
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
- **Recall Memory** — Conversation history with semantic search (table structure ready, integration pending).

### Agent Graph

```
START → route_intent
  ├── "chat"        → recall_memory → respond → END
  ├── "research"    → generate_query → [web_research × N] → reflection
  │                    ├── (gaps found) → [web_research] → reflection (loop)
  │                    └── (sufficient) → save_to_archival → respond → END
  ├── "recall"      → recall_memory → respond → END
  ├── "memory_edit" → recall_memory → respond → END
  └── "ingest"      → ingest_document → save_to_archival → respond → END
```

All non-research paths now go through `recall_memory` first, so the agent always has relevant knowledge context when responding.

### Document Ingestion Pipeline

```
User provides URL / plain text / file path
  ↓
ingest_document node
  ├── URL     → httpx fetch → BeautifulSoup extract
  ├── Text    → direct processing
  └── File    → read content
  ↓
chunk_text() — paragraph → sentence → character splitting with overlap
  ↓
DashScope Embedding batch vectorization
  ↓
ArchivalMemory.put_batch() → PostgreSQL + pgvector
```

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
│   │   └── test_recall.py            # Hybrid search recall benchmark
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
│       │   ├── archival.py           # ArchivalMemory (pgvector)
│       │   ├── recall.py             # RecallMemory
│       │   └── ingestion.py          # Document ingestion pipeline
│       └── nodes/
│           ├── memory_manager.py     # Intent routing + archival save
│           ├── researcher.py         # Search + research + reflection
│           └── responder.py          # Response generation
└── frontend/
    └── src/
        ├── App.tsx                   # Main app with LangGraph streaming
        └── components/
            ├── InputForm.tsx         # Mode selector (Chat/Research/Memory/Ingest)
            ├── WelcomeScreen.tsx
            ├── ChatMessagesView.tsx
            └── ActivityTimeline.tsx
```

## CLI Commands

| Command | Description |
|---|---|
| `/memory` | View Core Memory blocks |
| `/search <query>` | Search Archival Memory |
| `/ingest <url_or_path>` | Ingest a document (URL or file path) |
| `/quit` | Exit |

## Acknowledgments

This project draws design inspiration from:

- **[Letta](https://github.com/letta-ai/letta)** (formerly MemGPT) — The three-tier memory architecture (Core / Archival / Recall), Block-based memory model, and self-editing memory concept are adapted from Letta's design. The implementation is independent and simplified for LangGraph usage.
- **[Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)** — The fullstack agent pattern (LangGraph backend + React frontend with streaming), research loop with reflection, and activity timeline UI are inspired by this Google Gemini quickstart project.

## License

[MIT](LICENSE)
