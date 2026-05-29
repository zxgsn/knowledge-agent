# Deployment Guide

## Prerequisites

- Python 3.11+
- PostgreSQL 16 with pgvector extension
- Node.js 18+ (for frontend)
- 4GB+ RAM (for embedding + reranker models)

## Quick Start

### 1. Start PostgreSQL

```bash
cd backend
docker compose up -d
```

This starts PostgreSQL with pgvector on port 7432.

### 2. Configure Environment

Create `backend/.env`:

```bash
DATABASE_URL=postgresql://knowledge_agent:knowledge_agent@localhost:7432/knowledge_agent
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 3. Install Backend Dependencies

```bash
cd backend
pip install -e .
# or: uv sync
```

### 4. Download Models (Optional)

Models are auto-downloaded on first use, or pre-download:

```bash
python scripts/download_model.py
```

Local models are stored in `backend/models/`.

### 5. Start Backend

```bash
cd backend
make dev
# or: uvicorn src.api_server:app --reload --port 8000
```

### 6. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`.

## Docker Compose

The `docker-compose.yml` only defines the PostgreSQL service. The application itself runs directly.

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: knowledge_agent
      POSTGRES_PASSWORD: knowledge_agent
      POSTGRES_DB: knowledge_agent
    ports:
      - "7432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

## Configuration

### config.yaml

Feature flags and tuning parameters. See `backend/config.yaml` for all options.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `OPENAI_API_KEY` | Yes* | OpenAI API key (or compatible) |
| `TAVILY_API_KEY` | No | Tavily web search API key |
| `EMBEDDING_MODEL` | No | Embedding model (default: BAAI/bge-m3) |
| `RERANK_MODEL` | No | Reranker model (default: BAAI/bge-reranker-v2-m3) |
| `RERANK_ENABLED` | No | Enable reranking (default: true) |
| `MMR_ENABLED` | No | Enable MMR (default: false) |

*Can be replaced with any OpenAI-compatible API by setting `llm_base_url` in config.

## Testing

```bash
cd backend
make test
# or: python -m pytest tests/ -v
```

## Linting

```bash
cd backend
make format  # Auto-fix
make lint    # Check only
```
