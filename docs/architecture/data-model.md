# Data Model

## Database: PostgreSQL 16 + pgvector

The system uses a single PostgreSQL database with pgvector extension for vector similarity search.

### Connection Management

- Connection pool: `psycopg_pool.ConnectionPool` (min=2, max=10)
- Vector registration: `pgvector.psycopg.register_vector` auto-called per connection
- Defined in `backend/src/agent/storage/__init__.py`

## Tables

### `archival_memory`

Primary long-term storage for facts, research findings, and ingested documents.

```sql
CREATE TABLE archival_memory (
    id          TEXT PRIMARY KEY,           -- UUID
    namespace   TEXT NOT NULL,              -- e.g. "conversation_facts", "research_findings"
    content     TEXT NOT NULL,              -- The stored text
    metadata    JSONB DEFAULT '{}',         -- Source, entities, temporal refs, etc.
    embedding   vector(1024),               -- BGE-M3 embedding (1024-dim)
    content_tsv TSVECTOR,                   -- For BM25 full-text search
    status      TEXT DEFAULT 'active',      -- "active" | "archived" | "deleted"
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
```

**Indexes**:
- `ivfflat (embedding vector_cosine_ops)` - Vector similarity search
- GIN on `content_tsv` - BM25 full-text search
- B-tree on `namespace` - Namespace filtering
- B-tree on `status` - Status filtering

**Namespaces**:

| Namespace | Content | Source |
|-----------|---------|--------|
| `conversation_facts` | Facts extracted from conversations | memory_pipeline |
| `research_findings` | Summarized web research | save_to_archival |
| `document_ingested` | Chunks from ingested docs | ingest_document |
| `manual` | Manually added entries | API |

### `recall_memory`

Conversation history with semantic search capability.

```sql
CREATE TABLE recall_memory (
    id          TEXT PRIMARY KEY,
    thread_id   TEXT NOT NULL,              -- Thread/session isolation
    role        TEXT NOT NULL,              -- "user" | "assistant"
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   vector(1024),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

**Indexes**:
- `idx_recall_thread` on `thread_id` - Thread isolation
- `idx_recall_embedding` ivfflat on `embedding` - Vector similarity

### `documents`

Document registry for ingested content.

```sql
CREATE TABLE documents (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    source        TEXT,                     -- URL or file path
    source_type   TEXT,                     -- "url" | "pdf" | "text"
    content_full  TEXT,                     -- Full extracted text
    chunk_count   INTEGER DEFAULT 0,
    metadata      JSONB DEFAULT '{}',
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

### `conflict_review`

Queue for human review of uncertain memory merges.

```sql
CREATE TABLE conflict_review (
    id                TEXT PRIMARY KEY,
    new_fact          TEXT NOT NULL,
    existing_id       TEXT NOT NULL,
    existing_content  TEXT NOT NULL,
    similarity_score  FLOAT,
    llm_decision      TEXT,                 -- LLM's suggested action
    llm_merged_text   TEXT,                 -- LLM's merged version
    llm_confidence    FLOAT,               -- 0.0 - 1.0
    resolution        TEXT,                 -- "approved" | "rejected" | "modified"
    resolved_text     TEXT,                 -- Final accepted text
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
```

## Search Pipeline

The hybrid search in `search_archival()` combines multiple retrieval strategies:

```
Query
  |
  v
[HyDE] Generate hypothetical answer (optional)
  |
  v
[Embedding] BGE-M3 encodes query -> 1024-dim vector
  |
  v
[Hybrid Search]
  ├── Vector: cosine similarity (alpha weight)
  └── BM25: ts_rank on content_tsv (1-alpha weight)
  |
  v
[Candidate Pool] top 3x limit
  |
  v
[Cross-Encoder Rerank] BGE-Reranker-v2-m3 (optional)
  |
  v
[MMR Diversity] Maximal Marginal Relevance (optional)
  |
  v
[Content Dedup] Remove near-duplicates (>95% word overlap)
  |
  v
[Final Results] top limit
```

### Search Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.7 | Weight for vector score (1-alpha for BM25) |
| `limit` | 5 | Max final results |
| `candidate_limit` | 15 | Pre-rerank candidates (3x limit) |
| `rerank_enabled` | true | Cross-encoder re-ranking |
| `mmr_enabled` | false | MMR diversity |
| `mmr_lambda` | 0.5 | MMR relevance/diversity balance |
| `similarity_threshold` | 0.15 | Minimum cosine similarity |
| `dedup_threshold` | 0.95 | Content overlap for dedup |

## Metadata Schema

Archival entries can carry structured metadata:

```json
{
    "source": "conversation",
    "turn": 5,
    "entities": [
        {"name": "Alice", "type": "person"},
        {"name": "Google", "type": "org"}
    ],
    "temporal": {
        "reference": "last Monday",
        "absolute": "2026-05-25"
    },
    "document": "research-paper-title",
    "source_type": "url"
}
```

Entity and temporal metadata enable supplementary search paths (`search_by_entity`, `search_by_temporal`) for more precise recall.
