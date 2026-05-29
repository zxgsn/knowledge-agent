# Module: Storage

**Source**: `backend/src/agent/storage/`

Handles all storage operations: embeddings, reranking, document ingestion, and database access.

## Components

### `storage/__init__.py` - Connection Management

- **Singleton pattern**: Global connection pool and embedding model instance
- **Connection pool**: `psycopg_pool.ConnectionPool` (min=2, max=10)
- **Auto-registration**: pgvector type registration on each connection
- **Lazy initialization**: Pool and embeddings created on first use

### `storage/embedding.py` - Local Embeddings

**Model**: BAAI/bge-m3 (1024-dim, multilingual bi-encoder)

**Features**:
- Local model loading from `backend/models/bge-m3/`
- Fallback to HuggingFace Hub download
- In-memory LRU cache (512 entries) to avoid redundant encoding
- Device auto-detection: CUDA if available, else CPU
- Compatible with LangChain's `Embeddings` interface

**Usage**:
```python
from agent.storage import get_embeddings
embeddings = get_embeddings()
vector = embeddings.embed_query("text to encode")
vectors = embeddings.embed_documents(["text1", "text2"])
```

### `storage/reranker.py` - Cross-Encoder Reranking

**Model**: BAAI/bge-reranker-v2-m3 (multilingual cross-encoder, ~568M)

**Features**:
- Lazy-loaded singleton model
- Local model loading from `backend/models/bge-reranker-v2-m3/`
- Max input length: 512 tokens
- Graceful fallback to bi-encoder ranking on failure

**Usage**:
```python
from agent.storage.reranker import rerank
reranked = rerank(query, results, top_k=5, enabled=True)
```

### `storage/ingestion.py` - Document Ingestion Pipeline

**Pipeline**: Extract Text -> Chunk -> Embed -> Store

**Supported Sources**:

| Type | Extractor | Notes |
|------|-----------|-------|
| URL | `extract_text_from_url()` | BeautifulSoup, removes nav/footer/scripts |
| PDF | `extract_text_from_pdf()` | pypdf, page-by-page extraction |
| Text | Direct | Passthrough |

**Chunking Strategy**:
- Default chunk size: 1000 characters
- Overlap: 200 characters between chunks
- Each chunk gets a unique ID: `{source}_{index}`

**Document Enrichment** (optional):
When `document_enrichment` is enabled, each chunk gets LLM-assisted metadata:
- Summary
- Extracted entities
- Keywords

## Database Schema

See [Data Model](../architecture/data-model.md) for complete schema documentation.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | (required) | PostgreSQL connection string |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model name |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model name |
| `RERANK_ENABLED` | `true` | Enable cross-encoder reranking |
| `MMR_ENABLED` | `false` | Enable MMR diversity |
| `MMR_LAMBDA` | `0.5` | MMR relevance/diversity balance |
