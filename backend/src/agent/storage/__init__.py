"""Storage initialization with lazy singleton pattern."""

from __future__ import annotations

import os

from agent.storage.embedding import DashScopeEmbeddings

_embeddings: DashScopeEmbeddings | None = None
_recall_table_ready: bool = False


def get_embeddings() -> DashScopeEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = DashScopeEmbeddings()
    return _embeddings


def get_db_url() -> str:
    return os.environ["DATABASE_URL"]


def ensure_recall_table() -> None:
    """Create the recall_memory table and indexes if they don't exist."""
    global _recall_table_ready
    if _recall_table_ready:
        return

    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(get_db_url())
    register_vector(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recall_memory (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            embedding vector(1024),
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_recall_thread ON recall_memory (thread_id)")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_recall_embedding
        ON recall_memory USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    """)
    conn.commit()
    conn.close()
    _recall_table_ready = True
