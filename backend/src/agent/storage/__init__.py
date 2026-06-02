"""Storage initialization with lazy singleton pattern."""

from __future__ import annotations

import os
from contextlib import contextmanager
from threading import Lock

from agent.storage.embedding import LocalEmbeddings

_embeddings: LocalEmbeddings | None = None
_recall_table_ready: bool = False
_pool = None
_embeddings_lock = Lock()


def get_embeddings() -> LocalEmbeddings:
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = LocalEmbeddings()
    return _embeddings


def get_db_url() -> str:
    return os.environ["DATABASE_URL"]


def get_pool():
    """Get or create the connection pool singleton."""
    global _pool
    if _pool is None:
        from psycopg_pool import ConnectionPool
        _pool = ConnectionPool(
            get_db_url(),
            min_size=2,
            max_size=10,
            kwargs={"connect_timeout": 5},
        )
    return _pool


@contextmanager
def get_conn():
    """Get a connection from the pool. Auto-registers pgvector.

    Usage:
        with get_conn() as conn:
            conn.execute(...)
    """
    pool = get_pool()
    with pool.connection() as conn:
        from pgvector.psycopg import register_vector
        register_vector(conn)
        yield conn


def ensure_recall_table() -> None:
    """Create the recall_memory table and indexes if they don't exist."""
    global _recall_table_ready
    if _recall_table_ready:
        return

    with get_conn() as conn:
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
    _recall_table_ready = True
