"""Centralized sync DB helpers for archival and recall memory.

All functions use the connection pool from agent.storage.
Import these instead of duplicating DB logic across node files.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from agent.storage import get_conn

logger = logging.getLogger(__name__)

# Unified BM25 text search configuration
_TS_CONFIG = "english"


# --- Archival Memory ---


def search_archival(query: str, limit: int = 5, alpha: float = 0.7) -> list[dict]:
    """Hybrid search: vector similarity + BM25 keyword match, with cross-encoder re-ranking.

    Args:
        query: Search query text.
        limit: Max results after re-ranking.
        alpha: Weight for vector score (1-alpha for BM25). Default 0.7.
    """
    from pgvector import Vector

    from agent.storage import get_embeddings

    try:
        embeddings = get_embeddings()
        query_embedding = Vector(embeddings.embed_query(query))
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        return []

    try:
        with get_conn() as conn:
            candidate_limit = int(limit) * 3
            rows = conn.execute(
                f"""
                SELECT content, metadata,
                       %s * (1 - (embedding <=> %s::vector))
                         + (1 - %s) * LEAST(1, ts_rank(content_tsv, plainto_tsquery('{_TS_CONFIG}', %s)) * 5)
                       AS score
                FROM archival_memory
                WHERE 1 - (embedding <=> %s::vector) > 0.15
                ORDER BY score DESC
                LIMIT %s
                """,
                (alpha, query_embedding, alpha, query, query_embedding, candidate_limit),
            ).fetchall()
    except Exception as e:
        logger.error("Archival search failed: %s", e)
        return []

    results = []
    for row in rows:
        score = float(row[2])
        if score >= 0.01:
            meta = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            results.append({"content": row[0], "metadata": meta, "score": score})

    # Cross-encoder re-ranking
    try:
        from agent.storage.reranker import rerank
        results = rerank(query, results, top_k=int(limit))
    except Exception:
        results = results[: int(limit)]

    return results


def search_archival_for_dedup(
    query: str, namespace: str = "conversation_facts", limit: int = 3
) -> list[dict]:
    """Search archival for semantically similar facts. Pure cosine similarity."""
    from pgvector import Vector

    from agent.storage import get_embeddings

    try:
        embeddings = get_embeddings()
        query_embedding = Vector(embeddings.embed_query(query))
    except Exception:
        return []

    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM archival_memory
                WHERE namespace = %s
                  AND 1 - (embedding <=> %s::vector) > 0.5
                ORDER BY score DESC
                LIMIT %s
                """,
                (query_embedding, namespace, query_embedding, int(limit)),
            ).fetchall()
    except Exception:
        return []

    results = []
    for row in rows:
        score = float(row[3])
        meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
        results.append({"id": row[0], "content": row[1], "metadata": meta, "score": score})
    return results


def put_to_archival(
    content: str,
    namespace: str,
    metadata: dict,
    document_id: str | None = None,
    on_conflict: str = "nothing",
) -> str:
    """Store a single entry in archival memory.

    Args:
        content: Text content to store.
        namespace: Namespace for the entry.
        metadata: Metadata dict.
        document_id: Optional link to a document record.
        on_conflict: "nothing" (skip) or "update" (upsert content/metadata/embedding).
    """
    from pgvector import Vector

    from agent.storage import get_embeddings

    embeddings = get_embeddings()
    entry_id = str(uuid.uuid4())
    metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
    embedding = Vector(embeddings.embed_query(content))

    if on_conflict == "update":
        sql = (
            "INSERT INTO archival_memory (id, namespace, content, metadata, embedding, document_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
            "content=EXCLUDED.content, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding"
        )
    else:
        sql = (
            "INSERT INTO archival_memory (id, namespace, content, metadata, embedding, document_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING"
        )

    with get_conn() as conn:
        conn.execute(
            sql,
            (entry_id, namespace, content, json.dumps(metadata), embedding, document_id),
        )
        conn.commit()
    return entry_id


def put_batch_to_archival(
    entries: list[dict], namespace: str, document_id: str | None = None
) -> list[str]:
    """Batch insert entries into archival memory."""
    from pgvector import Vector

    from agent.storage import get_embeddings

    if not entries:
        return []

    embeddings = get_embeddings()
    embedding_vectors = embeddings.embed_documents([e["content"] for e in entries])
    now = datetime.now(timezone.utc).isoformat()
    ids: list[str] = []

    with get_conn() as conn:
        for entry, emb in zip(entries, embedding_vectors):
            entry_id = str(uuid.uuid4())
            meta = entry.get("metadata", {})
            meta["timestamp"] = now
            ids.append(entry_id)
            conn.execute(
                "INSERT INTO archival_memory (id, namespace, content, metadata, embedding, document_id) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
                "content=EXCLUDED.content, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
                (entry_id, namespace, entry["content"], json.dumps(meta), Vector(emb), document_id),
            )
        conn.commit()
    return ids


def insert_document(
    title: str, source: str, source_type: str, content_full: str, chunk_count: int
) -> tuple[str, bool]:
    """Insert a document record. Returns (doc_id, is_new)."""
    doc_id = str(uuid.uuid4())

    with get_conn() as conn:
        result = conn.execute(
            "INSERT INTO documents (id, title, source, source_type, content_full, chunk_count) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (title, source) DO NOTHING "
            "RETURNING id",
            (doc_id, title, source, source_type, content_full, chunk_count),
        )
        row = result.fetchone()
        if row:
            conn.commit()
            return row[0], True
        # Conflict — document already exists
        existing = conn.execute(
            "SELECT id FROM documents WHERE title = %s AND source = %s",
            (title, source),
        ).fetchone()
        return existing[0], False


def update_archival(entry_id: str, content: str, metadata: dict) -> None:
    """Update an existing archival entry's content and embedding."""
    from pgvector import Vector

    from agent.storage import get_embeddings

    embeddings = get_embeddings()
    embedding = Vector(embeddings.embed_query(content))
    metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
    metadata["updated_by"] = "memory_pipeline"

    with get_conn() as conn:
        conn.execute(
            "UPDATE archival_memory SET content = %s, metadata = %s, embedding = %s WHERE id = %s",
            (content, json.dumps(metadata), embedding, entry_id),
        )
        conn.commit()


def delete_archival(entry_id: str) -> bool:
    """Delete an archival entry by ID."""
    with get_conn() as conn:
        result = conn.execute("DELETE FROM archival_memory WHERE id = %s", (entry_id,))
        conn.commit()
        return result.rowcount > 0


def get_all_facts(namespace: str = "conversation_facts", limit: int = 500) -> list[dict]:
    """Fetch all facts in a namespace for consolidation."""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT id, content, metadata, embedding::text FROM archival_memory "
                "WHERE namespace = %s ORDER BY created_at DESC LIMIT %s",
                (namespace, int(limit)),
            ).fetchall()
    except Exception:
        return []

    results = []
    for row in rows:
        meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
        results.append({"id": row[0], "content": row[1], "metadata": meta, "embedding_text": row[3]})
    return results


def get_existing_memories(
    query: str, namespace: str = "conversation_facts", limit: int = 10
) -> list[dict]:
    """Get existing memories relevant to a query for the extraction prompt context."""
    from pgvector import Vector

    from agent.storage import get_embeddings

    try:
        embeddings = get_embeddings()
        query_embedding = Vector(embeddings.embed_query(query))
    except Exception:
        return []

    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, content, metadata
                FROM archival_memory
                WHERE namespace = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (namespace, query_embedding, int(limit)),
            ).fetchall()
    except Exception:
        return []

    results = []
    for row in rows:
        meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
        results.append({"id": row[0], "content": row[1], "metadata": meta})
    return results


def cleanup_namespace(namespace: str, max_age_days: int) -> int:
    """Delete entries older than max_age_days from a namespace. Returns count deleted."""
    try:
        with get_conn() as conn:
            result = conn.execute(
                "DELETE FROM archival_memory WHERE namespace = %s "
                "AND created_at < NOW() - INTERVAL '%s days'",
                (namespace, max_age_days),
            )
            conn.commit()
            return result.rowcount
    except Exception:
        return 0


def cleanup_excess(namespace: str, max_entries: int) -> int:
    """Keep only the newest max_entries in a namespace. Returns count deleted."""
    try:
        with get_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM archival_memory WHERE namespace = %s", (namespace,)
            ).fetchone()[0]

            if count <= max_entries:
                return 0

            conn.execute(
                "DELETE FROM archival_memory WHERE id IN ("
                "  SELECT id FROM archival_memory WHERE namespace = %s "
                "  ORDER BY created_at ASC LIMIT %s"
                ")",
                (namespace, count - max_entries),
            )
            conn.commit()
            return count - max_entries
    except Exception:
        return 0


# --- Recall Memory ---


def save_to_recall(role: str, content: str, thread_id: str = "default") -> None:
    """Save a message to the recall_memory table."""
    from pgvector import Vector

    from agent.storage import ensure_recall_table, get_embeddings

    ensure_recall_table()
    embeddings = get_embeddings()
    entry_id = str(uuid.uuid4())
    embedding = Vector(embeddings.embed_query(content))

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO recall_memory (id, thread_id, role, content, embedding) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
            (entry_id, thread_id, role, content, embedding),
        )
        conn.commit()


def search_recall(query: str, limit: int = 5) -> list[dict]:
    """Semantic search over recall_memory (conversation history).

    Uses pure cosine similarity with optional cross-encoder re-ranking.
    """
    from pgvector import Vector

    from agent.storage import get_embeddings

    try:
        embeddings = get_embeddings()
        query_embedding = Vector(embeddings.embed_query(query))
    except Exception as e:
        logger.error("Recall embedding failed: %s", e)
        return []

    try:
        with get_conn() as conn:
            candidate_limit = int(limit) * 3
            rows = conn.execute(
                """
                SELECT id, thread_id, role, content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM recall_memory
                WHERE 1 - (embedding <=> %s::vector) > 0.3
                ORDER BY score DESC
                LIMIT %s
                """,
                (query_embedding, query_embedding, candidate_limit),
            ).fetchall()
    except Exception as e:
        logger.error("Recall search failed: %s", e)
        return []

    results = []
    for row in rows:
        score = float(row[5])
        meta = row[4] if isinstance(row[4], dict) else json.loads(row[4])
        results.append({
            "id": row[0],
            "thread_id": row[1],
            "role": row[2],
            "content": row[3],
            "metadata": meta,
            "score": score,
        })

    # Cross-encoder re-ranking
    try:
        from agent.storage.reranker import rerank
        results = rerank(query, results, top_k=int(limit))
    except Exception:
        results = results[: int(limit)]

    return results
