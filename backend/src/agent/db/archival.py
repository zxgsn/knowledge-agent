"""Archival memory: search, store, update, delete, and maintenance."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from agent.db._base import get_conn

from agent.db._base import (
    _TS_CONFIG,
    _cosine_similarity,
    _deduplicate_results,
    _mmr_rerank,
    logger,
)


# --- Archival Memory ---


def search_archival(
    query: str,
    limit: int = 5,
    alpha: float = 0.7,
    rerank_enabled: bool | None = None,
    mmr_enabled: bool | None = None,
    mmr_lambda: float | None = None,
) -> list[dict]:
    """Hybrid search: vector similarity + BM25 keyword match, with cross-encoder re-ranking.

    Args:
        query: Search query text.
        limit: Max results after re-ranking.
        alpha: Weight for vector score (1-alpha for BM25). Default 0.7.
        rerank_enabled: Override env RERANK_ENABLED. None = use env.
        mmr_enabled: Override env MMR_ENABLED. None = use env.
        mmr_lambda: Override env MMR_LAMBDA. None = use env.
    """
    import os

    from pgvector import Vector

    from agent.storage import get_embeddings

    try:
        embeddings = get_embeddings()
        query_embedding_flat = embeddings.embed_query(query)
        query_embedding = Vector(query_embedding_flat)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        return []

    try:
        with get_conn() as conn:
            candidate_limit = int(limit) * 3
            rows = conn.execute(
                f"""
                SELECT content, metadata, namespace,
                       %s * (1 - (embedding <=> %s::vector))
                         + (1 - %s) * LEAST(1, ts_rank(content_tsv, websearch_to_tsquery('{_TS_CONFIG}', %s)) * 10)
                       AS score
                FROM archival_memory
                WHERE status = 'active' AND 1 - (embedding <=> %s::vector) > 0.15
                  AND namespace NOT LIKE 'locomo%%'
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
        score = float(row[3])
        if score >= 0.01:
            meta = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            results.append({"content": row[0], "metadata": meta, "namespace": row[2], "score": score})

    # Cross-encoder re-ranking
    try:
        from agent.storage.reranker import rerank
        results = rerank(query, results, top_k=int(limit), enabled=rerank_enabled)
    except Exception as e:
        # Catch ALL exceptions to ensure we never crash here
        import traceback
        from agent.logger import get_logger
        logger = get_logger(__name__)
        logger.error("Reranking failed in search_archival: %s", e, exc_info=True)
        # Fallback: sort by original score and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[: int(limit)]

    # MMR deduplication
    _mmr_on = mmr_enabled if mmr_enabled is not None else os.getenv("MMR_ENABLED", "false").lower() in ("true", "1", "yes")
    if _mmr_on and len(results) > 1:
        _lambda = mmr_lambda if mmr_lambda is not None else float(os.getenv("MMR_LAMBDA", "0.5"))
        results = _mmr_rerank(query_embedding_flat, results, lambda_param=_lambda, top_k=int(limit))

    # Content-based deduplication
    results = _deduplicate_results(results)

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
    except Exception as e:
        logger.error("Dedup embedding failed: %s", e)
        return []

    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM archival_memory
                WHERE namespace = %s
                  AND status = 'active'
                  AND 1 - (embedding <=> %s::vector) > 0.5
                ORDER BY score DESC
                LIMIT %s
                """,
                (query_embedding, namespace, query_embedding, int(limit)),
            ).fetchall()
    except Exception as e:
        logger.error("Dedup search failed: %s", e)
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


def update_archival(
    entry_id: str, content: str, metadata: dict, changed_by: str = "memory_pipeline"
) -> None:
    """Update an existing archival entry's content and embedding. Snapshots before mutation."""
    from pgvector import Vector

    from agent.storage import get_embeddings

    from agent.db.versioning import _snapshot_version

    embeddings = get_embeddings()
    embedding = Vector(embeddings.embed_query(content))
    metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
    metadata["updated_by"] = changed_by

    with get_conn() as conn:
        _snapshot_version(conn, entry_id, "update", changed_by)
        conn.execute(
            "UPDATE archival_memory SET content = %s, metadata = %s, embedding = %s, created_at = NOW() WHERE id = %s",
            (content, json.dumps(metadata), embedding, entry_id),
        )
        conn.commit()


def delete_archival(entry_id: str, changed_by: str = "memory_pipeline") -> bool:
    """Soft-delete an archival entry: snapshot then set status='superseded'."""
    from agent.db.versioning import _snapshot_version

    with get_conn() as conn:
        _snapshot_version(conn, entry_id, "delete", changed_by)
        result = conn.execute(
            "UPDATE archival_memory SET status = 'superseded' WHERE id = %s AND status = 'active'",
            (entry_id,),
        )
        conn.commit()
        return result.rowcount > 0


def get_all_facts(namespace: str = "conversation_facts", limit: int = 500) -> list[dict]:
    """Fetch all facts in a namespace for consolidation."""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT id, content, metadata, embedding::text FROM archival_memory "
                "WHERE namespace = %s AND status = 'active' ORDER BY created_at DESC LIMIT %s",
                (namespace, int(limit)),
            ).fetchall()
    except Exception as e:
        logger.error("get_all_facts failed: %s", e)
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
    except Exception as e:
        logger.error("get_existing_memories embedding failed: %s", e)
        return []

    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, content, metadata
                FROM archival_memory
                WHERE namespace = %s AND status = 'active'
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (namespace, query_embedding, int(limit)),
            ).fetchall()
    except Exception as e:
        logger.error("get_existing_memories search failed: %s", e)
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
    except Exception as e:
        logger.error("cleanup_namespace failed: %s", e)
        return 0


def cleanup_excess(namespace: str, max_entries: int) -> int:
    """Keep only the newest max_entries in a namespace. Returns count soft-deleted."""
    from agent.db.versioning import _snapshot_version

    try:
        with get_conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM archival_memory WHERE namespace = %s AND status = 'active'",
                (namespace,),
            ).fetchone()[0]

            if count <= max_entries:
                return 0

            excess_ids = conn.execute(
                "SELECT id FROM archival_memory WHERE namespace = %s AND status = 'active' "
                "ORDER BY created_at ASC LIMIT %s",
                (namespace, count - max_entries),
            ).fetchall()

            for (eid,) in excess_ids:
                _snapshot_version(conn, eid, "delete", "cleanup_excess")
                conn.execute(
                    "UPDATE archival_memory SET status = 'superseded' WHERE id = %s",
                    (eid,),
                )

            conn.commit()
            return len(excess_ids)
    except Exception as e:
        logger.error("cleanup_excess failed: %s", e)
        return 0


def get_recent_archival(limit: int = 5) -> list[dict]:
    """Fetch most recent archival entries across all namespaces."""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT content, metadata FROM archival_memory "
                "WHERE status = 'active' AND namespace NOT LIKE 'locomo%%' "
                "ORDER BY created_at DESC LIMIT %s",
                (int(limit),),
            ).fetchall()
    except Exception as e:
        logger.error("get_recent_archival failed: %s", e)
        return []

    results = []
    for row in rows:
        meta = row[1] if isinstance(row[1], dict) else json.loads(row[1])
        results.append({"content": row[0], "metadata": meta, "score": 1.0})
    return results


# --- Structured Metadata: Entity & Temporal Search ---


def ensure_structured_indexes() -> None:
    """Create GIN indexes on metadata->'entities' and metadata->'temporal' if missing."""
    try:
        with get_conn() as conn:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_archival_entities "
                "ON archival_memory USING GIN ((metadata->'entities'))"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_archival_temporal "
                "ON archival_memory USING GIN ((metadata->'temporal'))"
            )
            conn.commit()
    except Exception as e:
        logger.warning("ensure_structured_indexes failed: %s", e)


def search_by_entity(
    entity_name: str, namespace: str, limit: int = 10
) -> list[dict]:
    """Search archival entries by entity name in metadata->'entities'."""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, content, metadata, created_at
                FROM archival_memory
                WHERE namespace = %s
                  AND status = 'active'
                  AND metadata->'entities' @> %s::jsonb
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (namespace, json.dumps([{"name": entity_name}]), limit),
            ).fetchall()
            return [
                {
                    "id": str(r[0]),
                    "content": r[1],
                    "metadata": r[2] if isinstance(r[2], dict) else json.loads(r[2]),
                    "created_at": r[3].isoformat() if r[3] else None,
                }
                for r in rows
            ]
    except Exception as e:
        logger.error("search_by_entity failed: %s", e)
        return []


def search_by_temporal(
    date_str: str, namespace: str, limit: int = 10
) -> list[dict]:
    """Search archival entries by temporal absolute date in metadata->'temporal'."""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, content, metadata, created_at
                FROM archival_memory
                WHERE namespace = %s
                  AND status = 'active'
                  AND metadata->'temporal'->>'absolute' = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (namespace, date_str, limit),
            ).fetchall()
            return [
                {
                    "id": str(r[0]),
                    "content": r[1],
                    "metadata": r[2] if isinstance(r[2], dict) else json.loads(r[2]),
                    "created_at": r[3].isoformat() if r[3] else None,
                }
                for r in rows
            ]
    except Exception as e:
        logger.error("search_by_temporal failed: %s", e)
        return []


def search_by_temporal_range(
    start_date: str, end_date: str, namespace: str, limit: int = 10
) -> list[dict]:
    """Search archival entries by temporal date range."""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT id, content, metadata, created_at
                FROM archival_memory
                WHERE namespace = %s
                  AND status = 'active'
                  AND (metadata->'temporal'->>'absolute')::date BETWEEN %s AND %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (namespace, start_date, end_date, limit),
            ).fetchall()
            return [
                {
                    "id": str(r[0]),
                    "content": r[1],
                    "metadata": r[2] if isinstance(r[2], dict) else json.loads(r[2]),
                    "created_at": r[3].isoformat() if r[3] else None,
                }
                for r in rows
            ]
    except Exception as e:
        logger.error("search_by_temporal_range failed: %s", e)
        return []
