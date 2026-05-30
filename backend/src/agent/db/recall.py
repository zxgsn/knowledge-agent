"""Recall memory: search, save, recent retrieval, and cleanup."""

from __future__ import annotations

import json
import uuid

from agent.db._base import get_conn

from agent.db._base import logger


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


def search_recall(query: str, limit: int = 5, thread_id: str | None = None) -> list[dict]:
    """Semantic search over recall_memory (conversation history).

    Uses pure cosine similarity with optional cross-encoder re-ranking.

    Args:
        query: Search query text.
        limit: Max results after re-ranking.
        thread_id: If provided, only search within this thread/session.
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
            if thread_id:
                rows = conn.execute(
                    """
                    SELECT id, thread_id, role, content, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM recall_memory
                    WHERE thread_id = %s
                      AND 1 - (embedding <=> %s::vector) > 0.3
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query_embedding, thread_id, query_embedding, candidate_limit),
                ).fetchall()
            else:
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
        results = rerank(query, results, top_k=int(limit) * 2)
    except Exception:
        results = results[: int(limit) * 2]

    # Dedup by content similarity -- keep higher-scored entry
    deduped = []
    for r in results:
        r_words = set(r["content"].split())
        is_dup = False
        for kept in deduped:
            kept_words = set(kept["content"].split())
            if not r_words or not kept_words:
                continue
            overlap = len(r_words & kept_words) / min(len(r_words), len(kept_words))
            if overlap > 0.8:
                is_dup = True
                break
        if not is_dup:
            deduped.append(r)
    results = deduped[: int(limit)]

    return results


def get_recent_recall(limit: int = 5, thread_id: str | None = None) -> list[dict]:
    """Fetch most recent recall memory entries.

    Args:
        limit: Max entries to return.
        thread_id: If provided, only fetch from this thread/session.
    """
    from agent.storage import ensure_recall_table

    try:
        ensure_recall_table()
        with get_conn() as conn:
            if thread_id:
                rows = conn.execute(
                    "SELECT id, thread_id, role, content, metadata "
                    "FROM recall_memory WHERE thread_id = %s "
                    "ORDER BY created_at DESC LIMIT %s",
                    (thread_id, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, thread_id, role, content, metadata "
                    "FROM recall_memory ORDER BY created_at DESC LIMIT %s",
                    (int(limit),),
                ).fetchall()
    except Exception as e:
        logger.error("get_recent_recall failed: %s", e)
        return []

    results = []
    for row in rows:
        meta = row[4] if isinstance(row[4], dict) else json.loads(row[4])
        results.append({
            "id": row[0],
            "thread_id": row[1],
            "role": row[2],
            "content": row[3],
            "metadata": meta,
            "score": 1.0,
        })
    return results


def cleanup_recall(
    thread_id: str, max_age_days: int = 90, max_entries: int = 500
) -> tuple[int, int]:
    """Remove old recall entries and enforce a per-thread entry cap.

    Returns:
        (expired_deleted, excess_deleted) counts.
    """
    from agent.storage import ensure_recall_table

    try:
        ensure_recall_table()
        with get_conn() as conn:
            # TTL cleanup
            expired_result = conn.execute(
                "DELETE FROM recall_memory WHERE thread_id = %s "
                "AND created_at < NOW() - INTERVAL '%s days'",
                (thread_id, max_age_days),
            )
            expired_count = expired_result.rowcount

            # Excess cleanup: keep only the most recent N
            count = conn.execute(
                "SELECT COUNT(*) FROM recall_memory WHERE thread_id = %s",
                (thread_id,),
            ).fetchone()[0]

            excess_count = 0
            if count > max_entries:
                excess_ids = conn.execute(
                    "SELECT id FROM recall_memory WHERE thread_id = %s "
                    "ORDER BY created_at ASC LIMIT %s",
                    (thread_id, count - max_entries),
                ).fetchall()
                if excess_ids:
                    conn.execute(
                        "DELETE FROM recall_memory WHERE id = ANY(%s)",
                        ([r[0] for r in excess_ids],),
                    )
                    excess_count = len(excess_ids)

            conn.commit()
            return expired_count, excess_count
    except Exception as e:
        logger.error("cleanup_recall failed: %s", e)
        return 0, 0
