"""Importance scoring: DB schema helpers and query functions.

Manages importance_score / access_count / last_accessed columns
on the archival_memory table and provides importance-weighted queries.
"""

from __future__ import annotations

import json

from agent.db._base import get_conn

from agent.db._base import logger


def ensure_importance_columns() -> None:
    """Add importance_score, access_count, last_accessed columns to archival_memory if missing.

    Idempotent: safe to run multiple times.
    """
    try:
        with get_conn() as conn:
            for col, ddl in [
                ("importance_score", "ALTER TABLE archival_memory ADD COLUMN importance_score REAL DEFAULT 0.5"),
                ("access_count", "ALTER TABLE archival_memory ADD COLUMN access_count INTEGER DEFAULT 0"),
                ("last_accessed", "ALTER TABLE archival_memory ADD COLUMN last_accessed TIMESTAMPTZ"),
            ]:
                exists = conn.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = 'archival_memory' AND column_name = %s",
                    (col,),
                ).fetchone()
                if not exists:
                    conn.execute(ddl)
            # Index on importance_score for fast sorting
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_archival_importance "
                "ON archival_memory (importance_score DESC) WHERE status = 'active'"
            )
            conn.commit()
            logger.info("Importance columns ensured on archival_memory")
    except Exception as e:
        logger.warning("ensure_importance_columns failed: %s", e)


def update_importance_score(entry_id: str, score: float) -> None:
    """Update the importance_score for a single archival entry."""
    try:
        with get_conn() as conn:
            conn.execute(
                "UPDATE archival_memory SET importance_score = %s WHERE id = %s",
                (float(score), entry_id),
            )
            conn.commit()
    except Exception as e:
        logger.error("update_importance_score failed for %s: %s", entry_id, e)


def touch_memory_access(entry_id: str) -> None:
    """Increment access_count and set last_accessed for an entry."""
    try:
        with get_conn() as conn:
            conn.execute(
                "UPDATE archival_memory SET access_count = access_count + 1, "
                "last_accessed = NOW() WHERE id = %s",
                (entry_id,),
            )
            conn.commit()
    except Exception as e:
        logger.error("touch_memory_access failed for %s: %s", entry_id, e)


def get_memories_by_importance(
    namespace: str = "",
    limit: int = 20,
    min_score: float = 0.0,
) -> list[dict]:
    """Fetch memories ordered by importance_score (highest first).

    Args:
        namespace: If provided, filter to this namespace. Empty = all non-locomo.
        limit: Max results to return.
        min_score: Minimum importance score threshold.

    Returns:
        List of dicts with id, content, metadata, namespace, importance_score,
        access_count, last_accessed, created_at.
    """
    try:
        with get_conn() as conn:
            if namespace:
                rows = conn.execute(
                    "SELECT id, content, metadata, namespace, "
                    "COALESCE(importance_score, 0.5), access_count, "
                    "last_accessed::text, created_at::text "
                    "FROM archival_memory "
                    "WHERE namespace = %s AND status = 'active' AND importance_score >= %s "
                    "ORDER BY importance_score DESC LIMIT %s",
                    (namespace, min_score, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, content, metadata, namespace, "
                    "COALESCE(importance_score, 0.5), access_count, "
                    "last_accessed::text, created_at::text "
                    "FROM archival_memory "
                    "WHERE status = 'active' AND namespace NOT LIKE %s AND importance_score >= %s "
                    "ORDER BY importance_score DESC LIMIT %s",
                    ("locomo%", min_score, int(limit)),
                ).fetchall()
    except Exception as e:
        logger.error("get_memories_by_importance failed: %s", e)
        return []

    results = []
    for row in rows:
        meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
        results.append({
            "id": row[0],
            "content": row[1],
            "metadata": meta,
            "namespace": row[3],
            "importance_score": float(row[4]),
            "access_count": row[5] or 0,
            "last_accessed": row[6],
            "created_at": row[7],
        })
    return results


def search_importance_weighted(
    query: str,
    namespace: str = "",
    limit: int = 10,
    importance_weight: float = 0.3,
) -> list[dict]:
    """Hybrid search: vector similarity weighted by importance_score.

    Final score = (1 - importance_weight) * similarity + importance_weight * importance_score.

    Args:
        query: Search query text.
        namespace: Optional namespace filter. Empty string = all non-locomo.
        limit: Max results.
        importance_weight: How much importance contributes to final score (0-1).

    Returns:
        List of result dicts with content, metadata, score, cosine_score,
        importance_score, namespace.
    """
    from pgvector import Vector

    from agent.storage import get_embeddings

    try:
        embeddings = get_embeddings()
        query_embedding = Vector(embeddings.embed_query(query))
    except Exception as e:
        logger.error("search_importance_weighted embedding failed: %s", e)
        return []

    try:
        w_imp = importance_weight
        w_sim = 1.0 - w_imp
        with get_conn() as conn:
            if namespace:
                rows = conn.execute(
                    f"""
                    SELECT id, content, metadata, namespace,
                           1 - (embedding <=> %s::vector) AS cosine_score,
                           COALESCE(importance_score, 0.5) AS imp_score,
                           {w_sim} * (1 - (embedding <=> %s::vector))
                           + {w_imp} * COALESCE(importance_score, 0.5) AS weighted_score
                    FROM archival_memory
                    WHERE namespace = %s AND status = 'active'
                      AND 1 - (embedding <=> %s::vector) > 0.15
                    ORDER BY weighted_score DESC
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, namespace, query_embedding, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT id, content, metadata, namespace,
                           1 - (embedding <=> %s::vector) AS cosine_score,
                           COALESCE(importance_score, 0.5) AS imp_score,
                           {w_sim} * (1 - (embedding <=> %s::vector))
                           + {w_imp} * COALESCE(importance_score, 0.5) AS weighted_score
                    FROM archival_memory
                    WHERE namespace NOT LIKE %s AND status = 'active'
                      AND 1 - (embedding <=> %s::vector) > 0.15
                    ORDER BY weighted_score DESC
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, "locomo%", query_embedding, int(limit)),
                ).fetchall()
    except Exception as e:
        logger.error("search_importance_weighted search failed: %s", e)
        return []

    results = []
    for row in rows:
        meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
        cosine = float(row[4])
        imp = float(row[5])
        results.append({
            "id": row[0],
            "content": row[1],
            "metadata": meta,
            "namespace": row[3],
            "score": round(float(row[6]), 4),
            "cosine_score": round(cosine, 4),
            "importance_score": round(imp, 4),
        })
    return results


def get_memory_stats_extended() -> dict:
    """Extended stats including importance distribution, access stats, and top entries.

    Returns a dict with:
    - importance_distribution: list of bucket dicts for histogram
    - importance: summary stats (avg, min, max, median)
    - access: access count summary
    - top_important: entries with highest importance
    - recently_accessed: entries most recently accessed
    """
    stats: dict = {}
    try:
        with get_conn() as conn:
            # Importance distribution (5 buckets: 0-0.2, 0.2-0.4, etc.)
            buckets = []
            bucket_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
            bucket_bounds = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            for label, (lo, hi) in zip(bucket_labels, bucket_bounds):
                row = conn.execute(
                    "SELECT COUNT(*) FROM archival_memory "
                    "WHERE status = 'active' "
                    "AND COALESCE(importance_score, 0.5) >= %s "
                    "AND COALESCE(importance_score, 0.5) < %s",
                    (lo, hi),
                ).fetchone()
                buckets.append({"range": label, "count": row[0]})
            stats["importance_distribution"] = buckets

            # Importance summary stats
            row = conn.execute(
                "SELECT "
                "AVG(importance_score), MIN(importance_score), MAX(importance_score), "
                "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY importance_score) "
                "FROM archival_memory WHERE status = 'active' AND importance_score IS NOT NULL"
            ).fetchone()
            if row and row[0] is not None:
                stats["importance_summary"] = {
                    "avg": round(float(row[0]), 4),
                    "min": round(float(row[1]), 4),
                    "max": round(float(row[2]), 4),
                    "median": round(float(row[3]), 4),
                }

            # Access distribution
            row = conn.execute(
                "SELECT AVG(access_count), MAX(access_count), "
                "COUNT(*) FILTER (WHERE access_count > 0) "
                "FROM archival_memory WHERE status = 'active'"
            ).fetchone()
            if row:
                stats["access"] = {
                    "avg": round(float(row[0] or 0), 2),
                    "max": row[1] or 0,
                    "entries_with_access": row[2] or 0,
                }

            # Top entities from metadata->'entities'
            try:
                entity_rows = conn.execute(
                    """
                    SELECT elem->>'name' AS entity_name, COUNT(*) AS mention_count
                    FROM archival_memory,
                         jsonb_array_elements(metadata->'entities') AS elem
                    WHERE status = 'active'
                      AND metadata->'entities' IS NOT NULL
                      AND jsonb_typeof(metadata->'entities') = 'array'
                    GROUP BY elem->>'name'
                    ORDER BY mention_count DESC
                    LIMIT 20
                    """
                ).fetchall()
                stats["top_entities"] = [
                    {"entity": r[0], "count": r[1]} for r in entity_rows
                ]
            except Exception:
                stats["top_entities"] = []

            # Memories with highest importance
            top_rows = conn.execute(
                "SELECT id, content, COALESCE(importance_score, 0.5) "
                "FROM archival_memory "
                "WHERE status = 'active' "
                "ORDER BY importance_score DESC NULLS LAST LIMIT 5"
            ).fetchall()
            stats["top_important"] = [
                {"id": r[0], "content": r[1][:200], "score": float(r[2])}
                for r in top_rows
            ]

            # Recently accessed entries
            access_rows = conn.execute(
                "SELECT id, content, access_count, last_accessed::text "
                "FROM archival_memory "
                "WHERE status = 'active' AND last_accessed IS NOT NULL "
                "ORDER BY last_accessed DESC LIMIT 5"
            ).fetchall()
            stats["recently_accessed"] = [
                {"id": r[0], "content": r[1][:200], "access_count": r[2], "last_accessed": r[3]}
                for r in access_rows
            ]

    except Exception as e:
        logger.error("get_memory_stats_extended failed: %s", e)
        stats["error"] = str(e)

    return stats
