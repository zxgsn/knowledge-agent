"""Centralized sync DB helpers for archival and recall memory.

All functions use the connection pool from agent.storage.
Import these instead of duplicating DB logic across node files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone

from agent.storage import get_conn

logger = logging.getLogger(__name__)

# Unified BM25 text search configuration
_TS_CONFIG = "english"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mmr_rerank(
    query_embedding: list[float],
    results: list[dict],
    lambda_param: float = 0.5,
    top_k: int | None = None,
) -> list[dict]:
    """Maximal Marginal Relevance: balance relevance and diversity.

    Args:
        query_embedding: The query vector.
        results: List of dicts with 'content' and 'score' keys, sorted by score desc.
        lambda_param: 1.0 = pure relevance, 0.0 = pure diversity.
        top_k: Max results to return.
    """
    if len(results) <= 1:
        return results

    from agent.storage import get_embeddings

    embeddings = get_embeddings()
    doc_embeddings = [embeddings.embed_query(r["content"][:512]) for r in results]

    selected = [0]
    candidates = list(range(1, len(results)))

    while candidates:
        best_score, best_idx = float("-inf"), -1
        for c in candidates:
            relevance = results[c]["score"]
            max_sim = max(
                _cosine_similarity(doc_embeddings[c], doc_embeddings[s])
                for s in selected
            )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, c
        selected.append(best_idx)
        candidates.remove(best_idx)

    ordered = [results[i] for i in selected]
    return ordered[:top_k] if top_k else ordered


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
                         + (1 - %s) * LEAST(1, ts_rank(content_tsv, plainto_tsquery('{_TS_CONFIG}', %s)) * 5)
                       AS score
                FROM archival_memory
                WHERE status = 'active' AND 1 - (embedding <=> %s::vector) > 0.15
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
    except Exception:
        results = results[: int(limit)]

    # MMR deduplication
    _mmr_on = mmr_enabled if mmr_enabled is not None else os.getenv("MMR_ENABLED", "false").lower() in ("true", "1", "yes")
    if _mmr_on and len(results) > 1:
        _lambda = mmr_lambda if mmr_lambda is not None else float(os.getenv("MMR_LAMBDA", "0.5"))
        results = _mmr_rerank(query_embedding_flat, results, lambda_param=_lambda, top_k=int(limit))

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
                  AND status = 'active'
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


def _ensure_content_hash_column() -> None:
    """Add content_hash column and unique index to documents table if missing."""
    with get_conn() as conn:
        conn.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'documents' AND column_name = 'content_hash'
                ) THEN
                    ALTER TABLE documents ADD COLUMN content_hash TEXT;
                    UPDATE documents SET content_hash = encode(sha256(convert_to(content_full, 'UTF8')), 'hex')
                        WHERE content_hash IS NULL;
                    ALTER TABLE documents ALTER COLUMN content_hash SET NOT NULL;
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash ON documents (content_hash);
                END IF;
            END$$;
        """)
        conn.commit()


def insert_document(
    title: str, source: str, source_type: str, content_full: str, chunk_count: int
) -> tuple[str, bool]:
    """Insert a document record. Returns (doc_id, is_new).

    Deduplicates by SHA256 hash of content, not by filename.
    """
    _ensure_content_hash_column()

    content_hash = hashlib.sha256(content_full.encode("utf-8")).hexdigest()
    doc_id = str(uuid.uuid4())

    with get_conn() as conn:
        result = conn.execute(
            "INSERT INTO documents (id, title, source, source_type, content_full, chunk_count, content_hash) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (content_hash) DO NOTHING "
            "RETURNING id",
            (doc_id, title, source, source_type, content_full, chunk_count, content_hash),
        )
        row = result.fetchone()
        if row:
            conn.commit()
            return row[0], True
        # Conflict — document with same content already exists
        existing = conn.execute(
            "SELECT id FROM documents WHERE content_hash = %s",
            (content_hash,),
        ).fetchone()
        return existing[0], False


def update_archival(
    entry_id: str, content: str, metadata: dict, changed_by: str = "memory_pipeline"
) -> None:
    """Update an existing archival entry's content and embedding. Snapshots before mutation."""
    from pgvector import Vector

    from agent.storage import get_embeddings

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


# --- Version History ---


def _snapshot_version(
    conn,
    entry_id: str,
    change_type: str,
    changed_by: str = "memory_pipeline",
) -> None:
    """Snapshot the current state of an archival entry before mutation."""
    row = conn.execute(
        "SELECT content, metadata, embedding FROM archival_memory WHERE id = %s",
        (entry_id,),
    ).fetchone()
    if not row:
        return

    max_ver = conn.execute(
        "SELECT COALESCE(MAX(version_number), 0) FROM archival_versions WHERE archival_id = %s",
        (entry_id,),
    ).fetchone()[0]

    version_id = str(uuid.uuid4())
    meta = row[1] if isinstance(row[1], str) else json.dumps(row[1])
    conn.execute(
        "INSERT INTO archival_versions (id, archival_id, content, metadata, embedding, "
        "version_number, change_type, changed_by) "
        "VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s)",
        (version_id, entry_id, row[0], meta, row[2], max_ver + 1, change_type, changed_by),
    )


def get_version_history(entry_id: str) -> list[dict]:
    """Get all version snapshots for an archival entry, newest first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, content, metadata, version_number, change_type, changed_by, created_at::text "
            "FROM archival_versions WHERE archival_id = %s ORDER BY version_number DESC",
            (entry_id,),
        ).fetchall()
    results = []
    for row in rows:
        meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
        results.append({
            "version_id": row[0],
            "content": row[1],
            "metadata": meta,
            "version_number": row[3],
            "change_type": row[4],
            "changed_by": row[5],
            "created_at": row[6],
        })
    return results


def delete_version(version_id: str) -> bool:
    """Delete a single version snapshot."""
    with get_conn() as conn:
        result = conn.execute(
            "DELETE FROM archival_versions WHERE id = %s", (version_id,)
        )
        conn.commit()
        return result.rowcount > 0


def rollback_to_version(entry_id: str, version_number: int) -> bool:
    """Restore an archival entry to a specific version."""
    with get_conn() as conn:
        version_row = conn.execute(
            "SELECT content, metadata, embedding FROM archival_versions "
            "WHERE archival_id = %s AND version_number = %s",
            (entry_id, version_number),
        ).fetchone()
        if not version_row:
            return False

        meta = version_row[1] if isinstance(version_row[1], str) else json.dumps(version_row[1])

        current = conn.execute(
            "SELECT status FROM archival_memory WHERE id = %s", (entry_id,)
        ).fetchone()
        if current:
            _snapshot_version(conn, entry_id, "rollback", "user_rollback")
            conn.execute(
                "UPDATE archival_memory SET content = %s, metadata = %s::jsonb, embedding = %s, "
                "status = 'active' WHERE id = %s",
                (version_row[0], meta, version_row[2], entry_id),
            )
        else:
            conn.execute(
                "INSERT INTO archival_memory (id, content, metadata, embedding, status) "
                "VALUES (%s, %s, %s::jsonb, %s, 'active') ON CONFLICT (id) DO UPDATE SET "
                "content=EXCLUDED.content, metadata=EXCLUDED.metadata, "
                "embedding=EXCLUDED.embedding, status='active'",
                (entry_id, version_row[0], meta, version_row[2]),
            )
        conn.commit()
    return True


# --- Conflict Review ---


def queue_conflict_review(
    new_fact: str,
    existing_id: str,
    existing_content: str,
    similarity_score: float,
    llm_decision: str | None,
    llm_merged_text: str | None,
    llm_confidence: float | None,
) -> str:
    """Queue a low-confidence conflict for human review."""
    review_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conflict_reviews (id, new_fact, existing_id, existing_content, "
            "similarity_score, llm_decision, llm_merged_text, llm_confidence) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (review_id, new_fact, existing_id, existing_content,
             similarity_score, llm_decision, llm_merged_text, llm_confidence),
        )
        conn.commit()
    return review_id


def get_pending_conflicts(limit: int = 20) -> list[dict]:
    """Get pending conflict reviews for human resolution."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, new_fact, existing_id, existing_content, similarity_score, "
            "llm_decision, llm_merged_text, llm_confidence, created_at::text "
            "FROM conflict_reviews WHERE status = 'pending' ORDER BY created_at DESC LIMIT %s",
            (limit,),
        ).fetchall()
    return [
        {
            "id": r[0], "new_fact": r[1], "existing_id": r[2],
            "existing_content": r[3], "similarity_score": float(r[4]),
            "llm_decision": r[5], "llm_merged_text": r[6],
            "llm_confidence": float(r[7]) if r[7] else None,
            "created_at": r[8],
        }
        for r in rows
    ]


def resolve_conflict(
    review_id: str, action: str, resolution_text: str | None = None
) -> bool:
    """Resolve a conflict review. action: 'approve' | 'reject' | 'modify'."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT llm_decision, llm_merged_text, existing_id "
            "FROM conflict_reviews WHERE id = %s AND status = 'pending'",
            (review_id,),
        ).fetchone()
        if not row:
            return False

        conn.execute(
            "UPDATE conflict_reviews SET status = %s, resolution_text = %s, resolved_at = NOW() "
            "WHERE id = %s",
            (action, resolution_text, review_id),
        )

        if action == "approve":
            merged = resolution_text or row[1]
            if row[0] == "update" and merged:
                _snapshot_version(conn, row[2], "update", "conflict_review")
                from pgvector import Vector
                from agent.storage import get_embeddings
                emb = Vector(get_embeddings().embed_query(merged))
                meta = {"source": "conflict_review", "review_id": review_id}
                conn.execute(
                    "UPDATE archival_memory SET content = %s, metadata = %s, embedding = %s "
                    "WHERE id = %s",
                    (merged, json.dumps(meta), emb, row[2]),
                )
        elif action == "modify":
            if resolution_text:
                _snapshot_version(conn, row[2], "update", "conflict_review")
                from pgvector import Vector
                from agent.storage import get_embeddings
                emb = Vector(get_embeddings().embed_query(resolution_text))
                meta = {"source": "conflict_review", "review_id": review_id}
                conn.execute(
                    "UPDATE archival_memory SET content = %s, metadata = %s, embedding = %s "
                    "WHERE id = %s",
                    (resolution_text, json.dumps(meta), emb, row[2]),
                )

        conn.commit()
    return True


# --- Source Trust ---


SOURCE_TRUST = {
    "research_summary": 0.9,
    "manual": 1.0,
    "manual_save": 1.0,
    "manual_edit": 1.0,
    "conversation": 0.7,
    "memory_pipeline": 0.7,
    "consolidation": 0.8,
    "session_window": 0.5,
    "ingested": 0.8,
    "conflict_review": 0.9,
}


def get_source_trust(metadata: dict) -> tuple[float, str]:
    """Return (score, label) for a memory entry's source."""
    source = metadata.get("source", "unknown")
    score = SOURCE_TRUST.get(source, 0.5)
    if score >= 0.9:
        label = "verified"
    elif score >= 0.7:
        label = "default"
    else:
        label = "low"
    return score, label


# --- Delete (soft-delete) ---


def delete_archival(entry_id: str, changed_by: str = "memory_pipeline") -> bool:
    """Soft-delete an archival entry: snapshot then set status='superseded'."""
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
                WHERE namespace = %s AND status = 'active'
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
    """Keep only the newest max_entries in a namespace. Returns count soft-deleted."""
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
    except Exception:
        return 0


def get_recent_archival(limit: int = 5) -> list[dict]:
    """Fetch most recent archival entries across all namespaces."""
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT content, metadata FROM archival_memory "
                "WHERE status = 'active' "
                "ORDER BY created_at DESC LIMIT %s",
                (int(limit),),
            ).fetchall()
    except Exception:
        return []

    results = []
    for row in rows:
        meta = row[1] if isinstance(row[1], dict) else json.loads(row[1])
        results.append({"content": row[0], "metadata": meta, "score": 1.0})
    return results


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

    # Dedup by content similarity — keep higher-scored entry
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
    except Exception:
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
