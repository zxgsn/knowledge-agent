"""Conflict review: queue, list, and resolve memory conflicts."""

from __future__ import annotations

import json
import uuid

from agent.db._base import get_conn

from agent.db._base import logger


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
    from agent.db.versioning import _snapshot_version

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
