"""Version history: snapshot, query, delete, and rollback archival versions."""

from __future__ import annotations

import json
import uuid

from agent.db._base import get_conn

from agent.db._base import logger


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
