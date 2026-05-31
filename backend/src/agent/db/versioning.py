"""Version history: snapshot, query, delete, and rollback for archival, recall, and core memory."""

from __future__ import annotations

import json
import uuid

from agent.db._base import get_conn

from agent.db._base import logger


def _ensure_recall_versions_table(conn) -> None:
    """Create recall_versions table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recall_versions (
            id TEXT PRIMARY KEY,
            recall_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            embedding VECTOR(1024),
            version_number INT NOT NULL,
            change_type TEXT NOT NULL,
            changed_by TEXT DEFAULT 'system',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_recall_versions_recall_id "
        "ON recall_versions (recall_id)"
    )


def _ensure_core_memory_versions_table(conn) -> None:
    """Create core_memory_versions table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS core_memory_versions (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            block_label TEXT NOT NULL,
            block_value TEXT NOT NULL,
            version_number INT NOT NULL,
            change_type TEXT NOT NULL DEFAULT 'update',
            changed_by TEXT DEFAULT 'system',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_core_mem_versions_thread "
        "ON core_memory_versions (thread_id, block_label)"
    )


# --- Archival versioning (existing) ---


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


# --- Recall Memory versioning ---


def snapshot_recall_version(
    entry_id: str,
    change_type: str = "delete",
    changed_by: str = "user",
) -> None:
    """Snapshot a recall memory entry before mutation (e.g., delete)."""
    with get_conn() as conn:
        _ensure_recall_versions_table(conn)
        row = conn.execute(
            "SELECT thread_id, role, content, metadata, embedding "
            "FROM recall_memory WHERE id = %s",
            (entry_id,),
        ).fetchone()
        if not row:
            return

        max_ver = conn.execute(
            "SELECT COALESCE(MAX(version_number), 0) FROM recall_versions WHERE recall_id = %s",
            (entry_id,),
        ).fetchone()[0]

        version_id = str(uuid.uuid4())
        meta = row[3] if isinstance(row[3], str) else json.dumps(row[3] or {})
        conn.execute(
            "INSERT INTO recall_versions (id, recall_id, thread_id, role, content, metadata, "
            "embedding, version_number, change_type, changed_by) "
            "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)",
            (version_id, entry_id, row[0], row[1], row[2], meta,
             row[4], max_ver + 1, change_type, changed_by),
        )
        conn.commit()


def get_recall_version_history(entry_id: str) -> list[dict]:
    """Get all version snapshots for a recall entry, newest first."""
    with get_conn() as conn:
        _ensure_recall_versions_table(conn)
        rows = conn.execute(
            "SELECT id, content, metadata, version_number, change_type, changed_by, "
            "created_at::text, thread_id, role "
            "FROM recall_versions WHERE recall_id = %s ORDER BY version_number DESC",
            (entry_id,),
        ).fetchall()
    results = []
    for row in rows:
        meta = row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}")
        results.append({
            "version_id": row[0],
            "content": row[1],
            "metadata": meta,
            "version_number": row[3],
            "change_type": row[4],
            "changed_by": row[5],
            "created_at": row[6],
            "thread_id": row[7],
            "role": row[8],
        })
    return results


def rollback_recall_version(entry_id: str, version_number: int) -> bool:
    """Restore a deleted recall entry to a specific version."""
    with get_conn() as conn:
        _ensure_recall_versions_table(conn)
        version_row = conn.execute(
            "SELECT content, metadata, embedding, thread_id, role "
            "FROM recall_versions WHERE recall_id = %s AND version_number = %s",
            (entry_id, version_number),
        ).fetchone()
        if not version_row:
            return False

        meta = version_row[1] if isinstance(version_row[1], str) else json.dumps(version_row[1] or {})

        # Check if entry still exists
        existing = conn.execute(
            "SELECT id FROM recall_memory WHERE id = %s", (entry_id,)
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE recall_memory SET content = %s, metadata = %s::jsonb, embedding = %s "
                "WHERE id = %s",
                (version_row[0], meta, version_row[2], entry_id),
            )
        else:
            conn.execute(
                "INSERT INTO recall_memory (id, thread_id, role, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s, %s::jsonb, %s) ON CONFLICT (id) DO UPDATE SET "
                "content=EXCLUDED.content, metadata=EXCLUDED.metadata, "
                "embedding=EXCLUDED.embedding",
                (entry_id, version_row[3], version_row[4], version_row[0], meta, version_row[2]),
            )
        conn.commit()
    return True


def delete_recall_version(version_id: str) -> bool:
    """Delete a single recall version snapshot."""
    with get_conn() as conn:
        _ensure_recall_versions_table(conn)
        result = conn.execute(
            "DELETE FROM recall_versions WHERE id = %s", (version_id,)
        )
        conn.commit()
        return result.rowcount > 0


# --- Core Memory versioning ---


def snapshot_core_memory_block(
    thread_id: str,
    block_label: str,
    block_value: str,
    change_type: str = "update",
    changed_by: str = "agent",
) -> None:
    """Persist a snapshot of a core memory block before mutation."""
    with get_conn() as conn:
        _ensure_core_memory_versions_table(conn)

        max_ver = conn.execute(
            "SELECT COALESCE(MAX(version_number), 0) FROM core_memory_versions "
            "WHERE thread_id = %s AND block_label = %s",
            (thread_id, block_label),
        ).fetchone()[0]

        version_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO core_memory_versions (id, thread_id, block_label, block_value, "
            "version_number, change_type, changed_by) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (version_id, thread_id, block_label, block_value,
             max_ver + 1, change_type, changed_by),
        )
        conn.commit()


def get_core_memory_version_history(thread_id: str, block_label: str) -> list[dict]:
    """Get all version snapshots for a core memory block, newest first."""
    with get_conn() as conn:
        _ensure_core_memory_versions_table(conn)
        rows = conn.execute(
            "SELECT id, block_value, version_number, change_type, changed_by, created_at::text "
            "FROM core_memory_versions WHERE thread_id = %s AND block_label = %s "
            "ORDER BY version_number DESC",
            (thread_id, block_label),
        ).fetchall()
    return [
        {
            "version_id": row[0],
            "block_value": row[1],
            "version_number": row[2],
            "change_type": row[3],
            "changed_by": row[4],
            "created_at": row[5],
        }
        for row in rows
    ]


def rollback_core_memory_block(thread_id: str, block_label: str, version_number: int) -> str | None:
    """Restore a core memory block to a specific version. Returns the restored value."""
    with get_conn() as conn:
        _ensure_core_memory_versions_table(conn)
        version_row = conn.execute(
            "SELECT block_value FROM core_memory_versions "
            "WHERE thread_id = %s AND block_label = %s AND version_number = %s",
            (thread_id, block_label, version_number),
        ).fetchone()
        if not version_row:
            return None
        return version_row[0]
