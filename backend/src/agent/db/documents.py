"""Document management: content-hash dedup and document insertion."""

from __future__ import annotations

import hashlib
import uuid

from agent.db._base import get_conn

from agent.db._base import logger


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
        # Conflict -- document with same content already exists
        existing = conn.execute(
            "SELECT id FROM documents WHERE content_hash = %s",
            (content_hash,),
        ).fetchone()
        return existing[0], False
