"""Archival Memory: long-term semantic storage backed by PostgreSQL + pgvector."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import psycopg
from pgvector.psycopg import register_vector_async
from pgvector import Vector

from agent.storage.embedding import DashScopeEmbeddings


class ArchivalMemory:
    """Persistent semantic storage for knowledge entries.

    Each entry is stored with its text content, metadata, and an embedding vector.
    Retrieval uses cosine similarity search via pgvector.
    """

    def __init__(self, conn: psycopg.AsyncConnection, embeddings: DashScopeEmbeddings):
        self.conn = conn
        self.embeddings = embeddings

    @classmethod
    async def create(
        cls, database_url: str, embeddings: DashScopeEmbeddings
    ) -> ArchivalMemory:
        """Create an ArchivalMemory instance and initialize the database table."""
        conn = await psycopg.AsyncConnection.connect(database_url)
        await register_vector_async(conn)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS archival_memory (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                embedding vector(1024),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_archival_namespace
            ON archival_memory (namespace)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_archival_embedding
            ON archival_memory USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        # Hybrid search: tsvector column + GIN index for BM25
        await conn.execute(
            "ALTER TABLE archival_memory ADD COLUMN IF NOT EXISTS content_tsv tsvector"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_archival_tsv ON archival_memory USING GIN (content_tsv)"
        )
        await conn.execute(
            "UPDATE archival_memory SET content_tsv = to_tsvector('simple', content) "
            "WHERE content_tsv IS NULL"
        )
        await conn.execute("""
            CREATE OR REPLACE FUNCTION archival_memory_tsv_trigger() RETURNS trigger AS $$
            BEGIN
              NEW.content_tsv := to_tsvector('simple', NEW.content);
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """)
        await conn.execute("""
            DROP TRIGGER IF EXISTS tsvectorupdate ON archival_memory;
            CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE ON archival_memory
              FOR EACH ROW EXECUTE FUNCTION archival_memory_tsv_trigger()
        """)

        # Documents table: stores full documents before chunking
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT,
                source_type TEXT,
                content_full TEXT NOT NULL,
                chunk_count INT DEFAULT 0,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_created
            ON documents (created_at DESC)
        """)
        await conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_title_source
            ON documents (title, source)
        """)

        # Link archival_memory chunks to documents
        await conn.execute(
            "ALTER TABLE archival_memory ADD COLUMN IF NOT EXISTS document_id TEXT"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_archival_document "
            "ON archival_memory (document_id)"
        )

        await conn.commit()

        return cls(conn=conn, embeddings=embeddings)

    async def put(
        self,
        content: str,
        namespace: str = "default",
        metadata: dict | None = None,
        entry_id: str | None = None,
    ) -> str:
        """Store a knowledge entry with its embedding.

        Returns the entry ID.
        """
        entry_id = entry_id or str(uuid.uuid4())
        metadata = metadata or {}
        metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

        embedding = self.embeddings.embed_query(content)

        await self.conn.execute(
            """
            INSERT INTO archival_memory (id, namespace, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
            """,
            (entry_id, namespace, content, json.dumps(metadata), embedding),
        )
        await self.conn.commit()
        return entry_id

    async def put_batch(
        self,
        entries: list[dict],
        namespace: str = "default",
    ) -> list[str]:
        """Store multiple entries in batch.

        Each entry dict should have: {"content": str, "metadata": dict}
        Returns list of entry IDs.
        """
        if not entries:
            return []

        # Batch embed all contents at once
        contents = [e["content"] for e in entries]
        embeddings = self.embeddings.embed_documents(contents)

        now = datetime.now(timezone.utc).isoformat()
        ids: list[str] = []

        for entry, embedding in zip(entries, embeddings):
            entry_id = str(uuid.uuid4())
            metadata = entry.get("metadata", {})
            metadata["timestamp"] = now
            ids.append(entry_id)

            await self.conn.execute(
                """
                INSERT INTO archival_memory (id, namespace, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (entry_id, namespace, entry["content"], json.dumps(metadata), embedding),
            )

        await self.conn.commit()
        return ids

    async def search(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 5,
        metadata_filter: dict | None = None,
        alpha: float = 0.7,
    ) -> list[dict]:
        """Hybrid search: vector similarity + BM25 keyword match.

        Uses websearch_to_tsquery (OR logic) for BM25 and normalizes
        ts_rank to [0,1] so it's on the same scale as cosine similarity.

        Args:
            query: Search query.
            namespace: Namespace to search in.
            limit: Max results.
            metadata_filter: Optional key/value filters on metadata.
            alpha: Weight for vector score (1-alpha for BM25).

        Returns a list of {id, content, metadata, score} dicts.
        """
        query_embedding = Vector(self.embeddings.embed_query(query))

        sql = """
            SELECT id, content, metadata,
                   %s * (1 - (embedding <=> %s::vector))
                     + (1 - %s) * LEAST(1, ts_rank(content_tsv, plainto_tsquery('english', %s)) * 5)
                   AS score
            FROM archival_memory
            WHERE namespace = %s
              AND 1 - (embedding <=> %s::vector) > 0.15
        """
        params: list = [alpha, query_embedding, alpha, query, namespace, query_embedding]

        if metadata_filter:
            for key, value in metadata_filter.items():
                sql += " AND metadata->>%s = %s"
                params.extend([key, str(value)])

        sql += " ORDER BY score DESC LIMIT %s"
        params.append(limit)

        rows = await self.conn.execute(sql, params)
        results = []
        async for row in rows:
            results.append({
                "id": row[0],
                "content": row[1],
                "metadata": row[2] if isinstance(row[2], dict) else json.loads(row[2]),
                "score": float(row[3]),
            })
        return results

    async def delete(self, entry_id: str) -> bool:
        """Delete a knowledge entry by ID."""
        result = await self.conn.execute(
            "DELETE FROM archival_memory WHERE id = %s", (entry_id,)
        )
        await self.conn.commit()
        return result.rowcount > 0

    async def list_entries(
        self, namespace: str = "default", limit: int = 50
    ) -> list[dict]:
        """List recent entries in a namespace."""
        rows = await self.conn.execute(
            "SELECT id, content, metadata, created_at FROM archival_memory "
            "WHERE namespace = %s ORDER BY created_at DESC LIMIT %s",
            (namespace, limit),
        )
        results = []
        async for row in rows:
            results.append({
                "id": row[0],
                "content": row[1],
                "metadata": row[2] if isinstance(row[2], dict) else json.loads(row[2]),
                "created_at": row[3].isoformat() if row[3] else None,
            })
        return results

    async def close(self):
        await self.conn.close()
