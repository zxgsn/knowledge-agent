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
    ) -> list[dict]:
        """Semantic search over archival memory.

        Returns a list of {id, content, metadata, score} dicts.
        """
        query_embedding = Vector(self.embeddings.embed_query(query))

        sql = """
            SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS score
            FROM archival_memory
            WHERE namespace = %s
        """
        params: list = [query_embedding, namespace]

        if metadata_filter:
            for key, value in metadata_filter.items():
                sql += f" AND metadata->>%s = %s"
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
