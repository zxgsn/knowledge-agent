"""Recall Memory: searchable conversation history backed by PostgreSQL."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import psycopg
from pgvector import Vector

from agent.storage.embedding import DashScopeEmbeddings


class RecallMemory:
    """Persistent conversation history with semantic search.

    Stores all messages and supports both text and semantic retrieval.
    """

    def __init__(self, conn: psycopg.AsyncConnection, embeddings: DashScopeEmbeddings):
        self.conn = conn
        self.embeddings = embeddings

    @classmethod
    async def create(
        cls, conn: psycopg.AsyncConnection, embeddings: DashScopeEmbeddings
    ) -> RecallMemory:
        """Create a RecallMemory instance and initialize the database table."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS recall_memory (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                embedding vector(1024),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_recall_thread
            ON recall_memory (thread_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_recall_embedding
            ON recall_memory USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        await conn.commit()
        return cls(conn=conn, embeddings=embeddings)

    async def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        message_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Store a conversation message."""
        import uuid

        message_id = message_id or str(uuid.uuid4())
        metadata = metadata or {}
        embedding = self.embeddings.embed_query(content)

        await self.conn.execute(
            """
            INSERT INTO recall_memory (id, thread_id, role, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (message_id, thread_id, role, content, json.dumps(metadata), embedding),
        )
        await self.conn.commit()
        return message_id

    async def search(
        self,
        query: str,
        thread_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Semantic search over conversation history."""
        query_embedding = Vector(self.embeddings.embed_query(query))

        if thread_id:
            rows = await self.conn.execute(
                """
                SELECT id, thread_id, role, content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM recall_memory
                WHERE thread_id = %s
                ORDER BY score DESC
                LIMIT %s
                """,
                (query_embedding, thread_id, limit),
            )
        else:
            rows = await self.conn.execute(
                """
                SELECT id, thread_id, role, content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM recall_memory
                ORDER BY score DESC
                LIMIT %s
                """,
                (query_embedding, limit),
            )

        results = []
        async for row in rows:
            results.append({
                "id": row[0],
                "thread_id": row[1],
                "role": row[2],
                "content": row[3],
                "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
                "score": float(row[5]),
            })
        return results

    async def get_recent(
        self, thread_id: str, limit: int = 20
    ) -> list[dict]:
        """Get the most recent messages in a thread."""
        rows = await self.conn.execute(
            """
            SELECT id, role, content, metadata, created_at
            FROM recall_memory
            WHERE thread_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (thread_id, limit),
        )
        results = []
        async for row in rows:
            results.append({
                "id": row[0],
                "role": row[1],
                "content": row[2],
                "metadata": row[3] if isinstance(row[3], dict) else json.loads(row[3]),
                "created_at": row[4].isoformat() if row[4] else None,
            })
        return list(reversed(results))
