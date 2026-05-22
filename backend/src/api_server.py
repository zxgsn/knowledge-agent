"""REST API for browsing archival and recall memory.

Run with: uvicorn src.api_server:app --port 8000
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Knowledge Agent Library API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic models ---


class NamespaceStats(BaseModel):
    namespace: str
    count: int
    oldest: str | None
    newest: str | None


class ArchivalEntry(BaseModel):
    id: str
    namespace: str
    content: str
    metadata: dict
    created_at: str | None


class EntryListResponse(BaseModel):
    entries: list[ArchivalEntry]
    total: int


class RecallEntry(BaseModel):
    id: str
    thread_id: str
    role: str
    content: str
    metadata: dict
    created_at: str | None


class RecallListResponse(BaseModel):
    entries: list[RecallEntry]
    total: int


# --- DB helper ---


def _get_conn():
    import psycopg
    from pgvector.psycopg import register_vector

    from agent.storage import get_db_url

    conn = psycopg.connect(get_db_url(), connect_timeout=5)
    register_vector(conn)
    return conn


def _parse_meta(meta) -> dict:
    if isinstance(meta, dict):
        return meta
    try:
        return json.loads(meta)
    except (json.JSONDecodeError, TypeError):
        return {}


# --- Archival endpoints ---


@app.get("/api/archival/stats", response_model=list[NamespaceStats])
def get_archival_stats():
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT namespace, COUNT(*),
               MIN(created_at)::text, MAX(created_at)::text
        FROM archival_memory
        GROUP BY namespace
        ORDER BY COUNT(*) DESC
        """
    ).fetchall()
    conn.close()
    return [
        NamespaceStats(
            namespace=r[0], count=r[1],
            oldest=r[2], newest=r[3],
        )
        for r in rows
    ]


@app.get("/api/archival/entries", response_model=EntryListResponse)
def list_archival_entries(
    namespace: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search: str = Query(default=""),
):
    conn = _get_conn()
    params: list = []
    where_clauses: list[str] = []

    if namespace:
        where_clauses.append("namespace = %s")
        params.append(namespace)

    if search:
        where_clauses.append(
            "content_tsv @@ plainto_tsquery('simple', %s)"
        )
        params.append(search)

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # Total count
    count_params = list(params)
    total = conn.execute(
        f"SELECT COUNT(*) FROM archival_memory {where}", count_params
    ).fetchone()[0]

    # Fetch page
    params.extend([limit, offset])
    rows = conn.execute(
        f"""
        SELECT id, namespace, content, metadata, created_at::text
        FROM archival_memory {where}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """,
        params,
    ).fetchall()
    conn.close()

    entries = [
        ArchivalEntry(
            id=r[0], namespace=r[1], content=r[2],
            metadata=_parse_meta(r[3]), created_at=r[4],
        )
        for r in rows
    ]
    return EntryListResponse(entries=entries, total=total)


@app.get("/api/archival/entries/{entry_id}", response_model=ArchivalEntry)
def get_archival_entry(entry_id: str):
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, namespace, content, metadata, created_at::text "
        "FROM archival_memory WHERE id = %s",
        (entry_id,),
    ).fetchone()
    conn.close()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Entry not found")
    return ArchivalEntry(
        id=row[0], namespace=row[1], content=row[2],
        metadata=_parse_meta(row[3]), created_at=row[4],
    )


# --- Recall endpoints ---


@app.get("/api/recall/stats")
def get_recall_stats():
    conn = _get_conn()
    msg_count = conn.execute("SELECT COUNT(*) FROM recall_memory").fetchone()[0]
    thread_count = conn.execute(
        "SELECT COUNT(DISTINCT thread_id) FROM recall_memory"
    ).fetchone()[0]
    conn.close()
    return {"message_count": msg_count, "thread_count": thread_count}


@app.get("/api/recall/entries", response_model=RecallListResponse)
def list_recall_entries(
    thread_id: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search: str = Query(default=""),
):
    conn = _get_conn()
    params: list = []
    where_clauses: list[str] = []

    if thread_id:
        where_clauses.append("thread_id = %s")
        params.append(thread_id)

    if search:
        where_clauses.append(
            "to_tsvector('simple', content) @@ plainto_tsquery('simple', %s)"
        )
        params.append(search)

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    count_params = list(params)
    total = conn.execute(
        f"SELECT COUNT(*) FROM recall_memory {where}", count_params
    ).fetchone()[0]

    params.extend([limit, offset])
    rows = conn.execute(
        f"""
        SELECT id, thread_id, role, content, metadata, created_at::text
        FROM recall_memory {where}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """,
        params,
    ).fetchall()
    conn.close()

    entries = [
        RecallEntry(
            id=r[0], thread_id=r[1], role=r[2], content=r[3],
            metadata=_parse_meta(r[4]), created_at=r[5],
        )
        for r in rows
    ]
    return RecallListResponse(entries=entries, total=total)
