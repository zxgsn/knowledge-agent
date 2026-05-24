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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:2024", "http://127.0.0.1:2024"],
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


class DocumentSummary(BaseModel):
    id: str
    title: str
    source: str | None
    source_type: str | None
    chunk_count: int
    created_at: str | None


class DocumentDetail(DocumentSummary):
    content_full: str
    metadata: dict
    chunks: list[ArchivalEntry]


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int


class DocumentStats(BaseModel):
    total_documents: int
    by_source_type: dict[str, int]


class CoreMemoryBlock(BaseModel):
    label: str
    value: str
    description: str
    limit: int
    chars_current: int
    read_only: bool


class CoreMemoryResponse(BaseModel):
    blocks: list[CoreMemoryBlock]
    raw: dict[str, str]


class CreateEntryRequest(BaseModel):
    content: str
    namespace: str = "manual"
    metadata: dict = {}


class UpdateEntryRequest(BaseModel):
    content: str
    metadata: dict | None = None


# --- DB helper ---


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
    from agent.storage import get_conn

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT namespace, COUNT(*),
                   MIN(created_at)::text, MAX(created_at)::text
            FROM archival_memory
            GROUP BY namespace
            ORDER BY COUNT(*) DESC
            """
        ).fetchall()
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
    exclude_namespaces: str = Query(default=""),
):
    from agent.storage import get_conn

    params: list = []
    where_clauses: list[str] = []

    if namespace:
        where_clauses.append("namespace = %s")
        params.append(namespace)
    elif exclude_namespaces:
        ns_list = [ns.strip() for ns in exclude_namespaces.split(",") if ns.strip()]
        if ns_list:
            placeholders = ", ".join(["%s"] * len(ns_list))
            where_clauses.append(f"namespace NOT IN ({placeholders})")
            params.extend(ns_list)

    if search:
        where_clauses.append(
            "content_tsv @@ plainto_tsquery('english', %s)"
        )
        params.append(search)

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    with get_conn() as conn:
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
    from fastapi import HTTPException

    from agent.storage import get_conn

    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, namespace, content, metadata, created_at::text "
            "FROM archival_memory WHERE id = %s",
            (entry_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Entry not found")
    return ArchivalEntry(
        id=row[0], namespace=row[1], content=row[2],
        metadata=_parse_meta(row[3]), created_at=row[4],
    )


# --- Document endpoints ---


@app.get("/api/documents/stats", response_model=DocumentStats)
def get_document_stats():
    from agent.storage import get_conn

    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        rows = conn.execute(
            "SELECT COALESCE(source_type, 'unknown'), COUNT(*) "
            "FROM documents GROUP BY source_type ORDER BY COUNT(*) DESC"
        ).fetchall()
    return DocumentStats(
        total_documents=total,
        by_source_type={r[0]: r[1] for r in rows},
    )


@app.get("/api/documents", response_model=DocumentListResponse)
def list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search: str = Query(default=""),
):
    from agent.storage import get_conn

    params: list = []
    where_clauses: list[str] = []

    if search:
        where_clauses.append(
            "(title ILIKE %s OR source ILIKE %s)"
        )
        params.extend([f"%{search}%", f"%{search}%"])

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    with get_conn() as conn:
        count_params = list(params)
        total = conn.execute(
            f"SELECT COUNT(*) FROM documents {where}", count_params
        ).fetchone()[0]

        params.extend([limit, offset])
        rows = conn.execute(
            f"""
            SELECT id, title, source, source_type, chunk_count, created_at::text
            FROM documents {where}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            params,
        ).fetchall()

    documents = [
        DocumentSummary(
            id=r[0], title=r[1], source=r[2],
            source_type=r[3], chunk_count=r[4], created_at=r[5],
        )
        for r in rows
    ]
    return DocumentListResponse(documents=documents, total=total)


@app.get("/api/documents/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str):
    from fastapi import HTTPException

    from agent.storage import get_conn

    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, title, source, source_type, content_full, chunk_count, metadata, created_at::text "
            "FROM documents WHERE id = %s",
            (doc_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        # Fetch associated chunks
        chunk_rows = conn.execute(
            "SELECT id, namespace, content, metadata, created_at::text "
            "FROM archival_memory WHERE document_id = %s ORDER BY (metadata->>'chunk_index')::int",
            (doc_id,),
        ).fetchall()

    chunks = [
        ArchivalEntry(
            id=c[0], namespace=c[1], content=c[2],
            metadata=_parse_meta(c[3]), created_at=c[4],
        )
        for c in chunk_rows
    ]
    return DocumentDetail(
        id=row[0], title=row[1], source=row[2], source_type=row[3],
        content_full=row[4], chunk_count=row[5],
        metadata=_parse_meta(row[6]), created_at=row[7],
        chunks=chunks,
    )


# --- Recall endpoints ---


@app.get("/api/recall/stats")
def get_recall_stats():
    from agent.storage import get_conn

    with get_conn() as conn:
        msg_count = conn.execute("SELECT COUNT(*) FROM recall_memory").fetchone()[0]
        thread_count = conn.execute(
            "SELECT COUNT(DISTINCT thread_id) FROM recall_memory"
        ).fetchone()[0]
    return {"message_count": msg_count, "thread_count": thread_count}


@app.get("/api/recall/entries", response_model=RecallListResponse)
def list_recall_entries(
    thread_id: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search: str = Query(default=""),
):
    from agent.storage import get_conn

    params: list = []
    where_clauses: list[str] = []

    if thread_id:
        where_clauses.append("thread_id = %s")
        params.append(thread_id)

    if search:
        where_clauses.append(
            "to_tsvector('english', content) @@ plainto_tsquery('english', %s)"
        )
        params.append(search)

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    with get_conn() as conn:
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

    entries = [
        RecallEntry(
            id=r[0], thread_id=r[1], role=r[2], content=r[3],
            metadata=_parse_meta(r[4]), created_at=r[5],
        )
        for r in rows
    ]
    return RecallListResponse(entries=entries, total=total)


# --- Core Memory endpoints ---


@app.get("/api/core-memory/{thread_id}", response_model=CoreMemoryResponse)
def get_core_memory(thread_id: str):
    """Fetch core memory blocks for a thread from the LangGraph server."""
    import httpx
    from fastapi import HTTPException

    from agent.memory.block import DEFAULT_BLOCKS
    from agent.memory.core_memory import CoreMemory

    langgraph_url = os.environ.get("LANGGRAPH_API_URL", "http://localhost:2024")

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{langgraph_url}/threads/{thread_id}/state")
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail="Thread state not found")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LangGraph server error: {e}")

    state = resp.json().get("values", {})
    raw = state.get("core_memory", {})

    # Reconstruct CoreMemory to get full block metadata
    cm = CoreMemory.from_dict(raw)

    blocks = [
        CoreMemoryBlock(
            label=b.label,
            value=b.value,
            description=b.description,
            limit=b.limit,
            chars_current=b.chars_current,
            read_only=b.read_only,
        )
        for b in cm.blocks
    ]
    return CoreMemoryResponse(blocks=blocks, raw=raw)


# --- Delete endpoints ---


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and all its associated archival chunks."""
    from fastapi import HTTPException

    from agent.storage import get_conn

    with get_conn() as conn:
        # Delete associated chunks first
        chunk_result = conn.execute(
            "DELETE FROM archival_memory WHERE document_id = %s", (doc_id,)
        )
        # Delete the document itself
        doc_result = conn.execute(
            "DELETE FROM documents WHERE id = %s", (doc_id,)
        )
        conn.commit()

    if doc_result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "deleted": True,
        "document_id": doc_id,
        "chunks_deleted": chunk_result.rowcount,
    }


@app.delete("/api/archival/entries/{entry_id}")
def delete_archival_entry(entry_id: str):
    """Delete a single archival memory entry."""
    from fastapi import HTTPException

    from agent.storage import get_conn

    with get_conn() as conn:
        result = conn.execute(
            "DELETE FROM archival_memory WHERE id = %s", (entry_id,)
        )
        conn.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Entry not found")

    return {"deleted": True, "entry_id": entry_id}


@app.post("/api/archival/entries", response_model=ArchivalEntry)
def create_archival_entry(req: CreateEntryRequest):
    """Create a new archival memory entry."""
    from agent.db import put_to_archival
    from agent.storage import get_conn

    entry_id = put_to_archival(
        content=req.content,
        namespace=req.namespace,
        metadata=req.metadata,
    )

    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, namespace, content, metadata, created_at::text "
            "FROM archival_memory WHERE id = %s",
            (entry_id,),
        ).fetchone()

    return ArchivalEntry(
        id=row[0], namespace=row[1], content=row[2],
        metadata=_parse_meta(row[3]), created_at=row[4],
    )


@app.put("/api/archival/entries/{entry_id}", response_model=ArchivalEntry)
def update_archival_entry(entry_id: str, req: UpdateEntryRequest):
    """Update an existing archival memory entry."""
    from fastapi import HTTPException

    from agent.db import update_archival
    from agent.storage import get_conn

    # Check entry exists
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT metadata FROM archival_memory WHERE id = %s",
            (entry_id,),
        ).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Entry not found")

    # Merge metadata: use provided or keep existing
    meta = req.metadata if req.metadata is not None else _parse_meta(existing[0])
    update_archival(entry_id, req.content, meta)

    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, namespace, content, metadata, created_at::text "
            "FROM archival_memory WHERE id = %s",
            (entry_id,),
        ).fetchone()

    return ArchivalEntry(
        id=row[0], namespace=row[1], content=row[2],
        metadata=_parse_meta(row[3]), created_at=row[4],
    )


@app.delete("/api/recall/entries/{entry_id}")
def delete_recall_entry(entry_id: str):
    """Delete a single recall memory entry."""
    from fastapi import HTTPException

    from agent.storage import get_conn

    with get_conn() as conn:
        result = conn.execute(
            "DELETE FROM recall_memory WHERE id = %s", (entry_id,)
        )
        conn.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Entry not found")

    return {"deleted": True, "entry_id": entry_id}


@app.delete("/api/recall/entries")
def delete_recall_by_thread(thread_id: str = Query(...)):
    """Delete all recall memory entries for a thread."""
    from agent.storage import get_conn

    with get_conn() as conn:
        result = conn.execute(
            "DELETE FROM recall_memory WHERE thread_id = %s", (thread_id,)
        )
        conn.commit()

    return {"deleted": True, "thread_id": thread_id, "count": result.rowcount}

    with get_conn() as conn:
        result = conn.execute(
            "DELETE FROM recall_memory WHERE id = %s", (entry_id,)
        )
        conn.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Entry not found")

    return {"deleted": True, "entry_id": entry_id}
