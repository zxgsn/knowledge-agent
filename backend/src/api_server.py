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


class VersionEntry(BaseModel):
    version_id: str
    content: str
    metadata: dict
    version_number: int
    change_type: str
    changed_by: str
    created_at: str | None


class ConflictReview(BaseModel):
    id: str
    new_fact: str
    existing_id: str
    existing_content: str
    similarity_score: float
    llm_decision: str | None
    llm_merged_text: str | None
    llm_confidence: float | None
    created_at: str | None


class ResolveConflictRequest(BaseModel):
    action: str  # "approve" | "reject" | "modify"
    resolution_text: str | None = None


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
            WHERE status = 'active'
              AND namespace NOT LIKE %s
            GROUP BY namespace
            ORDER BY COUNT(*) DESC
            """,
            ("locomo%",),
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
    where_clauses: list[str] = ["status = 'active'"]

    if namespace:
        where_clauses.append("namespace = %s")
        params.append(namespace)
    else:
        # Always exclude locomo benchmark namespaces
        where_clauses.append("namespace NOT LIKE %s")
        params.append("locomo%")

        if exclude_namespaces:
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
            "FROM archival_memory WHERE document_id = %s AND status = 'active' "
            "ORDER BY (metadata->>'chunk_index')::int",
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



# --- Core Memory Version History endpoints ---


class CoreMemoryVersionEntry(BaseModel):
    version_id: str
    block_value: str
    version_number: int
    change_type: str
    changed_by: str
    created_at: str | None


@app.get("/api/core-memory/{thread_id}/{block_label}/versions", response_model=list[CoreMemoryVersionEntry])
def get_core_memory_versions(thread_id: str, block_label: str):
    from agent.db import get_core_memory_version_history
    return [CoreMemoryVersionEntry(**v) for v in get_core_memory_version_history(thread_id, block_label)]


@app.post("/api/core-memory/{thread_id}/{block_label}/rollback")
def rollback_core_memory(thread_id: str, block_label: str, version_number: int = Query(...)):
    from fastapi import HTTPException

    from agent.db import rollback_core_memory_block

    restored_value = rollback_core_memory_block(thread_id, block_label, version_number)
    if restored_value is None:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"rolled_back": True, "thread_id": thread_id, "block_label": block_label,
            "to_version": version_number, "value": restored_value}


# --- Version History & Rollback endpoints ---


@app.get("/api/archival/entries/{entry_id}/versions", response_model=list[VersionEntry])
def get_entry_versions(entry_id: str):
    from agent.db import get_version_history

    return [VersionEntry(**v) for v in get_version_history(entry_id)]


@app.post("/api/archival/entries/{entry_id}/rollback")
def rollback_entry(entry_id: str, version_number: int = Query(...)):
    from fastapi import HTTPException

    from agent.db import rollback_to_version

    success = rollback_to_version(entry_id, version_number)
    if not success:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"rolled_back": True, "entry_id": entry_id, "to_version": version_number}


@app.delete("/api/archival/versions/{version_id}")
def delete_version_entry(version_id: str):
    from fastapi import HTTPException

    from agent.db import delete_version

    success = delete_version(version_id)
    if not success:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"deleted": True, "version_id": version_id}


@app.get("/api/archival/entries/{entry_id}/diff")
def get_version_diff(entry_id: str, v1: int = Query(...), v2: int = Query(...)):
    """Get a diff between two versions of an archival entry."""
    from fastapi import HTTPException

    from agent.db import get_version_history

    versions = get_version_history(entry_id)
    ver1 = next((v for v in versions if v["version_number"] == v1), None)
    ver2 = next((v for v in versions if v["version_number"] == v2), None)

    if not ver1 or not ver2:
        raise HTTPException(status_code=404, detail="Version not found")

    # Simple word-level diff
    words1 = ver1["content"].split()
    words2 = ver2["content"].split()

    # Find common prefix and suffix
    prefix_len = 0
    while prefix_len < min(len(words1), len(words2)) and words1[prefix_len] == words2[prefix_len]:
        prefix_len += 1

    suffix_len = 0
    while (suffix_len < min(len(words1), len(words2)) - prefix_len
           and words1[len(words1) - 1 - suffix_len] == words2[len(words2) - 1 - suffix_len]):
        suffix_len += 1

    removed = words1[prefix_len:len(words1) - suffix_len] if suffix_len > 0 else words1[prefix_len:]
    added = words2[prefix_len:len(words2) - suffix_len] if suffix_len > 0 else words2[prefix_len:]

    return {
        "entry_id": entry_id,
        "v1": v1,
        "v2": v2,
        "v1_content": ver1["content"],
        "v2_content": ver2["content"],
        "removed": " ".join(removed),
        "added": " ".join(added),
        "v1_changed_by": ver1["changed_by"],
        "v2_changed_by": ver2["changed_by"],
    }


# --- Conflict Review endpoints ---


@app.get("/api/conflicts", response_model=list[ConflictReview])
def list_pending_conflicts(limit: int = Query(default=20, ge=1, le=100)):
    from agent.db import get_pending_conflicts

    return [ConflictReview(**c) for c in get_pending_conflicts(limit)]


@app.post("/api/conflicts/{review_id}/resolve")
def resolve_conflict_endpoint(review_id: str, req: ResolveConflictRequest):
    from fastapi import HTTPException

    from agent.db import resolve_conflict

    success = resolve_conflict(review_id, req.action, req.resolution_text)
    if not success:
        raise HTTPException(
            status_code=404, detail="Conflict review not found or already resolved"
        )
    return {"resolved": True, "review_id": review_id, "action": req.action}


class BulkResolveRequest(BaseModel):
    review_ids: list[str]
    action: str  # "approve" | "reject"


@app.post("/api/conflicts/bulk-resolve")
def bulk_resolve_conflicts(req: BulkResolveRequest):
    """Resolve multiple conflict reviews at once."""
    from agent.db import resolve_conflict

    resolved = []
    failed = []
    for review_id in req.review_ids:
        success = resolve_conflict(review_id, req.action)
        if success:
            resolved.append(review_id)
        else:
            failed.append(review_id)

    return {"resolved": resolved, "failed": failed}


# --- Delete endpoints ---


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and soft-delete all its associated archival chunks."""
    from fastapi import HTTPException

    from agent.db import delete_archival
    from agent.storage import get_conn

    with get_conn() as conn:
        # Find associated chunk IDs
        chunk_rows = conn.execute(
            "SELECT id FROM archival_memory WHERE document_id = %s AND status = 'active'",
            (doc_id,),
        ).fetchall()
        # Delete the document itself
        doc_result = conn.execute(
            "DELETE FROM documents WHERE id = %s", (doc_id,)
        )
        conn.commit()

    if doc_result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    # Soft-delete each chunk
    for (chunk_id,) in chunk_rows:
        delete_archival(chunk_id, changed_by="document_delete")

    return {
        "deleted": True,
        "document_id": doc_id,
        "chunks_deleted": len(chunk_rows),
    }


@app.delete("/api/archival/entries/{entry_id}")
def delete_archival_entry(entry_id: str):
    """Soft-delete a single archival memory entry."""
    from fastapi import HTTPException

    from agent.db import delete_archival

    success = delete_archival(entry_id, changed_by="manual_edit")
    if not success:
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
    update_archival(entry_id, req.content, meta, changed_by="manual_edit")

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
    """Delete a single recall memory entry (with version snapshot)."""
    from fastapi import HTTPException

    from agent.db import snapshot_recall_version
    from agent.storage import get_conn

    # Snapshot before delete for version history
    snapshot_recall_version(entry_id, change_type="delete", changed_by="manual_edit")

    with get_conn() as conn:
        result = conn.execute(
            "DELETE FROM recall_memory WHERE id = %s", (entry_id,)
        )
        conn.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Entry not found")

    return {"deleted": True, "entry_id": entry_id}


# --- Recall Version History endpoints ---


class RecallVersionEntry(BaseModel):
    version_id: str
    content: str
    metadata: dict
    version_number: int
    change_type: str
    changed_by: str
    created_at: str | None
    thread_id: str | None = None
    role: str | None = None


@app.get("/api/recall/entries/{entry_id}/versions", response_model=list[RecallVersionEntry])
def get_recall_entry_versions(entry_id: str):
    from agent.db import get_recall_version_history
    return [RecallVersionEntry(**v) for v in get_recall_version_history(entry_id)]


@app.post("/api/recall/entries/{entry_id}/rollback")
def rollback_recall_entry(entry_id: str, version_number: int = Query(...)):
    from fastapi import HTTPException

    from agent.db import rollback_recall_version

    success = rollback_recall_version(entry_id, version_number)
    if not success:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"rolled_back": True, "entry_id": entry_id, "to_version": version_number}


@app.delete("/api/recall/versions/{version_id}")
def delete_recall_version_entry(version_id: str):
    from fastapi import HTTPException

    from agent.db import delete_recall_version

    success = delete_recall_version(version_id)
    if not success:
        raise HTTPException(status_code=404, detail="Version not found")
    return {"deleted": True, "version_id": version_id}

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


# --- Analytics endpoints ---


@app.get("/api/analytics/stats")
def get_analytics_stats():
    """Get memory analytics statistics."""
    from agent.storage import get_conn

    stats = {}

    with get_conn() as conn:
        # Archival memory stats
        row = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT namespace) FROM archival_memory WHERE status = 'active'"
        ).fetchone()
        stats["archival_count"] = row[0]
        stats["namespace_count"] = row[1]

        # Namespace breakdown
        ns_rows = conn.execute(
            "SELECT namespace, COUNT(*) FROM archival_memory "
            "WHERE status = 'active' GROUP BY namespace ORDER BY COUNT(*) DESC"
        ).fetchall()
        stats["namespaces"] = [{"namespace": r[0], "count": r[1]} for r in ns_rows]

        # Recall memory stats
        row = conn.execute("SELECT COUNT(*), COUNT(DISTINCT thread_id) FROM recall_memory").fetchone()
        stats["recall_count"] = row[0]
        stats["thread_count"] = row[1]

        # Document stats
        row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        stats["document_count"] = row[0]

        # Conflict stats
        row = conn.execute("SELECT COUNT(*) FROM conflict_reviews WHERE status = 'pending'").fetchone()
        stats["pending_conflicts"] = row[0]

        # Version history stats
        row = conn.execute("SELECT COUNT(*) FROM archival_versions").fetchone()
        stats["version_count"] = row[0]

        # Age distribution (last 7 days, 30 days, older)
        row = conn.execute(
            "SELECT "
            "COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days'), "
            "COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days' AND created_at <= NOW() - INTERVAL '7 days'), "
            "COUNT(*) FILTER (WHERE created_at <= NOW() - INTERVAL '30 days') "
            "FROM archival_memory WHERE status = 'active'"
        ).fetchone()
        stats["age_distribution"] = {
            "last_7_days": row[0],
            "last_30_days": row[1],
            "older": row[2],
        }

    return stats


# --- Pydantic models for new endpoints ---


class ImportanceDistributionItem(BaseModel):
    range: str
    count: int


class ImportanceDistributionResponse(BaseModel):
    distribution: list[ImportanceDistributionItem]
    total_scored: int


class EntityInfo(BaseModel):
    entity: str
    memory_count: int | None = None
    count: int | None = None
    edge_count: int | None = None


class TopEntitiesResponse(BaseModel):
    entities: list[EntityInfo]


class KnowledgeGraphNode(BaseModel):
    id: str
    memory_count: int
    hop: int | None = None


class KnowledgeGraphEdge(BaseModel):
    source: str
    target: str
    weight: int


class KnowledgeGraphResponse(BaseModel):
    center: str
    nodes: list[KnowledgeGraphNode]
    edges: list[KnowledgeGraphEdge]


class ImportanceSearchResult(BaseModel):
    id: str
    content: str
    metadata: dict
    namespace: str
    score: float
    cosine_score: float
    importance_score: float
    created_at: str | None = None


class MemorySearchResponse(BaseModel):
    results: list[ImportanceSearchResult]
    total: int
    query: str
    namespace: str


class SetImportanceRequest(BaseModel):
    score: float


# --- Importance & Knowledge Graph endpoints ---


@app.get("/api/analytics/importance-distribution", response_model=ImportanceDistributionResponse)
def get_importance_distribution():
    """Get histogram of importance scores across all active memories."""
    from agent.db.importance import get_memory_stats_extended

    extended = get_memory_stats_extended()
    distribution = extended.get("importance_distribution", [])
    total = sum(item["count"] for item in distribution)
    return ImportanceDistributionResponse(
        distribution=[ImportanceDistributionItem(**d) for d in distribution],
        total_scored=total,
    )


@app.get("/api/analytics/knowledge-graph", response_model=KnowledgeGraphResponse)
def get_knowledge_graph(center: str = Query(default=""), radius: int = Query(default=2, ge=1, le=5)):
    """Get a subgraph around a center entity for visualization."""
    from fastapi import HTTPException

    from agent.memory.knowledge_graph import get_knowledge_graph, rebuild_graph_from_db

    graph = get_knowledge_graph()
    # If graph is empty, try to rebuild
    if graph.stats["entity_count"] == 0:
        graph = rebuild_graph_from_db()

    if not center:
        # Return top entity as center if none specified
        top = graph.get_top_entities(limit=1)
        if not top:
            raise HTTPException(status_code=404, detail="No entities found in knowledge graph")
        center = top[0]["entity"]

    subgraph = graph.get_subgraph(center, radius)
    return KnowledgeGraphResponse(
        center=center,
        nodes=[KnowledgeGraphNode(**n) for n in subgraph["nodes"]],
        edges=[KnowledgeGraphEdge(**e) for e in subgraph["edges"]],
    )


@app.get("/api/analytics/top-entities", response_model=TopEntitiesResponse)
def get_top_entities(limit: int = Query(default=20, ge=1, le=100)):
    """Get most frequently mentioned entities across all memories.

    Combines entities from both the knowledge graph (content-extracted)
    and structured metadata (entities field).
    """
    from agent.memory.knowledge_graph import get_knowledge_graph, rebuild_graph_from_db

    graph = get_knowledge_graph()
    if graph.stats["entity_count"] == 0:
        graph = rebuild_graph_from_db()

    graph_entities = graph.get_top_entities(limit)
    return TopEntitiesResponse(
        entities=[EntityInfo(**e) for e in graph_entities]
    )


@app.get("/api/memory/search", response_model=MemorySearchResponse)
def advanced_memory_search(
    query: str = Query(...),
    namespace: str = Query(default=""),
    limit: int = Query(default=10, ge=1, le=50),
    importance_weight: float = Query(default=0.3, ge=0.0, le=1.0),
):
    """Advanced memory search with importance-weighted results.

    Combines vector similarity with importance scoring for better ranking.
    """
    from agent.db.importance import search_importance_weighted

    results = search_importance_weighted(
        query=query,
        namespace=namespace,
        limit=limit,
        importance_weight=importance_weight,
    )
    return MemorySearchResponse(
        results=[ImportanceSearchResult(**r) for r in results],
        total=len(results),
        query=query,
        namespace=namespace,
    )


@app.post("/api/memory/{entry_id}/importance")
def set_importance_score(entry_id: str, req: SetImportanceRequest):
    """Manually set importance score for a memory entry."""
    from fastapi import HTTPException

    from agent.db.importance import update_importance_score

    score = max(0.0, min(1.0, req.score))
    update_importance_score(entry_id, score)
    return {"entry_id": entry_id, "importance_score": score}
