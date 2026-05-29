# REST API Reference

**Source**: `backend/src/api_server.py`

FastAPI application for browsing and managing the knowledge base. Runs on port 8000 by default.

## Base URL

```
http://localhost:8000
```

## CORS Origins

- `http://localhost:5173` (Vite dev server)
- `http://localhost:2024` (LangGraph Studio)

## Archival Memory Endpoints

### GET `/api/archival/stats`

List namespace statistics.

**Response**: `list[NamespaceStats]`
```json
[{"namespace": "conversation_facts", "count": 42, "oldest": "...", "newest": "..."}]
```

### GET `/api/archival/entries`

List archival entries with pagination and filtering.

**Query Parameters**:
- `namespace` (str, optional): Filter by namespace
- `limit` (int, 1-200, default=50): Page size
- `offset` (int, default=0): Pagination offset
- `search` (str, optional): Full-text search
- `exclude_namespaces` (str, optional): Comma-separated namespaces to exclude

**Response**: `EntryListResponse`
```json
{"entries": [...], "total": 42}
```

### GET `/api/archival/entries/{entry_id}`

Get a single archival entry by ID.

### POST `/api/archival/entries`

Create a new archival entry.

**Body**: `CreateEntryRequest`
```json
{"content": "...", "namespace": "manual", "metadata": {}}
```

### PUT `/api/archival/entries/{entry_id}`

Update an archival entry (creates a version snapshot).

### DELETE `/api/archival/entries/{entry_id}`

Soft-delete an archival entry (sets status to 'deleted').

### GET `/api/archival/entries/{entry_id}/versions`

List version history for an entry.

## Document Endpoints

### GET `/api/documents/stats`

Document statistics (total count, by source type).

### GET `/api/documents`

List documents with pagination and search.

### GET `/api/documents/{doc_id}`

Get document detail with chunks.

### DELETE `/api/documents/{doc_id}`

Delete document and all associated chunks.

## Recall Memory Endpoints

### GET `/api/recall/stats`

Recall memory statistics (total messages, by thread, by role).

### GET `/api/recall/entries`

List recall entries with pagination.

## Core Memory Endpoints

### GET `/api/core-memory`

Get current core memory blocks.

**Response**: `CoreMemoryResponse`
```json
{"blocks": [...], "raw": {"persona": "...", "human": "...", "knowledge_focus": "..."}}
```

## Conflict Review Endpoints

### GET `/api/conflicts`

List pending conflict reviews (low-confidence memory merges).

### POST `/api/conflicts/{conflict_id}/resolve`

Resolve a conflict (approve/reject/modify).

**Body**: `ResolveConflictRequest`
```json
{"action": "approve", "resolution_text": null}
```

## Health Check

### GET `/health`

Returns `{"status": "ok"}`.
