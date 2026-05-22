"""Memory editing tools for the agent.

These tools allow the agent to read, write, and edit its own memory blocks
(core memory) and interact with archival memory (long-term storage).

The tools are designed to be registered with LangGraph's tool system.
They operate on the agent's CoreMemory instance passed via a closure or context.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from langchain_core.tools import tool

if TYPE_CHECKING:
    from agent.memory.core_memory import CoreMemory


def _sync_search_archival(query: str, limit: int = 5, alpha: float = 0.7) -> list[dict]:
    """Hybrid search: vector similarity + BM25 keyword match, with optional cross-encoder re-ranking."""
    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector
    from agent.storage import get_db_url, get_embeddings

    try:
        embeddings = get_embeddings()
        query_embedding = Vector(embeddings.embed_query(query))
    except Exception:
        return []

    try:
        conn = psycopg.connect(get_db_url(), connect_timeout=5)
    except Exception:
        return []

    register_vector(conn)
    candidate_limit = int(limit) * 3
    rows = conn.execute(
        """
        SELECT content, metadata,
               %s * (1 - (embedding <=> %s::vector))
                 + (1 - %s) * LEAST(1, ts_rank(content_tsv, plainto_tsquery('english', %s)) * 5)
               AS score
        FROM archival_memory
        WHERE 1 - (embedding <=> %s::vector) > 0.15
        ORDER BY score DESC
        LIMIT %s
        """,
        (alpha, query_embedding, alpha, query, query_embedding, candidate_limit),
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        score = float(row[2])
        meta = row[1] if isinstance(row[1], dict) else json.loads(row[1])
        results.append({"content": row[0], "metadata": meta, "score": score})

    # Cross-encoder re-ranking
    try:
        from agent.storage.reranker import rerank
        results = rerank(query, results, top_k=int(limit))
    except Exception:
        results = results[:int(limit)]

    return results


def _sync_insert_document(title: str, source: str, source_type: str, content_full: str, chunk_count: int) -> tuple[str, bool]:
    """Insert a document record. Returns (doc_id, is_new)."""
    import psycopg
    from agent.storage import get_db_url

    conn = psycopg.connect(get_db_url(), connect_timeout=5)
    doc_id = str(uuid.uuid4())
    result = conn.execute(
        "INSERT INTO documents (id, title, source, source_type, content_full, chunk_count) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (title, source) DO NOTHING "
        "RETURNING id",
        (doc_id, title, source, source_type, content_full, chunk_count),
    )
    row = result.fetchone()
    if row:
        conn.commit()
        conn.close()
        return row[0], True
    existing = conn.execute(
        "SELECT id FROM documents WHERE title = %s AND source = %s",
        (title, source),
    ).fetchone()
    conn.close()
    return existing[0], False


def _sync_put_to_archival(content: str, namespace: str, metadata: dict, document_id: str | None = None) -> str:
    """Store a single entry in archival memory."""
    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector
    from agent.storage import get_db_url, get_embeddings

    embeddings = get_embeddings()
    entry_id = str(uuid.uuid4())
    metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
    embedding = Vector(embeddings.embed_query(content))

    conn = psycopg.connect(get_db_url(), connect_timeout=5)
    register_vector(conn)
    conn.execute(
        "INSERT INTO archival_memory (id, namespace, content, metadata, embedding, document_id) "
        "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
        (entry_id, namespace, content, json.dumps(metadata), embedding, document_id),
    )
    conn.commit()
    conn.close()
    return entry_id


def create_memory_tools(
    core_memory: CoreMemory,
    enable_archival: bool = False,
):
    """Factory that creates memory tools bound to specific memory instances.

    Returns a list of LangChain tools that can be passed to `bind_tools()`.

    Args:
        core_memory: The CoreMemory instance for reading/editing memory blocks.
        enable_archival: If True, include archival memory search/save tools.
    """

    # --- Core Memory Tools ---

    @tool
    def core_memory_replace(label: str, old_string: str, new_string: str) -> str:
        """Replace a substring in a core memory block. The old_string must appear exactly once.

        Args:
            label: The memory block label (e.g. 'persona', 'human', 'knowledge_focus').
            old_string: The exact substring to find and replace. Must be unique in the block.
            new_string: The replacement string.
        """
        try:
            block = core_memory.get_block(label)
        except KeyError:
            return f"Error: No block with label '{label}'. Available: {core_memory.list_labels()}"

        if block.read_only:
            return f"Error: Block '{label}' is read-only."

        current = block.value
        count = current.count(old_string)
        if count == 0:
            return f"Error: old_string not found in block '{label}'."
        if count > 1:
            return (
                f"Error: old_string appears {count} times in block '{label}'. "
                "It must be unique. Provide more context to make it unique."
            )

        new_value = current.replace(old_string, new_string)
        try:
            core_memory.update_block_value(label, new_value)
        except ValueError as e:
            return f"Error: {e}"

        return f"Block '{label}' updated successfully. New value:\n{new_value}"

    @tool
    def core_memory_insert(label: str, new_string: str, insert_line: int = -1) -> str:
        """Insert text at a specific line in a core memory block.

        Args:
            label: The memory block label.
            new_string: The text to insert.
            insert_line: Line number to insert at (0-indexed). -1 means append to end.
        """
        try:
            block = core_memory.get_block(label)
        except KeyError:
            return f"Error: No block with label '{label}'. Available: {core_memory.list_labels()}"

        if block.read_only:
            return f"Error: Block '{label}' is read-only."

        lines = block.value.split("\n")
        if insert_line == -1:
            insert_line = len(lines)
        if insert_line < 0 or insert_line > len(lines):
            return (
                f"Error: insert_line {insert_line} out of range. "
                f"Block has {len(lines)} lines (0-{len(lines)})."
            )

        new_lines = new_string.split("\n")
        lines[insert_line:insert_line] = new_lines
        new_value = "\n".join(lines)

        try:
            core_memory.update_block_value(label, new_value)
        except ValueError as e:
            return f"Error: {e}"

        return f"Block '{label}' updated. Inserted at line {insert_line}."

    @tool
    def core_memory_rethink(label: str, new_memory: str) -> str:
        """Completely rewrite a core memory block. Use for major reorganizations, not small edits.

        Args:
            label: The memory block label. If it doesn't exist, a new block is created.
            new_memory: The new full content for this block.
        """
        try:
            block = core_memory.get_block(label)
            if block.read_only:
                return f"Error: Block '{label}' is read-only."
        except KeyError:
            from agent.memory.block import Block
            block = Block(label=label)
            core_memory.set_block(block)

        try:
            core_memory.update_block_value(label, new_memory)
        except ValueError as e:
            return f"Error: {e}"

        return f"Block '{label}' completely rewritten."

    @tool
    def core_memory_view(label: str = "") -> str:
        """View the content of core memory blocks.

        Args:
            label: The block label to view. If empty, shows all blocks.
        """
        if label:
            try:
                block = core_memory.get_block(label)
                return (
                    f"<{block.label}>\n"
                    f"description: {block.description}\n"
                    f"chars: {block.chars_current}/{block.limit}\n"
                    f"read_only: {block.read_only}\n"
                    f"value:\n{block.value}\n"
                    f"</{block.label}>"
                )
            except KeyError:
                return f"Error: No block with label '{label}'. Available: {core_memory.list_labels()}"
        else:
            return core_memory.compile()

    tools = [core_memory_replace, core_memory_insert, core_memory_rethink, core_memory_view]

    # --- Archival Memory Tools ---

    if enable_archival:

        @tool
        def archival_memory_search(query: str, limit: int = 5) -> str:
            """Search long-term archival memory for relevant knowledge.

            Use this to find previously stored research findings, document excerpts,
            or any knowledge that was ingested into the knowledge base.

            Args:
                query: The search query.
                limit: Maximum number of results to return (default 5).
            """
            results = _sync_search_archival(query=query, limit=limit)
            if not results:
                return "No relevant results found in archival memory."

            parts = []
            for r in results:
                score = r.get("score", 0)
                content = r["content"]
                source = r.get("metadata", {}).get("source", "unknown")
                if score > 0.3:
                    parts.append(
                        f"[score={score:.2f}] (source: {source})\n{content}"
                    )
            return "\n\n---\n\n".join(parts) if parts else "No high-relevance results found."

        @tool
        def archival_memory_save(content: str, source: str = "manual") -> str:
            """Save a piece of text to long-term archival memory.

            Use this when the user asks you to remember something, or when you
            discover important information worth keeping for future reference.

            Args:
                content: The text to save. Should be self-contained and meaningful on its own.
                source: A label describing where this knowledge came from.
            """
            entry_id = _sync_put_to_archival(
                content=content,
                namespace="manual",
                metadata={"source": source, "type": "manual_save"},
            )
            return f"Saved to archival memory (id: {entry_id})."

        @tool
        def ingest_document(
            source: str,
            source_type: Literal["url", "text"] = "url",
            chunk_size: int = 800,
        ) -> str:
            """Ingest a document into the knowledge base.

            Supports URLs (web pages) and plain text. The document is automatically
            split into chunks, embedded, and stored in archival memory for future retrieval.

            Args:
                source: The URL to fetch, or the plain text content to ingest.
                source_type: "url" to fetch from a web URL, "text" for plain text input.
                chunk_size: Target chunk size in characters (default 800).
            """
            import httpx
            from agent.storage.ingestion import chunk_text

            if source_type == "url":
                try:
                    resp = httpx.get(source, follow_redirects=True, timeout=30)
                    resp.raise_for_status()
                except Exception as e:
                    return f"Failed to fetch URL: {e}"

                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                doc_label = soup.title.string if soup.title else source
            else:
                text = source
                doc_label = "manual_text"

            if not text or len(text.strip()) < 50:
                return "No meaningful text content to ingest."

            chunks = chunk_text(text, chunk_size=chunk_size, source=doc_label)
            if not chunks:
                return "Failed to chunk document."

            # Write full document to documents table first
            full_text = "\n\n".join(c.content for c in chunks)
            document_id, is_new = _sync_insert_document(
                doc_label,
                source if source_type == "url" else doc_label,
                source_type,
                full_text,
                len(chunks),
            )

            if not is_new:
                return f"Document '{doc_label}' already exists in the library. Skipping."

            ids = []
            for c in chunks:
                eid = _sync_put_to_archival(
                    content=c.content,
                    namespace="ingested",
                    metadata={
                        "source": c.source,
                        "source_type": source_type,
                        "chunk_index": c.index,
                        "document": doc_label,
                        "document_id": document_id,
                    },
                    document_id=document_id,
                )
                ids.append(eid)

            return (
                f"Ingested '{doc_label}': {len(ids)} chunks stored in archival memory. "
                f"Source: {source if source_type == 'url' else 'plain text'}"
            )

        tools.extend([archival_memory_search, archival_memory_save, ingest_document])

    return tools
