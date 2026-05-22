"""Memory-related nodes: intent routing, archival save, memory recall."""

from __future__ import annotations

import asyncio
import json
import re

import httpx

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agent.configuration import Configuration
from agent.prompts import ARCHIVAL_STORE_PROMPT, EVALUATE_RECALL_PROMPT, ROUTE_INTENT_PROMPT
from agent.state import AgentState


def _parse_json(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def route_intent(state: AgentState, config: RunnableConfig) -> dict:
    """Classify user intent: chat, research, or memory_edit."""
    configurable = Configuration.from_runnable_config(config)

    # If mode is already set (e.g. from frontend), keep it
    if state.get("mode") in ("research", "memory_edit", "ingest"):
        return {}

    llm = ChatOpenAI(
        model=configurable.llm_model,
        base_url=configurable.llm_base_url,
        api_key=configurable.llm_api_key,
        temperature=0,
        http_async_client=httpx.AsyncClient(proxy=None),
    )

    user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    if not user_msg:
        return {"mode": "chat"}

    prompt = ROUTE_INTENT_PROMPT.format(user_message=user_msg)
    response = await llm.ainvoke(prompt)
    parsed = _parse_json(response.content)
    mode = parsed.get("mode", "chat")
    if mode not in ("chat", "research", "memory_edit", "ingest", "recall"):
        mode = "chat"

    need_recall = parsed.get("need_recall", False)

    return {"mode": mode, "need_recall": need_recall}


def _sync_search_archival(query: str, limit: int = 5, alpha: float = 0.7) -> list[dict]:
    """Hybrid search: vector similarity + BM25 keyword match, with optional cross-encoder re-ranking.

    Args:
        query: Search query text.
        limit: Max results.
        alpha: Weight for vector score (1-alpha for BM25). Default 0.7.
    """
    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector

    from agent.storage import get_db_url, get_embeddings

    try:
        embeddings = get_embeddings()
        query_embedding = Vector(embeddings.embed_query(query))
    except Exception as e:
        import sys
        print(f"[memory_manager] Embedding failed: {e}", file=sys.stderr)
        return []

    try:
        conn = psycopg.connect(get_db_url(), connect_timeout=5)
    except Exception as e:
        import sys
        print(f"[memory_manager] DB connection failed: {e}", file=sys.stderr)
        return []

    register_vector(conn)
    # Fetch more candidates for re-ranking
    candidate_limit = int(limit) * 3
    rows = conn.execute(
        """
        SELECT content, metadata,
               %s * (1 - (embedding <=> %s::vector))
                 + (1 - %s) * ts_rank(content_tsv, plainto_tsquery('simple', %s))
               AS score
        FROM archival_memory
        WHERE content_tsv @@ plainto_tsquery('simple', %s)
           OR 1 - (embedding <=> %s::vector) > 0.2
        ORDER BY score DESC
        LIMIT %s
        """,
        (alpha, query_embedding, alpha, query, query, query_embedding, candidate_limit),
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        score = float(row[2])
        if score >= 0.01:
            meta = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            results.append({"content": row[0], "metadata": meta, "score": score})

    # Cross-encoder re-ranking
    try:
        from agent.storage.reranker import rerank
        results = rerank(query, results, top_k=int(limit))
    except Exception:
        results = results[:int(limit)]

    return results


def _sync_save_to_recall(role: str, content: str, thread_id: str = "default") -> None:
    """Save a message to the recall_memory table."""
    import uuid

    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector

    from agent.storage import ensure_recall_table, get_db_url, get_embeddings

    ensure_recall_table()
    embeddings = get_embeddings()
    entry_id = str(uuid.uuid4())
    embedding = Vector(embeddings.embed_query(content))

    conn = psycopg.connect(get_db_url(), connect_timeout=5)
    register_vector(conn)
    conn.execute(
        "INSERT INTO recall_memory (id, thread_id, role, content, embedding) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
        (entry_id, thread_id, role, content, embedding),
    )
    conn.commit()
    conn.close()


def _sync_search_recall(query: str, limit: int = 5) -> list[dict]:
    """Semantic search over recall_memory (conversation history).

    Searches across all threads for relevant past conversations.
    Uses pure cosine similarity (no BM25 — recall table has no content_tsv),
    with optional cross-encoder re-ranking.
    """
    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector

    from agent.storage import get_db_url, get_embeddings

    try:
        embeddings = get_embeddings()
        query_embedding = Vector(embeddings.embed_query(query))
    except Exception as e:
        import sys
        print(f"[memory_manager] Recall embedding failed: {e}", file=sys.stderr)
        return []

    try:
        conn = psycopg.connect(get_db_url(), connect_timeout=5)
    except Exception as e:
        import sys
        print(f"[memory_manager] Recall DB connection failed: {e}", file=sys.stderr)
        return []

    register_vector(conn)
    candidate_limit = int(limit) * 3
    rows = conn.execute(
        """
        SELECT id, thread_id, role, content, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM recall_memory
        WHERE 1 - (embedding <=> %s::vector) > 0.3
        ORDER BY score DESC
        LIMIT %s
        """,
        (query_embedding, query_embedding, candidate_limit),
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        score = float(row[5])
        meta = row[4] if isinstance(row[4], dict) else json.loads(row[4])
        results.append({
            "id": row[0],
            "thread_id": row[1],
            "role": row[2],
            "content": row[3],
            "metadata": meta,
            "score": score,
        })

    # Cross-encoder re-ranking
    try:
        from agent.storage.reranker import rerank
        results = rerank(query, results, top_k=int(limit))
    except Exception:
        results = results[:int(limit)]

    return results


async def recall_memory(state: AgentState, config: RunnableConfig) -> dict:
    """Search archival AND recall memory for content referenced by the user."""
    from datetime import datetime, timezone

    user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    if not user_msg:
        return {"archival_results": [], "recall_results": []}

    # Save user message to recall memory
    try:
        await asyncio.to_thread(_sync_save_to_recall, "user", user_msg)
    except Exception as e:
        import sys
        print(f"[memory_manager] Failed to save to recall: {e}", file=sys.stderr)

    # Search both archival and recall in parallel
    archival_task = asyncio.to_thread(_sync_search_archival, user_msg, 5)
    recall_task = asyncio.to_thread(_sync_search_recall, user_msg, 5)
    archival_results_raw, recall_results_raw = await asyncio.gather(
        archival_task, recall_task
    )

    # Log memory operation
    memory_ops = [{
        "type": "recall",
        "query": user_msg[:100] + "..." if len(user_msg) > 100 else user_msg,
        "archival_count": len(archival_results_raw),
        "recall_count": len(recall_results_raw),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]

    return {
        "archival_results": [
            {
                "content": r["content"],
                "source": r["metadata"].get("source", ""),
                "score": r["score"],
            }
            for r in archival_results_raw
        ],
        "recall_results": [
            {
                "content": r["content"],
                "role": r.get("role", ""),
                "thread_id": r.get("thread_id", ""),
                "score": r["score"],
            }
            for r in recall_results_raw
        ],
        "memory_operations": memory_ops,
    }


async def evaluate_recall(state: AgentState, config: RunnableConfig) -> dict:
    """Evaluate whether retrieved memory is sufficient to answer the user's question.

    Routes to respond (memory sufficient) or generate_query (needs web research).
    """
    from datetime import datetime, timezone

    configurable = Configuration.from_runnable_config(config)

    user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    archival_results = state.get("archival_results", [])
    recall_results = state.get("recall_results", [])

    # No memory results at all — insufficient
    if not archival_results and not recall_results:
        memory_ops = [{
            "type": "evaluate",
            "is_sufficient": False,
            "reason": "No relevant memories found.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
        return {"memory_sufficient": False, "memory_evaluation": "No relevant memories found.", "memory_operations": memory_ops}

    # Format memory content for evaluation — include both sources
    memory_parts = []
    for r in archival_results:
        memory_parts.append(
            f"- [archival:{r.get('source', 'unknown')}] {r['content']} (score: {r.get('score', 0):.2f})"
        )
    for r in recall_results:
        preview = r['content'][:200] + ("..." if len(r['content']) > 200 else "")
        memory_parts.append(
            f"- [recall:{r.get('role', '?')}] {preview} (score: {r.get('score', 0):.2f})"
        )
    memory_content = "\n".join(memory_parts)

    # Ask LLM to evaluate
    llm = ChatOpenAI(
        model=configurable.llm_model,
        base_url=configurable.llm_base_url,
        api_key=configurable.llm_api_key,
        temperature=0,
        http_async_client=httpx.AsyncClient(proxy=None),
    )

    prompt = EVALUATE_RECALL_PROMPT.format(
        question=user_msg,
        memory_content=memory_content,
    )

    try:
        response = await llm.ainvoke(prompt)
        parsed = _parse_json(response.content)
        is_sufficient = parsed.get("is_sufficient", False)
        reason = parsed.get("reason", "Evaluation completed.")
    except Exception:
        is_sufficient = False
        reason = "Failed to evaluate memory content."

    memory_ops = [{
        "type": "evaluate",
        "is_sufficient": is_sufficient,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]

    return {
        "memory_sufficient": is_sufficient,
        "memory_evaluation": reason,
        "memory_operations": memory_ops,
    }


def _sync_put_to_archival(content: str, namespace: str, metadata: dict) -> str:
    """Synchronous archival storage (runs in thread)."""
    import uuid
    from datetime import datetime, timezone

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
        "INSERT INTO archival_memory (id, namespace, content, metadata, embedding) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
        "content=EXCLUDED.content, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
        (entry_id, namespace, content, json.dumps(metadata), embedding),
    )
    conn.commit()
    conn.close()
    return entry_id


def _sync_put_batch_to_archival(entries: list[dict], namespace: str) -> list[str]:
    """Synchronous batch archival storage (runs in thread)."""
    import uuid
    from datetime import datetime, timezone

    import psycopg
    from pgvector import Vector
    from pgvector.psycopg import register_vector

    from agent.storage import get_db_url, get_embeddings

    if not entries:
        return []

    embeddings = get_embeddings()
    embedding_vectors = embeddings.embed_documents([e["content"] for e in entries])
    now = datetime.now(timezone.utc).isoformat()
    ids: list[str] = []

    conn = psycopg.connect(get_db_url(), connect_timeout=5)
    register_vector(conn)
    for entry, emb in zip(entries, embedding_vectors):
        entry_id = str(uuid.uuid4())
        meta = entry.get("metadata", {})
        meta["timestamp"] = now
        ids.append(entry_id)
        conn.execute(
            "INSERT INTO archival_memory (id, namespace, content, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET "
            "content=EXCLUDED.content, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
            (entry_id, namespace, entry["content"], json.dumps(meta), Vector(emb)),
        )
    conn.commit()
    conn.close()
    return ids


async def save_to_archival(state: AgentState, config: RunnableConfig) -> dict:
    """Extract key findings and save them to archival memory (PostgreSQL)."""
    from datetime import datetime, timezone

    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(
        model=configurable.llm_model,
        base_url=configurable.llm_base_url,
        api_key=configurable.llm_api_key,
        temperature=0.3,
        http_async_client=httpx.AsyncClient(proxy=None),
    )

    research_topic = _get_research_topic(state["messages"])
    summaries = "\n\n---\n\n".join(state.get("web_research_result", []))

    if not summaries:
        return {}

    prompt = ARCHIVAL_STORE_PROMPT.format(
        research_topic=research_topic,
        summaries=summaries,
    )
    response = await llm.ainvoke(prompt)

    try:
        entry_id = await asyncio.to_thread(
            _sync_put_to_archival,
            response.content,
            "research",
            {"source": "research_summary", "topic": research_topic},
        )
    except Exception as e:
        return {
            "archival_results": [{
                "content": response.content,
                "source": "research_summary",
                "topic": research_topic,
                "error": str(e),
            }]
        }

    # Log memory operation
    memory_ops = [{
        "type": "save",
        "content_preview": response.content[:100] + "..." if len(response.content) > 100 else response.content,
        "source": "research_summary",
        "topic": research_topic,
        "entry_id": entry_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]

    return {
        "archival_results": [{
            "id": entry_id,
            "content": response.content,
            "source": "research_summary",
            "topic": research_topic,
        }],
        "memory_operations": memory_ops,
    }


def _get_research_topic(messages: list) -> str:
    parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            parts.append(msg.content)
    return "\n".join(parts) if parts else ""


async def ingest_document_node(state: AgentState, config: RunnableConfig) -> dict:
    """Ingest a document (URL, text, or PDF) into the knowledge base.

    Extracts text -> chunks -> embeds -> stores in archival memory.
    Supports PDF upload via base64 encoding from the frontend.
    """
    import base64
    from datetime import datetime, timezone

    from agent.storage.ingestion import ingest_pdf, ingest_text, ingest_url

    doc_source = state.get("doc_source", "")
    source_type = state.get("doc_source_type", "url")
    pdf_filename = ""

    if not doc_source:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                content = msg.content

                # Check for PDF upload from frontend
                pdf_match = re.match(
                    r"\[UPLOAD_PDF:(.+?)\](.+?)\[/UPLOAD_PDF\]",
                    content,
                    re.DOTALL,
                )
                if pdf_match:
                    pdf_filename = pdf_match.group(1)
                    doc_source = pdf_match.group(2).strip()
                    source_type = "pdf"
                    break

                url_match = re.search(r"https?://\S+", content)
                if url_match:
                    doc_source = url_match.group(0)
                    source_type = "url"
                else:
                    doc_source = content
                    source_type = "text"
                break

    if not doc_source:
        return {"ingest_result": "Error: No document source provided."}

    try:
        if source_type == "pdf":
            pdf_bytes = base64.b64decode(doc_source)
            chunks = await ingest_pdf(pdf_bytes, filename=pdf_filename or "document.pdf")
            if not chunks:
                return {"ingest_result": f"Failed to extract text from PDF: {pdf_filename}"}
            doc_label = pdf_filename or "document.pdf"
        elif source_type == "url":
            title, chunks = await ingest_url(doc_source)
            if not chunks:
                return {"ingest_result": f"Failed to extract text from URL: {doc_source}"}
            doc_label = title
        else:
            chunks = await ingest_text(
                content=doc_source,
                source_name="manual_text",
                source_type="text",
            )
            if not chunks:
                return {"ingest_result": "No text content to ingest."}
            doc_label = "manual_text"

        entries = [
            {"content": c.content, "metadata": {"source": c.source, "source_type": c.source_type, "chunk_index": c.index, "document": doc_label}}
            for c in chunks
        ]
        entry_ids = await asyncio.to_thread(
            _sync_put_batch_to_archival, entries, "ingested"
        )

        if source_type == "pdf":
            source_desc = f"PDF: {doc_label}"
        elif source_type == "url":
            source_desc = f"URL: {doc_source}"
        else:
            source_desc = "plain text"

        result = (
            f"Ingested '{doc_label}': {len(chunks)} chunks extracted, "
            f"{len(entry_ids)} stored in archival memory. "
            f"Source: {source_desc}."
        )

        # Log memory operation
        memory_ops = [{
            "type": "ingest",
            "document": doc_label,
            "chunks_count": len(chunks),
            "stored_count": len(entry_ids),
            "source_type": source_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]

        return {
            "ingest_result": result,
            "archival_results": [
                {
                    "id": eid,
                    "content": c.content,
                    "source": c.source,
                    "source_type": c.source_type,
                    "chunk_index": c.index,
                    "document": doc_label,
                }
                for eid, c in zip(entry_ids, chunks)
            ],
            "memory_operations": memory_ops,
        }
    except Exception as e:
        return {"ingest_result": f"Error ingesting document: {e}"}
