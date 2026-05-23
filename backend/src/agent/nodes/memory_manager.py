"""Memory-related nodes: intent routing, archival save, memory recall."""

from __future__ import annotations

import asyncio
import json
import re

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.db import (
    insert_document,
    put_batch_to_archival,
    put_to_archival,
    save_to_recall,
    search_archival,
    search_recall,
)
from agent.prompts import ARCHIVAL_STORE_PROMPT, EVALUATE_RECALL_PROMPT, ROUTE_INTENT_PROMPT
from agent.state import AgentState
from agent.utils import get_llm, parse_json


async def route_intent(state: AgentState, config: RunnableConfig) -> dict:
    """Classify user intent: chat, research, or memory_edit."""
    configurable = Configuration.from_runnable_config(config)

    # If mode is already set (e.g. from frontend), keep it
    if state.get("mode") in ("research", "memory_edit", "ingest"):
        return {}

    llm = get_llm(configurable, temperature=0)

    user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    if not user_msg:
        return {"mode": "chat"}

    prompt = ROUTE_INTENT_PROMPT.format(user_message=user_msg)
    response = await llm.ainvoke(prompt)
    parsed = parse_json(response.content)
    mode = parsed.get("mode", "chat")
    if mode not in ("chat", "research", "memory_edit", "ingest", "recall"):
        mode = "chat"

    need_recall = parsed.get("need_recall", False)

    return {"mode": mode, "need_recall": need_recall}


async def _rewrite_query(user_msg: str, messages: list) -> str:
    """Rewrite ambiguous/referential queries into self-contained form."""
    recent = []
    for m in messages[-6:]:
        if hasattr(m, "content"):
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            recent.append(f"{role}: {m.content[:200]}")
    context = "\n".join(recent)

    prompt = (
        "Given the conversation history, rewrite the user's latest message "
        "as a self-contained search query. If the message is already self-contained, "
        "return it unchanged. Only return the rewritten query, nothing else.\n\n"
        f"Conversation history:\n{context}\n\n"
        f"Latest message: {user_msg}"
    )
    try:
        llm = get_llm(Configuration(), temperature=0.0)
        response = await llm.ainvoke(prompt)
        rewritten = response.content.strip().strip('"')
        return rewritten if rewritten else user_msg
    except Exception:
        return user_msg


async def _generate_hypothetical(query: str, config: Configuration) -> str:
    """Generate a hypothetical answer for HyDE retrieval."""
    hyde_prompt = (
        "Answer this question in 1-2 short sentences. "
        "Be specific with names and details. "
        "If you don't know, guess based on the question context.\n\n"
        f"Question: {query}"
    )
    try:
        llm = get_llm(config, temperature=0.7)
        response = await llm.ainvoke(hyde_prompt)
        return response.content.strip()
    except Exception:
        return query


async def recall_memory(state: AgentState, config: RunnableConfig) -> dict:
    """Search archival AND recall memory for content referenced by the user."""
    from datetime import datetime, timezone

    configurable = Configuration.from_runnable_config(config)

    user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break

    if not user_msg:
        return {"archival_results": [], "recall_results": []}

    # Save user message to recall memory
    try:
        await asyncio.to_thread(save_to_recall, "user", user_msg)
    except Exception as e:
        import sys
        print(f"[memory_manager] Failed to save to recall: {e}", file=sys.stderr)

    # Step 1: Query rewriting (if enabled)
    search_query = user_msg
    if configurable.memory_query_rewrite_enabled:
        search_query = await _rewrite_query(user_msg, state["messages"])

    # Step 2: HyDE (if enabled) — generate hypothetical answer for embedding
    search_query_for_embed = search_query
    if configurable.hyde_enabled:
        search_query_for_embed = await _generate_hypothetical(search_query, configurable)

    # Step 3: Search both archival and recall in parallel
    archival_task = asyncio.to_thread(search_archival, search_query_for_embed, 5)
    recall_task = asyncio.to_thread(search_recall, search_query, 5)
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
    llm = get_llm(configurable, temperature=0)

    prompt = EVALUATE_RECALL_PROMPT.format(
        question=user_msg,
        memory_content=memory_content,
    )

    try:
        response = await llm.ainvoke(prompt)
        parsed = parse_json(response.content)
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


async def save_to_archival(state: AgentState, config: RunnableConfig) -> dict:
    """Extract key findings and save them to archival memory (PostgreSQL)."""
    from datetime import datetime, timezone

    configurable = Configuration.from_runnable_config(config)
    llm = get_llm(configurable, temperature=0.3)

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
            put_to_archival,
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

    cfg = Configuration.from_runnable_config(config)
    chunk_params = dict(
        strategy=cfg.chunk_strategy,
        similarity_threshold=cfg.chunk_similarity_threshold,
        min_chunk_size=cfg.chunk_min_size,
        max_chunk_size=cfg.chunk_max_size,
    )

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
            chunks = await ingest_pdf(pdf_bytes, filename=pdf_filename or "document.pdf", **chunk_params)
            if not chunks:
                return {"ingest_result": f"Failed to extract text from PDF: {pdf_filename}"}
            doc_label = pdf_filename or "document.pdf"
        elif source_type == "url":
            title, chunks = await ingest_url(doc_source, **chunk_params)
            if not chunks:
                return {"ingest_result": f"Failed to extract text from URL: {doc_source}"}
            doc_label = title
        else:
            chunks = await ingest_text(
                content=doc_source,
                source_name="manual_text",
                source_type="text",
                **chunk_params,
            )
            if not chunks:
                return {"ingest_result": "No text content to ingest."}
            doc_label = "manual_text"

        # Reconstruct full text from chunks for document storage
        full_text = "\n\n".join(c.content for c in chunks)

        # Write to documents table first
        document_id, is_new = await asyncio.to_thread(
            insert_document,
            doc_label,
            doc_source if source_type == "url" else doc_label,
            source_type,
            full_text,
            len(chunks),
        )

        if not is_new:
            return {"ingest_result": f"Document '{doc_label}' already exists in the library. Skipping."}

        entries = [
            {"content": c.content, "metadata": {"source": c.source, "source_type": c.source_type, "chunk_index": c.index, "document": doc_label, "document_id": document_id}}
            for c in chunks
        ]
        entry_ids = await asyncio.to_thread(
            put_batch_to_archival, entries, "ingested", document_id
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
