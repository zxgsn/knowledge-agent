"""Memory-related nodes: intent routing, archival save, memory recall."""

from __future__ import annotations

import asyncio
import json
import re

import httpx
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.db import (
    get_recent_archival,
    get_recent_recall,
    insert_document,
    put_batch_to_archival,
    put_to_archival,
    save_to_recall,
    search_archival,
    search_by_entity,
    search_by_temporal,
    search_recall,
)
from agent.logger import get_logger
from agent.prompts import ARCHIVAL_STORE_PROMPT, EVALUATE_RECALL_PROMPT, ROUTE_INTENT_PROMPT
from agent.retry import with_retry
from agent.state import AgentState
from agent.utils import get_llm, parse_json

logger = get_logger(__name__)

# Only retry on transient network/timeout errors, not auth or validation errors
_TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    httpx.ConnectError,
    httpx.TimeoutException,
)


async def route_intent(state: AgentState, config: RunnableConfig) -> dict:
    """Classify user intent: chat, research, or memory_edit."""
    configurable = Configuration.from_runnable_config(config)

    # If mode is already set (e.g. from frontend), keep it
    # This avoids a costly LLM call when the user explicitly selected a mode.
    if state.get("mode") in ("chat", "research", "memory_edit", "ingest"):
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
    try:
        response = await with_retry(
            max_retries=2, retryable_exceptions=_TRANSIENT_ERRORS,
        )(llm.ainvoke)(prompt)
        parsed = parse_json(response.content)
        mode = parsed.get("mode", "chat")
        if mode not in ("chat", "research", "memory_edit", "ingest", "recall"):
            mode = "chat"
        need_recall = parsed.get("need_recall", False)
    except Exception as exc:
        logger.warning("Intent routing LLM failed, defaulting to chat: %s", exc)
        mode = "chat"
        need_recall = False

    return {"mode": mode, "need_recall": need_recall}


async def _rewrite_query(user_msg: str, messages: list, config: Configuration) -> str:
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
        llm = get_llm(config, temperature=0.0)
        response = await with_retry(
            max_retries=2, retryable_exceptions=_TRANSIENT_ERRORS,
        )(llm.ainvoke)(prompt)
        rewritten = response.content.strip().strip('"')
        if rewritten:
            logger.debug("Query rewritten: %s -> %s", user_msg[:60], rewritten[:60])
            return rewritten
        return user_msg
    except Exception as exc:
        logger.warning("Query rewrite LLM failed, using original: %s", exc)
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
        response = await with_retry(
            max_retries=2, retryable_exceptions=_TRANSIENT_ERRORS,
        )(llm.ainvoke)(hyde_prompt)
        result = response.content.strip()
        logger.debug("HyDE generated: %s", result[:80])
        return result
    except Exception as exc:
        logger.warning("HyDE LLM failed, using original query: %s", exc)
        return query


_TEMPORAL_KEYWORDS = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"yesterday|today|tomorrow|last\s+week|next\s+week|"
    r"last\s+month|next\s+month|last\s+year|next\s+year|"
    r"mon|tue|wed|thu|fri|sat|sun|"
    r"\u4e0a\u5468|\u4e0b\u5468|\u6628\u5929|\u4eca\u5929|\u660e\u5929|\u4e0a\u4e2a\u6708|\u4e0b\u4e2a\u6708|\u53bb\u5e74|\u660e\u5e74|"
    r"\u5468\u4e00|\u5468\u4e8c|\u5468\u4e09|\u5468\u56db|\u5468\u4e94|\u5468\u516d|\u5468\u65e5)\b",
    re.IGNORECASE,
)


def _detect_temporal_ref(query: str) -> str | None:
    """Extract a temporal reference keyword from the query, if any."""
    m = _TEMPORAL_KEYWORDS.search(query)
    return m.group(0).lower() if m else None


def _detect_entities(query: str, archival_results: list[dict]) -> list[str]:
    """Extract entity names from archival result metadata that also appear in the query."""
    entities = []
    query_lower = query.lower()
    for r in archival_results:
        for ent in r.get("metadata", {}).get("entities", []):
            name = ent.get("name", "")
            if name and name.lower() in query_lower and name not in entities:
                entities.append(name)
    return entities


def _estimate_query_complexity(query: str) -> int:
    """Estimate query complexity and return a retrieval limit (3-8).

    Simple heuristics based on:
    - Word count (more words -> more complex)
    - Presence of comparison/enumeration keywords
    - Question depth markers (why, how, compare, etc.)
    """
    words = query.split()
    word_count = len(words)
    query_lower = query.lower()

    # Base limit from word count
    if word_count <= 5:
        limit = 3
    elif word_count <= 12:
        limit = 5
    else:
        limit = 6

    # Boost for complex question patterns
    complex_patterns = [
        "compare", "difference", "versus", "vs", "pros and cons",
        "advantages", "disadvantages", "how does", "why does",
        "explain", "analyze", "evaluate", "summarize all",
        "list all", "what are the", "tell me everything",
        "比较", "区别", "优缺点", "为什么", "如何", "分析", "总结",
    ]
    for pat in complex_patterns:
        if pat in query_lower:
            limit = min(limit + 2, 8)
            break

    # Boost for multi-part questions (contains "and", ";", multiple "?")
    if query.count("?") >= 2 or ";" in query:
        limit = min(limit + 1, 8)

    return limit


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

    thread_id = state.get("thread_id", "default")

    # Proactive memory: push recent memories at conversation start
    mode = state.get("mode", "chat")
    turn_count = state.get("turn_count", 0)
    if (configurable.proactive_memory_enabled
            and mode == "chat"
            and turn_count < configurable.proactive_memory_turns):
        archival_results_raw, recall_results_raw = await asyncio.gather(
            asyncio.to_thread(get_recent_archival, 3),
            asyncio.to_thread(get_recent_recall, 3, thread_id=thread_id),
        )
        memory_ops = [{
            "type": "proactive_recall",
            "turn_count": turn_count,
            "archival_count": len(archival_results_raw),
            "recall_count": len(recall_results_raw),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
        return {
            "archival_results": [
                {
                    "content": r["content"],
                    "source": r.get("metadata", {}).get("source", ""),
                    "source_type": r.get("metadata", {}).get("source_type", ""),
                    "document": r.get("metadata", {}).get("document", ""),
                    "namespace": r.get("namespace", ""),
                    "timestamp": r.get("metadata", {}).get("timestamp", ""),
                    "score": r.get("score", 1.0),
                }
                for r in archival_results_raw
            ],
            "recall_results": [
                {"content": r["content"], "role": r.get("role", ""), "thread_id": r.get("thread_id", ""), "score": r.get("score", 1.0)}
                for r in recall_results_raw
            ],
            "memory_operations": memory_ops,
        }

    # Step 1: Query rewriting (if enabled)
    dispatch_custom_event("progress", {"stage": "memory_search", "detail": "Rewriting query..."}, config=config)
    search_query = user_msg
    if configurable.memory_query_rewrite_enabled:
        search_query = await _rewrite_query(user_msg, state["messages"], configurable)

    # Step 2: HyDE (if enabled) -- generate hypothetical answer for embedding
    dispatch_custom_event("progress", {"stage": "memory_search", "detail": "Generating search embedding..."}, config=config)
    search_query_for_embed = search_query
    if configurable.hyde_enabled:
        search_query_for_embed = await _generate_hypothetical(search_query, configurable)

    # Step 3: Search both archival and recall in parallel
    # Dynamic context allocation: adjust retrieval volume based on query complexity
    dispatch_custom_event("progress", {"stage": "memory_search", "detail": "Searching archival + recall memory..."}, config=config)
    retrieval_limit = _estimate_query_complexity(search_query)
    logger.info("Dynamic retrieval limit: %d (query: %s)", retrieval_limit, search_query[:60])
    archival_task = asyncio.to_thread(
        search_archival, search_query_for_embed, retrieval_limit,
        rerank_enabled=configurable.rerank_enabled,
        mmr_enabled=configurable.mmr_enabled,
        mmr_lambda=configurable.mmr_lambda,
    )
    recall_task = asyncio.to_thread(search_recall, search_query, retrieval_limit, thread_id=thread_id)
    archival_results_raw, recall_results_raw = await asyncio.gather(
        archival_task, recall_task
    )

    # Step 3b: Temporal/entity-aware supplementary search
    temporal_ref = _detect_temporal_ref(search_query)
    if temporal_ref and configurable.structured_extraction:
        temporal_results = await asyncio.to_thread(
            search_by_temporal, temporal_ref, "conversation_facts", 5
        )
        # Merge temporal results (avoid duplicates by id)
        existing_ids = {r.get("id") for r in archival_results_raw}
        for tr in temporal_results:
            if tr["id"] not in existing_ids:
                archival_results_raw.append({
                    "content": tr["content"],
                    "metadata": tr.get("metadata", {}),
                    "score": 0.6,  # moderate score for metadata match
                })

    # Save user message to recall AFTER searching to avoid matching itself
    try:
        await asyncio.to_thread(save_to_recall, "user", user_msg, thread_id=thread_id)
    except Exception as exc:
        logger.warning("Failed to save to recall: %s", exc)

    # Log memory operation
    memory_ops = [{
        "type": "recall",
        "query": user_msg[:100] + "..." if len(user_msg) > 100 else user_msg,
        "archival_count": len(archival_results_raw),
        "recall_count": len(recall_results_raw),
        "temporal_boost": temporal_ref is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]

    return {
        "archival_results": [
            {
                "content": r["content"],
                "source": r.get("metadata", {}).get("source", ""),
                "source_type": r.get("metadata", {}).get("source_type", ""),
                "document": r.get("metadata", {}).get("document", ""),
                "namespace": r.get("namespace", ""),
                "timestamp": r.get("metadata", {}).get("timestamp", ""),
                "score": r.get("score", 0),
            }
            for r in archival_results_raw
        ],
        "recall_results": [
            {
                "content": r["content"],
                "role": r.get("role", ""),
                "thread_id": r.get("thread_id", ""),
                "score": r.get("score", 0),
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

    # No memory results at all -- insufficient
    if not archival_results and not recall_results:
        memory_ops = [{
            "type": "evaluate",
            "is_sufficient": False,
            "reason": "No relevant memories found.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
        return {"memory_sufficient": False, "memory_evaluation": "No relevant memories found.", "memory_operations": memory_ops}

    # Format memory content for evaluation -- include both sources
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

    # Ask LLM to evaluate (with retry)
    dispatch_custom_event("progress", {"stage": "evaluate_recall", "detail": "Evaluating memory relevance..."}, config=config)
    llm = get_llm(configurable, temperature=0)

    prompt = EVALUATE_RECALL_PROMPT.format(
        question=user_msg,
        memory_content=memory_content,
    )

    try:
        response = await with_retry(
            max_retries=2, retryable_exceptions=_TRANSIENT_ERRORS,
        )(llm.ainvoke)(prompt)
        parsed = parse_json(response.content)
        is_sufficient = parsed.get("is_sufficient", False)
        reason = parsed.get("reason", "Evaluation completed.")
        logger.debug("Recall evaluation: sufficient=%s reason=%s", is_sufficient, reason[:80])
    except Exception as exc:
        # Safe fallback: assume memory is insufficient so research is triggered
        logger.warning("Recall evaluation LLM failed, assuming insufficient: %s", exc)
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

    dispatch_custom_event("progress", {"stage": "save_to_archival", "detail": "Extracting key findings..."}, config=config)
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

    try:
        response = await with_retry(
            max_retries=2, retryable_exceptions=_TRANSIENT_ERRORS,
        )(llm.ainvoke)(prompt)
    except Exception as exc:
        logger.warning("Archival store LLM failed after retries: %s", exc)
        return {}

    try:
        entry_id = await asyncio.to_thread(
            put_to_archival,
            response.content,
            "research",
            {"source": "research_summary", "topic": research_topic},
        )
    except Exception as exc:
        logger.error("Failed to store to archival: %s", exc)
        return {}

    # Log memory operation -- do NOT return archival_results here,
    # as that would overwrite the original memory search results
    # from recall_memory. The research content is already available
    # via web_research_result in state.
    memory_ops = [{
        "type": "save",
        "content_preview": response.content[:100] + "..." if len(response.content) > 100 else response.content,
        "source": "research_summary",
        "topic": research_topic,
        "entry_id": entry_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]

    return {"memory_operations": memory_ops}


def _get_research_topic(messages: list) -> str:
    parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            parts.append(msg.content)
    return "\n".join(parts) if parts else ""


def _extract_question(content: str, source_type: str, doc_source: str) -> str:
    """Extract user question from a combined ingest+question message.

    Returns the question text (stripped), or empty string if none found.
    """
    if source_type == "pdf":
        # Text before [UPLOAD_PDF:...] tag
        before = re.split(r"\[UPLOAD_PDF:", content, maxsplit=1)[0]
        return before.strip()
    if source_type == "url":
        # Remove the URL from content, keep the rest
        remaining = content.replace(doc_source, "").strip()
        return remaining
    # Text type: entire message is the document source
    return ""


async def ingest_document_node(state: AgentState, config: RunnableConfig) -> dict:
    """Ingest a document (URL, text, or PDF) into the knowledge base.

    Extracts text -> chunks -> embeds -> stores in archival memory.
    If the user also asked a question alongside the document, extracts it
    into ingest_question so the graph can route through recall -> respond.
    """
    import base64
    from datetime import datetime, timezone

    from agent.storage.ingestion import ingest_pdf, ingest_text, ingest_url

    dispatch_custom_event("progress", {"stage": "ingest_document", "detail": "Processing document..."}, config=config)
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
    original_content = ""

    if not doc_source:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                original_content = msg.content

                # Check for PDF upload from frontend
                pdf_match = re.search(
                    r"\[UPLOAD_PDF:(.+?)\](.+?)\[/UPLOAD_PDF\]",
                    original_content,
                    re.DOTALL,
                )
                if pdf_match:
                    pdf_filename = pdf_match.group(1)
                    doc_source = pdf_match.group(2).strip()
                    source_type = "pdf"
                    break

                url_match = re.search(r"https?://\S+", original_content)
                if url_match:
                    doc_source = url_match.group(0)
                    source_type = "url"
                else:
                    doc_source = original_content
                    source_type = "text"
                break

    if not doc_source:
        return {"ingest_result": "Error: No document source provided."}

    try:
        if source_type == "pdf":
            dispatch_custom_event("progress", {"stage": "ingest_document", "detail": f"Extracting text from PDF: {pdf_filename}..."}, config=config)
            pdf_bytes = base64.b64decode(doc_source)
            chunks = await ingest_pdf(pdf_bytes, filename=pdf_filename or "document.pdf", **chunk_params)
            if not chunks:
                return {"ingest_result": f"Failed to extract text from PDF: {pdf_filename}"}
            doc_label = pdf_filename or "document.pdf"
        elif source_type == "url":
            dispatch_custom_event("progress", {"stage": "ingest_document", "detail": f"Fetching URL: {doc_source[:60]}..."}, config=config)
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

        # Optional: LLM enrichment of chunks
        if cfg.document_enrichment:
            dispatch_custom_event("progress", {"stage": "ingest_document", "detail": f"Enriching {len(chunks)} chunks with LLM..."}, config=config)
            from agent.storage.ingestion import enrich_chunks
            enrich_llm = get_llm(cfg, temperature=0.0)
            chunks = await enrich_chunks(chunks, enrich_llm)

        dispatch_custom_event("progress", {"stage": "ingest_document", "detail": f"Embedding {len(chunks)} chunks..."}, config=config)
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

        entries = []
        for c in chunks:
            meta = {
                "source": c.source,
                "source_type": c.source_type,
                "chunk_index": c.index,
                "document": doc_label,
                "document_id": document_id,
            }
            # Include enriched metadata if available
            if c.metadata.get("summary"):
                meta["summary"] = c.metadata["summary"]
            if c.metadata.get("entities"):
                meta["entities"] = c.metadata["entities"]
            if c.metadata.get("keywords"):
                meta["keywords"] = c.metadata["keywords"]
            entries.append({"content": c.content, "metadata": meta})
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

        # Extract user question from combined message (if any)
        question = _extract_question(original_content, source_type, doc_source)

        # Rewrite user message so downstream nodes see clean text
        messages = list(state["messages"])
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                messages[i] = HumanMessage(
                    content=question or f"Ingested document: {doc_label}"
                )
                break

        return {
            "ingest_result": result,
            "ingest_question": question,
            "messages": messages,
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
