"""Response generation node."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import urllib.request

from langchain_core.callbacks import dispatch_custom_event
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.db import save_to_recall
from agent.heartbeat import heartbeat
from agent.memory.core_memory import CoreMemory
from agent.memory.tools import create_memory_tools
from agent.prompts import ANSWER_PROMPT, SUMMARIZE_HISTORY_PROMPT, SYSTEM_PROMPT, get_current_date
from agent.state import AgentState
from agent.utils import get_llm

MAX_TOOL_ROUNDS = 5
# Approximate chars per token for context budget estimation
CHARS_PER_TOKEN = 4
# Reserve tokens for system prompt, tools, and response
CONTEXT_TOKEN_BUDGET = 120_000
RESERVED_TOKENS = 10_000


# #region debug-point B:report-helper
def _debug_report(hypothesis_id: str, location: str, msg: str, data: dict | None = None) -> None:
    payload = {
        "sessionId": "frontend-network-error",
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": msg,
        "data": data or {},
        "ts": int(time.time() * 1000),
    }
    debug_url = "http://127.0.0.1:7777/event"
    env_path = os.path.join(".dbg", "frontend-network-error.env")
    try:
        with open(env_path, encoding="utf-8") as env_file:
            for line in env_file:
                if line.startswith("DEBUG_SERVER_URL="):
                    debug_url = line.split("=", 1)[1].strip() or debug_url
                    break
    except Exception:
        pass
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                debug_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=2,
        ).read()
    except Exception:
        pass
# #endregion


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    return len(text) // CHARS_PER_TOKEN


def _truncate_to_budget(
    context_parts: list[str], max_tokens: int
) -> list[str]:
    """Truncate context parts to fit within token budget.

    Priority order: core memory > archival > recall > web research.
    Truncates from the end of lower-priority sections first.
    """
    total_tokens = sum(_estimate_tokens(p) for p in context_parts)
    if total_tokens <= max_tokens:
        return context_parts

    # Work backwards, truncating lower-priority content first
    result = list(context_parts)
    for i in range(len(result) - 1, -1, -1):
        if total_tokens <= max_tokens:
            break
        part_tokens = _estimate_tokens(result[i])
        excess = total_tokens - max_tokens
        if excess >= part_tokens:
            # Remove this part entirely
            total_tokens -= part_tokens
            result[i] = ""
        else:
            # Truncate this part
            chars_to_keep = (part_tokens - excess) * CHARS_PER_TOKEN
            result[i] = result[i][:chars_to_keep] + "\n[... truncated for context budget]"
            total_tokens -= excess

    return [p for p in result if p]


async def respond(state: AgentState, config: RunnableConfig) -> dict:
    """Generate the final response to the user.

    Supports tool calling for core memory editing. The LLM can call
    core_memory_replace/insert/rethink/view tools in a loop before
    producing its final text response.
    """
    configurable = Configuration.from_runnable_config(config)
    llm = get_llm(configurable, temperature=0.5)
    # #region debug-point B:respond-start
    _debug_report("B", "backend/src/agent/nodes/responder.py:respond:start", "[DEBUG] respond node started", {
        "mode": state.get("mode", "chat"),
        "message_count": len(state.get("messages", [])),
        "archival_results": len(state.get("archival_results", [])),
        "recall_results": len(state.get("recall_results", [])),
    })
    # #endregion

    core_memory = CoreMemory.from_dict(state.get("core_memory", {}))
    summaries = "\n\n---\n\n".join(state.get("web_research_result", []))

    system = SYSTEM_PROMPT.format(
        current_date=get_current_date(),
        memory_blocks=core_memory.compile(),
    )

    archival_results = state.get("archival_results", [])
    recall_results = state.get("recall_results", [])
    archival_context = _format_archival_results(archival_results)
    recall_context = _format_recall_results(recall_results)

    # --- Conversation summarization ---
    existing_summary = state.get("conversation_summary", "")
    messages_list = list(state["messages"])
    new_summary = ""

    if (
        configurable.summarize_threshold > 0
        and len(messages_list) > configurable.summarize_threshold
    ):
        keep = configurable.summarize_keep_recent
        old_messages = messages_list[:-keep]
        if old_messages:
            # Build text from old messages
            history_parts = []
            for m in old_messages:
                if isinstance(m, HumanMessage):
                    history_parts.append(f"User: {m.content[:500]}")
                elif isinstance(m, AIMessage) and m.content:
                    history_parts.append(f"Assistant: {m.content[:500]}")
            old_history = "\n".join(history_parts)
            if existing_summary:
                old_history = f"Previous summary: {existing_summary}\n\nNew messages:\n{old_history}"
            try:
                summarize_llm = get_llm(configurable, temperature=0.2)
                resp = await summarize_llm.ainvoke(
                    SUMMARIZE_HISTORY_PROMPT.format(history=old_history)
                )
                new_summary = resp.content
            except Exception as e:
                print(f"[responder] Summarization failed: {e}", file=sys.stderr)
                new_summary = existing_summary
    elif existing_summary:
        new_summary = existing_summary

    if state.get("mode") == "research" and summaries:
        research_topic = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                research_topic = msg.content
                break
        user_content = ANSWER_PROMPT.format(
            research_topic=research_topic,
            summaries=summaries,
            archival_context=archival_context,
        )
        messages_for_llm = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
    else:
        messages_for_llm = [{"role": "system", "content": system}]
        # Include conversation summary as context if available
        if new_summary:
            messages_for_llm.append({
                "role": "user",
                "content": f"[Conversation Summary]\n{new_summary}",
            })
        for msg in state["messages"][-20:]:
            if isinstance(msg, HumanMessage):
                content = msg.content
                # Strip PDF base64 data — replace tag with short summary
                pdf_match = re.search(
                    r"\[UPLOAD_PDF:(.+?)\]",
                    content,
                )
                if pdf_match:
                    doc_name = pdf_match.group(1)
                    remaining = re.sub(
                        r"\[UPLOAD_PDF:.+?\].+?\[/UPLOAD_PDF\]",
                        "",
                        content,
                        flags=re.DOTALL,
                    ).strip()
                    suffix = f"[Uploaded PDF: {doc_name}. Document has been ingested into archival memory.]"
                    content = f"{remaining}\n\n{suffix}" if remaining else suffix
                messages_for_llm.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage) and msg.content:
                messages_for_llm.append({"role": "assistant", "content": msg.content})
        context_parts = []
        if archival_context:
            context_parts.append(
                "[Archival Memory]\n"
                "Each entry has an ID in brackets like [a0], [a1], etc. "
                "IMPORTANT: When you use information from an entry, you MUST cite it "
                "inline as [ref:a0], [ref:a1], etc. This is required for tracking.\n"
                f"{archival_context}"
            )
        if recall_context:
            context_parts.append(
                "[Conversation History]\n"
                "Each entry has an ID in brackets like [r0], [r1], etc. "
                "IMPORTANT: When you use information from an entry, you MUST cite it "
                "inline as [ref:r0], [ref:r1], etc. This is required for tracking.\n"
                f"{recall_context}"
            )
        if summaries:
            context_parts.append(f"[Web Research]\n{summaries}")
        if context_parts:
            # Truncate context to fit within token budget
            available_tokens = CONTEXT_TOKEN_BUDGET - RESERVED_TOKENS - _estimate_tokens(system)
            for msg in messages_for_llm[1:]:  # Exclude system message
                available_tokens -= _estimate_tokens(msg.get("content", ""))
            context_parts = _truncate_to_budget(context_parts, max(available_tokens, 1000))
            messages_for_llm.append({
                "role": "user",
                "content": "\n\n".join(context_parts),
            })

    # Bind core memory tools so the LLM can read/edit memory
    tools = create_memory_tools(core_memory, enable_archival=True)
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    # Tool-calling loop: LLM may call tools multiple times before final answer
    dispatch_custom_event("progress", {"stage": "respond", "detail": "Generating response..."}, config=config)

    response = None
    async with heartbeat(config, "respond", "Generating response...", interval=10):
        # #region debug-point B:respond-llm-call
        llm_call_started = time.perf_counter()
        _debug_report("B", "backend/src/agent/nodes/responder.py:respond:llm-before", "[DEBUG] response LLM call starting", {
            "messages_for_llm": len(messages_for_llm),
        })
        try:
            response = await llm_with_tools.ainvoke(messages_for_llm)
            _debug_report("B", "backend/src/agent/nodes/responder.py:respond:llm-after", "[DEBUG] response LLM call completed", {
                "elapsed_ms": round((time.perf_counter() - llm_call_started) * 1000, 2),
                "tool_calls": len(response.tool_calls or []),
                "content_length": len(response.content or ""),
            })
        except Exception as exc:
            _debug_report("B", "backend/src/agent/nodes/responder.py:respond:llm-error", "[DEBUG] response LLM call failed", {
                "elapsed_ms": round((time.perf_counter() - llm_call_started) * 1000, 2),
                "error_type": type(exc).__name__,
                "error": repr(exc),
            })
            logger.error("LLM call failed in respond: %s", exc)
            error_msg = f"I apologize, but I encountered an error while generating a response: {type(exc).__name__}. Please try again."
            return {
                "messages": [AIMessage(content=error_msg)],
                "core_memory": core_memory.to_dict_with_history(),
                "conversation_summary": new_summary,
            }
        # #endregion

        for _ in range(MAX_TOOL_ROUNDS):
            if not response.tool_calls:
                break

            # Execute each tool call and collect results.
            # Append the full assistant message (with reasoning_content) directly.
            messages_for_llm.append(response)
            for tc in response.tool_calls:
                tool = tools_by_name.get(tc["name"])
                if tool:
                    # Run tool in thread to avoid blocking the event loop
                    # (archival tools use sync psycopg which blockbuster flags)
                    result = await asyncio.to_thread(tool.invoke, tc["args"])
                else:
                    result = f"Unknown tool: {tc['name']}"
                messages_for_llm.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tc["id"],
                })

            # #region debug-point B:respond-tool-loop
            loop_started = time.perf_counter()
            try:
                response = await llm_with_tools.ainvoke(messages_for_llm)
                _debug_report("B", "backend/src/agent/nodes/responder.py:respond:tool-loop", "[DEBUG] response tool loop iteration completed", {
                    "elapsed_ms": round((time.perf_counter() - loop_started) * 1000, 2),
                    "tool_calls": len(response.tool_calls or []),
                    "content_length": len(response.content or ""),
                })
            except Exception as exc:
                _debug_report("B", "backend/src/agent/nodes/responder.py:respond:tool-loop-error", "[DEBUG] response tool loop failed", {
                    "elapsed_ms": round((time.perf_counter() - loop_started) * 1000, 2),
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                })
                logger.error("Tool loop failed in respond: %s", exc)
                # If we have partial content, use it; otherwise return error
                if response and response.content:
                    break  # Use existing response content
                error_msg = f"I encountered an error during tool execution: {type(exc).__name__}. Please try again."
                return {
                    "messages": [AIMessage(content=error_msg)],
                    "core_memory": core_memory.to_dict_with_history(),
                    "conversation_summary": new_summary,
                }
            # #endregion

        # If the LLM returned only tool calls with no text content, force one more
        # call without tools to produce an actual user-facing response.
        if response and not response.content and response.tool_calls:
            messages_for_llm.append(response)
            messages_for_llm.append({
                "role": "user",
                "content": "Please provide your response to the user now.",
            })
            # #region debug-point B:respond-finalize
            finalize_started = time.perf_counter()
            try:
                response = await llm.ainvoke(messages_for_llm)
                _debug_report("B", "backend/src/agent/nodes/responder.py:respond:finalize", "[DEBUG] finalize response call completed", {
                    "elapsed_ms": round((time.perf_counter() - finalize_started) * 1000, 2),
                    "content_length": len(response.content or ""),
                })
            except Exception as exc:
                _debug_report("B", "backend/src/agent/nodes/responder.py:respond:finalize-error", "[DEBUG] finalize response call failed", {
                    "elapsed_ms": round((time.perf_counter() - finalize_started) * 1000, 2),
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                })
                logger.error("Finalize call failed in respond: %s", exc)
                error_msg = f"I encountered an error while finalizing the response: {type(exc).__name__}. Please try again."
                return {
                    "messages": [AIMessage(content=error_msg)],
                    "core_memory": core_memory.to_dict_with_history(),
                    "conversation_summary": new_summary,
                }
            # #endregion

    # If response is still None after all attempts, return error
    if response is None:
        error_msg = "I apologize, but I was unable to generate a response. Please try again."
        return {
            "messages": [AIMessage(content=error_msg)],
            "core_memory": core_memory.to_dict_with_history(),
            "conversation_summary": new_summary,
        }

    # Auto-compress blocks that are approaching their character limit
    needs_compression = core_memory.get_blocks_needing_compression()
    if needs_compression:
        compress_llm = get_llm(configurable, temperature=0.2)
        for label in needs_compression:
            await core_memory.compress_block(label, compress_llm)

    # Save AI response to recall memory (thread-isolated)
    thread_id = state.get("thread_id", "default")
    if response.content:
        try:
            await asyncio.to_thread(save_to_recall, "assistant", response.content, thread_id=thread_id)
        except Exception as e:
            print(f"[responder] Failed to save to recall: {e}", file=sys.stderr)

    # Add memory indicator when archival/recall results were retrieved
    response_text = response.content or ""
    if archival_results or recall_results:
        # Detect which memories were cited via [ref:ID] tags in the response
        cited_ids = set(re.findall(r"\[ref:([ar]\d+)\]", response_text))
        # Strip [ref:ID] tags from the displayed response
        response_text = re.sub(r"\s*\[ref:[ar]\d+\]", "", response_text)

        total = len(archival_results) + len(recall_results)
        indicator = f"\n\n---\n> 📚 **从记忆中检索到 {total} 条相关内容**"
        for i, r in enumerate(archival_results):
            preview = r["content"][:60] + ("..." if len(r["content"]) > 60 else "")
            tag = "✓" if f"a{i}" in cited_ids else ""
            indicator += f"\n> - [archival:{r.get('source', '?')}] {preview} (score: {r['score']:.2f}) {tag}"
        for i, r in enumerate(recall_results):
            preview = r["content"][:60] + ("..." if len(r["content"]) > 60 else "")
            tag = "✓" if f"r{i}" in cited_ids else ""
            indicator += f"\n> - [recall:{r.get('role', '?')}] {preview} (score: {r['score']:.2f}) {tag}"
        response_text += indicator

    result = {
        "messages": [AIMessage(content=response_text)],
        "core_memory": core_memory.to_dict_with_history(),
        "conversation_summary": new_summary,
    }
    return result


def _format_archival_results(results: list) -> str:
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results):
        score = r.get("score", 0)
        parts.append(f"[a{i}] {r['content']} (relevance: {score:.2f})")
    return "\n\n".join(parts)


def _format_recall_results(results: list) -> str:
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results):
        score = r.get("score", 0)
        role = r.get("role", "unknown")
        content = r["content"][:300] + ("..." if len(r["content"]) > 300 else "")
        parts.append(f"[r{i}] [{role}] {content} (relevance: {score:.2f})")
    return "\n\n".join(parts)
