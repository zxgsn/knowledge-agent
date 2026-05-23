"""Response generation node."""

from __future__ import annotations

import asyncio
import sys

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.db import save_to_recall
from agent.memory.core_memory import CoreMemory
from agent.memory.tools import create_memory_tools
from agent.prompts import ANSWER_PROMPT, SYSTEM_PROMPT, get_current_date
from agent.state import AgentState
from agent.utils import get_llm

MAX_TOOL_ROUNDS = 5


async def respond(state: AgentState, config: RunnableConfig) -> dict:
    """Generate the final response to the user.

    Supports tool calling for core memory editing. The LLM can call
    core_memory_replace/insert/rethink/view tools in a loop before
    producing its final text response.
    """
    configurable = Configuration.from_runnable_config(config)
    llm = get_llm(configurable, temperature=0.5)

    core_memory = CoreMemory.from_dict(state.get("core_memory", {}))
    summaries = "\n\n---\n\n".join(state.get("web_research_result", []))
    archival_context = _format_archival_results(state.get("archival_results", []))
    recall_context = _format_recall_results(state.get("recall_results", []))

    system = SYSTEM_PROMPT.format(
        current_date=get_current_date(),
        memory_blocks=core_memory.compile(),
    )

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
        for msg in state["messages"][-20:]:
            if isinstance(msg, HumanMessage):
                content = msg.content
                # Replace PDF base64 data with a short summary
                import re as _re
                pdf_match = _re.match(
                    r"\[UPLOAD_PDF:(.+?)\].+?\[/UPLOAD_PDF\]",
                    content,
                    _re.DOTALL,
                )
                if pdf_match:
                    content = f"[Uploaded PDF: {pdf_match.group(1)}. Document has been ingested into archival memory.]"
                messages_for_llm.append({"role": "user", "content": content})
            elif isinstance(msg, AIMessage) and msg.content:
                messages_for_llm.append({"role": "assistant", "content": msg.content})
        if archival_context or recall_context:
            context_parts = []
            if archival_context:
                context_parts.append(f"[Archival Memory]\n{archival_context}")
            if recall_context:
                context_parts.append(f"[Conversation History]\n{recall_context}")
            messages_for_llm.append({
                "role": "user",
                "content": "\n\n".join(context_parts),
            })

    # Bind core memory tools so the LLM can read/edit memory
    tools = create_memory_tools(core_memory, enable_archival=True)
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    # Tool-calling loop: LLM may call tools multiple times before final answer
    response = await llm_with_tools.ainvoke(messages_for_llm)

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

        response = await llm_with_tools.ainvoke(messages_for_llm)

    # Auto-compress blocks that are approaching their character limit
    needs_compression = core_memory.get_blocks_needing_compression()
    if needs_compression:
        compress_llm = get_llm(configurable, temperature=0.2)
        for label in needs_compression:
            await core_memory.compress_block(label, compress_llm)

    # Save AI response to recall memory (thread-isolated)
    thread_id = state.get("thread_id", "default")
    try:
        await asyncio.to_thread(save_to_recall, "assistant", response.content, thread_id=thread_id)
    except Exception as e:
        print(f"[responder] Failed to save to recall: {e}", file=sys.stderr)

    # Add memory indicator when archival/recall results were used
    response_text = response.content
    archival_used = [r for r in state.get("archival_results", []) if r.get("score", 0) >= 0.3]
    recall_used = [r for r in state.get("recall_results", []) if r.get("score", 0) >= 0.3]
    if archival_used or recall_used:
        total = len(archival_used) + len(recall_used)
        indicator = f"\n\n---\n> 📚 **从记忆中检索到 {total} 条相关内容**"
        for r in archival_used[:2]:
            preview = r["content"][:60] + ("..." if len(r["content"]) > 60 else "")
            indicator += f"\n> - [archival:{r.get('source', '?')}] {preview} (score: {r['score']:.2f})"
        for r in recall_used[:2]:
            preview = r["content"][:60] + ("..." if len(r["content"]) > 60 else "")
            indicator += f"\n> - [recall:{r.get('role', '?')}] {preview} (score: {r['score']:.2f})"
        response_text += indicator

    result = {
        "messages": [AIMessage(content=response_text)],
        "core_memory": core_memory.to_dict_with_history(),
    }
    return result


def _format_archival_results(results: list) -> str:
    if not results:
        return ""
    parts = []
    for r in results:
        score = r.get("score", 0)
        if score >= 0.3:
            parts.append(f"- {r['content']} (relevance: {score:.2f})")
    return "\n\n".join(parts) if parts else ""


def _format_recall_results(results: list) -> str:
    if not results:
        return ""
    parts = []
    for r in results:
        score = r.get("score", 0)
        if score >= 0.3:
            role = r.get("role", "unknown")
            content = r["content"][:300] + ("..." if len(r["content"]) > 300 else "")
            parts.append(f"- [{role}] {content} (relevance: {score:.2f})")
    return "\n\n".join(parts) if parts else ""
