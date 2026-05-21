"""Response generation node."""

from __future__ import annotations

import asyncio
import sys

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agent.configuration import Configuration
from agent.memory.core_memory import CoreMemory
from agent.memory.tools import create_memory_tools
from agent.nodes.memory_manager import _sync_save_to_recall
from agent.prompts import ANSWER_PROMPT, SYSTEM_PROMPT, get_current_date
from agent.state import AgentState

MAX_TOOL_ROUNDS = 5


async def respond(state: AgentState, config: RunnableConfig) -> dict:
    """Generate the final response to the user.

    Supports tool calling for core memory editing. The LLM can call
    core_memory_replace/insert/rethink/view tools in a loop before
    producing its final text response.
    """
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(
        model=configurable.llm_model,
        base_url=configurable.llm_base_url,
        api_key=configurable.llm_api_key,
        temperature=0.5,
        http_async_client=httpx.AsyncClient(proxy=None),
        extra_body={"thinking": {"type": "enabled"}},
    )

    core_memory = CoreMemory.from_dict(state.get("core_memory", {}))
    summaries = "\n\n---\n\n".join(state.get("web_research_result", []))
    archival_context = _format_archival_results(state.get("archival_results", []))

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
                messages_for_llm.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                messages_for_llm.append({"role": "assistant", "content": msg.content})
        if archival_context:
            messages_for_llm.append({
                "role": "user",
                "content": f"[Relevant knowledge from archival memory]\n{archival_context}",
            })

    # Bind core memory tools so the LLM can read/edit memory
    tools = create_memory_tools(core_memory, archival_memory=None)
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
                result = tool.invoke(tc["args"])
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
        compress_llm = ChatOpenAI(
            model=configurable.llm_model,
            base_url=configurable.llm_base_url,
            api_key=configurable.llm_api_key,
            temperature=0.2,
            http_async_client=httpx.AsyncClient(proxy=None),
            extra_body={"thinking": {"type": "enabled"}},
        )
        for label in needs_compression:
            await core_memory.compress_block(label, compress_llm)

    # Save AI response to recall memory
    try:
        await asyncio.to_thread(_sync_save_to_recall, "assistant", response.content)
    except Exception as e:
        print(f"[responder] Failed to save to recall: {e}", file=sys.stderr)

    result = {
        "messages": [AIMessage(content=response.content)],
        "core_memory": core_memory.to_dict(),
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
