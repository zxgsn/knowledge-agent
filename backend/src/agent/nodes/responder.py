"""Response generation node."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agent.configuration import Configuration
from agent.memory.core_memory import CoreMemory
from agent.prompts import ANSWER_PROMPT, SYSTEM_PROMPT, get_current_date
from agent.state import AgentState


async def respond(state: AgentState, config: RunnableConfig) -> dict:
    """Generate the final response to the user."""
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(
        model=configurable.llm_model,
        base_url=configurable.llm_base_url,
        api_key=configurable.llm_api_key,
        temperature=0.5,
    )

    core_memory = CoreMemory.from_dict(state.get("core_memory", {}))
    research_topic = _get_research_topic(state["messages"])
    summaries = "\n\n---\n\n".join(state.get("web_research_result", []))
    archival_context = _format_archival_results(state.get("archival_results", []))

    system = SYSTEM_PROMPT.format(
        current_date=get_current_date(),
        memory_blocks=core_memory.compile(),
    )

    if state.get("mode") == "research" and summaries:
        user_content = ANSWER_PROMPT.format(
            research_topic=research_topic,
            summaries=summaries,
            archival_context=archival_context,
        )
    elif archival_context:
        user_content = (
            f"{research_topic}\n\n"
            f"[Relevant knowledge from your archival memory]\n{archival_context}"
        )
    else:
        user_content = research_topic

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    response = await llm.ainvoke(messages)

    # Auto-compress blocks that are approaching their character limit
    needs_compression = core_memory.get_blocks_needing_compression()
    if needs_compression:
        compress_llm = ChatOpenAI(
            model=configurable.llm_model,
            base_url=configurable.llm_base_url,
            api_key=configurable.llm_api_key,
            temperature=0.2,
        )
        for label in needs_compression:
            await core_memory.compress_block(label, compress_llm)

    result = {"messages": [AIMessage(content=response.content)]}
    if needs_compression:
        result["core_memory"] = core_memory.to_dict()
    return result


def _get_research_topic(messages: list) -> str:
    parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            parts.append(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            parts.append(msg.content[:200])
    return "\n".join(parts) if parts else ""


def _format_archival_results(results: list) -> str:
    if not results:
        return ""
    parts = []
    for r in results:
        score = r.get("score", 0)
        if score >= 0.3:
            parts.append(f"- {r['content']} (relevance: {score:.2f})")
    return "\n\n".join(parts) if parts else ""
