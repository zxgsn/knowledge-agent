"""Knowledge Agent - Main LangGraph graph.

Flow:
  START → route_intent
    ├── "chat" → respond → END
    ├── "research" → generate_query → web_research → reflection
    │     ├── (gaps) → web_research → reflection (loop)
    │     └── (sufficient) → save_to_archival → respond → END
    ├── "memory_edit" → respond → END
    └── "ingest" → ingest_document → save_to_archival → respond → END
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent.configuration import Configuration
from agent.nodes.memory_manager import (
    ingest_document_node,
    recall_memory,
    route_intent,
    save_to_archival,
)
from agent.nodes.researcher import generate_query, reflection, web_research
from agent.nodes.responder import respond
from agent.state import AgentState

load_dotenv()


# --- Conditional edge functions ---

def route_after_intent(state: AgentState) -> str:
    """Route to the appropriate sub-graph based on intent."""
    mode = state.get("mode", "chat")
    if mode == "research":
        return "generate_query"
    elif mode == "ingest":
        return "ingest_document"
    elif mode == "recall":
        return "recall_memory"
    else:
        # chat and memory_edit both go to respond
        return "respond"


def continue_to_web_research(state: AgentState):
    """Fan out: one web_research node per search query."""
    queries = state.get("search_query", [])
    if not queries:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                queries = [msg.content]
                break
    return [
        Send("web_research", {"search_query": q, "id": str(i)})
        for i, q in enumerate(queries)
    ]


def evaluate_research(state: AgentState, config: RunnableConfig):
    """Decide whether to continue researching or finalize."""
    configurable = Configuration.from_runnable_config(config)
    max_loops = state.get("max_research_loops", configurable.max_research_loops)
    loop_count = state.get("research_loop_count", 0)

    if state.get("is_sufficient") or loop_count >= max_loops:
        return "save_to_archival"

    follow_ups = state.get("follow_up_queries", [])
    if not follow_ups:
        return "save_to_archival"

    return [
        Send("web_research", {"search_query": q, "id": f"followup_{i}"})
        for i, q in enumerate(follow_ups)
    ]


# --- Build the graph ---

builder = StateGraph(AgentState, config_schema=Configuration)

# Add nodes
builder.add_node("route_intent", route_intent)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("ingest_document", ingest_document_node)
builder.add_node("save_to_archival", save_to_archival)
builder.add_node("recall_memory", recall_memory)
builder.add_node("respond", respond)

# Wire edges
builder.add_edge(START, "route_intent")
builder.add_conditional_edges(
    "route_intent", route_after_intent,
    ["generate_query", "ingest_document", "recall_memory", "respond"],
)
builder.add_conditional_edges("generate_query", continue_to_web_research, ["web_research"])
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "save_to_archival"]
)
builder.add_edge("ingest_document", "save_to_archival")
builder.add_edge("save_to_archival", "respond")
builder.add_edge("recall_memory", "respond")
builder.add_edge("respond", END)

graph = builder.compile(name="knowledge-agent")
