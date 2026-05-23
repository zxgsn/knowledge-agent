"""Knowledge Agent - Main LangGraph graph.

Flow:
  START → route_intent
    ├── "ingest" → ingest_document → [has_question?]
    │     ├── yes → recall_memory → respond → memory_pipeline → [consolidate?] → END
    │     └── no  → respond → memory_pipeline → [consolidate?] → END
    └── (其他)   → recall_memory → evaluate_recall
                     ├── (memory sufficient) → respond → memory_pipeline → [consolidate?] → END
                     └── (memory insufficient) → generate_query → [web_research × N] → reflection
                           ├── (gaps) → [web_research] → reflection (loop)
                           └── (sufficient) → save_to_archival → respond → memory_pipeline → [consolidate?] → END
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
    evaluate_recall,
    ingest_document_node,
    recall_memory,
    route_intent,
    save_to_archival,
)
from agent.nodes.memory_pipeline import consolidate_memory, memory_pipeline
from agent.nodes.researcher import generate_query, reflection, web_research
from agent.nodes.responder import respond
from agent.state import AgentState

load_dotenv()


# --- Conditional edge functions ---

def route_after_intent(state: AgentState, config: RunnableConfig) -> str:
    """Route to the appropriate sub-graph based on intent."""
    mode = state.get("mode", "chat")
    if mode == "ingest":
        return "ingest_document"
    if mode in ("chat", "memory_edit") or not state.get("need_recall", True):
        # Proactive memory: route early chat turns through recall_memory
        configurable = Configuration.from_runnable_config(config)
        turn_count = state.get("turn_count", 0)
        if (configurable.proactive_memory_enabled
                and turn_count < configurable.proactive_memory_turns):
            return "recall_memory"
        return "respond"
    return "recall_memory"


def route_after_ingest(state: AgentState) -> str:
    """Route after ingest_document: if user asked a question, search memory first."""
    if state.get("ingest_question"):
        return "recall_memory"
    return "respond"


def route_after_evaluation(state: AgentState) -> str:
    """Route after evaluate_recall based on memory sufficiency."""
    if state.get("memory_sufficient"):
        return "respond"
    return "generate_query"


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


def route_after_pipeline(state: AgentState, config: RunnableConfig) -> str:
    """Decide whether to run consolidation after memory pipeline."""
    configurable = Configuration.from_runnable_config(config)
    turn_count = state.get("turn_count", 0)
    if turn_count > 0 and turn_count % configurable.memory_consolidation_interval == 0:
        return "consolidate_memory"
    return END


# --- Build the graph ---

builder = StateGraph(AgentState, config_schema=Configuration)

# Add nodes
builder.add_node("route_intent", route_intent)
builder.add_node("recall_memory", recall_memory)
builder.add_node("evaluate_recall", evaluate_recall)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("ingest_document", ingest_document_node)
builder.add_node("save_to_archival", save_to_archival)
builder.add_node("respond", respond)
builder.add_node("memory_pipeline", memory_pipeline)
builder.add_node("consolidate_memory", consolidate_memory)

# Wire edges
builder.add_edge(START, "route_intent")
builder.add_conditional_edges(
    "route_intent", route_after_intent,
    ["ingest_document", "recall_memory", "respond"],
)
builder.add_edge("recall_memory", "evaluate_recall")
builder.add_conditional_edges(
    "evaluate_recall", route_after_evaluation,
    ["generate_query", "respond"],
)
builder.add_conditional_edges("generate_query", continue_to_web_research, ["web_research"])
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "save_to_archival"]
)
builder.add_conditional_edges(
    "ingest_document", route_after_ingest, ["recall_memory", "respond"]
)
builder.add_edge("save_to_archival", "respond")
builder.add_edge("respond", "memory_pipeline")
builder.add_conditional_edges(
    "memory_pipeline", route_after_pipeline,
    ["consolidate_memory", END],
)
builder.add_edge("consolidate_memory", END)

graph = builder.compile(name="knowledge-agent")
