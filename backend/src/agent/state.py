from __future__ import annotations

import operator

from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict


class AgentState(TypedDict):
    """Main agent state."""

    messages: Annotated[list, add_messages]
    core_memory: dict[str, str]  # {label: value} - persisted core memory blocks
    archival_results: Annotated[list, operator.add]  # results from archival search
    search_results: Annotated[list, operator.add]  # results from web search
    sources: Annotated[list, operator.add]  # collected source URLs
    research_loop_count: int
    mode: str  # "chat" | "research" | "memory_edit" | "ingest"
    doc_source: str  # URL or text content for document ingestion
    doc_source_type: str  # "url" | "text"
    ingest_result: str  # Result message from document ingestion
    # Research pipeline fields
    search_query: list[str]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: list[str]
    max_research_loops: int
