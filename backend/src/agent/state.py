from __future__ import annotations

import operator

from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict


class AgentState(TypedDict):
    """Main agent state."""

    # --- I/O ---
    messages: Annotated[list, add_messages]
    mode: str  # "chat" | "research" | "memory_edit" | "ingest" | "recall"
    need_recall: bool  # whether to query archival memory before responding
    core_memory: dict[str, str]  # {label: value} - persisted core memory blocks

    # --- Research subgraph ---
    search_query: list[str]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    research_loop_count: int
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: list[str]
    max_research_loops: int

    # --- Archival / Ingest ---
    archival_results: Annotated[list, operator.add]
    doc_source: str
    doc_source_type: str
    ingest_result: str

    # --- Memory Pipeline ---
    turn_count: int
    pipeline_facts: list[dict]

    # --- Memory Evaluation ---
    memory_sufficient: bool      # 记忆评估结果：是否足够回答问题
    memory_evaluation: str       # 评估理由

    # --- Memory Operations Log ---
    memory_operations: Annotated[list, operator.add]  # 记忆操作日志
