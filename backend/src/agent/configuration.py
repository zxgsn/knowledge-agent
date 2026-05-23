from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Runtime configuration for the knowledge agent."""

    llm_model: str = Field(
        default="gpt-4o-mini",
        metadata={"description": "LLM model name."},
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        metadata={"description": "LLM API base URL."},
    )
    llm_api_key: str = Field(
        default="",
        metadata={"description": "LLM API key."},
    )
    embedding_model: str = Field(
        default="text-embedding-v3",
        metadata={"description": "DashScope embedding model name."},
    )
    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "Number of initial search queries to generate."},
    )
    max_research_loops: int = Field(
        default=2,
        metadata={"description": "Maximum number of research loops."},
    )
    memory_consolidation_interval: int = Field(
        default=10,
        metadata={"description": "Run memory consolidation every N turns."},
    )
    memory_dedup_threshold: float = Field(
        default=0.8,
        metadata={"description": "Cosine similarity threshold for fact deduplication."},
    )
    memory_selective_enabled: bool = Field(
        default=True,
        metadata={"description": "Enable selective memory capture (judgment + extraction)."},
    )
    archival_cleanup_days: int = Field(
        default=30,
        metadata={"description": "Days before ingested entries are eligible for cleanup."},
    )
    archival_max_entries: int = Field(
        default=1000,
        metadata={"description": "Max entries per namespace before forced cleanup."},
    )
    rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        metadata={"description": "Cross-encoder model for re-ranking search results."},
    )
    rerank_enabled: bool = Field(
        default=True,
        metadata={"description": "Enable cross-encoder re-ranking."},
    )
    chunk_strategy: str = Field(
        default="fixed",
        metadata={"description": "Chunking strategy: 'fixed' or 'semantic'."},
    )
    chunk_similarity_threshold: float = Field(
        default=0.5,
        metadata={"description": "Cosine similarity threshold for semantic chunking boundaries."},
    )
    chunk_min_size: int = Field(
        default=200,
        metadata={"description": "Minimum chunk size in characters for semantic chunking."},
    )
    chunk_max_size: int = Field(
        default=1500,
        metadata={"description": "Maximum chunk size in characters for semantic chunking."},
    )
    hyde_enabled: bool = Field(
        default=False,
        metadata={"description": "Enable HyDE (Hypothetical Document Embeddings) for archival search."},
    )
    memory_query_rewrite_enabled: bool = Field(
        default=False,
        metadata={"description": "Rewrite ambiguous queries using conversation history before memory search."},
    )
    mmr_enabled: bool = Field(
        default=False,
        metadata={"description": "Enable MMR deduplication in search results."},
    )
    mmr_lambda: float = Field(
        default=0.5,
        metadata={"description": "MMR lambda parameter: 1.0=pure relevance, 0.0=pure diversity."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }
        values = {k: v for k, v in raw_values.items() if v is not None}
        return cls(**values)
