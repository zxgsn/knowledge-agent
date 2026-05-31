from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


def _load_yaml_config() -> dict[str, Any]:
    """Load feature config from config.yaml (next to backend/)."""
    try:
        import yaml
    except ImportError:
        return {}

    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    if not config_path.is_file():
        return {}

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data.get("features", {})


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
        default="BAAI/bge-m3",
        metadata={"description": "Embedding model name (sentence-transformers)."},
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
        default=True,
        metadata={"description": "Enable HyDE (Hypothetical Document Embeddings) for archival search."},
    )
    memory_query_rewrite_enabled: bool = Field(
        default=True,
        metadata={"description": "Rewrite ambiguous queries using conversation history before memory search."},
    )
    mmr_enabled: bool = Field(
        default=True,
        metadata={"description": "Enable MMR deduplication in search results."},
    )
    mmr_lambda: float = Field(
        default=0.7,
        metadata={"description": "MMR lambda parameter: 1.0=pure relevance, 0.0=pure diversity."},
    )
    proactive_memory_enabled: bool = Field(
        default=True,
        metadata={"description": "Proactively push relevant memories at conversation start."},
    )
    proactive_memory_turns: int = Field(
        default=3,
        metadata={"description": "Number of initial turns to activate proactive memory."},
    )
    conflict_confidence_threshold: float = Field(
        default=0.7,
        metadata={"description": "Confidence threshold below which conflicts are queued for human review."},
    )
    structured_extraction: bool = Field(
        default=True,
        metadata={"description": "Extract entities and temporal references from conversation facts."},
    )
    document_enrichment: bool = Field(
        default=True,
        metadata={"description": "LLM-assisted extraction of summary/entities/keywords during ingestion."},
    )
    summarize_threshold: int = Field(
        default=20,
        metadata={"description": "Summarize older conversation turns when message count exceeds this threshold."},
    )
    summarize_keep_recent: int = Field(
        default=10,
        metadata={"description": "Number of recent messages to keep as raw text when summarizing."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        # Priority: env vars > configurable dict > config.yaml > field defaults
        yaml_config = _load_yaml_config()

        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        raw_values: dict[str, Any] = {}
        for name in cls.model_fields.keys():
            env_val = os.environ.get(name.upper())
            if env_val is not None:
                raw_values[name] = env_val
            elif name in configurable:
                raw_values[name] = configurable[name]
            elif name in yaml_config:
                raw_values[name] = yaml_config[name]

        values = {k: v for k, v in raw_values.items() if v is not None}
        return cls(**values)
