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
