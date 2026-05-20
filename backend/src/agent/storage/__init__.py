"""Storage initialization with lazy singleton pattern."""

from __future__ import annotations

import os

from agent.storage.embedding import DashScopeEmbeddings

_embeddings: DashScopeEmbeddings | None = None


def get_embeddings() -> DashScopeEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = DashScopeEmbeddings()
    return _embeddings


def get_db_url() -> str:
    return os.environ["DATABASE_URL"]
