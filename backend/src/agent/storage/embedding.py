"""DashScope embedding wrapper for LangChain."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.embeddings import Embeddings


class DashScopeEmbeddings(Embeddings):
    """Alibaba Cloud DashScope text embedding model.

    Uses the dashscope SDK to call text-embedding-v3 or similar models.
    Compatible with LangChain's Embeddings interface.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model or os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        import dashscope

        dashscope.api_key = self.api_key
        resp = dashscope.TextEmbedding.call(
            model=self.model,
            input=texts,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"DashScope embedding failed: {resp.code} - {resp.message}"
            )
        return [item["embedding"] for item in resp.output["embeddings"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents. Handles batching if needed."""
        # DashScope supports batch embedding, but with limits
        # Process in chunks of 25 to be safe
        all_embeddings: list[list[float]] = []
        chunk_size = 25
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            all_embeddings.extend(self._call_api(chunk))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._call_api([text])[0]
