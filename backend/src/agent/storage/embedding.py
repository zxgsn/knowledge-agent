"""DashScope embedding wrapper for LangChain."""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict

from langchain_core.embeddings import Embeddings


class DashScopeEmbeddings(Embeddings):
    """Alibaba Cloud DashScope text embedding model.

    Uses the dashscope SDK to call text-embedding-v3 or similar models.
    Compatible with LangChain's Embeddings interface.
    Includes an in-memory LRU cache to avoid redundant API calls.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model or os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_maxsize = 512

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        uncached_texts = []
        uncached_indices = []
        results: list[list[float] | None] = [None] * len(texts)

        for i, t in enumerate(texts):
            key = self._cache_key(t)
            if key in self._cache:
                results[i] = self._cache[key]
                self._cache.move_to_end(key)
            else:
                uncached_texts.append(t)
                uncached_indices.append(i)

        if uncached_texts:
            import dashscope

            dashscope.api_key = self.api_key
            resp = dashscope.TextEmbedding.call(
                model=self.model,
                input=uncached_texts,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"DashScope embedding failed: {resp.code} - {resp.message}"
                )
            embeddings = [item["embedding"] for item in resp.output["embeddings"]]
            for idx, emb in zip(uncached_indices, embeddings):
                key = self._cache_key(texts[idx])
                self._cache[key] = emb
                self._cache.move_to_end(key)
                results[idx] = emb
                while len(self._cache) > self._cache_maxsize:
                    self._cache.popitem(last=False)

        return results  # type: ignore[return-value]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents. Handles batching if needed."""
        # DashScope batch limit is 10 per request
        all_embeddings: list[list[float]] = []
        chunk_size = 10
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            all_embeddings.extend(self._call_api(chunk))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._call_api([text])[0]
