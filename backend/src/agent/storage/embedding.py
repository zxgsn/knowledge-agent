"""Local embedding model wrapper for LangChain.

Uses sentence-transformers SentenceTransformer for bi-encoder embeddings.
Model: BAAI/bge-m3 (1024-dim, multilingual).
Checks backend/models/bge-m3/ first, falls back to HF hub download.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections import OrderedDict
from pathlib import Path

from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """Local sentence-transformers embedding model.

    Compatible with LangChain's Embeddings interface.
    Includes an in-memory LRU cache to avoid redundant encode calls.
    """

    def __init__(
        self,
        model: str | None = None,
    ):
        self.model_name = model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_maxsize = 512
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        import torch
        from sentence_transformers import SentenceTransformer

        local_dir = Path(__file__).resolve().parents[3] / "models" / self.model_name.split("/")[-1]
        if local_dir.is_dir() and (local_dir / "config.json").exists():
            model_path = str(local_dir)
            print(f"[embedding] Loading local model: {model_path}", file=sys.stderr)
        else:
            model_path = self.model_name
            print(f"[embedding] Loading model from HF: {model_path}", file=sys.stderr)

        # Force CPU and reduce memory usage to avoid segfault on Windows
        device = "cpu"
        if torch.cuda.is_available():
            try:
                # Test if CUDA actually works
                torch.tensor([1.0]).cuda()
                device = "cuda"
            except Exception:
                device = "cpu"

        self._client = SentenceTransformer(model_path, device=device)
        print(f"[embedding] Model loaded. dim={self._client.get_sentence_embedding_dimension()}", file=sys.stderr)
        return self._client

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _encode(self, texts: list[str]) -> list[list[float]]:
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
            client = self._get_client()
            embeddings = client.encode(uncached_texts, normalize_embeddings=True)
            for idx, emb in zip(uncached_indices, embeddings):
                emb_list = emb.tolist()
                key = self._cache_key(texts[idx])
                self._cache[key] = emb_list
                self._cache.move_to_end(key)
                results[idx] = emb_list
                while len(self._cache) > self._cache_maxsize:
                    self._cache.popitem(last=False)

        return results  # type: ignore[return-value]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._encode([text])[0]
