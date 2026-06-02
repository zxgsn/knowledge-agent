"""Local embedding model wrapper for LangChain.

Uses sentence-transformers SentenceTransformer for bi-encoder embeddings.
Model: BAAI/bge-m3 (1024-dim, multilingual).
Checks backend/models/bge-m3/ first, falls back to HF hub download.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from threading import Lock

from langchain_core.embeddings import Embeddings

from agent.logger import get_logger
from agent.retry import with_retry

logger = get_logger(__name__)

# How often to log cache stats (every N encode calls)
_CACHE_LOG_INTERVAL = 100


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

        # Cache stats counters
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._encode_calls: int = 0
        self._client_lock = Lock()
        self._cache_lock = Lock()

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client

            import torch
            from sentence_transformers import SentenceTransformer

            local_dir = Path(__file__).resolve().parents[3] / "models" / self.model_name.split("/")[-1]
            if local_dir.is_dir() and (local_dir / "config.json").exists():
                model_path = str(local_dir)
                logger.info("Loading local embedding model: %s", model_path)
            else:
                model_path = self.model_name
                logger.info("Loading embedding model from HF: %s", model_path)

            # Force CPU and reduce memory usage to avoid segfault on Windows
            device = "cpu"
            if torch.cuda.is_available():
                try:
                    # Test if CUDA actually works
                    torch.tensor([1.0]).cuda()
                    device = "cuda"
                except Exception:
                    device = "cpu"
                    logger.debug("CUDA test failed, using CPU for embeddings")

            self._client = SentenceTransformer(
                model_path,
                device=device,
                model_kwargs={"low_cpu_mem_usage": True},
            )
            logger.info(
                "Embedding model loaded: dim=%d, device=%s",
                self._client.get_sentence_embedding_dimension(),
                device,
            )
        return self._client

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    @with_retry(
        max_retries=2,
        base_delay=1.0,
        max_delay=10.0,
        retryable_exceptions=(RuntimeError, OSError),
    )
    def _encode_batch(self, texts: list[str]) -> list:
        """Encode a batch of texts with retry on transient failures."""
        client = self._get_client()
        return client.encode(texts, normalize_embeddings=True)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        uncached_texts = []
        uncached_indices = []
        results: list[list[float] | None] = [None] * len(texts)

        with self._cache_lock:
            for i, t in enumerate(texts):
                key = self._cache_key(t)
                if key in self._cache:
                    results[i] = self._cache[key]
                    self._cache.move_to_end(key)
                    self._cache_hits += 1
                else:
                    uncached_texts.append(t)
                    uncached_indices.append(i)
                    self._cache_misses += 1

        if uncached_texts:
            embeddings = self._encode_batch(uncached_texts)
            with self._cache_lock:
                for idx, emb in zip(uncached_indices, embeddings):
                    emb_list = emb.tolist()
                    key = self._cache_key(texts[idx])
                    self._cache[key] = emb_list
                    self._cache.move_to_end(key)
                    results[idx] = emb_list
                    while len(self._cache) > self._cache_maxsize:
                        self._cache.popitem(last=False)

        # Periodically log cache hit rate
        with self._cache_lock:
            self._encode_calls += 1
            if self._encode_calls % _CACHE_LOG_INTERVAL == 0:
                total = self._cache_hits + self._cache_misses
                rate = (self._cache_hits / total * 100) if total > 0 else 0.0
                logger.info(
                    "Embedding cache stats: hits=%d, misses=%d, hit_rate=%.1f%%, cache_size=%d",
                    self._cache_hits,
                    self._cache_misses,
                    rate,
                    len(self._cache),
                )

        return results  # type: ignore[return-value]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._encode([text])[0]
