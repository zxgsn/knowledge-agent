"""Tests for agent.storage.embedding — LRU cache behavior."""

from unittest.mock import MagicMock, patch

import numpy as np

from agent.storage.embedding import LocalEmbeddings


def _make_embedding(**kwargs):
    """Create a LocalEmbeddings with mocked sentence-transformers client."""
    emb = LocalEmbeddings(model="test-model", **kwargs)
    mock_client = MagicMock()
    # encode returns numpy array with shape (batch, 4)
    mock_client.encode.side_effect = lambda texts, normalize_embeddings=True: np.array(
        [[float(i)] * 4 for i in range(len(texts))]
    )
    mock_client.get_sentence_embedding_dimension.return_value = 4
    emb._client = mock_client
    return emb


class TestEmbeddingCacheInit:
    def test_cache_attributes_exist(self):
        emb = _make_embedding()
        assert hasattr(emb, "_cache")
        assert hasattr(emb, "_cache_maxsize")
        assert emb._cache_maxsize == 512

    def test_cache_starts_empty(self):
        emb = _make_embedding()
        assert len(emb._cache) == 0


class TestCacheKey:
    def test_deterministic(self):
        emb = _make_embedding()
        assert emb._cache_key("hello") == emb._cache_key("hello")

    def test_different_texts_different_keys(self):
        emb = _make_embedding()
        assert emb._cache_key("hello") != emb._cache_key("world")

    def test_returns_hex_string(self):
        emb = _make_embedding()
        key = emb._cache_key("test")
        assert len(key) == 32  # MD5 hex length
        assert all(c in "0123456789abcdef" for c in key)


class TestEncodeCache:
    def test_cache_miss_calls_encode(self):
        emb = _make_embedding()
        result = emb._encode(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 4
        assert len(emb._cache) == 1

    def test_cache_hit_skips_encode(self):
        emb = _make_embedding()
        call_count_before = emb._client.encode.call_count
        emb._encode(["hello"])
        emb._encode(["hello"])
        # encode should be called only once (first call), second is cache hit
        assert emb._client.encode.call_count == call_count_before + 1

    def test_partial_cache_hit(self):
        emb = _make_embedding()
        emb._encode(["hello"])  # cache "hello"
        call_count_before = emb._client.encode.call_count
        result = emb._encode(["hello", "world"])
        # Should encode only "world" (1 new text)
        assert emb._client.encode.call_count == call_count_before + 1
        assert len(result) == 2
        assert len(emb._cache) == 2

    def test_eviction_at_maxsize(self):
        emb = _make_embedding()
        emb._cache_maxsize = 3

        for i in range(5):
            emb._encode([f"text{i}"])

        assert len(emb._cache) == 3
        assert emb._cache_key("text0") not in emb._cache
        assert emb._cache_key("text1") not in emb._cache
        assert emb._cache_key("text2") in emb._cache

    def test_lru_ordering_on_hit(self):
        emb = _make_embedding()
        emb._cache_maxsize = 3

        # Fill cache with 3 items
        for i in range(3):
            emb._encode([f"text{i}"])

        # Access text0 to move it to end (most recently used)
        emb._encode(["text0"])

        # Add a new item — should evict text1 (least recently used)
        emb._encode(["text3"])

        assert emb._cache_key("text0") in emb._cache  # was accessed, so kept
        assert emb._cache_key("text1") not in emb._cache  # LRU evicted
        assert emb._cache_key("text3") in emb._cache


class TestEmbedQuery:
    def test_returns_single_vector(self):
        emb = _make_embedding()
        result = emb.embed_query("query")
        assert isinstance(result, list)
        assert len(result) == 4

    def test_caches_query(self):
        emb = _make_embedding()
        emb.embed_query("query")
        call_count_after = emb._client.encode.call_count
        emb.embed_query("query")
        assert emb._client.encode.call_count == call_count_after  # no new call


class TestEmbedDocuments:
    def test_batch_encode(self):
        emb = _make_embedding()
        result = emb.embed_documents(["a", "b", "c"])
        assert len(result) == 3
        assert all(len(v) == 4 for v in result)

    def test_large_batch(self):
        emb = _make_embedding()
        texts = [f"doc{i}" for i in range(15)]
        result = emb.embed_documents(texts)
        assert len(result) == 15
