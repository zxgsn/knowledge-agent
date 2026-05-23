"""Tests for agent.storage.embedding — LRU cache behavior."""

from unittest.mock import MagicMock, patch

from agent.storage.embedding import DashScopeEmbeddings


def _make_embedding(**kwargs):
    """Create a DashScopeEmbeddings with mocked API."""
    emb = DashScopeEmbeddings(api_key="test-key", **kwargs)
    return emb


def _mock_api_response(texts):
    """Create a mock DashScope API response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.output = {
        "embeddings": [
            {"embedding": [float(i)] * 4}
            for i in range(len(texts))
        ]
    }
    return resp


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


class TestCallApiCache:
    @patch("dashscope.TextEmbedding.call")
    def test_cache_miss_calls_api(self, mock_call):
        mock_call.return_value = _mock_api_response(["hello"])
        emb = _make_embedding()
        result = emb._call_api(["hello"])
        assert mock_call.call_count == 1
        assert len(result) == 1
        assert len(emb._cache) == 1

    @patch("dashscope.TextEmbedding.call")
    def test_cache_hit_skips_api(self, mock_call):
        mock_call.return_value = _mock_api_response(["hello"])
        emb = _make_embedding()
        emb._call_api(["hello"])  # fills cache
        emb._call_api(["hello"])  # should hit cache
        assert mock_call.call_count == 1  # API called only once

    @patch("dashscope.TextEmbedding.call")
    def test_partial_cache_hit(self, mock_call):
        mock_call.return_value = _mock_api_response(["hello"])
        emb = _make_embedding()
        emb._call_api(["hello"])  # cache "hello"

        mock_call.return_value = _mock_api_response(["world"])
        result = emb._call_api(["hello", "world"])

        # API called twice total, second call only for "world"
        assert mock_call.call_count == 2
        assert len(result) == 2
        assert len(emb._cache) == 2

    @patch("dashscope.TextEmbedding.call")
    def test_eviction_at_maxsize(self, mock_call):
        emb = _make_embedding()
        emb._cache_maxsize = 3

        for i in range(5):
            mock_call.return_value = _mock_api_response([f"text{i}"])
            emb._call_api([f"text{i}"])

        assert len(emb._cache) == 3
        # Oldest entries should be evicted
        assert emb._cache_key("text0") not in emb._cache
        assert emb._cache_key("text1") not in emb._cache
        assert emb._cache_key("text2") in emb._cache

    @patch("dashscope.TextEmbedding.call")
    def test_lru_ordering_on_hit(self, mock_call):
        emb = _make_embedding()
        emb._cache_maxsize = 3

        # Fill cache with 3 items
        for i in range(3):
            mock_call.return_value = _mock_api_response([f"text{i}"])
            emb._call_api([f"text{i}"])

        # Access text0 to move it to end (most recently used)
        emb._call_api(["text0"])

        # Add a new item — should evict text1 (least recently used)
        mock_call.return_value = _mock_api_response(["text3"])
        emb._call_api(["text3"])

        assert emb._cache_key("text0") in emb._cache  # was accessed, so kept
        assert emb._cache_key("text1") not in emb._cache  # LRU evicted
        assert emb._cache_key("text3") in emb._cache


class TestEmbedQuery:
    @patch("dashscope.TextEmbedding.call")
    def test_returns_single_vector(self, mock_call):
        mock_call.return_value = _mock_api_response(["query"])
        emb = _make_embedding()
        result = emb.embed_query("query")
        assert isinstance(result, list)
        assert len(result) == 4

    @patch("dashscope.TextEmbedding.call")
    def test_caches_query(self, mock_call):
        mock_call.return_value = _mock_api_response(["query"])
        emb = _make_embedding()
        emb.embed_query("query")
        emb.embed_query("query")
        assert mock_call.call_count == 1


class TestEmbedDocuments:
    @patch("dashscope.TextEmbedding.call")
    def test_batch_within_limit(self, mock_call):
        mock_call.return_value = _mock_api_response(["a", "b", "c"])
        emb = _make_embedding()
        result = emb.embed_documents(["a", "b", "c"])
        assert len(result) == 3
        assert mock_call.call_count == 1

    @patch("dashscope.TextEmbedding.call")
    def test_batch_exceeds_limit(self, mock_call):
        # DashScope limit is 10 per batch
        texts = [f"doc{i}" for i in range(15)]
        mock_call.return_value = _mock_api_response(texts[:10])
        emb = _make_embedding()
        emb.embed_documents(texts)
        # Should call API twice: 10 + 5
        assert mock_call.call_count == 2
