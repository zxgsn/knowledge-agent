"""Tests for agent.db._mmr_rerank and _cosine_similarity."""

from unittest.mock import MagicMock, patch

from agent.db import _cosine_similarity, _mmr_rerank


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        assert _cosine_similarity(a, a) == 1.0

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == -1.0

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_partial_similarity(self):
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / (2.0 ** 0.5)
        assert abs(_cosine_similarity(a, b) - expected) < 1e-9


class TestMmrRerank:
    def _make_results(self, contents, scores):
        return [
            {"content": c, "score": s, "metadata": {}}
            for c, s in zip(contents, scores)
        ]

    @patch("agent.storage.get_embeddings")
    def test_single_result_unchanged(self, mock_get_emb):
        results = self._make_results(["hello"], [0.9])
        out = _mmr_rerank([1.0, 0.0], results)
        assert len(out) == 1
        assert out[0]["content"] == "hello"

    @patch("agent.storage.get_embeddings")
    def test_empty_results(self, mock_get_emb):
        out = _mmr_rerank([1.0, 0.0], [])
        assert out == []

    @patch("agent.storage.get_embeddings")
    def test_diversity_over_relevance(self, mock_get_emb):
        """When lambda=0, MMR should maximize diversity (minimize similarity to selected)."""
        # Three results: A and B are identical (high sim), C is different
        emb = MagicMock()
        # Content: A, B identical embeddings, C orthogonal
        emb.embed_query = MagicMock(side_effect=lambda t: {
            "A": [1.0, 0.0],
            "B": [1.0, 0.0],  # identical to A
            "C": [0.0, 1.0],  # orthogonal to A
        }.get(t[:1], [0.5, 0.5]))
        mock_get_emb.return_value = emb

        results = self._make_results(["A", "B", "C"], [0.9, 0.85, 0.8])
        out = _mmr_rerank([1.0, 0.0], results, lambda_param=0.0)

        # With lambda=0 (pure diversity), after selecting A first,
        # C should be preferred over B because C is more different from A
        assert out[0]["content"] == "A"  # highest score, selected first
        assert out[1]["content"] == "C"  # more diverse than B

    @patch("agent.storage.get_embeddings")
    def test_pure_relevance_order(self, mock_get_emb):
        """When lambda=1, MMR should keep original relevance order."""
        emb = MagicMock()
        emb.embed_query = MagicMock(side_effect=lambda t: [1.0, 0.0])
        mock_get_emb.return_value = emb

        results = self._make_results(["A", "B", "C"], [0.9, 0.8, 0.7])
        out = _mmr_rerank([1.0, 0.0], results, lambda_param=1.0)

        assert [r["content"] for r in out] == ["A", "B", "C"]

    @patch("agent.storage.get_embeddings")
    def test_top_k_limits_results(self, mock_get_emb):
        emb = MagicMock()
        emb.embed_query = MagicMock(return_value=[1.0, 0.0])
        mock_get_emb.return_value = emb

        results = self._make_results(["A", "B", "C", "D"], [0.9, 0.8, 0.7, 0.6])
        out = _mmr_rerank([1.0, 0.0], results, lambda_param=0.5, top_k=2)
        assert len(out) == 2

    @patch("agent.storage.get_embeddings")
    def test_preserves_result_structure(self, mock_get_emb):
        """MMR should preserve all keys in result dicts."""
        emb = MagicMock()
        emb.embed_query = MagicMock(return_value=[1.0, 0.0])
        mock_get_emb.return_value = emb

        results = [
            {"content": "A", "score": 0.9, "metadata": {"source": "doc1"}},
            {"content": "B", "score": 0.8, "metadata": {"source": "doc2"}},
        ]
        out = _mmr_rerank([1.0, 0.0], results)
        for r in out:
            assert "content" in r
            assert "score" in r
            assert "metadata" in r
