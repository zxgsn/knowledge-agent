"""Cross-encoder re-ranking for search results.

Uses sentence-transformers CrossEncoder to re-rank retrieval results
with higher accuracy than bi-encoder cosine similarity alone.

Model: BAAI/bge-reranker-v2-m3 (multilingual, ~568M)
Downloaded to backend/models/bge-reranker-v2-m3/ (not HF cache).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model

    from sentence_transformers import CrossEncoder

    model_name = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    # Check for local model directory first
    local_dir = Path(__file__).resolve().parents[3] / "models" / model_name.split("/")[-1]
    if local_dir.is_dir() and (local_dir / "config.json").exists():
        model_path = str(local_dir)
        print(f"[reranker] Loading local model: {model_path}", file=sys.stderr)
    else:
        model_path = model_name
        print(f"[reranker] Loading model from HF: {model_path}", file=sys.stderr)

    _model = CrossEncoder(model_path, max_length=512)
    print(f"[reranker] Model loaded.", file=sys.stderr)
    return _model


def rerank(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """Re-rank search results using cross-encoder scoring.

    Args:
        query: The search query.
        results: List of dicts with at least 'content' and 'score' keys.
        top_k: Number of top results to return after re-ranking.

    Returns:
        Re-ranked results with 'score' replaced by cross-encoder score.
    """
    if not results:
        return []

    enabled = os.getenv("RERANK_ENABLED", "true").lower()
    if enabled in ("false", "0", "no"):
        return results[:top_k]

    try:
        model = _get_model()
    except Exception as e:
        print(f"[reranker] Failed to load model: {e}", file=sys.stderr)
        return results[:top_k]

    pairs = [[query, r["content"][:512]] for r in results]
    scores = model.predict(pairs)

    for r, s in zip(results, scores):
        r["score"] = float(s)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
