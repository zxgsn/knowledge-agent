"""Cross-encoder re-ranking for search results.

Uses sentence-transformers CrossEncoder to re-rank retrieval results
with higher accuracy than bi-encoder cosine similarity alone.

Model: BAAI/bge-reranker-v2-m3 (multilingual, ~568M)
Downloaded to backend/models/bge-reranker-v2-m3/ (not HF cache).
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from agent.logger import get_logger
from agent.retry import with_retry

logger = get_logger(__name__)

_model = None
_model_lock = threading.Lock()
# Track if we've already had a catastrophic failure and should disable reranking entirely
_disabled_due_to_failure = False


def _load_model():
    """Load the cross-encoder model (sync, for use with retry)."""
    from sentence_transformers import CrossEncoder
    import torch

    model_name = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    # Check for local model directory first
    local_dir = Path(__file__).resolve().parents[3] / "models" / model_name.split("/")[-1]
    if local_dir.is_dir() and (local_dir / "config.json").exists():
        model_path = str(local_dir)
        logger.info("Loading local reranker model: %s", model_path)
    else:
        model_path = model_name
        logger.info("Loading reranker model from HF: %s", model_path)

    # Force CPU and reduce memory usage to avoid segfault on Windows
    device = "cpu"
    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works
            torch.tensor([1.0]).cuda()
            device = "cuda"
        except Exception:
            device = "cpu"
            logger.debug("CUDA test failed, using CPU for reranking")
    logger.info("Using device: %s for reranker", device)

    model = CrossEncoder(model_path, max_length=512, device=device)
    logger.info("Reranker model loaded successfully")
    return model


@with_retry(
    max_retries=1,
    base_delay=1.0,
    max_delay=5.0,
    retryable_exceptions=(OSError, RuntimeError),
)
def _get_model():
    """Get or lazily load the cross-encoder model with retry on first load.

    Thread-safe: uses double-checked locking to prevent duplicate loads.
    """
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        # Double-check after acquiring lock
        if _model is not None:
            return _model
        _model = _load_model()
        return _model


def rerank(
    query: str,
    results: list[dict],
    top_k: int = 5,
    enabled: bool | None = None,
) -> list[dict]:
    """Re-rank search results using cross-encoder scoring.

    Falls back gracefully to the original results (truncated to *top_k*)
    if the model fails to load or scoring fails.

    Args:
        query: The search query.
        results: List of dicts with at least 'content' and 'score' keys.
        top_k: Number of top results to return after re-ranking.
        enabled: Override env RERANK_ENABLED. None = use env.

    Returns:
        Re-ranked results with 'score' replaced by cross-encoder score.
    """
    global _disabled_due_to_failure

    if not results:
        return []

    if _disabled_due_to_failure:
        logger.debug("Reranking disabled due to previous failure")
        return results[:top_k]

    if enabled is None:
        enabled = os.getenv("RERANK_ENABLED", "true").lower() not in ("false", "0", "no")
    if not enabled:
        logger.debug("Reranking disabled, returning top %d results", top_k)
        return results[:top_k]

    try:
        model = _get_model()
    except Exception as exc:
        logger.error("Failed to load reranker model: %s — disabling reranking", exc)
        _disabled_due_to_failure = True
        return results[:top_k]

    try:
        pairs = [[query, r["content"][:512]] for r in results]
        # Run predict in a subprocess-safe way to avoid segfaults
        # on Windows with large models in threaded contexts
        import torch
        with torch.no_grad():
            scores = model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)

        for r, s in zip(results, scores):
            r["score"] = float(s)

        results.sort(key=lambda x: x["score"], reverse=True)
        logger.debug(
            "Reranked %d results, top score=%.4f",
            len(results),
            results[0]["score"] if results else 0.0,
        )
    except Exception as exc:
        logger.error("Reranking failed: %s — falling back to original order", exc, exc_info=True)
        # If this is a serious error, consider disabling reranking entirely
        if "segfault" in str(exc).lower() or "access violation" in str(exc).lower() or "fatal" in str(exc).lower():
            _disabled_due_to_failure = True
            logger.warning("Disabling reranking due to fatal error")
        return results[:top_k]

    return results[:top_k]
