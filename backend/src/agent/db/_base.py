"""Shared helpers for the db package.

Contains low-level utility functions and constants used across
archival, recall, versioning, and other db submodules.
"""

from __future__ import annotations

import logging

from agent.storage import get_conn  # noqa: F401 — re-exported for test patching

logger = logging.getLogger(__name__)

# Unified BM25 text search configuration
_TS_CONFIG = "english"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _deduplicate_results(
    results: list[dict], threshold: float = 0.95
) -> list[dict]:
    """Remove near-duplicate results based on content similarity.

    Keeps the higher-scored version when two results have content
    overlap exceeding the threshold.
    """
    if len(results) <= 1:
        return results

    deduped = []
    for r in results:
        r_words = set(r["content"].lower().split())
        is_dup = False
        for kept in deduped:
            kept_words = set(kept["content"].lower().split())
            if not r_words or not kept_words:
                continue
            overlap = len(r_words & kept_words) / min(len(r_words), len(kept_words))
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            deduped.append(r)
    return deduped


def _mmr_rerank(
    query_embedding: list[float],
    results: list[dict],
    lambda_param: float = 0.5,
    top_k: int | None = None,
) -> list[dict]:
    """Maximal Marginal Relevance: balance relevance and diversity.

    Args:
        query_embedding: The query vector.
        results: List of dicts with 'content' and 'score' keys, sorted by score desc.
        lambda_param: 1.0 = pure relevance, 0.0 = pure diversity.
        top_k: Max results to return.
    """
    if len(results) <= 1:
        return results

    from agent.storage import get_embeddings

    embeddings = get_embeddings()
    doc_embeddings = [embeddings.embed_query(r["content"][:512]) for r in results]

    selected = [0]
    candidates = list(range(1, len(results)))

    while candidates:
        best_score, best_idx = float("-inf"), -1
        for c in candidates:
            relevance = results[c]["score"]
            max_sim = max(
                _cosine_similarity(doc_embeddings[c], doc_embeddings[s])
                for s in selected
            )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score, best_idx = mmr_score, c
        selected.append(best_idx)
        candidates.remove(best_idx)

    ordered = [results[i] for i in selected]
    return ordered[:top_k] if top_k else ordered
