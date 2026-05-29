# ADR-005: MMR (Maximal Marginal Relevance) for Diversity

## Status

Accepted (disabled by default)

## Context

Standard vector search returns the top-K most similar results. However, these results are often semantically redundant: multiple entries about the same fact with slight wording variations. This wastes the limited context window with duplicate information.

## Decision

Implement MMR (Maximal Marginal Relevance) as an optional post-retrieval diversification step:

```
MMR(d) = lambda * relevance(d) - (1 - lambda) * max_sim(d, selected)
```

Where:
- `relevance(d)`: Original retrieval score
- `max_sim(d, selected)`: Maximum cosine similarity to any already-selected document
- `lambda`: Balance parameter (1.0 = pure relevance, 0.0 = pure diversity)

Algorithm:
1. Start with the highest-scoring document
2. For each remaining slot, select the document that maximizes MMR score
3. Continue until top-K documents are selected

Configuration:
- `mmr_enabled: false` (default, opt-in)
- `mmr_lambda: 0.7` (default, in config.yaml)

Implementation in `db.py::_mmr_rerank()`:
- Re-embeds candidate documents for similarity computation
- Only applies when `mmr_enabled` is true and results > 1

## Consequences

### Advantages
- **Reduces redundancy**: Selected results cover more diverse aspects of the query
- **Better context utilization**: More information per token in the context window
- **Tunable**: Lambda parameter allows adjusting relevance/diversity trade-off

### Disadvantages
- **Latency**: Requires re-embedding candidates for pairwise similarity (~200-500ms)
- **May miss nuances**: Diversity penalty could exclude highly relevant near-duplicates that contain important subtle differences
- **Disabled by default**: Performance cost makes it opt-in

### When to Enable
- When search results contain many near-duplicates
- When context window is tight and every token matters
- For broad exploratory queries where diversity is more important than precision

## Alternatives Considered

1. **Post-retrieval dedup**: Simple word-overlap dedup (already implemented). Less sophisticated but zero cost.
2. **Diverse beam search**: More complex, designed for generation not retrieval.
3. **Clustering-based**: Group results by topic, pick from each cluster. More complex, similar outcome.
