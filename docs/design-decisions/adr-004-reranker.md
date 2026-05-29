# ADR-004: Cross-Encoder Reranking

## Status

Accepted

## Context

The initial retrieval uses bi-encoder embeddings (BGE-M3) for fast approximate nearest-neighbor search. Bi-encoders encode query and document independently, which limits their ability to capture fine-grained query-document interactions.

Cross-encoders, by contrast, process the query and document together through the full transformer, producing more accurate relevance scores at the cost of higher computation.

## Decision

Add an optional cross-encoder reranking stage after bi-encoder retrieval:

1. Bi-encoder retrieves `candidate_limit` (3x final limit) results
2. Cross-encoder scores each (query, document) pair
3. Results are re-sorted by cross-encoder score
4. Top `limit` results are returned

Model: `BAAI/bge-reranker-v2-m3` (multilingual, ~568M parameters)

Implementation in `storage/reranker.py`:
- Lazy-loaded singleton model
- Runs on CPU (with optional CUDA fallback)
- Max input length: 512 tokens
- Content truncated to 512 chars for reranking

Configuration:
- `rerank_enabled: true` (default)
- `RERANK_MODEL` env var for model override

## Consequences

### Advantages
- **Higher precision**: Cross-encoder captures query-document interactions that bi-encoder misses
- **Multilingual**: bge-reranker-v2-m3 supports Chinese + English
- **Graceful degradation**: If model fails to load, falls back to bi-encoder ranking

### Disadvantages
- **Latency**: ~100-300ms for scoring 15 candidates on CPU
- **Memory**: ~1.1GB for the reranker model in memory
- **Truncation**: Long documents lose information when truncated to 512 chars

### Performance Impact
- Typical latency: 100-300ms on CPU for 15 candidates
- Precision improvement: ~10-15% over bi-encoder alone (based on BEIR benchmarks)

## Alternatives Considered

1. **No reranking**: Simpler but lower precision. Current default includes reranking.
2. **ColBERT**: Late-interaction model. More complex to deploy, marginal gain over cross-encoder.
3. **API-based reranking (Cohere)**: Adds external dependency and per-query cost.
