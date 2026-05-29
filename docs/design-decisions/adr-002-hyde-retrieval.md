# ADR-002: HyDE (Hypothetical Document Embedding) for Retrieval

## Status

Accepted

## Context

Standard semantic search encodes the user's query directly into an embedding vector. However, user queries are often short, ambiguous, or phrased as questions ("What is quantum computing?"), while stored documents are typically longer, declarative statements. This creates a semantic gap between query and document embeddings.

## Decision

Implement HyDE (Hypothetical Document Embedding) as an optional preprocessing step:

1. Given a user query, generate a hypothetical answer using the LLM
2. Encode the hypothetical answer (not the original query) into an embedding
3. Use this embedding for vector similarity search

Configuration: `hyde_enabled: true` (default in config.yaml)

Implementation in `nodes/memory_manager.py::_generate_hypothetical()`:
```python
hyde_prompt = (
    "Answer this question in 1-2 short sentences. "
    "Be specific with names and details. "
    "If you don't know, guess based on the question context.\n\n"
    f"Question: {query}"
)
```

## Consequences

### Advantages
- **Better recall**: Hypothetical answers are semantically closer to stored documents
- **Handles questions**: Bridges the gap between question-form queries and declarative storage
- **Specificity**: LLM-generated answers include entity names and details that improve matching

### Disadvantages
- **Latency**: Adds one LLM call before search (typically 200-500ms)
- **Hallucination risk**: The hypothetical answer may contain incorrect details that bias the search
- **Cost**: Additional LLM token usage per search

### Mitigations
- HyDE is only used for archival search embedding, not for the final answer
- The hypothetical answer is discarded after embedding; it never reaches the user
- Can be disabled via config if latency is critical

## Alternatives Considered

1. **Query expansion**: Use synonyms/related terms. Less effective than HyDE for semantic gap.
2. **Multi-query**: Generate multiple query variations. More expensive than HyDE with similar benefit.
3. **No preprocessing**: Direct query embedding. Simpler but lower recall for question-form queries.
