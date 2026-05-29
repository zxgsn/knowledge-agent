# ADR-001: Three-Layer Memory Architecture

## Status

Accepted

## Context

The Knowledge Agent needs a persistent memory system that can:
- Maintain user context across conversations (preferences, personal info)
- Store and retrieve research findings and ingested documents
- Search conversation history semantically
- Scale to thousands of stored facts without performance degradation

Key requirements:
1. Some information (user profile, current focus) must be always available in context
2. Long-term knowledge must be retrievable via semantic search
3. Conversation history must be searchable but isolated per thread
4. The system must handle conflicting information gracefully

## Decision

Adopt a three-layer memory architecture inspired by Letta (formerly MemGPT):

### Layer 1: Core Memory (In-Context)
- Always present in the system prompt
- Implemented as editable `Block` objects with labels (persona, human, knowledge_focus)
- Agent can read and write via LangChain tools (`core_memory_replace`, `core_memory_insert`)
- Character limit per block (default: 5000 chars)
- No database storage; persisted as part of agent state

### Layer 2: Archival Memory (Long-Term)
- PostgreSQL + pgvector for vector storage
- Namespace-based organization (conversation_facts, research_findings, document_ingested)
- Hybrid search: vector similarity + BM25 keyword matching
- Optional cross-encoder re-ranking and MMR diversity
- Structured metadata: entities, temporal references, source info

### Layer 3: Recall Memory (Conversation History)
- PostgreSQL with vector embeddings
- Thread-isolated by `thread_id`
- Stores user and assistant messages
- Semantic search for referencing past conversations

## Consequences

### Advantages
- **Separation of concerns**: Each layer has a clear responsibility
- **Always-on context**: Core memory ensures critical info is never lost in search
- **Scalable**: Archival memory can grow to millions of entries via pgvector
- **Thread safety**: Recall memory isolation prevents cross-contamination between sessions
- **Graceful degradation**: If vector search fails, BM25 still works; if both fail, core memory persists

### Disadvantages
- **Context budget pressure**: Core memory competes with other content for prompt space
- **Consistency risk**: Core memory edits are immediate; archival is eventually consistent
- **Complexity**: Three layers mean three places to check when debugging memory issues

### Risks
- Core memory blocks could exceed their character limit if the agent is too aggressive with updates
- Archival memory search quality depends heavily on embedding model choice
- Recall memory grows unboundedly without periodic cleanup

## Alternatives Considered

1. **Single flat storage**: All memories in one vector store. Rejected: no guaranteed context for critical info.
2. **Two-layer (Core + Vector)**: Merge archival and recall into one store. Rejected: different isolation requirements (global vs per-thread).
3. **External memory service (Zep, Mem0)**: Use a dedicated memory SaaS. Rejected: adds external dependency and latency for a self-contained system.

## References

- Letta/MemGPT: [https://www.letta.com/](https://www.letta.com/)
- Mem0: [https://www.mem0.ai/](https://www.mem0.ai/)
- pgvector: [https://github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)
