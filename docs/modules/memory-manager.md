# Module: Memory Manager

**Source**: `backend/src/agent/nodes/memory_manager.py`

The memory manager contains the core memory-related LangGraph nodes: intent routing, memory recall, recall evaluation, document ingestion, and archival save.

## Nodes

### `route_intent(state, config) -> dict`

Classifies user intent into one of 5 modes using LLM.

**Input**: Latest user message from `state["messages"]`
**Output**: `{mode: str, need_recall: bool}`

**Logic**:
1. If mode already set by frontend (research, memory_edit, ingest), keep it
2. Otherwise, use `ROUTE_INTENT_PROMPT` to classify
3. Parse JSON response to get mode and need_recall flag

**Modes**: chat, research, recall, memory_edit, ingest

### `recall_memory(state, config) -> dict`

Searches both archival and recall memory for relevant content.

**Flow**:
1. **Query rewriting** (optional): Rewrite ambiguous/referential queries into self-contained form
2. **HyDE** (optional): Generate hypothetical answer for better embedding
3. **Parallel search**: Search archival (hybrid: vector + BM25) and recall (vector) simultaneously
4. **Temporal boost**: If temporal keywords detected, search by temporal metadata
5. **Save to recall**: Store user message AFTER search (avoid self-match)

**Output**: `{archival_results: [...], recall_results: [...], memory_operations: [...]}`

### `evaluate_recall(state, config) -> dict`

Determines if retrieved memory is sufficient to answer the user's question.

**Logic**:
- If no memory results: insufficient
- If pure chat mode and proactive memory: sufficient
- Otherwise: LLM evaluates using `EVALUATE_RECALL_PROMPT`

**Output**: `{memory_sufficient: bool, memory_evaluation: str}`

### `ingest_document_node(state, config) -> dict`

Processes documents (URL, PDF, text) and stores in archival memory.

**Pipeline**:
1. Detect source type (url, pdf, text)
2. Extract text content
3. Chunk into segments
4. Embed and store each chunk
5. Register document in `documents` table

### `save_to_archival(state, config) -> dict`

Extracts key findings from web research and stores in archival memory.

**Input**: Web research summaries from state
**Output**: Stored entries in `conversation_facts` namespace

## Helper Functions

### `_rewrite_query(user_msg, messages, config) -> str`

Rewrites ambiguous/referential queries using conversation context. Example: "What about that?" becomes "What is the capital of France?" given prior context.

### `_generate_hypothetical(query, config) -> str`

Generates a hypothetical answer for HyDE retrieval. The answer is used only for embedding, never shown to the user.

### `_detect_temporal_ref(query) -> str | None`

Extracts temporal keywords (yesterday, last Monday, etc.) from queries. Supports English and Chinese.

### `_detect_entities(query, archival_results) -> list[str]`

Extracts entity names from archival result metadata that also appear in the query. Used for entity-aware supplementary search.
