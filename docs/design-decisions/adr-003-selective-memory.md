# ADR-003: Selective Memory Capture

## Status

Accepted

## Context

After each conversation turn, the system could extract and store everything the user says. However, most conversational exchanges contain noise: greetings, acknowledgments, general knowledge Q&A, emotional expressions. Storing everything would:
- Pollute the vector store with low-value entries
- Increase storage and embedding costs
- Reduce search precision (more noise in results)
- Accelerate memory consolidation needs

## Decision

Implement a two-step selective memory pipeline:

### Step 1: Judgment (Lightweight)
Use a fast LLM call with `MEMORY_JUDGMENT_PROMPT` to determine if the turn is worth remembering.

Criteria for memorable=true:
- Explicit preferences ("I prefer X", "I use Y")
- Corrections ("actually, it's X not Y")
- Decisions or commitments
- Specific personal facts (names, dates, locations)
- Action items

Criteria for memorable=false:
- Greetings, small talk
- General knowledge Q&A
- Opinions, emotions, transient states
- Repetitions

### Step 2: Extraction (Only if memorable)
Use `SELECTIVE_EXTRACTION_PROMPT` to extract structured memory operations:
- **ADD**: New fact not in existing memory
- **UPDATE**: Refines or corrects an existing memory
- **DELETE**: Invalidates a previous memory

Each extracted memory includes:
- `text`: Self-contained factual statement
- `entities`: Named entities with types
- `temporal`: Time references with absolute dates

### Step 3: Conflict Resolution
Before executing ADD operations, check against existing memories:
- Use `FACT_CONFLICT_PROMPT` to decide merge vs skip
- High-confidence merges are automatic
- Low-confidence decisions are queued for human review

Configuration:
- `memory_selective_enabled: true` (default)
- `conflict_confidence_threshold: 0.7` (default)

## Consequences

### Advantages
- **Noise reduction**: Only meaningful facts are stored
- **Structured data**: Entities and temporal references enable metadata-aware search
- **Self-healing**: UPDATE/DELETE operations keep memory accurate over time
- **Human oversight**: Low-confidence conflicts can be reviewed

### Disadvantages
- **Two LLM calls per turn**: Judgment + extraction (can be skipped if not memorable)
- **Extraction quality**: LLM may miss facts or extract incorrectly
- **Latency**: Adds 1-3 seconds to each turn

### Trade-offs
- Judgment step is cheap (single short LLM call) and filters ~60-80% of turns
- Extraction only runs for memorable turns, so average cost is acceptable
- Conflict resolution prevents memory bloat from duplicate facts

## Alternatives Considered

1. **Store everything**: Simple but creates massive noise. Rejected.
2. **Rule-based extraction**: Regex patterns for facts. Too brittle for natural language.
3. **External memory service (mem0)**: Outsources the problem but adds dependency.
