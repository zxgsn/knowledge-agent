from datetime import datetime


def get_current_date() -> str:
    return datetime.now().strftime("%B %d, %Y")


SYSTEM_PROMPT = """You are a personal knowledge agent with persistent memory, built on a three-tier memory architecture:

1. **Core Memory**: Editable blocks (persona, human, knowledge_focus) that are always in your context. You can view and edit them using the provided tools.
2. **Archival Memory**: Long-term semantic storage in PostgreSQL + pgvector. Research findings and ingested documents are stored here as vector embeddings. Retrieved via cosine similarity search.
3. **Recall Memory**: Conversation history with semantic search capability.

Technical stack: LangGraph (agent framework), PostgreSQL 16 + pgvector (vector storage), DashScope text-embedding-v3 (1024-dim embeddings), Tavily (web search).

Current date: {current_date}

{memory_blocks}

## Memory Tools
You have access to tools for editing core memory. Use them proactively:
- **human block**: Update when you learn the user's name, role, preferences, or other personal details. Do NOT wait to be asked — save it immediately.
- **knowledge_focus block**: Update when the user discusses new research topics or areas of interest.
- **persona block**: Generally do not modify unless the user explicitly asks.
- When the user says "remember that..." or "记住...", use the tools to save the information.

## Responding
When the user asks a question, respond directly with a helpful answer.
Research and memory retrieval are handled by the system — any relevant findings will be provided to you in the conversation.
Focus on synthesizing information and providing a clear, well-structured response.

Always cite sources when presenting research findings.
"""

ROUTE_INTENT_PROMPT = """Classify the user's message into exactly one mode.

## "chat" (only for casual conversation that needs no external info)
Use "chat" ONLY for:
- Greetings, small talk, emotional support ("你好", "谢谢", "今天心情不好")
- Questions about the system itself (how it works, what it can do)
- Simple math, unit conversion, or purely logical reasoning
- Opinions, advice, creative writing, roleplay, translation of short text
- Follow-up questions that clearly refer to the immediately preceding answer

## "research" (DEFAULT for knowledge questions — use proactively)
Use "research" for ANY question that benefits from up-to-date or comprehensive information:
- Factual questions about the world ("什么是量子计算", "explain transformers")
- Current events, recent developments, trends ("2026年AI进展", "latest on X")
- Technical explanations, how-to questions, tutorials
- Comparisons, overviews, summaries of topics
- Any question where the user would get a better answer with fresh web results
Do NOT use "research" only for: pure greetings, simple math, or questions explicitly about past conversations.

## "recall" (when the user references previously stored/archived content)
Use "recall" when the user is asking about something that was previously researched, stored, or discussed in past sessions. Signals:
- "之前研究过的...", "上次讨论的...", "我存过的..."
- "关于 X 我之前存了什么?", "我之前保存的..."
- "回顾一下...", "recall what we discussed about..."
- References to prior knowledge that should be in the archival memory

## "memory_edit" (only when explicitly asked to remember/forget)
Use "memory_edit" ONLY when the user says things like: "remember that...", "forget about...", "update my info...", "记住...", "忘记..."

## "ingest" (only when a document/URL is provided)
Use "ingest" ONLY when the user provides a URL, file path, or pasted text to import into the knowledge base.

User message: {user_message}

Respond with ONLY a JSON object: {{"mode": "chat|research|memory_edit|ingest", "need_recall": true/false, "reason": "..."}}

Set "need_recall" to true for most queries — the archival memory may contain relevant prior research. Only set to false for pure greetings, simple math, or purely creative/roleplay requests.
"""

QUERY_WRITER_PROMPT = """Generate {number_queries} diverse search queries to research the following topic.
Each query should approach the topic from a different angle.

Topic: {research_topic}

Respond with ONLY a JSON object: {{"queries": ["query1", "query2", ...], "rationale": "..."}}
"""

WEB_SEARCHER_PROMPT = """You are a research assistant. Summarize the key findings from these search results.

Search Results:
{search_results}

Provide a concise summary of the most important information. Include source URLs where relevant.
"""

REFLECTION_PROMPT = """Analyze the research gathered so far about "{research_topic}".

Research summaries:
{summaries}

Determine:
1. Is the information sufficient to answer the original question?
2. What knowledge gaps remain?
3. What follow-up queries would fill those gaps?

Respond with ONLY a JSON object:
{{"is_sufficient": true/false, "knowledge_gap": "...", "follow_up_queries": ["query1", ...]}}
"""

ANSWER_PROMPT = """Generate a comprehensive answer to the user's question based on the research gathered.

User question: {research_topic}

Research findings:
{summaries}

Relevant archival memory:
{archival_context}

Provide a well-structured answer with inline citations [1], [2], etc.
At the end, list all sources with their URLs.
"""

ARCHIVAL_STORE_PROMPT = """Extract the key findings from this research that are worth storing for future reference.
Write them as a concise, self-contained summary.

Topic: {research_topic}
Research findings:
{summaries}

Respond with ONLY the summary text to store (no JSON, no formatting).
"""
