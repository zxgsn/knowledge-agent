"""Lightweight in-memory knowledge graph for linking related archival memories.

Nodes: entities extracted from memory content.
Edges: relationships between entities (co-occurrence, temporal, causal).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# Simple capitalised-noun phrase pattern for entity extraction
_CAPITALISED = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")
# Common stop words to exclude from entity extraction
_STOP_WORDS = frozenset({
    "The", "This", "That", "These", "Those", "What", "When", "Where",
    "Which", "While", "With", "From", "About", "After", "Before",
    "Between", "During", "Under", "Over", "Into", "Through",
    "Also", "Just", "Like", "More", "Most", "Much", "Many",
    "Some", "Such", "Then", "Than", "Here", "There", "Very",
    "User", "Assistant", "System", "Note", "However", "Therefore",
    "Additionally", "Furthermore", "Meanwhile", "Otherwise", "Finally",
})


def _extract_entities(text: str) -> list[str]:
    """Extract candidate entity phrases from text.

    Uses capitalised word detection as a cheap NER proxy, filters
    stop words, and deduplicates while preserving order.
    """
    if not text:
        return []

    matches = _CAPITALISED.findall(text)
    seen: set[str] = set()
    entities: list[str] = []
    for m in matches:
        if m in _STOP_WORDS:
            continue
        # Require at least 3 chars per word
        if any(len(w) < 3 for w in m.split()):
            continue
        norm = m.strip()
        if norm and norm not in seen:
            seen.add(norm)
            entities.append(norm)
    return entities


class KnowledgeGraph:
    """In-memory knowledge graph for linking related archival memories.

    Nodes: entities extracted from memory content.
    Edges: relationships between entities (co-occurrence weight).

    The graph is dict-based and can be rebuilt from the DB on startup.
    """

    def __init__(self) -> None:
        # entity_id -> set of memory_ids that mention it
        self._entity_memories: dict[str, set[str]] = defaultdict(set)
        # memory_id -> set of entity_ids
        self._memory_entities: dict[str, set[str]] = defaultdict(set)
        # (entity_a, entity_b) -> co-occurrence count (undirected, a < b)
        self._edges: dict[tuple[str, str], int] = defaultdict(int)
        # memory_id -> content snippet (for display)
        self._memory_content: dict[str, str] = {}

    def add_memory(
        self, memory_id: str, content: str, metadata: dict | None = None
    ) -> list[str]:
        """Extract entities from content and add to graph.

        Args:
            memory_id: Unique identifier for the memory entry.
            content: Text content to extract entities from.
            metadata: Optional metadata (entities may also come from
                metadata['entities'] if structured extraction was used).

        Returns:
            List of entity IDs extracted/registered.
        """
        # Extract entities from content
        entities = _extract_entities(content)

        # Also pull structured entities from metadata if available
        if metadata:
            structured = metadata.get("entities", [])
            if isinstance(structured, list):
                for ent in structured:
                    if isinstance(ent, dict):
                        name = ent.get("name", "").strip()
                    elif isinstance(ent, str):
                        name = ent.strip()
                    else:
                        continue
                    if name and name not in entities:
                        entities.append(name)

        if not entities:
            return []

        self._memory_content[memory_id] = content[:200]

        for entity in entities:
            self._entity_memories[entity].add(memory_id)
            self._memory_entities[memory_id].add(entity)

        # Create edges between co-occurring entities
        for i, a in enumerate(entities):
            for b in entities[i + 1 :]:
                key = (a, b) if a < b else (b, a)
                self._edges[key] += 1

        return entities

    def remove_memory(self, memory_id: str) -> None:
        """Remove a memory and its entity associations from the graph."""
        entities = self._memory_entities.pop(memory_id, set())
        self._memory_content.pop(memory_id, None)

        for entity in entities:
            mems = self._entity_memories.get(entity)
            if mems:
                mems.discard(memory_id)
                if not mems:
                    del self._entity_memories[entity]

        # Recalculate affected edges (brute force but fine for in-memory)
        self._rebuild_edges()

    def _rebuild_edges(self) -> None:
        """Rebuild all edges from current entity-memory mappings."""
        self._edges.clear()
        for memory_id, entities in self._memory_entities.items():
            ent_list = list(entities)
            for i, a in enumerate(ent_list):
                for b in ent_list[i + 1 :]:
                    key = (a, b) if a < b else (b, a)
                    self._edges[key] += 1

    def find_related(
        self, entity: str, max_hops: int = 2
    ) -> list[dict]:
        """Find memories related to an entity through graph traversal.

        BFS up to max_hops, collecting memories along the way.

        Returns:
            List of dicts with 'memory_id', 'entity', 'hops', 'content'.
        """
        if entity not in self._entity_memories:
            return []

        visited_entities: set[str] = {entity}
        visited_memories: set[str] = set()
        results: list[dict] = []
        frontier: set[str] = {entity}

        for hop in range(max_hops):
            next_frontier: set[str] = set()
            for ent in frontier:
                # Collect memories for this entity
                for mem_id in self._entity_memories.get(ent, set()):
                    if mem_id not in visited_memories:
                        visited_memories.add(mem_id)
                        results.append({
                            "memory_id": mem_id,
                            "entity": ent,
                            "hops": hop,
                            "content": self._memory_content.get(mem_id, ""),
                        })

                # Find adjacent entities via edges
                for (a, b), weight in self._edges.items():
                    neighbor = None
                    if a == ent and b not in visited_entities:
                        neighbor = b
                    elif b == ent and a not in visited_entities:
                        neighbor = a
                    if neighbor:
                        next_frontier.add(neighbor)
                        visited_entities.add(neighbor)

            frontier = next_frontier
            if not frontier:
                break

        return results

    def find_similar_entities(
        self, query: str, top_k: int = 5
    ) -> list[dict]:
        """Find entities similar to a query string.

        Uses simple substring/word overlap as similarity metric
        (no embeddings needed for lightweight matching).

        Returns:
            List of dicts with 'entity', 'score', 'memory_count'.
        """
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        if not query_words:
            return []

        scored: list[tuple[float, str]] = []
        for entity in self._entity_memories:
            ent_lower = entity.lower()
            # Exact substring match
            if query_lower in ent_lower or ent_lower in query_lower:
                scored.append((1.0, entity))
                continue
            # Word overlap
            ent_words = set(ent_lower.split())
            overlap = len(query_words & ent_words)
            if overlap > 0:
                score = overlap / max(len(query_words), len(ent_words))
                scored.append((score, entity))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "entity": entity,
                "score": round(score, 3),
                "memory_count": len(self._entity_memories.get(entity, set())),
            }
            for score, entity in scored[:top_k]
        ]

    def get_subgraph(self, center_entity: str, radius: int = 2) -> dict:
        """Get a subgraph around an entity for visualization.

        Returns:
            Dict with 'nodes' (entities) and 'edges' (relationships),
            suitable for JSON serialization and front-end rendering.
        """
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        visited_entities: set[str] = set()
        frontier: set[str] = {center_entity}

        for hop in range(radius + 1):
            next_frontier: set[str] = set()
            for ent in frontier:
                if ent in visited_entities:
                    continue
                visited_entities.add(ent)
                nodes[ent] = {
                    "id": ent,
                    "memory_count": len(self._entity_memories.get(ent, set())),
                    "hop": hop,
                }

                # Find edges to adjacent entities
                for (a, b), weight in self._edges.items():
                    neighbor = None
                    if a == ent:
                        neighbor = b
                    elif b == ent:
                        neighbor = a
                    if neighbor:
                        edges.append({
                            "source": a,
                            "target": b,
                            "weight": weight,
                        })
                        if neighbor not in visited_entities:
                            next_frontier.add(neighbor)

            frontier = next_frontier
            if not frontier:
                break

        # Deduplicate edges within the subgraph
        seen_edges: set[tuple[str, str]] = set()
        unique_edges: list[dict] = []
        for e in edges:
            key = (e["source"], e["target"]) if e["source"] < e["target"] else (e["target"], e["source"])
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)

        return {
            "center": center_entity,
            "nodes": list(nodes.values()),
            "edges": unique_edges,
        }

    def get_top_entities(self, limit: int = 20) -> list[dict]:
        """Return the most frequently mentioned entities across all memories.

        Returns:
            List of dicts with 'entity', 'memory_count', 'edge_count'.
        """
        entities = []
        for entity, mems in self._entity_memories.items():
            edge_count = sum(
                1 for (a, b) in self._edges if a == entity or b == entity
            )
            entities.append({
                "entity": entity,
                "memory_count": len(mems),
                "edge_count": edge_count,
            })
        entities.sort(key=lambda x: x["memory_count"], reverse=True)
        return entities[:limit]

    def to_json(self) -> str:
        """Serialize graph to JSON for API responses."""
        nodes = [
            {
                "id": entity,
                "memory_count": len(mems),
            }
            for entity, mems in self._entity_memories.items()
        ]
        edges = [
            {
                "source": a,
                "target": b,
                "weight": weight,
            }
            for (a, b), weight in self._edges.items()
        ]
        return json.dumps({
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to a dict for API responses."""
        nodes = [
            {
                "id": entity,
                "memory_count": len(mems),
            }
            for entity, mems in self._entity_memories.items()
        ]
        edges = [
            {
                "source": a,
                "target": b,
                "weight": weight,
            }
            for (a, b), weight in self._edges.items()
        ]
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    @property
    def stats(self) -> dict:
        """Return basic graph statistics."""
        return {
            "entity_count": len(self._entity_memories),
            "memory_count": len(self._memory_entities),
            "edge_count": len(self._edges),
        }


# Module-level singleton for reuse across the application
_knowledge_graph: KnowledgeGraph | None = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Get or create the global KnowledgeGraph singleton."""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph


def rebuild_graph_from_db() -> KnowledgeGraph:
    """Rebuild the knowledge graph from all active archival memories.

    Call this on startup or when the graph needs to be refreshed.
    """
    from agent.storage import get_conn

    graph = KnowledgeGraph()
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT id, content, metadata FROM archival_memory "
                "WHERE status = 'active' ORDER BY created_at DESC LIMIT 2000"
            ).fetchall()

        for row in rows:
            meta = row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {}
            graph.add_memory(str(row[0]), row[1], meta)

        logger.info(
            "Knowledge graph rebuilt: %d entities, %d memories, %d edges",
            graph.stats["entity_count"],
            graph.stats["memory_count"],
            graph.stats["edge_count"],
        )
    except Exception as e:
        logger.warning("Failed to rebuild knowledge graph from DB: %s", e)

    global _knowledge_graph
    _knowledge_graph = graph
    return graph
