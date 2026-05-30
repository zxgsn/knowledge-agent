"""Memory importance scoring system.

Calculates importance scores (0.0-1.0) for archival memory entries
based on recency, access frequency, source trust, content richness,
and entity density.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

# Source trust weights (mirrors agent.db.SOURCE_TRUST for consistency)
_SOURCE_TRUST: dict[str, float] = {
    "research_summary": 0.9,
    "manual": 1.0,
    "manual_save": 1.0,
    "manual_edit": 1.0,
    "conversation": 0.7,
    "memory_pipeline": 0.7,
    "consolidation": 0.8,
    "ingested": 0.8,
    "conflict_review": 0.9,
}

# Simple patterns that hint at named entities (capitalised words, proper nouns)
_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp string, tolerant of common formats."""
    if not value:
        return None
    try:
        # Handle timezone-aware ISO strings
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _recency_score(created_at: str | None, now: datetime | None = None) -> float:
    """Exponential decay score based on age. Newer memories score higher.

    Half-life of 30 days by default.
    """
    dt = _parse_dt(created_at)
    if dt is None:
        return 0.5  # Unknown age -> neutral

    now = now or datetime.now(timezone.utc)
    age_days = max((now - dt).total_seconds() / 86400, 0)
    half_life = 30.0
    lam = math.log(2) / half_life
    return math.exp(-lam * age_days)


def _access_score(access_count: int) -> float:
    """Logarithmic scaling of access frequency. More accesses -> higher score.

    Saturates around 10+ accesses.
    """
    if access_count <= 0:
        return 0.0
    return min(1.0, math.log1p(access_count) / math.log1p(10))


def _source_trust_score(metadata: dict) -> float:
    """Lookup source trust weight from metadata."""
    source = metadata.get("source", "unknown")
    return _SOURCE_TRUST.get(source, 0.5)


def _content_richness(content: str) -> float:
    """Longer, more detailed content scores slightly higher.

    Uses log scaling so very long docs don't dominate.
    Saturates around ~2000 chars.
    """
    length = len(content)
    if length == 0:
        return 0.0
    return min(1.0, math.log1p(length) / math.log1p(2000))


def _entity_density(content: str) -> float:
    """Memories with more named entities score higher.

    Extracts capitalised noun-like tokens as a cheap NER proxy.
    """
    if not content:
        return 0.0
    matches = _ENTITY_PATTERN.findall(content)
    words = content.split()
    if not words:
        return 0.0
    density = len(matches) / len(words)
    # Scale: 0 entities = 0.0, ~20% entity tokens = 1.0
    return min(1.0, density / 0.2)


def calculate_importance(
    content: str,
    metadata: dict,
    access_count: int = 0,
    last_accessed: str | None = None,
    created_at: str | None = None,
) -> float:
    """Calculate importance score (0.0-1.0) for a memory entry.

    Factors:
    - Recency: newer memories score higher (exponential decay)
    - Access frequency: more accessed = more important
    - Source trust: research_summary > manual > conversation
    - Content richness: longer, more detailed content scores slightly higher
    - Entity density: memories with more named entities score higher

    Args:
        content: The text content of the memory entry.
        metadata: Metadata dict (must contain 'source' for trust scoring).
        access_count: How many times this entry has been retrieved.
        last_accessed: ISO timestamp of last access (boosts recency).
        created_at: ISO timestamp of creation.

    Returns:
        Importance score between 0.0 and 1.0.
    """
    # Weights for each factor (sum to 1.0)
    W_RECENCY = 0.30
    W_ACCESS = 0.20
    W_TRUST = 0.25
    W_RICHNESS = 0.10
    W_ENTITIES = 0.15

    recency = _recency_score(created_at)
    # If recently accessed, use that as recency signal too
    if last_accessed:
        access_recency = _recency_score(last_accessed)
        recency = max(recency, access_recency * 0.8)

    access = _access_score(access_count)
    trust = _source_trust_score(metadata)
    richness = _content_richness(content)
    entities = _entity_density(content)

    raw_score = (
        W_RECENCY * recency
        + W_ACCESS * access
        + W_TRUST * trust
        + W_RICHNESS * richness
        + W_ENTITIES * entities
    )

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, raw_score))


def compute_decay_score(
    base_importance: float,
    created_at: str,
    last_accessed: str | None = None,
    half_life_days: float = 30.0,
) -> float:
    """Apply time-based decay to importance score.

    Uses exponential decay: score * exp(-lambda * age_days)
    where lambda = ln(2) / half_life_days.

    Args:
        base_importance: The initial importance score.
        created_at: ISO timestamp when the memory was created.
        last_accessed: ISO timestamp of last access (used instead of
            created_at if more recent).
        half_life_days: Number of days for score to halve. Default 30.

    Returns:
        Decayed importance score (0.0-1.0).
    """
    now = datetime.now(timezone.utc)

    # Use the most recent reference point
    reference = _parse_dt(created_at)
    accessed_dt = _parse_dt(last_accessed)
    if accessed_dt and reference and accessed_dt > reference:
        reference = accessed_dt

    if reference is None:
        return base_importance  # Can't decay without a timestamp

    age_days = max((now - reference).total_seconds() / 86400, 0)
    lam = math.log(2) / max(half_life_days, 0.01)
    decayed = base_importance * math.exp(-lam * age_days)
    return max(0.0, min(1.0, decayed))
