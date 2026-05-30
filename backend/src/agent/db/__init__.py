"""Centralized sync DB helpers for archival and recall memory.

All functions use the connection pool from agent.storage.
Import these instead of duplicating DB logic across node files.

This package is split into submodules for maintainability:
- _base: shared helpers and constants
- archival: archival memory CRUD and maintenance
- recall: recall memory search, save, cleanup
- versioning: version history snapshot, query, rollback
- conflicts: conflict review queue and resolution
- documents: document management and dedup
- analytics: analytics stats and source trust
- importance: importance scoring migration and queries
"""

from __future__ import annotations

# Re-export all public functions for backward compatibility.
# Imports from agent.db continue to work unchanged.

from agent.db._base import (
    _TS_CONFIG,
    _cosine_similarity,
    _deduplicate_results,
    _mmr_rerank,
    logger,
)

# Re-export get_conn so tests can patch agent.db.get_conn
from agent.db._base import get_conn  # noqa: F401

from agent.db.analytics import (
    SOURCE_TRUST,
    get_analytics_stats,
    get_source_trust,
)

from agent.db.archival import (
    cleanup_excess,
    cleanup_namespace,
    get_all_facts,
    get_existing_memories,
    get_recent_archival,
    put_batch_to_archival,
    put_to_archival,
    search_archival,
    search_archival_for_dedup,
    update_archival,
    delete_archival,
    ensure_structured_indexes,
    search_by_entity,
    search_by_temporal,
    search_by_temporal_range,
)

from agent.db.conflicts import (
    get_pending_conflicts,
    queue_conflict_review,
    resolve_conflict,
)

from agent.db.documents import (
    insert_document,
)

from agent.db.importance import (
    ensure_importance_columns,
    get_memories_by_importance,
    get_memory_stats_extended,
    search_importance_weighted,
    touch_memory_access,
    update_importance_score,
)

from agent.db.recall import (
    cleanup_recall,
    get_recent_recall,
    save_to_recall,
    search_recall,
)

from agent.db.versioning import (
    delete_version,
    get_version_history,
    rollback_to_version,
)

__all__ = [
    # _base
    "_TS_CONFIG", "_cosine_similarity", "_deduplicate_results", "_mmr_rerank", "logger",
    # analytics
    "SOURCE_TRUST", "get_analytics_stats", "get_source_trust",
    # archival
    "cleanup_excess", "cleanup_namespace", "get_all_facts", "get_existing_memories",
    "get_recent_archival", "put_batch_to_archival", "put_to_archival",
    "search_archival", "search_archival_for_dedup", "update_archival", "delete_archival",
    "ensure_structured_indexes", "search_by_entity", "search_by_temporal", "search_by_temporal_range",
    # conflicts
    "get_pending_conflicts", "queue_conflict_review", "resolve_conflict",
    # documents
    "insert_document",
    # importance
    "ensure_importance_columns", "get_memories_by_importance", "get_memory_stats_extended",
    "search_importance_weighted", "touch_memory_access", "update_importance_score",
    # recall
    "cleanup_recall", "get_recent_recall", "save_to_recall", "search_recall",
    # versioning
    "delete_version", "get_version_history", "rollback_to_version",
]
