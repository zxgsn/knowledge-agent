"""Analytics: memory statistics and source trust scoring."""

from __future__ import annotations

from agent.db._base import get_conn

from agent.db._base import logger


# --- Source Trust ---


SOURCE_TRUST = {
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


def get_source_trust(metadata: dict) -> tuple[float, str]:
    """Return (score, label) for a memory entry's source."""
    source = metadata.get("source", "unknown")
    score = SOURCE_TRUST.get(source, 0.5)
    if score >= 0.9:
        label = "verified"
    elif score >= 0.7:
        label = "default"
    else:
        label = "low"
    return score, label


def get_analytics_stats() -> dict:
    """Get memory analytics statistics.

    Returns a dict with counts for archival memory, recall memory,
    documents, conflicts, versions, and age distribution.
    """
    stats: dict = {}

    try:
        with get_conn() as conn:
            # Archival memory stats
            row = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT namespace) FROM archival_memory WHERE status = 'active'"
            ).fetchone()
            stats["archival_count"] = row[0]
            stats["namespace_count"] = row[1]

            # Namespace breakdown
            ns_rows = conn.execute(
                "SELECT namespace, COUNT(*) FROM archival_memory "
                "WHERE status = 'active' GROUP BY namespace ORDER BY COUNT(*) DESC"
            ).fetchall()
            stats["namespaces"] = [{"namespace": r[0], "count": r[1]} for r in ns_rows]

            # Recall memory stats
            row = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT thread_id) FROM recall_memory"
            ).fetchone()
            stats["recall_count"] = row[0]
            stats["thread_count"] = row[1]

            # Document stats
            row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            stats["document_count"] = row[0]

            # Conflict stats
            row = conn.execute(
                "SELECT COUNT(*) FROM conflict_reviews WHERE status = 'pending'"
            ).fetchone()
            stats["pending_conflicts"] = row[0]

            # Version history stats
            row = conn.execute("SELECT COUNT(*) FROM archival_versions").fetchone()
            stats["version_count"] = row[0]

            # Age distribution (last 7 days, 30 days, older)
            row = conn.execute(
                "SELECT "
                "COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days'), "
                "COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days' AND created_at <= NOW() - INTERVAL '7 days'), "
                "COUNT(*) FILTER (WHERE created_at <= NOW() - INTERVAL '30 days') "
                "FROM archival_memory WHERE status = 'active'"
            ).fetchone()
            stats["age_distribution"] = {
                "last_7_days": row[0],
                "last_30_days": row[1],
                "older": row[2],
            }
    except Exception as e:
        logger.error("get_analytics_stats failed: %s", e)
        stats["error"] = str(e)

    return stats
