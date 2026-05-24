"""Archival Memory management tool.

Manage research summaries, ingested documents, and other archival entries.
Supports statistics, listing, deduplication, cleanup, and consolidation.

Usage:
  python scripts/manage_archival.py stats                          # Show stats per namespace
  python scripts/manage_archival.py list --namespace research      # List entries
  python scripts/manage_archival.py dedup --namespace research     # Find & merge duplicates
  python scripts/manage_archival.py cleanup --namespace research --days 30  # Remove entries older than 30 days
  python scripts/manage_archival.py consolidate --namespace research        # Full consolidation
  python scripts/manage_archival.py drop --namespace locomo_baseline       # Delete all entries in namespace
  python scripts/manage_archival.py smart-dedup -n research       # LLM-assisted dedup (dry run)
  python scripts/manage_archival.py smart-dedup -n research --apply  # LLM-assisted dedup (execute)
  python scripts/manage_archival.py dedup-docs                    # Deduplicate documents by content hash (dry run)
  python scripts/manage_archival.py dedup-docs --apply            # Deduplicate documents by content hash (execute)
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import datetime, timezone

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector
from tqdm import tqdm

load_dotenv()

from agent.storage import get_db_url, get_embeddings

BATCH_SIZE = 10


# ============================================================
# Helpers
# ============================================================

def get_conn():
    conn = psycopg.connect(get_db_url())
    register_vector(conn)
    return conn


def format_age(days: float) -> str:
    if days < 1:
        return f"{days * 24:.0f}h"
    if days < 30:
        return f"{days:.0f}d"
    return f"{days / 30:.1f}mo"


# ============================================================
# Commands
# ============================================================

def cmd_stats(_args):
    """Show entry counts and age distribution per namespace."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT namespace,
               COUNT(*) as cnt,
               MIN(created_at) as oldest,
               MAX(created_at) as newest,
               AVG(LENGTH(content)) as avg_len
        FROM archival_memory
        GROUP BY namespace
        ORDER BY cnt DESC
    """).fetchall()
    conn.close()

    if not rows:
        print("No entries found in archival_memory.")
        return

    now = datetime.now(timezone.utc)
    print(f"\n  {'Namespace':<25} {'Count':>6} {'Avg Len':>8} {'Oldest':>10} {'Newest':>10}")
    print("  " + "-" * 65)
    total = 0
    for ns, cnt, oldest, newest, avg_len in rows:
        oldest_age = (now - oldest.replace(tzinfo=timezone.utc)).days if oldest else 0
        newest_age = (now - newest.replace(tzinfo=timezone.utc)).days if newest else 0
        print(f"  {ns:<25} {cnt:>6} {avg_len:>7.0f}c {format_age(oldest_age):>10} {format_age(newest_age):>10}")
        total += cnt
    print("  " + "-" * 65)
    print(f"  {'Total':<25} {total:>6}")


def cmd_list(args):
    """List entries in a namespace."""
    conn = get_conn()
    namespace = args.namespace
    limit = args.limit

    rows = conn.execute(
        "SELECT id, content, metadata, created_at FROM archival_memory "
        "WHERE namespace = %s ORDER BY created_at DESC LIMIT %s",
        (namespace, limit),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"No entries in namespace '{namespace}'.")
        return

    print(f"\n  Entries in '{namespace}' (showing {len(rows)}):\n")
    for i, (eid, content, meta, created) in enumerate(rows):
        preview = content[:120].replace("\n", " ")
        if len(content) > 120:
            preview += "..."
        meta = meta if isinstance(meta, dict) else json.loads(meta)
        source = meta.get("source", "")
        age = (datetime.now(timezone.utc) - created.replace(tzinfo=timezone.utc)).days
        print(f"  [{i+1}] ({format_age(age)} ago, source={source})")
        print(f"      {preview}")
        print(f"      id={eid[:12]}...")
        print()


def cmd_dedup(args):
    """Find and merge duplicate entries within a namespace."""
    namespace = args.namespace
    threshold = args.threshold
    dry_run = args.dry_run

    conn = get_conn()

    # Fetch all entries with embeddings
    rows = conn.execute(
        "SELECT id, content, embedding FROM archival_memory WHERE namespace = %s",
        (namespace,),
    ).fetchall()

    if len(rows) < 2:
        print(f"Only {len(rows)} entries in '{namespace}', nothing to dedup.")
        conn.close()
        return

    print(f"  Scanning {len(rows)} entries in '{namespace}' for duplicates (threshold={threshold})...")

    # Build similarity groups using Union-Find
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Compare all pairs
    dup_pairs = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            vec_i = rows[i][2]
            vec_j = rows[j][2]
            # Parse embeddings if they're strings
            if isinstance(vec_i, str):
                vec_i = json.loads(vec_i)
            if isinstance(vec_j, str):
                vec_j = json.loads(vec_j)
            # Compute cosine similarity
            dot = sum(a * b for a, b in zip(vec_i, vec_j))
            norm_i = sum(a * a for a in vec_i) ** 0.5
            norm_j = sum(b * b for b in vec_j) ** 0.5
            sim = dot / (norm_i * norm_j) if norm_i > 0 and norm_j > 0 else 0
            if sim >= threshold:
                union(i, j)
                dup_pairs += 1

    # Group by cluster
    clusters: dict[int, list[int]] = {}
    for i in range(len(rows)):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    merge_clusters = {k: v for k, v in clusters.items() if len(v) >= 2}

    if not merge_clusters:
        print(f"  No duplicates found (checked {dup_pairs} pairs above threshold).")
        conn.close()
        return

    print(f"  Found {len(merge_clusters)} duplicate groups ({dup_pairs} pairs).")

    if dry_run:
        for cluster_id, indices in merge_clusters.items():
            print(f"\n  --- Group {cluster_id} ({len(indices)} entries) ---")
            for idx in indices:
                preview = rows[idx][1][:80].replace("\n", " ")
                print(f"    [{idx}] {preview}...")
        conn.close()
        return

    # Merge with LLM
    from agent.storage.embedding import DashScopeEmbeddings
    embeddings = get_embeddings()

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("LLM_API_KEY", ""),
            temperature=0,
        )
        has_llm = True
    except Exception:
        has_llm = False
        print("  [warn] LLM not available, will keep longest entry per group.")

    merged_count = 0
    deleted_count = 0

    for cluster_id, indices in tqdm(merge_clusters.items(), desc="  Merging"):
        entries = [(rows[i][0], rows[i][1]) for i in indices]

        if has_llm:
            entries_text = "\n".join(f"- {content}" for _, content in entries)
            prompt = (
                "Merge these similar entries into one concise, complete entry. "
                "Remove redundancy. Keep all unique information. "
                "Output a single paragraph.\n\n"
                f"Entries:\n{entries_text}"
            )
            try:
                response = llm.invoke(prompt)
                merged = response.content.strip()
            except Exception:
                merged = max(entries, key=lambda e: len(e[1]))[1]
        else:
            merged = max(entries, key=lambda e: len(e[1]))[1]

        # Keep the first entry, update it, delete the rest
        keep_id = entries[0][0]
        delete_ids = [eid for eid, _ in entries[1:]]

        merged_vec = Vector(embeddings.embed_query(merged))
        conn.execute(
            "UPDATE archival_memory SET content = %s, embedding = %s WHERE id = %s",
            (merged, merged_vec, keep_id),
        )
        for did in delete_ids:
            conn.execute("DELETE FROM archival_memory WHERE id = %s", (did,))
            deleted_count += 1
        merged_count += 1

    conn.commit()
    conn.close()
    print(f"  Merged {merged_count} groups, deleted {deleted_count} entries.")


def cmd_cleanup(args):
    """Remove entries older than N days."""
    namespace = args.namespace
    days = args.days
    dry_run = args.dry_run

    conn = get_conn()

    rows = conn.execute(
        "SELECT id, content, created_at FROM archival_memory "
        "WHERE namespace = %s AND created_at < NOW() - INTERVAL '%s days'",
        (namespace, days),
    ).fetchall()

    if not rows:
        print(f"  No entries older than {days} days in '{namespace}'.")
        conn.close()
        return

    print(f"  Found {len(rows)} entries older than {days} days in '{namespace}'.")

    if dry_run:
        for eid, content, created in rows[:10]:
            preview = content[:80].replace("\n", " ")
            age = (datetime.now(timezone.utc) - created.replace(tzinfo=timezone.utc)).days
            print(f"    [{format_age(age)} ago] {preview}...")
        if len(rows) > 10:
            print(f"    ... and {len(rows) - 10} more")
        conn.close()
        return

    result = conn.execute(
        "DELETE FROM archival_memory WHERE namespace = %s AND created_at < NOW() - INTERVAL '%s days'",
        (namespace, days),
    )
    conn.commit()
    conn.close()
    print(f"  Deleted {result.rowcount} entries.")


def cmd_drop(args):
    """Delete all entries in a namespace."""
    namespace = args.namespace
    dry_run = args.dry_run

    conn = get_conn()
    count = conn.execute(
        "SELECT COUNT(*) FROM archival_memory WHERE namespace = %s", (namespace,)
    ).fetchone()[0]

    if count == 0:
        print(f"  Namespace '{namespace}' is empty.")
        conn.close()
        return

    print(f"  Namespace '{namespace}' has {count} entries.")

    if dry_run:
        print(f"  (dry run — no deletion)")
        conn.close()
        return

    result = conn.execute(
        "DELETE FROM archival_memory WHERE namespace = %s", (namespace,)
    )
    conn.commit()
    conn.close()
    print(f"  Deleted {result.rowcount} entries from '{namespace}'.")


def cmd_consolidate(args):
    """Full consolidation: dedup + merge within a namespace."""
    namespace = args.namespace
    threshold = args.threshold

    print(f"\n  Consolidating namespace '{namespace}' (threshold={threshold})")
    print("  Step 1: Finding duplicates...")
    args.dry_run = False
    cmd_dedup(args)
    print("  Consolidation complete.")


# ============================================================
# Smart Dedup — LLM-assisted duplicate detection
# ============================================================

BATCH_SIZE = 5


def _llm_judge_batch(llm, pairs: list[dict]) -> list[dict]:
    """Ask LLM to judge multiple candidate pairs at once.

    Args:
        pairs: list of {"pair_id": int, "a": str, "b": str}

    Returns list of {"pair_id": int, "duplicate": bool, "reason": str, "merged": str|None}
    """
    pair_blocks = []
    for p in pairs:
        pair_blocks.append(
            f"### Pair {p['pair_id']}\n"
            f"**Entry A:**\n{p['a']}\n\n"
            f"**Entry B:**\n{p['b']}"
        )

    prompt = (
        "You are a knowledge base curator. For each pair below, judge whether the two entries are "
        "**true duplicates** (same fact/info, redundant) or **distinct** "
        "(different facts, different angles, complementary).\n\n"
        "Respond with a JSON array, one object per pair:\n"
        '[{"pair_id": 0, "duplicate": true/false, "reason": "<brief>", '
        '"merged": "<merged text if duplicate, null otherwise>"}, ...]\n\n'
        + "\n\n".join(pair_blocks)
    )

    try:
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception as e:
        return [{"pair_id": p["pair_id"], "duplicate": False, "reason": f"LLM error: {e}", "merged": None} for p in pairs]


def cmd_smart_dedup(args):
    """Smart deduplication: cosine candidates + LLM judgment."""
    namespace = args.namespace
    threshold = args.threshold
    apply = args.apply

    # Init LLM
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
        temperature=0,
    )

    conn = get_conn()
    rows = conn.execute(
        "SELECT id, content, embedding FROM archival_memory WHERE namespace = %s",
        (namespace,),
    ).fetchall()

    if len(rows) < 2:
        print(f"  Only {len(rows)} entries in '{namespace}', nothing to check.")
        conn.close()
        return

    print(f"  Scanning {len(rows)} entries for candidates (cosine >= {threshold})...")

    # Find candidate pairs
    candidates = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            vec_i = rows[i][2]
            vec_j = rows[j][2]
            if isinstance(vec_i, str):
                vec_i = json.loads(vec_i)
            if isinstance(vec_j, str):
                vec_j = json.loads(vec_j)
            dot = sum(a * b for a, b in zip(vec_i, vec_j))
            norm_i = sum(a * a for a in vec_i) ** 0.5
            norm_j = sum(b * b for b in vec_j) ** 0.5
            sim = dot / (norm_i * norm_j) if norm_i > 0 and norm_j > 0 else 0
            if sim >= threshold:
                candidates.append((i, j, sim))

    if not candidates:
        print(f"  No candidate pairs found above threshold {threshold}.")
        conn.close()
        return

    # Sort by similarity descending
    candidates.sort(key=lambda x: x[2], reverse=True)

    # High-confidence auto-duplicates (cosine >= 0.95 skip LLM)
    AUTO_DUP_THRESHOLD = 0.95
    auto_dups = [(i, j, sim) for i, j, sim in candidates if sim >= AUTO_DUP_THRESHOLD]
    llm_candidates = [(i, j, sim) for i, j, sim in candidates if sim < AUTO_DUP_THRESHOLD]

    print(f"  Found {len(candidates)} candidate pairs.")
    print(f"  Auto-duplicate (cosine >= {AUTO_DUP_THRESHOLD}): {len(auto_dups)}")
    print(f"  LLM judgment needed: {len(llm_candidates)}")

    results = []

    # Auto-duplicates: no LLM needed
    for i, j, sim in auto_dups:
        results.append({
            "idx_a": i, "idx_b": j,
            "id_a": rows[i][0], "id_b": rows[j][0],
            "cosine": sim,
            "duplicate": True,
            "reason": f"auto (cosine={sim:.3f} >= {AUTO_DUP_THRESHOLD})",
            "merged": None,
        })

    # Batched LLM judgment
    if llm_candidates:
        batches = []
        for k in range(0, len(llm_candidates), BATCH_SIZE):
            batch = []
            for idx, (i, j, sim) in enumerate(llm_candidates[k:k + BATCH_SIZE], start=k):
                batch.append({"pair_id": idx, "a": rows[i][1], "b": rows[j][1]})
            batches.append(batch)

        print(f"  Judging in {len(batches)} batches (batch_size={BATCH_SIZE})...\n")

        for batch in tqdm(batches, desc="  Judging"):
            judgments = _llm_judge_batch(llm, batch)
            j_map = {j["pair_id"]: j for j in judgments}
            for p in batch:
                pid = p["pair_id"]
                i, j, sim = llm_candidates[pid]
                judgment = j_map.get(pid, {})
                results.append({
                    "idx_a": i, "idx_b": j,
                    "id_a": rows[i][0], "id_b": rows[j][0],
                    "cosine": sim,
                    "duplicate": judgment.get("duplicate", False),
                    "reason": judgment.get("reason", ""),
                    "merged": judgment.get("merged"),
                })

    # Display results
    confirmed = [r for r in results if r["duplicate"]]
    rejected = [r for r in results if not r["duplicate"]]

    print(f"\n  === Results ===")
    print(f"  Confirmed duplicates: {len(confirmed)}")
    print(f"  Rejected (distinct):  {len(rejected)}")

    if confirmed:
        print(f"\n  --- Confirmed Duplicates ---")
        for r in confirmed:
            preview_a = rows[r["idx_a"]][1][:60].replace("\n", " ")
            preview_b = rows[r["idx_b"]][1][:60].replace("\n", " ")
            print(f"  [cosine={r['cosine']:.3f}] {r['reason']}")
            print(f"    A: {preview_a}...")
            print(f"    B: {preview_b}...")
            print()

    if rejected:
        print(f"  --- Rejected (Distinct) ---")
        for r in rejected:
            preview_a = rows[r["idx_a"]][1][:60].replace("\n", " ")
            preview_b = rows[r["idx_b"]][1][:60].replace("\n", " ")
            print(f"  [cosine={r['cosine']:.3f}] {r['reason']}")
            print(f"    A: {preview_a}...")
            print(f"    B: {preview_b}...")
            print()

    if not apply:
        print("  Dry run. Use --apply to merge confirmed duplicates.")
        conn.close()
        return

    # Merge confirmed duplicates
    if not confirmed:
        conn.close()
        return

    embeddings = get_embeddings()
    merged_count = 0
    deleted_count = 0

    # Track which entries have already been merged/deleted
    deleted_ids = set()

    for r in tqdm(confirmed, desc="  Merging"):
        id_a, id_b = r["id_a"], r["id_b"]
        if id_a in deleted_ids or id_b in deleted_ids:
            continue

        merged_text = r.get("merged")
        if not merged_text:
            merged_text = max(rows[r["idx_a"]][1], rows[r["idx_b"]][1], key=len)

        merged_vec = Vector(embeddings.embed_query(merged_text))
        conn.execute(
            "UPDATE archival_memory SET content = %s, embedding = %s WHERE id = %s",
            (merged_text, merged_vec, id_a),
        )
        conn.execute("DELETE FROM archival_memory WHERE id = %s", (id_b,))
        deleted_ids.add(id_b)
        merged_count += 1
        deleted_count += 1

    conn.commit()
    conn.close()
    print(f"  Merged {merged_count} pairs, deleted {deleted_count} entries.")


# ============================================================
# Document Dedup — clean duplicate documents by content hash
# ============================================================

def cmd_dedup_docs(args):
    """Find and remove duplicate documents and orphaned ingested chunks."""
    apply = args.apply
    conn = get_conn()

    # Ensure content_hash column exists
    conn.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'documents' AND column_name = 'content_hash'
            ) THEN
                ALTER TABLE documents ADD COLUMN content_hash TEXT;
                UPDATE documents SET content_hash = encode(sha256(convert_to(content_full, 'UTF8')), 'hex')
                    WHERE content_hash IS NULL;
                ALTER TABLE documents ALTER COLUMN content_hash SET NOT NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash ON documents (content_hash);
            END IF;
        END$$;
    """)
    conn.commit()

    total_deleted_docs = 0
    total_deleted_chunks = 0

    # --- Part 1: Duplicate documents by content_hash ---
    rows = conn.execute("""
        SELECT content_hash, array_agg(id ORDER BY created_at) AS ids,
               array_agg(title ORDER BY created_at) AS titles,
               array_agg(created_at ORDER BY created_at) AS dates,
               COUNT(*) AS cnt
        FROM documents
        GROUP BY content_hash
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC
    """).fetchall()

    if rows:
        total_dup_docs = sum(r[4] - 1 for r in rows)
        print(f"  [Documents] Found {len(rows)} content groups with duplicates ({total_dup_docs} to remove).\n")

        for content_hash, ids, titles, dates, cnt in rows:
            print(f"  Content hash: {content_hash[:16]}... ({cnt} copies)")
            for i, (doc_id, title, date) in enumerate(zip(ids, titles, dates)):
                tag = "KEEP" if i == 0 else "DELETE"
                print(f"    [{tag}] {title} (created {date}, id={doc_id[:12]}...)")
            print()

        if apply:
            for content_hash, ids, titles, dates, cnt in rows:
                for doc_id in ids[1:]:
                    result = conn.execute(
                        "DELETE FROM archival_memory WHERE document_id = %s", (doc_id,)
                    )
                    total_deleted_chunks += result.rowcount
                    conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                    total_deleted_docs += 1
    else:
        print("  [Documents] No duplicate documents found.\n")

    # --- Part 2: Orphaned ingested chunks (document_id IS NULL) with matching content ---
    orphan_count = conn.execute("""
        SELECT COUNT(*) FROM archival_memory
        WHERE namespace = 'ingested' AND document_id IS NULL
    """).fetchone()[0]

    if orphan_count > 0:
        matched = conn.execute("""
            SELECT COUNT(*) FROM archival_memory o
            WHERE o.namespace = 'ingested' AND o.document_id IS NULL
            AND EXISTS (
                SELECT 1 FROM archival_memory l
                WHERE l.namespace = 'ingested' AND l.document_id IS NOT NULL
                AND l.content = o.content
            )
        """).fetchone()[0]

        print(f"  [Ingested Chunks] Found {orphan_count} orphaned entries (no document_id).")
        print(f"  [Ingested Chunks] {matched} have matching content in linked entries — safe to delete.\n")

        if apply and matched > 0:
            result = conn.execute("""
                DELETE FROM archival_memory o
                WHERE o.namespace = 'ingested' AND o.document_id IS NULL
                AND EXISTS (
                    SELECT 1 FROM archival_memory l
                    WHERE l.namespace = 'ingested' AND l.document_id IS NOT NULL
                    AND l.content = o.content
                )
            """)
            total_deleted_chunks += result.rowcount
    else:
        print("  [Ingested Chunks] No orphaned entries found.\n")

    if apply:
        conn.commit()
        print(f"  Total: deleted {total_deleted_docs} documents and {total_deleted_chunks} archival chunks.")
    else:
        print("  Dry run. Use --apply to execute deletion.")
    conn.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Archival Memory management tool")
    sub = parser.add_subparsers(dest="command")

    # stats
    sub.add_parser("stats", help="Show entry counts per namespace")

    # list
    p_list = sub.add_parser("list", help="List entries in a namespace")
    p_list.add_argument("--namespace", "-n", required=True)
    p_list.add_argument("--limit", "-l", type=int, default=20)

    # dedup
    p_dedup = sub.add_parser("dedup", help="Find and merge duplicates")
    p_dedup.add_argument("--namespace", "-n", required=True)
    p_dedup.add_argument("--threshold", "-t", type=float, default=0.85)
    p_dedup.add_argument("--dry-run", action="store_true")

    # cleanup
    p_clean = sub.add_parser("cleanup", help="Remove entries older than N days")
    p_clean.add_argument("--namespace", "-n", required=True)
    p_clean.add_argument("--days", "-d", type=int, required=True)
    p_clean.add_argument("--dry-run", action="store_true")

    # drop
    p_drop = sub.add_parser("drop", help="Delete all entries in a namespace")
    p_drop.add_argument("--namespace", "-n", required=True)
    p_drop.add_argument("--dry-run", action="store_true")

    # consolidate
    p_con = sub.add_parser("consolidate", help="Full consolidation (dedup + merge)")
    p_con.add_argument("--namespace", "-n", required=True)
    p_con.add_argument("--threshold", "-t", type=float, default=0.85)

    # smart-dedup
    p_smart = sub.add_parser("smart-dedup", help="LLM-assisted duplicate detection")
    p_smart.add_argument("--namespace", "-n", required=True)
    p_smart.add_argument("--threshold", "-t", type=float, default=0.7,
                         help="Cosine similarity threshold for candidates (default 0.7)")
    p_smart.add_argument("--apply", action="store_true",
                         help="Execute merges (default is dry run)")

    # dedup-docs
    p_ddocs = sub.add_parser("dedup-docs", help="Remove duplicate documents by content hash")
    p_ddocs.add_argument("--apply", action="store_true",
                         help="Execute deletion (default is dry run)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    {"stats": cmd_stats, "list": cmd_list, "dedup": cmd_dedup,
     "cleanup": cmd_cleanup, "drop": cmd_drop, "consolidate": cmd_consolidate,
     "smart-dedup": cmd_smart_dedup, "dedup-docs": cmd_dedup_docs}[args.command](args)
