"""Standalone database explorer for Knowledge Agent.

Usage:
    python scripts/db_explorer.py list archival [--namespace NS] [--limit N]
    python scripts/db_explorer.py list recall [--thread ID] [--limit N]
    python scripts/db_explorer.py search archival "query" [--namespace NS] [--limit N]
    python scripts/db_explorer.py search recall "query" [--thread ID] [--limit N]
    python scripts/db_explorer.py stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# Load .env from backend root
_backend = Path(__file__).resolve().parent.parent
load_dotenv(_backend / ".env")

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector


def get_conn():
    url = os.environ["DATABASE_URL"]
    conn = psycopg.connect(url)
    register_vector(conn)
    return conn


def get_embeddings():
    from agent.storage.embedding import DashScopeEmbeddings

    return DashScopeEmbeddings()


def fmt_content(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


# ── list ──────────────────────────────────────────────────────────────

def cmd_list_archival(args):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, namespace, content, metadata, created_at "
        "FROM archival_memory WHERE namespace = %s "
        "ORDER BY created_at DESC LIMIT %s",
        (args.namespace, args.limit),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"No entries in archival_memory (namespace='{args.namespace}')")
        return

    print(f"archival_memory [{args.namespace}] — {len(rows)} entries\n")
    for row in rows:
        meta = row[3] if isinstance(row[3], dict) else json.loads(row[3])
        ts = row[4].strftime("%Y-%m-%d %H:%M") if row[4] else "?"
        print(f"  [{row[0][:8]}] {ts}  {fmt_content(row[2])}")
        if meta:
            tags = " ".join(f"{k}={v}" for k, v in meta.items() if k != "timestamp")
            if tags:
                print(f"           {tags}")
    print()


def cmd_list_recall(args):
    conn = get_conn()
    if args.thread:
        rows = conn.execute(
            "SELECT id, thread_id, role, content, created_at "
            "FROM recall_memory WHERE thread_id = %s "
            "ORDER BY created_at DESC LIMIT %s",
            (args.thread, args.limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, thread_id, role, content, created_at "
            "FROM recall_memory ORDER BY created_at DESC LIMIT %s",
            (args.limit,),
        ).fetchall()
    conn.close()

    if not rows:
        print("No entries in recall_memory")
        return

    print(f"recall_memory — {len(rows)} entries\n")
    for row in rows:
        ts = row[4].strftime("%Y-%m-%d %H:%M") if row[4] else "?"
        print(f"  [{row[0][:8]}] {row[1][:12]:<12} {row[2]:<9} {ts}  {fmt_content(row[3])}")
    print()


# ── search ────────────────────────────────────────────────────────────

def cmd_search_archival(args):
    emb = get_embeddings()
    query_vec = Vector(embed_emb_query(emb, args.query))
    conn = get_conn()

    rows = conn.execute(
        """
        SELECT id, namespace, content, metadata,
               0.7 * (1 - (embedding <=> %s::vector))
                 + 0.3 * ts_rank(content_tsv, plainto_tsquery('simple', %s))
               AS score
        FROM archival_memory
        WHERE namespace = %s
          AND (content_tsv @@ plainto_tsquery('simple', %s)
               OR 1 - (embedding <=> %s::vector) > 0.2)
        ORDER BY score DESC
        LIMIT %s
        """,
        (query_vec, args.query, args.namespace, args.query, query_vec, args.limit),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"No results for '{args.query}' in archival_memory [{args.namespace}]")
        return

    print(f"archival search: '{args.query}' [{args.namespace}] — {len(rows)} results\n")
    for row in rows:
        meta = row[3] if isinstance(row[3], dict) else json.loads(row[3])
        src = meta.get("source", meta.get("topic", ""))
        print(f"  [{row[0][:8]}] score={row[4]:.3f}  {src}")
        print(f"    {fmt_content(row[2], 200)}")
        print()


def cmd_search_recall(args):
    emb = get_embeddings()
    query_vec = Vector(embed_emb_query(emb, args.query))
    conn = get_conn()

    if args.thread:
        rows = conn.execute(
            """
            SELECT id, thread_id, role, content,
                   1 - (embedding <=> %s::vector) AS score
            FROM recall_memory
            WHERE thread_id = %s
            ORDER BY score DESC
            LIMIT %s
            """,
            (query_vec, args.thread, args.limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, thread_id, role, content,
                   1 - (embedding <=> %s::vector) AS score
            FROM recall_memory
            ORDER BY score DESC
            LIMIT %s
            """,
            (query_vec, args.limit),
        ).fetchall()
    conn.close()

    if not rows:
        print(f"No results for '{args.query}' in recall_memory")
        return

    print(f"recall search: '{args.query}' — {len(rows)} results\n")
    for row in rows:
        print(f"  [{row[0][:8]}] {row[1][:12]:<12} {row[2]:<9} score={row[4]:.3f}")
        print(f"    {fmt_content(row[3], 200)}")
        print()


def embed_emb_query(emb, text: str) -> list[float]:
    return emb.embed_query(text)


# ── stats ─────────────────────────────────────────────────────────────

def cmd_stats(_args):
    conn = get_conn()

    arch_count = conn.execute("SELECT COUNT(*) FROM archival_memory").fetchone()[0]
    recall_count = conn.execute("SELECT COUNT(*) FROM recall_memory").fetchone()[0]

    namespaces = conn.execute(
        "SELECT namespace, COUNT(*) FROM archival_memory GROUP BY namespace ORDER BY COUNT(*) DESC"
    ).fetchall()

    threads = conn.execute(
        "SELECT thread_id, COUNT(*) FROM recall_memory GROUP BY thread_id ORDER BY COUNT(*) DESC LIMIT 10"
    ).fetchall()

    conn.close()

    print("Knowledge Agent Database Stats\n")
    print(f"  archival_memory : {arch_count} entries")
    print(f"  recall_memory   : {recall_count} entries")

    if namespaces:
        print(f"\nArchival namespaces:")
        for ns, cnt in namespaces:
            print(f"  {ns:<20} {cnt}")

    if threads:
        print(f"\nRecall threads (top 10):")
        for tid, cnt in threads:
            print(f"  {tid:<20} {cnt}")
    print()


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Agent database explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="List database entries")
    list_sub = p_list.add_subparsers(dest="table")

    p_la = list_sub.add_parser("archival", help="List archival memory entries")
    p_la.add_argument("--namespace", default="default")
    p_la.add_argument("--limit", type=int, default=20)

    p_lr = list_sub.add_parser("recall", help="List recall memory entries")
    p_lr.add_argument("--thread", default=None)
    p_lr.add_argument("--limit", type=int, default=20)

    # search
    p_search = sub.add_parser("search", help="Semantic search")
    search_sub = p_search.add_subparsers(dest="table")

    p_sa = search_sub.add_parser("archival", help="Search archival memory")
    p_sa.add_argument("query")
    p_sa.add_argument("--namespace", default="default")
    p_sa.add_argument("--limit", type=int, default=5)

    p_sr = search_sub.add_parser("recall", help="Search recall memory")
    p_sr.add_argument("query")
    p_sr.add_argument("--thread", default=None)
    p_sr.add_argument("--limit", type=int, default=5)

    # stats
    sub.add_parser("stats", help="Show database statistics")

    args = parser.parse_args()

    if args.command == "list":
        if args.table == "archival":
            cmd_list_archival(args)
        elif args.table == "recall":
            cmd_list_recall(args)
        else:
            parser.parse_args(["list", "--help"])
    elif args.command == "search":
        if args.table == "archival":
            cmd_search_archival(args)
        elif args.table == "recall":
            cmd_search_recall(args)
        else:
            parser.parse_args(["search", "--help"])
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
