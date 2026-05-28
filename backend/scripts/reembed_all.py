"""Re-embed all existing data after switching embedding models.

Usage:
    python scripts/reembed_all.py
    python scripts/reembed_all.py --dry-run
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from tqdm import tqdm

from agent.storage import get_db_url, get_embeddings


def reembed_table(table: str, batch_size: int = 32) -> int:
    conn = psycopg.connect(get_db_url())
    register_vector(conn)

    rows = conn.execute(f"SELECT id, content FROM {table}").fetchall()
    total = len(rows)
    if total == 0:
        conn.close()
        print(f"  {table}: 0 rows, skipped")
        return 0

    embeddings = get_embeddings()
    pbar = tqdm(total=total, desc=f"  {table}", unit="row", ncols=80)

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        ids = [r[0] for r in batch]
        contents = [r[1] for r in batch]
        vectors = embeddings.embed_documents(contents)

        for entry_id, vec in zip(ids, vectors):
            conn.execute(
                f"UPDATE {table} SET embedding = %s WHERE id = %s",
                (Vector(vec), entry_id),
            )
        conn.commit()
        pbar.update(len(batch))

    pbar.close()
    conn.close()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    print(f"Model: {model}")

    conn = psycopg.connect(get_db_url())
    arch_count = conn.execute("SELECT count(*) FROM archival_memory").fetchone()[0]
    rec_count = conn.execute("SELECT count(*) FROM recall_memory").fetchone()[0]
    conn.close()
    total = arch_count + rec_count

    print(f"archival_memory: {arch_count} rows")
    print(f"recall_memory:   {rec_count} rows")
    print(f"Total to re-embed: {total} rows\n")

    if args.dry_run:
        print("Dry run — no changes made.")
        return

    t_start = time.time()
    reembed_table("archival_memory")
    reembed_table("recall_memory")
    elapsed = time.time() - t_start

    print(f"\nDone. {total} rows re-embedded in {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
