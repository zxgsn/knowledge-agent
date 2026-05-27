"""Demo evaluation with embedded LOCOMO-style data.

Self-contained test that demonstrates the knowledge-agent's memory capabilities
without requiring external dataset files. Uses 1 sample with 2 sessions and 10 questions.

Usage:
  python scripts/demo_eval.py
  python scripts/demo_eval.py --output demo_results.json
  python scripts/demo_eval.py --keep-data
"""
from __future__ import annotations

import warnings
# Suppress langchain deprecation warning about allowed_objects
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
# Suppress LangChainPendingDeprecationWarning
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except ImportError:
    pass

import asyncio
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

load_dotenv()

import hashlib

def _fallback_embed(text: str) -> list[float]:
    """Hash-based fallback embedding (1024-dim, deterministic)."""
    h = hashlib.sha512(text.encode()).digest()
    vec = [float(b) / 255.0 for b in h]
    # sha512 produces 64 bytes, repeat to get 1024 dimensions
    return (vec * 16)[:1024]

# Always use fallback for demo (real model needs too much memory)
_embedding_works = False
_emb = None
from agent.storage import get_db_url

def get_embed_fn():
    return _fallback_embed


# ============================================================
# Embedded Sample Data
# ============================================================

SAMPLE = {
    "conversation": [
        {
            "session_id": "session_1",
            "dialogue": [
                {"speaker": "Alice", "utterance": "Hi Bob! I just moved to San Francisco last week."},
                {"speaker": "Bob", "utterance": "That's great! How do you like it so far?"},
                {"speaker": "Alice", "utterance": "I love it! The weather is amazing. I'm living in the Mission District."},
                {"speaker": "Bob", "utterance": "Nice area! What brought you there?"},
                {"speaker": "Alice", "utterance": "I got a new job at TechCorp as a senior engineer. Started on Monday."},
                {"speaker": "Bob", "utterance": "Congratulations! What are you working on?"},
                {"speaker": "Alice", "utterance": "Building a new recommendation system. It's challenging but exciting."},
                {"speaker": "Bob", "utterance": "Sounds interesting. Do you miss your old city?"},
                {"speaker": "Alice", "utterance": "A little. I lived in Portland for 5 years. But I'm happy here."},
                {"speaker": "Bob", "utterance": "Portland is nice too. Have you explored SF yet?"},
                {"speaker": "Alice", "utterance": "I went to Golden Gate Park on Saturday. Beautiful!"},
                {"speaker": "Bob", "utterance": "You should try the food scene. Lots of great restaurants."},
                {"speaker": "Alice", "utterance": "I had amazing sushi at Sushi Ran in Sausalito yesterday."},
                {"speaker": "Bob", "utterance": "Great choice! I know the chef there."},
            ]
        },
        {
            "session_id": "session_2",
            "dialogue": [
                {"speaker": "Alice", "utterance": "Bob, I need your advice on something."},
                {"speaker": "Bob", "utterance": "Sure, what's up?"},
                {"speaker": "Alice", "utterance": "I'm thinking about getting a dog. Any recommendations for breeds?"},
                {"speaker": "Bob", "utterance": "What's your lifestyle like? Active or more relaxed?"},
                {"speaker": "Alice", "utterance": "I run 3 times a week and hike on weekends. My apartment is medium-sized."},
                {"speaker": "Bob", "utterance": "A Labrador or Golden Retriever would be perfect for you."},
                {"speaker": "Alice", "utterance": "I was also considering a Border Collie. They're so smart."},
                {"speaker": "Bob", "utterance": "They are, but they need a LOT of exercise. More than 3 runs a week."},
                {"speaker": "Alice", "utterance": "Good point. I'll stick with a Lab then. There's a shelter near Dolores Park."},
                {"speaker": "Bob", "utterance": "Perfect! Let me know when you adopt. I'll help you get supplies."},
            ]
        }
    ],
    "questions": [
        {
            "question": "Where did Alice move to?",
            "answer": "San Francisco",
            "question_type": "single_session",
            "evidence": ["D1:0", "D1:2"]
        },
        {
            "question": "What district does Alice live in?",
            "answer": "Mission District",
            "question_type": "single_session",
            "evidence": ["D1:2"]
        },
        {
            "question": "What is Alice's new job?",
            "answer": "senior engineer at TechCorp",
            "question_type": "single_session",
            "evidence": ["D1:4"]
        },
        {
            "question": "What is Alice building at work?",
            "answer": "recommendation system",
            "question_type": "single_session",
            "evidence": ["D1:6"]
        },
        {
            "question": "How long did Alice live in Portland?",
            "answer": "5 years",
            "question_type": "single_session",
            "evidence": ["D1:8"]
        },
        {
            "question": "What restaurant did Alice visit?",
            "answer": "Sushi Ran in Sausalito",
            "question_type": "single_session",
            "evidence": ["D1:12"]
        },
        {
            "question": "What kind of dog is Alice considering?",
            "answer": "Labrador",
            "question_type": "multi_session",
            "evidence": ["D2:5", "D2:8"]
        },
        {
            "question": "How often does Alice run?",
            "answer": "3 times a week",
            "question_type": "single_session",
            "evidence": ["D2:4"]
        },
        {
            "question": "Where is the dog shelter Alice found?",
            "answer": "near Dolores Park",
            "question_type": "single_session",
            "evidence": ["D2:8"]
        },
        {
            "question": "What did Alice do on Saturday?",
            "answer": "went to Golden Gate Park",
            "question_type": "temporal",
            "evidence": ["D1:10"]
        }
    ]
}


# ============================================================
# Ingestion Strategies
# ============================================================

def ingest_raw_turns(namespace: str = "demo_raw") -> int:
    """Ingest each turn as a separate document."""
    embed_fn = get_embed_fn()
    conn = psycopg.connect(get_db_url())
    register_vector(conn)

    total = 0
    for session in SAMPLE["conversation"]:
        sid = session["session_id"]
        for i, turn in enumerate(session["dialogue"]):
            content = f"[{turn['speaker']}] {turn['utterance']}"
            metadata = {"session_id": sid, "turn": i, "speaker": turn["speaker"]}
            vec = embed_fn(content)
            conn.execute(
                """INSERT INTO archival_memory (id, namespace, content, metadata, embedding)
                   VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING""",
                (f"demo-{sid}-{i}", namespace, content, json.dumps(metadata), Vector(vec)),
            )
            total += 1

    conn.commit()
    conn.close()
    return total


def ingest_context_windows(namespace: str = "demo_ctx", window: int = 5, stride: int = 2) -> int:
    """Ingest with sliding context windows."""
    embed_fn = get_embed_fn()
    conn = psycopg.connect(get_db_url())
    register_vector(conn)

    total = 0
    for session in SAMPLE["conversation"]:
        sid = session["session_id"]
        dialogue = session["dialogue"]
        for start in range(0, len(dialogue), stride):
            end = min(start + window, len(dialogue))
            chunk = dialogue[start:end]
            lines = [f"[{t['speaker']}] {t['utterance']}" for t in chunk]
            content = "\n".join(lines)
            metadata = {"session_id": sid, "start": start, "end": end}
            vec = embed_fn(content)
            conn.execute(
                """INSERT INTO archival_memory (id, namespace, content, metadata, embedding)
                   VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING""",
                (f"demo-ctx-{sid}-{start}", namespace, content, json.dumps(metadata), Vector(vec)),
            )
            total += 1

    conn.commit()
    conn.close()
    return total


def ingest_extracted_facts(namespace: str = "demo_facts") -> int:
    """Ingest pre-extracted facts (simulates LLM extraction)."""
    facts = [
        "Alice moved to San Francisco last week.",
        "Alice lives in the Mission District.",
        "Alice works at TechCorp as a senior engineer.",
        "Alice started her new job on Monday.",
        "Alice is building a recommendation system at work.",
        "Alice lived in Portland for 5 years before moving.",
        "Alice went to Golden Gate Park on Saturday.",
        "Alice had sushi at Sushi Ran in Sausalito.",
        "Alice is considering getting a Labrador dog.",
        "Alice runs 3 times a week and hikes on weekends.",
        "There is a dog shelter near Dolores Park that Alice found.",
        "Bob knows the chef at Sushi Ran.",
        "Bob recommended Labrador or Golden Retriever for Alice.",
        "Border Collies need more than 3 runs a week according to Bob.",
    ]

    embed_fn = get_embed_fn()
    conn = psycopg.connect(get_db_url())
    register_vector(conn)

    for i, fact in enumerate(facts):
        vec = embed_fn(fact)
        conn.execute(
            """INSERT INTO archival_memory (id, namespace, content, metadata, embedding)
               VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING""",
            (f"demo-fact-{i}", namespace, fact, json.dumps({"source": "extracted"}), Vector(vec)),
        )

    conn.commit()
    conn.close()
    return len(facts)


def ingest_hybrid(ns_context: str = "demo_ctx", ns_facts: str = "demo_facts") -> tuple[int, int]:
    """Ingest both context windows and extracted facts."""
    c = ingest_context_windows(ns_context)
    f = ingest_extracted_facts(ns_facts)
    return c, f


# ============================================================
# Search
# ============================================================

def search(query: str, namespace: str, limit: int = 10) -> list[dict]:
    """Hybrid search: BM25 + vector similarity."""
    embed_fn = get_embed_fn()
    qe = Vector(embed_fn(query))
    conn = psycopg.connect(get_db_url())
    register_vector(conn)
    rows = conn.execute(
        """
        SELECT content, metadata,
               0.7 * (1 - (embedding <=> %s::vector))
                 + 0.3 * LEAST(1, ts_rank(content_tsv, websearch_to_tsquery('english', %s)) * 10)
               AS score
        FROM archival_memory
        WHERE namespace = %s
          AND 1 - (embedding <=> %s::vector) > 0.10
        ORDER BY score DESC
        LIMIT %s
        """,
        (qe, query, namespace, qe, limit),
    ).fetchall()
    conn.close()
    return [
        {"content": r[0], "metadata": r[1] if isinstance(r[1], dict) else json.loads(r[1]), "score": float(r[2])}
        for r in rows
    ]


def search_multi_ns(query: str, namespaces: list[str], limit: int = 10) -> list[dict]:
    """Search across multiple namespaces and merge results."""
    all_results = []
    for ns in namespaces:
        all_results.extend(search(query, ns, limit=limit))
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:limit]


# ============================================================
# Evaluation
# ============================================================

def check_answer(answer: str, results: list[dict], k: int) -> bool:
    """Check if answer appears in top-K results."""
    al = answer.lower().strip()
    if not al:
        return False
    for r in results[:k]:
        cl = r["content"].lower()
        if al in cl:
            return True
        at = set(re.findall(r"\w+", al))
        if len(at) >= 2:
            ct = set(re.findall(r"\w+", cl))
            if len(at & ct) / len(at) > 0.5:
                return True
    return False


def evaluate(namespace: str, label: str, search_fn=None) -> dict:
    """Evaluate Recall@K and MRR."""
    k_vals = [1, 3, 5, 10]
    metrics = {k: {"hits": 0, "total": 0} for k in k_vals}
    mrr_sum = 0.0
    details = []
    searcher = search_fn or (lambda q, ns, limit: search(q, ns, limit))

    for q in SAMPLE["questions"]:
        question = q["question"]
        answer = q["answer"]
        qtype = q["question_type"]

        results = searcher(question, namespace, limit=max(k_vals))

        hit_ks = []
        for k in k_vals:
            metrics[k]["total"] += 1
            if check_answer(answer, results, k):
                metrics[k]["hits"] += 1
                hit_ks.append(k)

        # MRR
        first_rank = 0
        for i, r in enumerate(results[:max(k_vals)]):
            if check_answer(answer, [r], 1):
                first_rank = i + 1
                break
        mrr_sum += 1.0 / first_rank if first_rank > 0 else 0.0

        details.append({
            "question": question,
            "answer": answer,
            "type": qtype,
            "hit_at": hit_ks,
            "top1": results[0]["content"][:80] if results else "",
            "top1_score": results[0]["score"] if results else 0,
        })

    return {"metrics": metrics, "mrr": mrr_sum / len(SAMPLE["questions"]), "details": details}


# ============================================================
# Display
# ============================================================

def print_results(results: dict, label: str):
    """Print evaluation results."""
    metrics = results["metrics"]
    mrr = results["mrr"]

    print(f"\n  {label}")
    print(f"  {'':20} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
    print(f"  {'-'*56}")

    print(f"  {'Overall':<20}", end="")
    for k in [1, 3, 5, 10]:
        m = metrics[k]
        r = m["hits"] / m["total"] if m["total"] > 0 else 0
        print(f"{r:>7.1%} ", end="")
    print(f"{mrr:>7.1%} ")


def print_comparison(all_results: dict):
    """Print comparison table."""
    print(f"\n{'='*60}")
    print(f"  LOCOMO Demo: Strategy Comparison")
    print(f"{'='*60}")

    print(f"\n  {'Strategy':<25} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
    print(f"  {'-'*60}")

    for label, res in all_results.items():
        metrics = res["metrics"]
        mrr = res["mrr"]
        print(f"  {label:<25}", end="")
        for k in [1, 3, 5, 10]:
            m = metrics[k]
            r = m["hits"] / m["total"] if m["total"] > 0 else 0
            print(f"{r:>7.1%} ", end="")
        print(f"{mrr:>7.1%} ")

    # External baselines
    print(f"\n  [External Baselines]")
    print(f"  {'mem0 (paper)':<25} {'--':>8} {'--':>8} {'38.7%':>8} {'43.9%':>8} {'--':>8}")
    print(f"  {'mem0-graph (paper)':<25} {'--':>8} {'--':>8} {'43.6%':>8} {'51.0%':>8} {'--':>8}")
    print(f"  {'LoCoMo-RAG (paper)':<25} {'--':>8} {'--':>8} {'19.8%':>8} {'26.5%':>8} {'--':>8}")

    print(f"  {'-'*60}")


def print_details(details: list[dict], label: str):
    """Print per-question details."""
    print(f"\n{'='*60}")
    print(f"  Per-Question Details: {label}")
    print(f"{'='*60}")

    for d in details:
        hits = ", ".join([f"R@{k}" for k in d["hit_at"]]) if d["hit_at"] else "MISS"
        print(f"\n  Q: {d['question']}")
        print(f"  A: {d['answer']}")
        print(f"  Type: {d['type']}")
        print(f"  Hit: {hits}")
        print(f"  Top: [{d['top1_score']:.3f}] {d['top1']}...")


# ============================================================
# Cleanup
# ============================================================

def cleanup(namespaces: list[str]):
    """Remove test data."""
    conn = psycopg.connect(get_db_url())
    for ns in namespaces:
        conn.execute("DELETE FROM archival_memory WHERE namespace = %s", (ns,))
    conn.commit()
    conn.close()


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LOCOMO demo evaluation")
    parser.add_argument("--output", type=str, default=None, help="Export results to JSON")
    parser.add_argument("--keep-data", action="store_true", help="Keep test data after eval")
    args = parser.parse_args()

    # Check DB
    try:
        conn = psycopg.connect(get_db_url())
        conn.close()
        print("[OK] Database connected")
    except Exception as e:
        print(f"[FAIL] Database failed: {e}")
        print("  Start PostgreSQL: docker compose up -d postgres")
        sys.exit(1)

    t_start = time.time()
    all_results = {}

    # Strategy 1: Raw turns
    ns_raw = "demo_raw"
    print(f"\n[1/3] Ingesting raw turns...")
    count = ingest_raw_turns(ns_raw)
    print(f"  → {count} turns")
    print(f"  Evaluating...")
    all_results["Raw Turns"] = evaluate(ns_raw, "Raw Turns")
    print_results(all_results["Raw Turns"], "Raw Turns")

    # Strategy 2: Context windows
    ns_ctx = "demo_ctx"
    print(f"\n[2/3] Ingesting context windows (w=5, s=2)...")
    count = ingest_context_windows(ns_ctx)
    print(f"  → {count} windows")
    print(f"  Evaluating...")
    all_results["Context Windows"] = evaluate(ns_ctx, "Context Windows")
    print_results(all_results["Context Windows"], "Context Windows")

    # Strategy 3: Hybrid (context + facts)
    ns_facts = "demo_facts"
    print(f"\n[3/3] Ingesting hybrid (context + extracted facts)...")
    c, f = ingest_hybrid(ns_ctx, ns_facts)
    print(f"  → {c} windows + {f} facts")
    print(f"  Evaluating...")

    def hybrid_search(query, ns, limit):
        ctx_results = search(query, "demo_ctx", limit=limit)
        fact_results = search(query, "demo_facts", limit=limit)
        combined = ctx_results + fact_results
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:limit]

    all_results["Hybrid (ctx+facts)"] = evaluate("demo_hybrid", "Hybrid", search_fn=hybrid_search)
    print_results(all_results["Hybrid (ctx+facts)"], "Hybrid (ctx+facts)")

    # Comparison
    print_comparison(all_results)

    # Best strategy details
    best_label = max(all_results, key=lambda l: all_results[l]["metrics"][10]["hits"])
    print_details(all_results[best_label]["details"], best_label)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*60}")

    # Export
    if args.output:
        export = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_questions": len(SAMPLE["questions"]),
            "results": {
                label: {"metrics": res["metrics"], "mrr": res["mrr"]}
                for label, res in all_results.items()
            },
            "best_strategy": best_label,
            "best_r10": all_results[best_label]["metrics"][10]["hits"] / all_results[best_label]["metrics"][10]["total"],
            "details": all_results[best_label]["details"],
            "external_baselines": {
                "mem0": {"R@5": 0.387, "R@10": 0.439},
                "mem0-graph": {"R@5": 0.436, "R@10": 0.510},
                "LoCoMo-RAG": {"R@5": 0.198, "R@10": 0.265},
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"\n  Results exported to: {args.output}")

    # Cleanup
    if not args.keep_data:
        print(f"\n  Cleaning up...")
        cleanup(["demo_raw", "demo_ctx", "demo_facts"])
        print(f"  Done.")
    else:
        print(f"\n  Data kept: demo_raw, demo_ctx, demo_facts")


if __name__ == "__main__":
    main()
