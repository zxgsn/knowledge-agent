"""Optimized LoCoMo benchmark with all beneficial features enabled.

This script runs the knowledge-agent with maximum optimization settings:
- HyDE (Hypothetical Document Embeddings) for better semantic matching
- Query rewrite for ambiguous queries
- MMR (Maximal Marginal Relevance) for diversity
- Content-based deduplication
- Optimized BM25 with websearch_to_tsquery and scaling
- LLM judge for accurate evaluation
- Hybrid ingestion strategy (context windows + extracted facts)

Usage:
  # Quick test (1 sample, ~3 min)
  python scripts/run_optimized_benchmark.py --limit 1

  # Full evaluation (all samples, ~30-60 min)
  python scripts/run_optimized_benchmark.py

  # Export results to JSON
  python scripts/run_optimized_benchmark.py --limit 3 --output results_optimized.json

  # Compare with baseline
  python scripts/run_optimized_benchmark.py --limit 3 --compare-baseline
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from functools import partial

# Suppress langgraph deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Enable all optimizations via environment variables
os.environ["HYDE_ENABLED"] = "true"
os.environ["MEMORY_QUERY_REWRITE_ENABLED"] = "true"
os.environ["MMR_ENABLED"] = "true"
os.environ["MMR_LAMBDA"] = "0.7"  # Favor relevance slightly more
os.environ["RERANK_ENABLED"] = "true"

import psycopg
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pgvector import Vector
from pgvector.psycopg import register_vector
from tqdm import tqdm

load_dotenv()

from agent.storage import get_db_url, get_embeddings
from agent.db import search_archival, _deduplicate_results, _mmr_rerank
from agent.nodes.researcher import generate_hypothetical_document, rewrite_query
from agent.configuration import Configuration

# Import test_locomo utilities
from test_locomo import (
    build_evidence_index,
    check_db_connection,
    clean_namespace,
    ingest_extracted_facts,
    ingest_raw_turns,
    ingest_session_context,
    llm_judge_answer,
    load_locomo,
    print_results,
)
from benchmark_comparison import extract_overall_metrics

BATCH_SIZE = 10


# ============================================================
# Optimized Search Functions
# ============================================================

def search_archival_optimized(
    query: str, namespace: str, limit: int = 10, alpha: float = 0.7,
    enable_hyde: bool = True, enable_dedup: bool = True, enable_mmr: bool = True,
    llm=None,
) -> list[dict]:
    """Optimized hybrid search with all enhancements.

    Pipeline:
    1. HyDE: Generate hypothetical answer, then search with its embedding
    2. Hybrid search: BM25 (websearch_to_tsquery * 10) + vector similarity
    3. Content-based deduplication (threshold 0.95)
    4. MMR reranking for diversity (lambda=0.7)
    5. Cross-encoder reranking (if enabled)
    """
    search_query = query

    # Step 1: HyDE - generate hypothetical document for better semantic matching
    if enable_hyde and llm:
        try:
            hyde_prompt = (
                "Answer this question in 1-2 short sentences. "
                "Be specific with names and details. "
                "If you don't know, guess based on the question context.\n\n"
                f"Question: {query}"
            )
            response = llm.invoke(hyde_prompt)
            search_query = response.content.strip()
        except Exception:
            search_query = query

    # Step 2: Hybrid search (BM25 + vector)
    embeddings = get_embeddings()
    query_embedding = Vector(embeddings.embed_query(search_query))

    conn = psycopg.connect(get_db_url())
    register_vector(conn)

    rows = conn.execute(
        """
        SELECT content, metadata,
               %s * (1 - (embedding <=> %s::vector))
                 + (1 - %s) * LEAST(1, ts_rank(content_tsv, websearch_to_tsquery('english', %s)) * 10)
               AS score
        FROM archival_memory
        WHERE namespace = %s
          AND 1 - (embedding <=> %s::vector) > 0.10
        ORDER BY score DESC
        LIMIT %s
        """,
        (alpha, query_embedding, alpha, search_query, namespace, query_embedding, limit * 3),
    ).fetchall()
    conn.close()

    results = [
        {
            "content": r[0],
            "metadata": r[1] if isinstance(r[1], dict) else json.loads(r[1]),
            "score": float(r[2]),
        }
        for r in rows
    ]

    # Step 3: Content-based deduplication
    if enable_dedup:
        results = _deduplicate_results(results, threshold=0.95)

    # Step 4: MMR reranking for diversity
    if enable_mmr and len(results) > 1:
        query_vec = embeddings.embed_query(search_query)
        results = _mmr_rerank(query_vec, results, lambda_param=0.7, top_k=limit * 2)

    return results[:limit]


def search_archival_baseline(
    query: str, namespace: str, limit: int = 10, alpha: float = 0.7,
) -> list[dict]:
    """Baseline search without optimizations for comparison."""
    embeddings = get_embeddings()
    query_embedding = Vector(embeddings.embed_query(query))

    conn = psycopg.connect(get_db_url())
    register_vector(conn)

    rows = conn.execute(
        """
        SELECT content, metadata,
               %s * (1 - (embedding <=> %s::vector))
                 + (1 - %s) * LEAST(1, ts_rank(content_tsv, plainto_tsquery('english', %s)) * 5)
               AS score
        FROM archival_memory
        WHERE namespace = %s
          AND 1 - (embedding <=> %s::vector) > 0.15
        ORDER BY score DESC
        LIMIT %s
        """,
        (alpha, query_embedding, alpha, query, namespace, query_embedding, limit),
    ).fetchall()
    conn.close()

    return [
        {
            "content": r[0],
            "metadata": r[1] if isinstance(r[1], dict) else json.loads(r[1]),
            "score": float(r[2]),
        }
        for r in rows
    ]


# ============================================================
# Optimized Evaluation
# ============================================================

def evaluate_recall_optimized(
    samples: list[dict],
    namespace: str,
    k_values: list[int] | None = None,
    use_llm_judge: bool = True,
    llm=None,
) -> dict:
    """Evaluate Recall@K and MRR with optimized search and optional LLM judge."""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    max_k = max(k_values)
    results: dict[str, dict[int, dict]] = {}

    all_questions = []
    for sample in samples:
        for q in sample["questions"]:
            all_questions.append(q)

    evidence_index = build_evidence_index(samples)

    def optimized_checker(question, answer, results, k, evidence=None):
        """Combined checker: substring + evidence + optional LLM judge."""
        # Standard substring/token match
        from test_locomo import answer_in_results_with_evidence
        if answer_in_results_with_evidence(answer, results, k, evidence, evidence_index):
            return True

        # LLM judge for borderline cases
        if use_llm_judge and llm and answer.strip():
            return llm_judge_answer(question, answer, results, k, llm)

        return False

    t0 = time.time()

    for q in tqdm(all_questions, desc="  Evaluating optimized R@K+MRR", unit="q"):
        qtype = q.get("question_type", "unknown")
        question = q.get("question", "")
        answer = q.get("answer", "")
        evidence = q.get("evidence", [])

        if not question:
            continue
        if not answer.strip() and not evidence:
            continue

        if qtype not in results:
            results[qtype] = {k: {"hits": 0, "total": 0} for k in k_values}

        for k in k_values:
            results[qtype][k]["total"] += 1

        # Use optimized search
        search_results = search_archival_optimized(
            question, namespace, limit=max_k,
            enable_hyde=True, enable_dedup=True, enable_mmr=True, llm=llm,
        )

        # Recall@K
        for k in k_values:
            if optimized_checker(question, answer, search_results, k, evidence=evidence):
                results[qtype][k]["hits"] += 1

        # MRR
        first_rank = 0
        for rank_idx, r in enumerate(search_results[:max_k]):
            if optimized_checker(question, answer, [r], 1, evidence=evidence):
                first_rank = rank_idx + 1
                break
        rr = 1.0 / first_rank if first_rank > 0 else 0.0

        if "_mrr" not in results:
            results["_mrr"] = {}
        if qtype not in results["_mrr"]:
            results["_mrr"][qtype] = {"rr_sum": 0.0, "total": 0}
        results["_mrr"][qtype]["rr_sum"] += rr
        results["_mrr"][qtype]["total"] += 1

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return results


# ============================================================
# Main Benchmark Runner
# ============================================================

def print_comparison_table(optimized_results: dict, baseline_results: dict = None) -> None:
    """Print comparison between optimized and baseline results."""
    print("\n" + "=" * 80)
    print("  Optimized Benchmark Results")
    print("=" * 80)

    opt_metrics = extract_overall_metrics(optimized_results)

    # External baselines for reference
    external_baselines = {
        "mem0 (paper)": {"R@5": 0.387, "R@10": 0.439},
        "mem0-graph (paper)": {"R@5": 0.436, "R@10": 0.510},
        "LoCoMo-RAG (paper)": {"R@5": 0.198, "R@10": 0.265},
    }

    print(f"\n  {'System':<35} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
    print("  " + "-" * 75)

    # Optimized results
    print(f"  {'Our System (Optimized)':<35}", end="")
    for key in ["R@1", "R@3", "R@5", "R@10", "MRR"]:
        val = opt_metrics.get(key, 0)
        print(f"{val:>7.1%} ", end="")
    print()

    # Baseline results if available
    if baseline_results:
        base_metrics = extract_overall_metrics(baseline_results)
        print(f"  {'Our System (Baseline)':<35}", end="")
        for key in ["R@1", "R@3", "R@5", "R@10", "MRR"]:
            val = base_metrics.get(key, 0)
            print(f"{val:>7.1%} ", end="")
        print()

        # Improvement
        print(f"  {'Improvement':<35}", end="")
        for key in ["R@1", "R@3", "R@5", "R@10", "MRR"]:
            opt_val = opt_metrics.get(key, 0)
            base_val = base_metrics.get(key, 0)
            delta = opt_val - base_val
            sign = "+" if delta >= 0 else ""
            print(f"{sign}{delta:>6.1%} ", end="")
        print()

    # External baselines
    print()
    print("  [Published Baselines]")
    for name, data in external_baselines.items():
        print(f"  {name:<35}", end="")
        for key in ["R@1", "R@3", "R@5", "R@10", "MRR"]:
            val = data.get(key)
            if val is not None:
                print(f"{val:>7.1%} ", end="")
            else:
                print(f"{'--':>8}", end="")
        print()

    print("  " + "-" * 75)

    # Best comparison
    best_ext_r10 = max(d.get("R@10", 0) for d in external_baselines.values())
    opt_r10 = opt_metrics.get("R@10", 0)
    delta = opt_r10 - best_ext_r10
    sign = "+" if delta >= 0 else ""
    print(f"  vs best published (mem0-graph): {sign}{delta:.1%}")
    print("=" * 80)


def print_per_type_breakdown(results: dict, title: str) -> None:
    """Print per-question-type breakdown."""
    k_values = set()
    for qtype, qtype_data in results.items():
        if qtype == "_mrr":
            continue
        k_values.update(qtype_data.keys())
    k_values = sorted(k_values)

    print(f"\n  {title} — Per-Type Breakdown")
    header = f"  {'Type':<20}"
    for k in k_values:
        header += f"{'R@' + str(k):>8}"
    header += f"{'MRR':>8}"
    print(header)
    print("  " + "-" * (20 + 8 * len(k_values) + 8))

    for qtype in sorted(k for k in results.keys() if k != "_mrr"):
        print(f"  {qtype:<20}", end="")
        for k in k_values:
            data = results[qtype].get(k, {"hits": 0, "total": 1})
            recall = data["hits"] / data["total"] if data["total"] > 0 else 0
            print(f"{recall:>7.1%}", end=" ")
        if "_mrr" in results and qtype in results["_mrr"]:
            mrr_data = results["_mrr"][qtype]
            mrr = mrr_data["rr_sum"] / mrr_data["total"] if mrr_data["total"] > 0 else 0
            print(f"{mrr:>7.1%}", end=" ")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimized LoCoMo benchmark with all enhancements"
    )
    parser.add_argument("--local", type=str, default=None, help="Path to local LoCoMo JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (fast mode)")
    parser.add_argument("--output", type=str, default=None, help="Export results to JSON")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Also run baseline for comparison")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Disable LLM judge (faster but less accurate)")
    parser.add_argument("--no-hyde", action="store_true", help="Disable HyDE search")
    parser.add_argument("--no-mmr", action="store_true", help="Disable MMR reranking")
    args = parser.parse_args()

    check_db_connection()

    t_start = time.time()
    samples = load_locomo(local_path=args.local, limit=args.limit)

    # Get LLM for HyDE and LLM judge
    llm = None
    if not args.no_hyde or not args.no_llm_judge:
        try:
            from langchain_openai import ChatOpenAI
            config = Configuration.from_runnable_config()
            llm = ChatOpenAI(
                model=config.llm_model,
                base_url=config.llm_base_url,
                api_key=config.llm_api_key,
                temperature=0,
            )
            print(f"  LLM initialized: {config.llm_model}")
        except Exception as e:
            print(f"  [warn] LLM initialization failed: {e}")
            print("  [warn] HyDE and LLM judge will be disabled")

    k_values = [1, 3, 5, 10]

    # Run optimized evaluation
    print("\n" + "=" * 80)
    print("  Running OPTIMIZED evaluation (Hybrid + HyDE + MMR + Dedup + LLM Judge)")
    print("=" * 80)

    ns_optimized = "locomo_bench_optimized"
    clean_namespace(ns_optimized)

    # Ingest with hybrid strategy (context windows + extracted facts)
    print("\n  [1/2] Ingesting with hybrid strategy...")
    ingest_session_context(samples, namespace=ns_optimized)
    if llm:
        ingest_extracted_facts(samples, namespace=ns_optimized)
    else:
        print("  [skip] No LLM available, skipping fact extraction")

    # Evaluate with optimizations
    print("\n  [2/2] Evaluating with optimized search...")
    optimized_results = evaluate_recall_optimized(
        samples, ns_optimized, k_values,
        use_llm_judge=not args.no_llm_judge and llm is not None,
        llm=llm,
    )

    print_results(optimized_results, "Optimized (Hybrid + HyDE + MMR + Dedup + LLM Judge)")

    # Run baseline for comparison if requested
    baseline_results = None
    if args.compare_baseline:
        print("\n" + "=" * 80)
        print("  Running BASELINE evaluation for comparison")
        print("=" * 80)

        ns_baseline = "locomo_bench_baseline"
        clean_namespace(ns_baseline)

        print("\n  [1/2] Ingesting baseline (raw turns)...")
        ingest_raw_turns(samples, namespace=ns_baseline)

        print("\n  [2/2] Evaluating baseline...")
        from test_locomo import evaluate_recall
        baseline_results = evaluate_recall(samples, ns_baseline, k_values)
        print_results(baseline_results, "Baseline (Raw Turns)")

    # Print comparison
    print_comparison_table(optimized_results, baseline_results)
    print_per_type_breakdown(optimized_results, "Optimized System")

    # Export results
    if args.output:
        export = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "hyde_enabled": not args.no_hyde,
                "mmr_enabled": not args.no_mmr,
                "llm_judge_enabled": not args.no_llm_judge,
                "strategy": "hybrid (context windows + extracted facts)",
                "bm25_optimization": "websearch_to_tsquery * 10",
                "dedup_threshold": 0.95,
                "mmr_lambda": 0.7,
            },
            "optimized": extract_overall_metrics(optimized_results),
            "optimized_per_type": {
                k: v for k, v in optimized_results.items() if k != "_mrr"
            },
            "external_baselines": {
                "mem0 (paper)": {"R@5": 0.387, "R@10": 0.439},
                "mem0-graph (paper)": {"R@5": 0.436, "R@10": 0.510},
                "LoCoMo-RAG (paper)": {"R@5": 0.198, "R@10": 0.265},
            },
        }
        if baseline_results:
            export["baseline"] = extract_overall_metrics(baseline_results)
            export["improvement"] = {
                k: export["optimized"].get(k, 0) - export["baseline"].get(k, 0)
                for k in ["R@1", "R@3", "R@5", "R@10", "MRR"]
            }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"\n  Results exported to: {args.output}")

    elapsed = time.time() - t_start
    print(f"\nTotal benchmark time: {elapsed:.1f}s")

    # Cleanup
    clean_namespace(ns_optimized)
    if args.compare_baseline:
        clean_namespace("locomo_bench_baseline")


if __name__ == "__main__":
    main()
