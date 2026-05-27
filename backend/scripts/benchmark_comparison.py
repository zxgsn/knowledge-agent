"""Comprehensive LoCoMo benchmark: strategy comparison + external baselines.

Runs all internal strategies (baseline, context windows, extracted facts, hybrid),
computes Recall@K + MRR, and compares against published baselines from other systems.

Usage:
  # Quick run (1 sample, ~2 min)
  python scripts/benchmark_comparison.py --limit 1

  # Full run (all samples, slower)
  python scripts/benchmark_comparison.py

  # Skip LLM-based strategies (fast, no API calls for extraction)
  python scripts/benchmark_comparison.py --limit 1 --skip-llm

  # Export results to JSON
  python scripts/benchmark_comparison.py --limit 1 --output results.json
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import time

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # stdout/stderr wrapping is handled by test_locomo at import time
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Reuse all evaluation infrastructure from test_locomo
from test_locomo import (
    build_evidence_index,
    check_db_connection,
    clean_namespace,
    compare_strategies,
    evaluate_recall,
    ingest_cross_session_summary,
    ingest_extracted_facts,
    ingest_raw_turns,
    ingest_session_context,
    load_locomo,
    print_results,
    search_archival,
)


# ============================================================
# Published Baselines (from literature)
# ============================================================

EXTERNAL_BASELINES = {
    # mem0 paper (arXiv 2409.04667) - LoCoMo results
    # Note: these used gpt-4o-mini for extraction + a different embedding model,
    # so absolute numbers are NOT directly comparable. The value is in showing
    # the relative improvement of each pipeline stage.
    "mem0 (paper)": {
        "source": "mem0 paper, arXiv 2409.04667, Table 3",
        "note": "gpt-4o-mini + text-embedding-3-small, LoCoMo full",
        "R@1": None,  # not reported at this granularity
        "R@3": None,
        "R@5": 0.387,
        "R@10": 0.439,
        "MRR": None,
    },
    "mem0-graph (paper)": {
        "source": "mem0 paper, arXiv 2409.04667, Table 3",
        "note": "graph-augmented mem0 variant",
        "R@1": None,
        "R@3": None,
        "R@5": 0.436,
        "R@10": 0.510,
        "MRR": None,
    },
    # LoCoMo paper baselines (arXiv 2311.00176)
    "LoCoMo-RAG (paper)": {
        "source": "LoCoMo paper, arXiv 2311.00176",
        "note": "standard RAG on conversation turns",
        "R@1": None,
        "R@3": None,
        "R@5": 0.198,
        "R@10": 0.265,
        "MRR": None,
    },
    "LoCoMo-full-context (paper)": {
        "source": "LoCoMo paper, arXiv 2311.00176",
        "note": "full conversation as context (upper bound)",
        "R@1": None,
        "R@3": None,
        "R@5": None,
        "R@10": None,
        "MRR": None,
    },
}


def extract_overall_metrics(results: dict) -> dict:
    """Extract overall Recall@K and MRR from evaluation results."""
    k_values = set()
    for qtype, qtype_data in results.items():
        if qtype == "_mrr":
            continue
        k_values.update(qtype_data.keys())
    k_values = sorted(k_values)

    metrics = {}
    total_hits = {k: 0 for k in k_values}
    total_count = {k: 0 for k in k_values}

    for qtype, qtype_data in results.items():
        if qtype == "_mrr":
            continue
        for k in k_values:
            data = qtype_data.get(k, {"hits": 0, "total": 0})
            total_hits[k] += data["hits"]
            total_count[k] += data["total"]

    for k in k_values:
        metrics[f"R@{k}"] = total_hits[k] / total_count[k] if total_count[k] > 0 else 0

    # MRR
    if "_mrr" in results:
        rr_sum = sum(d["rr_sum"] for d in results["_mrr"].values())
        mrr_count = sum(d["total"] for d in results["_mrr"].values())
        metrics["MRR"] = rr_sum / mrr_count if mrr_count > 0 else 0

    return metrics


def print_comparison_table(all_results: dict[str, dict], external: dict) -> None:
    """Print a unified comparison table: internal strategies + external baselines."""
    # Determine available K values from internal results
    k_values = set()
    for strategy_results in all_results.values():
        for qtype, qtype_data in strategy_results.items():
            if qtype == "_mrr":
                continue
            k_values.update(qtype_data.keys())
    k_values = sorted(k_values)

    has_mrr = any("_mrr" in r for r in all_results.values())
    col_labels = [f"R@{k}" for k in k_values] + (["MRR"] if has_mrr else [])

    # Header
    print("\n" + "=" * 78)
    print("  LoCoMo Benchmark: Full Comparison")
    print("=" * 78)
    print(f"  {'System / Strategy':<30}", end="")
    for label in col_labels:
        print(f"{label:>8}", end="")
    print()
    print("  " + "-" * (30 + 8 * len(col_labels)))

    # Internal strategies
    print("  [Our System]")
    for label, results in all_results.items():
        metrics = extract_overall_metrics(results)
        print(f"  {'  ' + label:<30}", end="")
        for k in k_values:
            val = metrics.get(f"R@{k}", 0)
            print(f"{val:>7.1%} ", end="")
        if has_mrr:
            mrr_val = metrics.get("MRR", 0)
            print(f"{mrr_val:>7.1%} ", end="")
        print()

    # External baselines
    print()
    print("  [Published Baselines]")
    for name, data in external.items():
        print(f"  {'  ' + name:<30}", end="")
        for k in k_values:
            key = f"R@{k}"
            val = data.get(key)
            if val is not None:
                print(f"{val:>7.1%} ", end="")
            else:
                print(f"{'--':>8}", end="")
        mrr_val = data.get("MRR")
        if has_mrr:
            if mrr_val is not None:
                print(f"{mrr_val:>7.1%} ", end="")
            else:
                print(f"{'--':>8}", end="")
        print()

    print("  " + "-" * (30 + 8 * len(col_labels)))
    print()

    # Best internal strategy highlight
    best_label = None
    best_r10 = 0
    for label, results in all_results.items():
        metrics = extract_overall_metrics(results)
        r10 = metrics.get("R@10", 0)
        if r10 > best_r10:
            best_r10 = r10
            best_label = label

    if best_label:
        print(f"  Best strategy: {best_label} (R@10={best_r10:.1%})")

        # Compare with best external baseline that has R@10
        best_ext_name = None
        best_ext_r10 = 0
        for name, data in external.items():
            r10 = data.get("R@10")
            if r10 is not None and r10 > best_ext_r10:
                best_ext_r10 = r10
                best_ext_name = name

        if best_ext_name and best_r10 > 0:
            delta = best_r10 - best_ext_r10
            direction = "+" if delta >= 0 else ""
            print(f"  vs best published ({best_ext_name}): {direction}{delta:.1%}")

    print("=" * 78)


def print_per_type_breakdown(results: dict, title: str) -> None:
    """Print per-question-type breakdown with MRR."""
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


def run_benchmark(
    samples: list[dict],
    skip_llm: bool = False,
) -> dict[str, dict]:
    """Run all strategies and return their evaluation results."""
    k_values = [1, 3, 5, 10]
    ev_index = build_evidence_index(samples)
    all_results = {}

    # Strategy 1: Baseline
    print("\n" + "-" * 50)
    print("  [1/4] Baseline: Raw conversation turns")
    print("-" * 50)
    ns = "locomo_bench_baseline"
    clean_namespace(ns)
    ingest_raw_turns(samples, namespace=ns)
    results = evaluate_recall(samples, ns, k_values, evidence_index=ev_index)
    print_results(results, "Baseline")
    all_results["Baseline (raw turns)"] = results

    # Strategy 2: Context windows (w=10, s=3)
    print("\n" + "-" * 50)
    print("  [2/4] Session context windows (w=10, s=3)")
    print("-" * 50)
    ns = "locomo_bench_context"
    clean_namespace(ns)
    ingest_session_context(samples, namespace=ns)
    results = evaluate_recall(samples, ns, k_values, evidence_index=ev_index)
    print_results(results, "Context Windows")
    all_results["Context windows (w=10)"] = results

    if not skip_llm:
        # Strategy 3: Extracted facts (mem0-style)
        print("\n" + "-" * 50)
        print("  [3/4] Extracted facts (mem0 pipeline)")
        print("-" * 50)
        ns = "locomo_bench_extracted"
        clean_namespace(ns)
        ingest_extracted_facts(samples, namespace=ns)
        results = evaluate_recall(samples, ns, k_values, evidence_index=ev_index)
        print_results(results, "Extracted Facts")
        all_results["Extracted facts (mem0)"] = results

        # Strategy 4: Hybrid (context + extracted + cross-session)
        print("\n" + "-" * 50)
        print("  [4/4] Hybrid: Context + Extracted + Cross-session")
        print("-" * 50)
        ns = "locomo_bench_hybrid"
        clean_namespace(ns)
        ingest_session_context(samples, namespace=ns)
        ingest_extracted_facts(samples, namespace=ns)
        ingest_cross_session_summary(samples, namespace=ns)
        results = evaluate_recall(samples, ns, k_values, evidence_index=ev_index)
        print_results(results, "Hybrid")
        all_results["Hybrid (ctx+facts+cross)"] = results
    else:
        print("\n  [skip-llm] Skipping strategies 3 & 4 (require LLM API calls)")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive LoCoMo benchmark with external comparison"
    )
    parser.add_argument("--local", type=str, default=None, help="Path to local LoCoMo JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (fast mode)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM-based strategies")
    parser.add_argument("--output", type=str, default=None, help="Export results to JSON file")
    parser.add_argument(
        "--external", action="store_true", default=True,
        help="Show external baselines (default: on)",
    )
    parser.add_argument("--no-external", action="store_false", dest="external",
                        help="Hide external baselines")
    args = parser.parse_args()

    check_db_connection()

    t_start = time.time()
    samples = load_locomo(local_path=args.local, limit=args.limit)

    # Run all strategies
    all_results = run_benchmark(samples, skip_llm=args.skip_llm)

    # Print unified comparison
    if args.external:
        print_comparison_table(all_results, EXTERNAL_BASELINES)
    else:
        # At least print internal summary
        print("\n" + "=" * 78)
        print("  Internal Strategy Comparison")
        print("=" * 78)
        for label, results in all_results.items():
            metrics = extract_overall_metrics(results)
            line = f"  {label:<30}"
            for key in ["R@1", "R@3", "R@5", "R@10", "MRR"]:
                val = metrics.get(key, 0)
                line += f"{val:>7.1%} "
            print(line)
        print("=" * 78)

    # Per-type breakdown for best strategy
    if all_results:
        best_label = max(all_results, key=lambda l: extract_overall_metrics(all_results[l]).get("R@10", 0))
        print_per_type_breakdown(all_results[best_label], best_label)

    # Export JSON
    if args.output:
        export = {}
        for label, results in all_results.items():
            export[label] = extract_overall_metrics(results)
        if args.external:
            export["_external_baselines"] = EXTERNAL_BASELINES
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"\n  Results exported to: {args.output}")

    elapsed = time.time() - t_start
    print(f"\nTotal benchmark time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
